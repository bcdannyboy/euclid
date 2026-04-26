from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.fit.multi_horizon import resolve_fit_strategy
from euclid.modules.features import FeatureSpec, default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.search.backends import (
    DescriptiveSearchProposal,
    run_descriptive_search_backends,
)

DEFAULT_PROPOSAL_IDS = (
    "analytic_intercept",
    "analytic_lag1_affine",
    "recursive_level_smoother",
    "recursive_running_mean",
    "spectral_harmonic_1",
    "spectral_harmonic_2",
)


def test_exact_descriptive_search_materializes_all_retained_families_into_cir() -> None:
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    canonicalization_policy = build_canonicalization_policy()
    search_plan = build_search_plan(
        evaluation_plan=evaluation_plan,
        canonicalization_policy_ref=TypedRef(
            "canonicalization_policy_manifest@1.0.0",
            canonicalization_policy.canonicalization_policy_id,
        ),
        codelength_policy_ref=TypedRef(
            "codelength_policy_manifest@1.1.0",
            "mdl_policy_default",
        ),
        reference_description_policy_ref=TypedRef(
            "reference_description_policy_manifest@1.1.0",
            "reference_description_default",
        ),
        observation_model_ref=TypedRef(
            "observation_model_manifest@1.1.0",
            "observation_model_default",
        ),
        candidate_family_ids=DEFAULT_PROPOSAL_IDS,
        search_class="exact_finite_enumeration",
        proposal_limit=len(DEFAULT_PROPOSAL_IDS),
        seasonal_period=4,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )

    assert {
        candidate.structural_layer.cir_family_id
        for candidate in result.accepted_candidates
    } == {"analytic", "recursive"}
    assert result.coverage.canonical_program_count == len(DEFAULT_PROPOSAL_IDS)
    assert result.coverage.attempted_candidate_count == len(DEFAULT_PROPOSAL_IDS)
    assert result.coverage.omitted_candidate_count == 0
    assert result.coverage.accepted_candidate_count == 1
    assert result.coverage.law_eligible_candidate_count == 3
    assert result.coverage.rejected_candidate_count == 3
    assert (
        result.coverage.coverage_statement
        == "complete_over_declared_canonical_program_space"
    )
    assert result.coverage.exactness_ceiling == "exact_over_declared_fragment_only"
    assert (
        result.coverage.scope_declaration
        == "finite_exactness_limited_to_declared_canonical_program_space"
    )
    assert len(
        {candidate.canonical_hash() for candidate in result.accepted_candidates}
    ) == len(result.accepted_candidates)
    assert {
        family_result.family_id: family_result.coverage.canonical_program_count
        for family_result in result.family_results
    } == {
        "analytic": 2,
        "recursive": 2,
        "spectral": 2,
        "algorithmic": 0,
    }
    assert {
        family_result.family_id: family_result.coverage.accepted_candidate_count
        for family_result in result.family_results
    } == {
        "analytic": 1,
        "recursive": 1,
        "spectral": 0,
        "algorithmic": 0,
    }
    assert {
        family_result.family_id: family_result.coverage.law_eligible_candidate_count
        for family_result in result.family_results
    } == {
        "analytic": 2,
        "recursive": 1,
        "spectral": 0,
        "algorithmic": 0,
    }
    assert all(
        candidate.evidence_layer.backend_origin_record.search_class
        == "exact_finite_enumeration"
        for candidate in result.accepted_candidates
    )
    assert result.description_artifacts
    assert len(result.description_artifacts) == len(result.admissibility_diagnostics)
    assert {
        diagnostic.candidate_id
        for diagnostic in result.admissibility_diagnostics
        if diagnostic.is_admissible
    } == {
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.accepted_candidates
    }
    assert {
        diagnostic.candidate_id
        for diagnostic in result.admissibility_diagnostics
        if not diagnostic.is_admissible
    } == {
        "recursive_running_mean",
        "spectral_harmonic_1",
        "spectral_harmonic_2",
    }
    assert result.frontier.coverage.comparable_axes == (
        "structure_code_bits",
        "description_gain_bits",
        "inner_primary_score",
    )
    assert result.frontier.coverage.incomparable_axes == ()
    assert [record.candidate_id for record in result.frontier.frontier_candidates] == [
        "analytic_lag1_affine",
        "analytic_intercept",
        "recursive_level_smoother",
    ]
    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.frontier.retained_frontier_cir_candidates
    ] == [
        "analytic_lag1_affine",
        "analytic_intercept",
        "recursive_level_smoother",
    ]
    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.frontier.frozen_shortlist_cir_candidates
    ] == [
        "analytic_lag1_affine",
    ]
    assert all(
        artifact.description_gain_bits
        == pytest.approx(artifact.reference_bits - artifact.L_total_bits)
        for artifact in result.description_artifacts
    )


def test_exact_enumeration_orders_family_members_by_candidate_id() -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=(
            "analytic_zed",
            "analytic_alpha",
            "recursive_level_smoother",
        ),
        search_class="exact_finite_enumeration",
        proposal_limit=3,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=(
            DescriptiveSearchProposal(
                candidate_id="analytic_zed",
                primitive_family="analytic",
                form_class="closed_form_expression",
                parameter_values={"intercept": 15.0},
            ),
            DescriptiveSearchProposal(
                candidate_id="analytic_alpha",
                primitive_family="analytic",
                form_class="closed_form_expression",
                parameter_values={"intercept": 14.0},
            ),
            DescriptiveSearchProposal(
                candidate_id="recursive_level_smoother",
                primitive_family="recursive",
                form_class="state_recurrence",
                literal_values={"alpha": 0.5},
                persistent_state={"level": 10.0, "step_count": 0},
                history_access_mode="bounded_lag_window",
                max_lag=1,
            ),
        ),
    )

    proposal_ranks = {
        candidate.evidence_layer.backend_origin_record.source_candidate_id: (
            candidate.evidence_layer.backend_origin_record.proposal_rank
        )
        for candidate in result.descriptive_scope
    }
    assert proposal_ranks["analytic_alpha"] == 0
    assert proposal_ranks["analytic_zed"] == 1
    assert proposal_ranks["recursive_level_smoother"] == 2
    assert result.coverage.disclosures["canonical_enumerator"] == (
        "declared_adapter_family_order_then_candidate_id"
    )


def test_bounded_heuristic_search_reports_omitted_space_honestly() -> None:
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    canonicalization_policy = build_canonicalization_policy()
    search_plan = build_search_plan(
        evaluation_plan=evaluation_plan,
        canonicalization_policy_ref=TypedRef(
            "canonicalization_policy_manifest@1.0.0",
            canonicalization_policy.canonicalization_policy_id,
        ),
        codelength_policy_ref=TypedRef(
            "codelength_policy_manifest@1.1.0",
            "mdl_policy_default",
        ),
        reference_description_policy_ref=TypedRef(
            "reference_description_policy_manifest@1.1.0",
            "reference_description_default",
        ),
        observation_model_ref=TypedRef(
            "observation_model_manifest@1.1.0",
            "observation_model_default",
        ),
        candidate_family_ids=DEFAULT_PROPOSAL_IDS,
        search_class="bounded_heuristic",
        proposal_limit=3,
        seasonal_period=4,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )

    assert result.coverage.canonical_program_count == len(DEFAULT_PROPOSAL_IDS)
    assert result.coverage.attempted_candidate_count == 3
    assert result.coverage.accepted_candidate_count == 1
    assert result.coverage.law_eligible_candidate_count == 3
    assert result.coverage.omitted_candidate_count == 3
    assert result.coverage.coverage_statement == "incomplete_search_disclosed"
    assert result.coverage.exactness_ceiling == "no_global_exactness_claim"
    assert (
        result.coverage.scope_declaration
        == "heuristic_prefix_over_declared_candidate_space"
    )
    assert result.frontier.coverage.comparable_axes == (
        "structure_code_bits",
        "description_gain_bits",
        "inner_primary_score",
    )
    assert result.frontier.coverage.incomparable_axes == ()
    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.accepted_candidates
    ] == [
        "analytic_intercept",
        "analytic_lag1_affine",
        "recursive_level_smoother",
    ]
    for candidate in result.accepted_candidates:
        assert (
            candidate.evidence_layer.backend_origin_record.search_class
            == "bounded_heuristic"
        )
        search_evidence = candidate.evidence_layer.transient_diagnostics[
            "search_evidence"
        ]
        assert search_evidence["search_class"] == "bounded_heuristic"
        assert search_evidence["proposer_mechanism"] == (
            "declared_adapter_default_or_user_supplied_proposals"
        )
        assert search_evidence["pruning_rules"] == (
            "proposal_limit_prefix_then_canonical_duplicate_screen"
            "_then_descriptive_admissibility"
        )
        assert search_evidence["exactness_scope"] == (
            "heuristic_prefix_over_declared_candidate_space"
        )


def test_search_inner_primary_score_uses_rollout_objective_when_configured() -> None:
    feature_view, audit = _feature_view(
        values=(1.0, 1.5, 2.8, 4.9, 8.1, 12.0, 17.5, 23.0, 31.0)
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=5,
        horizon=3,
    )
    canonicalization_policy = build_canonicalization_policy()
    rollout_strategy = resolve_fit_strategy(
        strategy_id="joint",
        horizon_set=(1, 3),
        horizon_weights=((1, "0.25"), (3, "0.75")),
    )
    search_plan = build_search_plan(
        evaluation_plan=evaluation_plan,
        canonicalization_policy_ref=TypedRef(
            "canonicalization_policy_manifest@1.0.0",
            canonicalization_policy.canonicalization_policy_id,
        ),
        codelength_policy_ref=TypedRef(
            "codelength_policy_manifest@1.1.0",
            "mdl_policy_default",
        ),
        reference_description_policy_ref=TypedRef(
            "reference_description_policy_manifest@1.1.0",
            "reference_description_default",
        ),
        observation_model_ref=TypedRef(
            "observation_model_manifest@1.1.0",
            "observation_model_default",
        ),
        candidate_family_ids=("analytic_intercept", "analytic_lag1_affine"),
        search_class="exact_finite_enumeration",
        proposal_limit=2,
        fit_strategy=rollout_strategy.as_dict(),
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )

    for candidate in result.descriptive_scope:
        diagnostics = candidate.evidence_layer.transient_diagnostics
        rollout_objective = diagnostics["search_rollout_primary_objective"]
        assert diagnostics["inner_primary_score"] == pytest.approx(
            rollout_objective["aggregated_primary_score"]
        )
        assert rollout_objective["fit_strategy_id"] == "joint"
        assert rollout_objective["horizon_set"] == [1, 3]
        assert rollout_objective["horizon_weights"] == [
            {"horizon": 1, "weight": "0.25"},
            {"horizon": 3, "weight": "0.75"},
        ]


def test_search_threads_coding_policy_metadata_into_description_artifacts() -> None:
    feature_view, audit = _feature_view(values=(10.0, 10.0, 10.5, 10.0, 10.5))
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=("analytic_intercept",),
        search_class="exact_finite_enumeration",
        proposal_limit=1,
        data_code_family="prequential_laplace_residual_bin_v1",
        reference_policy={
            "reference_family_id": "naive_last_observation",
            "policy_id": "naive_last_observation_reference_v1",
        },
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )

    assert result.description_artifacts
    artifact = result.description_artifacts[0]
    assert artifact.data_code_family == "prequential_laplace_residual_bin_v1"
    assert artifact.reference_policy_id == "naive_last_observation_reference_v1"
    assert artifact.reference_family_id == "naive_last_observation"
    assert artifact.coding_row_set_id.startswith("sha256:")
    assert artifact.codelength_comparison_key["data_code_family"] == (
        "prequential_laplace_residual_bin_v1"
    )


def test_retained_family_expansion_emits_selected_lag_and_harmonic_group_proposals() -> None:
    feature_view, audit = _feature_view(
        values=(1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0),
        feature_names=("lag_1", "lag_2"),
    )
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=(
            "analytic_selected_lag_1_2_affine",
            "spectral_harmonic_group_1_2",
        ),
        search_class="exact_finite_enumeration",
        proposal_limit=2,
        seasonal_period=6,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )

    proposal_ids = {
        diagnostic.candidate_id for diagnostic in result.admissibility_diagnostics
    }
    assert proposal_ids == {
        "analytic_selected_lag_1_2_affine",
        "spectral_harmonic_group_1_2",
    }
    analytic = next(
        candidate
        for candidate in result.descriptive_scope
        if candidate.evidence_layer.backend_origin_record.source_candidate_id
        == "analytic_selected_lag_1_2_affine"
    )
    spectral = next(
        candidate
        for candidate in result.descriptive_scope
        if candidate.evidence_layer.backend_origin_record.source_candidate_id
        == "spectral_harmonic_group_1_2"
    )
    assert analytic.execution_layer.history_access_contract.max_lag == 2
    assert analytic.execution_layer.history_access_contract.allowed_side_information == (
        "lag_1",
        "lag_2",
    )
    assert {
        parameter.name
        for parameter in analytic.structural_layer.parameter_block.parameters
    } >= {"lag_1__coefficient", "lag_2__coefficient"}
    assert {
        literal.name: literal.value
        for literal in spectral.structural_layer.literal_block.literals
    }["harmonic_group"] == "1,2"


def test_equality_saturation_search_no_longer_uses_sort_only_extractor() -> None:
    feature_view, audit = _feature_view()
    proposals = (
        DescriptiveSearchProposal(
            candidate_id="analytic_intercept_simple",
            primitive_family="analytic",
            form_class="closed_form_expression",
            parameter_values={"intercept": 14.0},
        ),
        DescriptiveSearchProposal(
            candidate_id="analytic_piecewise_complex",
            primitive_family="analytic",
            form_class="closed_form_expression",
            feature_dependencies=("lag_1",),
            parameter_values={"intercept": 14.0},
            literal_values={"upper_cut": 3.0, "lower_cut": 1.0},
            persistent_state={"step_count": 1, "running_total": 0.0},
            composition_payload={
                "operator_id": "piecewise",
                "ordered_partition": [
                    {"start_literal": 0.0, "end_literal": 1.0, "reducer_id": "head"},
                    {"start_literal": 1.0, "end_literal": 3.0, "reducer_id": "tail"},
                ],
            },
            history_access_mode="bounded_lag_window",
            max_lag=1,
        ),
    )
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=tuple(proposal.candidate_id for proposal in proposals),
        search_class="equality_saturation_heuristic",
        proposal_limit=1,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=proposals,
    )

    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.accepted_candidates
    ] == ["analytic_intercept_simple"]
    assert result.coverage.coverage_statement == "incomplete_search_disclosed"
    assert result.coverage.exactness_ceiling == "no_global_exactness_claim"
    assert (
        result.coverage.scope_declaration
        == "heuristic_rewrite_neighborhood_with_cost_extraction"
    )
    assert result.coverage.disclosures == {
        "rewrite_system": "egraph_engine_required_for_expression_cir_rewrites",
        "extractor_cost": "declared_by_egraph_engine_rewrite_trace",
        "legacy_fragment_backend_mode": "no_sort_only_equality_saturation",
        "stop_rule": "proposal_limit=1",
    }
    replay_hook_names = {
        hook.hook_name
        for hook in result.accepted_candidates[0].evidence_layer.replay_hooks.hooks
    }
    assert replay_hook_names >= {
        "budget_record",
        "search_seed",
        "rewrite_system",
        "extractor_cost",
        "search_stop_rule",
    }


def test_stochastic_search_replays_by_seed_and_discloses_restart_policy() -> None:
    feature_view, audit = _feature_view()
    proposals = tuple(
        DescriptiveSearchProposal(
            candidate_id=f"analytic_seed_{index}",
            primitive_family="analytic",
            form_class="closed_form_expression",
            parameter_values={"intercept": 10.0 + index},
        )
        for index in range(4)
    )
    candidate_ids = tuple(proposal.candidate_id for proposal in proposals)
    seed_17_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=candidate_ids,
        search_class="stochastic_heuristic",
        proposal_limit=3,
        random_seed="17",
        candidate_batch_size=1,
    )
    seed_17_repeat_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=candidate_ids,
        search_class="stochastic_heuristic",
        proposal_limit=3,
        random_seed="17",
        candidate_batch_size=1,
    )
    seed_23_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=candidate_ids,
        search_class="stochastic_heuristic",
        proposal_limit=3,
        random_seed="23",
        candidate_batch_size=1,
    )

    seed_17_result = run_descriptive_search_backends(
        search_plan=seed_17_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=proposals,
    )
    seed_17_repeat_result = run_descriptive_search_backends(
        search_plan=seed_17_repeat_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=proposals,
    )
    seed_23_result = run_descriptive_search_backends(
        search_plan=seed_23_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=proposals,
    )

    seed_17_ids = [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in seed_17_result.accepted_candidates
    ]
    seed_23_ids = [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in seed_23_result.accepted_candidates
    ]

    assert seed_17_ids == [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in seed_17_repeat_result.accepted_candidates
    ]
    assert seed_17_ids != seed_23_ids
    assert seed_17_result.coverage.attempted_candidate_count == 3
    assert seed_17_result.coverage.omitted_candidate_count == 1
    assert seed_17_result.coverage.coverage_statement == "incomplete_search_disclosed"
    assert seed_17_result.coverage.exactness_ceiling == "no_global_exactness_claim"
    assert (
        seed_17_result.coverage.scope_declaration
        == "heuristic_seeded_search_with_declared_restart_policy"
    )
    assert seed_17_result.coverage.disclosures == {
        "proposal_distribution": "seeded_sha256_permutation_without_replacement",
        "seed_policy": "root_seed=17;derivation=deterministic_scope_hash",
        "restart_policy": (
            "batch_size=1;restart_until_budget_or_exhaustion;restarts_used=3"
        ),
        "stop_rule": "proposal_limit=3",
    }
    replay_hook_names = {
        hook.hook_name
        for hook in seed_17_result.accepted_candidates[
            0
        ].evidence_layer.replay_hooks.hooks
    }
    assert replay_hook_names >= {
        "budget_record",
        "search_seed",
        "proposal_distribution",
        "restart_policy",
        "search_stop_rule",
    }


def test_rejected_diagnostics_capture_realization_failures() -> None:
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    canonicalization_policy = build_canonicalization_policy()
    proposal_ids = (
        "bad_syntax",
        "bad_bounds",
        "bad_family",
        "bad_observation",
    )
    search_plan = build_search_plan(
        evaluation_plan=evaluation_plan,
        canonicalization_policy_ref=TypedRef(
            "canonicalization_policy_manifest@1.0.0",
            canonicalization_policy.canonicalization_policy_id,
        ),
        codelength_policy_ref=TypedRef(
            "codelength_policy_manifest@1.1.0",
            "mdl_policy_default",
        ),
        reference_description_policy_ref=TypedRef(
            "reference_description_policy_manifest@1.1.0",
            "reference_description_default",
        ),
        observation_model_ref=TypedRef(
            "observation_model_manifest@1.1.0",
            "observation_model_default",
        ),
        candidate_family_ids=proposal_ids,
        search_class="exact_finite_enumeration",
        proposal_limit=len(proposal_ids),
        seasonal_period=4,
    )

    with pytest.raises(ContractValidationError) as exc_info:
        run_descriptive_search_backends(
            search_plan=search_plan,
            feature_view=feature_view,
            include_default_grammar=False,
            proposal_specs=(
                DescriptiveSearchProposal(
                    candidate_id="bad_syntax",
                    primitive_family="analytic",
                    form_class="closed_form_expression",
                    composition_payload={"operator_id": "unknown_operator"},
                ),
                DescriptiveSearchProposal(
                    candidate_id="bad_bounds",
                    primitive_family="recursive",
                    form_class="state_recurrence",
                    max_lag=99,
                ),
                DescriptiveSearchProposal(
                    candidate_id="bad_family",
                    primitive_family="algorithmic",
                    form_class="bounded_program",
                ),
                DescriptiveSearchProposal(
                    candidate_id="bad_observation",
                    primitive_family="spectral",
                    form_class="spectral_basis_expansion",
                    required_observation_model_family="poisson_count",
                ),
            ),
        )

    assert exc_info.value.code == "descriptive_fallback_bank_unavailable"


def test_glm_style_observation_aware_reducer_fails_closed_without_bound_observation_model() -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=("analytic_poisson_glm_reducer",),
        search_class="exact_finite_enumeration",
        proposal_limit=1,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=(
            DescriptiveSearchProposal(
                candidate_id="analytic_poisson_glm_reducer",
                primitive_family="analytic",
                form_class="glm_link_reducer",
                feature_dependencies=("lag_1",),
                parameter_values={"intercept": 0.0, "lag_1__coefficient": 0.0},
                literal_values={"link": "log"},
                history_access_mode="bounded_lag_window",
                max_lag=1,
                required_observation_model_family="poisson_rate",
            ),
        ),
    )

    assert result.accepted_candidates == ()
    assert result.rejected_diagnostics[0].reason_code == "observation_model_incompatible"


def test_descriptive_scope_keeps_ranked_candidate_when_gain_floor_blocks_acceptance(
) -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=("analytic_floor_blocked",),
        search_class="exact_finite_enumeration",
        proposal_limit=1,
        minimum_description_gain_bits=1_000_000.0,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=(
            DescriptiveSearchProposal(
                candidate_id="analytic_floor_blocked",
                primitive_family="analytic",
                form_class="closed_form_expression",
                parameter_values={"intercept": 14.0},
            ),
        ),
    )

    assert result.accepted_candidates == ()
    assert result.law_eligible_scope == ()
    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.descriptive_scope
    ] == ["analytic_floor_blocked"]
    assert (
        result.best_overall_candidate
        is result.descriptive_scope[0]
    )
    assert result.accepted_candidate is None
    assert result.gap_report_reason_codes == ("description_gain_below_floor",)

    scope_metadata = result.best_overall_candidate.evidence_layer.transient_diagnostics[
        "descriptive_scope"
    ]
    assert scope_metadata == {
        "scope_rank": 1,
        "source": "primary_search",
        "selection_rule": (
            "min_total_code_bits_then_max_description_gain_then_"
            "min_structure_code_bits_then_min_canonical_byte_length_then_candidate_id"
        ),
        "law_eligible": False,
        "law_rejection_reason_codes": ["description_gain_below_floor"],
    }


def test_banned_synthetic_candidate_stays_out_of_descriptive_ranking_bank() -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=(
            "analytic_intercept",
            "analytic_additive_residual_surface",
        ),
        search_class="exact_finite_enumeration",
        proposal_limit=2,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=(
            DescriptiveSearchProposal(
                candidate_id="analytic_intercept",
                primitive_family="analytic",
                form_class="closed_form_expression",
                parameter_values={"intercept": 14.0},
            ),
            DescriptiveSearchProposal(
                candidate_id="analytic_additive_residual_surface",
                primitive_family="analytic",
                form_class="closed_form_expression",
                feature_dependencies=("lag_1",),
                parameter_values={"intercept": 14.0},
                composition_payload={
                    "operator_id": "additive_residual",
                    "base_reducer": "trend_component",
                    "residual_reducer": "seasonal_component",
                    "shared_observation_model": "point_identity",
                },
                history_access_mode="bounded_lag_window",
                max_lag=1,
            ),
        ),
    )

    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.descriptive_scope
    ] == ["analytic_intercept"]
    assert (
        result.best_overall_candidate
        .evidence_layer.backend_origin_record.source_candidate_id
        == "analytic_intercept"
    )
    assert (
        result.accepted_candidate
        .evidence_layer.backend_origin_record.source_candidate_id
        == "analytic_intercept"
    )
    assert [
        diagnostic.candidate_id for diagnostic in result.admissibility_diagnostics
    ] == ["analytic_intercept"]
    rejected = next(
        diagnostic
        for diagnostic in result.rejected_diagnostics
        if diagnostic.candidate_id == "analytic_additive_residual_surface"
    )
    assert rejected.reason_code == "descriptive_scope_excluded"
    assert rejected.details["reason_codes"] == ["requires_lookup_residual_wrapper"]
    assert isinstance(rejected.details["candidate_hash"], str)


def test_exact_closure_and_posthoc_symbolic_candidates_stay_out_of_descriptive_scope(
) -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=(
            "analytic_intercept",
            "analytic_closure_candidate",
            "analytic_symbolic_candidate",
        ),
        search_class="exact_finite_enumeration",
        proposal_limit=3,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=(
            DescriptiveSearchProposal(
                candidate_id="analytic_intercept",
                primitive_family="analytic",
                form_class="closed_form_expression",
                parameter_values={"intercept": 14.0},
            ),
            DescriptiveSearchProposal(
                candidate_id="analytic_closure_candidate",
                primitive_family="analytic",
                form_class="exact_sample_closure",
                parameter_values={"intercept": 13.0, "slope": 0.2},
            ),
            DescriptiveSearchProposal(
                candidate_id="analytic_symbolic_candidate",
                primitive_family="analytic",
                form_class="posthoc_symbolic_synthesis",
                parameter_values={"intercept": 12.0, "slope": 0.1},
            ),
        ),
    )

    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.descriptive_scope
    ] == ["analytic_intercept"]
    assert (
        result.best_overall_candidate
        .evidence_layer.backend_origin_record.source_candidate_id
        == "analytic_intercept"
    )
    assert (
        result.accepted_candidate
        .evidence_layer.backend_origin_record.source_candidate_id
        == "analytic_intercept"
    )
    rejected = {
        diagnostic.candidate_id: diagnostic
        for diagnostic in result.rejected_diagnostics
        if diagnostic.reason_code == "descriptive_scope_excluded"
    }
    assert rejected["analytic_closure_candidate"].details["reason_codes"] == [
        "requires_exact_sample_closure"
    ]
    assert rejected["analytic_symbolic_candidate"].details["reason_codes"] == [
        "requires_posthoc_symbolic_synthesis"
    ]


def test_candidate_id_fragments_alone_do_not_exclude_descriptive_scope() -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=(
            "analytic_intercept",
            "analytic_exact_closure_alias",
            "analytic_symbolic_synthesis_alias",
            "analytic_holistic_alias",
        ),
        search_class="exact_finite_enumeration",
        proposal_limit=4,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=(
            DescriptiveSearchProposal(
                candidate_id="analytic_intercept",
                primitive_family="analytic",
                form_class="closed_form_expression",
                parameter_values={"intercept": 14.0},
            ),
            DescriptiveSearchProposal(
                candidate_id="analytic_exact_closure_alias",
                primitive_family="analytic",
                form_class="closed_form_expression",
                parameter_values={"intercept": 13.0},
            ),
            DescriptiveSearchProposal(
                candidate_id="analytic_symbolic_synthesis_alias",
                primitive_family="analytic",
                form_class="closed_form_expression",
                parameter_values={"intercept": 12.0},
            ),
            DescriptiveSearchProposal(
                candidate_id="analytic_holistic_alias",
                primitive_family="analytic",
                form_class="closed_form_expression",
                parameter_values={"intercept": 11.0},
            ),
        ),
    )

    assert {
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.descriptive_scope
    } == {
        "analytic_intercept",
        "analytic_exact_closure_alias",
        "analytic_symbolic_synthesis_alias",
        "analytic_holistic_alias",
    }
    assert not any(
        diagnostic.reason_code == "descriptive_scope_excluded"
        for diagnostic in result.rejected_diagnostics
    )


def test_descriptive_scope_uses_compact_fallback_bank_before_abstaining() -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=("bad_syntax", "analytic_intercept"),
        search_class="exact_finite_enumeration",
        proposal_limit=2,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=(
            DescriptiveSearchProposal(
                candidate_id="bad_syntax",
                primitive_family="analytic",
                form_class="closed_form_expression",
                composition_payload={"operator_id": "unknown_operator"},
            ),
        ),
    )

    assert result.accepted_candidates == ()
    assert result.accepted_candidate is None
    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.descriptive_scope
    ] == ["analytic_intercept"]
    assert (
        result.best_overall_candidate
        .evidence_layer.backend_origin_record.source_candidate_id
        == "analytic_intercept"
    )
    assert result.gap_report_reason_codes == ("descriptive_only_fallback_bank",)
    assert result.coverage.canonical_program_count == 2
    assert result.coverage.attempted_candidate_count == 2
    assert result.coverage.omitted_candidate_count == 0
    assert result.coverage.fallback_candidate_count == 1
    assert result.coverage.disclosures == {
        "fragment_bounds": "proposal_limit=2",
        "canonical_enumerator": "declared_adapter_family_order_then_candidate_id",
        "enumeration_cardinality": 2,
        "stop_rule": "exhaust_declared_canonical_program_space(cardinality=2)",
    }
    assert {diagnostic.reason_code for diagnostic in result.rejected_diagnostics} >= {
        "syntax_invalid",
    }
    analytic_family = next(
        family_result
        for family_result in result.family_results
        if family_result.family_id == "analytic"
    )
    assert analytic_family.coverage.canonical_program_count == 2
    assert analytic_family.coverage.attempted_candidate_count == 2
    assert analytic_family.coverage.omitted_candidate_count == 0
    assert analytic_family.coverage.fallback_candidate_count == 1

    scope_metadata = result.best_overall_candidate.evidence_layer.transient_diagnostics[
        "descriptive_scope"
    ]
    assert scope_metadata == {
        "scope_rank": 1,
        "source": "fallback_bank",
        "selection_rule": (
            "min_total_code_bits_then_max_description_gain_then_"
            "min_structure_code_bits_then_min_canonical_byte_length_then_candidate_id"
        ),
        "law_eligible": False,
        "law_rejection_reason_codes": ["descriptive_only_fallback_bank"],
    }
    assert result.best_overall_candidate.evidence_layer.transient_diagnostics[
        "search_evidence"
    ] == {
        "search_class": "exact_finite_enumeration",
        "exactness_scope": (
            "finite_exactness_limited_to_declared_canonical_program_space"
        ),
        "fragment_bounds": "proposal_limit=2",
        "canonical_enumerator": "declared_adapter_family_order_then_candidate_id",
        "enumeration_cardinality": 2,
        "stop_rule": "exhaust_declared_canonical_program_space(cardinality=2)",
    }


def test_bounded_search_fallback_bank_counts_fallback_attempts_honestly() -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=("bad_syntax", "analytic_intercept"),
        search_class="bounded_heuristic",
        proposal_limit=1,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=(
            DescriptiveSearchProposal(
                candidate_id="bad_syntax",
                primitive_family="analytic",
                form_class="closed_form_expression",
                composition_payload={"operator_id": "unknown_operator"},
            ),
        ),
    )

    assert result.accepted_candidates == ()
    assert (
        result.best_overall_candidate
        .evidence_layer.backend_origin_record.source_candidate_id
        == "analytic_intercept"
    )
    assert result.coverage.canonical_program_count == 2
    assert result.coverage.attempted_candidate_count == 2
    assert result.coverage.omitted_candidate_count == 0
    assert result.coverage.fallback_candidate_count == 1
    analytic_family = next(
        family_result
        for family_result in result.family_results
        if family_result.family_id == "analytic"
    )
    assert analytic_family.coverage.canonical_program_count == 2
    assert analytic_family.coverage.attempted_candidate_count == 2
    assert analytic_family.coverage.omitted_candidate_count == 0
    assert analytic_family.coverage.fallback_candidate_count == 1


def test_bounded_search_retries_compact_fallback_omitted_by_budget() -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=("bad_syntax", "recursive_level_smoother"),
        search_class="bounded_heuristic",
        proposal_limit=1,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=True,
        proposal_specs=(
            DescriptiveSearchProposal(
                candidate_id="bad_syntax",
                primitive_family="analytic",
                form_class="closed_form_expression",
                composition_payload={"operator_id": "unknown_operator"},
            ),
        ),
    )

    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.descriptive_scope
    ] == ["recursive_level_smoother"]
    assert result.gap_report_reason_codes == ("descriptive_only_fallback_bank",)
    assert result.coverage.fallback_candidate_count == 1


def test_bounded_search_retries_compact_fallback_shadowed_by_bad_user_proposal(
) -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=("analytic_intercept",),
        search_class="bounded_heuristic",
        proposal_limit=1,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=(
            DescriptiveSearchProposal(
                candidate_id="analytic_intercept",
                primitive_family="analytic",
                form_class="closed_form_expression",
                composition_payload={"operator_id": "unknown_operator"},
            ),
        ),
    )

    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.descriptive_scope
    ] == ["analytic_intercept"]
    assert result.gap_report_reason_codes == ("descriptive_only_fallback_bank",)
    assert result.coverage.fallback_candidate_count == 1


def test_exact_fallback_candidates_count_against_budget_guard() -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=("bad_syntax", "analytic_intercept"),
        search_class="exact_finite_enumeration",
        proposal_limit=1,
    )

    with pytest.raises(ContractValidationError) as exc_info:
        run_descriptive_search_backends(
            search_plan=search_plan,
            feature_view=feature_view,
            include_default_grammar=False,
            proposal_specs=(
                DescriptiveSearchProposal(
                    candidate_id="bad_syntax",
                    primitive_family="analytic",
                    form_class="closed_form_expression",
                    composition_payload={"operator_id": "unknown_operator"},
                ),
            ),
        )

    assert exc_info.value.code == "exact_search_budget_too_small"
    assert exc_info.value.field_path == "search_plan.proposal_limit"
    assert exc_info.value.details == {
        "proposal_limit": 1,
        "canonical_program_count": 2,
    }


def test_inner_frontier_scoring_uses_common_entity_local_training_span_for_ragged_panels() -> None:
    feature_view, audit = _ragged_panel_feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=("analytic_intercept",),
        search_class="exact_finite_enumeration",
        proposal_limit=1,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )

    candidate = result.descriptive_scope[0]
    assert (
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        == "analytic_intercept"
    )
    assert "inner_primary_score" in candidate.evidence_layer.transient_diagnostics
    assert result.frontier.coverage.comparable_axes == (
        "structure_code_bits",
        "description_gain_bits",
        "inner_primary_score",
    )


def _feature_view(
    values: tuple[float, ...] | None = None,
    *,
    feature_names: tuple[str, ...] = ("lag_1",),
):
    values = values or (10.0, 12.0, 13.0, 15.0, 16.0, 18.0)
    snapshot = FrozenDatasetSnapshot(
        series_id="demo-series",
        cutoff_available_at=f"2026-01-{len(values):02d}T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-{index + 1:02d}T00:00:00Z",
                available_at=f"2026-01-{index + 1:02d}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:{index}",
            )
            for index, value in enumerate(values)
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    feature_spec = FeatureSpec(
        feature_spec_id="test_feature_spec_v1",
        features=tuple(
            {"feature_id": feature_name, "kind": "lag", "lag_steps": index + 1}
            for index, feature_name in enumerate(feature_names)
        ),
    )
    return materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=feature_spec if feature_names != ("lag_1",) else default_feature_spec(),
    ), audit


def _ragged_panel_feature_view():
    rows = []
    values_by_entity = {
        "entity-a": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
        "entity-b": (20.0, 21.0, 22.0, 23.0, 24.0),
    }
    for entity, values in values_by_entity.items():
        for index, value in enumerate(values):
            rows.append(
                SnapshotRow(
                    entity=entity,
                    event_time=f"2026-01-{index + 1:02d}T00:00:00Z",
                    available_at=f"2026-01-{index + 1:02d}T00:00:00Z",
                    observed_value=value,
                    revision_id=0,
                    payload_hash=f"sha256:{entity}:{index}",
                )
            )
    snapshot = FrozenDatasetSnapshot(
        series_id="ragged-panel-demo",
        cutoff_available_at="2026-01-06T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(rows),
        entity_panel=("entity-a", "entity-b"),
    )
    audit = audit_snapshot_time_safety(snapshot)
    return materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    ), audit


def _build_search_plan(
    *,
    feature_view,
    audit,
    candidate_family_ids: tuple[str, ...],
    search_class: str,
    proposal_limit: int,
    random_seed: str = "0",
    candidate_batch_size: int = 1,
    minimum_description_gain_bits: float | None = None,
    data_code_family: str | None = None,
    reference_policy: dict[str, object] | None = None,
    seasonal_period: int = 4,
):
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    canonicalization_policy = build_canonicalization_policy()
    return build_search_plan(
        evaluation_plan=evaluation_plan,
        canonicalization_policy_ref=TypedRef(
            "canonicalization_policy_manifest@1.0.0",
            canonicalization_policy.canonicalization_policy_id,
        ),
        codelength_policy_ref=TypedRef(
            "codelength_policy_manifest@1.1.0",
            "mdl_policy_default",
        ),
        reference_description_policy_ref=TypedRef(
            "reference_description_policy_manifest@1.1.0",
            "reference_description_default",
        ),
        observation_model_ref=TypedRef(
            "observation_model_manifest@1.1.0",
            "observation_model_default",
        ),
        candidate_family_ids=candidate_family_ids,
        search_class=search_class,
        proposal_limit=proposal_limit,
        candidate_batch_size=candidate_batch_size,
        random_seed=random_seed,
        minimum_description_gain_bits=minimum_description_gain_bits,
        seasonal_period=seasonal_period,
        data_code_family=data_code_family,
        reference_policy=reference_policy,
    )
