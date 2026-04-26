from __future__ import annotations

from euclid.contracts.refs import TypedRef
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.search.backends import DescriptiveSearchProposal
from euclid.search.portfolio import run_descriptive_search_portfolio

DEFAULT_PORTFOLIO_CANDIDATE_IDS = (
    "analytic_intercept",
    "analytic_lag1_affine",
    "recursive_level_smoother",
    "recursive_running_mean",
    "spectral_harmonic_1",
    "spectral_harmonic_2",
    "algorithmic_last_observation",
    "algorithmic_running_half_average",
)


def test_portfolio_selects_best_family_finalist_and_records_machine_readable_decision(
) -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=DEFAULT_PORTFOLIO_CANDIDATE_IDS,
        search_class="exact_finite_enumeration",
        proposal_limit=len(DEFAULT_PORTFOLIO_CANDIDATE_IDS),
        seasonal_period=4,
    )

    result = run_descriptive_search_portfolio(
        search_plan=search_plan,
        feature_view=feature_view,
    )

    assert (
        result.selected_candidate.evidence_layer.backend_origin_record.source_candidate_id
        == "analytic_lag1_affine"
    )
    assert result.selection_record.selected_candidate_id == "analytic_lag1_affine"
    assert result.selection_record.selected_backend_family == "analytic"
    assert result.selection_record.selection_scope == "shared_planning_cir_only"
    assert result.selection_record.selection_rule == (
        "min_total_code_bits_then_max_description_gain_then_"
        "min_structure_code_bits_then_min_canonical_byte_length_then_candidate_id"
    )
    assert {
        finalist["family_id"] for finalist in result.selection_record.compared_finalists
    } == {"analytic", "recursive", "algorithmic"}
    assert [step["step"] for step in result.selection_record.decision_trace] == [
        "collect_family_finalists",
        "rank_family_finalists",
        "select_portfolio_winner",
    ]

    ledgers = {ledger.family_id: ledger for ledger in result.backend_ledgers}
    assert ledgers["analytic"].finalist_candidate_id == "analytic_lag1_affine"
    assert [
        entry.candidate_id
        for entry in ledgers["analytic"].candidate_ledger
        if entry.ledger_status == "accepted"
    ] == ["analytic_intercept", "analytic_lag1_affine"]
    assert {
        entry.candidate_id: entry.ledger_status
        for entry in ledgers["algorithmic"].candidate_ledger
    } == {
        "algorithmic_last_observation": "accepted",
        "algorithmic_running_half_average": "rejected",
    }
    assert all(
        entry.ledger_status == "rejected"
        for entry in ledgers["spectral"].candidate_ledger
    )
    assert ledgers["recursive"].budget_consumption == {
        "canonical_program_count": 2,
        "attempted_candidate_count": 2,
        "accepted_candidate_count": 1,
        "rejected_candidate_count": 1,
        "omitted_candidate_count": 0,
    }
    assert (
        ledgers["algorithmic"].scope_declaration
        == "finite_exactness_limited_to_declared_canonical_program_space"
    )


def test_portfolio_ledgers_preserve_backend_omissions_and_search_honesty() -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=DEFAULT_PORTFOLIO_CANDIDATE_IDS,
        search_class="bounded_heuristic",
        proposal_limit=3,
        seasonal_period=4,
    )

    result = run_descriptive_search_portfolio(
        search_plan=search_plan,
        feature_view=feature_view,
    )

    ledgers = {ledger.family_id: ledger for ledger in result.backend_ledgers}
    assert {
        entry.candidate_id: entry.ledger_status
        for entry in ledgers["recursive"].candidate_ledger
    } == {
        "recursive_level_smoother": "accepted",
        "recursive_running_mean": "omitted",
    }
    assert {entry.candidate_id for entry in ledgers["spectral"].candidate_ledger} == {
        "spectral_harmonic_1",
        "spectral_harmonic_2",
    }
    assert all(
        entry.ledger_status == "omitted"
        for entry in ledgers["algorithmic"].candidate_ledger
    )
    assert ledgers["algorithmic"].coverage_statement == "incomplete_search_disclosed"
    assert ledgers["algorithmic"].exactness_ceiling == "no_global_exactness_claim"
    assert (
        ledgers["algorithmic"].scope_declaration
        == "heuristic_prefix_over_declared_candidate_space"
    )
    assert result.selection_record.selected_candidate_id == "analytic_lag1_affine"


def test_portfolio_excludes_exact_closure_and_posthoc_symbolic_candidates_from_finalists(
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
        seasonal_period=4,
    )

    result = run_descriptive_search_portfolio(
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

    assert (
        result.selected_candidate.evidence_layer.backend_origin_record.source_candidate_id
        == "analytic_intercept"
    )
    assert result.selection_record.selected_candidate_id == "analytic_intercept"
    assert result.selection_record.selected_backend_family == "analytic"
    assert [item["candidate_id"] for item in result.selection_record.compared_finalists] == [
        "analytic_intercept"
    ]

    analytic_ledger = next(
        ledger for ledger in result.backend_ledgers if ledger.family_id == "analytic"
    )
    entries = {entry.candidate_id: entry for entry in analytic_ledger.candidate_ledger}
    assert analytic_ledger.finalist_candidate_id == "analytic_intercept"
    assert entries["analytic_intercept"].ledger_status == "accepted"
    assert entries["analytic_closure_candidate"].ledger_status == "rejected"
    assert entries["analytic_closure_candidate"].reason_codes == (
        "descriptive_scope_excluded",
    )
    assert (
        entries["analytic_closure_candidate"].details["diagnostics"][0][
            "reason_codes"
        ]
        == ["requires_exact_sample_closure"]
    )
    assert entries["analytic_symbolic_candidate"].ledger_status == "rejected"
    assert entries["analytic_symbolic_candidate"].reason_codes == (
        "descriptive_scope_excluded",
    )
    assert (
        entries["analytic_symbolic_candidate"].details["diagnostics"][0][
            "reason_codes"
        ]
        == ["requires_posthoc_symbolic_synthesis"]
    )


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="portfolio-series",
        cutoff_available_at="2026-01-06T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=(
            SnapshotRow(
                event_time="2026-01-01T00:00:00Z",
                available_at="2026-01-01T00:00:00Z",
                observed_value=10.0,
                revision_id=0,
                payload_hash="sha256:a",
            ),
            SnapshotRow(
                event_time="2026-01-02T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                observed_value=12.0,
                revision_id=0,
                payload_hash="sha256:b",
            ),
            SnapshotRow(
                event_time="2026-01-03T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                observed_value=13.0,
                revision_id=0,
                payload_hash="sha256:c",
            ),
            SnapshotRow(
                event_time="2026-01-04T00:00:00Z",
                available_at="2026-01-04T00:00:00Z",
                observed_value=15.0,
                revision_id=0,
                payload_hash="sha256:d",
            ),
            SnapshotRow(
                event_time="2026-01-05T00:00:00Z",
                available_at="2026-01-05T00:00:00Z",
                observed_value=16.0,
                revision_id=0,
                payload_hash="sha256:e",
            ),
            SnapshotRow(
                event_time="2026-01-06T00:00:00Z",
                available_at="2026-01-06T00:00:00Z",
                observed_value=18.0,
                revision_id=0,
                payload_hash="sha256:f",
            ),
        ),
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
    seasonal_period: int,
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
        seasonal_period=seasonal_period,
    )
