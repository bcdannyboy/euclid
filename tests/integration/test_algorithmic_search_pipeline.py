from __future__ import annotations

from pathlib import Path

from euclid.adapters.algorithmic_dsl import enumerate_algorithmic_proposal_specs
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.modules.candidate_fitting import fit_candidate_window
from euclid.modules.evaluation import emit_point_prediction_artifact
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.search.backends import (
    AlgorithmicSearchBackendAdapter,
    DescriptiveSearchProposal,
    run_descriptive_search_backends,
)
from euclid.search.dsl.parser import parse_algorithmic_program

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_algorithmic_search_pipeline_enumerates_candidates_and_fits_via_cir() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=2,
    )
    proposal_specs = enumerate_algorithmic_proposal_specs()
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
        candidate_family_ids=tuple(spec.candidate_id for spec in proposal_specs),
        search_class="exact_finite_enumeration",
        proposal_limit=len(proposal_specs),
    )
    score_policy = _score_policy_manifest(catalog, evaluation_plan)

    first = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        adapters=(AlgorithmicSearchBackendAdapter(),),
    )
    second = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        adapters=(AlgorithmicSearchBackendAdapter(),),
    )

    assert first.coverage.canonical_program_count == len(proposal_specs)
    assert first.family_results[0].coverage.canonical_program_count == len(
        proposal_specs
    )
    assert [
        candidate.canonical_hash() for candidate in first.accepted_candidates
    ] == [candidate.canonical_hash() for candidate in second.accepted_candidates]

    candidate = next(
        candidate
        for candidate in first.accepted_candidates
        if candidate.evidence_layer.backend_origin_record.source_candidate_id
        == "algorithmic_last_observation"
    )
    literal_names = {
        literal.name for literal in candidate.structural_layer.literal_block.literals
    }
    assert candidate.structural_layer.cir_family_id == "algorithmic"
    assert {
        "algorithmic_dsl_id",
        "algorithmic_program",
        "program_node_count",
    }.issubset(literal_names)
    assert candidate.execution_layer.history_access_contract.contract_id.startswith(
        "algorithmic_history__slots_1__lags_0"
    )

    fit_result = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )
    artifact = emit_point_prediction_artifact(
        catalog=catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.development_segments[0],
        fit_result=fit_result,
        score_policy_manifest=score_policy,
        stage_id="outer_test",
    )

    assert fit_result.family_id == "algorithmic"
    assert fit_result.final_state == {"state_0": "15"}
    assert [
        (row["horizon"], row["point_forecast"]) for row in artifact.body["rows"]
    ] == [
        (1, 15.0),
        (2, 15.0),
    ]


def test_algorithmic_pipeline_replays_v1_fragment_candidate() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=2,
    )
    program = parse_algorithmic_program(
        (
            "(program "
            "(state (lit 0) (lit 0)) "
            "(next "
            "(obs 1) "
            "(if (not false) (obs 0) (obs 1))) "
            "(emit (state 1)))"
        ),
        state_slot_count=2,
        max_program_nodes=9,
        allowed_observation_lags=(0, 1),
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
        candidate_family_ids=("algorithmic_conditional_v1",),
        search_class="exact_finite_enumeration",
        proposal_limit=1,
    )
    score_policy = _score_policy_manifest(catalog, evaluation_plan)
    proposal = DescriptiveSearchProposal(
        candidate_id="algorithmic_conditional_v1",
        primitive_family="algorithmic",
        form_class="bounded_program",
        literal_values={
            "algorithmic_program": program.canonical_source,
            "algorithmic_state_slot_count": program.state_slot_count,
            "program_node_count": program.node_count,
        },
        history_access_mode="bounded_lag_window",
        max_lag=1,
    )

    first = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        adapters=(AlgorithmicSearchBackendAdapter(),),
        include_default_grammar=False,
        proposal_specs=(proposal,),
    )
    second = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        adapters=(AlgorithmicSearchBackendAdapter(),),
        include_default_grammar=False,
        proposal_specs=(proposal,),
    )
    first_candidate = first.accepted_candidates[0]
    second_candidate = second.accepted_candidates[0]

    first_fit = fit_candidate_window(
        candidate=first_candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )
    second_fit = fit_candidate_window(
        candidate=second_candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )
    first_artifact = emit_point_prediction_artifact(
        catalog=catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.development_segments[0],
        fit_result=first_fit,
        score_policy_manifest=score_policy,
        stage_id="outer_test",
    )
    second_artifact = emit_point_prediction_artifact(
        catalog=catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.development_segments[0],
        fit_result=second_fit,
        score_policy_manifest=score_policy,
        stage_id="outer_test",
    )

    literal_names = {
        literal.name
        for literal in first_candidate.structural_layer.literal_block.literals
    }

    assert first_candidate.structural_layer.cir_family_id == "algorithmic"
    assert {
        "algorithmic_state_slot_count",
        "algorithmic_allowed_observation_lags",
    } <= literal_names
    assert first_fit.final_state == second_fit.final_state
    assert first_artifact.body["rows"] == second_artifact.body["rows"]


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="phase06-algorithmic-series",
        cutoff_available_at="2026-01-07T00:00:00Z",
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
            SnapshotRow(
                event_time="2026-01-07T00:00:00Z",
                available_at="2026-01-07T00:00:00Z",
                observed_value=19.0,
                revision_id=0,
                payload_hash="sha256:g",
            ),
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    return materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    ), audit


def _score_policy_manifest(catalog, evaluation_plan):
    from euclid.manifests.base import ManifestEnvelope

    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": "phase06_algorithmic_prediction_policy_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": "point",
            "point_loss_id": "absolute_error",
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [
                weight.as_dict() for weight in evaluation_plan.horizon_weights
            ],
            "entity_aggregation_mode": (
                "single_entity_only_no_cross_entity_aggregation"
            ),
            "secondary_diagnostic_ids": [],
            "forbidden_primary_metric_ids": [],
            "lower_is_better": True,
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )
