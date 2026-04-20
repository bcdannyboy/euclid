from __future__ import annotations

from pathlib import Path

from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.modules.candidate_fitting import fit_candidate_window
from euclid.modules.evaluation import emit_point_prediction_artifact
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import EvaluationPlan, build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.search.backends import run_descriptive_search_backends
from euclid.search_planning import build_canonicalization_policy, build_search_plan

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_search_fit_handoff_emits_fold_local_point_prediction_artifact() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=2,
    )
    search_plan = _search_plan(evaluation_plan)
    score_policy = _score_policy_manifest(catalog, evaluation_plan)
    search_result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )
    candidate = search_result.accepted_candidates[0]
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

    assert artifact.schema_name == "prediction_artifact_manifest@1.1.0"
    assert (
        artifact.body["fit_window_id"]
        == evaluation_plan.development_segments[0].segment_id
    )
    assert (
        artifact.body["test_window_id"]
        == evaluation_plan.development_segments[0].segment_id
    )
    assert artifact.body["comparison_key"]["horizon_set"] == [1, 2]
    assert [row["horizon"] for row in artifact.body["rows"]] == [1, 2]
    assert artifact.body["missing_scored_origins"] == []


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="phase05-prediction-series",
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


def _search_plan(evaluation_plan: EvaluationPlan):
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
        candidate_family_ids=(
            "analytic_intercept",
            "analytic_lag1_affine",
            "recursive_level_smoother",
            "recursive_running_mean",
            "spectral_harmonic_1",
        ),
        proposal_limit=5,
        search_class="exact_finite_enumeration",
        seasonal_period=4,
    )


def _score_policy_manifest(
    catalog, evaluation_plan: EvaluationPlan
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": "phase05_prediction_policy_v1",
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
