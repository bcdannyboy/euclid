from __future__ import annotations

from pathlib import Path

from euclid.adapters.decomposition import (
    LegacyDecompositionProposal,
    normalize_legacy_decomposition_candidate,
)
from euclid.adapters.sparse_library import (
    LegacySparseProposal,
    normalize_legacy_sparse_candidate,
)
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.math.observation_models import PointObservationModel
from euclid.modules.candidate_fitting import fit_candidate_window
from euclid.modules.evaluation import emit_point_prediction_artifact
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.scoring import score_point_prediction_artifact
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.reducers.models import BoundObservationModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_heterogeneous_adapter_candidates_flow_through_common_fit_and_scoring() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, evaluation_plan, search_plan = _context()
    observation_model = BoundObservationModel.from_runtime(PointObservationModel())
    score_policy = _score_policy_manifest(catalog, evaluation_plan)

    decomposition_candidate = normalize_legacy_decomposition_candidate(
        spec=LegacyDecompositionProposal(
            candidate_id="decomposition_lag1",
            feature_dependencies=("lag_1",),
            parameter_values={"intercept": 1.0, "lag_coefficient": 0.9},
            max_lag=1,
        ),
        search_plan=search_plan,
        feature_view=feature_view,
        observation_model=observation_model,
        proposal_rank=0,
    )
    sparse_candidate = normalize_legacy_sparse_candidate(
        spec=LegacySparseProposal(
            candidate_id="sparse_recursive",
            primitive_family="recursive",
            form_class="state_recurrence",
            literal_values={"alpha": 0.5},
            persistent_state={"level": 10.0, "step_count": 0},
            max_lag=1,
        ),
        search_plan=search_plan,
        feature_view=feature_view,
        observation_model=observation_model,
        proposal_rank=1,
    )

    for candidate in (decomposition_candidate, sparse_candidate):
        fit_result = fit_candidate_window(
            candidate=candidate,
            feature_view=feature_view,
            fit_window=evaluation_plan.confirmatory_segment,
            search_plan=search_plan,
            stage_id="confirmatory_holdout",
        )
        artifact = emit_point_prediction_artifact(
            catalog=catalog,
            feature_view=feature_view,
            evaluation_plan=evaluation_plan,
            evaluation_segment=evaluation_plan.confirmatory_segment,
            fit_result=fit_result,
            score_policy_manifest=score_policy,
            stage_id="confirmatory_holdout",
        )
        score_result = score_point_prediction_artifact(
            catalog=catalog,
            score_policy_manifest=score_policy,
            prediction_artifact_manifest=artifact,
        )

        assert artifact.body["forecast_object_type"] == "point"
        assert score_result.body["comparison_status"] == "comparable"
        assert (
            candidate.evidence_layer.transient_diagnostics["legacy_non_claim_adapter"][
                "production_evidence_allowed"
            ]
            is False
        )
        assert artifact.body["candidate_id"] in {
            "decomposition_lag1",
            "sparse_recursive",
        }


def _context():
    snapshot = FrozenDatasetSnapshot(
        series_id="multi-backend-cir-series",
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
    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )
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
        candidate_family_ids=("analytic_intercept", "analytic_lag1_affine"),
        search_class="exact_finite_enumeration",
        proposal_limit=2,
        seasonal_period=4,
    )
    return feature_view, evaluation_plan, search_plan


def _score_policy_manifest(catalog, evaluation_plan):
    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": "multi_backend_cir_policy",
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
