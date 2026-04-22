from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.math.observation_models import PointObservationModel
from euclid.modules.candidate_fitting import fit_candidate_window
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.probabilistic_evaluation import (
    emit_probabilistic_prediction_artifact,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import EvaluationPlan, build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.reducers.models import BoundObservationModel
from euclid.search.backends import (
    AlgorithmicSearchBackendAdapter,
    AnalyticSearchBackendAdapter,
    RecursiveSearchBackendAdapter,
    SpectralSearchBackendAdapter,
    run_descriptive_search_backends,
)
from euclid.search_planning import build_canonicalization_policy, build_search_plan

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_SCORE_POLICY_SCHEMAS = {
    "distribution": "probabilistic_score_policy_manifest@1.0.0",
    "interval": "interval_score_policy_manifest@1.0.0",
    "quantile": "quantile_score_policy_manifest@1.0.0",
    "event_probability": "event_probability_score_policy_manifest@1.0.0",
}


@pytest.mark.parametrize(
    ("candidate_id", "forecast_object_type", "required_fields"),
    (
        (
            "analytic_lag1_affine",
            "distribution",
            ("distribution_family", "location", "scale", "support_kind"),
        ),
        (
            "recursive_level_smoother",
            "interval",
            ("nominal_coverage", "lower_bound", "upper_bound"),
        ),
        (
            "spectral_harmonic_1",
            "quantile",
            ("quantiles",),
        ),
        (
            "algorithmic_last_observation",
            "event_probability",
            ("event_definition", "event_probability", "realized_event"),
        ),
    ),
)
def test_emit_probabilistic_prediction_artifact_preserves_forecast_object_typing(
    candidate_id: str,
    forecast_object_type: str,
    required_fields: tuple[str, ...],
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view(candidate_id)
    min_train_size = 4 if candidate_id.startswith("spectral_") else 3
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=min_train_size,
        horizon=2,
    )
    search_plan = _search_plan(evaluation_plan)
    score_policy = _probabilistic_score_policy_manifest(
        catalog=catalog,
        evaluation_plan=evaluation_plan,
        forecast_object_type=forecast_object_type,
    )
    candidate = _candidate(feature_view, search_plan, candidate_id)
    fit_result = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )

    artifact = emit_probabilistic_prediction_artifact(
        catalog=catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.development_segments[0],
        fit_result=fit_result,
        score_policy_manifest=score_policy,
        stage_id="outer_test",
        forecast_object_type=forecast_object_type,
    )

    assert artifact.body["forecast_object_type"] == forecast_object_type
    assert artifact.body["comparison_key"] == {
        "forecast_object_type": forecast_object_type,
        "horizon_set": [1, 2],
        "score_law_id": artifact.body["score_law_id"],
        "scored_origin_set_id": artifact.body["scored_origin_set_id"],
    }
    assert (
        artifact.body["score_policy_ref"]["schema_name"]
        == _SCORE_POLICY_SCHEMAS[forecast_object_type]
    )
    assert artifact.body["missing_scored_origins"] == []
    assert all(
        check["status"] == "passed" for check in artifact.body["timeguard_checks"]
    )
    assert len(artifact.body["rows"]) == 2
    for row in artifact.body["rows"]:
        assert row["horizon"] in {1, 2}
        assert "realized_observation" in row
        assert "point_forecast" not in row
        for field_name in required_fields:
            assert field_name in row
        if forecast_object_type == "event_probability":
            assert (
                row["event_definition"]["threshold_source"]
                == "declared_literal"
            )
            assert row["event_definition"]["event_id"] != "target_ge_origin_target"


def test_emit_probabilistic_prediction_artifact_rejects_wrong_policy_type() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view("analytic_lag1_affine")
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=2,
    )
    search_plan = _search_plan(evaluation_plan)
    interval_policy = _probabilistic_score_policy_manifest(
        catalog=catalog,
        evaluation_plan=evaluation_plan,
        forecast_object_type="interval",
    )
    candidate = _candidate(feature_view, search_plan, "analytic_lag1_affine")
    fit_result = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )

    with pytest.raises(ContractValidationError) as excinfo:
        emit_probabilistic_prediction_artifact(
            catalog=catalog,
            feature_view=feature_view,
            evaluation_plan=evaluation_plan,
            evaluation_segment=evaluation_plan.development_segments[0],
            fit_result=fit_result,
            score_policy_manifest=interval_policy,
            stage_id="outer_test",
            forecast_object_type="distribution",
        )

    assert excinfo.value.code == "probabilistic_policy_forecast_object_type_mismatch"


def _feature_view(candidate_id: str):
    observations = (
        (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0)
        if candidate_id.startswith("spectral_")
        else (1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0)
    )
    snapshot = FrozenDatasetSnapshot(
        series_id="probabilistic-evaluation-series",
        cutoff_available_at="2026-01-08T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-0{index + 1}T00:00:00Z",
                available_at=f"2026-01-0{index + 1}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:{chr(ord('a') + index)}",
            )
            for index, value in enumerate(observations)
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
            "analytic_lag1_affine",
            "recursive_level_smoother",
            "spectral_harmonic_1",
            "algorithmic_last_observation",
        ),
        proposal_limit=4,
        search_class="exact_finite_enumeration",
        seasonal_period=4,
        minimum_description_gain_bits=-2.0,
    )


def _candidate(feature_view, search_plan, candidate_id: str):
    adapter = _adapter_for_candidate(candidate_id)
    if adapter is not None:
        observation_model = BoundObservationModel.from_runtime(PointObservationModel())
        for rank, proposal in enumerate(
            adapter.default_proposals(
                search_plan=search_plan,
                feature_view=feature_view,
            )
        ):
            if proposal.candidate_id != candidate_id:
                continue
            return adapter.realize_proposal(
                proposal=proposal,
                proposal_rank=rank,
                search_plan=search_plan,
                feature_view=feature_view,
                observation_model=observation_model,
            )

    search_result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )
    for candidate in search_result.accepted_candidates:
        if (
            candidate.evidence_layer.backend_origin_record.source_candidate_id
            == candidate_id
        ):
            return candidate
    raise AssertionError(f"missing candidate {candidate_id}")


def _adapter_for_candidate(candidate_id: str):
    if candidate_id.startswith("analytic_"):
        return AnalyticSearchBackendAdapter()
    if candidate_id.startswith("recursive_"):
        return RecursiveSearchBackendAdapter()
    if candidate_id.startswith("spectral_"):
        return SpectralSearchBackendAdapter()
    if candidate_id.startswith("algorithmic_"):
        return AlgorithmicSearchBackendAdapter()
    return None


def _probabilistic_score_policy_manifest(
    *,
    catalog,
    evaluation_plan: EvaluationPlan,
    forecast_object_type: str,
) -> ManifestEnvelope:
    schema_name = _SCORE_POLICY_SCHEMAS[forecast_object_type]
    primary_score = {
        "distribution": "continuous_ranked_probability_score",
        "interval": "interval_score",
        "quantile": "pinball_loss",
        "event_probability": "brier_score",
    }[forecast_object_type]
    return ManifestEnvelope.build(
        schema_name=schema_name,
        module_id="scoring",
        body={
            "score_policy_id": f"test_{forecast_object_type}_policy_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": forecast_object_type,
            "primary_score": primary_score,
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [
                weight.as_dict() for weight in evaluation_plan.horizon_weights
            ],
            "entity_aggregation_mode": (
                "single_entity_only_no_cross_entity_aggregation"
            ),
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )
