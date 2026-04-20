from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    DistributionPredictionRow,
    EventProbabilityPredictionRow,
    IntervalPredictionRow,
    PerHorizonPrimaryScore,
    PerHorizonScore,
    PointScoreResultManifest,
    PredictionArtifactManifest,
    PredictionRow,
    ProbabilisticScoreResultManifest,
    QuantilePredictionRow,
    QuantileValue,
)
from euclid.modules.scoring import (
    score_point_prediction_artifact,
    score_probabilistic_prediction_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ADMITTED_OBJECT_TYPES = (
    "point",
    "distribution",
    "interval",
    "quantile",
    "event_probability",
)


@pytest.mark.parametrize("forecast_object_type", ADMITTED_OBJECT_TYPES)
def test_each_object_type_accepts_only_admitted_score_laws(
    forecast_object_type: str,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    valid_artifact = _prediction_artifact_manifest(
        catalog=catalog,
        forecast_object_type=forecast_object_type,
        score_policy=_score_policy_manifest(
            catalog=catalog,
            forecast_object_type=forecast_object_type,
            score_law_id=_valid_score_law_id(forecast_object_type),
        ),
    )

    if forecast_object_type == "point":
        valid_result = score_point_prediction_artifact(
            catalog=catalog,
            score_policy_manifest=_score_policy_manifest(
                catalog=catalog,
                forecast_object_type="point",
                score_law_id="absolute_error",
            ),
            prediction_artifact_manifest=valid_artifact,
        )
        invalid_policy = _score_policy_manifest(
            catalog=catalog,
            forecast_object_type="point",
            score_law_id="log_score",
        )
        assert valid_result.body["comparison_status"] == "comparable"
        with pytest.raises(ContractValidationError) as exc_info:
            score_point_prediction_artifact(
                catalog=catalog,
                score_policy_manifest=invalid_policy,
                prediction_artifact_manifest=_prediction_artifact_manifest(
                    catalog=catalog,
                    forecast_object_type="point",
                    score_policy=invalid_policy,
                ),
            )
        assert exc_info.value.code == "unsupported_point_loss_id"
        return

    valid_policy = _score_policy_manifest(
        catalog=catalog,
        forecast_object_type=forecast_object_type,
        score_law_id=_valid_score_law_id(forecast_object_type),
    )
    valid_result = score_probabilistic_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=valid_policy,
        prediction_artifact_manifest=valid_artifact,
    )
    invalid_policy = _score_policy_manifest(
        catalog=catalog,
        forecast_object_type=forecast_object_type,
        score_law_id=_invalid_score_law_id(forecast_object_type),
    )

    assert valid_result.body["comparison_status"] == "comparable"
    with pytest.raises(ContractValidationError) as exc_info:
        score_probabilistic_prediction_artifact(
            catalog=catalog,
            score_policy_manifest=invalid_policy,
            prediction_artifact_manifest=_prediction_artifact_manifest(
                catalog=catalog,
                forecast_object_type=forecast_object_type,
                score_policy=invalid_policy,
            ),
        )
    assert exc_info.value.code == "unsupported_primary_score_id"


@pytest.mark.parametrize("forecast_object_type", ADMITTED_OBJECT_TYPES)
def test_score_result_roundtrip_for_all_object_types(
    forecast_object_type: str,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(
        catalog=catalog,
        forecast_object_type=forecast_object_type,
        score_law_id=_valid_score_law_id(forecast_object_type),
    )
    prediction_artifact = _prediction_artifact_manifest(
        catalog=catalog,
        forecast_object_type=forecast_object_type,
        score_policy=score_policy,
    )
    if forecast_object_type == "point":
        score_result = score_point_prediction_artifact(
            catalog=catalog,
            score_policy_manifest=score_policy,
            prediction_artifact_manifest=prediction_artifact,
        )
        model = PointScoreResultManifest(
            object_id=score_result.object_id,
            score_result_id=str(score_result.body["score_result_id"]),
            score_policy_ref=score_policy.ref,
            prediction_artifact_ref=prediction_artifact.ref,
            forecast_object_type="point",
            per_horizon=tuple(
                PerHorizonScore(
                    horizon=int(item["horizon"]),
                    valid_origin_count=int(item["valid_origin_count"]),
                    mean_point_loss=float(item["mean_point_loss"]),
                )
                for item in score_result.body["per_horizon"]
            ),
            aggregated_primary_score=float(
                score_result.body["aggregated_primary_score"]
            ),
            comparison_status=str(score_result.body["comparison_status"]),
            failure_reason_code=score_result.body["failure_reason_code"],
        )
    else:
        score_result = score_probabilistic_prediction_artifact(
            catalog=catalog,
            score_policy_manifest=score_policy,
            prediction_artifact_manifest=prediction_artifact,
        )
        model = ProbabilisticScoreResultManifest(
            object_id=score_result.object_id,
            score_result_id=str(score_result.body["score_result_id"]),
            score_policy_ref=score_policy.ref,
            prediction_artifact_ref=prediction_artifact.ref,
            forecast_object_type=forecast_object_type,
            per_horizon=tuple(
                PerHorizonPrimaryScore(
                    horizon=int(item["horizon"]),
                    valid_origin_count=int(item["valid_origin_count"]),
                    mean_primary_score=float(item["mean_primary_score"]),
                )
                for item in score_result.body["per_horizon"]
            ),
            aggregated_primary_score=float(
                score_result.body["aggregated_primary_score"]
            ),
            comparison_status=str(score_result.body["comparison_status"]),
            failure_reason_code=score_result.body["failure_reason_code"],
        )

    roundtripped = model.to_manifest(catalog)
    assert roundtripped.schema_name == score_result.schema_name
    assert roundtripped.body == score_result.body
    assert roundtripped.body["forecast_object_type"] == forecast_object_type


def _score_policy_manifest(
    *,
    catalog,
    forecast_object_type: str,
    score_law_id: str,
) -> ManifestEnvelope:
    if forecast_object_type == "point":
        return ManifestEnvelope.build(
            schema_name="point_score_policy_manifest@1.0.0",
            module_id="scoring",
            body={
                "score_policy_id": "point_score_policy_v1",
                "owner_prompt_id": "prompt.scoring-calibration-v1",
                "scope_id": "euclid_v1_binding_scope@1.0.0",
                "forecast_object_type": "point",
                "point_loss_id": score_law_id,
                "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
                "horizon_weights": [{"horizon": 1, "weight": "1.0"}],
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
    schema_name = {
        "distribution": "probabilistic_score_policy_manifest@1.0.0",
        "interval": "interval_score_policy_manifest@1.0.0",
        "quantile": "quantile_score_policy_manifest@1.0.0",
        "event_probability": "event_probability_score_policy_manifest@1.0.0",
    }[forecast_object_type]
    return ManifestEnvelope.build(
        schema_name=schema_name,
        module_id="scoring",
        body={
            "score_policy_id": f"{forecast_object_type}_score_policy_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": forecast_object_type,
            "primary_score": score_law_id,
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [{"horizon": 1, "weight": "1.0"}],
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


def _prediction_artifact_manifest(
    *,
    catalog,
    forecast_object_type: str,
    score_policy: ManifestEnvelope,
) -> ManifestEnvelope:
    rows = {
        "point": (
            PredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                point_forecast=10.0,
                realized_observation=11.0,
            ),
            PredictionRow(
                origin_time="2026-01-02T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                horizon=1,
                point_forecast=12.0,
                realized_observation=12.0,
            ),
        ),
        "distribution": (
            DistributionPredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                distribution_family="gaussian_location_scale",
                location=10.0,
                scale=1.0,
                support_kind="all_real",
                realized_observation=10.5,
            ),
            DistributionPredictionRow(
                origin_time="2026-01-02T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                horizon=1,
                distribution_family="gaussian_location_scale",
                location=11.0,
                scale=1.0,
                support_kind="all_real",
                realized_observation=11.0,
            ),
        ),
        "interval": (
            IntervalPredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                nominal_coverage=0.8,
                lower_bound=9.0,
                upper_bound=11.0,
                realized_observation=10.0,
            ),
            IntervalPredictionRow(
                origin_time="2026-01-02T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                horizon=1,
                nominal_coverage=0.8,
                lower_bound=10.0,
                upper_bound=12.0,
                realized_observation=11.0,
            ),
        ),
        "quantile": (
            QuantilePredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                quantiles=(
                    QuantileValue(level=0.1, value=9.5),
                    QuantileValue(level=0.5, value=10.0),
                    QuantileValue(level=0.9, value=10.5),
                ),
                realized_observation=10.0,
            ),
            QuantilePredictionRow(
                origin_time="2026-01-02T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                horizon=1,
                quantiles=(
                    QuantileValue(level=0.1, value=10.5),
                    QuantileValue(level=0.5, value=11.0),
                    QuantileValue(level=0.9, value=11.5),
                ),
                realized_observation=11.0,
            ),
        ),
        "event_probability": (
            EventProbabilityPredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                event_definition={"kind": "greater_than", "threshold": 10.0},
                event_probability=0.2,
                realized_observation=9.0,
                realized_event=False,
            ),
            EventProbabilityPredictionRow(
                origin_time="2026-01-02T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                horizon=1,
                event_definition={"kind": "greater_than", "threshold": 10.0},
                event_probability=0.8,
                realized_observation=12.0,
                realized_event=True,
            ),
        ),
    }[forecast_object_type]
    return PredictionArtifactManifest(
        prediction_artifact_id=f"{forecast_object_type}_prediction",
        candidate_id=f"{forecast_object_type}_candidate",
        stage_id="confirmatory_holdout",
        fit_window_id="fit_window",
        test_window_id="confirmatory_segment",
        model_freeze_status="global_finalist_frozen",
        refit_rule_applied="pre_holdout_development_refit",
        score_policy_ref=score_policy.ref,
        rows=rows,
        forecast_object_type=forecast_object_type,
        score_law_id=_valid_score_law_id(forecast_object_type),
        horizon_weights=({"horizon": 1, "weight": "1.0"},),
        scored_origin_panel=tuple(
            {
                "scored_origin_id": f"{forecast_object_type}_origin_{index}",
                "origin_time": row.origin_time,
                "available_at": row.available_at,
                "horizon": row.horizon,
            }
            for index, row in enumerate(rows)
        ),
        scored_origin_set_id=f"{forecast_object_type}_panel",
        comparison_key={
            "forecast_object_type": forecast_object_type,
            "horizon_set": [1],
            "scored_origin_set_id": f"{forecast_object_type}_panel",
        },
        missing_scored_origins=(),
        timeguard_checks=tuple(
            {
                "scored_origin_id": f"{forecast_object_type}_origin_{index}",
                "expected_available_at": row.available_at,
                "observed_available_at": row.available_at,
                "status": "passed",
            }
            for index, row in enumerate(rows)
        ),
    ).to_manifest(catalog)


def _valid_score_law_id(forecast_object_type: str) -> str:
    return {
        "point": "absolute_error",
        "distribution": "continuous_ranked_probability_score",
        "interval": "interval_score",
        "quantile": "pinball_loss",
        "event_probability": "brier_score",
    }[forecast_object_type]


def _invalid_score_law_id(forecast_object_type: str) -> str:
    return {
        "distribution": "interval_score",
        "interval": "pinball_loss",
        "quantile": "brier_score",
        "event_probability": "interval_score",
    }[forecast_object_type]
