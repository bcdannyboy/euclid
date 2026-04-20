from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.loader import load_contract_catalog
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    DistributionPredictionRow,
    EventProbabilityPredictionRow,
    IntervalPredictionRow,
    PredictionArtifactManifest,
    QuantilePredictionRow,
    QuantileValue,
)
from euclid.modules.calibration import (
    build_calibration_contract,
    evaluate_prediction_calibration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("forecast_object_type", "thresholds"),
    (
        ("distribution", {"max_ks_distance": 1.0}),
        ("interval", {"max_abs_coverage_gap": 1.0}),
        ("quantile", {"max_abs_hit_balance_gap": 1.0}),
        ("event_probability", {"max_reliability_gap": 0.25}),
    ),
)
def test_calibration_verdicts_exist_for_all_probabilistic_types(
    forecast_object_type: str,
    thresholds: dict[str, float],
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    prediction_artifact = _prediction_artifact_manifest(
        catalog=catalog,
        forecast_object_type=forecast_object_type,
    )
    calibration_contract = build_calibration_contract(
        catalog=catalog,
        forecast_object_type=forecast_object_type,
        thresholds=thresholds,
    )

    calibration_result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=calibration_contract,
        prediction_artifact_manifest=prediction_artifact,
    )

    assert calibration_result.body["forecast_object_type"] == forecast_object_type
    assert calibration_result.body["status"] in {"passed", "failed"}
    assert calibration_result.body["gate_effect"] == (
        "required_for_probabilistic_publication"
    )
    assert isinstance(calibration_result.body["diagnostics"], list)
    assert calibration_result.body["diagnostics"]


def _score_policy_manifest(
    *,
    catalog,
    forecast_object_type: str,
) -> ManifestEnvelope:
    schema_name = {
        "distribution": "probabilistic_score_policy_manifest@1.0.0",
        "interval": "interval_score_policy_manifest@1.0.0",
        "quantile": "quantile_score_policy_manifest@1.0.0",
        "event_probability": "event_probability_score_policy_manifest@1.0.0",
    }[forecast_object_type]
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
            "score_policy_id": f"{forecast_object_type}_score_policy_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": forecast_object_type,
            "primary_score": primary_score,
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
) -> ManifestEnvelope:
    score_policy = _score_policy_manifest(
        catalog=catalog,
        forecast_object_type=forecast_object_type,
    )
    rows = {
        "distribution": (
            DistributionPredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                distribution_family="gaussian_location_scale",
                location=10.0,
                scale=1.0,
                support_kind="all_real",
                realized_observation=10.0,
            ),
            DistributionPredictionRow(
                origin_time="2026-01-02T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                horizon=1,
                distribution_family="gaussian_location_scale",
                location=11.0,
                scale=1.0,
                support_kind="all_real",
                realized_observation=11.2,
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
        score_law_id=str(score_policy.body["primary_score"]),
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
