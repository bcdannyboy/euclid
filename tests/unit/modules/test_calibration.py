from __future__ import annotations

from pathlib import Path

import pytest
from scipy import stats

from euclid.contracts.loader import load_contract_catalog
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    DistributionPredictionRow,
    EventProbabilityPredictionRow,
    IntervalPredictionRow,
    PredictionArtifactManifest,
)
from euclid.modules.calibration import (
    build_calibration_contract,
    evaluate_prediction_calibration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_evaluate_prediction_calibration_marks_point_forecasts_not_applicable() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="point_candidate",
        forecast_object_type="point",
        score_policy=_score_policy_manifest(catalog, "point"),
        rows=(),
    )
    contract = build_calibration_contract(
        catalog=catalog,
        forecast_object_type="point",
    )

    result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=contract,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["forecast_object_type"] == "point"
    assert result.body["status"] == "not_applicable_for_forecast_type"
    assert result.body["pass"] is None
    assert result.body["gate_effect"] == "none"
    assert result.body["diagnostics"] == []


def test_interval_calibration_keeps_diagnostics_and_blocks_publication() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(catalog, "interval")
    artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="interval_candidate",
        forecast_object_type="interval",
        score_policy=score_policy,
        rows=(
            IntervalPredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                nominal_coverage=0.8,
                lower_bound=9.0,
                upper_bound=11.0,
                realized_observation=12.0,
            ),
            IntervalPredictionRow(
                origin_time="2026-01-02T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                horizon=1,
                nominal_coverage=0.8,
                lower_bound=9.0,
                upper_bound=11.0,
                realized_observation=12.5,
            ),
        ),
    )
    contract = build_calibration_contract(
        catalog=catalog,
        forecast_object_type="interval",
        thresholds={"max_abs_coverage_gap": 0.1},
    )

    result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=contract,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["status"] == "failed"
    assert result.body["pass"] is False
    assert result.body["gate_effect"] == "required_for_probabilistic_publication"
    assert result.body["failure_reason_code"] == "calibration_failed"
    assert result.body["diagnostics"] == [
        {
            "diagnostic_id": "nominal_coverage",
            "nominal_coverage": 0.8,
            "empirical_coverage": 0.0,
            "absolute_gap": 0.8,
            "status": "failed",
        }
    ]


def test_distribution_calibration_pit_matches_scipy_normal_cdf() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(catalog, "distribution")
    artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="distribution_candidate",
        forecast_object_type="distribution",
        score_policy=score_policy,
        rows=(
            DistributionPredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                distribution_family="gaussian",
                location=0.0,
                scale=2.0,
                support_kind="continuous",
                realized_observation=1.0,
            ),
        ),
    )
    contract = build_calibration_contract(
        catalog=catalog,
        forecast_object_type="distribution",
        thresholds={"max_ks_distance": 1.0},
    )

    result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=contract,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["diagnostics"][0]["mean_pit"] == pytest.approx(
        stats.norm.cdf(0.5)
    )


def test_evaluate_prediction_calibration_passes_reliable_event_probabilities() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(catalog, "event_probability")
    artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="event_candidate",
        forecast_object_type="event_probability",
        score_policy=score_policy,
        rows=(
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
    )
    contract = build_calibration_contract(
        catalog=catalog,
        forecast_object_type="event_probability",
        thresholds={"max_reliability_gap": 0.25},
    )

    result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=contract,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["status"] == "passed"
    assert result.body["pass"] is True
    assert result.body["failure_reason_code"] is None
    assert result.body["gate_effect"] == "required_for_probabilistic_publication"
    assert result.body["diagnostics"] == [
        {
            "diagnostic_id": "reliability_curve_or_binned_frequency",
            "bin_count": 2,
            "max_reliability_gap": 0.2,
            "status": "passed",
        }
    ]


def _score_policy_manifest(catalog, forecast_object_type: str) -> ManifestEnvelope:
    schema_name = {
        "point": "point_score_policy_manifest@1.0.0",
        "distribution": "probabilistic_score_policy_manifest@1.0.0",
        "interval": "interval_score_policy_manifest@1.0.0",
        "event_probability": "event_probability_score_policy_manifest@1.0.0",
    }[forecast_object_type]
    primary_score = {
        "point": "absolute_error",
        "distribution": "continuous_ranked_probability_score",
        "interval": "interval_score",
        "event_probability": "brier_score",
    }[forecast_object_type]
    body = {
        "score_policy_id": f"{forecast_object_type}_score_policy_v1",
        "owner_prompt_id": "prompt.scoring-calibration-v1",
        "scope_id": "euclid_v1_binding_scope@1.0.0",
        "forecast_object_type": forecast_object_type,
        "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
        "horizon_weights": [{"horizon": 1, "weight": "1.0"}],
        "entity_aggregation_mode": "single_entity_only_no_cross_entity_aggregation",
        "comparison_class_rule": "identical_score_policy_required",
    }
    if forecast_object_type == "point":
        body["point_loss_id"] = primary_score
        body["secondary_diagnostic_ids"] = []
        body["forbidden_primary_metric_ids"] = []
        body["lower_is_better"] = True
    else:
        body["primary_score"] = primary_score
        body["secondary_diagnostic_ids"] = []
        body["forbidden_primary_metric_ids"] = []
        body["lower_is_better"] = True
    return ManifestEnvelope.build(
        schema_name=schema_name,
        module_id="scoring",
        body=body,
        catalog=catalog,
    )


def _prediction_artifact(
    *,
    catalog,
    candidate_id: str,
    forecast_object_type: str,
    score_policy: ManifestEnvelope,
    rows,
) -> ManifestEnvelope:
    scored_origin_panel = [
        {
            "scored_origin_id": f"{candidate_id}_origin_{index}",
            "origin_time": row.origin_time,
            "available_at": row.available_at,
            "horizon": row.horizon,
        }
        for index, row in enumerate(rows)
    ]
    return PredictionArtifactManifest(
        prediction_artifact_id=f"{candidate_id}_prediction",
        candidate_id=candidate_id,
        stage_id="confirmatory_holdout",
        fit_window_id="fit_window",
        test_window_id="confirmatory_segment",
        model_freeze_status="global_finalist_frozen",
        refit_rule_applied="pre_holdout_development_refit",
        score_policy_ref=score_policy.ref,
        rows=tuple(rows),
        forecast_object_type=forecast_object_type,
        score_law_id=str(
            score_policy.body.get(
                "primary_score", score_policy.body.get("point_loss_id")
            )
        ),
        horizon_weights=({"horizon": 1, "weight": "1.0"},),
        scored_origin_panel=tuple(scored_origin_panel),
        scored_origin_set_id=f"{candidate_id}_panel",
        comparison_key={
            "forecast_object_type": forecast_object_type,
            "horizon_set": [1],
            "score_law_id": score_policy.body.get(
                "primary_score", score_policy.body.get("point_loss_id")
            ),
            "scored_origin_set_id": f"{candidate_id}_panel",
        },
        missing_scored_origins=(),
        timeguard_checks=tuple(
            {
                "scored_origin_id": scored_origin["scored_origin_id"],
                "expected_available_at": scored_origin["available_at"],
                "observed_available_at": scored_origin["available_at"],
                "status": "passed",
            }
            for scored_origin in scored_origin_panel
        ),
    ).to_manifest(catalog)
