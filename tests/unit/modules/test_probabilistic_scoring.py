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
from euclid.modules.claims import resolve_claim_publication
from euclid.modules.gate_lifecycle import resolve_scorecard_status
from euclid.modules.scoring import score_probabilistic_prediction_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("forecast_object_type", "primary_score", "rows", "thresholds"),
    (
        (
            "distribution",
            "continuous_ranked_probability_score",
            (
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
            {"max_ks_distance": 1.0},
        ),
        (
            "interval",
            "interval_score",
            (
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
            {"max_abs_coverage_gap": 1.0},
        ),
        (
            "quantile",
            "pinball_loss",
            (
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
            {"max_abs_hit_balance_gap": 1.0},
        ),
        (
            "event_probability",
            "brier_score",
            (
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
            {"max_reliability_gap": 0.25},
        ),
    ),
)
def test_probabilistic_scoring_and_calibration_support_all_admitted_object_types(
    forecast_object_type: str,
    primary_score: str,
    rows,
    thresholds: dict[str, float],
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _probabilistic_score_policy_manifest(
        catalog=catalog,
        forecast_object_type=forecast_object_type,
        primary_score=primary_score,
    )
    prediction_artifact = _probabilistic_prediction_artifact(
        catalog=catalog,
        candidate_id=f"{forecast_object_type}_candidate",
        score_policy=score_policy,
        forecast_object_type=forecast_object_type,
        rows=rows,
    )
    calibration_contract = build_calibration_contract(
        catalog=catalog,
        forecast_object_type=forecast_object_type,
        thresholds=thresholds,
    )

    score_result = score_probabilistic_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy,
        prediction_artifact_manifest=prediction_artifact,
    )
    calibration_result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=calibration_contract,
        prediction_artifact_manifest=prediction_artifact,
    )
    scorecard_decision = resolve_scorecard_status(
        candidate_admissible=True,
        robustness_status="passed",
        candidate_beats_baseline=True,
        confirmatory_promotion_allowed=True,
        point_score_comparison_status=str(score_result.body["comparison_status"]),
        time_safety_status="passed",
        calibration_status=str(calibration_result.body["status"]),
    )
    claim_decision = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": scorecard_decision.descriptive_status,
            "descriptive_reason_codes": list(
                scorecard_decision.descriptive_reason_codes
            ),
            "predictive_status": scorecard_decision.predictive_status,
            "predictive_reason_codes": list(scorecard_decision.predictive_reason_codes),
            "forecast_object_type": forecast_object_type,
        }
    )

    assert score_result.schema_name == "probabilistic_score_result_manifest@1.0.0"
    assert score_result.body["forecast_object_type"] == forecast_object_type
    assert score_result.body["comparison_status"] == "comparable"
    assert calibration_result.body["status"] == "passed"
    assert claim_decision.publication_mode == "candidate_publication"
    assert claim_decision.claim_type == "predictive_within_declared_scope"
    assert claim_decision.predictive_support_status == "confirmatory_supported"
    assert claim_decision.allowed_interpretation_codes == (
        "historical_structure_summary",
        "probabilistic_forecast_within_declared_validation_scope",
    )


def test_probabilistic_calibration_failure_downgrades_claim_to_descriptive_structure() -> (
    None
):
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _probabilistic_score_policy_manifest(
        catalog=catalog,
        forecast_object_type="interval",
        primary_score="interval_score",
    )
    prediction_artifact = _probabilistic_prediction_artifact(
        catalog=catalog,
        candidate_id="interval_candidate",
        score_policy=score_policy,
        forecast_object_type="interval",
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
                lower_bound=10.0,
                upper_bound=12.0,
                realized_observation=13.0,
            ),
        ),
    )
    calibration_contract = build_calibration_contract(
        catalog=catalog,
        forecast_object_type="interval",
        thresholds={"max_abs_coverage_gap": 0.1},
    )

    score_result = score_probabilistic_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy,
        prediction_artifact_manifest=prediction_artifact,
    )
    calibration_result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=calibration_contract,
        prediction_artifact_manifest=prediction_artifact,
    )
    scorecard_decision = resolve_scorecard_status(
        candidate_admissible=True,
        robustness_status="passed",
        candidate_beats_baseline=True,
        confirmatory_promotion_allowed=True,
        point_score_comparison_status=str(score_result.body["comparison_status"]),
        time_safety_status="passed",
        calibration_status=str(calibration_result.body["status"]),
    )
    claim_decision = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": scorecard_decision.descriptive_status,
            "descriptive_reason_codes": list(
                scorecard_decision.descriptive_reason_codes
            ),
            "predictive_status": scorecard_decision.predictive_status,
            "predictive_reason_codes": list(scorecard_decision.predictive_reason_codes),
            "forecast_object_type": "interval",
        }
    )

    assert calibration_result.body["status"] == "failed"
    assert calibration_result.body["failure_reason_code"] == "calibration_failed"
    assert scorecard_decision.predictive_status == "blocked"
    assert scorecard_decision.predictive_reason_codes == ("calibration_failed",)
    assert claim_decision.publication_mode == "candidate_publication"
    assert claim_decision.claim_type == "descriptive_structure"
    assert claim_decision.predictive_support_status == "blocked"
    assert claim_decision.allowed_interpretation_codes == (
        "historical_structure_summary",
    )


def test_distribution_scoring_accepts_non_gaussian_stochastic_model() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _probabilistic_score_policy_manifest(
        catalog=catalog,
        forecast_object_type="distribution",
        primary_score="log_score",
    )
    prediction_artifact = _probabilistic_prediction_artifact(
        catalog=catalog,
        candidate_id="student_t_candidate",
        score_policy=score_policy,
        forecast_object_type="distribution",
        rows=(
            DistributionPredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                distribution_family="student_t_location_scale",
                location=10.0,
                scale=1.0,
                support_kind="all_real",
                realized_observation=10.5,
            ),
        ),
    )

    score_result = score_probabilistic_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy,
        prediction_artifact_manifest=prediction_artifact,
    )

    assert score_result.body["comparison_status"] == "comparable"
    assert score_result.body["aggregated_primary_score"] > 0


def _probabilistic_score_policy_manifest(
    *,
    catalog,
    forecast_object_type: str,
    primary_score: str,
) -> ManifestEnvelope:
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
            "primary_score": primary_score,
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [{"horizon": 1, "weight": "1.0"}],
            "entity_aggregation_mode": "single_entity_only_no_cross_entity_aggregation",
            "secondary_diagnostic_ids": [],
            "forbidden_primary_metric_ids": [],
            "lower_is_better": True,
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )


def _probabilistic_prediction_artifact(
    *,
    catalog,
    candidate_id: str,
    score_policy: ManifestEnvelope,
    forecast_object_type: str,
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
        score_law_id=str(score_policy.body["primary_score"]),
        horizon_weights=({"horizon": 1, "weight": "1.0"},),
        scored_origin_panel=tuple(scored_origin_panel),
        scored_origin_set_id=f"{candidate_id}_panel",
        comparison_key={
            "forecast_object_type": forecast_object_type,
            "horizon_set": [1],
            "score_law_id": score_policy.body["primary_score"],
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
