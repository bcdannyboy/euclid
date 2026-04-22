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
    PredictionArtifactManifest,
    PredictionRow,
    QuantilePredictionRow,
    QuantileValue,
)
from euclid.modules.scoring import (
    evaluate_point_comparators,
    score_point_prediction_artifact,
    score_probabilistic_prediction_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_score_point_prediction_artifact_computes_weighted_squared_error() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(
        catalog=catalog,
        point_loss_id="squared_error",
        horizon_weights=((1, "0.25"), (2, "0.75")),
    )
    artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="candidate",
        score_policy=score_policy,
        rows=(
            ("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 1, 10.0, 11.0),
            ("2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", 2, 20.0, 23.0),
            ("2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", 1, 12.0, 10.0),
            ("2026-01-02T00:00:00Z", "2026-01-04T00:00:00Z", 2, 18.0, 21.0),
        ),
        horizon_weights=((1, "0.25"), (2, "0.75")),
        scored_origin_set_id="shared-origin-panel",
    )

    result = score_point_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy,
        prediction_artifact_manifest=artifact,
    )

    assert result.schema_name == "point_score_result_manifest@1.0.0"
    assert result.body["comparison_status"] == "comparable"
    assert result.body["failure_reason_code"] is None
    assert result.body["per_horizon"] == [
        {"horizon": 1, "valid_origin_count": 2, "mean_point_loss": 2.5},
        {"horizon": 2, "valid_origin_count": 2, "mean_point_loss": 9.0},
    ]
    assert result.body["aggregated_primary_score"] == pytest.approx(7.375)


def test_score_point_prediction_artifact_marks_incomplete_horizon_panel() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(
        catalog=catalog,
        point_loss_id="absolute_error",
        horizon_weights=((1, "0.5"), (2, "0.5")),
    )
    artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="candidate",
        score_policy=score_policy,
        rows=(
            ("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 1, 10.0, 11.0),
            ("2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", 1, 12.0, 10.0),
            ("2026-01-02T00:00:00Z", "2026-01-04T00:00:00Z", 2, 18.0, 21.0),
        ),
        horizon_weights=((1, "0.5"), (2, "0.5")),
        scored_origin_set_id="shared-origin-panel",
    )

    result = score_point_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["comparison_status"] == "not_comparable"
    assert result.body["failure_reason_code"] == "incomplete_declared_horizon_panel"
    assert result.body["per_horizon"] == []


def test_evaluate_point_comparators_emits_paired_records_and_significance() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(
        catalog=catalog,
        point_loss_id="absolute_error",
        horizon_weights=((1, "1.0"),),
    )
    candidate_artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="candidate",
        score_policy=score_policy,
        rows=(
            ("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 1, 10.0, 11.0),
            ("2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", 1, 12.0, 12.0),
        ),
        horizon_weights=((1, "1.0"),),
        scored_origin_set_id="shared-origin-panel",
    )
    baseline_registry = _baseline_registry_manifest(
        catalog=catalog,
        score_policy=score_policy,
        declarations=(
            ("constant_baseline", "baseline", "constant"),
            ("seasonal_baseline", "baseline", "seasonal_naive"),
        ),
        primary_baseline_id="constant_baseline",
    )

    result = evaluate_point_comparators(
        catalog=catalog,
        score_policy_manifest=score_policy,
        candidate_prediction_artifact=candidate_artifact,
        baseline_registry_manifest=baseline_registry,
        comparator_prediction_artifacts={
            "constant_baseline": _prediction_artifact(
                catalog=catalog,
                candidate_id="constant_baseline",
                score_policy=score_policy,
                rows=(
                    ("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 1, 9.0, 11.0),
                    ("2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", 1, 11.0, 12.0),
                ),
                horizon_weights=((1, "1.0"),),
                scored_origin_set_id="shared-origin-panel",
            ),
            "seasonal_baseline": _prediction_artifact(
                catalog=catalog,
                candidate_id="seasonal_baseline",
                score_policy=score_policy,
                rows=(
                    ("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 1, 11.0, 11.0),
                    ("2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", 1, 12.4, 12.0),
                ),
                horizon_weights=((1, "1.0"),),
                scored_origin_set_id="shared-origin-panel",
            ),
        },
        practical_significance_margin=0.5,
    )

    assert result.candidate_score_result.body[
        "aggregated_primary_score"
    ] == pytest.approx(0.5)
    assert [
        item.body["aggregated_primary_score"]
        for item in result.comparator_score_results
    ] == pytest.approx([1.5, 0.2])
    assert result.comparison_universe.body["candidate_score_result_ref"] == (
        result.candidate_score_result.ref.as_dict()
    )
    assert result.comparison_universe.body["baseline_score_result_ref"] == (
        result.comparator_score_results[0].ref.as_dict()
    )
    paired_records = result.comparison_universe.body["paired_comparison_records"]
    predictive_tests = [
        record.pop("paired_predictive_test_result") for record in paired_records
    ]
    assert paired_records == [
        {
            "comparator_id": "constant_baseline",
            "comparator_kind": "baseline",
            "comparison_status": "comparable",
            "failure_reason_code": None,
            "candidate_primary_score": 0.5,
            "comparator_primary_score": 1.5,
            "primary_score_delta": 1.0,
            "paired_origin_count": 2,
            "mean_loss_differential": 1.0,
            "per_horizon_mean_loss_differentials": [
                {"horizon": 1, "mean_loss_differential": 1.0}
            ],
            "practical_significance_margin": 0.5,
            "practical_significance_status": "candidate_better_than_margin",
            "score_result_ref": result.comparator_score_results[0].ref.as_dict(),
        },
        {
            "comparator_id": "seasonal_baseline",
            "comparator_kind": "baseline",
            "comparison_status": "comparable",
            "failure_reason_code": None,
            "candidate_primary_score": 0.5,
            "comparator_primary_score": 0.2,
            "primary_score_delta": -0.3,
            "paired_origin_count": 2,
            "mean_loss_differential": -0.3,
            "per_horizon_mean_loss_differentials": [
                {"horizon": 1, "mean_loss_differential": -0.3}
            ],
            "practical_significance_margin": 0.5,
            "practical_significance_status": "within_margin",
            "score_result_ref": result.comparator_score_results[1].ref.as_dict(),
        },
    ]
    assert predictive_tests[0]["schema_name"] == "paired_predictive_test_result@1.0.0"
    assert predictive_tests[0]["status"] == "passed"
    assert predictive_tests[0]["promotion_allowed"] is True
    assert predictive_tests[0]["raw_metric_comparison_role"] == "diagnostic_only"
    assert predictive_tests[0]["replay_identity"].startswith("predictive-promotion:")
    assert predictive_tests[1]["status"] == "downgraded"
    assert predictive_tests[1]["promotion_allowed"] is False


def test_evaluate_point_comparators_rejects_mismatched_primary_baseline_geometry() -> (
    None
):
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(
        catalog=catalog,
        point_loss_id="absolute_error",
        horizon_weights=((1, "1.0"),),
    )
    candidate_artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="candidate",
        score_policy=score_policy,
        rows=(
            ("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 1, 10.0, 11.0),
            ("2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", 1, 12.0, 12.0),
        ),
        horizon_weights=((1, "1.0"),),
        scored_origin_set_id="candidate-origin-panel",
    )
    baseline_registry = _baseline_registry_manifest(
        catalog=catalog,
        score_policy=score_policy,
        declarations=(("constant_baseline", "baseline", "constant"),),
        primary_baseline_id="constant_baseline",
    )

    with pytest.raises(ContractValidationError) as exc_info:
        evaluate_point_comparators(
            catalog=catalog,
            score_policy_manifest=score_policy,
            candidate_prediction_artifact=candidate_artifact,
            baseline_registry_manifest=baseline_registry,
            comparator_prediction_artifacts={
                "constant_baseline": _prediction_artifact(
                    catalog=catalog,
                    candidate_id="constant_baseline",
                    score_policy=score_policy,
                    rows=(
                        ("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 1, 9.0, 11.0),
                        ("2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", 1, 11.0, 12.0),
                    ),
                    horizon_weights=((1, "1.0"),),
                    scored_origin_set_id="baseline-origin-panel",
                )
            },
            practical_significance_margin=0.0,
        )

    assert exc_info.value.code == "primary_comparator_not_comparable"


@pytest.mark.parametrize(
    ("forecast_object_type", "primary_score", "rows", "expected_score"),
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
            ),
            0.233694977255,
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
                    lower_bound=9.0,
                    upper_bound=11.0,
                    realized_observation=12.0,
                ),
            ),
            7.0,
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
                        QuantileValue(level=0.1, value=8.0),
                        QuantileValue(level=0.5, value=10.0),
                        QuantileValue(level=0.9, value=12.0),
                    ),
                    realized_observation=11.0,
                ),
            ),
            0.3,
        ),
        (
            "event_probability",
            "brier_score",
            (
                EventProbabilityPredictionRow(
                    origin_time="2026-01-01T00:00:00Z",
                    available_at="2026-01-02T00:00:00Z",
                    horizon=1,
                    event_definition={"kind": "greater_than", "threshold": 10.5},
                    event_probability=0.8,
                    realized_observation=11.0,
                    realized_event=True,
                ),
                EventProbabilityPredictionRow(
                    origin_time="2026-01-02T00:00:00Z",
                    available_at="2026-01-03T00:00:00Z",
                    horizon=1,
                    event_definition={"kind": "greater_than", "threshold": 10.5},
                    event_probability=0.3,
                    realized_observation=10.0,
                    realized_event=False,
                ),
            ),
            0.065,
        ),
    ),
)
def test_score_probabilistic_prediction_artifact_computes_type_specific_primary_scores(
    forecast_object_type: str,
    primary_score: str,
    rows,
    expected_score: float,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _probabilistic_score_policy_manifest(
        catalog=catalog,
        forecast_object_type=forecast_object_type,
        primary_score=primary_score,
        horizon_weights=((1, "1.0"),),
    )
    artifact = _probabilistic_prediction_artifact(
        catalog=catalog,
        candidate_id=f"{forecast_object_type}_candidate",
        score_policy=score_policy,
        forecast_object_type=forecast_object_type,
        rows=rows,
        horizon_weights=((1, "1.0"),),
        scored_origin_set_id=f"{forecast_object_type}_panel",
    )

    result = score_probabilistic_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy,
        prediction_artifact_manifest=artifact,
    )

    assert result.schema_name == "probabilistic_score_result_manifest@1.0.0"
    assert result.body["forecast_object_type"] == forecast_object_type
    assert result.body["comparison_status"] == "comparable"
    assert result.body["failure_reason_code"] is None
    assert result.body["per_horizon"] == [
        {
            "horizon": 1,
            "valid_origin_count": len(rows),
            "mean_primary_score": pytest.approx(expected_score),
        }
    ]
    assert result.body["aggregated_primary_score"] == pytest.approx(expected_score)


def test_score_probabilistic_prediction_artifact_marks_policy_object_mismatch() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    interval_policy = _probabilistic_score_policy_manifest(
        catalog=catalog,
        forecast_object_type="interval",
        primary_score="interval_score",
        horizon_weights=((1, "1.0"),),
    )
    distribution_policy = _probabilistic_score_policy_manifest(
        catalog=catalog,
        forecast_object_type="distribution",
        primary_score="continuous_ranked_probability_score",
        horizon_weights=((1, "1.0"),),
    )
    artifact = _probabilistic_prediction_artifact(
        catalog=catalog,
        candidate_id="distribution_candidate",
        score_policy=distribution_policy,
        forecast_object_type="distribution",
        rows=(
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
        ),
        horizon_weights=((1, "1.0"),),
        scored_origin_set_id="distribution_panel",
    )

    result = score_probabilistic_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=interval_policy,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["comparison_status"] == "not_comparable"
    assert result.body["failure_reason_code"] == "unsupported_forecast_object_type"


def _score_policy_manifest(
    *,
    catalog,
    point_loss_id: str,
    horizon_weights: tuple[tuple[int, str], ...],
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": f"policy_{point_loss_id}",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": "point",
            "point_loss_id": point_loss_id,
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [
                {"horizon": horizon, "weight": weight}
                for horizon, weight in horizon_weights
            ],
            "entity_aggregation_mode": "single_entity_only_no_cross_entity_aggregation",
            "secondary_diagnostic_ids": [],
            "forbidden_primary_metric_ids": [
                "root_mean_squared_error",
                "mean_absolute_percentage_error",
            ],
            "lower_is_better": True,
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )


def _baseline_registry_manifest(
    *,
    catalog,
    score_policy: ManifestEnvelope,
    declarations: tuple[tuple[str, str, str], ...],
    primary_baseline_id: str,
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="baseline_registry_manifest@1.1.0",
        module_id="evaluation_governance",
        body={
            "baseline_registry_id": "test_baseline_registry_v1",
            "owner_prompt_id": "prompt.predictive-validation-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "primary_baseline_id": primary_baseline_id,
            "baseline_ids": [baseline_id for baseline_id, _, _ in declarations],
            "baseline_declarations": [
                {
                    "baseline_id": baseline_id,
                    "comparator_declaration_id": f"{baseline_id}_declaration",
                    "comparator_kind": comparator_kind,
                    "family_id": family_id,
                    "forecast_object_type": "point",
                    "freeze_rule": "frozen_before_confirmatory_access",
                }
                for baseline_id, comparator_kind, family_id in declarations
            ],
            "compatible_point_score_policy_ref": score_policy.ref.as_dict(),
        },
        catalog=catalog,
    )


def _prediction_artifact(
    *,
    catalog,
    candidate_id: str,
    score_policy: ManifestEnvelope,
    rows: tuple[tuple[str, str, int, float, float], ...],
    horizon_weights: tuple[tuple[int, str], ...],
    scored_origin_set_id: str,
) -> ManifestEnvelope:
    scored_origin_panel = [
        {
            "scored_origin_id": f"{candidate_id}_origin_{index}",
            "origin_time": origin_time,
            "available_at": available_at,
            "horizon": horizon,
        }
        for index, (origin_time, available_at, horizon, _, _) in enumerate(rows)
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
        rows=tuple(
            PredictionRow(
                origin_time=origin_time,
                available_at=available_at,
                horizon=horizon,
                point_forecast=point_forecast,
                realized_observation=realized_observation,
            )
            for (
                origin_time,
                available_at,
                horizon,
                point_forecast,
                realized_observation,
            ) in rows
        ),
        score_law_id=str(score_policy.body["point_loss_id"]),
        horizon_weights=tuple(
            {
                "horizon": horizon,
                "weight": weight,
            }
            for horizon, weight in horizon_weights
        ),
        scored_origin_panel=tuple(scored_origin_panel),
        scored_origin_set_id=scored_origin_set_id,
        comparison_key={
            "forecast_object_type": "point",
            "horizon_set": [horizon for horizon, _ in horizon_weights],
            "score_law_id": score_policy.body["point_loss_id"],
            "scored_origin_set_id": scored_origin_set_id,
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


_PROBABILISTIC_SCORE_POLICY_SCHEMAS = {
    "distribution": "probabilistic_score_policy_manifest@1.0.0",
    "interval": "interval_score_policy_manifest@1.0.0",
    "quantile": "quantile_score_policy_manifest@1.0.0",
    "event_probability": "event_probability_score_policy_manifest@1.0.0",
}


def _probabilistic_score_policy_manifest(
    *,
    catalog,
    forecast_object_type: str,
    primary_score: str,
    horizon_weights: tuple[tuple[int, str], ...],
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name=_PROBABILISTIC_SCORE_POLICY_SCHEMAS[forecast_object_type],
        module_id="scoring",
        body={
            "score_policy_id": f"{forecast_object_type}_{primary_score}_policy_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": forecast_object_type,
            "primary_score": primary_score,
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [
                {"horizon": horizon, "weight": weight}
                for horizon, weight in horizon_weights
            ],
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
    horizon_weights: tuple[tuple[int, str], ...],
    scored_origin_set_id: str,
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
        horizon_weights=tuple(
            {
                "horizon": horizon,
                "weight": weight,
            }
            for horizon, weight in horizon_weights
        ),
        scored_origin_panel=tuple(scored_origin_panel),
        scored_origin_set_id=scored_origin_set_id,
        comparison_key={
            "forecast_object_type": forecast_object_type,
            "horizon_set": [horizon for horizon, _ in horizon_weights],
            "score_law_id": score_policy.body["primary_score"],
            "scored_origin_set_id": scored_origin_set_id,
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
