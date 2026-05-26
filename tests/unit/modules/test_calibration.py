from __future__ import annotations

from dataclasses import replace
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
    QuantilePredictionRow,
    QuantileValue,
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
    diagnostic = result.body["diagnostics"][0]
    assert diagnostic["diagnostic_id"] == "nominal_coverage"
    assert diagnostic["nominal_coverage"] == 0.8
    assert diagnostic["empirical_coverage"] == 0.0
    assert diagnostic["absolute_gap"] == 0.8
    assert diagnostic["status"] == "failed"
    assert diagnostic["level_diagnostics"] == [
        {
            "nominal_coverage": 0.8,
            "sample_size": 2,
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


def test_distribution_calibration_pit_uses_declared_student_t_family() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(catalog, "distribution")
    artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="student_t_distribution_candidate",
        forecast_object_type="distribution",
        score_policy=score_policy,
        rows=(
            DistributionPredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                distribution_family="student_t_location_scale",
                location=0.0,
                scale=1.0,
                support_kind="all_real",
                realized_observation=2.0,
                distribution_parameters={"location": 0.0, "scale": 1.0, "df": 3.0},
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

    assert result.body["diagnostics"][0]["family_id"] == "student_t"
    assert result.body["diagnostics"][0]["mean_pit"] == pytest.approx(
        stats.t(df=3.0, loc=0.0, scale=1.0).cdf(2.0)
    )
    assert result.body["diagnostics"][0]["mean_pit"] != pytest.approx(
        stats.norm(loc=0.0, scale=1.0).cdf(2.0)
    )


@pytest.mark.parametrize(
    ("strategy", "expected_min_bins"),
    (
        ("equal_width", 3),
        ("equal_mass", 3),
        ("adaptive_min_count", 2),
    ),
)
def test_event_probability_reliability_bin_strategies(
    strategy: str,
    expected_min_bins: int,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(catalog, "event_probability")
    probabilities = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)
    artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id=f"{strategy}_event_candidate",
        forecast_object_type="event_probability",
        score_policy=score_policy,
        rows=tuple(
            EventProbabilityPredictionRow(
                origin_time=f"2026-01-0{index + 1}T00:00:00Z",
                available_at=f"2026-01-0{index + 2}T00:00:00Z",
                horizon=1,
                event_definition={"kind": "greater_than", "threshold": 10.0},
                event_probability=probability,
                realized_observation=11.0 if probability >= 0.5 else 9.0,
                realized_event=probability >= 0.5,
            )
            for index, probability in enumerate(probabilities)
        ),
    )
    contract = build_calibration_contract(
        catalog=catalog,
        forecast_object_type="event_probability",
        thresholds={"max_reliability_gap": 1.0},
        reliability_bins={
            "strategy": strategy,
            "bin_count": 3,
            "minimum_bin_count": 2,
        },
    )

    result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=contract,
        prediction_artifact_manifest=artifact,
    )

    diagnostic = result.body["diagnostics"][0]
    assert diagnostic["bin_strategy"] == strategy
    assert diagnostic["bin_count"] >= expected_min_bins
    assert all("sample_count" in item for item in diagnostic["bins"])


def test_interval_and_quantile_calibration_reports_per_level_diagnostics() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    interval_policy = _score_policy_manifest(catalog, "interval")
    quantile_policy = _score_policy_manifest(catalog, "quantile")
    interval_artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="interval_levels_candidate",
        forecast_object_type="interval",
        score_policy=interval_policy,
        rows=(
            IntervalPredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                nominal_coverage=0.8,
                lower_bound=9.0,
                upper_bound=11.0,
                realized_observation=10.0,
                intervals=(
                    {"nominal_coverage": 0.8, "lower_bound": 9.0, "upper_bound": 11.0},
                    {"nominal_coverage": 0.9, "lower_bound": 8.0, "upper_bound": 12.0},
                ),
            ),
        ),
    )
    quantile_artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="quantile_levels_candidate",
        forecast_object_type="quantile",
        score_policy=quantile_policy,
        rows=(
            QuantilePredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                quantiles=(
                    QuantileValue(level=0.1, value=9.0),
                    QuantileValue(level=0.5, value=10.0),
                    QuantileValue(level=0.9, value=11.0),
                ),
                realized_observation=10.0,
            ),
        ),
    )

    interval_result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=build_calibration_contract(
            catalog=catalog,
            forecast_object_type="interval",
            thresholds={"max_abs_coverage_gap": 1.0},
            interval_levels=(0.8, 0.9),
        ),
        prediction_artifact_manifest=interval_artifact,
    )
    quantile_result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=build_calibration_contract(
            catalog=catalog,
            forecast_object_type="quantile",
            thresholds={"max_abs_hit_balance_gap": 1.0},
            quantile_levels=(0.1, 0.5, 0.9),
        ),
        prediction_artifact_manifest=quantile_artifact,
    )

    assert [
        item["nominal_coverage"]
        for item in interval_result.body["diagnostics"][0]["level_diagnostics"]
    ] == [0.8, 0.9]
    assert [
        item["quantile_level"]
        for item in quantile_result.body["diagnostics"][0]["level_diagnostics"]
    ] == [0.1, 0.5, 0.9]


def test_calibration_fails_on_insufficient_sample_count() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(catalog, "event_probability")
    artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="small_event_candidate",
        forecast_object_type="event_probability",
        score_policy=score_policy,
        rows=(
            EventProbabilityPredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
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
        thresholds={"minimum_sample_count": 2},
    )

    result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=contract,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["status"] == "failed"
    assert result.body["failure_reason_code"] == "insufficient_calibration_sample_count"
    assert result.body["lane_status"] == "failed"


def test_recalibration_lanes_cannot_fit_on_confirmatory_rows() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(catalog, "distribution")
    artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="recalibration_candidate",
        forecast_object_type="distribution",
        score_policy=score_policy,
        rows=(
            DistributionPredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                distribution_family="gaussian_location_scale",
                location=0.0,
                scale=1.0,
                support_kind="all_real",
                realized_observation=0.0,
            ),
        ),
    )
    contract = build_calibration_contract(
        catalog=catalog,
        forecast_object_type="distribution",
        calibration_lane="recalibration_fit",
    )

    result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=contract,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["status"] == "failed"
    assert result.body["failure_reason_code"] == (
        "confirmatory_rows_forbidden_for_recalibration"
    )
    assert result.body["lane_status"] == "blocked"


def test_recalibration_fit_requires_independent_calibration_split_roles() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(catalog, "distribution")
    artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="mixed_split_candidate",
        forecast_object_type="distribution",
        score_policy=score_policy,
        rows=(
            DistributionPredictionRow(
                origin_time="2026-01-01T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                horizon=1,
                distribution_family="gaussian_location_scale",
                location=0.0,
                scale=1.0,
                support_kind="all_real",
                realized_observation=0.0,
            ),
        ),
    )
    body = dict(artifact.body)
    body["stage_id"] = "calibration_fit"
    body["calibration_split_id"] = "calibration_split_alpha"
    row = dict(body["rows"][0])
    row["split_role"] = "confirmatory_holdout"
    body["rows"] = (row,)
    artifact = replace(artifact, body=body)
    contract = build_calibration_contract(
        catalog=catalog,
        forecast_object_type="distribution",
        calibration_lane="recalibration_fit",
    )

    result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=contract,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["status"] == "failed"
    assert result.body["failure_reason_code"] == (
        "confirmatory_rows_forbidden_for_recalibration"
    )
    diagnostic = result.body["diagnostics"][0]
    assert diagnostic["diagnostic_id"] == "calibration_split_independence"
    assert diagnostic["split_role_counts"] == {"confirmatory_holdout": 1}
    assert result.body["calibration_identity"]["calibration_split_id"] == (
        "calibration_split_alpha"
    )
    assert result.body["calibration_identity"]["split_role_counts"] == {
        "confirmatory_holdout": 1
    }


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
    diagnostic = result.body["diagnostics"][0]
    assert diagnostic["diagnostic_id"] == "reliability_curve_or_binned_frequency"
    assert diagnostic["bin_strategy"] == "exact"
    assert diagnostic["bin_count"] == 2
    assert diagnostic["max_reliability_gap"] == 0.2
    assert diagnostic["status"] == "passed"
    assert [item["sample_count"] for item in diagnostic["bins"]] == [1, 1]


def test_split_conformal_exchangeable_requires_exchangeability_declaration() -> None:
    manifest = _conformal_method_claim_scope(
        method_id="split_conformal_exchangeable_v1",
        assumption_declarations={},
    )

    assert manifest["method_id"] == "split_conformal_exchangeable_v1"
    assert manifest["calibration_split_id"] == "calibration_split_alpha"
    assert manifest["horizon_ids"] == [1, 3]
    assert manifest["guarantee_tier"] == "finite_sample_exchangeable"
    assert manifest["status"] == "blocked"
    assert manifest["finite_sample_distribution_free_claim_allowed"] is False
    assert manifest["fixed_time_finite_sample_claim_allowed"] is False
    assert manifest["reason_codes"] == ["missing_exchangeability_declaration"]


def test_enbpi_time_series_is_approximate_mixing_not_exact_finite_sample() -> None:
    manifest = _conformal_method_claim_scope(
        method_id="enbpi_time_series_v1",
        assumption_declarations={"mixing": "beta_mixing_declared"},
    )

    assert manifest["method_id"] == "enbpi_time_series_v1"
    assert manifest["guarantee_tier"] == "approximate_mixing_time_series"
    assert manifest["assumption_scope"] == "mixing_time_series"
    assert manifest["finite_sample_distribution_free_claim_allowed"] is False
    assert manifest["fixed_time_finite_sample_claim_allowed"] is False
    assert manifest["approximate_coverage_claim_allowed"] is True
    assert "finite_sample_distribution_free_not_supported" in manifest["reason_codes"]


def test_adaptive_conformal_time_series_is_long_run_not_fixed_time_finite_sample() -> None:
    manifest = _conformal_method_claim_scope(
        method_id="adaptive_conformal_time_series_v1",
        assumption_declarations={"adaptation": "online_residual_update_declared"},
    )

    assert manifest["method_id"] == "adaptive_conformal_time_series_v1"
    assert manifest["guarantee_tier"] == "long_run_frequency_control"
    assert manifest["assumption_scope"] == "time_series_adaptive"
    assert manifest["long_run_coverage_claim_allowed"] is True
    assert manifest["fixed_time_finite_sample_claim_allowed"] is False
    assert manifest["finite_sample_distribution_free_claim_allowed"] is False
    assert "fixed_time_finite_sample_not_supported" in manifest["reason_codes"]


def test_unknown_conformal_method_fails_closed() -> None:
    manifest = _conformal_method_claim_scope(
        method_id="made_up_conformal_method_v1",
        assumption_declarations={"exchangeability": True},
    )

    assert manifest["method_id"] == "made_up_conformal_method_v1"
    assert manifest["status"] == "failed"
    assert manifest["guarantee_tier"] == "diagnostic_only"
    assert manifest["finite_sample_distribution_free_claim_allowed"] is False
    assert manifest["fixed_time_finite_sample_claim_allowed"] is False
    assert manifest["reason_codes"] == ["unknown_conformal_method"]


def test_horizon_one_residuals_cannot_calibrate_horizon_three_without_pooling() -> None:
    manifest = _partitioned_calibration_manifest(
        partition_payloads=(
            {
                "partition_id": "h1_residuals_for_h3",
                "calibration_split_id": "calibration_split_alpha",
                "horizon_id": 1,
                "target_horizon_id": 3,
                "residuals": (0.1, -0.2, 0.05, 0.0),
                "coverage_hits": (True, True, True, True),
            },
        ),
        target_horizon_ids=(3,),
        partition_policy={"horizon_pooling": "none"},
        minimum_calibration_count=2,
    )

    assert manifest["status"] == "blocked"
    assert manifest["promotion_allowed"] is False
    assert manifest["reason_codes"] == [
        "cross_horizon_calibration_without_valid_pooling_policy"
    ]
    diagnostic = manifest["partition_diagnostics"][0]
    assert diagnostic["partition_id"] == "h1_residuals_for_h3"
    assert diagnostic["horizon_id"] == 1
    assert diagnostic["target_horizon_id"] == 3
    assert diagnostic["pooling_policy"] == "none"


def test_minimum_calibration_count_is_enforced_per_partition() -> None:
    manifest = _partitioned_calibration_manifest(
        partition_payloads=(
            {
                "partition_id": "h1_calm",
                "calibration_split_id": "calibration_split_alpha",
                "horizon_id": 1,
                "target_horizon_id": 1,
                "regime_id": "calm",
                "residuals": (0.1,),
                "coverage_hits": (True,),
            },
            {
                "partition_id": "h3_calm",
                "calibration_split_id": "calibration_split_alpha",
                "horizon_id": 3,
                "target_horizon_id": 3,
                "regime_id": "calm",
                "residuals": (0.2, -0.1, 0.0),
                "coverage_hits": (True, True, True),
            },
        ),
        target_horizon_ids=(1, 3),
        partition_policy={"horizon_pooling": "none"},
        minimum_calibration_count=2,
    )

    assert manifest["status"] == "blocked"
    assert manifest["promotion_allowed"] is False
    assert "insufficient_calibration_count_for_partition" in manifest["reason_codes"]
    failed_partitions = [
        item
        for item in manifest["partition_diagnostics"]
        if item["status"] == "blocked"
    ]
    assert failed_partitions == [
        {
            "partition_id": "h1_calm",
            "horizon_id": 1,
            "target_horizon_id": 1,
            "entity_id": None,
            "regime_id": "calm",
            "calibration_count": 1,
            "minimum_calibration_count": 2,
            "status": "blocked",
            "reason_codes": ["insufficient_calibration_count_for_partition"],
        }
    ]


def test_regime_slice_undercoverage_blocks_promotion() -> None:
    manifest = _partitioned_calibration_manifest(
        partition_payloads=(
            {
                "partition_id": "h1_calm",
                "calibration_split_id": "calibration_split_alpha",
                "horizon_id": 1,
                "target_horizon_id": 1,
                "regime_id": "calm",
                "residuals": (0.1, 0.0, -0.1, 0.2, -0.2, 0.05),
                "coverage_hits": (True, True, True, True, True, True),
            },
            {
                "partition_id": "h1_stress",
                "calibration_split_id": "calibration_split_alpha",
                "horizon_id": 1,
                "target_horizon_id": 1,
                "regime_id": "stress",
                "residuals": (1.4, 1.2, -1.1, 1.6, -1.3, 1.5),
                "coverage_hits": (False, False, True, False, False, True),
            },
        ),
        target_horizon_ids=(1,),
        partition_policy={"horizon_pooling": "none", "regime_slices": "separate"},
        minimum_calibration_count=2,
        nominal_coverage=0.8,
        minimum_empirical_coverage_lower_bound=0.65,
    )

    assert manifest["status"] == "failed"
    assert manifest["promotion_allowed"] is False
    assert "regime_slice_undercoverage" in manifest["reason_codes"]
    stress = next(
        item
        for item in manifest["partition_diagnostics"]
        if item["regime_id"] == "stress"
    )
    assert stress["status"] == "failed"
    assert stress["empirical_coverage"] == pytest.approx(2 / 6)
    assert stress["coverage_lower_bound"] < 0.65
    assert stress["reason_codes"] == ["regime_slice_undercoverage"]


def test_mapie_unavailable_returns_calibration_backend_unavailable() -> None:
    manifest = _mapie_time_series_manifest(optional_backend_overrides={"mapie": None})

    assert manifest["status"] == "unavailable"
    assert manifest["promotion_allowed"] is False
    assert manifest["backend"] == "mapie"
    assert manifest["failure_reason_code"] == "calibration_backend_unavailable"
    assert manifest["reason_codes"] == ["calibration_backend_unavailable"]


def test_mapie_time_series_method_records_version_indices_and_assumptions() -> None:
    manifest = _mapie_time_series_manifest(
        method_name="EnbPI",
        calibration_indices=(2, 3, 5, 8),
        assumptions={
            "dependence": "weak_mixing",
            "calibration_window": "pre_confirmatory_only",
        },
        optional_backend_overrides={
            "mapie": {
                "available": True,
                "version": "0.9.2",
            }
        },
    )

    assert manifest["status"] == "passed"
    assert manifest["backend"] == "mapie"
    assert manifest["backend_version"] == "0.9.2"
    assert manifest["method_name"] == "EnbPI"
    assert manifest["calibration_indices"] == [2, 3, 5, 8]
    assert manifest["assumptions"] == {
        "calibration_window": "pre_confirmatory_only",
        "dependence": "weak_mixing",
    }
    assert "serialized_mapie_object" not in manifest


def _conformal_method_claim_scope(
    *,
    method_id: str,
    assumption_declarations: dict[str, object],
) -> dict[str, object]:
    from euclid.modules.conformal import resolve_conformal_method

    result = resolve_conformal_method(
        method_id=method_id,
        calibration_split_id="calibration_split_alpha",
        horizon_ids=(1, 3),
        assumption_declarations=assumption_declarations,
    )
    return result.as_manifest()


def _partitioned_calibration_manifest(
    *,
    partition_payloads: tuple[dict[str, object], ...],
    target_horizon_ids: tuple[int, ...],
    partition_policy: dict[str, object],
    minimum_calibration_count: int,
    nominal_coverage: float = 0.8,
    minimum_empirical_coverage_lower_bound: float = 0.0,
) -> dict[str, object]:
    from euclid.modules.conformal import (
        CalibrationPartition,
        evaluate_calibration_partitions,
    )

    result = evaluate_calibration_partitions(
        calibration_partitions=tuple(
            CalibrationPartition(**payload) for payload in partition_payloads
        ),
        target_horizon_ids=target_horizon_ids,
        partition_policy=partition_policy,
        minimum_calibration_count=minimum_calibration_count,
        nominal_coverage=nominal_coverage,
        minimum_empirical_coverage_lower_bound=(
            minimum_empirical_coverage_lower_bound
        ),
    )
    return result.as_manifest()


def _mapie_time_series_manifest(
    *,
    method_name: str = "EnbPI",
    calibration_indices: tuple[int, ...] = (0, 1, 2),
    assumptions: dict[str, object] | None = None,
    optional_backend_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    from euclid.modules.conformal import run_mapie_time_series_adapter

    result = run_mapie_time_series_adapter(
        method_name=method_name,
        calibration_indices=calibration_indices,
        assumptions=assumptions or {"dependence": "weak_mixing"},
        optional_backend_overrides=optional_backend_overrides or {},
    )
    return result.as_manifest()


def _score_policy_manifest(catalog, forecast_object_type: str) -> ManifestEnvelope:
    schema_name = {
        "point": "point_score_policy_manifest@1.0.0",
        "distribution": "probabilistic_score_policy_manifest@1.0.0",
        "interval": "interval_score_policy_manifest@1.0.0",
        "quantile": "quantile_score_policy_manifest@1.0.0",
        "event_probability": "event_probability_score_policy_manifest@1.0.0",
    }[forecast_object_type]
    primary_score = {
        "point": "absolute_error",
        "distribution": "continuous_ranked_probability_score",
        "interval": "interval_score",
        "quantile": "pinball_loss",
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
