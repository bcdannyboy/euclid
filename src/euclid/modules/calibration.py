from __future__ import annotations

import math
from statistics import NormalDist, fmean
from typing import Any, Mapping

from euclid.contracts.loader import ContractCatalog
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    CalibrationContractManifest,
    CalibrationResultManifest,
)

_STANDARD_NORMAL = NormalDist()

_CALIBRATION_POLICIES = {
    "point": {
        "calibration_mode": "not_applicable",
        "required_diagnostic_ids": (),
        "optional_diagnostic_ids": (),
        "pass_rule": "typed_not_applicable_result",
        "gate_effect": "none",
        "thresholds": {},
    },
    "distribution": {
        "calibration_mode": "required",
        "required_diagnostic_ids": ("pit_or_randomized_pit_uniformity",),
        "optional_diagnostic_ids": ("calibration_sharpness_decomposition",),
        "pass_rule": "declared_distributional_calibration_suite_passes",
        "gate_effect": "required_for_probabilistic_publication",
        "thresholds": {"max_ks_distance": 0.25},
    },
    "interval": {
        "calibration_mode": "required",
        "required_diagnostic_ids": ("nominal_coverage",),
        "optional_diagnostic_ids": ("conditional_coverage",),
        "pass_rule": "declared_interval_calibration_suite_passes",
        "gate_effect": "required_for_probabilistic_publication",
        "thresholds": {"max_abs_coverage_gap": 0.1},
    },
    "quantile": {
        "calibration_mode": "required",
        "required_diagnostic_ids": ("quantile_hit_balance",),
        "optional_diagnostic_ids": ("multi_quantile_crossing_check",),
        "pass_rule": "declared_quantile_calibration_suite_passes",
        "gate_effect": "required_for_probabilistic_publication",
        "thresholds": {"max_abs_hit_balance_gap": 0.15},
    },
    "event_probability": {
        "calibration_mode": "required",
        "required_diagnostic_ids": ("reliability_curve_or_binned_frequency",),
        "optional_diagnostic_ids": ("brier_decomposition",),
        "pass_rule": "declared_event_probability_calibration_suite_passes",
        "gate_effect": "required_for_probabilistic_publication",
        "thresholds": {"max_reliability_gap": 0.2},
    },
}


def build_calibration_contract(
    *,
    catalog: ContractCatalog,
    forecast_object_type: str,
    thresholds: Mapping[str, float] | None = None,
) -> ManifestEnvelope:
    policy = _CALIBRATION_POLICIES[forecast_object_type]
    resolved_thresholds = dict(policy["thresholds"])
    if thresholds:
        resolved_thresholds.update(
            {key: float(value) for key, value in thresholds.items()}
        )
    return CalibrationContractManifest(
        calibration_contract_id=f"{forecast_object_type}_calibration_contract_v1",
        forecast_object_type=forecast_object_type,
        calibration_mode=str(policy["calibration_mode"]),
        required_diagnostic_ids=tuple(policy["required_diagnostic_ids"]),
        optional_diagnostic_ids=tuple(policy["optional_diagnostic_ids"]),
        pass_rule=str(policy["pass_rule"]),
        gate_effect=str(policy["gate_effect"]),
        thresholds=resolved_thresholds,
    ).to_manifest(catalog)


def evaluate_prediction_calibration(
    *,
    catalog: ContractCatalog,
    calibration_contract_manifest: ManifestEnvelope,
    prediction_artifact_manifest: ManifestEnvelope,
) -> ManifestEnvelope:
    forecast_object_type = str(
        prediction_artifact_manifest.body["forecast_object_type"]
    )
    contract_forecast_object_type = str(
        calibration_contract_manifest.body["forecast_object_type"]
    )
    gate_effect = str(calibration_contract_manifest.body["gate_effect"])
    result_id = (
        f"{prediction_artifact_manifest.body['prediction_artifact_id']}" "__calibration"
    )

    if forecast_object_type != contract_forecast_object_type:
        return CalibrationResultManifest(
            calibration_result_id=result_id,
            calibration_contract_ref=calibration_contract_manifest.ref,
            prediction_artifact_ref=prediction_artifact_manifest.ref,
            forecast_object_type=forecast_object_type,
            status="failed",
            failure_reason_code="unsupported_forecast_object_type",
            pass_value=False,
            gate_effect=gate_effect,
            diagnostics=(),
        ).to_manifest(catalog)

    if forecast_object_type == "point":
        return CalibrationResultManifest(
            calibration_result_id=result_id,
            calibration_contract_ref=calibration_contract_manifest.ref,
            prediction_artifact_ref=prediction_artifact_manifest.ref,
            forecast_object_type="point",
            status="not_applicable_for_forecast_type",
            failure_reason_code=None,
            pass_value=None,
            gate_effect="none",
            diagnostics=(),
        ).to_manifest(catalog)

    diagnostics = _diagnostics_for_forecast_object_type(
        forecast_object_type=forecast_object_type,
        rows=tuple(prediction_artifact_manifest.body["rows"]),
        thresholds=dict(calibration_contract_manifest.body.get("thresholds", {})),
    )
    passed = all(item["status"] == "passed" for item in diagnostics)
    return CalibrationResultManifest(
        calibration_result_id=result_id,
        calibration_contract_ref=calibration_contract_manifest.ref,
        prediction_artifact_ref=prediction_artifact_manifest.ref,
        forecast_object_type=forecast_object_type,
        status="passed" if passed else "failed",
        failure_reason_code=None if passed else "calibration_failed",
        pass_value=passed,
        gate_effect=gate_effect,
        diagnostics=tuple(diagnostics),
    ).to_manifest(catalog)


def _diagnostics_for_forecast_object_type(
    *,
    forecast_object_type: str,
    rows: tuple[Mapping[str, Any], ...],
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    if forecast_object_type == "distribution":
        return [_distribution_diagnostic(rows, thresholds)]
    if forecast_object_type == "interval":
        return [_interval_diagnostic(rows, thresholds)]
    if forecast_object_type == "quantile":
        return [_quantile_diagnostic(rows, thresholds)]
    if forecast_object_type == "event_probability":
        return [_event_probability_diagnostic(rows, thresholds)]
    raise ValueError(f"unsupported forecast_object_type {forecast_object_type!r}")


def _distribution_diagnostic(
    rows: tuple[Mapping[str, Any], ...],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    pit_values = []
    for row in rows:
        location = float(row["location"])
        scale = float(row["scale"])
        realized_observation = float(row["realized_observation"])
        z = (realized_observation - location) / scale
        pit_values.append(_STANDARD_NORMAL.cdf(z))
    ks_distance = _ks_uniform_distance(tuple(pit_values))
    threshold = float(thresholds.get("max_ks_distance", 0.25))
    return {
        "diagnostic_id": "pit_or_randomized_pit_uniformity",
        "sample_size": len(pit_values),
        "mean_pit": _stable_float(fmean(pit_values)),
        "max_ks_distance": _stable_float(ks_distance),
        "status": "passed" if ks_distance <= threshold else "failed",
    }


def _interval_diagnostic(
    rows: tuple[Mapping[str, Any], ...],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    nominal_coverage = fmean(float(row["nominal_coverage"]) for row in rows)
    empirical_coverage = fmean(
        1.0
        if float(row["lower_bound"])
        <= float(row["realized_observation"])
        <= float(row["upper_bound"])
        else 0.0
        for row in rows
    )
    absolute_gap = abs(empirical_coverage - nominal_coverage)
    threshold = float(thresholds.get("max_abs_coverage_gap", 0.1))
    return {
        "diagnostic_id": "nominal_coverage",
        "nominal_coverage": _stable_float(nominal_coverage),
        "empirical_coverage": _stable_float(empirical_coverage),
        "absolute_gap": _stable_float(absolute_gap),
        "status": "passed" if absolute_gap <= threshold else "failed",
    }


def _quantile_diagnostic(
    rows: tuple[Mapping[str, Any], ...],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    levels = sorted(
        {float(item["level"]) for row in rows for item in row.get("quantiles", [])}
    )
    gaps = []
    for level in levels:
        hits = []
        for row in rows:
            quantile_value = next(
                float(item["value"])
                for item in row["quantiles"]
                if float(item["level"]) == level
            )
            hits.append(
                1.0 if float(row["realized_observation"]) <= quantile_value else 0.0
            )
        gaps.append(abs(fmean(hits) - level))
    max_abs_gap = max(gaps) if gaps else 0.0
    threshold = float(thresholds.get("max_abs_hit_balance_gap", 0.15))
    return {
        "diagnostic_id": "quantile_hit_balance",
        "quantile_levels": [_stable_float(level) for level in levels],
        "max_abs_gap": _stable_float(max_abs_gap),
        "status": "passed" if max_abs_gap <= threshold else "failed",
    }


def _event_probability_diagnostic(
    rows: tuple[Mapping[str, Any], ...],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        probability = _stable_float(float(row["event_probability"]))
        grouped.setdefault(probability, []).append(
            1.0 if bool(row["realized_event"]) else 0.0
        )
    max_reliability_gap = max(
        abs(fmean(outcomes) - probability) for probability, outcomes in grouped.items()
    )
    threshold = float(thresholds.get("max_reliability_gap", 0.2))
    return {
        "diagnostic_id": "reliability_curve_or_binned_frequency",
        "bin_count": len(grouped),
        "max_reliability_gap": _stable_float(max_reliability_gap),
        "status": "passed" if max_reliability_gap <= threshold else "failed",
    }


def _ks_uniform_distance(values: tuple[float, ...]) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    sample_size = len(ordered)
    return max(
        max(
            index / sample_size - value,
            value - ((index - 1) / sample_size),
        )
        for index, value in enumerate(ordered, start=1)
    )


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


__all__ = [
    "build_calibration_contract",
    "evaluate_prediction_calibration",
]
