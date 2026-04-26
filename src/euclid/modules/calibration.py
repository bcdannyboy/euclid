from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Mapping, Sequence

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    CalibrationContractManifest,
    CalibrationResultManifest,
)
from euclid.modules.scoring import bind_distribution_row_observation_model

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
    reliability_bins: Mapping[str, Any] | None = None,
    pit_config: Mapping[str, Any] | None = None,
    interval_levels: Sequence[float] = (),
    quantile_levels: Sequence[float] = (),
    calibration_lane: str = "evaluation_only",
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
        reliability_bins=dict(reliability_bins or {}),
        pit_config=dict(pit_config or {}),
        interval_levels=tuple(float(level) for level in interval_levels),
        quantile_levels=tuple(float(level) for level in quantile_levels),
        calibration_lane=calibration_lane,
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
    effective_config = _effective_calibration_config(calibration_contract_manifest)
    calibration_identity = _calibration_identity(
        calibration_contract_manifest=calibration_contract_manifest,
        prediction_artifact_manifest=prediction_artifact_manifest,
    )
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
            effective_calibration_config=effective_config,
            calibration_identity=calibration_identity,
            lane_status="failed",
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
            effective_calibration_config=effective_config,
            calibration_identity=calibration_identity,
            lane_status="not_applicable",
        ).to_manifest(catalog)

    lane = str(effective_config["calibration_lane"])
    stage_id = str(prediction_artifact_manifest.body.get("stage_id", ""))
    is_fit_lane = (
        lane in {"recalibration_fit", "conformal_fit"}
        or (("recalibration" in lane or "conformal" in lane) and lane.endswith("_fit"))
    )
    if is_fit_lane and stage_id.startswith("confirmatory"):
        return CalibrationResultManifest(
            calibration_result_id=result_id,
            calibration_contract_ref=calibration_contract_manifest.ref,
            prediction_artifact_ref=prediction_artifact_manifest.ref,
            forecast_object_type=forecast_object_type,
            status="failed",
            failure_reason_code="confirmatory_rows_forbidden_for_recalibration",
            pass_value=False,
            gate_effect=gate_effect,
            diagnostics=(),
            effective_calibration_config=effective_config,
            calibration_identity=calibration_identity,
            lane_status="blocked",
        ).to_manifest(catalog)

    rows = tuple(prediction_artifact_manifest.body["rows"])
    minimum_sample_count = int(
        effective_config.get("thresholds", {}).get("minimum_sample_count", 1)
    )
    if len(rows) < minimum_sample_count:
        return CalibrationResultManifest(
            calibration_result_id=result_id,
            calibration_contract_ref=calibration_contract_manifest.ref,
            prediction_artifact_ref=prediction_artifact_manifest.ref,
            forecast_object_type=forecast_object_type,
            status="failed",
            failure_reason_code="insufficient_calibration_sample_count",
            pass_value=False,
            gate_effect=gate_effect,
            diagnostics=(),
            effective_calibration_config=effective_config,
            calibration_identity=calibration_identity,
            lane_status="failed",
        ).to_manifest(catalog)

    try:
        diagnostics = _diagnostics_for_forecast_object_type(
            forecast_object_type=forecast_object_type,
            rows=rows,
            thresholds=dict(effective_config.get("thresholds", {})),
            effective_config=effective_config,
        )
    except ContractValidationError as exc:
        return CalibrationResultManifest(
            calibration_result_id=result_id,
            calibration_contract_ref=calibration_contract_manifest.ref,
            prediction_artifact_ref=prediction_artifact_manifest.ref,
            forecast_object_type=forecast_object_type,
            status="failed",
            failure_reason_code=exc.code,
            pass_value=False,
            gate_effect=gate_effect,
            diagnostics=(),
            effective_calibration_config=effective_config,
            calibration_identity=calibration_identity,
            lane_status="failed",
        ).to_manifest(catalog)
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
        effective_calibration_config=effective_config,
        calibration_identity=calibration_identity,
        lane_status="passed" if passed else "failed",
    ).to_manifest(catalog)


def _diagnostics_for_forecast_object_type(
    *,
    forecast_object_type: str,
    rows: tuple[Mapping[str, Any], ...],
    thresholds: Mapping[str, float],
    effective_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if forecast_object_type == "distribution":
        return [_distribution_diagnostic(rows, thresholds, effective_config)]
    if forecast_object_type == "interval":
        return [_interval_diagnostic(rows, thresholds, effective_config)]
    if forecast_object_type == "quantile":
        return [_quantile_diagnostic(rows, thresholds, effective_config)]
    if forecast_object_type == "event_probability":
        return [_event_probability_diagnostic(rows, thresholds, effective_config)]
    raise ValueError(f"unsupported forecast_object_type {forecast_object_type!r}")


def _distribution_diagnostic(
    rows: tuple[Mapping[str, Any], ...],
    thresholds: Mapping[str, float],
    effective_config: Mapping[str, Any],
) -> dict[str, Any]:
    pit_values = []
    family_ids: set[str] = set()
    pit_config = dict(effective_config.get("pit", {}))
    randomized = bool(
        pit_config.get(
            "randomized",
            pit_config.get("method") == "randomized_pit",
        )
    )
    randomization_seed = str(pit_config.get("seed", "0"))
    for row in rows:
        model = bind_distribution_row_observation_model(row)
        family_ids.add(model.family_id)
        realized_observation = float(row["realized_observation"])
        pit_values.append(
            model.pit(
                realized_observation,
                randomized=randomized,
                row_key=_calibration_row_key(row),
                randomization_seed=randomization_seed,
            )
        )
    ks_distance = _ks_uniform_distance(tuple(pit_values))
    threshold = float(thresholds.get("max_ks_distance", 0.25))
    family_id = next(iter(family_ids)) if len(family_ids) == 1 else "mixed"
    return {
        "diagnostic_id": "pit_or_randomized_pit_uniformity",
        "family_id": family_id,
        "family_ids": sorted(family_ids),
        "pit_method": "randomized_pit" if randomized else "pit",
        "sample_size": len(pit_values),
        "mean_pit": _stable_float(fmean(pit_values)),
        "max_ks_distance": _stable_float(ks_distance),
        "status": "passed" if ks_distance <= threshold else "failed",
    }


def _interval_diagnostic(
    rows: tuple[Mapping[str, Any], ...],
    thresholds: Mapping[str, float],
    effective_config: Mapping[str, Any],
) -> dict[str, Any]:
    requested_levels = tuple(
        float(level) for level in effective_config.get("interval_levels", ())
    )
    observed_levels = sorted(
        {
            float(entry["nominal_coverage"])
            for row in rows
            for entry in _interval_entries(row)
        }
    )
    levels = requested_levels or tuple(observed_levels)
    threshold = float(thresholds.get("max_abs_coverage_gap", 0.1))
    level_diagnostics = []
    for level in levels:
        hits = []
        for row in rows:
            entry = _matching_interval_entry(row, level)
            if entry is None:
                continue
            observed = float(row["realized_observation"])
            hits.append(
                1.0
                if float(entry["lower_bound"])
                <= observed
                <= float(entry["upper_bound"])
                else 0.0
            )
        empirical_coverage = fmean(hits) if hits else math.nan
        absolute_gap = (
            abs(empirical_coverage - level) if math.isfinite(empirical_coverage) else math.inf
        )
        level_diagnostics.append(
            {
                "nominal_coverage": _stable_float(level),
                "sample_size": len(hits),
                "empirical_coverage": _stable_float(empirical_coverage),
                "absolute_gap": _stable_float(absolute_gap),
                "status": "passed" if absolute_gap <= threshold else "failed",
            }
        )
    absolute_gap = max(
        (float(item["absolute_gap"]) for item in level_diagnostics),
        default=math.inf,
    )
    nominal_coverage = fmean(levels) if levels else math.nan
    empirical_coverage = (
        fmean(float(item["empirical_coverage"]) for item in level_diagnostics)
        if level_diagnostics
        else math.nan
    )
    return {
        "diagnostic_id": "nominal_coverage",
        "nominal_coverage": _stable_float(nominal_coverage),
        "empirical_coverage": _stable_float(empirical_coverage),
        "absolute_gap": _stable_float(absolute_gap),
        "level_diagnostics": level_diagnostics,
        "status": "passed" if absolute_gap <= threshold else "failed",
    }


def _quantile_diagnostic(
    rows: tuple[Mapping[str, Any], ...],
    thresholds: Mapping[str, float],
    effective_config: Mapping[str, Any],
) -> dict[str, Any]:
    requested_levels = tuple(
        float(level) for level in effective_config.get("quantile_levels", ())
    )
    observed_levels = sorted(
        {float(item["level"]) for row in rows for item in row.get("quantiles", [])}
    )
    levels = requested_levels or tuple(observed_levels)
    threshold = float(thresholds.get("max_abs_hit_balance_gap", 0.15))
    level_diagnostics = []
    for level in levels:
        hits = []
        for row in rows:
            quantile_value = _matching_quantile_value(row, level)
            if quantile_value is None:
                continue
            hits.append(
                1.0 if float(row["realized_observation"]) <= quantile_value else 0.0
            )
        empirical_hit_rate = fmean(hits) if hits else math.nan
        absolute_gap = (
            abs(empirical_hit_rate - level)
            if math.isfinite(empirical_hit_rate)
            else math.inf
        )
        level_diagnostics.append(
            {
                "quantile_level": _stable_float(level),
                "sample_size": len(hits),
                "empirical_hit_rate": _stable_float(empirical_hit_rate),
                "absolute_gap": _stable_float(absolute_gap),
                "status": "passed" if absolute_gap <= threshold else "failed",
            }
        )
    max_abs_gap = max(
        (float(item["absolute_gap"]) for item in level_diagnostics),
        default=math.inf,
    )
    return {
        "diagnostic_id": "quantile_hit_balance",
        "quantile_levels": [_stable_float(level) for level in levels],
        "max_abs_gap": _stable_float(max_abs_gap),
        "level_diagnostics": level_diagnostics,
        "status": "passed" if max_abs_gap <= threshold else "failed",
    }


def _event_probability_diagnostic(
    rows: tuple[Mapping[str, Any], ...],
    thresholds: Mapping[str, float],
    effective_config: Mapping[str, Any],
) -> dict[str, Any]:
    records = tuple(
        (
            _stable_float(float(row["event_probability"])),
            1.0 if bool(row["realized_event"]) else 0.0,
        )
        for row in rows
    )
    bin_config = dict(effective_config.get("reliability_bins", {}))
    strategy = str(bin_config.get("strategy", "exact"))
    bins = _event_reliability_bins(records, bin_config)
    max_reliability_gap = max(
        (float(item["absolute_gap"]) for item in bins),
        default=math.inf,
    )
    threshold = float(thresholds.get("max_reliability_gap", 0.2))
    return {
        "diagnostic_id": "reliability_curve_or_binned_frequency",
        "bin_strategy": strategy,
        "bin_count": len(bins),
        "bins": bins,
        "max_reliability_gap": _stable_float(max_reliability_gap),
        "status": "passed" if max_reliability_gap <= threshold else "failed",
    }


def _effective_calibration_config(
    calibration_contract_manifest: ManifestEnvelope,
) -> dict[str, Any]:
    body = calibration_contract_manifest.body
    return {
        "forecast_object_type": str(body["forecast_object_type"]),
        "thresholds": dict(body.get("thresholds", {})),
        "reliability_bins": dict(body.get("reliability_bins", {})),
        "pit": dict(body.get("pit", {})),
        "interval_levels": [float(level) for level in body.get("interval_levels", [])],
        "quantile_levels": [float(level) for level in body.get("quantile_levels", [])],
        "calibration_lane": str(body.get("calibration_lane", "evaluation_only")),
    }


def _calibration_identity(
    *,
    calibration_contract_manifest: ManifestEnvelope,
    prediction_artifact_manifest: ManifestEnvelope,
) -> dict[str, Any]:
    rows = tuple(prediction_artifact_manifest.body.get("rows", ()))
    distribution_families = sorted(
        {
            str(row["distribution_family"])
            for row in rows
            if row.get("distribution_family") is not None
        }
    )
    identity = {
        "calibration_contract_ref": calibration_contract_manifest.ref.as_dict(),
        "prediction_artifact_ref": prediction_artifact_manifest.ref.as_dict(),
        "forecast_object_type": str(
            prediction_artifact_manifest.body.get("forecast_object_type", "")
        ),
        "score_policy_ref": dict(
            prediction_artifact_manifest.body.get("score_policy_ref", {})
        ),
        "horizon_set": sorted({int(row["horizon"]) for row in rows}),
        "scored_origin_set_id": prediction_artifact_manifest.body.get(
            "scored_origin_set_id"
        ),
        "entity_panel": list(prediction_artifact_manifest.body.get("entity_panel", [])),
        "calibration_lane": str(
            calibration_contract_manifest.body.get(
                "calibration_lane",
                "evaluation_only",
            )
        ),
    }
    if distribution_families:
        identity["distribution_families"] = distribution_families
    return identity


def _calibration_row_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(row.get("entity", "")),
            str(row["origin_time"]),
            str(row["horizon"]),
        )
    )


def _interval_entries(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_intervals = row.get("intervals")
    if isinstance(raw_intervals, list) and raw_intervals:
        return tuple(dict(item) for item in raw_intervals)
    return (
        {
            "nominal_coverage": float(row["nominal_coverage"]),
            "lower_bound": float(row["lower_bound"]),
            "upper_bound": float(row["upper_bound"]),
        },
    )


def _matching_interval_entry(
    row: Mapping[str, Any],
    level: float,
) -> Mapping[str, Any] | None:
    for entry in _interval_entries(row):
        if float(entry["nominal_coverage"]) == float(level):
            return entry
    return None


def _matching_quantile_value(
    row: Mapping[str, Any],
    level: float,
) -> float | None:
    for item in row.get("quantiles", []):
        if float(item["level"]) == float(level):
            return float(item["value"])
    return None


def _event_reliability_bins(
    records: tuple[tuple[float, float], ...],
    bin_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    strategy = str(bin_config.get("strategy", "exact"))
    if strategy == "equal_width":
        groups = _equal_width_groups(
            records,
            bin_count=max(1, int(bin_config.get("bin_count", 10))),
        )
    elif strategy == "equal_mass":
        groups = _equal_mass_groups(
            records,
            bin_count=max(1, int(bin_config.get("bin_count", 10))),
        )
    elif strategy == "adaptive_min_count":
        groups = _adaptive_min_count_groups(
            records,
            minimum_bin_count=max(1, int(bin_config.get("minimum_bin_count", 10))),
        )
    else:
        groups = _exact_probability_groups(records)
    return [_event_bin_summary(group) for group in groups if group]


def _exact_probability_groups(
    records: tuple[tuple[float, float], ...],
) -> list[tuple[tuple[float, float], ...]]:
    grouped: dict[float, list[tuple[float, float]]] = {}
    for record in records:
        grouped.setdefault(record[0], []).append(record)
    return [tuple(grouped[key]) for key in sorted(grouped)]


def _equal_width_groups(
    records: tuple[tuple[float, float], ...],
    *,
    bin_count: int,
) -> list[tuple[tuple[float, float], ...]]:
    groups: list[list[tuple[float, float]]] = [[] for _ in range(bin_count)]
    for record in records:
        probability = max(0.0, min(1.0, record[0]))
        index = min(int(probability * bin_count), bin_count - 1)
        groups[index].append(record)
    return [tuple(group) for group in groups]


def _equal_mass_groups(
    records: tuple[tuple[float, float], ...],
    *,
    bin_count: int,
) -> list[tuple[tuple[float, float], ...]]:
    if not records:
        return []
    ordered = tuple(sorted(records, key=lambda item: item[0]))
    resolved_bin_count = min(bin_count, len(ordered))
    chunk_size = math.ceil(len(ordered) / resolved_bin_count)
    return [
        ordered[index : index + chunk_size]
        for index in range(0, len(ordered), chunk_size)
    ]


def _adaptive_min_count_groups(
    records: tuple[tuple[float, float], ...],
    *,
    minimum_bin_count: int,
) -> list[tuple[tuple[float, float], ...]]:
    if not records:
        return []
    ordered = tuple(sorted(records, key=lambda item: item[0]))
    groups = [
        ordered[index : index + minimum_bin_count]
        for index in range(0, len(ordered), minimum_bin_count)
    ]
    if len(groups) > 1 and len(groups[-1]) < minimum_bin_count:
        groups[-2] = groups[-2] + groups[-1]
        groups.pop()
    return groups


def _event_bin_summary(group: tuple[tuple[float, float], ...]) -> dict[str, Any]:
    probabilities = tuple(probability for probability, _ in group)
    outcomes = tuple(outcome for _, outcome in group)
    mean_probability = fmean(probabilities)
    empirical_frequency = fmean(outcomes)
    return {
        "probability_min": _stable_float(min(probabilities)),
        "probability_max": _stable_float(max(probabilities)),
        "mean_probability": _stable_float(mean_probability),
        "empirical_frequency": _stable_float(empirical_frequency),
        "sample_count": len(group),
        "absolute_gap": _stable_float(abs(empirical_frequency - mean_probability)),
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
