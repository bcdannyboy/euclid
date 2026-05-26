from __future__ import annotations

import math
from dataclasses import dataclass, field
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

_VALID_HORIZON_POOLING_POLICIES = {
    "explicit_horizon_pooling_v1",
    "valid_explicit_horizon_pooling",
}


@dataclass(frozen=True)
class CalibrationPartition:
    horizon: int | str | None = None
    entity_id: str | None = None
    regime_id: str | None = None
    partition_id: str | None = None
    calibration_split_id: str | None = None
    horizon_id: int | None = None
    target_horizon_id: int | None = None
    residuals: tuple[float, ...] = ()
    coverage_hits: tuple[bool, ...] = ()
    pooled_horizons: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "residuals",
            tuple(float(value) for value in self.residuals),
        )
        object.__setattr__(
            self,
            "coverage_hits",
            tuple(bool(value) for value in self.coverage_hits),
        )
        object.__setattr__(
            self,
            "pooled_horizons",
            tuple(int(value) for value in self.pooled_horizons),
        )

    @property
    def resolved_horizon(self) -> int | str | None:
        if self.horizon is not None:
            return self.horizon
        return self.horizon_id

    @property
    def resolved_partition_id(self) -> str:
        if self.partition_id:
            return self.partition_id
        tokens = [f"horizon={self.resolved_horizon}"]
        if self.entity_id is not None:
            tokens.append(f"entity={self.entity_id}")
        if self.regime_id is not None:
            tokens.append(f"regime={self.regime_id}")
        return "|".join(tokens)

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "partition_id": self.resolved_partition_id,
            "horizon": self.resolved_horizon,
        }
        if self.entity_id is not None:
            body["entity_id"] = self.entity_id
        if self.regime_id is not None:
            body["regime_id"] = self.regime_id
        if self.pooled_horizons:
            body["pooled_horizons"] = list(self.pooled_horizons)
        return body


@dataclass(frozen=True)
class CalibrationPartitionEvaluation:
    status: str
    promotion_allowed: bool
    reason_codes: tuple[str, ...]
    partition_diagnostics: tuple[Mapping[str, Any], ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "calibration_partition_evaluation@1.0.0",
            "status": self.status,
            "promotion_allowed": self.promotion_allowed,
            "reason_codes": list(self.reason_codes),
            "partition_diagnostics": [
                dict(diagnostic) for diagnostic in self.partition_diagnostics
            ],
        }


@dataclass(frozen=True)
class MapieTimeSeriesAdapterResult:
    status: str
    backend: str
    method_name: str
    calibration_indices: tuple[int, ...]
    assumptions: Mapping[str, Any] = field(default_factory=dict)
    backend_version: str | None = None
    failure_reason_code: str | None = None
    reason_codes: tuple[str, ...] = ()
    promotion_allowed: bool = False

    def as_manifest(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_name": "mapie_time_series_adapter_result@1.0.0",
            "status": self.status,
            "promotion_allowed": self.promotion_allowed,
            "backend": self.backend,
            "method_name": self.method_name,
            "calibration_indices": list(self.calibration_indices),
            "assumptions": {
                str(key): self.assumptions[key] for key in sorted(self.assumptions)
            },
            "reason_codes": list(self.reason_codes),
        }
        if self.backend_version is not None:
            body["backend_version"] = self.backend_version
        if self.failure_reason_code is not None:
            body["failure_reason_code"] = self.failure_reason_code
        return body


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
    is_fit_lane = lane in {"recalibration_fit", "conformal_fit"} or (
        ("recalibration" in lane or "conformal" in lane) and lane.endswith("_fit")
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
    split_role_failure = _calibration_split_role_failure(
        rows=rows,
        is_fit_lane=is_fit_lane,
    )
    if split_role_failure is not None:
        return CalibrationResultManifest(
            calibration_result_id=result_id,
            calibration_contract_ref=calibration_contract_manifest.ref,
            prediction_artifact_ref=prediction_artifact_manifest.ref,
            forecast_object_type=forecast_object_type,
            status="failed",
            failure_reason_code=split_role_failure,
            pass_value=False,
            gate_effect=gate_effect,
            diagnostics=(
                _calibration_split_role_diagnostic(
                    rows=rows,
                    failure_reason_code=split_role_failure,
                ),
            ),
            effective_calibration_config=effective_config,
            calibration_identity=calibration_identity,
            lane_status="blocked",
        ).to_manifest(catalog)

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

    partition_groups = _calibration_partition_groups(
        rows=rows,
        effective_config=effective_config,
    )
    insufficient_partitions = [
        (partition, partition_rows)
        for partition, partition_rows in partition_groups
        if len(partition_rows) < minimum_sample_count
    ]
    if insufficient_partitions:
        return CalibrationResultManifest(
            calibration_result_id=result_id,
            calibration_contract_ref=calibration_contract_manifest.ref,
            prediction_artifact_ref=prediction_artifact_manifest.ref,
            forecast_object_type=forecast_object_type,
            status="failed",
            failure_reason_code="insufficient_calibration_partition_count",
            pass_value=False,
            gate_effect=gate_effect,
            diagnostics=(
                _partition_sample_count_diagnostic(
                    insufficient_partitions=tuple(insufficient_partitions),
                    minimum_sample_count=minimum_sample_count,
                ),
            ),
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
    if _diagnostics_require_partitioning(rows, effective_config):
        return [
            _partitioned_diagnostic(
                forecast_object_type=forecast_object_type,
                rows=rows,
                thresholds=thresholds,
                effective_config=effective_config,
            )
        ]
    return [
        _single_diagnostic_for_forecast_object_type(
            forecast_object_type=forecast_object_type,
            rows=rows,
            thresholds=thresholds,
            effective_config=effective_config,
        )
    ]


def _single_diagnostic_for_forecast_object_type(
    *,
    forecast_object_type: str,
    rows: tuple[Mapping[str, Any], ...],
    thresholds: Mapping[str, float],
    effective_config: Mapping[str, Any],
) -> dict[str, Any]:
    if forecast_object_type == "distribution":
        return _distribution_diagnostic(rows, thresholds, effective_config)
    if forecast_object_type == "interval":
        return _interval_diagnostic(rows, thresholds, effective_config)
    if forecast_object_type == "quantile":
        return _quantile_diagnostic(rows, thresholds, effective_config)
    if forecast_object_type == "event_probability":
        return _event_probability_diagnostic(rows, thresholds, effective_config)
    raise ValueError(f"unsupported forecast_object_type {forecast_object_type!r}")


def _partitioned_diagnostic(
    *,
    forecast_object_type: str,
    rows: tuple[Mapping[str, Any], ...],
    thresholds: Mapping[str, float],
    effective_config: Mapping[str, Any],
) -> dict[str, Any]:
    partition_diagnostics = []
    for partition, partition_rows in _calibration_partition_groups(
        rows=rows,
        effective_config=effective_config,
    ):
        diagnostic = _single_diagnostic_for_forecast_object_type(
            forecast_object_type=forecast_object_type,
            rows=partition_rows,
            thresholds=thresholds,
            effective_config=effective_config,
        )
        partition_diagnostics.append(
            {
                "partition": partition.as_dict(),
                **diagnostic,
            }
        )
    combined = _single_diagnostic_for_forecast_object_type(
        forecast_object_type=forecast_object_type,
        rows=rows,
        thresholds=thresholds,
        effective_config=effective_config,
    )
    combined["partition_diagnostics"] = partition_diagnostics
    combined["status"] = (
        "passed"
        if partition_diagnostics
        and all(item["status"] == "passed" for item in partition_diagnostics)
        else "failed"
    )
    if forecast_object_type == "interval":
        combined["absolute_gap"] = _stable_float(
            max(
                (
                    float(item.get("absolute_gap", math.inf))
                    for item in partition_diagnostics
                ),
                default=math.inf,
            )
        )
        combined["empirical_coverage"] = _stable_float(
            min(
                (
                    float(item.get("empirical_coverage", math.inf))
                    for item in partition_diagnostics
                ),
                default=math.nan,
            )
        )
    return combined


def _diagnostics_require_partitioning(
    rows: tuple[Mapping[str, Any], ...],
    effective_config: Mapping[str, Any],
) -> bool:
    if _explicit_horizon_pooling_policy(effective_config) is not None:
        return True
    if bool(effective_config.get("partition_by_entity", False)):
        return True
    if bool(effective_config.get("partition_by_regime", False)):
        return True
    return len({int(row["horizon"]) for row in rows}) > 1


def _calibration_partition_groups(
    *,
    rows: tuple[Mapping[str, Any], ...],
    effective_config: Mapping[str, Any],
) -> tuple[tuple[CalibrationPartition, tuple[Mapping[str, Any], ...]], ...]:
    horizon_pooling_policy = _explicit_horizon_pooling_policy(effective_config)
    partition_by_entity = bool(effective_config.get("partition_by_entity", False))
    partition_by_regime = bool(effective_config.get("partition_by_regime", False))
    grouped: dict[tuple[Any, str | None, str | None], list[Mapping[str, Any]]] = {}
    for row in rows:
        horizon_key: int | str = (
            "pooled" if horizon_pooling_policy is not None else int(row["horizon"])
        )
        entity_id = (
            str(row["entity"]) if partition_by_entity and row.get("entity") else None
        )
        regime_id = (
            str(row["regime"]) if partition_by_regime and row.get("regime") else None
        )
        grouped.setdefault((horizon_key, entity_id, regime_id), []).append(row)

    pooled_horizons = tuple(sorted({int(row["horizon"]) for row in rows}))
    partitions = []
    for horizon, entity_id, regime_id in sorted(
        grouped,
        key=lambda item: (_partition_sort_value(item[0]), item[1] or "", item[2] or ""),
    ):
        partition = CalibrationPartition(
            horizon=horizon,
            entity_id=entity_id,
            regime_id=regime_id,
            pooled_horizons=pooled_horizons if horizon == "pooled" else (),
        )
        partitions.append((partition, tuple(grouped[(horizon, entity_id, regime_id)])))
    return tuple(partitions)


def _explicit_horizon_pooling_policy(
    effective_config: Mapping[str, Any],
) -> str | None:
    raw_policy = effective_config.get("horizon_pooling_policy")
    if raw_policy is None:
        return None
    policy = str(raw_policy)
    if policy in _VALID_HORIZON_POOLING_POLICIES:
        return policy
    return None


def _partition_sort_value(value: Any) -> tuple[int, Any]:
    if isinstance(value, int):
        return (0, value)
    return (1, str(value))


def _partition_sample_count_diagnostic(
    *,
    insufficient_partitions: tuple[
        tuple[CalibrationPartition, tuple[Mapping[str, Any], ...]], ...
    ],
    minimum_sample_count: int,
) -> dict[str, Any]:
    return {
        "diagnostic_id": "calibration_partition_sample_count",
        "minimum_sample_count": minimum_sample_count,
        "partition_diagnostics": [
            {
                "partition": partition.as_dict(),
                "sample_size": len(partition_rows),
                "minimum_sample_count": minimum_sample_count,
                "status": "failed",
                "reason_code": "insufficient_calibration_partition_count",
            }
            for partition, partition_rows in insufficient_partitions
        ],
        "status": "failed",
    }


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
    lower_bound_threshold = thresholds.get("minimum_empirical_coverage_lower_bound")
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
            abs(empirical_coverage - level)
            if math.isfinite(empirical_coverage)
            else math.inf
        )
        lower_bound_status = "not_evaluated"
        empirical_coverage_lower_bound = math.nan
        if lower_bound_threshold is not None:
            empirical_coverage_lower_bound = _wilson_lower_bound(
                success_count=sum(1 for hit in hits if hit),
                sample_size=len(hits),
            )
            lower_bound_status = (
                "passed"
                if empirical_coverage_lower_bound >= float(lower_bound_threshold)
                else "failed"
            )
        status = "passed" if absolute_gap <= threshold else "failed"
        if lower_bound_status == "failed":
            status = "failed"
        level_diagnostic = {
            "nominal_coverage": _stable_float(level),
            "sample_size": len(hits),
            "empirical_coverage": _stable_float(empirical_coverage),
            "absolute_gap": _stable_float(absolute_gap),
            "status": status,
        }
        if lower_bound_threshold is not None:
            level_diagnostic.update(
                {
                    "empirical_coverage_lower_bound": _stable_float(
                        empirical_coverage_lower_bound
                    ),
                    "empirical_coverage_lower_bound_method": "wilson_score",
                    "minimum_empirical_coverage_lower_bound": _stable_float(
                        float(lower_bound_threshold)
                    ),
                    "empirical_coverage_lower_bound_status": lower_bound_status,
                }
            )
        level_diagnostics.append(level_diagnostic)
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
        "status": (
            "passed"
            if level_diagnostics
            and all(item["status"] == "passed" for item in level_diagnostics)
            else "failed"
        ),
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
        "partition_by_entity": bool(body.get("partition_by_entity", False)),
        "partition_by_regime": bool(body.get("partition_by_regime", False)),
        "horizon_pooling_policy": body.get("horizon_pooling_policy"),
    }


def _calibration_split_role_failure(
    *,
    rows: tuple[Mapping[str, Any], ...],
    is_fit_lane: bool,
) -> str | None:
    if not is_fit_lane:
        return None
    if any(not _row_split_role(row) for row in rows):
        return "missing_calibration_split_role_metadata"
    if any(_row_split_role(row) in _CONFIRMATORY_SPLIT_ROLES for row in rows):
        return "confirmatory_rows_forbidden_for_recalibration"
    return None


def _calibration_split_role_diagnostic(
    *,
    rows: tuple[Mapping[str, Any], ...],
    failure_reason_code: str,
) -> dict[str, Any]:
    counts = _split_role_counts(rows)
    return {
        "diagnostic_id": "calibration_split_independence",
        "status": "failed",
        "failure_reason_code": failure_reason_code,
        "split_role_counts": counts,
        "confirmatory_roles": sorted(_CONFIRMATORY_SPLIT_ROLES),
    }


_CONFIRMATORY_SPLIT_ROLES = {
    "confirmatory",
    "confirmatory_holdout",
    "holdout",
    "test",
    "test_holdout",
}


def _row_split_role(row: Mapping[str, Any]) -> str:
    value = row.get("split_role")
    return str(value).strip().lower() if value is not None else ""


def _split_role_counts(rows: tuple[Mapping[str, Any], ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        role = _row_split_role(row) or "missing"
        counts[role] = counts.get(role, 0) + 1
    return dict(sorted(counts.items()))


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
        "stage_id": prediction_artifact_manifest.body.get("stage_id"),
        "fit_window_id": prediction_artifact_manifest.body.get("fit_window_id"),
        "test_window_id": prediction_artifact_manifest.body.get("test_window_id"),
        "calibration_split_id": prediction_artifact_manifest.body.get(
            "calibration_split_id"
        )
        or calibration_contract_manifest.body.get("calibration_split_id"),
        "split_role_counts": _split_role_counts(rows),
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


def evaluate_calibration_partitions(
    *,
    calibration_partitions: Sequence[CalibrationPartition],
    target_horizon_ids: Sequence[int],
    partition_policy: Mapping[str, Any],
    minimum_calibration_count: int,
    nominal_coverage: float = 0.8,
    minimum_empirical_coverage_lower_bound: float = 0.0,
) -> CalibrationPartitionEvaluation:
    pooling_policy = str(partition_policy.get("horizon_pooling", "none"))
    target_horizons = tuple(int(horizon) for horizon in target_horizon_ids)
    diagnostics = []
    reason_codes: list[str] = []
    for partition in calibration_partitions:
        diagnostic = _calibration_partition_payload_diagnostic(
            partition=partition,
            pooling_policy=pooling_policy,
            minimum_calibration_count=int(minimum_calibration_count),
            nominal_coverage=float(nominal_coverage),
            minimum_empirical_coverage_lower_bound=float(
                minimum_empirical_coverage_lower_bound
            ),
        )
        diagnostics.append(diagnostic)
        for reason_code in diagnostic.get("reason_codes", []):
            if reason_code not in reason_codes:
                reason_codes.append(str(reason_code))

    cross_horizon_blocked = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic["horizon_id"] != diagnostic["target_horizon_id"]
        and diagnostic["target_horizon_id"] in target_horizons
        and pooling_policy not in _VALID_HORIZON_POOLING_POLICIES
    ]
    if cross_horizon_blocked:
        reason_codes = ["cross_horizon_calibration_without_valid_pooling_policy"]
        diagnostics = [
            {
                "partition_id": diagnostic["partition_id"],
                "horizon_id": diagnostic["horizon_id"],
                "target_horizon_id": diagnostic["target_horizon_id"],
                "pooling_policy": pooling_policy,
            }
            for diagnostic in diagnostics
        ]
        return CalibrationPartitionEvaluation(
            status="blocked",
            promotion_allowed=False,
            reason_codes=tuple(reason_codes),
            partition_diagnostics=tuple(diagnostics),
        )

    if any(diagnostic["status"] == "blocked" for diagnostic in diagnostics):
        status = "blocked"
    elif any(diagnostic["status"] == "failed" for diagnostic in diagnostics):
        status = "failed"
    else:
        status = "passed"
    return CalibrationPartitionEvaluation(
        status=status,
        promotion_allowed=status == "passed",
        reason_codes=tuple(reason_codes),
        partition_diagnostics=tuple(diagnostics),
    )


def run_mapie_time_series_adapter(
    *,
    method_name: str,
    calibration_indices: Sequence[int],
    assumptions: Mapping[str, Any],
    optional_backend_overrides: Mapping[str, Any] | None = None,
) -> MapieTimeSeriesAdapterResult:
    backend = dict(optional_backend_overrides or {}).get("mapie", "auto")
    if backend is None:
        return MapieTimeSeriesAdapterResult(
            status="unavailable",
            backend="mapie",
            method_name=str(method_name),
            calibration_indices=tuple(int(index) for index in calibration_indices),
            assumptions=dict(assumptions),
            failure_reason_code="calibration_backend_unavailable",
            reason_codes=("calibration_backend_unavailable",),
            promotion_allowed=False,
        )
    if isinstance(backend, Mapping):
        available = bool(backend.get("available", False))
        version = str(backend.get("version", "unknown")) if available else None
        if not available:
            return MapieTimeSeriesAdapterResult(
                status="unavailable",
                backend="mapie",
                method_name=str(method_name),
                calibration_indices=tuple(int(index) for index in calibration_indices),
                assumptions=dict(assumptions),
                failure_reason_code="calibration_backend_unavailable",
                reason_codes=("calibration_backend_unavailable",),
                promotion_allowed=False,
            )
        return MapieTimeSeriesAdapterResult(
            status="passed",
            backend="mapie",
            backend_version=version,
            method_name=str(method_name),
            calibration_indices=tuple(int(index) for index in calibration_indices),
            assumptions=dict(assumptions),
            reason_codes=(),
            promotion_allowed=True,
        )
    return MapieTimeSeriesAdapterResult(
        status="unavailable",
        backend="mapie",
        method_name=str(method_name),
        calibration_indices=tuple(int(index) for index in calibration_indices),
        assumptions=dict(assumptions),
        failure_reason_code="calibration_backend_unavailable",
        reason_codes=("calibration_backend_unavailable",),
        promotion_allowed=False,
    )


def normalize_calibration_method_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if metadata is None:
        return None
    calibration_partition_ids = tuple(
        str(value) for value in metadata.get("calibration_partition_ids", ())
    )
    calibration_partition_id = metadata.get("calibration_partition_id")
    if calibration_partition_id is not None:
        calibration_partition_ids = (
            str(calibration_partition_id),
            *calibration_partition_ids,
        )
    resolved_partition_ids = tuple(_unique_values(calibration_partition_ids))
    backend = dict(metadata.get("backend", {}))
    body = {
        "method_id": str(metadata.get("method_id", "")),
        "status": str(metadata.get("status", "unknown")),
        "reason_codes": [
            str(reason_code) for reason_code in metadata.get("reason_codes", ())
        ],
        "guarantee_tier": str(metadata.get("guarantee_tier", "diagnostic_only")),
        "assumption_ids": [
            str(assumption_id) for assumption_id in metadata.get("assumption_ids", ())
        ],
        "assumptions": _stable_any_mapping(dict(metadata.get("assumptions", {}))),
        "assumption_scope": metadata.get("assumption_scope"),
        "calibration_partition_id": (
            resolved_partition_ids[0] if resolved_partition_ids else None
        ),
        "calibration_partition_ids": list(resolved_partition_ids),
        "horizon_ids": [int(horizon) for horizon in metadata.get("horizon_ids", ())],
        "calibration_indices": [
            int(index) for index in metadata.get("calibration_indices", ())
        ],
        "backend": _stable_any_mapping(backend),
    }
    return body


def mapie_backend_metadata() -> dict[str, Any]:
    try:
        from euclid.modules import probabilistic_evaluation

        version = probabilistic_evaluation.importlib_metadata.version("mapie")
    except Exception:
        return {
            "backend": "mapie",
            "status": "unavailable",
            "backend_version": None,
            "failure_reason_code": "calibration_backend_unavailable",
            "reason_codes": ["calibration_backend_unavailable"],
        }
    return {
        "backend": "mapie",
        "status": "available",
        "backend_version": str(version),
        "reason_codes": [],
    }


def _calibration_partition_payload_diagnostic(
    *,
    partition: CalibrationPartition,
    pooling_policy: str,
    minimum_calibration_count: int,
    nominal_coverage: float,
    minimum_empirical_coverage_lower_bound: float,
) -> dict[str, Any]:
    calibration_count = len(partition.coverage_hits)
    reason_codes: list[str] = []
    if calibration_count < minimum_calibration_count:
        reason_codes.append("insufficient_calibration_count_for_partition")
        return {
            "partition_id": partition.resolved_partition_id,
            "horizon_id": partition.horizon_id,
            "target_horizon_id": partition.target_horizon_id,
            "entity_id": partition.entity_id,
            "regime_id": partition.regime_id,
            "calibration_count": calibration_count,
            "minimum_calibration_count": minimum_calibration_count,
            "status": "blocked",
            "reason_codes": reason_codes,
        }

    empirical_coverage = (
        fmean(1.0 if hit else 0.0 for hit in partition.coverage_hits)
        if partition.coverage_hits
        else math.nan
    )
    coverage_lower_bound = _wilson_lower_bound(
        success_count=sum(1 for hit in partition.coverage_hits if hit),
        sample_size=calibration_count,
    )
    absolute_gap = (
        abs(empirical_coverage - nominal_coverage)
        if math.isfinite(empirical_coverage)
        else math.inf
    )
    if (
        partition.regime_id is not None
        and coverage_lower_bound < minimum_empirical_coverage_lower_bound
    ):
        reason_codes.append("regime_slice_undercoverage")
    status = "failed" if reason_codes else "passed"
    return {
        "partition_id": partition.resolved_partition_id,
        "horizon_id": partition.horizon_id,
        "target_horizon_id": partition.target_horizon_id,
        "entity_id": partition.entity_id,
        "regime_id": partition.regime_id,
        "calibration_count": calibration_count,
        "minimum_calibration_count": minimum_calibration_count,
        "nominal_coverage": _stable_float(nominal_coverage),
        "empirical_coverage": _stable_float(empirical_coverage),
        "coverage_lower_bound": _stable_float(coverage_lower_bound),
        "absolute_gap": _stable_float(absolute_gap),
        "pooling_policy": pooling_policy,
        "status": status,
        "reason_codes": reason_codes,
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


def _wilson_lower_bound(
    *,
    success_count: int,
    sample_size: int,
    z_value: float = 1.959963984540054,
) -> float:
    if sample_size <= 0:
        return math.nan
    successes = float(success_count)
    n = float(sample_size)
    proportion = successes / n
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / n
    center = proportion + z_squared / (2.0 * n)
    margin = z_value * math.sqrt(
        (proportion * (1.0 - proportion) / n) + (z_squared / (4.0 * n * n))
    )
    return max(0.0, (center - margin) / denominator)


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


def _stable_any_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): mapping[key] for key in sorted(mapping)}


def _unique_values(values: Sequence[Any]) -> tuple[Any, ...]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


__all__ = [
    "CalibrationPartition",
    "CalibrationPartitionEvaluation",
    "MapieTimeSeriesAdapterResult",
    "build_calibration_contract",
    "evaluate_calibration_partitions",
    "evaluate_prediction_calibration",
    "mapie_backend_metadata",
    "normalize_calibration_method_metadata",
    "run_mapie_time_series_adapter",
]
