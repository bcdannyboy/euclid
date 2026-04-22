from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS


@dataclass(frozen=True)
class PredictivePromotionResult:
    status: str
    promotion_allowed: bool
    reason_codes: tuple[str, ...]
    mean_loss_differential: float
    confidence_interval: tuple[float, float] | None
    practical_margin: float
    raw_metric_comparison_role: str
    statistical_test_backend: str
    confidence_interval_method: str
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "paired_predictive_test_result@1.0.0",
            "status": self.status,
            "promotion_allowed": self.promotion_allowed,
            "reason_codes": list(self.reason_codes),
            "mean_loss_differential": self.mean_loss_differential,
            "confidence_interval": (
                list(self.confidence_interval)
                if self.confidence_interval is not None
                else None
            ),
            "confidence_interval_method": self.confidence_interval_method,
            "practical_margin": self.practical_margin,
            "raw_metric_comparison_role": self.raw_metric_comparison_role,
            "statistical_test_backend": self.statistical_test_backend,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class PrequentialScoreStream:
    stream_id: str
    candidate_id: str
    baseline_id: str
    per_origin: tuple[Mapping[str, Any], ...]
    per_horizon: tuple[Mapping[str, Any], ...]
    per_entity: tuple[Mapping[str, Any], ...]
    per_regime: tuple[Mapping[str, Any], ...]
    rolling_degradation: tuple[Mapping[str, Any], ...]
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "prequential_score_stream@1.0.0",
            "stream_id": self.stream_id,
            "candidate_id": self.candidate_id,
            "baseline_id": self.baseline_id,
            "per_origin": [dict(item) for item in self.per_origin],
            "per_horizon": [dict(item) for item in self.per_horizon],
            "per_entity": [dict(item) for item in self.per_entity],
            "per_regime": [dict(item) for item in self.per_regime],
            "rolling_degradation": [dict(item) for item in self.rolling_degradation],
            "replay_identity": self.replay_identity,
        }


def evaluate_predictive_promotion(
    *,
    candidate_losses: Sequence[float],
    baseline_losses: Sequence[float],
    split_protocol_id: str,
    baseline_id: str | None,
    practical_margin: float,
    calibration_status: str = "not_applicable_for_forecast_type",
    leakage_status: str = "passed",
) -> PredictivePromotionResult:
    reason_codes: list[str] = []
    candidate = _finite_tuple(candidate_losses)
    baseline = _finite_tuple(baseline_losses)
    if baseline_id is None or not baseline:
        reason_codes.append("missing_baseline")
    if split_protocol_id in {"", "train_only", "in_sample_only"}:
        reason_codes.append("unstable_split_protocol")
    if leakage_status != "passed":
        reason_codes.append("leakage_detected")
    if calibration_status in {"failed", "coverage_failed", "poor_coverage"}:
        reason_codes.append("calibration_failed")
        if calibration_status in {"coverage_failed", "poor_coverage"}:
            reason_codes.append("poor_coverage")
    if len(candidate) != len(baseline) or not candidate:
        reason_codes.append("unpaired_loss_stream")

    if reason_codes:
        return _promotion_result(
            status="abstained",
            promotion_allowed=False,
            reason_codes=_unique(reason_codes),
            mean_loss_differential=0.0,
            confidence_interval=None,
            practical_margin=practical_margin,
        )

    differentials = tuple(
        baseline_loss - candidate_loss
        for candidate_loss, baseline_loss in zip(candidate, baseline, strict=True)
    )
    mean_differential = _stable_float(fmean(differentials))
    if math.isclose(mean_differential, 0.0, abs_tol=1e-12):
        reason_codes.append("baseline_tie")
    if mean_differential <= float(practical_margin):
        reason_codes.append("insignificant_improvement")
    confidence_interval = _statsmodels_hac_ci(
        differentials=differentials,
        maxlags=_hac_maxlags(len(differentials)),
    )
    if confidence_interval[0] <= float(practical_margin):
        reason_codes.append("uncertainty_interval_crosses_margin")

    unique_reasons = _unique(reason_codes)
    status = "passed" if not unique_reasons else "downgraded"
    return _promotion_result(
        status=status,
        promotion_allowed=status == "passed",
        reason_codes=unique_reasons,
        mean_loss_differential=mean_differential,
        confidence_interval=confidence_interval,
        practical_margin=practical_margin,
    )


def build_prequential_score_stream(
    *,
    stream_id: str,
    candidate_id: str,
    baseline_id: str,
    rows: Sequence[Mapping[str, Any]],
    rolling_window: int = 5,
) -> PrequentialScoreStream:
    per_origin: list[dict[str, Any]] = []
    for row in rows:
        candidate_loss = float(row["candidate_loss"])
        baseline_loss = float(row["baseline_loss"])
        per_origin.append(
            {
                "origin_id": str(row["origin_id"]),
                "horizon": int(row["horizon"]),
                "entity": str(row.get("entity", "")),
                "regime": str(row.get("regime", "")),
                "candidate_loss": _stable_float(candidate_loss),
                "baseline_loss": _stable_float(baseline_loss),
                "loss_difference": _stable_float(baseline_loss - candidate_loss),
            }
        )
    per_horizon = _group_mean(per_origin, "horizon")
    per_entity = _group_mean(per_origin, "entity")
    per_regime = _group_mean(per_origin, "regime")
    rolling = _rolling_degradation(per_origin, rolling_window=max(1, rolling_window))
    identity_payload = {
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "per_origin": per_origin,
        "stream_id": stream_id,
    }
    return PrequentialScoreStream(
        stream_id=str(stream_id),
        candidate_id=str(candidate_id),
        baseline_id=str(baseline_id),
        per_origin=tuple(per_origin),
        per_horizon=tuple(per_horizon),
        per_entity=tuple(per_entity),
        per_regime=tuple(per_regime),
        rolling_degradation=tuple(rolling),
        replay_identity=f"prequential-stream:{_digest(identity_payload)}",
    )


def _promotion_result(
    *,
    status: str,
    promotion_allowed: bool,
    reason_codes: tuple[str, ...],
    mean_loss_differential: float,
    confidence_interval: tuple[float, float] | None,
    practical_margin: float,
) -> PredictivePromotionResult:
    payload = {
        "confidence_interval": list(confidence_interval)
        if confidence_interval is not None
        else None,
        "mean_loss_differential": mean_loss_differential,
        "practical_margin": float(practical_margin),
        "reason_codes": list(reason_codes),
        "statistical_test_backend": "statsmodels_hac_mean_loss_differential",
        "status": status,
    }
    return PredictivePromotionResult(
        status=status,
        promotion_allowed=promotion_allowed,
        reason_codes=reason_codes,
        mean_loss_differential=_stable_float(mean_loss_differential),
        confidence_interval=confidence_interval,
        practical_margin=_stable_float(float(practical_margin)),
        raw_metric_comparison_role="diagnostic_only",
        statistical_test_backend="statsmodels_hac_mean_loss_differential",
        confidence_interval_method=(
            "newey_west_hac_t_interval"
            if confidence_interval is not None
            else "not_applicable"
        ),
        replay_identity=f"predictive-promotion:{_digest(payload)}",
    )


def _statsmodels_hac_ci(
    *,
    differentials: tuple[float, ...],
    maxlags: int,
) -> tuple[float, float]:
    values = np.asarray(differentials, dtype=float)
    design = np.ones((len(values), 1), dtype=float)
    model = OLS(values, design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max(0, int(maxlags))},
    )
    mean = float(model.params[0])
    covariance = np.asarray(model.cov_params(), dtype=float)
    variance = max(0.0, float(covariance[0, 0]))
    standard_error = math.sqrt(variance)
    if math.isclose(standard_error, 0.0, abs_tol=1e-15):
        return (_stable_float(mean), _stable_float(mean))
    degrees_of_freedom = max(len(values) - 1, 1)
    critical_value = float(stats.t.ppf(0.975, degrees_of_freedom))
    return (
        _stable_float(mean - (critical_value * standard_error)),
        _stable_float(mean + (critical_value * standard_error)),
    )


def _hac_maxlags(sample_size: int) -> int:
    if sample_size < 8:
        return 0
    return min(sample_size - 1, max(1, int(math.sqrt(sample_size))))


def _finite_tuple(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        return ()
    return result


def _group_mean(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[float]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(float(row["loss_difference"]))
    return [
        {key: group_key, "mean_loss_difference": _stable_float(fmean(values))}
        for group_key, values in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _rolling_degradation(
    rows: Sequence[Mapping[str, Any]],
    *,
    rolling_window: int,
) -> list[dict[str, Any]]:
    if len(rows) < rolling_window:
        return []
    windows: list[dict[str, Any]] = []
    for index in range(rolling_window - 1, len(rows)):
        window = rows[index - rolling_window + 1 : index + 1]
        latest = float(window[-1]["loss_difference"])
        mean_difference = fmean(float(row["loss_difference"]) for row in window)
        windows.append(
            {
                "end_origin_id": window[-1]["origin_id"],
                "mean_loss_difference": _stable_float(mean_difference),
                "status": "degraded" if latest < 0 else "stable",
            }
        )
    return windows


def _unique(codes: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for code in codes:
        if code:
            seen.setdefault(str(code), None)
    return tuple(seen)


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "PredictivePromotionResult",
    "PrequentialScoreStream",
    "build_prequential_score_stream",
    "evaluate_predictive_promotion",
]
