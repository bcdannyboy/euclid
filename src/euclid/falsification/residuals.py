from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

from euclid.falsification._identity import (
    finite_tuple,
    replay_identity,
    stable_float,
    unique_codes,
)


@dataclass(frozen=True)
class ResidualDiagnosticResult:
    candidate_id: str
    status: str
    reason_codes: tuple[str, ...]
    claim_effect: str
    residual_count: int
    mean_residual: float | None
    lag1_autocorrelation: float | None
    sign_run_count: int | None
    max_abs_standardized_residual: float | None
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "residual_diagnostics@1.0.0",
            "candidate_id": self.candidate_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "claim_effect": self.claim_effect,
            "residual_count": self.residual_count,
            "mean_residual": self.mean_residual,
            "lag1_autocorrelation": self.lag1_autocorrelation,
            "sign_run_count": self.sign_run_count,
            "max_abs_standardized_residual": self.max_abs_standardized_residual,
            "replay_identity": self.replay_identity,
        }


def evaluate_residual_diagnostics(
    *,
    candidate_id: str,
    residuals: Sequence[float],
    min_residual_count: int = 4,
    autocorrelation_threshold: float = 0.5,
    max_abs_mean_standardized: float = 0.5,
    max_abs_standardized_residual: float = 4.0,
) -> ResidualDiagnosticResult:
    residual_tuple = finite_tuple(residuals)
    if residual_tuple is None:
        return _result(
            candidate_id=candidate_id,
            status="abstained",
            reason_codes=("nonfinite_residual",),
            residuals=(),
            claim_effect="block_claim",
        )
    if len(residual_tuple) < int(min_residual_count):
        return _result(
            candidate_id=candidate_id,
            status="abstained",
            reason_codes=("insufficient_residual_support",),
            residuals=residual_tuple,
            claim_effect="block_claim",
        )

    mean = fmean(residual_tuple)
    scale = _residual_scale(residual_tuple, mean)
    standardized_mean = abs(mean) / scale if scale > 0 else 0.0
    max_standardized = max(abs(value - mean) / scale for value in residual_tuple)
    lag1 = _lag1_autocorrelation(residual_tuple)
    runs = _sign_run_count(residual_tuple)

    reasons: list[str] = []
    if lag1 is not None and abs(lag1) >= float(autocorrelation_threshold):
        reasons.append("structured_residuals")
    if standardized_mean > float(max_abs_mean_standardized):
        reasons.append("biased_residuals")
    if max_standardized > float(max_abs_standardized_residual):
        reasons.append("residual_outlier")

    return _result(
        candidate_id=candidate_id,
        status="failed" if reasons else "passed",
        reason_codes=unique_codes(reasons),
        residuals=residual_tuple,
        mean_residual=stable_float(mean),
        lag1_autocorrelation=None if lag1 is None else stable_float(lag1),
        sign_run_count=runs,
        max_abs_standardized_residual=stable_float(max_standardized),
        claim_effect="downgrade_predictive_claim" if reasons else "allow_claim",
    )


def _result(
    *,
    candidate_id: str,
    status: str,
    reason_codes: Sequence[str],
    residuals: Sequence[float],
    claim_effect: str,
    mean_residual: float | None = None,
    lag1_autocorrelation: float | None = None,
    sign_run_count: int | None = None,
    max_abs_standardized_residual: float | None = None,
) -> ResidualDiagnosticResult:
    payload: Mapping[str, Any] = {
        "candidate_id": candidate_id,
        "claim_effect": claim_effect,
        "lag1_autocorrelation": lag1_autocorrelation,
        "max_abs_standardized_residual": max_abs_standardized_residual,
        "mean_residual": mean_residual,
        "reason_codes": list(reason_codes),
        "residuals": list(residuals),
        "sign_run_count": sign_run_count,
        "status": status,
    }
    return ResidualDiagnosticResult(
        candidate_id=str(candidate_id),
        status=status,
        reason_codes=unique_codes(reason_codes),
        claim_effect=claim_effect,
        residual_count=len(tuple(residuals)),
        mean_residual=mean_residual,
        lag1_autocorrelation=lag1_autocorrelation,
        sign_run_count=sign_run_count,
        max_abs_standardized_residual=max_abs_standardized_residual,
        replay_identity=replay_identity("residual-diagnostics", payload),
    )


def _residual_scale(residuals: Sequence[float], mean: float) -> float:
    if len(residuals) < 2:
        return 0.0
    variance = fmean((value - mean) ** 2 for value in residuals)
    return max(math.sqrt(max(variance, 0.0)), 1e-12)


def _lag1_autocorrelation(residuals: Sequence[float]) -> float | None:
    if len(residuals) < 3:
        return None
    mean = fmean(residuals)
    numerator = sum(
        (residuals[index] - mean) * (residuals[index - 1] - mean)
        for index in range(1, len(residuals))
    )
    denominator = sum((value - mean) ** 2 for value in residuals)
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _sign_run_count(residuals: Sequence[float]) -> int:
    signs = [1 if value >= 0 else -1 for value in residuals if value != 0]
    if not signs:
        return 1
    return 1 + sum(
        1 for index in range(1, len(signs)) if signs[index] != signs[index - 1]
    )


__all__ = ["ResidualDiagnosticResult", "evaluate_residual_diagnostics"]
