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
class ParameterStabilityResult:
    candidate_id: str
    status: str
    reason_codes: tuple[str, ...]
    claim_effect: str
    parameter_relative_ranges: Mapping[str, float]
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "parameter_stability@1.0.0",
            "candidate_id": self.candidate_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "claim_effect": self.claim_effect,
            "parameter_relative_ranges": dict(self.parameter_relative_ranges),
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class SurrogateResidualTestResult:
    candidate_id: str
    status: str
    reason_codes: tuple[str, ...]
    claim_effect: str
    observed_statistic: float
    surrogate_statistics: tuple[float, ...]
    monte_carlo_p_value: float
    max_p_value: float
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "surrogate_residual_test@1.0.0",
            "candidate_id": self.candidate_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "claim_effect": self.claim_effect,
            "observed_statistic": self.observed_statistic,
            "surrogate_statistics": list(self.surrogate_statistics),
            "monte_carlo_p_value": self.monte_carlo_p_value,
            "max_p_value": self.max_p_value,
            "replay_identity": self.replay_identity,
        }


def evaluate_parameter_stability(
    *,
    candidate_id: str,
    window_parameters: Sequence[Mapping[str, float]],
    max_relative_range: float = 0.5,
) -> ParameterStabilityResult:
    if len(window_parameters) < 2:
        return _parameter_result(
            candidate_id=candidate_id,
            status="abstained",
            reason_codes=("insufficient_parameter_windows",),
            relative_ranges={},
            claim_effect="block_claim",
        )
    parameter_names = sorted(
        set.intersection(
            *(set(str(key) for key in window) for window in window_parameters)
        )
    )
    relative_ranges: dict[str, float] = {}
    for parameter in parameter_names:
        values = finite_tuple([float(window[parameter]) for window in window_parameters])
        if values is None:
            return _parameter_result(
                candidate_id=candidate_id,
                status="abstained",
                reason_codes=("nonfinite_parameter_estimate",),
                relative_ranges=relative_ranges,
                claim_effect="block_claim",
            )
        center = max(abs(fmean(values)), 1e-12)
        relative_ranges[parameter] = stable_float((max(values) - min(values)) / center)
    unstable = [
        parameter
        for parameter, relative_range in relative_ranges.items()
        if relative_range > float(max_relative_range)
    ]
    return _parameter_result(
        candidate_id=candidate_id,
        status="failed" if unstable else "passed",
        reason_codes=("parameter_instability",) if unstable else (),
        relative_ranges=relative_ranges,
        claim_effect="downgrade_predictive_claim" if unstable else "allow_claim",
    )


def evaluate_surrogate_residual_test(
    *,
    candidate_id: str,
    observed_statistic: float,
    surrogate_statistics: Sequence[float],
    max_p_value: float,
) -> SurrogateResidualTestResult:
    observed = float(observed_statistic)
    surrogates = finite_tuple(surrogate_statistics)
    threshold = float(max_p_value)
    if not math.isfinite(observed) or surrogates is None or not surrogates:
        return _surrogate_result(
            candidate_id=candidate_id,
            status="abstained",
            reason_codes=("invalid_surrogate_residual_test",),
            observed_statistic=0.0,
            surrogate_statistics=(),
            monte_carlo_p_value=1.0,
            max_p_value=threshold,
            claim_effect="block_claim",
        )
    exceedances = sum(1 for value in surrogates if value >= observed)
    p_value = (exceedances + 1.0) / (len(surrogates) + 1.0)
    failed = p_value <= threshold
    return _surrogate_result(
        candidate_id=candidate_id,
        status="failed" if failed else "passed",
        reason_codes=("structured_residuals_vs_surrogate",) if failed else (),
        observed_statistic=stable_float(observed),
        surrogate_statistics=tuple(stable_float(value) for value in surrogates),
        monte_carlo_p_value=stable_float(p_value),
        max_p_value=stable_float(threshold),
        claim_effect="downgrade_predictive_claim" if failed else "allow_claim",
    )


def _parameter_result(
    *,
    candidate_id: str,
    status: str,
    reason_codes: Sequence[str],
    relative_ranges: Mapping[str, float],
    claim_effect: str,
) -> ParameterStabilityResult:
    payload = {
        "candidate_id": candidate_id,
        "claim_effect": claim_effect,
        "parameter_relative_ranges": dict(relative_ranges),
        "reason_codes": list(reason_codes),
        "status": status,
    }
    return ParameterStabilityResult(
        candidate_id=str(candidate_id),
        status=status,
        reason_codes=unique_codes(reason_codes),
        claim_effect=claim_effect,
        parameter_relative_ranges=dict(relative_ranges),
        replay_identity=replay_identity("parameter-stability", payload),
    )


def _surrogate_result(
    *,
    candidate_id: str,
    status: str,
    reason_codes: Sequence[str],
    observed_statistic: float,
    surrogate_statistics: Sequence[float],
    monte_carlo_p_value: float,
    max_p_value: float,
    claim_effect: str,
) -> SurrogateResidualTestResult:
    payload = {
        "candidate_id": candidate_id,
        "claim_effect": claim_effect,
        "max_p_value": max_p_value,
        "monte_carlo_p_value": monte_carlo_p_value,
        "observed_statistic": observed_statistic,
        "reason_codes": list(reason_codes),
        "status": status,
        "surrogate_statistics": list(surrogate_statistics),
    }
    return SurrogateResidualTestResult(
        candidate_id=str(candidate_id),
        status=status,
        reason_codes=unique_codes(reason_codes),
        claim_effect=claim_effect,
        observed_statistic=observed_statistic,
        surrogate_statistics=tuple(surrogate_statistics),
        monte_carlo_p_value=monte_carlo_p_value,
        max_p_value=max_p_value,
        replay_identity=replay_identity("surrogate-residual-test", payload),
    )


__all__ = [
    "ParameterStabilityResult",
    "SurrogateResidualTestResult",
    "evaluate_parameter_stability",
    "evaluate_surrogate_residual_test",
]
