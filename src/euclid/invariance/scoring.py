from __future__ import annotations

import math
from statistics import fmean
from typing import Mapping, Sequence


def residual_invariance_metrics(
    residuals_by_environment: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    means = {
        environment: _mean_abs(residuals)
        for environment, residuals in residuals_by_environment.items()
        if residuals
    }
    if not means:
        return {
            "max_environment_residual": math.inf,
            "min_environment_residual": math.inf,
            "residual_spread": math.inf,
        }
    values = tuple(means.values())
    return {
        "max_environment_residual": _stable_float(max(values)),
        "min_environment_residual": _stable_float(min(values)),
        "residual_spread": _stable_float(max(values) - min(values)),
    }


def parameter_stability_metrics(
    parameters_by_environment: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    parameter_names = sorted(
        {
            name
            for parameters in parameters_by_environment.values()
            for name in parameters.keys()
        }
    )
    max_drift = 0.0
    for name in parameter_names:
        values = [
            float(parameters[name])
            for parameters in parameters_by_environment.values()
            if name in parameters
        ]
        if len(values) < len(parameters_by_environment):
            return {"max_parameter_drift": math.inf}
        max_drift = max(max_drift, max(values) - min(values))
    return {"max_parameter_drift": _stable_float(max_drift)}


def support_stability_metrics(
    supports_by_environment: Mapping[str, set[str] | frozenset[str]],
) -> dict[str, float]:
    supports = [set(support) for support in supports_by_environment.values()]
    if not supports:
        return {"min_support_jaccard": 0.0}
    min_jaccard = 1.0
    for left_index, left in enumerate(supports):
        for right in supports[left_index + 1 :]:
            union = left | right
            jaccard = 1.0 if not union else len(left & right) / len(union)
            min_jaccard = min(min_jaccard, jaccard)
    return {"min_support_jaccard": _stable_float(min_jaccard)}


def holdout_stability_metrics(
    holdout_losses_by_environment: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, float]:
    if not holdout_losses_by_environment:
        return {"max_holdout_degradation": 0.0}
    max_degradation = 0.0
    for losses in holdout_losses_by_environment.values():
        train = float(losses.get("train", 0.0))
        holdout = float(losses.get("holdout", losses.get("test", 0.0)))
        max_degradation = max(max_degradation, holdout - train)
    return {"max_holdout_degradation": _stable_float(max_degradation)}


def _mean_abs(values: Sequence[float]) -> float:
    return _stable_float(fmean(abs(float(value)) for value in values))


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


__all__ = [
    "holdout_stability_metrics",
    "parameter_stability_metrics",
    "residual_invariance_metrics",
    "support_stability_metrics",
]
