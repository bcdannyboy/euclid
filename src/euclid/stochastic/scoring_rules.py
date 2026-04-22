from __future__ import annotations

import math
from typing import Any

from scipy import stats

from euclid.contracts.errors import ContractValidationError
from euclid.stochastic.observation_models import BoundObservationModel


def log_score(model: Any, observed: float) -> float:
    try:
        score = -float(model.log_likelihood(float(observed)))
    except ContractValidationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise ContractValidationError(
            code="nonfinite_likelihood",
            message="log score requires a finite likelihood",
            field_path="model",
        ) from exc
    if not math.isfinite(score):
        raise ContractValidationError(
            code="nonfinite_likelihood",
            message="log score requires a finite likelihood",
            field_path="model",
        )
    return _stable_float(score)


def continuous_ranked_probability_score(
    model: BoundObservationModel,
    observed: float,
) -> float:
    if model.family_id != "gaussian":
        raise ContractValidationError(
            code="unsupported_crps_family",
            message="closed-form CRPS is currently admitted for Gaussian models",
            field_path="model.family_id",
            details={"family_id": model.family_id},
        )
    model._validate_parameters()
    location = float(model.parameters["location"])
    scale = float(model.parameters["scale"])
    z = (float(observed) - location) / scale
    score = scale * (
        z * (2.0 * float(stats.norm.cdf(z)) - 1.0)
        + (2.0 * float(stats.norm.pdf(z)))
        - (1.0 / math.sqrt(math.pi))
    )
    return _finite_score(score)


def interval_score(
    *,
    nominal_coverage: float,
    lower_bound: float,
    upper_bound: float,
    observed: float,
) -> float:
    coverage = float(nominal_coverage)
    lower = float(lower_bound)
    upper = float(upper_bound)
    value = float(observed)
    if not (0.0 < coverage < 1.0) or lower > upper:
        raise ContractValidationError(
            code="invalid_interval_forecast",
            message="interval score requires 0 < coverage < 1 and lower <= upper",
            field_path="interval",
        )
    alpha = 1.0 - coverage
    score = upper - lower
    if value < lower:
        score += (2.0 / alpha) * (lower - value)
    elif value > upper:
        score += (2.0 / alpha) * (value - upper)
    return _finite_score(score)


def pinball_loss(*, level: float, quantile: float, observed: float) -> float:
    alpha = float(level)
    if not 0.0 < alpha < 1.0:
        raise ContractValidationError(
            code="invalid_quantile_level",
            message="pinball loss requires 0 < level < 1",
            field_path="level",
        )
    residual = float(observed) - float(quantile)
    score = alpha * residual if residual >= 0 else (alpha - 1.0) * residual
    return _finite_score(score)


def brier_score(*, probability: float, realized_event: bool) -> float:
    probability_value = float(probability)
    if not math.isfinite(probability_value) or not 0.0 <= probability_value <= 1.0:
        raise ContractValidationError(
            code="invalid_event_probability",
            message="Brier score requires a probability in [0, 1]",
            field_path="probability",
        )
    score = (probability_value - (1.0 if realized_event else 0.0)) ** 2
    return _finite_score(score)


def _finite_score(value: float) -> float:
    if not math.isfinite(value):
        raise ContractValidationError(
            code="nonfinite_score_value",
            message="proper scoring rule produced a nonfinite value",
            field_path="score",
        )
    return _stable_float(value)


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


__all__ = [
    "brier_score",
    "continuous_ranked_probability_score",
    "interval_score",
    "log_score",
    "pinball_loss",
]
