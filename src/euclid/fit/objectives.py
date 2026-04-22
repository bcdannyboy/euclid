from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from euclid.contracts.errors import ContractValidationError
from euclid.fit.parameterization import ParameterDeclaration


LossFn = Callable[..., float]


@dataclass(frozen=True)
class ObjectiveDefinition:
    objective_id: str
    scipy_loss: str
    _loss_fn: LossFn

    def residuals(
        self,
        observed: tuple[float, ...],
        predicted: tuple[float, ...],
    ) -> tuple[float, ...]:
        _require_aligned(observed, predicted)
        return tuple(float(predicted_value) - float(observed_value) for observed_value, predicted_value in zip(observed, predicted, strict=True))

    def scalar_loss(
        self,
        observed: tuple[float, ...],
        predicted: tuple[float, ...],
        **kwargs,
    ) -> float:
        _require_aligned(observed, predicted)
        return float(self._loss_fn(observed, predicted, **kwargs))


def get_objective(objective_id: str) -> ObjectiveDefinition:
    try:
        return _OBJECTIVES[objective_id]
    except KeyError as exc:
        raise ContractValidationError(
            code="unknown_fit_objective",
            message=f"unknown fit objective {objective_id!r}",
            field_path="objective_id",
            details={"objective_id": objective_id},
        ) from exc


def regularization_penalty(
    parameter_values: dict[str, float],
    declarations: tuple[ParameterDeclaration, ...],
) -> float:
    total = 0.0
    for declaration in declarations:
        if declaration.penalty is None:
            continue
        value = float(parameter_values[declaration.name])
        delta = value - declaration.penalty.center
        if declaration.penalty.kind == "l1":
            total += declaration.penalty.weight * abs(delta)
        elif declaration.penalty.kind == "l2":
            total += declaration.penalty.weight * (delta**2)
        else:
            raise ContractValidationError(
                code="unknown_fit_penalty",
                message=f"unknown parameter penalty {declaration.penalty.kind!r}",
                field_path=f"parameter[{declaration.name}].penalty.kind",
            )
    return float(total)


def _squared_loss(observed: tuple[float, ...], predicted: tuple[float, ...]) -> float:
    return sum((pred - obs) ** 2 for obs, pred in zip(observed, predicted, strict=True))


def _absolute_loss(observed: tuple[float, ...], predicted: tuple[float, ...]) -> float:
    return sum(abs(pred - obs) for obs, pred in zip(observed, predicted, strict=True))


def _huber_loss(
    observed: tuple[float, ...],
    predicted: tuple[float, ...],
    *,
    delta: float = 1.0,
) -> float:
    if delta <= 0.0 or not math.isfinite(delta):
        raise ContractValidationError(
            code="invalid_fit_objective",
            message="Huber delta must be positive and finite",
            field_path="delta",
        )
    total = 0.0
    for obs, pred in zip(observed, predicted, strict=True):
        residual = abs(pred - obs)
        total += 0.5 * residual * residual if residual <= delta else delta * residual
    return float(total)


def _gaussian_nll(
    observed: tuple[float, ...],
    predicted: tuple[float, ...],
    *,
    scale: float = 1.0,
) -> float:
    if scale <= 0.0 or not math.isfinite(scale):
        raise ContractValidationError(
            code="invalid_fit_objective",
            message="Gaussian negative log likelihood scale must be positive",
            field_path="scale",
        )
    log_normalizer = math.log(scale * math.sqrt(2.0 * math.pi))
    return float(
        sum(
            log_normalizer + (((obs - pred) / scale) ** 2) / 2.0
            for obs, pred in zip(observed, predicted, strict=True)
        )
    )


def _require_aligned(
    observed: tuple[float, ...],
    predicted: tuple[float, ...],
) -> None:
    if len(observed) != len(predicted):
        raise ContractValidationError(
            code="invalid_fit_objective_inputs",
            message="observed and predicted values must have identical lengths",
            field_path="predicted",
            details={"observed": len(observed), "predicted": len(predicted)},
        )


_OBJECTIVES = {
    "squared_error": ObjectiveDefinition("squared_error", "linear", _squared_loss),
    "absolute_error": ObjectiveDefinition("absolute_error", "soft_l1", _absolute_loss),
    "huber_loss": ObjectiveDefinition("huber_loss", "huber", _huber_loss),
    "gaussian_nll": ObjectiveDefinition("gaussian_nll", "linear", _gaussian_nll),
}


__all__ = ["ObjectiveDefinition", "get_objective", "regularization_penalty"]
