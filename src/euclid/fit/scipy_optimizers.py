from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize

from euclid.fit.parameterization import ParameterDeclaration, ParameterVector


@dataclass(frozen=True)
class OptimizerResult:
    converged: bool
    parameter_estimates: dict[str, float]
    loss: float
    diagnostics: dict[str, Any]
    failure_reasons: tuple[str, ...]
    replay_metadata: dict[str, Any]


def fit_least_squares(
    *,
    parameter_declarations: Sequence[ParameterDeclaration],
    residual_fn: Callable[[Mapping[str, float]], Sequence[float]],
    objective_id: str,
    seed: int = 0,
    max_nfev: int | None = None,
    scipy_loss: str = "linear",
) -> OptimizerResult:
    vector = ParameterVector(tuple(parameter_declarations))
    bounds = vector.free_bounds()
    initial = np.asarray(vector.initial_free_vector(), dtype=float)
    failure_reasons: list[str] = []

    def wrapped(external_values):
        params = vector.bind_free_values(tuple(float(value) for value in external_values))
        return np.asarray(tuple(float(value) for value in residual_fn(params)), dtype=float)

    try:
        result = least_squares(
            wrapped,
            initial,
            bounds=bounds,
            loss=scipy_loss,
            max_nfev=max_nfev,
        )
        converged = bool(result.success)
        if not converged:
            failure_reasons.append("optimizer_nonconvergence")
        estimates = vector.bind_free_values(tuple(float(value) for value in result.x))
        residuals = np.asarray(result.fun, dtype=float)
        loss = float(np.dot(residuals, residuals))
        diagnostics = {
            "optimizer_backend": "scipy.optimize.least_squares",
            "objective_id": objective_id,
            "status": int(result.status),
            "message": str(result.message),
            "iteration_count": int(result.nfev),
            "function_evaluations": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(getattr(result, "optimality", 0.0)),
            "initial_parameters": _initial_parameters(vector),
            "final_parameters": dict(estimates),
            "bounds": _bounds_payload(vector),
            "bound_hits": _bound_hits(estimates, vector),
            "jacobian_shape": list(result.jac.shape),
        }
    except Exception as exc:  # pragma: no cover - exact SciPy exception shape varies
        failure_reasons.append("optimizer_numerical_failure")
        estimates = {
            declaration.name: declaration.initial_value
            for declaration in vector.declarations
        }
        converged = False
        loss = float("inf")
        diagnostics = {
            "optimizer_backend": "scipy.optimize.least_squares",
            "objective_id": objective_id,
            "status": "exception",
            "message": type(exc).__name__,
            "initial_parameters": _initial_parameters(vector),
            "final_parameters": dict(estimates),
            "bounds": _bounds_payload(vector),
            "bound_hits": [],
        }

    return OptimizerResult(
        converged=converged,
        parameter_estimates=dict(estimates),
        loss=loss,
        diagnostics=diagnostics,
        failure_reasons=tuple(failure_reasons),
        replay_metadata=_replay_metadata(seed=seed, objective_id=objective_id),
    )


def fit_minimize(
    *,
    parameter_declarations: Sequence[ParameterDeclaration],
    objective_fn: Callable[[Mapping[str, float]], float],
    objective_id: str,
    seed: int = 0,
    maxiter: int | None = None,
) -> OptimizerResult:
    vector = ParameterVector(tuple(parameter_declarations))
    bounds = list(zip(*vector.free_bounds(), strict=True))
    initial = np.asarray(vector.initial_free_vector(), dtype=float)
    failure_reasons: list[str] = []

    def wrapped(external_values):
        params = vector.bind_free_values(tuple(float(value) for value in external_values))
        return float(objective_fn(params))

    try:
        result = minimize(
            wrapped,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={} if maxiter is None else {"maxiter": int(maxiter)},
        )
        converged = bool(result.success)
        if not converged:
            failure_reasons.append("optimizer_nonconvergence")
        estimates = vector.bind_free_values(tuple(float(value) for value in result.x))
        loss = float(result.fun)
        diagnostics = {
            "optimizer_backend": "scipy.optimize.minimize",
            "objective_id": objective_id,
            "status": int(result.status),
            "message": str(result.message),
            "iteration_count": int(getattr(result, "nit", 0)),
            "function_evaluations": int(getattr(result, "nfev", 0)),
            "initial_parameters": _initial_parameters(vector),
            "final_parameters": dict(estimates),
            "bounds": _bounds_payload(vector),
            "bound_hits": _bound_hits(estimates, vector),
        }
    except Exception as exc:  # pragma: no cover - exact SciPy exception shape varies
        failure_reasons.append("optimizer_numerical_failure")
        estimates = {
            declaration.name: declaration.initial_value
            for declaration in vector.declarations
        }
        converged = False
        loss = float("inf")
        diagnostics = {
            "optimizer_backend": "scipy.optimize.minimize",
            "objective_id": objective_id,
            "status": "exception",
            "message": type(exc).__name__,
            "initial_parameters": _initial_parameters(vector),
            "final_parameters": dict(estimates),
            "bounds": _bounds_payload(vector),
            "bound_hits": [],
        }

    return OptimizerResult(
        converged=converged,
        parameter_estimates=dict(estimates),
        loss=loss,
        diagnostics=diagnostics,
        failure_reasons=tuple(failure_reasons),
        replay_metadata=_replay_metadata(seed=seed, objective_id=objective_id),
    )


def _initial_parameters(vector: ParameterVector) -> dict[str, float]:
    return {declaration.name: declaration.initial_value for declaration in vector.declarations}


def _bounds_payload(vector: ParameterVector) -> dict[str, dict[str, float | None]]:
    return {
        declaration.name: (
            {"lower": None, "upper": None}
            if declaration.bounds is None
            else declaration.bounds.as_dict()
        )
        for declaration in vector.declarations
    }


def _bound_hits(
    estimates: Mapping[str, float],
    vector: ParameterVector,
) -> list[str]:
    hits: list[str] = []
    for declaration in vector.declarations:
        bounds = declaration.bounds
        if bounds is None:
            continue
        value = float(estimates[declaration.name])
        if bounds.lower is not None and abs(value - bounds.lower) <= 1e-8:
            hits.append(f"{declaration.name}:lower")
        if bounds.upper is not None and abs(value - bounds.upper) <= 1e-8:
            hits.append(f"{declaration.name}:upper")
    return hits


def _replay_metadata(*, seed: int, objective_id: str) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "objective_id": objective_id,
        "library_versions": {
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
    }


__all__ = ["OptimizerResult", "fit_least_squares", "fit_minimize"]
