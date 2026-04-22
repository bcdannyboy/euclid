from __future__ import annotations

import pytest

from euclid.fit.parameterization import ParameterBounds, ParameterDeclaration
from euclid.fit.scipy_optimizers import fit_least_squares, fit_minimize


def test_scipy_least_squares_recovers_linear_parameters_with_replay_metadata() -> None:
    x_values = (0.0, 1.0, 2.0, 3.0)
    y_values = tuple(1.0 + 2.0 * x for x in x_values)

    result = fit_least_squares(
        parameter_declarations=(
            ParameterDeclaration("intercept", initial_value=0.0),
            ParameterDeclaration("slope", initial_value=0.0),
        ),
        residual_fn=lambda params: tuple(
            params["intercept"] + params["slope"] * x - y
            for x, y in zip(x_values, y_values, strict=True)
        ),
        objective_id="squared_error",
        seed=17,
    )

    assert result.converged is True
    assert result.parameter_estimates["intercept"] == pytest.approx(1.0)
    assert result.parameter_estimates["slope"] == pytest.approx(2.0)
    assert result.diagnostics["optimizer_backend"] == "scipy.optimize.least_squares"
    assert result.replay_metadata["seed"] == 17
    assert "scipy" in result.replay_metadata["library_versions"]


def test_scipy_minimize_respects_bounds_and_records_bound_hit() -> None:
    result = fit_minimize(
        parameter_declarations=(
            ParameterDeclaration(
                "x",
                initial_value=0.0,
                bounds=ParameterBounds(lower=0.0, upper=1.0),
            ),
        ),
        objective_fn=lambda params: (params["x"] - 5.0) ** 2,
        objective_id="squared_error",
        seed=3,
    )

    assert result.converged is True
    assert result.parameter_estimates["x"] == pytest.approx(1.0)
    assert result.diagnostics["bound_hits"] == ["x:upper"]


def test_optimizer_non_convergence_is_typed_not_silent() -> None:
    result = fit_least_squares(
        parameter_declarations=(ParameterDeclaration("x", initial_value=0.0),),
        residual_fn=lambda params: (params["x"] - 100.0,),
        objective_id="squared_error",
        seed=11,
        max_nfev=1,
    )

    assert result.converged is False
    assert result.failure_reasons
    assert "optimizer_nonconvergence" in result.failure_reasons
