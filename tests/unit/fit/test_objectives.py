from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.fit.objectives import get_objective, regularization_penalty
from euclid.fit.parameterization import ParameterDeclaration, ParameterPenalty


def test_squared_absolute_and_huber_objectives_expose_residual_and_scalar_loss() -> None:
    observed = (1.0, 2.0, 4.0)
    predicted = (1.5, 1.0, 6.0)

    squared = get_objective("squared_error")
    absolute = get_objective("absolute_error")
    huber = get_objective("huber_loss")

    assert squared.residuals(observed, predicted) == pytest.approx((0.5, -1.0, 2.0))
    assert squared.scalar_loss(observed, predicted) == pytest.approx(5.25)
    assert absolute.scalar_loss(observed, predicted) == pytest.approx(3.5)
    assert huber.scalar_loss(observed, predicted, delta=1.0) == pytest.approx(2.625)


def test_gaussian_negative_log_likelihood_and_regularization_are_finite() -> None:
    gaussian = get_objective("gaussian_nll")

    assert gaussian.scalar_loss((1.0, 2.0), (1.0, 2.5), scale=1.0) > 0.0
    assert regularization_penalty(
        {"alpha": 3.0},
        (
            ParameterDeclaration(
                "alpha",
                initial_value=0.0,
                penalty=ParameterPenalty(kind="l2", weight=0.5, center=1.0),
            ),
        ),
    ) == pytest.approx(2.0)


def test_objectives_fail_closed_for_unknown_objective_or_bad_scale() -> None:
    with pytest.raises(ContractValidationError, match="unknown"):
        get_objective("not_registered")

    with pytest.raises(ContractValidationError, match="scale"):
        get_objective("gaussian_nll").scalar_loss((1.0,), (1.0,), scale=0.0)
