from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.fit.parameterization import (
    ParameterBounds,
    ParameterDeclaration,
    ParameterPenalty,
    ParameterPrior,
    ParameterVector,
)


def test_parameter_declaration_serializes_constraints_and_scope_flags() -> None:
    declaration = ParameterDeclaration(
        name="growth_rate",
        initial_value=0.25,
        bounds=ParameterBounds(lower=0.0, upper=2.0),
        transform="positive_log",
        fixed=False,
        shared=True,
        entity_local=True,
        regime_local=False,
        prior=ParameterPrior(kind="normal", location=0.0, scale=1.0),
        penalty=ParameterPenalty(kind="l2", weight=0.125),
        unit="1 / day",
        description_length_bits=3.5,
    )

    assert declaration.as_dict() == {
        "name": "growth_rate",
        "initial_value": 0.25,
        "bounds": {"lower": 0.0, "upper": 2.0},
        "transform": "positive_log",
        "fixed": False,
        "shared": True,
        "entity_local": True,
        "regime_local": False,
        "prior": {"kind": "normal", "location": 0.0, "scale": 1.0},
        "penalty": {"kind": "l2", "weight": 0.125, "center": 0.0},
        "unit": "1 / day",
        "description_length_bits": 3.5,
    }


def test_parameter_vector_binds_only_free_values_and_preserves_fixed_parameters() -> None:
    vector = ParameterVector(
        declarations=(
            ParameterDeclaration("intercept", initial_value=10.0, fixed=True),
            ParameterDeclaration("slope", initial_value=0.0),
            ParameterDeclaration(
                "scale",
                initial_value=1.0,
                bounds=ParameterBounds(lower=0.0, upper=None),
                transform="positive_log",
            ),
        )
    )

    assert vector.free_names == ("slope", "scale")
    assert vector.initial_free_vector() == pytest.approx((0.0, 0.0))
    assert vector.bind_free_values((2.5, 0.0)) == {
        "intercept": 10.0,
        "slope": 2.5,
        "scale": 1.0,
    }


def test_parameterization_fails_closed_on_duplicate_or_invalid_bounds() -> None:
    with pytest.raises(ContractValidationError, match="duplicate"):
        ParameterVector(
            declarations=(
                ParameterDeclaration("alpha", initial_value=0.0),
                ParameterDeclaration("alpha", initial_value=1.0),
            )
        )

    with pytest.raises(ContractValidationError, match="lower"):
        ParameterDeclaration(
            "bad",
            initial_value=0.0,
            bounds=ParameterBounds(lower=1.0, upper=0.0),
        )
