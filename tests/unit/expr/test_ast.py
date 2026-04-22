from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import (
    BinaryOp,
    Conditional,
    Derivative,
    DistributionParameter,
    Feature,
    FunctionCall,
    Integral,
    Lag,
    Literal,
    NaryOp,
    NoiseTerm,
    Parameter,
    Piecewise,
    State,
    UnaryOp,
    expression_hash,
)
from euclid.expr.serialization import expression_from_dict, expression_to_dict


def test_commutative_expression_nodes_have_stable_canonical_identity() -> None:
    left = NaryOp(
        "add",
        (
            Feature("x", domain="real", unit="meter"),
            Literal(1.0, unit="meter"),
            Parameter("alpha", domain="real", unit="meter"),
        ),
    )
    right = NaryOp(
        "add",
        (
            Parameter("alpha", domain="real", unit="meter"),
            Literal(1.0, unit="meter"),
            Feature("x", domain="real", unit="meter"),
        ),
    )

    assert expression_to_dict(left) == expression_to_dict(right)
    assert expression_hash(left) == expression_hash(right)
    assert [child.node_kind for child in left.children] == [
        child.node_kind for child in right.children
    ]


def test_expression_nodes_are_immutable_and_reject_nonfinite_literals() -> None:
    feature = Feature("x")

    with pytest.raises(FrozenInstanceError):
        feature.name = "y"  # type: ignore[misc]

    with pytest.raises(ContractValidationError) as exc_info:
        Literal(float("nan"))

    assert exc_info.value.code == "invalid_expression_literal"


def test_all_planned_node_types_round_trip_through_stable_serialization() -> None:
    expression = Piecewise(
        cases=(
            (
                BinaryOp("gt", Feature("x"), Literal(0.0)),
                NaryOp(
                    "add",
                    (
                        UnaryOp("log", Feature("x", domain="positive_real")),
                        Lag(Feature("y"), periods=1),
                        Derivative(Feature("x"), variable="t", order=1),
                    ),
                ),
            ),
        ),
        default=FunctionCall(
            "custom_kernel",
            (
                Integral(Feature("z"), variable="t"),
                State("level", unit="meter"),
                NoiseTerm("process", distribution="gaussian"),
                DistributionParameter("scale", domain="positive_real"),
            ),
        ),
    )

    payload = expression_to_dict(expression)
    restored = expression_from_dict(payload)

    assert expression_to_dict(restored) == payload
    assert expression_hash(restored) == expression_hash(expression)


def test_invalid_expression_operator_fails_closed_before_serialization() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        UnaryOp("not_registered", Feature("x"))

    assert exc_info.value.code == "unsupported_expression_operator"
    assert exc_info.value.field_path == "operator"

