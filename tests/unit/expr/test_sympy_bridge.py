from __future__ import annotations

import pytest
import sympy as sp

from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import BinaryOp, Feature, Literal, NaryOp, UnaryOp
from euclid.expr.serialization import expression_to_dict
from euclid.expr.sympy_bridge import (
    are_equivalent,
    derivative,
    from_sympy,
    simplify_expression,
    to_sympy,
)


def test_sympy_simplification_round_trip_preserves_canonical_equivalence() -> None:
    expression = NaryOp(
        "add",
        (
            Feature("x"),
            Literal(0.0),
            BinaryOp("mul", Literal(2.0), Feature("y")),
        ),
    )

    simplified = simplify_expression(expression)
    restored = from_sympy(to_sympy(simplified))

    assert are_equivalent(simplified, NaryOp("add", (Feature("x"), BinaryOp("mul", Literal(2.0), Feature("y")))))
    assert expression_to_dict(restored) == expression_to_dict(simplified)


def test_sympy_derivative_returns_expression_ir() -> None:
    expression = BinaryOp("pow", Feature("x"), Literal(2.0))

    result = derivative(expression, variable="x")

    assert are_equivalent(result, BinaryOp("mul", Literal(2.0), Feature("x")))


def test_domain_sensitive_expressions_do_not_collapse_without_assumptions() -> None:
    expression = UnaryOp("sqrt", BinaryOp("pow", Feature("x"), Literal(2.0)))

    without_assumptions = simplify_expression(expression)
    with_positive = simplify_expression(
        expression,
        assumptions={"x": {"domain": "positive_real"}},
    )

    assert not are_equivalent(without_assumptions, Feature("x"))
    assert are_equivalent(with_positive, Feature("x"))


def test_unsupported_sympy_function_is_rejected_with_typed_diagnostic() -> None:
    x = sp.Symbol("x")

    with pytest.raises(ContractValidationError) as exc_info:
        from_sympy(sp.gamma(x))

    assert exc_info.value.code == "unsupported_sympy_expression"

