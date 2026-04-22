from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import BinaryOp, Feature, Literal, NaryOp, UnaryOp
from euclid.expr.serialization import expression_canonical_json
from euclid.rewrites.sympy_simplifier import simplify_expression_with_trace


def test_sympy_simplifier_removes_neutral_elements_with_replayable_trace() -> None:
    x = Feature("x", unit="meter")
    expression = NaryOp(
        "mul",
        (
            NaryOp("add", (x, Literal(0, unit="meter"))),
            Literal(1),
        ),
    )

    result = simplify_expression_with_trace(expression)

    assert result.expression == x
    assert result.equivalence_evidence["status"] == "verified"
    assert "additive_identity" in result.applied_rule_ids
    assert "multiplicative_identity" in result.applied_rule_ids
    assert result.replay_identity.startswith("sha256:")


def test_sympy_simplifier_preserves_domain_for_sqrt_square() -> None:
    x = Feature("x", domain="real")
    expression = UnaryOp("sqrt", UnaryOp("pow2", x))

    result = simplify_expression_with_trace(expression)

    assert expression_canonical_json(result.expression) != expression_canonical_json(x)
    assert "sqrt_square_nonnegative" in result.rejected_rule_ids
    assert result.equivalence_evidence["status"] == "verified"


def test_sympy_simplifier_accepts_sqrt_square_under_nonnegative_assumption() -> None:
    x = Feature("x", domain="real")
    expression = UnaryOp("sqrt", UnaryOp("pow2", x))

    result = simplify_expression_with_trace(
        expression,
        assumptions={"x": {"domain": "nonnegative_real"}},
    )

    assert result.expression == x
    assert "sqrt_square_nonnegative" in result.applied_rule_ids


def test_sympy_simplifier_rejects_dimensioned_log_exp_rewrite() -> None:
    temperature = Feature("temperature", unit="kelvin")
    expression = UnaryOp("log", UnaryOp("exp", temperature))

    with pytest.raises(ContractValidationError) as excinfo:
        simplify_expression_with_trace(expression)

    assert excinfo.value.code == "unsafe_rewrite_rejected"
    assert excinfo.value.details["reason_code"] == "dimensionless_required"


def test_sympy_simplifier_rejects_unit_changing_neutral_element() -> None:
    expression = NaryOp(
        "add",
        (Feature("distance", unit="meter"), Literal(0, unit="second")),
    )

    with pytest.raises(ContractValidationError) as excinfo:
        simplify_expression_with_trace(expression)

    assert excinfo.value.code == "unsafe_rewrite_rejected"
    assert excinfo.value.details["reason_code"] == "unit_mismatch"


def test_sympy_simplifier_keeps_rational_assumption_evidence() -> None:
    x = Feature("x")
    y = Feature("y")
    expression = BinaryOp("div", BinaryOp("mul", x, y), y)

    result = simplify_expression_with_trace(
        expression,
        assumptions={"y": {"nonzero": True}},
    )

    assert result.expression == x
    assert "divide_common_factor" in result.applied_rule_ids
    assert result.equivalence_evidence["assumptions"]["y"]["nonzero"] is True
