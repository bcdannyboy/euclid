from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import BinaryOp, Feature, Literal, UnaryOp
from euclid.expr.evaluators import evaluate_expression
from euclid.expr.units import infer_expression_unit, validate_expression_units


def test_pint_unit_validation_allows_compatible_arithmetic() -> None:
    expression = BinaryOp("add", Feature("distance"), Literal(2.0, unit="meter"))

    result = validate_expression_units(
        expression,
        variable_units={"distance": "meter"},
    )

    assert result.status == "passed"
    assert str(infer_expression_unit(expression, variable_units={"distance": "meter"})) == "meter"


def test_pint_unit_validation_rejects_incompatible_binary_operations() -> None:
    expression = BinaryOp("add", Feature("distance"), Feature("elapsed"))

    with pytest.raises(ContractValidationError) as exc_info:
        validate_expression_units(
            expression,
            variable_units={"distance": "meter", "elapsed": "second"},
        )

    assert exc_info.value.code == "expression_unit_incompatible"


def test_domain_validation_rejects_log_nonpositive_and_division_by_zero() -> None:
    with pytest.raises(ContractValidationError) as log_exc:
        evaluate_expression(UnaryOp("log", Feature("x")), {"x": 0.0})
    assert log_exc.value.code == "expression_domain_violation"

    with pytest.raises(ContractValidationError) as div_exc:
        evaluate_expression(BinaryOp("div", Feature("x"), Feature("y")), {"x": 1.0, "y": 0.0})
    assert div_exc.value.code == "expression_domain_violation"


def test_evaluator_rejects_missing_and_nonfinite_values() -> None:
    expression = BinaryOp("add", Feature("x"), Literal(1.0))

    with pytest.raises(ContractValidationError) as missing_exc:
        evaluate_expression(expression, {})
    assert missing_exc.value.code == "expression_value_missing"

    with pytest.raises(ContractValidationError) as nonfinite_exc:
        evaluate_expression(expression, {"x": float("inf")})
    assert nonfinite_exc.value.code == "expression_value_nonfinite"

