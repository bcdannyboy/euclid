from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from pint import DimensionalityError, UnitRegistry

from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import (
    BinaryOp,
    Conditional,
    DistributionParameter,
    Expr,
    Feature,
    FunctionCall,
    Lag,
    Literal,
    NaryOp,
    NoiseTerm,
    Parameter,
    Piecewise,
    State,
    UnaryOp,
)

_UREG = UnitRegistry()
_DIMENSIONLESS = _UREG.Unit("")


@dataclass(frozen=True)
class UnitValidationResult:
    status: str
    unit: str
    reason_codes: tuple[str, ...] = ()


def unit_registry() -> UnitRegistry:
    return _UREG


def infer_expression_unit(
    expression: Expr,
    *,
    variable_units: Mapping[str, str] | None = None,
):
    units = dict(variable_units or {})
    return _infer(expression, units)


def validate_expression_units(
    expression: Expr,
    *,
    variable_units: Mapping[str, str] | None = None,
) -> UnitValidationResult:
    unit = infer_expression_unit(expression, variable_units=variable_units)
    return UnitValidationResult(status="passed", unit=str(unit))


def _infer(expression: Expr, variable_units: Mapping[str, str]):
    if isinstance(expression, Literal):
        return _parse_unit(expression.unit)
    if isinstance(expression, (Feature, Parameter, State, DistributionParameter, NoiseTerm)):
        return _parse_unit(variable_units.get(expression.name) or expression.unit)
    if isinstance(expression, Lag):
        return _infer(expression.expression, variable_units)
    if isinstance(expression, UnaryOp):
        unit = _infer(expression.operand, variable_units)
        if expression.operator in {"neg", "abs", "floor", "ceil"}:
            return unit
        if expression.operator == "pow2":
            return unit * unit
        if expression.operator == "sqrt":
            return unit ** 0.5
        if expression.operator in {
            "exp",
            "log",
            "protected_log",
            "sin",
            "cos",
            "tan",
            "tanh",
            "sigmoid",
            "logit",
        }:
            _require_dimensionless(unit, expression.operator)
            return _DIMENSIONLESS
        return unit
    if isinstance(expression, BinaryOp):
        left = _infer(expression.left, variable_units)
        right = _infer(expression.right, variable_units)
        if expression.operator in {"add", "sub", "gt", "lt"}:
            _require_compatible(left, right, expression.operator)
            return left if expression.operator in {"add", "sub"} else _DIMENSIONLESS
        if expression.operator == "mul":
            return left * right
        if expression.operator in {"div", "protected_div"}:
            return left / right
        if expression.operator == "pow":
            _require_dimensionless(right, "pow_exponent")
            return left
    if isinstance(expression, NaryOp):
        child_units = [_infer(child, variable_units) for child in expression.children]
        if not child_units:
            return _DIMENSIONLESS
        if expression.operator in {"add", "min", "max"}:
            first = child_units[0]
            for unit in child_units[1:]:
                _require_compatible(first, unit, expression.operator)
            return first
        if expression.operator == "mul":
            result = _DIMENSIONLESS
            for unit in child_units:
                result = result * unit
            return result
    if isinstance(expression, Conditional):
        true_unit = _infer(expression.if_true, variable_units)
        false_unit = _infer(expression.if_false, variable_units)
        _require_compatible(true_unit, false_unit, "where")
        return true_unit
    if isinstance(expression, Piecewise):
        result = _infer(expression.default, variable_units)
        for _, value in expression.cases:
            _require_compatible(result, _infer(value, variable_units), "piecewise")
        return result
    if isinstance(expression, FunctionCall):
        return _DIMENSIONLESS
    raise ContractValidationError(
        code="expression_unit_unknown",
        message=f"{type(expression).__name__} has no unit inference rule",
        field_path="expression",
    )


def _parse_unit(unit: str | None):
    if unit is None:
        return _DIMENSIONLESS
    try:
        return _UREG.Unit(unit)
    except Exception as exc:  # pragma: no cover - Pint exception surface varies
        raise ContractValidationError(
            code="expression_unit_invalid",
            message=f"{unit!r} is not a valid Pint unit",
            field_path="unit",
        ) from exc


def _require_compatible(left, right, operator: str) -> None:
    try:
        (1 * left).to(right)
    except DimensionalityError as exc:
        raise ContractValidationError(
            code="expression_unit_incompatible",
            message=f"{operator} requires compatible units",
            field_path="operator",
            details={"left_unit": str(left), "right_unit": str(right)},
        ) from exc


def _require_dimensionless(unit, operator: str) -> None:
    try:
        (1 * unit).to(_DIMENSIONLESS)
    except DimensionalityError as exc:
        raise ContractValidationError(
            code="expression_unit_incompatible",
            message=f"{operator} requires a dimensionless argument",
            field_path="operator",
            details={"unit": str(unit)},
        ) from exc


__all__ = [
    "UnitValidationResult",
    "infer_expression_unit",
    "unit_registry",
    "validate_expression_units",
]

