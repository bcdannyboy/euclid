from __future__ import annotations

import math
from collections.abc import Mapping

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

_PROTECTED_EPSILON = 1e-12


def evaluate_expression(expression: Expr, values: Mapping[str, float]) -> float | bool:
    result = _eval(expression, values)
    if isinstance(result, bool):
        return result
    if not math.isfinite(float(result)):
        raise ContractValidationError(
            code="expression_value_nonfinite",
            message="expression evaluation produced a non-finite value",
            field_path="expression",
        )
    return float(result)


def _eval(expression: Expr, values: Mapping[str, float]) -> float | bool:
    if isinstance(expression, Literal):
        if isinstance(expression.value, bool):
            return expression.value
        if isinstance(expression.value, (int, float)):
            return _finite(float(expression.value), field_path="value")
        raise ContractValidationError(
            code="expression_value_nonfinite",
            message="non-numeric literals cannot be evaluated as numeric expressions",
            field_path="value",
        )
    if isinstance(expression, (Feature, Parameter, State, DistributionParameter, NoiseTerm)):
        return _lookup(expression.name, values)
    if isinstance(expression, Lag):
        return _eval(expression.expression, values)
    if isinstance(expression, UnaryOp):
        value = float(_eval(expression.operand, values))
        if expression.operator == "neg":
            return -value
        if expression.operator == "abs":
            return abs(value)
        if expression.operator == "sqrt":
            if value < 0.0:
                raise _domain("sqrt requires a nonnegative argument", "operand")
            return math.sqrt(value)
        if expression.operator == "log":
            if value <= 0.0:
                raise _domain("log requires a positive argument", "operand")
            return math.log(value)
        if expression.operator == "protected_log":
            return math.log(max(value, _PROTECTED_EPSILON))
        if expression.operator == "exp":
            return math.exp(value)
        if expression.operator == "sin":
            return math.sin(value)
        if expression.operator == "cos":
            return math.cos(value)
        if expression.operator == "tan":
            return math.tan(value)
        if expression.operator == "tanh":
            return math.tanh(value)
        if expression.operator == "sigmoid":
            return 1.0 / (1.0 + math.exp(-value))
        if expression.operator == "logit":
            if value <= 0.0 or value >= 1.0:
                raise _domain("logit requires an open probability", "operand")
            return math.log(value / (1.0 - value))
        if expression.operator == "pow2":
            return value * value
        if expression.operator == "floor":
            return float(math.floor(value))
        if expression.operator == "ceil":
            return float(math.ceil(value))
    if isinstance(expression, BinaryOp):
        left = _eval(expression.left, values)
        right = _eval(expression.right, values)
        if expression.operator == "add":
            return float(left) + float(right)
        if expression.operator == "sub":
            return float(left) - float(right)
        if expression.operator == "mul":
            return float(left) * float(right)
        if expression.operator == "div":
            if float(right) == 0.0:
                raise _domain("division by zero", "right")
            return float(left) / float(right)
        if expression.operator == "protected_div":
            return float(left) / float(right) if abs(float(right)) > _PROTECTED_EPSILON else 0.0
        if expression.operator == "pow":
            return float(left) ** float(right)
        if expression.operator == "gt":
            return float(left) > float(right)
        if expression.operator == "lt":
            return float(left) < float(right)
    if isinstance(expression, NaryOp):
        values_tuple = tuple(float(_eval(child, values)) for child in expression.children)
        if expression.operator == "add":
            return sum(values_tuple)
        if expression.operator == "mul":
            result = 1.0
            for value in values_tuple:
                result *= value
            return result
        if expression.operator == "min":
            return min(values_tuple)
        if expression.operator == "max":
            return max(values_tuple)
    if isinstance(expression, Conditional):
        return _eval(expression.if_true if _eval(expression.condition, values) else expression.if_false, values)
    if isinstance(expression, Piecewise):
        for condition, value in expression.cases:
            if _eval(condition, values):
                return _eval(value, values)
        return _eval(expression.default, values)
    if isinstance(expression, FunctionCall):
        raise ContractValidationError(
            code="unsupported_expression_evaluator",
            message="opaque function calls are not directly evaluable",
            field_path="function_name",
        )
    raise ContractValidationError(
        code="unsupported_expression_evaluator",
        message=f"{type(expression).__name__} is not evaluable",
        field_path="expression",
    )


def _lookup(name: str, values: Mapping[str, float]) -> float:
    if name not in values:
        raise ContractValidationError(
            code="expression_value_missing",
            message=f"missing value for expression symbol {name!r}",
            field_path=name,
        )
    return _finite(float(values[name]), field_path=name)


def _finite(value: float, *, field_path: str) -> float:
    if not math.isfinite(value):
        raise ContractValidationError(
            code="expression_value_nonfinite",
            message=f"{field_path} must be finite",
            field_path=field_path,
        )
    return value


def _domain(message: str, field_path: str) -> ContractValidationError:
    return ContractValidationError(
        code="expression_domain_violation",
        message=message,
        field_path=field_path,
    )


__all__ = ["evaluate_expression"]

