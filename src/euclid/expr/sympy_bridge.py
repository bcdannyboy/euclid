from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import sympy as sp

from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import (
    BinaryOp,
    Conditional,
    Derivative,
    Expr,
    Feature,
    FunctionCall,
    Integral,
    Literal,
    NaryOp,
    Parameter,
    Piecewise,
    State,
    UnaryOp,
)


def to_sympy(
    expression: Expr,
    *,
    assumptions: Mapping[str, Mapping[str, Any]] | None = None,
):
    symbols: dict[str, sp.Symbol] = {}
    assumption_payload = dict(assumptions or {})

    def symbol(name: str, domain: str = "real") -> sp.Symbol:
        payload = dict(assumption_payload.get(name, {}))
        effective_domain = str(payload.get("domain", domain))
        kwargs: dict[str, bool] = {}
        if effective_domain in {"real", "positive_real", "nonnegative_real", "integer", "probability", "bounded_interval"}:
            kwargs["real"] = True
        if effective_domain == "positive_real":
            kwargs["positive"] = True
        if effective_domain == "nonnegative_real":
            kwargs["nonnegative"] = True
        if effective_domain == "integer":
            kwargs["integer"] = True
        key = f"{name}:{sorted(kwargs.items())}"
        if key not in symbols:
            symbols[key] = sp.Symbol(name, **kwargs)
        return symbols[key]

    return _to_sympy(expression, symbol)


def from_sympy(value) -> Expr:
    if value == sp.S.true:
        return Literal(True, domain="boolean")
    if value == sp.S.false:
        return Literal(False, domain="boolean")
    if value.is_Integer:
        return Literal(int(value))
    if value.is_Number:
        return Literal(float(value))
    if isinstance(value, sp.Symbol):
        return Feature(str(value))
    if isinstance(value, sp.Add):
        return NaryOp("add", tuple(from_sympy(arg) for arg in value.args))
    if isinstance(value, sp.Mul):
        return NaryOp("mul", tuple(from_sympy(arg) for arg in value.args))
    if isinstance(value, sp.Pow):
        base, exponent = value.args
        if exponent.is_number and sp.simplify(exponent - 1) == 0:
            return from_sympy(base)
        if exponent == sp.Rational(1, 2):
            return UnaryOp("sqrt", from_sympy(base))
        return BinaryOp("pow", from_sympy(base), from_sympy(exponent))
    if isinstance(value, sp.StrictGreaterThan):
        left, right = value.args
        return BinaryOp("gt", from_sympy(left), from_sympy(right))
    if isinstance(value, sp.StrictLessThan):
        left, right = value.args
        return BinaryOp("lt", from_sympy(left), from_sympy(right))
    if isinstance(value, sp.GreaterThan):
        left, right = value.args
        return BinaryOp("ge", from_sympy(left), from_sympy(right))
    if isinstance(value, sp.LessThan):
        left, right = value.args
        return BinaryOp("le", from_sympy(left), from_sympy(right))
    if isinstance(value, sp.Equality):
        left, right = value.args
        return BinaryOp("eq", from_sympy(left), from_sympy(right))
    if isinstance(value, sp.And):
        return NaryOp("and", tuple(from_sympy(arg) for arg in value.args))
    if isinstance(value, sp.Or):
        return NaryOp("or", tuple(from_sympy(arg) for arg in value.args))
    if isinstance(value, sp.Not):
        return UnaryOp("not", from_sympy(value.args[0]))
    if isinstance(value, sp.Piecewise):
        cases = []
        default = Literal(0.0)
        for expression, condition in value.args:
            if condition == True:  # noqa: E712 - SymPy uses literal True here
                default = from_sympy(expression)
            else:
                cases.append((from_sympy(condition), from_sympy(expression)))
        return Piecewise(cases=tuple(cases), default=default)
    if isinstance(value, sp.Derivative):
        expr = from_sympy(value.expr)
        variable, order = value.variable_count[0]
        return Derivative(expr, variable=str(variable), order=int(order))
    if isinstance(value, sp.Integral):
        expr = from_sympy(value.function)
        variable = value.variables[0]
        return Integral(expr, variable=str(variable))
    if isinstance(value, sp.Function):
        function_name = value.func.__name__
        mapping = {
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "tanh": "tanh",
            "exp": "exp",
            "log": "log",
            "Abs": "abs",
        }
        operator = mapping.get(function_name)
        if operator is not None and len(value.args) == 1:
            return UnaryOp(operator, from_sympy(value.args[0]))
    raise ContractValidationError(
        code="unsupported_sympy_expression",
        message=f"SymPy expression {value!r} is not supported by Euclid expression IR",
        field_path="sympy_expression",
        details={"sympy_type": type(value).__name__},
    )


def simplify_expression(
    expression: Expr,
    *,
    assumptions: Mapping[str, Mapping[str, Any]] | None = None,
) -> Expr:
    return from_sympy(sp.simplify(to_sympy(expression, assumptions=assumptions)))


def are_equivalent(
    left: Expr,
    right: Expr,
    *,
    assumptions: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    left_sympy = to_sympy(left, assumptions=assumptions)
    right_sympy = to_sympy(right, assumptions=assumptions)
    try:
        return bool(sp.simplify(left_sympy - right_sympy) == 0)
    except TypeError:
        return bool(sp.simplify(sp.Eq(left_sympy, right_sympy)) == sp.S.true)


def derivative(expression: Expr, *, variable: str) -> Expr:
    symbol = sp.Symbol(variable, real=True)
    return from_sympy(sp.diff(to_sympy(expression), symbol))


def _to_sympy(expression: Expr, symbol_factory):
    if isinstance(expression, Literal):
        if isinstance(expression.value, bool):
            return sp.S.true if expression.value else sp.S.false
        if isinstance(expression.value, int):
            return sp.Integer(expression.value)
        if isinstance(expression.value, float):
            return sp.Float(expression.value)
        raise ContractValidationError(
            code="unsupported_sympy_expression",
            message="non-numeric literals cannot lower to SymPy",
            field_path="value",
        )
    if isinstance(expression, Feature):
        return symbol_factory(expression.name, expression.domain)
    if isinstance(expression, Parameter):
        return symbol_factory(expression.name, expression.domain)
    if isinstance(expression, State):
        return symbol_factory(expression.name, expression.domain)
    if isinstance(expression, UnaryOp):
        operand = _to_sympy(expression.operand, symbol_factory)
        return {
            "neg": lambda: -operand,
            "abs": lambda: sp.Abs(operand),
            "sqrt": lambda: sp.sqrt(operand),
            "exp": lambda: sp.exp(operand),
            "log": lambda: sp.log(operand),
            "protected_log": lambda: sp.log(operand),
            "sin": lambda: sp.sin(operand),
            "cos": lambda: sp.cos(operand),
            "tan": lambda: sp.tan(operand),
            "tanh": lambda: sp.tanh(operand),
            "pow2": lambda: operand**2,
            "not": lambda: sp.Not(operand),
        }.get(expression.operator, lambda: _unsupported(expression.operator))()
    if isinstance(expression, BinaryOp):
        left = _to_sympy(expression.left, symbol_factory)
        right = _to_sympy(expression.right, symbol_factory)
        return {
            "add": lambda: left + right,
            "sub": lambda: left - right,
            "mul": lambda: left * right,
            "div": lambda: left / right,
            "protected_div": lambda: left / right,
            "pow": lambda: left**right,
            "gt": lambda: sp.StrictGreaterThan(left, right),
            "lt": lambda: sp.StrictLessThan(left, right),
            "ge": lambda: sp.GreaterThan(left, right),
            "le": lambda: sp.LessThan(left, right),
            "eq": lambda: sp.Eq(left, right),
            "and": lambda: sp.And(left, right),
            "or": lambda: sp.Or(left, right),
        }.get(expression.operator, lambda: _unsupported(expression.operator))()
    if isinstance(expression, NaryOp):
        args = tuple(_to_sympy(child, symbol_factory) for child in expression.children)
        if expression.operator == "add":
            return sp.Add(*args)
        if expression.operator == "mul":
            return sp.Mul(*args)
        if expression.operator == "min":
            return sp.Min(*args)
        if expression.operator == "max":
            return sp.Max(*args)
        if expression.operator == "and":
            return sp.And(*args)
        if expression.operator == "or":
            return sp.Or(*args)
        _unsupported(expression.operator)
    if isinstance(expression, Conditional):
        return sp.Piecewise(
            (_to_sympy(expression.if_true, symbol_factory), _to_sympy(expression.condition, symbol_factory)),
            (_to_sympy(expression.if_false, symbol_factory), True),
        )
    if isinstance(expression, Piecewise):
        args = [
            (_to_sympy(value, symbol_factory), _to_sympy(condition, symbol_factory))
            for condition, value in expression.cases
        ]
        args.append((_to_sympy(expression.default, symbol_factory), True))
        return sp.Piecewise(*args)
    if isinstance(expression, Derivative):
        return sp.diff(
            _to_sympy(expression.expression, symbol_factory),
            sp.Symbol(expression.variable),
            expression.order,
        )
    if isinstance(expression, Integral):
        return sp.integrate(
            _to_sympy(expression.expression, symbol_factory),
            sp.Symbol(expression.variable),
        )
    if isinstance(expression, FunctionCall):
        raise ContractValidationError(
            code="unsupported_sympy_expression",
            message="opaque function calls cannot lower to SymPy",
            field_path="function_name",
        )
    raise ContractValidationError(
        code="unsupported_sympy_expression",
        message=f"{type(expression).__name__} cannot lower to SymPy",
        field_path="expression",
    )


def _unsupported(operator: str):
    raise ContractValidationError(
        code="unsupported_sympy_expression",
        message=f"{operator!r} has no SymPy lowering",
        field_path="operator",
    )


__all__ = [
    "are_equivalent",
    "derivative",
    "from_sympy",
    "simplify_expression",
    "to_sympy",
]
