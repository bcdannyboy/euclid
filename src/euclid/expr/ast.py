from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from euclid.contracts.errors import ContractValidationError
from euclid.expr.operators import get_operator

ScalarLiteral: TypeAlias = str | bool | int | float | None


def _require_identifier(value: str, *, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(
            code="invalid_expression_identifier",
            message=f"{field_path} must be a non-empty string",
            field_path=field_path,
        )
    return value.strip()


def _normalize_domain(domain: str | None) -> str:
    if domain is None:
        return "real"
    return _require_identifier(domain, field_path="domain")


def _normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    return _require_identifier(unit, field_path="unit")


def _normalize_literal(value: ScalarLiteral) -> ScalarLiteral:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(
                code="invalid_expression_literal",
                message="expression literal values must be finite",
                field_path="value",
            )
        return value
    raise ContractValidationError(
        code="invalid_expression_literal",
        message=f"{type(value).__name__} is not a supported expression literal",
        field_path="value",
    )


@dataclass(frozen=True)
class ExprNode:
    @property
    def node_kind(self) -> str:
        name = type(self).__name__
        result = []
        for index, character in enumerate(name):
            if index and character.isupper():
                result.append("_")
            result.append(character.lower())
        return "".join(result)


@dataclass(frozen=True)
class Literal(ExprNode):
    value: ScalarLiteral
    domain: str = "real"
    unit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _normalize_literal(self.value))
        object.__setattr__(self, "domain", _normalize_domain(self.domain))
        object.__setattr__(self, "unit", _normalize_unit(self.unit))


@dataclass(frozen=True)
class Parameter(ExprNode):
    name: str
    domain: str = "real"
    unit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_identifier(self.name, field_path="name"))
        object.__setattr__(self, "domain", _normalize_domain(self.domain))
        object.__setattr__(self, "unit", _normalize_unit(self.unit))


@dataclass(frozen=True)
class Feature(ExprNode):
    name: str
    domain: str = "real"
    unit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_identifier(self.name, field_path="name"))
        object.__setattr__(self, "domain", _normalize_domain(self.domain))
        object.__setattr__(self, "unit", _normalize_unit(self.unit))


@dataclass(frozen=True)
class State(ExprNode):
    name: str
    domain: str = "real"
    unit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_identifier(self.name, field_path="name"))
        object.__setattr__(self, "domain", _normalize_domain(self.domain))
        object.__setattr__(self, "unit", _normalize_unit(self.unit))


@dataclass(frozen=True)
class Lag(ExprNode):
    expression: Expr
    periods: int = 1

    def __post_init__(self) -> None:
        if self.periods <= 0:
            raise ContractValidationError(
                code="invalid_expression_lag",
                message="lag periods must be positive",
                field_path="periods",
            )


@dataclass(frozen=True)
class Delay(ExprNode):
    expression: Expr
    periods: int = 1

    def __post_init__(self) -> None:
        if self.periods <= 0:
            raise ContractValidationError(
                code="invalid_expression_delay",
                message="delay periods must be positive",
                field_path="periods",
            )


@dataclass(frozen=True)
class Derivative(ExprNode):
    expression: Expr
    variable: str
    order: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "variable", _require_identifier(self.variable, field_path="variable"))
        if self.order <= 0:
            raise ContractValidationError(
                code="invalid_expression_derivative",
                message="derivative order must be positive",
                field_path="order",
            )


@dataclass(frozen=True)
class Integral(ExprNode):
    expression: Expr
    variable: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "variable", _require_identifier(self.variable, field_path="variable"))


@dataclass(frozen=True)
class UnaryOp(ExprNode):
    operator: str
    operand: Expr

    def __post_init__(self) -> None:
        operator = get_operator(self.operator)
        if not operator.allows_arity(1):
            raise ContractValidationError(
                code="invalid_expression_arity",
                message=f"{self.operator!r} does not accept one operand",
                field_path="operator",
            )


@dataclass(frozen=True)
class BinaryOp(ExprNode):
    operator: str
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        operator = get_operator(self.operator)
        if not operator.allows_arity(2):
            raise ContractValidationError(
                code="invalid_expression_arity",
                message=f"{self.operator!r} does not accept two operands",
                field_path="operator",
            )
        if operator.commutative:
            from euclid.expr.serialization import expression_canonical_json

            if expression_canonical_json(self.right) < expression_canonical_json(self.left):
                left, right = self.left, self.right
                object.__setattr__(self, "left", right)
                object.__setattr__(self, "right", left)


@dataclass(frozen=True)
class NaryOp(ExprNode):
    operator: str
    children: tuple[Expr, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        operator = get_operator(self.operator)
        if not operator.allows_arity(len(self.children)):
            raise ContractValidationError(
                code="invalid_expression_arity",
                message=(
                    f"{self.operator!r} does not accept {len(self.children)} operands"
                ),
                field_path="children",
            )
        if operator.commutative:
            from euclid.expr.serialization import expression_canonical_json

            object.__setattr__(
                self,
                "children",
                tuple(sorted(self.children, key=expression_canonical_json)),
            )


@dataclass(frozen=True)
class Conditional(ExprNode):
    condition: Expr
    if_true: Expr
    if_false: Expr


@dataclass(frozen=True)
class Piecewise(ExprNode):
    cases: tuple[tuple[Expr, Expr], ...]
    default: Expr

    def __post_init__(self) -> None:
        if not self.cases:
            raise ContractValidationError(
                code="invalid_expression_piecewise",
                message="piecewise expressions require at least one case",
                field_path="cases",
            )


@dataclass(frozen=True)
class NoiseTerm(ExprNode):
    name: str
    distribution: str
    domain: str = "real"
    unit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_identifier(self.name, field_path="name"))
        object.__setattr__(
            self,
            "distribution",
            _require_identifier(self.distribution, field_path="distribution"),
        )
        object.__setattr__(self, "domain", _normalize_domain(self.domain))
        object.__setattr__(self, "unit", _normalize_unit(self.unit))


@dataclass(frozen=True)
class DistributionParameter(ExprNode):
    name: str
    domain: str = "real"
    unit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_identifier(self.name, field_path="name"))
        object.__setattr__(self, "domain", _normalize_domain(self.domain))
        object.__setattr__(self, "unit", _normalize_unit(self.unit))


@dataclass(frozen=True)
class FunctionCall(ExprNode):
    function_name: str
    args: tuple[Expr, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "function_name",
            _require_identifier(self.function_name, field_path="function_name"),
        )


Expr: TypeAlias = (
    Literal
    | Parameter
    | Feature
    | State
    | Lag
    | Delay
    | Derivative
    | Integral
    | UnaryOp
    | BinaryOp
    | NaryOp
    | Conditional
    | Piecewise
    | NoiseTerm
    | DistributionParameter
    | FunctionCall
)


def expression_hash(expression: Expr) -> str:
    from euclid.expr.serialization import expression_hash as _expression_hash

    return _expression_hash(expression)


def walk_expression(expression: Expr) -> tuple[Expr, ...]:
    children: list[Expr] = [expression]
    if isinstance(expression, (Lag, Delay, Derivative, Integral)):
        children.extend(walk_expression(expression.expression))
    elif isinstance(expression, UnaryOp):
        children.extend(walk_expression(expression.operand))
    elif isinstance(expression, BinaryOp):
        children.extend(walk_expression(expression.left))
        children.extend(walk_expression(expression.right))
    elif isinstance(expression, NaryOp):
        for child in expression.children:
            children.extend(walk_expression(child))
    elif isinstance(expression, Conditional):
        children.extend(walk_expression(expression.condition))
        children.extend(walk_expression(expression.if_true))
        children.extend(walk_expression(expression.if_false))
    elif isinstance(expression, Piecewise):
        for condition, value in expression.cases:
            children.extend(walk_expression(condition))
            children.extend(walk_expression(value))
        children.extend(walk_expression(expression.default))
    elif isinstance(expression, FunctionCall):
        for arg in expression.args:
            children.extend(walk_expression(arg))
    return tuple(children)


__all__ = [
    "BinaryOp",
    "Conditional",
    "Delay",
    "Derivative",
    "DistributionParameter",
    "Expr",
    "ExprNode",
    "Feature",
    "FunctionCall",
    "Integral",
    "Lag",
    "Literal",
    "NaryOp",
    "NoiseTerm",
    "Parameter",
    "Piecewise",
    "State",
    "UnaryOp",
    "expression_hash",
    "walk_expression",
]
