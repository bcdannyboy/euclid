from __future__ import annotations

from typing import Any, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import (
    BinaryOp,
    Conditional,
    Delay,
    Derivative,
    DistributionParameter,
    Expr,
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
)
from euclid.runtime.hashing import canonicalize_json, sha256_digest


def expression_to_dict(expression: Expr) -> dict[str, Any]:
    if isinstance(expression, Literal):
        return _with_optional_unit(
            {
                "kind": "literal",
                "value": expression.value,
                "domain": expression.domain,
            },
            expression.unit,
        )
    if isinstance(expression, Parameter):
        return _name_payload("parameter", expression.name, expression.domain, expression.unit)
    if isinstance(expression, Feature):
        return _name_payload("feature", expression.name, expression.domain, expression.unit)
    if isinstance(expression, State):
        return _name_payload("state", expression.name, expression.domain, expression.unit)
    if isinstance(expression, Lag):
        return {
            "kind": "lag",
            "periods": expression.periods,
            "expression": expression_to_dict(expression.expression),
        }
    if isinstance(expression, Delay):
        return {
            "kind": "delay",
            "periods": expression.periods,
            "expression": expression_to_dict(expression.expression),
        }
    if isinstance(expression, Derivative):
        return {
            "kind": "derivative",
            "variable": expression.variable,
            "order": expression.order,
            "expression": expression_to_dict(expression.expression),
        }
    if isinstance(expression, Integral):
        return {
            "kind": "integral",
            "variable": expression.variable,
            "expression": expression_to_dict(expression.expression),
        }
    if isinstance(expression, UnaryOp):
        return {
            "kind": "unary_op",
            "operator": expression.operator,
            "operand": expression_to_dict(expression.operand),
        }
    if isinstance(expression, BinaryOp):
        return {
            "kind": "binary_op",
            "operator": expression.operator,
            "left": expression_to_dict(expression.left),
            "right": expression_to_dict(expression.right),
        }
    if isinstance(expression, NaryOp):
        return {
            "kind": "nary_op",
            "operator": expression.operator,
            "children": [expression_to_dict(child) for child in expression.children],
        }
    if isinstance(expression, Conditional):
        return {
            "kind": "conditional",
            "condition": expression_to_dict(expression.condition),
            "if_true": expression_to_dict(expression.if_true),
            "if_false": expression_to_dict(expression.if_false),
        }
    if isinstance(expression, Piecewise):
        return {
            "kind": "piecewise",
            "cases": [
                {
                    "condition": expression_to_dict(condition),
                    "value": expression_to_dict(value),
                }
                for condition, value in expression.cases
            ],
            "default": expression_to_dict(expression.default),
        }
    if isinstance(expression, NoiseTerm):
        return _with_optional_unit(
            {
                "kind": "noise_term",
                "name": expression.name,
                "distribution": expression.distribution,
                "domain": expression.domain,
            },
            expression.unit,
        )
    if isinstance(expression, DistributionParameter):
        return _name_payload(
            "distribution_parameter",
            expression.name,
            expression.domain,
            expression.unit,
        )
    if isinstance(expression, FunctionCall):
        return {
            "kind": "function_call",
            "function_name": expression.function_name,
            "args": [expression_to_dict(arg) for arg in expression.args],
        }
    raise ContractValidationError(
        code="unsupported_expression_node",
        message=f"{type(expression).__name__} is not serializable as expression IR",
        field_path="expression",
    )


def expression_from_dict(payload: Mapping[str, Any]) -> Expr:
    kind = str(payload.get("kind", ""))
    if kind == "literal":
        return Literal(
            payload.get("value"),
            domain=str(payload.get("domain", "real")),
            unit=_optional_str(payload.get("unit")),
        )
    if kind == "parameter":
        return Parameter(
            str(payload["name"]),
            domain=str(payload.get("domain", "real")),
            unit=_optional_str(payload.get("unit")),
        )
    if kind == "feature":
        return Feature(
            str(payload["name"]),
            domain=str(payload.get("domain", "real")),
            unit=_optional_str(payload.get("unit")),
        )
    if kind == "state":
        return State(
            str(payload["name"]),
            domain=str(payload.get("domain", "real")),
            unit=_optional_str(payload.get("unit")),
        )
    if kind == "lag":
        return Lag(
            expression_from_dict(_mapping(payload["expression"])),
            periods=int(payload.get("periods", 1)),
        )
    if kind == "delay":
        return Delay(
            expression_from_dict(_mapping(payload["expression"])),
            periods=int(payload.get("periods", 1)),
        )
    if kind == "derivative":
        return Derivative(
            expression_from_dict(_mapping(payload["expression"])),
            variable=str(payload["variable"]),
            order=int(payload.get("order", 1)),
        )
    if kind == "integral":
        return Integral(
            expression_from_dict(_mapping(payload["expression"])),
            variable=str(payload["variable"]),
        )
    if kind == "unary_op":
        return UnaryOp(str(payload["operator"]), expression_from_dict(_mapping(payload["operand"])))
    if kind == "binary_op":
        return BinaryOp(
            str(payload["operator"]),
            expression_from_dict(_mapping(payload["left"])),
            expression_from_dict(_mapping(payload["right"])),
        )
    if kind == "nary_op":
        return NaryOp(
            str(payload["operator"]),
            tuple(expression_from_dict(_mapping(child)) for child in payload["children"]),
        )
    if kind == "conditional":
        return Conditional(
            expression_from_dict(_mapping(payload["condition"])),
            expression_from_dict(_mapping(payload["if_true"])),
            expression_from_dict(_mapping(payload["if_false"])),
        )
    if kind == "piecewise":
        return Piecewise(
            cases=tuple(
                (
                    expression_from_dict(_mapping(case["condition"])),
                    expression_from_dict(_mapping(case["value"])),
                )
                for case in payload["cases"]
            ),
            default=expression_from_dict(_mapping(payload["default"])),
        )
    if kind == "noise_term":
        return NoiseTerm(
            str(payload["name"]),
            distribution=str(payload["distribution"]),
            domain=str(payload.get("domain", "real")),
            unit=_optional_str(payload.get("unit")),
        )
    if kind == "distribution_parameter":
        return DistributionParameter(
            str(payload["name"]),
            domain=str(payload.get("domain", "real")),
            unit=_optional_str(payload.get("unit")),
        )
    if kind == "function_call":
        return FunctionCall(
            str(payload["function_name"]),
            tuple(expression_from_dict(_mapping(arg)) for arg in payload.get("args", ())),
        )
    raise ContractValidationError(
        code="unsupported_expression_node",
        message=f"{kind!r} is not a supported expression node kind",
        field_path="kind",
    )


def expression_canonical_json(expression: Expr) -> str:
    return canonicalize_json(expression_to_dict(expression))


def expression_hash(expression: Expr) -> str:
    return sha256_digest(expression_to_dict(expression))


def _name_payload(kind: str, name: str, domain: str, unit: str | None) -> dict[str, Any]:
    return _with_optional_unit(
        {"kind": kind, "name": name, "domain": domain},
        unit,
    )


def _with_optional_unit(payload: dict[str, Any], unit: str | None) -> dict[str, Any]:
    if unit is not None:
        payload["unit"] = unit
    return payload


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(
            code="invalid_expression_serialization",
            message="nested expression payloads must be mappings",
            field_path="expression",
        )
    return value


__all__ = [
    "expression_canonical_json",
    "expression_from_dict",
    "expression_hash",
    "expression_to_dict",
]

