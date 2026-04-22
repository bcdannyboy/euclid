from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import BinaryOp, Conditional, Expr, Feature, Literal, State, UnaryOp
from euclid.search.dsl.ast import AlgorithmicExpr, AlgorithmicProgram


@dataclass(frozen=True)
class LoweredAlgorithmicProgram:
    initial_state_expressions: tuple[Expr, ...]
    next_state_expressions: tuple[Expr, ...]
    emit_expression: Expr
    legacy_source_ref: str
    primary_runtime: str = "expression_ir"


def lower_algorithmic_program_to_expression_ir(
    program: AlgorithmicProgram,
) -> LoweredAlgorithmicProgram:
    return LoweredAlgorithmicProgram(
        initial_state_expressions=tuple(
            _lower_algorithmic_expr(expr) for expr in program.initial_state_exprs
        ),
        next_state_expressions=tuple(
            _lower_algorithmic_expr(expr) for expr in program.next_state_exprs
        ),
        emit_expression=_lower_algorithmic_expr(program.emit_expr),
        legacy_source_ref=program.canonical_source,
    )


def _lower_algorithmic_expr(expr: AlgorithmicExpr) -> Expr:
    op = expr.op
    if op == "lit":
        return Literal(_literal_value(expr.args[0]))
    if op == "true":
        return Literal(True, domain="boolean")
    if op == "false":
        return Literal(False, domain="boolean")
    if op == "state":
        return State(f"state_{expr.args[0]}")
    if op == "obs":
        return Feature(f"obs_{expr.args[0]}")
    if op in {"add", "sub", "mul", "div", "min", "max", "lt", "le", "eq", "and", "or"}:
        return BinaryOp(
            _OPERATOR_MAP[op],
            _lower_algorithmic_expr(expr.args[0]),
            _lower_algorithmic_expr(expr.args[1]),
        )
    if op in {"neg", "abs", "not"}:
        return UnaryOp(_OPERATOR_MAP[op], _lower_algorithmic_expr(expr.args[0]))
    if op == "if":
        return Conditional(
            _lower_algorithmic_expr(expr.args[0]),
            _lower_algorithmic_expr(expr.args[1]),
            _lower_algorithmic_expr(expr.args[2]),
        )
    raise ContractValidationError(
        code="unsupported_algorithmic_expression_lowering",
        message=f"{op!r} is not supported by the expression IR lowering adapter",
        field_path="algorithmic_expr.op",
        details={"operator": op},
    )


def _literal_value(value: object) -> int | float:
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return int(value.numerator)
        return float(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    raise ContractValidationError(
        code="unsupported_algorithmic_literal_lowering",
        message="algorithmic literal lowering requires a finite numeric value",
        field_path="algorithmic_expr.args[0]",
        details={"value_type": type(value).__name__},
    )


_OPERATOR_MAP = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div",
    "min": "min",
    "max": "max",
    "neg": "neg",
    "abs": "abs",
    "lt": "lt",
    "le": "le",
    "eq": "eq",
    "and": "and",
    "or": "or",
    "not": "not",
}


__all__ = [
    "LoweredAlgorithmicProgram",
    "lower_algorithmic_program_to_expression_ir",
]
