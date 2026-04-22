from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from euclid.expr.ast import (
    BinaryOp,
    Conditional,
    Delay,
    Derivative,
    DistributionParameter,
    Expr,
    FunctionCall,
    Integral,
    Lag,
    Literal,
    NaryOp,
    NoiseTerm,
    Piecewise,
    UnaryOp,
)
from euclid.expr.serialization import expression_canonical_json, expression_hash


@dataclass(frozen=True)
class ExtractionResult:
    original_expression_hash: str
    best_expression_hash: str
    original_cost: int
    best_cost: int
    best_expression_canonical_json: str
    tie_break_rule: str = "min_cost_then_min_canonical_expression_json"

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_expression_hash": self.original_expression_hash,
            "best_expression_hash": self.best_expression_hash,
            "original_cost": self.original_cost,
            "best_cost": self.best_cost,
            "best_expression_canonical_json": self.best_expression_canonical_json,
            "tie_break_rule": self.tie_break_rule,
        }


def expression_cost(expression: Expr) -> int:
    if isinstance(expression, (Literal, NoiseTerm, DistributionParameter)):
        return 1
    if expression.__class__.__name__ in {"Feature", "Parameter", "State"}:
        return 1
    if isinstance(expression, (Lag, Delay, Derivative, Integral)):
        return 2 + expression_cost(expression.expression)
    if isinstance(expression, UnaryOp):
        return 1 + expression_cost(expression.operand)
    if isinstance(expression, BinaryOp):
        return 1 + expression_cost(expression.left) + expression_cost(expression.right)
    if isinstance(expression, NaryOp):
        return 1 + sum(expression_cost(child) for child in expression.children)
    if isinstance(expression, Conditional):
        return (
            2
            + expression_cost(expression.condition)
            + expression_cost(expression.if_true)
            + expression_cost(expression.if_false)
        )
    if isinstance(expression, Piecewise):
        return (
            2
            + expression_cost(expression.default)
            + sum(
                expression_cost(condition) + expression_cost(value)
                for condition, value in expression.cases
            )
        )
    if isinstance(expression, FunctionCall):
        return 3 + sum(expression_cost(arg) for arg in expression.args)
    return 1000


def extract_lowest_cost_expression(
    *,
    original: Expr,
    candidates: Sequence[Expr],
) -> tuple[Expr, ExtractionResult]:
    candidate_tuple = tuple(candidates) or (original,)
    best = sorted(
        candidate_tuple,
        key=lambda expression: (
            expression_cost(expression),
            expression_canonical_json(expression),
        ),
    )[0]
    return best, ExtractionResult(
        original_expression_hash=expression_hash(original),
        best_expression_hash=expression_hash(best),
        original_cost=expression_cost(original),
        best_cost=expression_cost(best),
        best_expression_canonical_json=expression_canonical_json(best),
    )


__all__ = ["ExtractionResult", "extract_lowest_cost_expression", "expression_cost"]
