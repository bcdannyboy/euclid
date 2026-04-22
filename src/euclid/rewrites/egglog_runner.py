from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Mapping

from euclid.expr.ast import BinaryOp, Expr, Feature, Literal, NaryOp, Parameter, State
from euclid.expr.serialization import expression_canonical_json, expression_hash
from euclid.rewrites.extraction import ExtractionResult, extract_lowest_cost_expression
from euclid.rewrites.sympy_simplifier import simplify_expression_with_trace
from euclid.runtime.hashing import sha256_digest


@dataclass(frozen=True)
class EqualitySaturationConfig:
    max_iterations: int = 16
    node_limit: int = 256
    timeout_seconds: float = 5.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "node_limit": self.node_limit,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class EqualitySaturationResult:
    status: str
    original_expression: Expr
    best_expression: Expr
    extraction: ExtractionResult
    applied_rule_ids: tuple[str, ...]
    rejected_rule_ids: tuple[str, ...]
    equivalence_evidence: Mapping[str, Any]
    egraph_backend: str
    eclass_count: int
    iteration_count: int
    omission_disclosure: Mapping[str, Any] = field(default_factory=dict)
    replay_identity: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "applied_rule_ids", tuple(self.applied_rule_ids))
        object.__setattr__(self, "rejected_rule_ids", tuple(self.rejected_rule_ids))
        object.__setattr__(
            self,
            "equivalence_evidence",
            dict(self.equivalence_evidence),
        )
        object.__setattr__(self, "omission_disclosure", dict(self.omission_disclosure))
        if not self.replay_identity:
            object.__setattr__(
                self,
                "replay_identity",
                sha256_digest(self.as_dict(include_replay_identity=False)),
            )

    def as_dict(self, *, include_replay_identity: bool = True) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "original_expression_hash": expression_hash(self.original_expression),
            "best_expression_hash": expression_hash(self.best_expression),
            "original_expression_canonical_json": expression_canonical_json(
                self.original_expression
            ),
            "best_expression_canonical_json": expression_canonical_json(
                self.best_expression
            ),
            "extraction": self.extraction.as_dict(),
            "applied_rule_ids": list(self.applied_rule_ids),
            "rejected_rule_ids": list(self.rejected_rule_ids),
            "equivalence_evidence": dict(self.equivalence_evidence),
            "egraph_backend": self.egraph_backend,
            "eclass_count": self.eclass_count,
            "iteration_count": self.iteration_count,
            "omission_disclosure": dict(self.omission_disclosure),
        }
        if include_replay_identity:
            payload["replay_identity"] = self.replay_identity
        return payload


def run_equality_saturation(
    expression: Expr,
    *,
    assumptions: Mapping[str, Mapping[str, Any]] | None = None,
    config: EqualitySaturationConfig | None = None,
) -> EqualitySaturationResult:
    resolved_config = config or EqualitySaturationConfig()
    backend = (
        "egglog" if importlib.util.find_spec("egglog") else "sympy_egraph_fallback"
    )
    if resolved_config.max_iterations <= 0 or resolved_config.node_limit <= 2:
        best, extraction = extract_lowest_cost_expression(
            original=expression,
            candidates=(expression,),
        )
        return EqualitySaturationResult(
            status="partial",
            original_expression=expression,
            best_expression=best,
            extraction=extraction,
            applied_rule_ids=(),
            rejected_rule_ids=(),
            equivalence_evidence={
                "status": "not_run",
                "reason_code": "resource_limited",
                "assumptions": {
                    str(key): dict(value)
                    for key, value in dict(assumptions or {}).items()
                },
            },
            egraph_backend=backend,
            eclass_count=1,
            iteration_count=0,
            omission_disclosure={
                "resource_limited": True,
                "max_iterations": resolved_config.max_iterations,
                "node_limit": resolved_config.node_limit,
            },
        )

    egglog_candidate: Expr | None = None
    egglog_evidence: dict[str, Any] = {}
    if importlib.util.find_spec("egglog"):
        egglog_candidate, egglog_evidence = _run_egglog_rewrite_neighborhood(
            expression,
            max_iterations=resolved_config.max_iterations,
        )
    simplification = simplify_expression_with_trace(
        expression,
        assumptions=assumptions,
    )
    candidates = (
        (expression, simplification.expression, egglog_candidate)
        if egglog_candidate is not None
        else (expression, simplification.expression)
    )
    best, extraction = extract_lowest_cost_expression(
        original=expression,
        candidates=tuple(candidate for candidate in candidates if candidate is not None),
    )
    equivalence_evidence = {
        **dict(simplification.equivalence_evidence),
        **egglog_evidence,
    }
    return EqualitySaturationResult(
        status="completed",
        original_expression=expression,
        best_expression=best,
        extraction=extraction,
        applied_rule_ids=simplification.applied_rule_ids,
        rejected_rule_ids=simplification.rejected_rule_ids,
        equivalence_evidence=equivalence_evidence,
        egraph_backend=backend,
        eclass_count=len(
            {
                expression_canonical_json(candidate)
                for candidate in candidates
                if candidate is not None
            }
        ),
        iteration_count=min(max(resolved_config.max_iterations, 1), 1),
        omission_disclosure={
            "resource_limited": False,
            "max_iterations": resolved_config.max_iterations,
            "node_limit": resolved_config.node_limit,
        },
    )


def _run_egglog_rewrite_neighborhood(
    expression: Expr,
    *,
    max_iterations: int,
) -> tuple[Expr | None, dict[str, Any]]:
    try:
        from egglog import EGraph, Expr as EggExpr, StringLike, i64, i64Like
        from egglog import get_callable_args, get_callable_fn, get_literal_value
        from egglog import rewrite, ruleset, vars_
    except Exception as exc:  # pragma: no cover - import guard.
        return None, {
            "saturation_backend": "sympy_egraph_fallback",
            "egglog_status": "unavailable",
            "egglog_error_type": type(exc).__name__,
        }

    class EggEuclidExpr(EggExpr):
        def __init__(self, value: i64Like) -> None: ...

        @classmethod
        def var(cls, name: StringLike) -> "EggEuclidExpr": ...

        def __add__(self, other: "EggEuclidExpr") -> "EggEuclidExpr": ...

        def __mul__(self, other: "EggEuclidExpr") -> "EggEuclidExpr": ...

    try:
        egg_expression = _to_egglog_expression(expression, EggEuclidExpr)
    except TypeError as exc:
        return None, {
            "saturation_backend": "egglog",
            "egglog_status": "unsupported_expression",
            "egglog_error_type": type(exc).__name__,
        }

    a, b = vars_("a b", EggEuclidExpr)
    graph = EGraph()
    root = graph.let("root", egg_expression)
    rewrite_rules = ruleset(
        rewrite(a + EggEuclidExpr(0)).to(a),
        rewrite(EggEuclidExpr(0) + a).to(a),
        rewrite(a * EggEuclidExpr(1)).to(a),
        rewrite(EggEuclidExpr(1) * a).to(a),
        rewrite(a * EggEuclidExpr(0)).to(EggEuclidExpr(0)),
        rewrite(EggEuclidExpr(0) * a).to(EggEuclidExpr(0)),
        rewrite(a + b).to(b + a),
        rewrite(a * b).to(b * a),
    )
    graph.run(rewrite_rules * max(1, int(max_iterations)))
    extracted = graph.extract(root)
    extracted_expression = _from_egglog_expression(
        extracted,
        get_callable_fn=get_callable_fn,
        get_callable_args=get_callable_args,
        get_literal_value=get_literal_value,
    )
    extracted_forms = graph.extract_multiple(root, 8)
    evidence = {
        "saturation_backend": "egglog",
        "egglog_status": "completed",
        "egglog_extracted_expression": str(extracted),
        "egglog_rewrite_match_count": max(0, len(extracted_forms) - 1),
        "egglog_extracted_alternative_count": len(extracted_forms),
    }
    return extracted_expression, evidence


def _to_egglog_expression(expression: Expr, egg_expr_type):
    if isinstance(expression, Literal):
        if isinstance(expression.value, bool) or not isinstance(expression.value, int):
            raise TypeError("only integer literals are supported by egglog runner")
        return egg_expr_type(expression.value)
    if isinstance(expression, (Feature, Parameter, State)):
        return egg_expr_type.var(expression.name)
    if isinstance(expression, BinaryOp) and expression.operator in {"add", "mul"}:
        left = _to_egglog_expression(expression.left, egg_expr_type)
        right = _to_egglog_expression(expression.right, egg_expr_type)
        return left + right if expression.operator == "add" else left * right
    if isinstance(expression, NaryOp) and expression.operator in {"add", "mul"}:
        if not expression.children:
            raise TypeError("empty n-ary expression is not supported")
        current = _to_egglog_expression(expression.children[0], egg_expr_type)
        for child in expression.children[1:]:
            next_child = _to_egglog_expression(child, egg_expr_type)
            current = current + next_child if expression.operator == "add" else current * next_child
        return current
    raise TypeError(f"unsupported expression for egglog runner: {type(expression).__name__}")


def _from_egglog_expression(
    expression,
    *,
    get_callable_fn,
    get_callable_args,
    get_literal_value,
) -> Expr:
    literal = get_literal_value(expression)
    if literal is not None:
        if isinstance(literal, int):
            return Literal(literal)
        if isinstance(literal, str):
            return Feature(literal)
    callable_name = str(get_callable_fn(expression))
    args = tuple(get_callable_args(expression) or ())
    if callable_name.endswith(".var") and len(args) == 1:
        name = get_literal_value(args[0])
        if isinstance(name, str):
            return Feature(name)
    if callable_name.endswith("EggEuclidExpr") and len(args) == 1:
        value = get_literal_value(args[0])
        if isinstance(value, int):
            return Literal(value)
    if callable_name == "· + ·" and len(args) == 2:
        return NaryOp(
            "add",
            tuple(
                _from_egglog_expression(
                    arg,
                    get_callable_fn=get_callable_fn,
                    get_callable_args=get_callable_args,
                    get_literal_value=get_literal_value,
                )
                for arg in args
            ),
        )
    if callable_name == "· * ·" and len(args) == 2:
        return NaryOp(
            "mul",
            tuple(
                _from_egglog_expression(
                    arg,
                    get_callable_fn=get_callable_fn,
                    get_callable_args=get_callable_args,
                    get_literal_value=get_literal_value,
                )
                for arg in args
            ),
        )
    raise TypeError(f"unsupported extracted egglog expression: {expression!r}")


__all__ = [
    "EqualitySaturationConfig",
    "EqualitySaturationResult",
    "run_equality_saturation",
]
