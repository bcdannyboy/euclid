from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import BinaryOp, Expr, Literal, NaryOp, UnaryOp
from euclid.expr.serialization import expression_canonical_json, expression_hash
from euclid.expr.sympy_bridge import are_equivalent
from euclid.rewrites.rules import (
    RewriteApplicabilityContext,
    RewriteEvidence,
    validate_rewrite_application,
)
from euclid.runtime.hashing import sha256_digest


@dataclass(frozen=True)
class SimplificationResult:
    original_expression: Expr
    expression: Expr
    applied_rule_evidence: tuple[RewriteEvidence, ...] = ()
    rejected_rules: tuple[Mapping[str, Any], ...] = ()
    equivalence_evidence: Mapping[str, Any] = field(default_factory=dict)
    replay_identity: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "applied_rule_evidence",
            tuple(self.applied_rule_evidence),
        )
        object.__setattr__(
            self,
            "rejected_rules",
            tuple(dict(item) for item in self.rejected_rules),
        )
        object.__setattr__(
            self,
            "equivalence_evidence",
            dict(self.equivalence_evidence),
        )
        if not self.replay_identity:
            object.__setattr__(
                self,
                "replay_identity",
                sha256_digest(self.as_dict(include_replay_identity=False)),
            )

    @property
    def applied_rule_ids(self) -> tuple[str, ...]:
        return tuple(evidence.rule_id for evidence in self.applied_rule_evidence)

    @property
    def rejected_rule_ids(self) -> tuple[str, ...]:
        return tuple(str(rule["rule_id"]) for rule in self.rejected_rules)

    def as_dict(self, *, include_replay_identity: bool = True) -> dict[str, Any]:
        payload = {
            "original_expression_hash": expression_hash(self.original_expression),
            "expression_hash": expression_hash(self.expression),
            "original_expression_canonical_json": expression_canonical_json(
                self.original_expression
            ),
            "expression_canonical_json": expression_canonical_json(self.expression),
            "applied_rule_evidence": [
                evidence.as_dict() for evidence in self.applied_rule_evidence
            ],
            "rejected_rules": [dict(rule) for rule in self.rejected_rules],
            "equivalence_evidence": dict(self.equivalence_evidence),
        }
        if include_replay_identity:
            payload["replay_identity"] = self.replay_identity
        return payload


def simplify_expression_with_trace(
    expression: Expr,
    *,
    assumptions: Mapping[str, Mapping[str, Any]] | None = None,
) -> SimplificationResult:
    context = RewriteApplicabilityContext(assumptions=assumptions or {})
    applied: list[RewriteEvidence] = []
    rejected: list[Mapping[str, Any]] = []
    simplified = _simplify(
        expression,
        context=context,
        applied=applied,
        rejected=rejected,
    )
    equivalence_evidence = _verify_equivalence(
        original=expression,
        simplified=simplified,
        assumptions=context.assumptions,
    )
    return SimplificationResult(
        original_expression=expression,
        expression=simplified,
        applied_rule_evidence=tuple(applied),
        rejected_rules=tuple(rejected),
        equivalence_evidence=equivalence_evidence,
    )


def _simplify(
    expression: Expr,
    *,
    context: RewriteApplicabilityContext,
    applied: list[RewriteEvidence],
    rejected: list[Mapping[str, Any]],
) -> Expr:
    if isinstance(expression, UnaryOp):
        operand = _simplify(
            expression.operand,
            context=context,
            applied=applied,
            rejected=rejected,
        )
        rewritten = UnaryOp(expression.operator, operand)
        if (
            rewritten.operator == "sqrt"
            and isinstance(rewritten.operand, UnaryOp)
            and rewritten.operand.operator == "pow2"
        ):
            candidate = rewritten.operand.operand
            try:
                applied.append(
                    validate_rewrite_application(
                        rule_id="sqrt_square_nonnegative",
                        before=rewritten,
                        after=candidate,
                        context=context,
                    )
                )
                return candidate
            except ContractValidationError as exc:
                rejected.append({"rule_id": "sqrt_square_nonnegative", **exc.as_dict()})
                return UnaryOp("abs", candidate)
        if (
            rewritten.operator == "log"
            and isinstance(rewritten.operand, UnaryOp)
            and rewritten.operand.operator == "exp"
        ):
            candidate = rewritten.operand.operand
            applied.append(
                validate_rewrite_application(
                    rule_id="log_exp_inverse",
                    before=rewritten,
                    after=candidate,
                    context=context,
                )
            )
            return candidate
        return rewritten
    if isinstance(expression, BinaryOp):
        left = _simplify(
            expression.left,
            context=context,
            applied=applied,
            rejected=rejected,
        )
        right = _simplify(
            expression.right,
            context=context,
            applied=applied,
            rejected=rejected,
        )
        rewritten = BinaryOp(expression.operator, left, right)
        if rewritten.operator == "div":
            common = _common_division_factor(rewritten)
            if common is not None:
                try:
                    applied.append(
                        validate_rewrite_application(
                            rule_id="divide_common_factor",
                            before=rewritten,
                            after=common,
                            context=context,
                        )
                    )
                    return common
                except ContractValidationError as exc:
                    rejected.append(
                        {"rule_id": "divide_common_factor", **exc.as_dict()}
                    )
        return rewritten
    if isinstance(expression, NaryOp):
        children = tuple(
            _simplify(child, context=context, applied=applied, rejected=rejected)
            for child in expression.children
        )
        if expression.operator == "add":
            return _simplify_add(children, context=context, applied=applied)
        if expression.operator == "mul":
            return _simplify_mul(children, context=context, applied=applied)
        return NaryOp(expression.operator, children)
    return expression


def _simplify_add(
    children: tuple[Expr, ...],
    *,
    context: RewriteApplicabilityContext,
    applied: list[RewriteEvidence],
) -> Expr:
    flattened: list[Expr] = []
    for child in children:
        if isinstance(child, NaryOp) and child.operator == "add":
            flattened.extend(child.children)
        else:
            flattened.append(child)
    nonzero = [child for child in flattened if not _is_literal(child, 0)]
    for child in flattened:
        if _is_literal(child, 0) and nonzero:
            before = NaryOp("add", (nonzero[0], child))
            applied.append(
                validate_rewrite_application(
                    rule_id="additive_identity",
                    before=before,
                    after=nonzero[0],
                    context=context,
                )
            )
    if not nonzero:
        return Literal(0)
    if len(nonzero) == 1:
        return nonzero[0]
    return NaryOp("add", tuple(nonzero))


def _simplify_mul(
    children: tuple[Expr, ...],
    *,
    context: RewriteApplicabilityContext,
    applied: list[RewriteEvidence],
) -> Expr:
    flattened: list[Expr] = []
    for child in children:
        if isinstance(child, NaryOp) and child.operator == "mul":
            flattened.extend(child.children)
        else:
            flattened.append(child)
    for child in flattened:
        if _is_literal(child, 0):
            before = NaryOp("mul", tuple(flattened))
            result = Literal(0, unit=getattr(child, "unit", None))
            applied.append(
                validate_rewrite_application(
                    rule_id="zero_product",
                    before=before,
                    after=result,
                    context=context,
                )
            )
            return result
    nonone = [child for child in flattened if not _is_literal(child, 1)]
    for child in flattened:
        if _is_literal(child, 1) and nonone:
            before = NaryOp("mul", (nonone[0], child))
            applied.append(
                validate_rewrite_application(
                    rule_id="multiplicative_identity",
                    before=before,
                    after=nonone[0],
                    context=context,
                )
            )
    if not nonone:
        return Literal(1)
    if len(nonone) == 1:
        return nonone[0]
    return NaryOp("mul", tuple(nonone))


def _common_division_factor(expression: BinaryOp) -> Expr | None:
    numerator = expression.left
    denominator = expression.right
    denominator_key = expression_canonical_json(denominator)
    if isinstance(numerator, BinaryOp) and numerator.operator == "mul":
        factors = (numerator.left, numerator.right)
    elif isinstance(numerator, NaryOp) and numerator.operator == "mul":
        factors = numerator.children
    else:
        return None
    remaining = [
        factor
        for factor in factors
        if expression_canonical_json(factor) != denominator_key
    ]
    if len(remaining) != len(factors) - 1:
        return None
    if len(remaining) == 1:
        return remaining[0]
    return NaryOp("mul", tuple(remaining))


def _verify_equivalence(
    *,
    original: Expr,
    simplified: Expr,
    assumptions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    verified = are_equivalent(original, simplified, assumptions=assumptions)
    if not verified:
        raise ContractValidationError(
            code="unsafe_rewrite_rejected",
            message="rewritten expression could not be verified equivalent",
            field_path="expression",
            details={"reason_code": "equivalence_not_verified"},
        )
    return {
        "status": "verified",
        "verification_method": "sympy_simplify_under_declared_assumptions",
        "assumptions": {str(key): dict(value) for key, value in assumptions.items()},
    }


def _is_literal(expression: Expr, value: int | float) -> bool:
    return isinstance(expression, Literal) and expression.value == value


__all__ = ["SimplificationResult", "simplify_expression_with_trace"]
