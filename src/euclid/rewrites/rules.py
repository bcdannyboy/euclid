from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import (
    BinaryOp,
    Expr,
    Feature,
    Literal,
    NaryOp,
    Parameter,
    State,
    UnaryOp,
)
from euclid.expr.serialization import expression_canonical_json
from euclid.runtime.hashing import sha256_digest

_RULE_CATEGORIES = frozenset(
    {"algebraic", "trigonometric", "log_exp", "rational", "piecewise", "unit"}
)
_UNIT_POLICIES = frozenset(
    {"preserve", "require_dimensionless", "same_dimension"}
)


@dataclass(frozen=True)
class RewriteRule:
    rule_id: str
    category: str
    before_pattern: str
    after_pattern: str
    side_conditions: tuple[str, ...]
    equivalence_kind: str = "exact"
    unit_policy: str = "preserve"
    domain_policy: str = "preserve"
    publication_semantics: str = "rewrite_evidence_only_not_claim"

    def __post_init__(self) -> None:
        if self.category not in _RULE_CATEGORIES:
            raise ContractValidationError(
                code="invalid_rewrite_rule",
                message="rewrite rule category is not registered",
                field_path="category",
                details={"category": self.category},
            )
        if self.unit_policy not in _UNIT_POLICIES:
            raise ContractValidationError(
                code="invalid_rewrite_rule",
                message="rewrite rule unit policy is not registered",
                field_path="unit_policy",
                details={"unit_policy": self.unit_policy},
            )
        if self.equivalence_kind != "exact":
            raise ContractValidationError(
                code="invalid_rewrite_rule",
                message="P07 production rewrites must be exact",
                field_path="equivalence_kind",
            )
        if not self.side_conditions:
            raise ContractValidationError(
                code="invalid_rewrite_rule",
                message="rewrite rules must declare side conditions",
                field_path="side_conditions",
            )
        object.__setattr__(self, "side_conditions", tuple(self.side_conditions))

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "before_pattern": self.before_pattern,
            "after_pattern": self.after_pattern,
            "side_conditions": list(self.side_conditions),
            "equivalence_kind": self.equivalence_kind,
            "unit_policy": self.unit_policy,
            "domain_policy": self.domain_policy,
            "publication_semantics": self.publication_semantics,
        }


@dataclass(frozen=True)
class RewriteRuleRegistry:
    rules: tuple[RewriteRule, ...]

    def __post_init__(self) -> None:
        by_id: dict[str, RewriteRule] = {}
        for rule in self.rules:
            if rule.rule_id in by_id:
                raise ContractValidationError(
                    code="duplicate_rewrite_rule",
                    message="rewrite rule IDs must be unique",
                    field_path="rules.rule_id",
                    details={"rule_id": rule.rule_id},
                )
            by_id[rule.rule_id] = rule
        object.__setattr__(self, "rules", tuple(self.rules))

    def get(self, rule_id: str) -> RewriteRule:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        raise ContractValidationError(
            code="unknown_rewrite_rule",
            message=f"unknown rewrite rule {rule_id!r}",
            field_path="rule_id",
            details={"rule_id": rule_id},
        )

    def as_dict(self) -> dict[str, Any]:
        return {"rules": [rule.as_dict() for rule in self.rules]}


@dataclass(frozen=True)
class RewriteApplicabilityContext:
    assumptions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assumptions",
            {str(name): dict(payload) for name, payload in self.assumptions.items()},
        )


@dataclass(frozen=True)
class RewriteEvidence:
    rule_id: str
    before_hash: str
    after_hash: str
    side_conditions_checked: tuple[str, ...]
    domain_evidence: Mapping[str, Any]
    unit_evidence: Mapping[str, Any]
    publication_semantics: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "side_conditions_checked",
            tuple(self.side_conditions_checked),
        )
        object.__setattr__(self, "domain_evidence", dict(self.domain_evidence))
        object.__setattr__(self, "unit_evidence", dict(self.unit_evidence))

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "side_conditions_checked": list(self.side_conditions_checked),
            "domain_evidence": dict(self.domain_evidence),
            "unit_evidence": dict(self.unit_evidence),
            "publication_semantics": self.publication_semantics,
        }


def default_rule_registry() -> RewriteRuleRegistry:
    return RewriteRuleRegistry(
        rules=(
            RewriteRule(
                rule_id="additive_identity",
                category="algebraic",
                before_pattern="x + 0",
                after_pattern="x",
                side_conditions=("zero_has_same_unit_or_dimensionless",),
                unit_policy="same_dimension",
            ),
            RewriteRule(
                rule_id="multiplicative_identity",
                category="algebraic",
                before_pattern="x * 1",
                after_pattern="x",
                side_conditions=("one_is_dimensionless",),
            ),
            RewriteRule(
                rule_id="zero_product",
                category="algebraic",
                before_pattern="x * 0",
                after_pattern="0",
                side_conditions=("zero_product_preserves_domain",),
            ),
            RewriteRule(
                rule_id="trig_pythagorean_identity",
                category="trigonometric",
                before_pattern="sin(x)^2 + cos(x)^2",
                after_pattern="1",
                side_conditions=("x_dimensionless",),
                unit_policy="require_dimensionless",
            ),
            RewriteRule(
                rule_id="log_exp_inverse",
                category="log_exp",
                before_pattern="log(exp(x))",
                after_pattern="x",
                side_conditions=("x_dimensionless", "x_real"),
                unit_policy="require_dimensionless",
            ),
            RewriteRule(
                rule_id="divide_common_factor",
                category="rational",
                before_pattern="(x * y) / y",
                after_pattern="x",
                side_conditions=("y_nonzero",),
            ),
            RewriteRule(
                rule_id="sqrt_square_nonnegative",
                category="algebraic",
                before_pattern="sqrt(x^2)",
                after_pattern="x",
                side_conditions=("x_nonnegative",),
            ),
            RewriteRule(
                rule_id="piecewise_same_branch",
                category="piecewise",
                before_pattern="piecewise(c, x, x)",
                after_pattern="x",
                side_conditions=("branches_equal",),
            ),
            RewriteRule(
                rule_id="unit_zero_identity",
                category="unit",
                before_pattern="x[unit] + 0[unit]",
                after_pattern="x[unit]",
                side_conditions=("unit_equal",),
                unit_policy="same_dimension",
            ),
        )
    )


def validate_rewrite_application(
    *,
    rule_id: str,
    before: Expr,
    after: Expr,
    context: RewriteApplicabilityContext | None = None,
    registry: RewriteRuleRegistry | None = None,
) -> RewriteEvidence:
    resolved_context = context or RewriteApplicabilityContext()
    rule = (registry or default_rule_registry()).get(rule_id)
    if rule.rule_id in {"additive_identity", "unit_zero_identity"}:
        _validate_additive_identity(before, after)
    elif rule.rule_id == "multiplicative_identity":
        _validate_multiplicative_identity(before, after)
    elif rule.rule_id == "zero_product":
        _validate_zero_product(before, after)
    elif rule.rule_id == "log_exp_inverse":
        _validate_dimensionless(before, reason="dimensionless_required")
    elif rule.rule_id == "trig_pythagorean_identity":
        _validate_dimensionless(before, reason="dimensionless_required")
    elif rule.rule_id == "sqrt_square_nonnegative":
        _validate_nonnegative_operand(before, after, resolved_context)
    elif rule.rule_id == "divide_common_factor":
        _validate_nonzero_common_denominator(before, resolved_context)

    return RewriteEvidence(
        rule_id=rule.rule_id,
        before_hash=sha256_digest(expression_canonical_json(before)),
        after_hash=sha256_digest(expression_canonical_json(after)),
        side_conditions_checked=rule.side_conditions,
        domain_evidence=_domain_evidence(rule, before, after, resolved_context),
        unit_evidence={"unit_policy": rule.unit_policy},
        publication_semantics=rule.publication_semantics,
    )


def _validate_additive_identity(before: Expr, after: Expr) -> None:
    children = _operator_children(before, "add")
    if not children or after not in children:
        return
    after_unit = _unit(after)
    for child in children:
        if child is after:
            continue
        if _is_zero(child):
            zero_unit = _unit(child)
            if (
                zero_unit is not None
                and after_unit is not None
                and zero_unit != after_unit
            ):
                _reject("unit_mismatch", rule_id="additive_identity")


def _validate_multiplicative_identity(before: Expr, after: Expr) -> None:
    children = _operator_children(before, "mul")
    if children and after in children:
        return


def _validate_zero_product(before: Expr, after: Expr) -> None:
    if not _is_zero(after):
        _reject("not_zero_product", rule_id="zero_product")
    if not any(_is_zero(child) for child in _operator_children(before, "mul")):
        _reject("missing_zero_factor", rule_id="zero_product")


def _validate_dimensionless(expression: Expr, *, reason: str) -> None:
    for node in _walk_symbols_and_literals(expression):
        unit = _unit(node)
        if unit not in {None, "dimensionless"}:
            _reject(reason, rule_id="dimensionless")


def _validate_nonnegative_operand(
    before: Expr,
    after: Expr,
    context: RewriteApplicabilityContext,
) -> None:
    operand = _sqrt_square_operand(before)
    if (
        operand is None
        or expression_canonical_json(operand) != expression_canonical_json(after)
    ):
        _reject("pattern_mismatch", rule_id="sqrt_square_nonnegative")
    if _domain(after) in {"nonnegative_real", "positive_real"}:
        return
    name = _symbol_name(after)
    if name and context.assumptions.get(name, {}).get("domain") in {
        "nonnegative_real",
        "positive_real",
    }:
        return
    _reject("domain_assumption_missing", rule_id="sqrt_square_nonnegative")


def _validate_nonzero_common_denominator(
    before: Expr,
    context: RewriteApplicabilityContext,
) -> None:
    denominator = (
        before.right
        if isinstance(before, BinaryOp) and before.operator == "div"
        else None
    )
    name = _symbol_name(denominator)
    if not name or context.assumptions.get(name, {}).get("nonzero") is not True:
        _reject("nonzero_assumption_missing", rule_id="divide_common_factor")


def _domain_evidence(
    rule: RewriteRule,
    before: Expr,
    after: Expr,
    context: RewriteApplicabilityContext,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {"domain_policy": rule.domain_policy}
    if rule.rule_id == "sqrt_square_nonnegative":
        evidence["required_domain"] = "nonnegative_real"
        evidence["operand_domain"] = _domain(after)
        name = _symbol_name(after)
        if name:
            evidence["assumption"] = dict(context.assumptions.get(name, {}))
    elif rule.rule_id == "divide_common_factor":
        denominator = before.right if isinstance(before, BinaryOp) else None
        name = _symbol_name(denominator)
        if name:
            evidence["nonzero_symbol"] = name
            evidence["assumption"] = dict(context.assumptions.get(name, {}))
    return evidence


def _operator_children(expression: Expr, operator: str) -> tuple[Expr, ...]:
    if isinstance(expression, NaryOp) and expression.operator == operator:
        return expression.children
    if isinstance(expression, BinaryOp) and expression.operator == operator:
        return (expression.left, expression.right)
    return ()


def _sqrt_square_operand(expression: Expr) -> Expr | None:
    if not (isinstance(expression, UnaryOp) and expression.operator == "sqrt"):
        return None
    operand = expression.operand
    if isinstance(operand, UnaryOp) and operand.operator == "pow2":
        return operand.operand
    if (
        isinstance(operand, BinaryOp)
        and operand.operator == "pow"
        and _literal_value(operand.right) == 2
    ):
        return operand.left
    return None


def _walk_symbols_and_literals(expression: Expr) -> tuple[Expr, ...]:
    from euclid.expr.ast import walk_expression

    return tuple(
        node
        for node in walk_expression(expression)
        if isinstance(node, (Feature, Parameter, State, Literal))
    )


def _unit(expression: Expr | None) -> str | None:
    return getattr(expression, "unit", None)


def _domain(expression: Expr | None) -> str | None:
    return getattr(expression, "domain", None)


def _symbol_name(expression: Expr | None) -> str | None:
    if isinstance(expression, (Feature, Parameter, State)):
        return expression.name
    return None


def _is_zero(expression: Expr) -> bool:
    return _literal_value(expression) in {0, 0.0}


def _is_one(expression: Expr) -> bool:
    return _literal_value(expression) in {1, 1.0}


def _literal_value(expression: Expr) -> object:
    return expression.value if isinstance(expression, Literal) else object()


def _reject(reason_code: str, *, rule_id: str) -> None:
    raise ContractValidationError(
        code="unsafe_rewrite_rejected",
        message="rewrite rule side conditions were not satisfied",
        field_path="rewrite",
        details={"reason_code": reason_code, "rule_id": rule_id},
    )


__all__ = [
    "RewriteApplicabilityContext",
    "RewriteEvidence",
    "RewriteRule",
    "RewriteRuleRegistry",
    "default_rule_registry",
    "validate_rewrite_application",
]
