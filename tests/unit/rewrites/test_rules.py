from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import BinaryOp, Feature, Literal, NaryOp, UnaryOp
from euclid.rewrites.rules import (
    RewriteApplicabilityContext,
    default_rule_registry,
    validate_rewrite_application,
)


def test_default_rule_registry_declares_assumptions_side_conditions_and_units() -> None:
    registry = default_rule_registry()

    expected_rules = {
        "additive_identity",
        "multiplicative_identity",
        "zero_product",
        "trig_pythagorean_identity",
        "log_exp_inverse",
        "divide_common_factor",
        "sqrt_square_nonnegative",
    }

    assert expected_rules <= {rule.rule_id for rule in registry.rules}
    for rule in registry.rules:
        assert rule.category in {
            "algebraic",
            "trigonometric",
            "log_exp",
            "rational",
            "piecewise",
            "unit",
        }
        assert rule.equivalence_kind == "exact"
        assert rule.side_conditions
        assert rule.unit_policy in {
            "preserve",
            "require_dimensionless",
            "same_dimension",
        }
        assert rule.publication_semantics == "rewrite_evidence_only_not_claim"


def test_safe_algebraic_identity_is_validated_with_unit_evidence() -> None:
    x = Feature("x", unit="meter")
    original = NaryOp("add", (x, Literal(0, unit="meter")))

    evidence = validate_rewrite_application(
        rule_id="additive_identity",
        before=original,
        after=x,
        context=RewriteApplicabilityContext(),
    )

    assert evidence.rule_id == "additive_identity"
    assert evidence.unit_evidence["unit_policy"] == "same_dimension"
    assert evidence.domain_evidence["domain_policy"] == "preserve"
    assert evidence.publication_semantics == "rewrite_evidence_only_not_claim"


def test_unit_changing_rewrite_is_rejected() -> None:
    x = Feature("x", unit="meter")
    original = NaryOp("add", (x, Literal(0, unit="second")))

    with pytest.raises(ContractValidationError) as excinfo:
        validate_rewrite_application(
            rule_id="additive_identity",
            before=original,
            after=x,
            context=RewriteApplicabilityContext(),
        )

    assert excinfo.value.code == "unsafe_rewrite_rejected"
    assert excinfo.value.details["reason_code"] == "unit_mismatch"


def test_domain_restricted_rewrite_requires_declared_assumptions() -> None:
    x = Feature("x", domain="real")
    original = UnaryOp("sqrt", UnaryOp("pow2", x))

    with pytest.raises(ContractValidationError) as excinfo:
        validate_rewrite_application(
            rule_id="sqrt_square_nonnegative",
            before=original,
            after=x,
            context=RewriteApplicabilityContext(),
        )
    assert excinfo.value.details["reason_code"] == "domain_assumption_missing"

    evidence = validate_rewrite_application(
        rule_id="sqrt_square_nonnegative",
        before=original,
        after=x,
        context=RewriteApplicabilityContext(
            assumptions={"x": {"domain": "nonnegative_real"}}
        ),
    )
    assert evidence.domain_evidence["required_domain"] == "nonnegative_real"


def test_unsafe_rational_rewrite_rejects_missing_nonzero_denominator() -> None:
    x = Feature("x")
    y = Feature("y")
    original = BinaryOp("div", BinaryOp("mul", x, y), y)

    with pytest.raises(ContractValidationError) as excinfo:
        validate_rewrite_application(
            rule_id="divide_common_factor",
            before=original,
            after=x,
            context=RewriteApplicabilityContext(),
        )

    assert excinfo.value.code == "unsafe_rewrite_rejected"
    assert excinfo.value.details["reason_code"] == "nonzero_assumption_missing"
