from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import Expr
from euclid.expr.sympy_bridge import from_sympy

_SAFE_SYMPY_GLOBALS = {
    "__builtins__": {},
    "Abs": sp.Abs,
    "Add": sp.Add,
    "Float": sp.Float,
    "Integer": sp.Integer,
    "Max": sp.Max,
    "Min": sp.Min,
    "Mul": sp.Mul,
    "Pow": sp.Pow,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
    "cos": sp.cos,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
    "tanh": sp.tanh,
}


@dataclass(frozen=True)
class ExternalEngineCandidate:
    engine_name: str
    engine_candidate_id: str
    expression_source: str
    feature_names: tuple[str, ...]
    raw_score: float | None = None


@dataclass(frozen=True)
class ClaimBoundary:
    claim_publication_allowed: bool
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_publication_allowed": self.claim_publication_allowed,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class LoweredExternalExpression:
    expression: Expr
    provenance: Mapping[str, Any]
    claim_boundary: ClaimBoundary


def lower_external_engine_candidate(
    candidate: ExternalEngineCandidate,
) -> LoweredExternalExpression:
    local_dict = {name: sp.Symbol(name, real=True) for name in candidate.feature_names}
    parsed = parse_expr(
        candidate.expression_source,
        local_dict=local_dict,
        global_dict=_SAFE_SYMPY_GLOBALS,
        evaluate=True,
    )
    return LoweredExternalExpression(
        expression=from_sympy(parsed),
        provenance={
            "engine_name": candidate.engine_name,
            "engine_candidate_id": candidate.engine_candidate_id,
            "raw_score": candidate.raw_score,
        },
        claim_boundary=ClaimBoundary(
            claim_publication_allowed=False,
            reason_codes=("external_engine_not_claim_authority",),
        ),
    )


def require_euclid_claim_boundary(lowered: LoweredExternalExpression) -> None:
    if lowered.claim_boundary.claim_publication_allowed:
        return
    raise ContractValidationError(
        code="external_engine_claim_boundary",
        message=(
            "external engine output must pass Euclid CIR, fitting, scoring, replay, "
            "and publication gates before any claim can be published"
        ),
        field_path="claim_boundary",
        details={
            "reason_codes": list(lowered.claim_boundary.reason_codes),
            "engine_name": lowered.provenance.get("engine_name"),
        },
    )


__all__ = [
    "ClaimBoundary",
    "ExternalEngineCandidate",
    "LoweredExternalExpression",
    "lower_external_engine_candidate",
    "require_euclid_claim_boundary",
]
