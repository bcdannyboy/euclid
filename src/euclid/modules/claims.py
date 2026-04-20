from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

_CLAIM_INTERPRETATION_HISTORY = "historical_structure_summary"
_CLAIM_INTERPRETATION_POINT = "point_forecast_within_declared_validation_scope"
_CLAIM_INTERPRETATION_CROSS_ENTITY_PANEL = (
    "cross_entity_panel_forecast_within_declared_validation_scope"
)
_CLAIM_INTERPRETATION_PROBABILISTIC = (
    "probabilistic_forecast_within_declared_validation_scope"
)
_CLAIM_INTERPRETATION_MECHANISTIC = "mechanism_claim"
_FORBIDDEN_INTERPRETATIONS = (
    "causal_claim",
    "mechanism_claim",
    "transport_claim",
    "cross_entity_generalization",
    "probabilistic_forecast_claim",
    "calibration_claim",
)
_ROBUSTNESS_REASON_CODES = {
    "robustness_failed",
    "null_protocol_failed",
    "perturbation_protocol_failed",
    "leakage_canary_failed",
}


@dataclass(frozen=True)
class ClaimPublicationDecision:
    publication_mode: str
    claim_type: str | None
    claim_ceiling: str | None
    predictive_support_status: str | None
    allowed_interpretation_codes: tuple[str, ...]
    forbidden_interpretation_codes: tuple[str, ...]
    mechanistic_support_status: str | None = None
    abstention_type: str | None = None
    abstention_reason_codes: tuple[str, ...] = ()
    blocked_ceiling: str | None = None


def resolve_claim_publication(
    *,
    scorecard_body: Mapping[str, Any],
) -> ClaimPublicationDecision:
    descriptive_status = str(scorecard_body["descriptive_status"])
    descriptive_reason_codes = _normalized_codes(
        scorecard_body.get("descriptive_reason_codes", ())
    )
    predictive_status = str(scorecard_body.get("predictive_status", "not_requested"))
    predictive_reason_codes = _normalized_codes(
        scorecard_body.get("predictive_reason_codes", ())
    )
    forecast_object_type = str(scorecard_body.get("forecast_object_type", "point"))
    entity_panel = _normalized_codes(scorecard_body.get("entity_panel", ()))
    mechanistic_status = str(scorecard_body.get("mechanistic_status", "not_requested"))

    if descriptive_status == "passed":
        lower_claim_type = (
            "predictively_supported"
            if predictive_status == "passed"
            else "descriptive_only"
        )
        claim_type = lower_claim_type
        allowed_interpretation_codes = [_CLAIM_INTERPRETATION_HISTORY]
        if lower_claim_type == "predictively_supported":
            if forecast_object_type == "point":
                allowed_interpretation_codes.append(_CLAIM_INTERPRETATION_POINT)
                if len(entity_panel) > 1:
                    allowed_interpretation_codes.append(
                        _CLAIM_INTERPRETATION_CROSS_ENTITY_PANEL
                    )
            else:
                allowed_interpretation_codes.append(
                    _CLAIM_INTERPRETATION_PROBABILISTIC
                )
            if mechanistic_status == "passed":
                claim_type = "mechanistically_compatible_hypothesis"
                allowed_interpretation_codes.append(
                    _CLAIM_INTERPRETATION_MECHANISTIC
                )
        forbidden_interpretation_codes = tuple(
            code
            for code in _FORBIDDEN_INTERPRETATIONS
            if not (
                code == _CLAIM_INTERPRETATION_MECHANISTIC
                and mechanistic_status == "passed"
            )
        )
        return ClaimPublicationDecision(
            publication_mode="candidate_publication",
            claim_type=claim_type,
            claim_ceiling=claim_type,
            predictive_support_status=_predictive_support_status(
                predictive_status=predictive_status,
                predictive_reason_codes=predictive_reason_codes,
            ),
            allowed_interpretation_codes=tuple(allowed_interpretation_codes),
            forbidden_interpretation_codes=forbidden_interpretation_codes,
            mechanistic_support_status=mechanistic_status,
        )

    return ClaimPublicationDecision(
        publication_mode="abstention_only_publication",
        claim_type=None,
        claim_ceiling=None,
        predictive_support_status=None,
        allowed_interpretation_codes=(),
        forbidden_interpretation_codes=_FORBIDDEN_INTERPRETATIONS,
        mechanistic_support_status=mechanistic_status,
        abstention_type=_resolve_abstention_type(
            descriptive_reason_codes=descriptive_reason_codes
        ),
        abstention_reason_codes=descriptive_reason_codes,
        blocked_ceiling="descriptive_only",
    )


def _predictive_support_status(
    *,
    predictive_status: str,
    predictive_reason_codes: tuple[str, ...],
) -> str:
    if predictive_status == "passed":
        return "confirmatory_supported"
    if predictive_status == "exploratory_only":
        return "exploratory_only"
    if predictive_status == "not_requested":
        return "not_requested"
    if "holdout_exhausted" in predictive_reason_codes:
        return "exhausted"
    return "blocked"


def _resolve_abstention_type(
    *,
    descriptive_reason_codes: tuple[str, ...],
) -> str:
    if "codelength_comparability_failed" in descriptive_reason_codes:
        return "codelength_comparability_failed"
    if any(code in _ROBUSTNESS_REASON_CODES for code in descriptive_reason_codes):
        return "robustness_failed"
    return "no_admissible_reducer"


def _normalized_codes(codes: Any) -> tuple[str, ...]:
    if not isinstance(codes, (list, tuple)):
        return ()
    seen: dict[str, None] = {}
    for code in codes:
        text = str(code)
        if text:
            seen.setdefault(text, None)
    return tuple(seen)
