from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from euclid.contracts.errors import ContractValidationError

CLAIM_LANE_DESCRIPTIVE = "descriptive_structure"
CLAIM_LANE_PREDICTIVE = "predictive_within_declared_scope"
CLAIM_LANE_INVARIANT = "invariant_predictive_law"
CLAIM_LANE_STOCHASTIC = "stochastic_law"
CLAIM_LANE_MECHANISTIC = "mechanistically_compatible_law"
CLAIM_LANE_TRANSPORT = "transport_supported_law"
CLAIM_LANE_ABSTAINED = "abstained"
CLAIM_LANE_DOWNGRADED = "downgraded"

PRODUCTION_CLAIM_LANES = frozenset(
    {
        CLAIM_LANE_DESCRIPTIVE,
        CLAIM_LANE_PREDICTIVE,
        CLAIM_LANE_INVARIANT,
        CLAIM_LANE_STOCHASTIC,
        CLAIM_LANE_MECHANISTIC,
        CLAIM_LANE_TRANSPORT,
        CLAIM_LANE_ABSTAINED,
        CLAIM_LANE_DOWNGRADED,
    }
)

LEGACY_CLAIM_LANE_ALIASES = {
    "descriptive_only": CLAIM_LANE_DESCRIPTIVE,
    "predictively_supported": CLAIM_LANE_PREDICTIVE,
    "mechanistically_compatible_hypothesis": CLAIM_LANE_MECHANISTIC,
}

_CLAIM_INTERPRETATION_HISTORY = "historical_structure_summary"
_CLAIM_INTERPRETATION_POINT = "point_forecast_within_declared_validation_scope"
_CLAIM_INTERPRETATION_CROSS_ENTITY_PANEL = (
    "cross_entity_panel_forecast_within_declared_validation_scope"
)
_CLAIM_INTERPRETATION_PROBABILISTIC = (
    "probabilistic_forecast_within_declared_validation_scope"
)
_CLAIM_INTERPRETATION_MECHANISTIC = "mechanism_claim"
_CLAIM_INTERPRETATION_INVARIANT = "invariant_claim"
_CLAIM_INTERPRETATION_TRANSPORT = "transport_claim"
_CLAIM_INTERPRETATION_STOCHASTIC = "stochastic_law_within_declared_scope"
_FORBIDDEN_INTERPRETATIONS = (
    "causal_claim",
    "mechanism_claim",
    "transport_claim",
    "invariant_claim",
    "universal_claim",
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
    invariance_support_status: str | None = None
    transport_support_status: str | None = None
    stochastic_support_status: str | None = None
    downgrade_reason_codes: tuple[str, ...] = ()
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
    invariance_status = str(scorecard_body.get("invariance_status", "not_requested"))
    transport_status = str(scorecard_body.get("transport_status", "not_requested"))
    stochastic_status = str(scorecard_body.get("stochastic_status", "not_requested"))
    requested_claim_lane = _optional_claim_lane(scorecard_body.get("requested_claim_lane"))

    if descriptive_status == "passed":
        lower_claim_type = (
            CLAIM_LANE_PREDICTIVE
            if predictive_status == "passed"
            else CLAIM_LANE_DESCRIPTIVE
        )
        claim_type = lower_claim_type
        allowed_interpretation_codes = [_CLAIM_INTERPRETATION_HISTORY]
        downgrade_reason_codes: list[str] = []
        if lower_claim_type == CLAIM_LANE_PREDICTIVE:
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
                claim_type = CLAIM_LANE_MECHANISTIC
                allowed_interpretation_codes.append(
                    _CLAIM_INTERPRETATION_MECHANISTIC
                )
            claim_type, downgrade_reason_codes = _apply_requested_lane(
                current_claim_type=claim_type,
                requested_claim_lane=requested_claim_lane,
                allowed_interpretation_codes=allowed_interpretation_codes,
                forecast_object_type=forecast_object_type,
                invariance_status=invariance_status,
                transport_status=transport_status,
                stochastic_status=stochastic_status,
            )
        forbidden_interpretation_codes = tuple(
            code
            for code in _FORBIDDEN_INTERPRETATIONS
            if not (
                code == _CLAIM_INTERPRETATION_MECHANISTIC
                and claim_type == CLAIM_LANE_MECHANISTIC
            )
            and not (
                code == _CLAIM_INTERPRETATION_INVARIANT
                and claim_type == CLAIM_LANE_INVARIANT
            )
            and not (
                code == _CLAIM_INTERPRETATION_TRANSPORT
                and claim_type == CLAIM_LANE_TRANSPORT
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
            invariance_support_status=invariance_status,
            transport_support_status=transport_status,
            stochastic_support_status=stochastic_status,
            downgrade_reason_codes=tuple(downgrade_reason_codes),
        )

    return ClaimPublicationDecision(
        publication_mode="abstention_only_publication",
        claim_type=None,
        claim_ceiling=None,
        predictive_support_status=None,
        allowed_interpretation_codes=(),
        forbidden_interpretation_codes=_FORBIDDEN_INTERPRETATIONS,
        mechanistic_support_status=mechanistic_status,
        invariance_support_status=invariance_status,
        transport_support_status=transport_status,
        stochastic_support_status=stochastic_status,
        abstention_type=_resolve_abstention_type(
            descriptive_reason_codes=descriptive_reason_codes
        ),
        abstention_reason_codes=descriptive_reason_codes,
        blocked_ceiling=CLAIM_LANE_DESCRIPTIVE,
    )


def normalize_claim_lane(value: str, *, allow_legacy: bool = False) -> str:
    lane = str(value)
    if lane in PRODUCTION_CLAIM_LANES:
        return lane
    if lane in LEGACY_CLAIM_LANE_ALIASES:
        if allow_legacy:
            return LEGACY_CLAIM_LANE_ALIASES[lane]
        raise ContractValidationError(
            code="legacy_claim_lane_not_production",
            message=(
                f"{lane!r} is a legacy claim-lane alias and may not be used as "
                "production evidence"
            ),
            field_path="claim_lane",
            details={"legacy_claim_lane": lane},
        )
    raise ContractValidationError(
        code="invalid_claim_lane",
        message=f"unsupported claim_lane {lane!r}",
        field_path="claim_lane",
        details={"claim_lane": lane},
    )


def assert_claim_scope_publication(claim_card_body: Mapping[str, Any]) -> None:
    claim_lane = normalize_claim_lane(str(claim_card_body.get("claim_ceiling", "")))
    claim_text = str(claim_card_body.get("claim_text", "")).lower()
    invariance_status = str(claim_card_body.get("invariance_support_status", ""))
    transport_status = str(claim_card_body.get("transport_support_status", ""))
    stochastic_status = str(claim_card_body.get("stochastic_support_status", ""))

    blocked_reasons: list[str] = []
    if claim_lane == CLAIM_LANE_INVARIANT and invariance_status != "passed":
        blocked_reasons.append("invariance_required_for_invariant_claim")
    if claim_lane == CLAIM_LANE_TRANSPORT and transport_status != "passed":
        blocked_reasons.append("transport_required_for_transport_supported_claim")
    if claim_lane == CLAIM_LANE_STOCHASTIC and stochastic_status != "passed":
        blocked_reasons.append("stochastic_evidence_required_for_stochastic_claim")
    if "universal" in claim_text and claim_lane not in {
        CLAIM_LANE_INVARIANT,
        CLAIM_LANE_TRANSPORT,
    }:
        blocked_reasons.append("universal_language_requires_invariance_or_transport")
    if "invariant" in claim_text and claim_lane != CLAIM_LANE_INVARIANT:
        blocked_reasons.append("invariant_language_requires_invariant_lane")
    if "transport" in claim_text and claim_lane != CLAIM_LANE_TRANSPORT:
        blocked_reasons.append("transport_language_requires_transport_lane")
    if blocked_reasons:
        raise ContractValidationError(
            code="claim_scope_overstatement",
            message="claim publication overstates its validated scope",
            field_path="claim_card.claim_text",
            details={
                "claim_lane": claim_lane,
                "reason_codes": _normalized_codes(blocked_reasons),
            },
        )


def _optional_claim_lane(value: Any) -> str | None:
    if value is None:
        return None
    return normalize_claim_lane(str(value))


def _apply_requested_lane(
    *,
    current_claim_type: str,
    requested_claim_lane: str | None,
    allowed_interpretation_codes: list[str],
    forecast_object_type: str,
    invariance_status: str,
    transport_status: str,
    stochastic_status: str,
) -> tuple[str, list[str]]:
    if requested_claim_lane is None:
        return current_claim_type, []
    if requested_claim_lane == CLAIM_LANE_DESCRIPTIVE:
        return CLAIM_LANE_DESCRIPTIVE, []
    if requested_claim_lane == CLAIM_LANE_PREDICTIVE:
        return CLAIM_LANE_PREDICTIVE, []
    if requested_claim_lane == CLAIM_LANE_MECHANISTIC:
        return current_claim_type, (
            []
            if current_claim_type == CLAIM_LANE_MECHANISTIC
            else ["mechanistic_evidence_required_for_mechanistic_claim"]
        )
    if requested_claim_lane == CLAIM_LANE_INVARIANT:
        if invariance_status == "passed":
            allowed_interpretation_codes.append(_CLAIM_INTERPRETATION_INVARIANT)
            return CLAIM_LANE_INVARIANT, []
        return current_claim_type, ["invariance_required_for_invariant_claim"]
    if requested_claim_lane == CLAIM_LANE_TRANSPORT:
        if transport_status == "passed":
            allowed_interpretation_codes.append(_CLAIM_INTERPRETATION_TRANSPORT)
            return CLAIM_LANE_TRANSPORT, []
        return current_claim_type, [
            "transport_required_for_transport_supported_claim"
        ]
    if requested_claim_lane == CLAIM_LANE_STOCHASTIC:
        if forecast_object_type != "point" and stochastic_status == "passed":
            allowed_interpretation_codes.append(_CLAIM_INTERPRETATION_STOCHASTIC)
            return CLAIM_LANE_STOCHASTIC, []
        return current_claim_type, [
            "stochastic_evidence_required_for_stochastic_claim"
        ]
    return current_claim_type, ["unsupported_claim_lane_request"]


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
