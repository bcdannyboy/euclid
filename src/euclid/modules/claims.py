from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from euclid.contracts.errors import ContractValidationError
from euclid.modules.conformal import assert_conformal_claim_scope

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
_MDL_ELIGIBLE_CODELENGTH_TIERS = frozenset(
    {
        "exact_prequential_symbol_code",
        "mdl_based_universal_code",
    }
)
_UNIVERSAL_CODING_ELIGIBLE_CODELENGTH_TIERS = frozenset(
    {
        "mdl_based_universal_code",
    }
)
_VALID_GIVEN_REGIME_SCOPE = "valid_given_regime"
_VALID_GIVEN_STATE_SCOPE = "valid_given_state"


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


def build_scorecard_body(
    *,
    scorecard_id: str,
    candidate_ref: Any,
    point_score_policy_ref: Any,
    point_score_result_ref: Any,
    calibration_contract_ref: Any,
    calibration_result_ref: Any,
    evaluation_plan_ref: Any,
    baseline_registry_ref: Any,
    forecast_comparison_policy_ref: Any,
    comparison_universe_ref: Any,
    evaluation_event_log_ref: Any,
    evaluation_governance_ref: Any,
    predictive_gate_policy_ref: Any,
    null_protocol_ref: Any,
    perturbation_protocol_ref: Any,
    robustness_report_ref: Any,
    time_safety_audit_ref: Any,
    scorecard_decision: Any,
    description_gain_bits: float | int | None = None,
    forecast_object_type: str | None = None,
    entity_panel: Sequence[str] = (),
    observation_model_ref: Any | None = None,
    canonical_structure_code_ref: Any | None = None,
    target_transform_ref: Any | None = None,
    base_measure_policy_ref: Any | None = None,
    codelength_policy_ref: Any | None = None,
    reference_description_policy_ref: Any | None = None,
    description_components: Mapping[str, float | int] | None = None,
    prediction_artifact_ref: Any | None = None,
    residual_history_refs: Sequence[Any] = (),
    stochastic_model_refs: Sequence[Any] = (),
    stochastic_evidence_status: str | None = None,
    stochastic_evidence_reason_codes: Sequence[str] = (),
    stochastic_status: str | None = None,
    stochastic_reason_codes: Sequence[str] = (),
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "scorecard_id": str(scorecard_id),
        "candidate_ref": _ref_payload(candidate_ref, field_path="candidate_ref"),
    }
    optional_refs = {
        "observation_model_ref": observation_model_ref,
        "canonical_structure_code_ref": canonical_structure_code_ref,
        "target_transform_ref": target_transform_ref,
        "base_measure_policy_ref": base_measure_policy_ref,
        "codelength_policy_ref": codelength_policy_ref,
        "reference_description_policy_ref": reference_description_policy_ref,
    }
    for field_name, ref in optional_refs.items():
        if ref is not None:
            body[field_name] = _ref_payload(ref, field_path=field_name)

    if description_components is not None:
        for field_name in (
            "L_family_bits",
            "L_structure_bits",
            "L_literals_bits",
            "L_params_bits",
            "L_state_bits",
            "L_data_bits",
            "L_total_bits",
            "reference_bits",
        ):
            if field_name in description_components:
                body[field_name] = float(description_components[field_name])
    if description_gain_bits is not None:
        body["description_gain_bits"] = float(description_gain_bits)

    body.update(
        {
            "point_score_policy_ref": _ref_payload(
                point_score_policy_ref,
                field_path="point_score_policy_ref",
            ),
            "point_score_result_ref": _ref_payload(
                point_score_result_ref,
                field_path="point_score_result_ref",
            ),
            "calibration_contract_ref": _ref_payload(
                calibration_contract_ref,
                field_path="calibration_contract_ref",
            ),
            "calibration_result_ref": _ref_payload(
                calibration_result_ref,
                field_path="calibration_result_ref",
            ),
            "evaluation_plan_ref": _ref_payload(
                evaluation_plan_ref,
                field_path="evaluation_plan_ref",
            ),
            "baseline_registry_ref": _ref_payload(
                baseline_registry_ref,
                field_path="baseline_registry_ref",
            ),
            "forecast_comparison_policy_ref": _ref_payload(
                forecast_comparison_policy_ref,
                field_path="forecast_comparison_policy_ref",
            ),
            "comparison_universe_ref": _ref_payload(
                comparison_universe_ref,
                field_path="comparison_universe_ref",
            ),
            "evaluation_event_log_ref": _ref_payload(
                evaluation_event_log_ref,
                field_path="evaluation_event_log_ref",
            ),
            "evaluation_governance_ref": _ref_payload(
                evaluation_governance_ref,
                field_path="evaluation_governance_ref",
            ),
            "predictive_gate_policy_ref": _ref_payload(
                predictive_gate_policy_ref,
                field_path="predictive_gate_policy_ref",
            ),
            "null_protocol_ref": _ref_payload(
                null_protocol_ref,
                field_path="null_protocol_ref",
            ),
            "perturbation_protocol_ref": _ref_payload(
                perturbation_protocol_ref,
                field_path="perturbation_protocol_ref",
            ),
            "robustness_report_ref": _ref_payload(
                robustness_report_ref,
                field_path="robustness_report_ref",
            ),
            "time_safety_audit_ref": _ref_payload(
                time_safety_audit_ref,
                field_path="time_safety_audit_ref",
            ),
            "descriptive_status": str(scorecard_decision.descriptive_status),
            "descriptive_reason_codes": list(
                _normalized_codes(scorecard_decision.descriptive_reason_codes)
            ),
            "predictive_status": str(scorecard_decision.predictive_status),
            "predictive_reason_codes": list(
                _normalized_codes(scorecard_decision.predictive_reason_codes)
            ),
        }
    )
    if hasattr(scorecard_decision, "mechanistic_status"):
        body["mechanistic_status"] = str(scorecard_decision.mechanistic_status)
        body["mechanistic_reason_codes"] = list(
            _normalized_codes(
                getattr(scorecard_decision, "mechanistic_reason_codes", ())
            )
        )
    if forecast_object_type is not None:
        body["forecast_object_type"] = str(forecast_object_type)
    if entity_panel:
        body["entity_panel"] = [str(entity) for entity in entity_panel]
    if prediction_artifact_ref is not None:
        body["prediction_artifact_ref"] = _ref_payload(
            prediction_artifact_ref,
            field_path="prediction_artifact_ref",
        )
    if residual_history_refs:
        body["residual_history_refs"] = [
            _ref_payload(ref, field_path="residual_history_refs")
            for ref in residual_history_refs
        ]
    if stochastic_model_refs:
        body["stochastic_model_refs"] = [
            _ref_payload(ref, field_path="stochastic_model_refs")
            for ref in stochastic_model_refs
        ]
    if stochastic_evidence_status is not None:
        body["stochastic_evidence_status"] = str(stochastic_evidence_status)
        body["stochastic_evidence_reason_codes"] = list(
            _normalized_codes(stochastic_evidence_reason_codes)
        )
    if stochastic_status is not None:
        body["stochastic_status"] = str(stochastic_status)
        body["stochastic_reason_codes"] = list(
            _normalized_codes(stochastic_reason_codes)
        )
    if extra_fields:
        body.update(extra_fields)
    return body


def build_claim_card_body(
    *,
    claim_card_id: str,
    candidate_ref: Any,
    scorecard_ref: Any,
    validation_scope_ref: Any,
    claim_decision: ClaimPublicationDecision,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if claim_decision.publication_mode != "candidate_publication":
        raise ContractValidationError(
            code="claim_card_requires_candidate_publication_decision",
            message="claim-card body requires a candidate publication decision",
            field_path="claim_decision.publication_mode",
            details={"publication_mode": claim_decision.publication_mode},
        )
    if claim_decision.claim_type is None or claim_decision.claim_ceiling is None:
        raise ContractValidationError(
            code="claim_card_requires_claim_scope",
            message="claim-card body requires claim type and ceiling",
            field_path="claim_decision.claim_ceiling",
        )

    body = _build_claim_card_base_body(
        claim_card_id=claim_card_id,
        candidate_ref=candidate_ref,
        scorecard_ref=scorecard_ref,
        validation_scope_ref=validation_scope_ref,
        claim_decision=claim_decision,
    )
    body.update(
        {
            "invariance_support_status": claim_decision.invariance_support_status,
            "transport_support_status": claim_decision.transport_support_status,
            "stochastic_support_status": claim_decision.stochastic_support_status,
            "downgrade_reason_codes": list(claim_decision.downgrade_reason_codes),
        }
    )
    if extra_fields:
        body.update(extra_fields)
    return body


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
    coding_claim_tier = _claim_card_coding_claim_tier(claim_card_body)
    invariance_status = str(claim_card_body.get("invariance_support_status", ""))
    transport_status = str(claim_card_body.get("transport_support_status", ""))
    stochastic_status = str(claim_card_body.get("stochastic_support_status", ""))

    assert_conformal_claim_scope(claim_card_body)

    blocked_reasons: list[str] = list(
        _nonstationarity_claim_scope_reason_codes(
            claim_card_body=claim_card_body,
            claim_text=claim_text,
            claim_lane=claim_lane,
        )
    )
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
    if "stochastic" in claim_text and claim_lane != CLAIM_LANE_STOCHASTIC:
        blocked_reasons.append("stochastic_language_requires_stochastic_lane")
    if _uses_mdl_language(claim_text) and (
        coding_claim_tier not in _MDL_ELIGIBLE_CODELENGTH_TIERS
    ):
        blocked_reasons.append("mdl_language_requires_eligible_codelength_claim_tier")
    if _uses_universal_coding_language(claim_text) and (
        coding_claim_tier not in _UNIVERSAL_CODING_ELIGIBLE_CODELENGTH_TIERS
    ):
        blocked_reasons.append(
            "universal_coding_language_requires_universal_codelength_claim_tier"
        )
    if blocked_reasons:
        raise ContractValidationError(
            code="claim_scope_overstatement",
            message="claim publication overstates its validated scope",
            field_path="claim_card.claim_text",
            details={
                "claim_lane": claim_lane,
                "coding_claim_tier": coding_claim_tier,
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


def _uses_mdl_language(claim_text: str) -> bool:
    return (
        "minimum description length" in claim_text
        or "minimum-description-length" in claim_text
        or "mdl" in claim_text.split()
        or "mdl-" in claim_text
        or "mdl_" in claim_text
    )


def _uses_universal_coding_language(claim_text: str) -> bool:
    return (
        "universal coding" in claim_text
        or "universal code" in claim_text
        or "universal codelength" in claim_text
    )


def _nonstationarity_claim_scope_reason_codes(
    *,
    claim_card_body: Mapping[str, Any],
    claim_text: str,
    claim_lane: str,
) -> tuple[str, ...]:
    reason_codes: list[str] = []
    stability = _mapping_or_none(claim_card_body.get("stability_diagnostic_artifact"))
    change_point = _mapping_or_none(claim_card_body.get("change_point_artifact"))
    regime_switching = _mapping_or_none(
        claim_card_body.get("regime_switching_artifact")
    )
    state_space = _mapping_or_none(claim_card_body.get("state_space_artifact"))
    if stability is not None:
        if _nonstationarity_artifact_is_instability_evidence(
            stability
        ) and _uses_law_language(claim_text):
            reason_codes.append("instability_evidence_cannot_be_law_claim")
        if _nonstationarity_artifact_is_diagnostic_only(stability) and _uses_law_language(
            claim_text
        ):
            reason_codes.append("stability_diagnostic_is_diagnostic_only")
        if _unresolved_instability(stability) and _uses_stationary_law_language(
            claim_text
        ):
            reason_codes.append(
                "unresolved_instability_blocks_stationary_law_claim"
            )
    if regime_switching is not None:
        if _unresolved_instability(
            regime_switching
        ) and _uses_stationary_law_language(claim_text):
            reason_codes.append(
                "regime_switching_instability_blocks_stationary_law_claim"
            )
        if _regime_scoped_artifact(regime_switching):
            if not _artifact_has_valid_given_regime_scope(
                regime_switching,
                claim_text=claim_text,
            ):
                reason_codes.append(
                    "regime_scoped_evidence_requires_valid_given_regime_scope"
                )
            if claim_lane == CLAIM_LANE_INVARIANT or _uses_stationary_law_language(
                claim_text
            ):
                reason_codes.append(
                    "regime_scoped_evidence_cannot_support_stationary_law_claim"
                )
    if state_space is not None:
        if _unresolved_instability(state_space) and _uses_stationary_law_language(
            claim_text
        ):
            reason_codes.append(
                "state_space_instability_blocks_stationary_law_claim"
            )
        if _state_scoped_artifact(state_space):
            if not _artifact_has_valid_given_state_scope(
                state_space,
                claim_text=claim_text,
            ):
                reason_codes.append(
                    "state_scoped_evidence_requires_valid_given_state_scope"
                )
            if claim_lane == CLAIM_LANE_INVARIANT or _uses_stationary_law_language(
                claim_text
            ):
                reason_codes.append(
                    "state_scoped_evidence_cannot_support_stationary_law_claim"
                )
    if change_point is not None and _change_point_requires_scoped_wording(
        change_point
    ):
        if _uses_stationary_law_language(claim_text) or "universal" in claim_text:
            reason_codes.append(
                "change_point_artifact_requires_scoped_nonstationary_wording"
            )
    if _uses_stationary_law_language(claim_text):
        if _has_regime_conditioned_evidence(
            claim_card_body
        ) and not _contains_claim_scope(
            claim_card_body,
            _VALID_GIVEN_REGIME_SCOPE,
        ):
            reason_codes.append(
                "regime_conditioned_evidence_requires_valid_given_regime_scope"
            )
        if _has_state_space_evidence(claim_card_body) and not _contains_claim_scope(
            claim_card_body,
            _VALID_GIVEN_STATE_SCOPE,
        ):
            if _state_space_whiteness_failure_unresolved(claim_card_body):
                reason_codes.append(
                    "state_space_whiteness_failure_blocks_unscoped_stationary_claim"
                )
            else:
                reason_codes.append(
                    "state_space_evidence_requires_valid_given_state_scope"
                )
    return _normalized_codes(reason_codes)


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _nonstationarity_artifact_is_diagnostic_only(
    artifact: Mapping[str, Any],
) -> bool:
    return (
        str(artifact.get("evidence_role", "")) == "diagnostic_only"
        or str(artifact.get("claim_scope", "")) in {
            "diagnostic_evidence_only",
            "diagnostic_only",
            "not_a_law_claim",
        }
        or artifact.get("law_claim_allowed") is False
        or artifact.get("is_law_claim") is False
    )


def _nonstationarity_artifact_is_instability_evidence(
    artifact: Mapping[str, Any],
) -> bool:
    return (
        str(artifact.get("evidence_role", "")) == "instability_evidence"
        or str(artifact.get("claim_scope", "")) == "instability_evidence_only"
    )


def _unresolved_instability(artifact: Mapping[str, Any]) -> bool:
    if artifact.get("resolved_by_later_lane") is True:
        return False
    if artifact.get("handled_by_nonstationary_lane") is True:
        return False
    handling = artifact.get("nonstationarity_handling")
    if isinstance(handling, Mapping) and str(handling.get("status", "")) == "passed":
        return False
    reason_codes = {str(code) for code in artifact.get("reason_codes", ())}
    return str(artifact.get("status", "")) in {
        "detected",
        "failed",
        "nonstationary",
        "unstable",
    } or bool(
        reason_codes
        & {
            "instability_evidence_unresolved",
            "recursive_residual_instability_detected",
            "regime_switching_instability_detected",
            "stability_test_failed",
            "state_space_instability_unresolved",
            "structural_break_detected",
        }
    )


def _regime_scoped_artifact(artifact: Mapping[str, Any]) -> bool:
    return (
        _artifact_schema_contains(artifact, "regime_switching")
        or "regime_id" in artifact
        or "regime_ids" in artifact
        or str(artifact.get("claim_scope", "")) == _VALID_GIVEN_REGIME_SCOPE
        or "valid_given_regime" in artifact
    )


def _state_scoped_artifact(artifact: Mapping[str, Any]) -> bool:
    return (
        _artifact_schema_contains(artifact, "state_space")
        or "state_id" in artifact
        or str(artifact.get("claim_scope", "")) == _VALID_GIVEN_STATE_SCOPE
        or "valid_given_state" in artifact
    )


def _artifact_schema_contains(artifact: Mapping[str, Any], text: str) -> bool:
    descriptor_fields = (
        artifact.get("schema_name"),
        artifact.get("artifact_type"),
        artifact.get("method"),
        artifact.get("lane_id"),
        artifact.get("model_class"),
    )
    return any(text in str(field).lower() for field in descriptor_fields)


def _artifact_has_valid_given_regime_scope(
    artifact: Mapping[str, Any],
    *,
    claim_text: str,
) -> bool:
    return _contains_claim_scope(artifact, _VALID_GIVEN_REGIME_SCOPE) or (
        _claim_text_has_valid_given_scope(claim_text, "regime")
    )


def _artifact_has_valid_given_state_scope(
    artifact: Mapping[str, Any],
    *,
    claim_text: str,
) -> bool:
    return _contains_claim_scope(artifact, _VALID_GIVEN_STATE_SCOPE) or (
        _claim_text_has_valid_given_scope(claim_text, "state")
    )


def _claim_text_has_valid_given_scope(claim_text: str, scope_token: str) -> bool:
    return (
        ("valid_given" in claim_text or "valid given" in claim_text)
        and scope_token in claim_text
    )


def _change_point_requires_scoped_wording(artifact: Mapping[str, Any]) -> bool:
    if artifact.get("resolved_by_later_lane") is True:
        return False
    return str(artifact.get("status", "")) in {"passed", "detected", "failed"}


def _uses_law_language(claim_text: str) -> bool:
    return any(
        token in claim_text
        for token in (
            "law",
            "universal",
            "invariant",
            "stationary",
        )
    )


def _uses_stationary_law_language(claim_text: str) -> bool:
    return (
        "stationary" in claim_text
        or "universal" in claim_text
        or "all regimes" in claim_text
        or "all states" in claim_text
    ) and "nonstationary" not in claim_text


def _has_regime_conditioned_evidence(value: Any) -> bool:
    if isinstance(value, Mapping):
        operator_id = str(value.get("operator_id", "")).lower()
        composition_operator = str(value.get("composition_operator", "")).lower()
        if operator_id == "regime_conditioned" or (
            composition_operator == "regime_conditioned"
        ):
            return True
        return any(_has_regime_conditioned_evidence(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_regime_conditioned_evidence(child) for child in value)
    return False


def _has_state_space_evidence(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any("state_space" in str(key).lower() for key in value):
            return True
        descriptor_fields = (
            value.get("schema_name"),
            value.get("lane_id"),
            value.get("artifact_type"),
            value.get("model_class"),
        )
        if any("state_space" in str(field).lower() for field in descriptor_fields):
            return True
        return any(_has_state_space_evidence(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_state_space_evidence(child) for child in value)
    return False


def _contains_claim_scope(value: Any, expected_scope: str) -> bool:
    if isinstance(value, Mapping):
        for field_name in (
            "claim_scope",
            "supported_claim_scope",
            "claim_scopes",
            "supported_claim_scopes",
        ):
            if _scope_value_matches(value.get(field_name), expected_scope):
                return True
        if expected_scope in value and _scope_payload_is_explicit(
            value.get(expected_scope)
        ):
            return True
        return any(
            _contains_claim_scope(child, expected_scope) for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_claim_scope(child, expected_scope) for child in value)
    return False


def _scope_value_matches(value: Any, expected_scope: str) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value == expected_scope
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_scope_value_matches(item, expected_scope) for item in value)
    return False


def _scope_payload_is_explicit(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, tuple, set, frozenset, Mapping)):
        return bool(value)
    return True


def _state_space_whiteness_failure_unresolved(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("resolved_by_later_lane") is True:
            return False
        if value.get("handled_by_nonstationary_lane") is True:
            return False
        statuses = (
            value.get("innovation_whiteness_status"),
            value.get("whiteness_status"),
            value.get("innovation_whiteness_test_status"),
        )
        if any(
            str(status).lower() in {"failed", "blocked", "rejected", "nonwhite"}
            for status in statuses
            if status is not None
        ):
            return True
        reason_codes = {str(code) for code in value.get("reason_codes", ())}
        if reason_codes & {
            "innovation_whiteness_failed",
            "innovation_whiteness_test_failed",
            "nonwhite_innovations_detected",
            "state_space_innovation_whiteness_failed",
        }:
            return True
        return any(
            _state_space_whiteness_failure_unresolved(child)
            for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_state_space_whiteness_failure_unresolved(child) for child in value)
    return False


def _claim_card_coding_claim_tier(claim_card_body: Mapping[str, Any]) -> str | None:
    direct_tier = claim_card_body.get("coding_claim_tier")
    if direct_tier is not None:
        return str(direct_tier)
    for field_name in (
        "codelength_policy_manifest",
        "codelength_policy",
        "codelength_policy_body",
    ):
        tier = _mapping_coding_claim_tier(claim_card_body.get(field_name))
        if tier is not None:
            return tier
    return None


def _mapping_coding_claim_tier(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    tier = value.get("coding_claim_tier")
    if tier is not None:
        return str(tier)
    body = value.get("body")
    if isinstance(body, Mapping):
        tier = body.get("coding_claim_tier")
        if tier is not None:
            return str(tier)
    return None


def _ref_payload(value: Any, *, field_path: str) -> dict[str, str]:
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    if isinstance(value, Mapping):
        schema_name = value.get("schema_name")
        object_id = value.get("object_id")
        if isinstance(schema_name, str) and isinstance(object_id, str):
            return {"schema_name": schema_name, "object_id": object_id}
    raise ContractValidationError(
        code="typed_ref_payload_required",
        message=f"{field_path} must be a typed reference payload",
        field_path=field_path,
    )


def _build_claim_card_base_body(
    *,
    claim_card_id: str,
    candidate_ref: Any,
    scorecard_ref: Any,
    validation_scope_ref: Any,
    claim_decision: ClaimPublicationDecision,
) -> dict[str, Any]:
    try:
        from euclid.modules.evidence_contracts import build_claim_card_manifest_body
    except ImportError:
        return {
            "claim_card_id": str(claim_card_id),
            "candidate_ref": _ref_payload(candidate_ref, field_path="candidate_ref"),
            "scorecard_ref": _ref_payload(scorecard_ref, field_path="scorecard_ref"),
            "validation_scope_ref": _ref_payload(
                validation_scope_ref,
                field_path="validation_scope_ref",
            ),
            "claim_type": claim_decision.claim_type,
            "claim_ceiling": claim_decision.claim_ceiling,
            "predictive_support_status": claim_decision.predictive_support_status,
            "allowed_interpretation_codes": list(
                claim_decision.allowed_interpretation_codes
            ),
            "forbidden_interpretation_codes": list(
                claim_decision.forbidden_interpretation_codes
            ),
        }
    return dict(
        build_claim_card_manifest_body(
            claim_card_id=str(claim_card_id),
            candidate_ref=candidate_ref,
            scorecard_ref=scorecard_ref,
            validation_scope_ref=validation_scope_ref,
            claim_decision=claim_decision,
        )
    )
