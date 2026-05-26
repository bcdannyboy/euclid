from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.modules.claims import (
    assert_claim_scope_publication,
    build_claim_card_body,
    build_scorecard_body,
    resolve_claim_publication,
)
from euclid.modules.gate_lifecycle import ScorecardStatusDecision


def _ref(schema_name: str, object_id: str) -> TypedRef:
    return TypedRef(schema_name=schema_name, object_id=object_id)


def test_build_scorecard_body_preserves_required_manifest_shape() -> None:
    decision = ScorecardStatusDecision(
        descriptive_status="passed",
        descriptive_reason_codes=(),
        predictive_status="blocked",
        predictive_reason_codes=("baseline_rule_failed",),
        mechanistic_status="not_requested",
        mechanistic_reason_codes=(),
    )

    body = build_scorecard_body(
        scorecard_id="scorecard-v1",
        candidate_ref=_ref("reducer_artifact_manifest@1.0.0", "candidate"),
        point_score_policy_ref=_ref(
            "point_score_policy_manifest@1.0.0", "score-policy"
        ),
        point_score_result_ref=_ref(
            "point_score_result_manifest@1.0.0", "score-result"
        ),
        calibration_contract_ref=_ref(
            "calibration_contract_manifest@1.0.0", "calibration-contract"
        ),
        calibration_result_ref=_ref(
            "calibration_result_manifest@1.0.0", "calibration-result"
        ),
        evaluation_plan_ref=_ref(
            "evaluation_plan_manifest@1.1.0", "evaluation-plan"
        ),
        baseline_registry_ref=_ref(
            "baseline_registry_manifest@1.0.0", "baseline-registry"
        ),
        forecast_comparison_policy_ref=_ref(
            "forecast_comparison_policy_manifest@1.0.0", "comparison-policy"
        ),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison-universe"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event-log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        predictive_gate_policy_ref=_ref(
            "predictive_gate_policy_manifest@1.0.0", "predictive-gate"
        ),
        null_protocol_ref=_ref("null_protocol_manifest@1.0.0", "null-protocol"),
        perturbation_protocol_ref=_ref(
            "perturbation_protocol_manifest@1.0.0", "perturbation-protocol"
        ),
        robustness_report_ref=_ref(
            "robustness_report_manifest@1.1.0", "robustness"
        ),
        time_safety_audit_ref=_ref(
            "time_safety_audit_manifest@1.0.0", "time-safety"
        ),
        scorecard_decision=decision,
        description_gain_bits=12.5,
        forecast_object_type="point",
        entity_panel=("entity-a",),
        extra_fields={"predictive_law": None},
    )

    assert body["scorecard_id"] == "scorecard-v1"
    assert body["candidate_ref"] == {
        "schema_name": "reducer_artifact_manifest@1.0.0",
        "object_id": "candidate",
    }
    assert body["description_gain_bits"] == 12.5
    assert body["descriptive_status"] == "passed"
    assert body["descriptive_reason_codes"] == []
    assert body["predictive_status"] == "blocked"
    assert body["predictive_reason_codes"] == ["baseline_rule_failed"]
    assert body["mechanistic_status"] == "not_requested"
    assert body["mechanistic_reason_codes"] == []
    assert body["forecast_object_type"] == "point"
    assert body["entity_panel"] == ["entity-a"]
    assert body["predictive_law"] is None


def test_build_claim_card_body_preserves_required_manifest_shape() -> None:
    decision = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": "passed",
            "descriptive_reason_codes": [],
            "predictive_status": "passed",
            "predictive_reason_codes": [],
            "mechanistic_status": "not_requested",
            "invariance_status": "not_requested",
            "transport_status": "not_requested",
            "stochastic_status": "not_requested",
        }
    )

    body = build_claim_card_body(
        claim_card_id="claim-card-v1",
        candidate_ref=_ref("reducer_artifact_manifest@1.0.0", "candidate"),
        scorecard_ref=_ref("scorecard_manifest@1.1.0", "scorecard"),
        validation_scope_ref=_ref(
            "validation_scope_manifest@1.0.0", "validation-scope"
        ),
        claim_decision=decision,
        extra_fields={"predictive_law": None},
    )

    assert body == {
        "claim_card_id": "claim-card-v1",
        "candidate_ref": {
            "schema_name": "reducer_artifact_manifest@1.0.0",
            "object_id": "candidate",
        },
        "scorecard_ref": {
            "schema_name": "scorecard_manifest@1.1.0",
            "object_id": "scorecard",
        },
        "validation_scope_ref": {
            "schema_name": "validation_scope_manifest@1.0.0",
            "object_id": "validation-scope",
        },
        "claim_type": "predictive_within_declared_scope",
        "claim_ceiling": "predictive_within_declared_scope",
        "predictive_support_status": "confirmatory_supported",
        "invariance_support_status": "not_requested",
        "transport_support_status": "not_requested",
        "stochastic_support_status": "not_requested",
        "downgrade_reason_codes": [],
        "allowed_interpretation_codes": [
            "historical_structure_summary",
            "point_forecast_within_declared_validation_scope",
        ],
        "forbidden_interpretation_codes": [
            "causal_claim",
            "mechanism_claim",
            "transport_claim",
            "invariant_claim",
            "universal_claim",
            "cross_entity_generalization",
            "probabilistic_forecast_claim",
            "calibration_claim",
        ],
        "predictive_law": None,
    }


def test_resolve_claim_publication_downgrades_failed_predictive_support() -> None:
    decision = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": "passed",
            "descriptive_reason_codes": [],
            "predictive_status": "blocked",
            "predictive_reason_codes": ["baseline_rule_failed"],
        }
    )

    assert decision.publication_mode == "candidate_publication"
    assert decision.claim_type == "descriptive_structure"
    assert decision.claim_ceiling == "descriptive_structure"
    assert decision.predictive_support_status == "blocked"
    assert decision.abstention_type is None
    assert decision.allowed_interpretation_codes == ("historical_structure_summary",)


def test_resolve_claim_publication_emits_predictive_lane_only_on_passed_gate() -> None:
    decision = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": "passed",
            "descriptive_reason_codes": [],
            "predictive_status": "passed",
            "predictive_reason_codes": [],
        }
    )

    assert decision.publication_mode == "candidate_publication"
    assert decision.claim_type == "predictive_within_declared_scope"
    assert decision.claim_ceiling == "predictive_within_declared_scope"
    assert decision.predictive_support_status == "confirmatory_supported"
    assert decision.allowed_interpretation_codes == (
        "historical_structure_summary",
        "point_forecast_within_declared_validation_scope",
    )


def test_resolve_claim_publication_emits_cross_entity_panel_lane_within_panel_only(  # noqa: E501
) -> None:
    decision = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": "passed",
            "descriptive_reason_codes": [],
            "predictive_status": "passed",
            "predictive_reason_codes": [],
            "forecast_object_type": "point",
            "entity_panel": ["entity-a", "entity-b"],
        }
    )

    assert decision.allowed_interpretation_codes == (
        "historical_structure_summary",
        "point_forecast_within_declared_validation_scope",
        "cross_entity_panel_forecast_within_declared_validation_scope",
    )


def test_resolve_claim_publication_emits_no_admissible_reducer_abstention() -> None:
    decision = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": "blocked",
            "descriptive_reason_codes": [
                "descriptive_gate_failed",
                "no_candidate_survived_search",
            ],
            "predictive_status": "not_requested",
            "predictive_reason_codes": ["predictive_not_requested"],
        }
    )

    assert decision.publication_mode == "abstention_only_publication"
    assert decision.abstention_type == "no_admissible_reducer"
    assert decision.abstention_reason_codes == (
        "descriptive_gate_failed",
        "no_candidate_survived_search",
    )
    assert decision.claim_type is None


def test_resolve_claim_publication_emits_codelength_abstention() -> None:
    decision = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": "blocked_codelength_comparability_failed",
            "descriptive_reason_codes": ["codelength_comparability_failed"],
            "predictive_status": "not_requested",
            "predictive_reason_codes": ["predictive_not_requested"],
        }
    )

    assert decision.publication_mode == "abstention_only_publication"
    assert decision.abstention_type == "codelength_comparability_failed"
    assert decision.abstention_reason_codes == ("codelength_comparability_failed",)


def test_resolve_claim_publication_emits_robustness_abstention() -> None:
    decision = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": "blocked_robustness_failed",
            "descriptive_reason_codes": [
                "robustness_failed",
                "leakage_canary_failed",
            ],
            "predictive_status": "not_requested",
            "predictive_reason_codes": ["predictive_not_requested"],
        }
    )

    assert decision.publication_mode == "abstention_only_publication"
    assert decision.abstention_type == "robustness_failed"
    assert decision.abstention_reason_codes == (
        "robustness_failed",
        "leakage_canary_failed",
    )


def test_claim_scope_rejects_universal_coding_language_without_eligible_policy_tier() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        assert_claim_scope_publication(
            {
                "claim_type": "invariant_predictive_law",
                "claim_ceiling": "invariant_predictive_law",
                "claim_text": (
                    "This invariant law is backed by universal coding evidence."
                ),
                "invariance_support_status": "passed",
                "transport_support_status": "not_requested",
                "stochastic_support_status": "not_requested",
                "codelength_policy_manifest": {
                    "coding_claim_tier": "mdl_inspired_proxy_score",
                    "coding_claim_tier_reason_code": (
                        "legacy_fixed_step_raw_reference_policy_is_proxy_score"
                    ),
                },
            }
        )

    assert exc_info.value.code == "claim_scope_overstatement"
    assert "universal_coding_language_requires_universal_codelength_claim_tier" in (
        exc_info.value.details["reason_codes"]
    )
