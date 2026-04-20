from __future__ import annotations

from euclid.modules.claims import resolve_claim_publication


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
    assert decision.claim_type == "descriptive_only"
    assert decision.claim_ceiling == "descriptive_only"
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
    assert decision.claim_type == "predictively_supported"
    assert decision.claim_ceiling == "predictively_supported"
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
