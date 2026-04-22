from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.modules.claims import (
    assert_claim_scope_publication,
    normalize_claim_lane,
    resolve_claim_publication,
)


def test_claim_taxonomy_uses_planned_lanes_and_rejects_legacy_production_names() -> None:
    assert normalize_claim_lane("descriptive_structure") == "descriptive_structure"
    assert (
        normalize_claim_lane("predictive_within_declared_scope")
        == "predictive_within_declared_scope"
    )
    assert (
        normalize_claim_lane("mechanistically_compatible_law")
        == "mechanistically_compatible_law"
    )

    with pytest.raises(ContractValidationError) as exc_info:
        normalize_claim_lane("predictively_supported")

    assert exc_info.value.code == "legacy_claim_lane_not_production"


def test_invariant_claim_downgrades_without_passed_invariance_evidence() -> None:
    decision = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": "passed",
            "descriptive_reason_codes": [],
            "predictive_status": "passed",
            "predictive_reason_codes": [],
            "requested_claim_lane": "invariant_predictive_law",
            "invariance_status": "failed",
            "invariance_reason_codes": ["parameter_drift_failed"],
        }
    )

    assert decision.publication_mode == "candidate_publication"
    assert decision.claim_type == "predictive_within_declared_scope"
    assert decision.claim_ceiling == "predictive_within_declared_scope"
    assert decision.invariance_support_status == "failed"
    assert "invariance_required_for_invariant_claim" in decision.downgrade_reason_codes
    assert "invariant_claim" in decision.forbidden_interpretation_codes


def test_transport_supported_claim_requires_transport_evidence() -> None:
    decision = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": "passed",
            "descriptive_reason_codes": [],
            "predictive_status": "passed",
            "predictive_reason_codes": [],
            "requested_claim_lane": "transport_supported_law",
            "transport_status": "failed",
            "transport_reason_codes": ["transport_holdout_failed"],
        }
    )

    assert decision.claim_type == "predictive_within_declared_scope"
    assert decision.transport_support_status == "failed"
    assert "transport_required_for_transport_supported_claim" in (
        decision.downgrade_reason_codes
    )
    assert "transport_claim" in decision.forbidden_interpretation_codes


def test_publication_scope_assertion_blocks_unbacked_universal_language() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        assert_claim_scope_publication(
            {
                "claim_type": "predictive_within_declared_scope",
                "claim_ceiling": "predictive_within_declared_scope",
                "claim_text": (
                    "This is a universal invariant law that transports to new "
                    "entities."
                ),
                "invariance_support_status": "not_requested",
                "transport_support_status": "not_requested",
            }
        )

    assert exc_info.value.code == "claim_scope_overstatement"


def test_invariant_and_transport_lanes_publish_only_with_required_evidence() -> None:
    invariant = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": "passed",
            "descriptive_reason_codes": [],
            "predictive_status": "passed",
            "predictive_reason_codes": [],
            "requested_claim_lane": "invariant_predictive_law",
            "invariance_status": "passed",
        }
    )
    transport = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": "passed",
            "descriptive_reason_codes": [],
            "predictive_status": "passed",
            "predictive_reason_codes": [],
            "requested_claim_lane": "transport_supported_law",
            "transport_status": "passed",
        }
    )

    assert invariant.claim_type == "invariant_predictive_law"
    assert "invariant_claim" in invariant.allowed_interpretation_codes
    assert transport.claim_type == "transport_supported_law"
    assert "transport_claim" in transport.allowed_interpretation_codes
