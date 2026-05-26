from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.modules.claims import assert_claim_scope_publication


GENERIC_REASON_CODES = {"failed", "invalid", "error"}


def _reason_codes(error: ContractValidationError) -> set[str]:
    return {str(code) for code in error.details.get("reason_codes", ())}


def _base_claim_card(
    *,
    claim_text: str,
    extra_fields: dict[str, object],
) -> dict[str, object]:
    return {
        "claim_type": "predictive_within_declared_scope",
        "claim_ceiling": "predictive_within_declared_scope",
        "claim_text": claim_text,
        "invariance_support_status": "not_requested",
        "transport_support_status": "not_requested",
        "stochastic_support_status": "not_requested",
        **extra_fields,
    }


def test_regime_conditioned_claim_rejects_stationary_wording_without_scope() -> None:
    claim_card = _base_claim_card(
        claim_text="The candidate is a stationary predictive law for all regimes.",
        extra_fields={
            "composition_runtime_evidence": {
                "scored_origins": [
                    {
                        "operator_id": "regime_conditioned",
                        "selection_mode": "hard_switch",
                        "selected_branch_id": "volatile_branch",
                    }
                ]
            }
        },
    )

    with pytest.raises(ContractValidationError) as exc_info:
        assert_claim_scope_publication(claim_card)

    assert exc_info.value.code == "claim_scope_overstatement"
    observed_reason_codes = _reason_codes(exc_info.value)
    assert "regime_conditioned_evidence_requires_valid_given_regime_scope" in (
        observed_reason_codes
    )
    assert not (GENERIC_REASON_CODES & observed_reason_codes)


def test_regime_conditioned_claim_allows_stationary_wording_when_scope_is_explicit() -> (
    None
):
    claim_card = _base_claim_card(
        claim_text=(
            "The candidate is a stationary predictive law valid given the "
            "observed regime."
        ),
        extra_fields={
            "composition_runtime_evidence": {
                "scored_origins": [
                    {
                        "operator_id": "regime_conditioned",
                        "claim_scope": "valid_given_regime",
                        "valid_given_regime": {
                            "signal_fields": ["regime_flag"],
                            "observed_regime_value": "volatile",
                            "selected_branch_id": "volatile_branch",
                        },
                    }
                ]
            }
        },
    )

    assert_claim_scope_publication(claim_card)


def test_state_space_claim_rejects_stationary_wording_after_whiteness_failure() -> None:
    claim_card = _base_claim_card(
        claim_text="The state-space candidate is a stationary predictive law.",
        extra_fields={
            "state_space_evidence": {
                "schema_name": "state_space_artifact@1.0.0",
                "lane_id": "state_space_local_level_v1",
                "innovation_whiteness_status": "failed",
                "reason_codes": ["innovation_whiteness_failed"],
            }
        },
    )

    with pytest.raises(ContractValidationError) as exc_info:
        assert_claim_scope_publication(claim_card)

    assert exc_info.value.code == "claim_scope_overstatement"
    observed_reason_codes = _reason_codes(exc_info.value)
    assert "state_space_whiteness_failure_blocks_unscoped_stationary_claim" in (
        observed_reason_codes
    )
    assert not (GENERIC_REASON_CODES & observed_reason_codes)


def test_state_space_claim_allows_stationary_wording_when_state_scope_is_explicit() -> (
    None
):
    claim_card = _base_claim_card(
        claim_text=(
            "The state-space candidate is a stationary predictive law valid "
            "given the filtered state."
        ),
        extra_fields={
            "state_space_evidence": {
                "schema_name": "state_space_artifact@1.0.0",
                "lane_id": "state_space_local_level_v1",
                "claim_scope": "valid_given_state",
                "valid_given_state": {
                    "state_variable": "local_level",
                    "conditioning": "filtered_state",
                },
                "innovation_whiteness_status": "failed",
                "reason_codes": ["innovation_whiteness_failed"],
            }
        },
    )

    assert_claim_scope_publication(claim_card)
