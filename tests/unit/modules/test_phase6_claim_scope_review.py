from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.modules.claims import assert_claim_scope_publication


GENERIC_REASON_CODES = {"failed", "invalid", "error"}


def _reason_codes(error: ContractValidationError) -> set[str]:
    if "reason_codes" in error.details:
        return {str(code) for code in error.details["reason_codes"]}
    if "reason_code" in error.details:
        return {str(error.details["reason_code"])}
    return set()


def _base_claim_card(
    *,
    claim_ceiling: str = "predictive_within_declared_scope",
    claim_text: str,
    invariance_support_status: str = "not_requested",
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    claim_card: dict[str, object] = {
        "claim_type": claim_ceiling,
        "claim_ceiling": claim_ceiling,
        "claim_text": claim_text,
        "invariance_support_status": invariance_support_status,
        "transport_support_status": "not_requested",
        "stochastic_support_status": "not_requested",
    }
    if extra_fields:
        claim_card.update(extra_fields)
    return claim_card


def test_stability_diagnostic_evidence_does_not_publish_universal_law_claim() -> None:
    claim_card = _base_claim_card(
        claim_ceiling="invariant_predictive_law",
        claim_text=(
            "The recursive residual stability diagnostic proves a universal "
            "stationary predictive law."
        ),
        invariance_support_status="passed",
        extra_fields={
            "stability_diagnostic_artifact": {
                "schema_name": "stability_diagnostic_artifact@1.0.0",
                "method": "recursive_residual_cusum_v1",
                "status": "passed",
                "evidence_role": "diagnostic_only",
                "reason_codes": [],
            }
        },
    )

    with pytest.raises(ContractValidationError) as exc_info:
        assert_claim_scope_publication(claim_card)

    assert exc_info.value.code == "claim_scope_overstatement"
    observed_reason_codes = _reason_codes(exc_info.value)
    assert "stability_diagnostic_is_diagnostic_only" in observed_reason_codes
    assert not (GENERIC_REASON_CODES & observed_reason_codes)


def test_unresolved_instability_blocks_stationary_predictive_law_wording() -> None:
    claim_card = _base_claim_card(
        claim_text=(
            "The candidate is a stationary predictive law across the declared "
            "validation window."
        ),
        extra_fields={
            "stability_diagnostic_artifact": {
                "schema_name": "stability_diagnostic_artifact@1.0.0",
                "method": "recursive_residual_cusum_v1",
                "status": "unstable",
                "reason_codes": ["stability_test_failed"],
                "handled_by_nonstationary_lane": False,
            }
        },
    )

    with pytest.raises(ContractValidationError) as exc_info:
        assert_claim_scope_publication(claim_card)

    assert exc_info.value.code == "claim_scope_overstatement"
    observed_reason_codes = _reason_codes(exc_info.value)
    assert "unresolved_instability_blocks_stationary_law_claim" in (
        observed_reason_codes
    )
    assert not (GENERIC_REASON_CODES & observed_reason_codes)


def test_change_point_artifact_requires_scoped_nonstationary_wording_until_resolved() -> (
    None
):
    claim_card = _base_claim_card(
        claim_ceiling="invariant_predictive_law",
        claim_text=(
            "The detected change point proves one universal stationary "
            "predictive law for all regimes."
        ),
        invariance_support_status="passed",
        extra_fields={
            "change_point_artifact": {
                "schema_name": "change_point_artifact@1.0.0",
                "method": "pelt_l2_v1",
                "status": "passed",
                "breakpoints": [48],
                "penalty": 3.0,
                "min_segment_size": 12,
                "tolerance": 3,
                "resolved_by_later_lane": False,
                "supported_claim_scope": "scoped_nonstationary_evidence",
            }
        },
    )

    with pytest.raises(ContractValidationError) as exc_info:
        assert_claim_scope_publication(claim_card)

    assert exc_info.value.code == "claim_scope_overstatement"
    observed_reason_codes = _reason_codes(exc_info.value)
    assert "change_point_artifact_requires_scoped_nonstationary_wording" in (
        observed_reason_codes
    )
    assert not (GENERIC_REASON_CODES & observed_reason_codes)
