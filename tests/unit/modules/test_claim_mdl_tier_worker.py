from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.modules.claims import assert_claim_scope_publication


def _claim_card(*, claim_text: str, coding_claim_tier: str) -> dict[str, object]:
    return {
        "claim_type": "invariant_predictive_law",
        "claim_ceiling": "invariant_predictive_law",
        "claim_text": claim_text,
        "invariance_support_status": "passed",
        "transport_support_status": "not_requested",
        "stochastic_support_status": "not_requested",
        "codelength_policy_manifest": {
            "body": {
                "coding_claim_tier": coding_claim_tier,
            },
        },
    }


@pytest.mark.parametrize(
    "coding_claim_tier",
    (
        "mdl_inspired_proxy_score",
        "not_mdl_claim_eligible",
    ),
)
def test_claim_publication_rejects_mdl_language_for_ineligible_coding_tiers(
    coding_claim_tier: str,
) -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        assert_claim_scope_publication(
            _claim_card(
                claim_text=(
                    "The candidate is published with minimum description length "
                    "support."
                ),
                coding_claim_tier=coding_claim_tier,
            )
        )

    assert exc_info.value.code == "claim_scope_overstatement"
    assert (
        "mdl_language_requires_eligible_codelength_claim_tier"
        in exc_info.value.details["reason_codes"]
    )


@pytest.mark.parametrize(
    "coding_claim_tier",
    (
        "mdl_inspired_proxy_score",
        "not_mdl_claim_eligible",
    ),
)
def test_claim_publication_rejects_universal_coding_language_for_ineligible_tiers(
    coding_claim_tier: str,
) -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        assert_claim_scope_publication(
            _claim_card(
                claim_text="The residual stream uses universal coding evidence.",
                coding_claim_tier=coding_claim_tier,
            )
        )

    assert exc_info.value.code == "claim_scope_overstatement"
    assert (
        "universal_coding_language_requires_universal_codelength_claim_tier"
        in exc_info.value.details["reason_codes"]
    )


def test_claim_publication_allows_mdl_language_for_exact_prequential_symbol_code() -> None:
    assert_claim_scope_publication(
        _claim_card(
            claim_text=(
                "The candidate is published with minimum description length "
                "support from an exact prequential symbol code."
            ),
            coding_claim_tier="exact_prequential_symbol_code",
        )
    )


def test_claim_publication_allows_universal_coding_language_for_mdl_universal_code() -> None:
    assert_claim_scope_publication(
        _claim_card(
            claim_text="The residual stream uses universal coding evidence.",
            coding_claim_tier="mdl_based_universal_code",
        )
    )
