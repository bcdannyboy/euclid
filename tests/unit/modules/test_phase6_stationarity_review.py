from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.modules.claims import (
    assert_claim_scope_publication,
    build_claim_card_body,
    resolve_claim_publication,
)


GENERIC_REASON_CODES = {"failed", "invalid", "error"}


def _ref(schema_name: str, object_id: str) -> TypedRef:
    return TypedRef(schema_name=schema_name, object_id=object_id)


def _reason_codes(error: ContractValidationError) -> set[str]:
    if "reason_codes" in error.details:
        return {str(code) for code in error.details["reason_codes"]}
    if "reason_code" in error.details:
        return {str(error.details["reason_code"])}
    return set()


def _claim_card(
    *,
    claim_text: str,
    extra_fields: dict[str, object],
    requested_claim_lane: str = "predictive_within_declared_scope",
) -> dict[str, object]:
    scorecard_body: dict[str, object] = {
        "descriptive_status": "passed",
        "descriptive_reason_codes": [],
        "predictive_status": "passed",
        "predictive_reason_codes": [],
        "mechanistic_status": "not_requested",
        "transport_status": "not_requested",
        "stochastic_status": "not_requested",
    }
    if requested_claim_lane == "invariant_predictive_law":
        scorecard_body["invariance_status"] = "passed"
        scorecard_body["requested_claim_lane"] = requested_claim_lane
    else:
        scorecard_body["invariance_status"] = "not_requested"

    claim_decision = resolve_claim_publication(scorecard_body=scorecard_body)
    return build_claim_card_body(
        claim_card_id="phase6-stationarity-review-card",
        candidate_ref=_ref("reducer_artifact_manifest@1.0.0", "candidate"),
        scorecard_ref=_ref("scorecard_manifest@1.1.0", "scorecard"),
        validation_scope_ref=_ref(
            "validation_scope_manifest@1.0.0",
            "validation-scope",
        ),
        claim_decision=claim_decision,
        extra_fields={"claim_text": claim_text, **extra_fields},
    )


def _assert_scope_overstatement(
    claim_card: dict[str, object],
    *,
    expected_reason_code: str,
) -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        assert_claim_scope_publication(claim_card)

    assert exc_info.value.code == "claim_scope_overstatement"
    observed_reason_codes = _reason_codes(exc_info.value)
    assert expected_reason_code in observed_reason_codes
    assert not (GENERIC_REASON_CODES & observed_reason_codes)


@pytest.mark.parametrize(
    ("artifact_field", "artifact", "expected_reason_code"),
    (
        (
            "regime_switching_artifact",
            {
                "schema_name": "regime_switching_artifact@1.0.0",
                "method": "hidden_markov_regime_switching_v1",
                "status": "detected",
                "regime_ids": ["calm", "stress"],
                "switch_count": 3,
                "resolved_by_later_lane": False,
                "reason_codes": ["regime_switching_instability_detected"],
            },
            "regime_switching_instability_blocks_stationary_law_claim",
        ),
        (
            "state_space_artifact",
            {
                "schema_name": "state_space_artifact@1.0.0",
                "method": "local_level_state_space_v1",
                "status": "unstable",
                "state_evolution_status": "time_varying_unresolved",
                "resolved_by_later_lane": False,
                "reason_codes": ["state_space_instability_unresolved"],
            },
            "state_space_instability_blocks_stationary_law_claim",
        ),
    ),
)
def test_unresolved_instability_artifacts_block_stationary_law_claims(
    artifact_field: str,
    artifact: dict[str, object],
    expected_reason_code: str,
) -> None:
    claim_card = _claim_card(
        claim_text=(
            "The candidate is a universal stationary predictive law across "
            "all regimes."
        ),
        requested_claim_lane="invariant_predictive_law",
        extra_fields={artifact_field: artifact},
    )

    _assert_scope_overstatement(
        claim_card,
        expected_reason_code=expected_reason_code,
    )


def test_instability_evidence_cannot_publish_as_law_claim() -> None:
    claim_card = _claim_card(
        claim_text=(
            "The residual instability evidence is published as an invariant "
            "predictive law."
        ),
        requested_claim_lane="invariant_predictive_law",
        extra_fields={
            "stability_diagnostic_artifact": {
                "schema_name": "stability_diagnostic_artifact@1.0.0",
                "method": "recursive_residual_cusum_v1",
                "status": "passed",
                "evidence_role": "instability_evidence",
                "claim_scope": "instability_evidence_only",
                "reason_codes": [],
            }
        },
    )

    _assert_scope_overstatement(
        claim_card,
        expected_reason_code="instability_evidence_cannot_be_law_claim",
    )


@pytest.mark.parametrize(
    ("artifact_field", "artifact", "claim_text", "expected_reason_code"),
    (
        (
            "regime_switching_artifact",
            {
                "schema_name": "regime_switching_artifact@1.0.0",
                "method": "hidden_markov_regime_switching_v1",
                "status": "passed",
                "regime_id": "stress",
                "reason_codes": [],
            },
            "The predictive relationship is supported in the stress regime.",
            "regime_scoped_evidence_requires_valid_given_regime_scope",
        ),
        (
            "state_space_artifact",
            {
                "schema_name": "state_space_artifact@1.0.0",
                "method": "local_level_state_space_v1",
                "status": "passed",
                "state_id": "high-load-latent-state",
                "reason_codes": [],
            },
            "The predictive relationship is supported in the high-load state.",
            "state_scoped_evidence_requires_valid_given_state_scope",
        ),
    ),
)
def test_scoped_evidence_requires_explicit_valid_given_wording_or_manifest_scope(
    artifact_field: str,
    artifact: dict[str, object],
    claim_text: str,
    expected_reason_code: str,
) -> None:
    claim_card = _claim_card(
        claim_text=claim_text,
        extra_fields={artifact_field: artifact},
    )

    _assert_scope_overstatement(
        claim_card,
        expected_reason_code=expected_reason_code,
    )


@pytest.mark.parametrize(
    ("artifact_field", "artifact", "expected_reason_code"),
    (
        (
            "regime_switching_artifact",
            {
                "schema_name": "regime_switching_artifact@1.0.0",
                "method": "hidden_markov_regime_switching_v1",
                "status": "passed",
                "claim_scope": "valid_given_regime",
                "valid_given_regime": "stress",
                "reason_codes": [],
            },
            "regime_scoped_evidence_cannot_support_stationary_law_claim",
        ),
        (
            "state_space_artifact",
            {
                "schema_name": "state_space_artifact@1.0.0",
                "method": "local_level_state_space_v1",
                "status": "passed",
                "claim_scope": "valid_given_state",
                "valid_given_state": "high-load-latent-state",
                "reason_codes": [],
            },
            "state_scoped_evidence_cannot_support_stationary_law_claim",
        ),
    ),
)
def test_scoped_evidence_does_not_launder_into_stationary_law_claims(
    artifact_field: str,
    artifact: dict[str, object],
    expected_reason_code: str,
) -> None:
    claim_card = _claim_card(
        claim_text=(
            "The valid_given scope is promoted to a universal stationary "
            "predictive law across all regimes."
        ),
        requested_claim_lane="invariant_predictive_law",
        extra_fields={artifact_field: artifact},
    )

    _assert_scope_overstatement(
        claim_card,
        expected_reason_code=expected_reason_code,
    )
