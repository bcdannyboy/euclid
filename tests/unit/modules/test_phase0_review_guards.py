from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.modules.claims import assert_claim_scope_publication
from euclid.modules.gate_lifecycle import resolve_scorecard_status


GENERIC_REASON_CODES = {"failed", "invalid"}


def _reason_codes(error: ContractValidationError) -> set[str]:
    if "reason_codes" in error.details:
        return {str(code) for code in error.details["reason_codes"]}
    if "reason_code" in error.details:
        return {str(error.details["reason_code"])}
    return set()


def test_typed_evidence_status_rejects_empty_and_generic_non_passed_reasons() -> (
    None
):
    from euclid.modules.evidence_contracts import EvidenceStatus

    for factory_name in ("failed", "abstained", "downgraded"):
        factory = getattr(EvidenceStatus, factory_name)

        with pytest.raises(ContractValidationError) as missing_reasons:
            factory(reason_codes=())
        assert missing_reasons.value.code in {
            "evidence_status_missing_reason_codes",
            "missing_evidence_reason_code",
        }

        with pytest.raises(ContractValidationError) as generic_reason:
            factory(reason_codes=("failed",))
        assert generic_reason.value.code in {
            "generic_reason_code_forbidden",
            "unknown_evidence_reason_code",
        }
        assert _reason_codes(generic_reason.value) == {"failed"}


def test_passed_gate_decision_requires_evidence_refs_when_artifacts_are_required() -> (
    None
):
    from euclid.modules.evidence_contracts import EvidenceStatus

    with pytest.raises(ContractValidationError) as exc_info:
        EvidenceStatus.passed(
            artifacts_required=True,
            evidence_refs=(),
        )

    assert exc_info.value.code in {
        "passed_gate_missing_evidence_refs",
        "missing_required_evidence_refs",
    }


@pytest.mark.parametrize(
    ("decision_class_name", "base_kwargs", "status_factory_name"),
    (
        (
            "EvidenceGateDecision",
            {
                "gate_id": "calibration_gate",
            },
            "failed",
        ),
        (
            "ClaimScopeDecision",
            {
                "claim_scope": "predictive_within_declared_scope",
            },
            "downgraded",
        ),
        (
            "PromotionDecision",
            {
                "promotion_id": "candidate_to_predictive",
            },
            "failed",
        ),
    ),
)
def test_non_passed_gate_claim_and_promotion_manifests_require_specific_reasons(
    decision_class_name: str,
    base_kwargs: dict[str, str],
    status_factory_name: str,
) -> None:
    import euclid.modules.evidence_contracts as evidence_contracts

    decision_class = getattr(evidence_contracts, decision_class_name)
    status_factory = getattr(evidence_contracts.EvidenceStatus, status_factory_name)

    with pytest.raises(ContractValidationError) as missing_reasons:
        decision_class(
            **base_kwargs,
            status=status_factory(reason_codes=()),
        ).as_manifest()
    assert missing_reasons.value.code in {
        "evidence_status_missing_reason_codes",
        "missing_evidence_reason_code",
    }

    with pytest.raises(ContractValidationError) as generic_reason:
        decision_class(
            **base_kwargs,
            status=status_factory(reason_codes=("invalid",)),
        ).as_manifest()
    assert generic_reason.value.code in {
        "generic_reason_code_forbidden",
        "unknown_evidence_reason_code",
    }
    assert _reason_codes(generic_reason.value) == {"invalid"}

    manifest = decision_class(
        **base_kwargs,
        status=status_factory(reason_codes=("baseline_rule_failed",)),
    ).as_manifest()
    assert manifest["reason_codes"] == ["baseline_rule_failed"]


@pytest.mark.parametrize(
    ("claim_text", "expected_reason_codes"),
    (
        (
            "This law is universal across the declared domain.",
            {"universal_language_requires_invariance_or_transport"},
        ),
        (
            "The discovered equation is an invariant relationship.",
            {"invariant_language_requires_invariant_lane"},
        ),
        (
            "The relationship transports to future unseen domains.",
            {"transport_language_requires_transport_lane"},
        ),
        (
            "The candidate is a stochastic law with calibrated randomness.",
            {"stochastic_language_requires_stochastic_lane"},
        ),
    ),
)
def test_claim_scope_publication_blocks_text_overclaims_with_specific_reason_codes(
    claim_text: str,
    expected_reason_codes: set[str],
) -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        assert_claim_scope_publication(
            {
                "claim_type": "predictive_within_declared_scope",
                "claim_ceiling": "predictive_within_declared_scope",
                "claim_text": claim_text,
                "invariance_support_status": "not_requested",
                "transport_support_status": "not_requested",
                "stochastic_support_status": "not_requested",
            }
        )

    assert exc_info.value.code == "claim_scope_overstatement"
    observed_reason_codes = _reason_codes(exc_info.value)
    assert expected_reason_codes <= observed_reason_codes
    assert not (GENERIC_REASON_CODES & observed_reason_codes)


def test_scorecard_decision_preserves_legacy_shape_with_typed_statuses() -> None:
    decision = resolve_scorecard_status(
        candidate_admissible=True,
        robustness_status="passed",
        candidate_beats_baseline=False,
        confirmatory_promotion_allowed=True,
        point_score_comparison_status="comparable",
        time_safety_status="passed",
        calibration_status="not_applicable_for_forecast_type",
    )

    assert decision.descriptive_status == "passed"
    assert decision.descriptive_reason_codes == ()
    assert decision.predictive_status == "blocked"
    assert decision.predictive_reason_codes == ("baseline_rule_failed",)

    assert decision.descriptive_gate is not None
    assert decision.descriptive_gate.status.status == "passed"
    assert decision.descriptive_gate.status.reason_codes == ()
    assert decision.predictive_gate is not None
    assert decision.predictive_gate.status.status == "failed"
    assert decision.predictive_gate.status.reason_codes == (
        "baseline_rule_failed",
    )
    assert decision.as_manifest() == {
        "descriptive_status": "passed",
        "descriptive_reason_codes": [],
        "predictive_status": "blocked",
        "predictive_reason_codes": ["baseline_rule_failed"],
        "mechanistic_status": "not_requested",
        "mechanistic_reason_codes": [],
    }
