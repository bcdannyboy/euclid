from __future__ import annotations

import importlib

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.modules.claims import resolve_claim_publication


def _evidence_contracts():
    return importlib.import_module("euclid.modules.evidence_contracts")


def _ref(schema_name: str, object_id: str) -> TypedRef:
    return TypedRef(schema_name=schema_name, object_id=object_id)


def test_passed_evidence_status_requires_artifact_refs_when_required() -> None:
    contracts = _evidence_contracts()

    with pytest.raises(ContractValidationError) as excinfo:
        contracts.EvidenceStatus.passed(
            artifacts_required=True,
            evidence_refs=(),
        )

    assert excinfo.value.code == "missing_required_evidence_refs"
    assert excinfo.value.field_path == "evidence_refs"


@pytest.mark.parametrize("factory_name", ["failed", "abstained"])
def test_non_passed_evidence_status_requires_reason_code(
    factory_name: str,
) -> None:
    contracts = _evidence_contracts()
    factory = getattr(contracts.EvidenceStatus, factory_name)

    with pytest.raises(ContractValidationError) as excinfo:
        factory(reason_codes=())

    assert excinfo.value.code == "missing_evidence_reason_code"
    assert excinfo.value.field_path == "reason_codes"


def test_unknown_evidence_status_raises_contract_validation_error() -> None:
    contracts = _evidence_contracts()

    with pytest.raises(ContractValidationError) as excinfo:
        contracts.EvidenceStatus(
            status="promoted_without_confirmatory_gate",
            reason_codes=("baseline_rule_failed",),
            artifacts_required=False,
        )

    assert excinfo.value.code == "unknown_evidence_status"
    assert excinfo.value.details["status"] == "promoted_without_confirmatory_gate"


def test_unknown_reason_code_raises_contract_validation_error() -> None:
    contracts = _evidence_contracts()

    with pytest.raises(ContractValidationError) as excinfo:
        contracts.EvidenceStatus.failed(
            ("calibration_artifact_missing",),
            allowed_reason_codes={"baseline_rule_failed"},
        )

    assert excinfo.value.code == "unknown_evidence_reason_code"
    assert excinfo.value.details["reason_code"] == "calibration_artifact_missing"


def test_falsification_reason_codes_are_valid_evidence_contract_codes() -> None:
    contracts = _evidence_contracts()

    status = contracts.EvidenceStatus.failed(
        (
            "falsification_failed",
            "structured_residuals",
            "counterexample_discovered",
            "extrapolation_failure",
        )
    )

    assert status.reason_codes == (
        "falsification_failed",
        "structured_residuals",
        "counterexample_discovered",
        "extrapolation_failure",
    )


def test_passed_evidence_status_preserves_typed_artifact_refs() -> None:
    contracts = _evidence_contracts()
    score_ref = _ref("point_score_result_manifest@1.0.0", "score-result-1")

    status = contracts.EvidenceStatus.passed(
        artifacts_required=True,
        evidence_refs=(score_ref,),
    )

    assert status.status == "passed"
    assert status.reason_codes == ()
    assert status.evidence_refs == (score_ref,)


def test_scorecard_builder_preserves_required_manifest_shape() -> None:
    contracts = _evidence_contracts()
    score_ref = _ref("point_score_result_manifest@1.0.0", "score-result-1")
    calibration_ref = _ref("calibration_result_manifest@1.0.0", "calibration-1")
    robustness_ref = _ref("robustness_report_manifest@1.0.0", "robustness-1")

    scorecard_body = contracts.build_scorecard_manifest_body(
        scorecard_id="scorecard-1",
        candidate_ref=_ref("reducer_artifact_manifest@1.0.0", "candidate-1"),
        point_score_result_ref=score_ref,
        calibration_result_ref=calibration_ref,
        robustness_report_ref=robustness_ref,
        forecast_object_type="point",
        descriptive=contracts.EvidenceStatus.passed(
            artifacts_required=True,
            evidence_refs=(robustness_ref,),
        ),
        predictive=contracts.EvidenceStatus.failed(
            ("baseline_rule_failed",),
            allowed_reason_codes={"baseline_rule_failed"},
            evidence_refs=(score_ref, calibration_ref),
        ),
    )

    assert scorecard_body["scorecard_id"] == "scorecard-1"
    assert scorecard_body["candidate_ref"] == {
        "schema_name": "reducer_artifact_manifest@1.0.0",
        "object_id": "candidate-1",
    }
    assert scorecard_body["point_score_result_ref"] == score_ref.as_dict()
    assert scorecard_body["calibration_result_ref"] == calibration_ref.as_dict()
    assert scorecard_body["robustness_report_ref"] == robustness_ref.as_dict()
    assert scorecard_body["forecast_object_type"] == "point"
    assert scorecard_body["descriptive_status"] == "passed"
    assert scorecard_body["descriptive_reason_codes"] == []
    assert scorecard_body["descriptive_evidence_refs"] == [
        robustness_ref.as_dict()
    ]
    assert scorecard_body["predictive_status"] == "failed"
    assert scorecard_body["predictive_reason_codes"] == ["baseline_rule_failed"]
    assert scorecard_body["predictive_evidence_refs"] == [
        score_ref.as_dict(),
        calibration_ref.as_dict(),
    ]


def test_claim_card_builder_uses_claim_decision_without_workflow_local_shape() -> None:
    contracts = _evidence_contracts()
    scorecard_ref = _ref("scorecard_manifest@1.1.0", "scorecard-1")
    validation_scope_ref = _ref(
        "validation_scope_manifest@1.0.0",
        "validation-scope-1",
    )
    scorecard_body = {
        "descriptive_status": "passed",
        "descriptive_reason_codes": [],
        "predictive_status": "passed",
        "predictive_reason_codes": [],
        "forecast_object_type": "point",
    }
    decision = resolve_claim_publication(scorecard_body=scorecard_body)

    claim_card_body = contracts.build_claim_card_manifest_body(
        claim_card_id="claim-card-1",
        candidate_ref=_ref("reducer_artifact_manifest@1.0.0", "candidate-1"),
        scorecard_ref=scorecard_ref,
        validation_scope_ref=validation_scope_ref,
        claim_decision=decision,
    )

    assert claim_card_body == {
        "claim_card_id": "claim-card-1",
        "candidate_ref": {
            "schema_name": "reducer_artifact_manifest@1.0.0",
            "object_id": "candidate-1",
        },
        "scorecard_ref": scorecard_ref.as_dict(),
        "validation_scope_ref": validation_scope_ref.as_dict(),
        "claim_type": "predictive_within_declared_scope",
        "claim_ceiling": "predictive_within_declared_scope",
        "predictive_support_status": "confirmatory_supported",
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
    }
