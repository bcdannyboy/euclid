from __future__ import annotations

from pathlib import Path

from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.modules.claims import resolve_claim_publication
from euclid.modules.gate_lifecycle import resolve_scorecard_status
from euclid.modules.mechanistic_evidence import evaluate_mechanistic_evidence

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _ref(schema_name: str, object_id: str) -> TypedRef:
    return TypedRef(schema_name=schema_name, object_id=object_id)


def test_mechanistic_evidence_upgrades_claim_after_floor_and_checks_pass(
) -> None:
    _ = load_contract_catalog(PROJECT_ROOT)
    evaluation = evaluate_mechanistic_evidence(
        mechanistic_evidence_id="glucose_mechanistic_support",
        candidate_ref=_ref("reducer_artifact_manifest@1.0.0", "candidate"),
        prediction_artifact_ref=_ref(
            "prediction_artifact_manifest@1.1.0", "prediction"
        ),
        external_evidence_ref=_ref(
            "external_evidence_manifest@1.0.0", "glucose_bundle"
        ),
        lower_claim_ceiling="predictively_supported",
        term_bindings=(
            {
                "term_id": "lag1_state",
                "domain_entity": "circulating_glucose",
                "activity": "state_persistence",
            },
        ),
        term_units=(
            {"term_id": "lag1_state", "unit_id": "mmol_per_l", "compatible": True},
        ),
        invariance_checks=(
            {"check_id": "meal_shift", "status": "passed"},
        ),
        external_evidence_records=(
            {"source_id": "paper-a", "independence_mode": "external_domain_source"},
        ),
        predictive_evidence_refs=(
            _ref("point_score_result_manifest@1.0.0", "score_result"),
        ),
    )

    assert evaluation.mechanism_mapping.status == "passed"
    assert evaluation.units_check.status == "passed"
    assert evaluation.invariance_test.status == "passed"
    assert evaluation.evidence_independence.status == "passed"
    assert evaluation.dossier.status == "passed"
    assert evaluation.dossier.resolved_claim_ceiling == (
        "mechanistically_compatible_hypothesis"
    )

    scorecard = resolve_scorecard_status(
        candidate_admissible=True,
        robustness_status="passed",
        candidate_beats_baseline=True,
        confirmatory_promotion_allowed=True,
        point_score_comparison_status="comparable",
        time_safety_status="passed",
        calibration_status="not_applicable_for_forecast_type",
        mechanistic_requested=True,
        mechanistic_evidence_status=evaluation.dossier.status,
        mechanistic_reason_codes=evaluation.dossier.reason_codes,
    )
    claim = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": scorecard.descriptive_status,
            "descriptive_reason_codes": list(scorecard.descriptive_reason_codes),
            "predictive_status": scorecard.predictive_status,
            "predictive_reason_codes": list(scorecard.predictive_reason_codes),
            "mechanistic_status": scorecard.mechanistic_status,
            "mechanistic_reason_codes": list(scorecard.mechanistic_reason_codes),
            "forecast_object_type": "point",
        }
    )

    assert scorecard.mechanistic_status == "passed"
    assert claim.claim_type == "mechanistically_compatible_hypothesis"
    assert claim.claim_ceiling == "mechanistically_compatible_hypothesis"
    assert "mechanism_claim" in claim.allowed_interpretation_codes
    assert "mechanism_claim" not in claim.forbidden_interpretation_codes


def test_mechanistic_evidence_overlap_downgrades_to_predictive_support() -> None:
    evaluation = evaluate_mechanistic_evidence(
        mechanistic_evidence_id="glucose_mechanistic_overlap",
        candidate_ref=_ref("reducer_artifact_manifest@1.0.0", "candidate"),
        prediction_artifact_ref=_ref(
            "prediction_artifact_manifest@1.1.0", "prediction"
        ),
        external_evidence_ref=_ref(
            "external_evidence_manifest@1.0.0", "glucose_bundle"
        ),
        lower_claim_ceiling="predictively_supported",
        term_bindings=(
            {
                "term_id": "lag1_state",
                "domain_entity": "circulating_glucose",
                "activity": "state_persistence",
            },
        ),
        term_units=(
            {"term_id": "lag1_state", "unit_id": "mmol_per_l", "compatible": True},
        ),
        invariance_checks=(
            {"check_id": "meal_shift", "status": "passed"},
        ),
        external_evidence_records=(
            {
                "source_id": "paper-a",
                "independence_mode": "derived_from_predictive_artifact",
                "derived_from_predictive_ref": _ref(
                    "point_score_result_manifest@1.0.0",
                    "score_result",
                ).as_dict(),
            },
        ),
        predictive_evidence_refs=(
            _ref("point_score_result_manifest@1.0.0", "score_result"),
        ),
    )

    assert evaluation.evidence_independence.status == "failed"
    assert evaluation.dossier.status == "downgraded_to_predictively_supported"
    assert "predictive_evidence_overlap" in evaluation.dossier.reason_codes
    assert evaluation.dossier.resolved_claim_ceiling == "predictively_supported"

    scorecard = resolve_scorecard_status(
        candidate_admissible=True,
        robustness_status="passed",
        candidate_beats_baseline=True,
        confirmatory_promotion_allowed=True,
        point_score_comparison_status="comparable",
        time_safety_status="passed",
        calibration_status="not_applicable_for_forecast_type",
        mechanistic_requested=True,
        mechanistic_evidence_status=evaluation.dossier.status,
        mechanistic_reason_codes=evaluation.dossier.reason_codes,
    )
    claim = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": scorecard.descriptive_status,
            "descriptive_reason_codes": list(scorecard.descriptive_reason_codes),
            "predictive_status": scorecard.predictive_status,
            "predictive_reason_codes": list(scorecard.predictive_reason_codes),
            "mechanistic_status": scorecard.mechanistic_status,
            "mechanistic_reason_codes": list(scorecard.mechanistic_reason_codes),
            "forecast_object_type": "point",
        }
    )

    assert scorecard.mechanistic_status == "downgraded_to_predictively_supported"
    assert claim.claim_type == "predictively_supported"
    assert "mechanism_claim" not in claim.allowed_interpretation_codes
    assert "mechanism_claim" in claim.forbidden_interpretation_codes


def test_mechanistic_evidence_cannot_outrun_predictive_floor() -> None:
    evaluation = evaluate_mechanistic_evidence(
        mechanistic_evidence_id="glucose_mechanistic_no_floor",
        candidate_ref=_ref("reducer_artifact_manifest@1.0.0", "candidate"),
        prediction_artifact_ref=_ref(
            "prediction_artifact_manifest@1.1.0", "prediction"
        ),
        external_evidence_ref=_ref(
            "external_evidence_manifest@1.0.0", "glucose_bundle"
        ),
        lower_claim_ceiling="descriptive_only",
        term_bindings=(
            {
                "term_id": "lag1_state",
                "domain_entity": "circulating_glucose",
                "activity": "state_persistence",
            },
        ),
        term_units=(
            {"term_id": "lag1_state", "unit_id": "mmol_per_l", "compatible": True},
        ),
        invariance_checks=(
            {"check_id": "meal_shift", "status": "passed"},
        ),
        external_evidence_records=(
            {"source_id": "paper-a", "independence_mode": "external_domain_source"},
        ),
        predictive_evidence_refs=(),
    )

    assert evaluation.dossier.status == "blocked_predictive_floor"
    assert evaluation.dossier.resolved_claim_ceiling == "descriptive_only"
    assert evaluation.dossier.reason_codes == ("predictive_floor_required",)
