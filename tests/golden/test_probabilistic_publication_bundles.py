from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import euclid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "fixtures/runtime/phase06"
DISTRIBUTION_MANIFEST = FIXTURE_ROOT / "probabilistic-distribution-demo.yaml"
CALIBRATION_FAILURE_MANIFEST = (
    FIXTURE_ROOT / "probabilistic-distribution-calibration-failure-demo.yaml"
)


def _ref_string(payload: object) -> str | None:
    if isinstance(payload, euclid.TypedRef):
        return f"{payload.schema_name}:{payload.object_id}"
    if not isinstance(payload, Mapping):
        return None
    schema_name = payload.get("schema_name")
    object_id = payload.get("object_id")
    if isinstance(schema_name, str) and isinstance(object_id, str):
        return f"{schema_name}:{object_id}"
    return None


def _probabilistic_publication_snapshot(
    *,
    output_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=manifest_path,
        output_root=output_root,
    )
    replay = euclid.replay_demo(output_root=output_root)
    published = euclid.publish_demo_run_to_catalog(output_root=output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=output_root,
        publication_id=published.publication_id,
    )
    run_graph = euclid.load_demo_run_artifact_graph(output_root=output_root)
    run_result = run_graph.inspect(run_graph.root_ref).manifest
    validation_scope_ref = run_result.body["primary_validation_scope_ref"]
    assert isinstance(validation_scope_ref, Mapping)
    scorecard_body = (
        inspection.scorecard.manifest.body if inspection.scorecard is not None else {}
    )
    claim_body = (
        inspection.claim_card.manifest.body if inspection.claim_card is not None else {}
    )
    comparison = euclid.compare_demo_baseline(output_root=output_root)

    return {
        "request_id": result.request.request_id,
        "run_result_ref": _ref_string(run_result.ref),
        "forecast_object_type": str(run_result.body["forecast_object_type"]),
        "publication_mode": str(run_result.body["result_mode"]),
        "run_result_stochastic_evidence": {
            "stochastic_support_status": run_result.body.get(
                "stochastic_support_status"
            ),
            "stochastic_support_reason_codes": list(
                run_result.body.get("stochastic_support_reason_codes", ())
            ),
            "residual_history_refs": [
                ref
                for ref in (
                    _ref_string(item)
                    for item in run_result.body.get("residual_history_refs", ())
                )
                if ref is not None
            ],
            "stochastic_model_refs": [
                ref
                for ref in (
                    _ref_string(item)
                    for item in run_result.body.get("stochastic_model_refs", ())
                )
                if ref is not None
            ],
        },
        "summary_refs": {
            "prediction_artifact_ref": _ref_string(
                result.summary.prediction_artifact_ref
            ),
            "score_result_ref": _ref_string(result.summary.score_result_ref),
            "calibration_result_ref": _ref_string(
                result.summary.calibration_result_ref
            ),
            "claim_card_ref": _ref_string(result.summary.claim_card_ref),
            "abstention_ref": _ref_string(result.summary.abstention_ref),
        },
        "scorecard": {
            "descriptive_status": scorecard_body.get("descriptive_status"),
            "predictive_status": scorecard_body.get("predictive_status"),
            "predictive_reason_codes": list(
                scorecard_body.get("predictive_reason_codes", ())
            ),
            "stochastic_status": scorecard_body.get("stochastic_status"),
            "stochastic_evidence_status": scorecard_body.get(
                "stochastic_evidence_status"
            ),
            "residual_history_refs": [
                ref
                for ref in (
                    _ref_string(item)
                    for item in scorecard_body.get("residual_history_refs", ())
                )
                if ref is not None
            ],
            "stochastic_model_refs": [
                ref
                for ref in (
                    _ref_string(item)
                    for item in scorecard_body.get("stochastic_model_refs", ())
                )
                if ref is not None
            ],
        },
        "claim": {
            "claim_type": claim_body.get("claim_type"),
            "predictive_support_status": claim_body.get("predictive_support_status"),
            "stochastic_support_status": claim_body.get(
                "stochastic_support_status"
            ),
            "stochastic_evidence_status": claim_body.get(
                "stochastic_evidence_status"
            ),
            "residual_history_refs": [
                ref
                for ref in (
                    _ref_string(item)
                    for item in claim_body.get("residual_history_refs", ())
                )
                if ref is not None
            ],
            "stochastic_model_refs": [
                ref
                for ref in (
                    _ref_string(item)
                    for item in claim_body.get("stochastic_model_refs", ())
                )
                if ref is not None
            ],
            "allowed_interpretation_codes": list(
                claim_body.get("allowed_interpretation_codes", ())
            ),
        },
        "catalog_entry": {
            "publication_mode": published.publication_mode,
            "forecast_object_type": published.forecast_object_type,
            "comparator_exposure_status": published.comparator_exposure_status,
            "scorecard_ref": _ref_string(published.scorecard_ref),
            "claim_card_ref": _ref_string(published.claim_card_ref),
            "abstention_ref": _ref_string(published.abstention_ref),
            "validation_scope_ref": _ref_string(published.validation_scope_ref),
            "primary_score_result_ref": _ref_string(published.primary_score_result_ref),
            "primary_calibration_result_ref": _ref_string(
                published.primary_calibration_result_ref
            ),
            "stochastic_support_status": published.stochastic_support_status,
            "residual_history_refs": [
                f"{ref.schema_name}:{ref.object_id}"
                for ref in published.residual_history_refs
            ],
            "stochastic_model_refs": [
                f"{ref.schema_name}:{ref.object_id}"
                for ref in published.stochastic_model_refs
            ],
        },
        "comparison": {
            "baseline_id": comparison.baseline_id,
            "comparison_class_status": comparison.comparison_class_status,
            "candidate_beats_baseline": comparison.candidate_beats_baseline,
            "candidate_primary_score": comparison.candidate_primary_score,
            "baseline_primary_score": comparison.baseline_primary_score,
            "score_delta": comparison.score_delta,
            "paired_comparison_records": list(comparison.paired_comparison_records),
        },
        "replay": {
            "bundle_ref": _ref_string(replay.summary.bundle_ref),
            "replay_verification_status": replay.summary.replay_verification_status,
            "required_manifest_refs": [
                f"{ref.schema_name}:{ref.object_id}"
                for ref in inspection.replay_bundle.required_manifest_refs
            ],
            "validation_scope_ref": _ref_string(validation_scope_ref),
            "artifact_hash_roles": [
                record.artifact_role
                for record in inspection.replay_bundle.artifact_hash_records
            ],
            "stage_order": list(inspection.replay_bundle.recorded_stage_order),
        },
    }


def _expected_fixture(name: str) -> dict[str, Any]:
    path = FIXTURE_ROOT / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_probabilistic_distribution_publication_matches_golden_fixture(
    tmp_path: Path,
) -> None:
    assert _probabilistic_publication_snapshot(
        output_root=tmp_path / "probabilistic-publication-output",
        manifest_path=DISTRIBUTION_MANIFEST,
    ) == _expected_fixture("probabilistic-distribution-publication-golden.json")


def test_probabilistic_distribution_downgrade_matches_golden_fixture(
    tmp_path: Path,
) -> None:
    assert _probabilistic_publication_snapshot(
        output_root=tmp_path / "probabilistic-downgrade-output",
        manifest_path=CALIBRATION_FAILURE_MANIFEST,
    ) == _expected_fixture("probabilistic-distribution-downgrade-golden.json")
