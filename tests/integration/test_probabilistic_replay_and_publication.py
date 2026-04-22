from __future__ import annotations

from pathlib import Path

import euclid
from euclid.contracts.refs import TypedRef

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "fixtures/runtime/phase06"
DISTRIBUTION_MANIFEST = FIXTURE_ROOT / "probabilistic-distribution-demo.yaml"
QUANTILE_MANIFEST = FIXTURE_ROOT / "probabilistic-quantile-demo.yaml"
CALIBRATION_FAILURE_MANIFEST = (
    FIXTURE_ROOT / "probabilistic-distribution-calibration-failure-demo.yaml"
)


def test_distribution_run_replays_and_publishes(tmp_path: Path) -> None:
    _assert_probabilistic_run_replays_and_publishes(
        output_root=tmp_path / "distribution",
        manifest_path=DISTRIBUTION_MANIFEST,
        forecast_object_type="distribution",
    )


def test_quantile_run_replays_and_publishes(tmp_path: Path) -> None:
    _assert_probabilistic_run_replays_and_publishes(
        output_root=tmp_path / "quantile",
        manifest_path=QUANTILE_MANIFEST,
        forecast_object_type="quantile",
    )


def test_failed_probabilistic_support_downgrades_or_abstains_correctly(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "distribution-calibration-failure"
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=CALIBRATION_FAILURE_MANIFEST,
        output_root=output_root,
    )
    replay = euclid.replay_demo(output_root=output_root)
    entry = euclid.publish_demo_run_to_catalog(output_root=output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=output_root,
        publication_id=entry.publication_id,
    )
    run_result, validation_scope, scope_ledger = _runtime_scope_artifacts(
        output_root=output_root,
        run_id=result.summary.run_result_ref.object_id,
    )

    assert result.summary.forecast_object_type == "distribution"
    assert result.calibration.status == "failed"
    assert replay.summary.replay_verification_status == "verified"
    assert entry.replay_verification_status == "verified"
    assert entry.publication_mode == "candidate_publication"
    assert entry.validation_scope_ref == validation_scope.ref
    assert (
        run_result.body["primary_validation_scope_ref"]
        == validation_scope.ref.as_dict()
    )
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == "descriptive_structure"
    assert inspection.claim_card.manifest.body["predictive_support_status"] == "blocked"
    assert inspection.scorecard is not None
    assert inspection.scorecard.manifest.body["predictive_status"] == "blocked"
    assert inspection.scorecard.manifest.body["predictive_reason_codes"] == [
        "calibration_failed"
    ]
    assert (
        validation_scope.body["run_support_object_ids"]
        == scope_ledger.body["run_support_object_ids"]
    )
    assert (
        validation_scope.body["admissibility_rule_ids"]
        == scope_ledger.body["admissibility_rule_ids"]
    )
    assert validation_scope.ref in inspection.replay_bundle.required_manifest_refs
    assert "validation_scope" in {
        record.artifact_role
        for record in inspection.replay_bundle.artifact_hash_records
    }


def _assert_probabilistic_run_replays_and_publishes(
    *,
    output_root: Path,
    manifest_path: Path,
    forecast_object_type: str,
) -> None:
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=manifest_path,
        output_root=output_root,
    )
    replay = euclid.replay_demo(output_root=output_root)
    entry = euclid.publish_demo_run_to_catalog(output_root=output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=output_root,
        publication_id=entry.publication_id,
    )
    run_result, validation_scope, scope_ledger = _runtime_scope_artifacts(
        output_root=output_root,
        run_id=result.summary.run_result_ref.object_id,
    )

    assert result.summary.forecast_object_type == forecast_object_type
    assert run_result.body["forecast_object_type"] == forecast_object_type
    assert run_result.body["result_mode"] == "candidate_publication"
    assert (
        run_result.body["primary_validation_scope_ref"]
        == validation_scope.ref.as_dict()
    )
    assert replay.summary.replay_verification_status == "verified"
    assert entry.replay_verification_status == "verified"
    assert entry.publication_mode == "candidate_publication"
    assert entry.forecast_object_type == forecast_object_type
    assert entry.validation_scope_ref == validation_scope.ref
    assert entry.primary_score_result_ref == result.summary.score_result_ref
    assert entry.primary_calibration_result_ref == result.summary.calibration_result_ref
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == "predictive_within_declared_scope"
    assert (
        inspection.claim_card.manifest.body["predictive_support_status"]
        == "confirmatory_supported"
    )
    assert validation_scope.body["public_forecast_object_type"] == forecast_object_type
    assert validation_scope.body["scope_ledger_ref"] == scope_ledger.ref.as_dict()
    assert (
        validation_scope.body["run_support_object_ids"]
        == scope_ledger.body["run_support_object_ids"]
    )
    assert (
        validation_scope.body["admissibility_rule_ids"]
        == scope_ledger.body["admissibility_rule_ids"]
    )
    assert inspection.replay_bundle.replay_verification_status == "verified"
    assert validation_scope.ref in inspection.replay_bundle.required_manifest_refs
    assert {
        result.summary.prediction_artifact_ref,
        result.summary.score_result_ref,
        result.summary.calibration_result_ref,
        validation_scope.ref,
    } <= set(inspection.replay_bundle.required_manifest_refs)
    assert "validation_scope" in {
        record.artifact_role
        for record in inspection.replay_bundle.artifact_hash_records
    }


def _runtime_scope_artifacts(*, output_root: Path, run_id: str):
    graph = euclid.load_demo_run_artifact_graph(output_root=output_root, run_id=run_id)
    run_result = graph.inspect(graph.root_ref).manifest
    validation_scope = graph.inspect(
        TypedRef(**run_result.body["primary_validation_scope_ref"])
    ).manifest
    scope_ledger = graph.inspect(
        TypedRef(**run_result.body["scope_ledger_ref"])
    ).manifest
    return run_result, validation_scope, scope_ledger
