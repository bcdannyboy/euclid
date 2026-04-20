from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from euclid.artifacts import FilesystemArtifactStore
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.control_plane import SQLiteMetadataStore
from euclid.manifest_registry import ManifestRegistry
from euclid.manifests.runtime_models import (
    ArtifactHashRecord,
    ReproducibilityBundleManifest,
)
from euclid.modules.replay import (
    ReplayedOutcome,
    build_replay_stage_order,
    build_reproducibility_bundle_manifest,
    inspect_reproducibility_bundle,
    required_manifest_refs_from_run_result,
    verify_replayed_outcome,
)
from euclid.prototype.workflow import run_prototype_reducer_workflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _ref(schema_name: str, object_id: str) -> TypedRef:
    return TypedRef(schema_name=schema_name, object_id=object_id)


def _build_registry(tmp_path: Path) -> tuple[object, ManifestRegistry]:
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
    )
    return catalog, registry


def _run_workflow(tmp_path: Path):
    catalog, registry = _build_registry(tmp_path)
    result = run_prototype_reducer_workflow(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
    )
    return catalog, registry, result


def _matching_outcome(result) -> ReplayedOutcome:
    bundle = ReproducibilityBundleManifest.from_manifest(
        result.reproducibility_bundle.manifest
    )
    return ReplayedOutcome(
        selected_candidate_id=str(
            result.selected_candidate.manifest.body["candidate_id"]
        ),
        confirmatory_primary_score=result.confirmatory_primary_score,
        publication_mode=str(result.run_result.manifest.body["result_mode"]),
        descriptive_status=str(result.scorecard.manifest.body["descriptive_status"]),
        descriptive_reason_codes=tuple(
            str(item)
            for item in result.scorecard.manifest.body["descriptive_reason_codes"]
        ),
        predictive_status=str(result.scorecard.manifest.body["predictive_status"]),
        predictive_reason_codes=tuple(
            str(item)
            for item in result.scorecard.manifest.body["predictive_reason_codes"]
        ),
        replayed_stage_order=tuple(
            record.stage_id for record in bundle.stage_order_records
        ),
    )


def test_build_reproducibility_bundle_manifest_captures_environment_and_stage_order(
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    stage_order = build_replay_stage_order(
        dataset_snapshot_ref=_ref("dataset_snapshot_manifest@1.0.0", "snapshot"),
        feature_view_ref=_ref("feature_view_manifest@1.0.0", "feature_view"),
        search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
        evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event_log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        scorecard_ref=_ref("scorecard_manifest@1.1.0", "scorecard"),
        candidate_or_abstention_ref=_ref("abstention_manifest@1.1.0", "abstention"),
        run_result_ref=_ref("run_result_manifest@1.1.0", "run_result"),
    )
    model = build_reproducibility_bundle_manifest(
        object_id="demo_bundle",
        bundle_id="demo_bundle",
        bundle_mode="abstention_only_publication",
        dataset_snapshot_ref=_ref("dataset_snapshot_manifest@1.0.0", "snapshot"),
        feature_view_ref=_ref("feature_view_manifest@1.0.0", "feature_view"),
        search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
        evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event_log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        run_result_ref=_ref("run_result_manifest@1.1.0", "run_result"),
        required_manifest_refs=(_ref("abstention_manifest@1.1.0", "abstention"),),
        artifact_hash_records=(
            ArtifactHashRecord(
                artifact_role="dataset_snapshot",
                sha256="sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            ),
        ),
        stage_order_records=stage_order,
    )

    inspection = inspect_reproducibility_bundle(model.to_manifest(catalog))

    assert inspection.environment_metadata["python_version"]
    assert inspection.environment_metadata["python_implementation"]
    assert inspection.stage_order_records == stage_order
    assert inspection.recorded_stage_order[0] == "dataset_snapshot_frozen"
    assert {record.seed_scope for record in inspection.seed_records} == {
        "perturbation",
        "search",
        "surrogate_generation",
    }


def test_build_replay_stage_order_and_required_refs_include_external_evidence() -> None:
    required_refs = required_manifest_refs_from_run_result(
        {
            "result_mode": "candidate_publication",
            "forecast_object_type": "point",
            "primary_reducer_artifact_ref": _ref(
                "reducer_artifact_manifest@1.0.0", "candidate"
            ).as_dict(),
            "primary_scorecard_ref": _ref(
                "scorecard_manifest@1.1.0", "scorecard"
            ).as_dict(),
            "primary_claim_card_ref": _ref(
                "claim_card_manifest@1.1.0", "claim_card"
            ).as_dict(),
            "primary_validation_scope_ref": _ref(
                "validation_scope_manifest@1.0.0", "validation_scope"
            ).as_dict(),
            "primary_external_evidence_ref": _ref(
                "external_evidence_manifest@1.0.0", "external_bundle"
            ).as_dict(),
            "primary_mechanistic_evidence_ref": _ref(
                "mechanistic_evidence_dossier_manifest@1.0.0", "mechanistic"
            ).as_dict(),
        }
    )
    stage_order = build_replay_stage_order(
        dataset_snapshot_ref=_ref("dataset_snapshot_manifest@1.0.0", "snapshot"),
        feature_view_ref=_ref("feature_view_manifest@1.0.0", "feature_view"),
        search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
        evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event_log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        scorecard_ref=_ref("scorecard_manifest@1.1.0", "scorecard"),
        candidate_or_abstention_ref=_ref("claim_card_manifest@1.1.0", "claim_card"),
        run_result_ref=_ref("run_result_manifest@1.1.0", "run_result"),
        external_evidence_ref=_ref(
            "external_evidence_manifest@1.0.0", "external_bundle"
        ),
        mechanistic_evidence_ref=_ref(
            "mechanistic_evidence_dossier_manifest@1.0.0", "mechanistic"
        ),
    )

    assert _ref(
        "external_evidence_manifest@1.0.0", "external_bundle"
    ) in required_refs
    assert _ref(
        "validation_scope_manifest@1.0.0", "validation_scope"
    ) in required_refs
    assert [record.stage_id for record in stage_order][-4:] == [
        "external_evidence_resolved",
        "mechanistic_evidence_resolved",
        "publication_decision_resolved",
        "run_result_assembled",
    ]


def test_required_refs_include_external_evidence_for_abstention_runs() -> None:
    required_refs = required_manifest_refs_from_run_result(
        {
            "result_mode": "abstention_only_publication",
            "forecast_object_type": "point",
            "primary_abstention_ref": _ref(
                "abstention_manifest@1.1.0", "abstention"
            ).as_dict(),
            "primary_validation_scope_ref": _ref(
                "validation_scope_manifest@1.0.0", "validation_scope"
            ).as_dict(),
            "primary_external_evidence_ref": _ref(
                "external_evidence_manifest@1.0.0", "external_bundle"
            ).as_dict(),
        }
    )

    assert required_refs == (
        _ref("abstention_manifest@1.1.0", "abstention"),
        _ref("validation_scope_manifest@1.0.0", "validation_scope"),
        _ref("external_evidence_manifest@1.0.0", "external_bundle"),
    )


def test_verify_replayed_outcome_detects_hash_mismatch(tmp_path: Path) -> None:
    catalog, registry, result = _run_workflow(tmp_path)
    bundle_model = ReproducibilityBundleManifest.from_manifest(
        result.reproducibility_bundle.manifest
    )
    tampered_hashes = tuple(
        ArtifactHashRecord(
            artifact_role=record.artifact_role,
            sha256=(
                "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                if record.artifact_role == "run_result"
                else record.sha256
            ),
        )
        for record in bundle_model.artifact_hash_records
    )
    tampered_bundle = registry.register(
        build_reproducibility_bundle_manifest(
            object_id="tampered_bundle",
            bundle_id="tampered_bundle",
            bundle_mode=bundle_model.bundle_mode,
            dataset_snapshot_ref=bundle_model.dataset_snapshot_ref,
            feature_view_ref=bundle_model.feature_view_ref,
            search_plan_ref=bundle_model.search_plan_ref,
            evaluation_plan_ref=bundle_model.evaluation_plan_ref,
            comparison_universe_ref=bundle_model.comparison_universe_ref,
            evaluation_event_log_ref=bundle_model.evaluation_event_log_ref,
            evaluation_governance_ref=bundle_model.evaluation_governance_ref,
            run_result_ref=bundle_model.run_result_ref,
            required_manifest_refs=bundle_model.required_manifest_refs,
            artifact_hash_records=tampered_hashes,
            seed_records=bundle_model.seed_records,
            environment_metadata=bundle_model.environment_metadata,
            stage_order_records=bundle_model.stage_order_records,
            replay_verification_status="failed",
            failure_reason_codes=("artifact_hash_mismatch",),
        ).to_manifest(catalog),
    )

    verification = verify_replayed_outcome(
        bundle=tampered_bundle,
        registry=registry,
        outcome=_matching_outcome(result),
    )

    assert verification.replay_verification_status == "failed"
    assert verification.failure_reason_codes == ("artifact_hash_mismatch",)


def test_verify_replayed_outcome_detects_missing_seed_record(tmp_path: Path) -> None:
    catalog, registry, result = _run_workflow(tmp_path)
    bundle_model = ReproducibilityBundleManifest.from_manifest(
        result.reproducibility_bundle.manifest
    )
    tampered_bundle = registry.register(
        build_reproducibility_bundle_manifest(
            object_id="seedless_bundle",
            bundle_id="seedless_bundle",
            bundle_mode=bundle_model.bundle_mode,
            dataset_snapshot_ref=bundle_model.dataset_snapshot_ref,
            feature_view_ref=bundle_model.feature_view_ref,
            search_plan_ref=bundle_model.search_plan_ref,
            evaluation_plan_ref=bundle_model.evaluation_plan_ref,
            comparison_universe_ref=bundle_model.comparison_universe_ref,
            evaluation_event_log_ref=bundle_model.evaluation_event_log_ref,
            evaluation_governance_ref=bundle_model.evaluation_governance_ref,
            run_result_ref=bundle_model.run_result_ref,
            required_manifest_refs=bundle_model.required_manifest_refs,
            artifact_hash_records=bundle_model.artifact_hash_records,
            seed_records=(bundle_model.seed_records[0],),
            environment_metadata=bundle_model.environment_metadata,
            stage_order_records=bundle_model.stage_order_records,
            replay_verification_status="failed",
            failure_reason_codes=("missing_seed_record",),
        ).to_manifest(catalog),
    )

    verification = verify_replayed_outcome(
        bundle=tampered_bundle,
        registry=registry,
        outcome=_matching_outcome(result),
    )

    assert verification.replay_verification_status == "failed"
    assert verification.failure_reason_codes == ("missing_seed_record",)


def test_verify_replayed_outcome_detects_stage_order_mismatch(tmp_path: Path) -> None:
    _, registry, result = _run_workflow(tmp_path)
    outcome = _matching_outcome(result)

    verification = verify_replayed_outcome(
        bundle=result.reproducibility_bundle,
        registry=registry,
        outcome=replace(
            outcome,
            replayed_stage_order=tuple(reversed(outcome.replayed_stage_order)),
        ),
    )

    assert verification.replay_verification_status == "failed"
    assert verification.failure_reason_codes == ("nondeterministic_replay",)
