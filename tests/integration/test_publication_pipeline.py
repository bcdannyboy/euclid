from __future__ import annotations

from pathlib import Path

import pytest

from euclid.artifacts import FilesystemArtifactStore
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.control_plane import SQLiteMetadataStore
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    ArtifactHashRecord,
    ReadinessJudgmentManifest,
    ReproducibilityBundleManifest,
    SchemaLifecycleIntegrationClosureManifest,
    SeedRecord,
)
from euclid.manifest_registry import ManifestRegistry
from euclid.modules.catalog_publishing import (
    build_publication_record_manifest,
    build_run_result_manifest,
)
from euclid.modules.evaluation_governance import (
    ComparisonKey,
    build_comparison_universe,
)
from euclid.prototype.workflow import run_prototype_reducer_workflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ref(schema_name: str, object_id: str) -> TypedRef:
    return TypedRef(schema_name=schema_name, object_id=object_id)


def _comparison_universe_manifest():
    catalog = load_contract_catalog(PROJECT_ROOT)
    key = ComparisonKey(
        forecast_object_type="point",
        score_policy_ref=_ref("point_score_policy_manifest@1.0.0", "score_policy"),
        horizon_set=(1,),
        scored_origin_set_id="origins",
    )
    universe = build_comparison_universe(
        selected_candidate_id="candidate",
        baseline_id="constant_baseline",
        candidate_primary_score=0.1,
        baseline_primary_score=0.2,
        candidate_comparison_key=key,
        baseline_comparison_key=key,
        candidate_score_result_ref=_ref(
            "point_score_result_manifest@1.0.0", "candidate_score"
        ),
        baseline_score_result_ref=_ref(
            "point_score_result_manifest@1.0.0", "baseline_score"
        ),
        comparator_score_result_refs=(
            _ref("point_score_result_manifest@1.0.0", "baseline_score"),
        ),
        paired_comparison_records=(
            {
                "comparator_id": "constant_baseline",
                "comparator_kind": "baseline",
                "comparison_status": "comparable",
                "candidate_primary_score": 0.1,
                "comparator_primary_score": 0.2,
                "primary_score_delta": -0.1,
                "mean_loss_differential": -0.1,
                "score_result_ref": _ref(
                    "point_score_result_manifest@1.0.0", "baseline_score"
                ).as_dict(),
            },
        ),
    )
    return universe.to_manifest(catalog)


def _bundle_manifest():
    catalog = load_contract_catalog(PROJECT_ROOT)
    bundle = ReproducibilityBundleManifest(
        object_id="demo_bundle",
        bundle_id="demo_bundle",
        scope_id="euclid_v1_binding_scope@1.0.0",
        bundle_mode="candidate_publication",
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
        required_manifest_refs=(
            _ref("reducer_artifact_manifest@1.0.0", "candidate"),
            _ref("scorecard_manifest@1.1.0", "scorecard"),
            _ref("claim_card_manifest@1.1.0", "claim"),
        ),
        artifact_hash_records=(
            ArtifactHashRecord(
                artifact_role="dataset_snapshot",
                sha256="sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            ),
        ),
        seed_records=(SeedRecord(seed_scope="search", seed_value="0"),),
        replay_verification_status="verified",
        failure_reason_codes=(),
    )
    return bundle.to_manifest(catalog)


def _readiness_manifest():
    catalog = load_contract_catalog(PROJECT_ROOT)
    return ReadinessJudgmentManifest(
        object_id="demo_readiness",
        judgment_id="demo_readiness",
        final_verdict="ready",
        catalog_scope="public",
    ).to_manifest(catalog)


def _closure_ref() -> TypedRef:
    catalog = load_contract_catalog(PROJECT_ROOT)
    return SchemaLifecycleIntegrationClosureManifest(
        object_id="demo_closure",
        closure_id="demo_closure",
        status="passed",
    ).to_manifest(catalog).ref


def test_publication_record_validates_candidate_claim_card_scope() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    run_result = build_run_result_manifest(
        object_id="run_result",
        run_id="run_result",
        scope_ledger_ref=_ref("scope_ledger_manifest@1.0.0", "scope"),
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
        reproducibility_bundle_ref=_ref(
            "reproducibility_bundle_manifest@1.0.0", "demo_bundle"
        ),
        publication_mode="candidate_publication",
        selected_candidate_ref=_ref("reducer_artifact_manifest@1.0.0", "candidate"),
        scorecard_ref=_ref("scorecard_manifest@1.1.0", "scorecard"),
        claim_card_ref=_ref("claim_card_manifest@1.1.0", "claim"),
    ).to_manifest(catalog)
    claim_card = ManifestEnvelope.build(
        schema_name="claim_card_manifest@1.1.0",
        module_id="claims",
        object_id="claim",
        catalog=catalog,
        body={
            "claim_card_id": "claim",
            "candidate_ref": _ref(
                "reducer_artifact_manifest@1.0.0", "candidate"
            ).as_dict(),
            "scorecard_ref": _ref("scorecard_manifest@1.1.0", "scorecard").as_dict(),
            "validation_scope_ref": _ref(
                "validation_scope_manifest@1.0.0", "validation"
            ).as_dict(),
            "claim_type": "descriptive_structure",
            "claim_ceiling": "descriptive_structure",
            "claim_text": "universal invariant law across all future systems",
            "predictive_support_status": "blocked",
            "allowed_interpretation_codes": ["historical_structure_summary"],
            "forbidden_interpretation_codes": ["universal_claim"],
        },
    )

    with pytest.raises(ContractValidationError) as excinfo:
        build_publication_record_manifest(
            object_id="publication_record",
            publication_id="publication_record",
            run_result_manifest=run_result,
            comparison_universe_manifest=_comparison_universe_manifest(),
            reproducibility_bundle_manifest=_bundle_manifest(),
            readiness_judgment_manifest=_readiness_manifest(),
            schema_lifecycle_integration_closure_ref=_closure_ref(),
            catalog_scope="public",
            published_at="2026-04-26T00:00:00Z",
            claim_card_manifest=claim_card,
        )

    assert excinfo.value.code == "claim_scope_overstatement"


def test_abstention_publication_stays_free_of_candidate_refs(tmp_path: Path) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
    )

    result = run_prototype_reducer_workflow(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
        minimum_description_gain_bits=10_000.0,
    )

    run_result_body = result.run_result.manifest.body
    assert run_result_body["result_mode"] == "abstention_only_publication"
    assert "primary_reducer_artifact_ref" not in run_result_body
    assert "primary_scorecard_ref" not in run_result_body
    assert "primary_claim_card_ref" not in run_result_body
    assert result.publication_record.manifest.body["comparator_exposure_status"] == (
        "not_applicable_abstention_only"
    )
