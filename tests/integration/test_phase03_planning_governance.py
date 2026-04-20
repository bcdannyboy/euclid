from __future__ import annotations

from pathlib import Path

from euclid.artifacts import FilesystemArtifactStore
from euclid.contracts.loader import load_contract_catalog
from euclid.control_plane import SQLiteMetadataStore
from euclid.manifest_registry import ManifestRegistry
from euclid.prototype.intake_planning import build_prototype_intake_plan
from euclid.prototype.workflow import run_prototype_reducer_workflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_registry(tmp_path: Path) -> ManifestRegistry:
    catalog = load_contract_catalog(PROJECT_ROOT)
    return ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
    )


def test_intake_plan_registers_split_and_search_freeze_lineage(
    tmp_path: Path,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = _build_registry(tmp_path)

    result = build_prototype_intake_plan(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
    )

    assert set(registry.list_lineage_parents(result.evaluation_plan.manifest.ref)) == {
        result.time_safety_audit.manifest.ref,
        result.feature_view.manifest.ref,
    }
    assert set(registry.list_lineage_parents(result.search_plan.manifest.ref)) == {
        result.canonicalization_policy.manifest.ref,
        result.codelength_policy.manifest.ref,
        result.reference_description_policy.manifest.ref,
        result.observation_model.manifest.ref,
        result.evaluation_plan.manifest.ref,
    }
    assert result.evaluation_plan.manifest.body["comparison_key"] == {
        "forecast_object_type": "point",
        "horizon_set": [1],
        "scored_origin_set_id": result.evaluation_plan_object.scored_origin_set_id,
    }


def test_reducer_workflow_binds_comparison_universe_to_frozen_search_plan(
    tmp_path: Path,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = _build_registry(tmp_path)

    result = run_prototype_reducer_workflow(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
    )

    candidate_key = result.comparison_universe.manifest.body["candidate_comparison_key"]
    baseline_key = result.comparison_universe.manifest.body["baseline_comparison_key"]

    assert candidate_key == baseline_key
    assert candidate_key["forecast_object_type"] == "point"
    assert candidate_key["horizon_set"] == [1]
    assert candidate_key["scored_origin_set_id"] == (
        result.intake.evaluation_plan.manifest.body["scored_origin_set_id"]
    )
    event_kinds = [
        event["ref_kind"]
        for event in result.evaluation_event_log.manifest.body["events"]
    ]

    assert event_kinds == [
        "search_plan",
        "frozen_shortlist",
        "freeze_event",
        "comparison_universe",
        "freeze_event",
        "prediction_artifact",
        "run_result",
    ]
    assert result.evaluation_event_log.manifest.body["stage_bookkeeping"] == [
        {
            "stage_id": "inner_search",
            "stage_kind": "search_local_fitting",
            "segment_ids": [
                segment.segment_id
                for segment in result.intake.evaluation_plan_object.development_segments
            ],
            "status": "completed",
            "uses_confirmatory_holdout": False,
        },
        {
            "stage_id": "global_pair_freeze_pre_holdout",
            "stage_kind": "shortlist_freeze",
            "segment_ids": ["global_pair_freeze_pre_holdout"],
            "status": "completed",
            "uses_confirmatory_holdout": False,
        },
        {
            "stage_id": "confirmatory_holdout",
            "stage_kind": "confirmatory_access",
            "segment_ids": [
                result.intake.evaluation_plan_object.confirmatory_segment.segment_id
            ],
            "status": "materialized",
            "access_count": 1,
            "uses_confirmatory_holdout": True,
        },
    ]
    assert set(
        registry.list_lineage_parents(result.evaluation_governance.manifest.ref)
    ) == {
        result.comparison_universe.manifest.ref,
        result.evaluation_event_log.manifest.ref,
        result.freeze_event.manifest.ref,
        result.frozen_shortlist.manifest.ref,
    }
