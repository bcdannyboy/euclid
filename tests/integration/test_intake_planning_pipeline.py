from __future__ import annotations

from pathlib import Path

from euclid.artifacts import FilesystemArtifactStore
from euclid.contracts.loader import load_contract_catalog
from euclid.control_plane import SQLiteMetadataStore
from euclid.manifest_registry import ManifestRegistry
from euclid.prototype.intake_planning import build_prototype_intake_plan
from euclid.runtime.hashing import sha256_digest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_build_prototype_intake_plan_registers_retained_scope_slice(tmp_path) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
    )

    result = build_prototype_intake_plan(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
    )

    assert len(result.observation_records) == 6
    assert result.snapshot.manifest.schema_name == "dataset_snapshot_manifest@1.0.0"
    assert result.time_safety_audit.manifest.body["status"] == "passed"
    assert result.time_safety_audit.manifest.body["causal_availability_window"][
        "coordinate_count"
    ] == 6
    assert result.feature_view.manifest.body["feature_names"] == ["lag_1"]
    assert result.feature_view.manifest.body["materialization_report"][
        "reusable_stage_ids"
    ] == ["search", "candidate_fitting", "evaluation"]
    assert result.canonicalization_policy.manifest.schema_name == (
        "canonicalization_policy_manifest@1.0.0"
    )
    assert result.search_plan.manifest.schema_name == "search_plan_manifest@1.0.0"
    assert result.search_plan.manifest.body["search_class"] == "bounded_heuristic"
    assert result.search_plan.manifest.body["search_budget"] == {
        "proposal_limit": 4,
        "frontier_width": 4,
        "shortlist_limit": 1,
        "wall_clock_budget_seconds": 1,
        "budget_accounting_rule": "proposal_count_then_candidate_id_tie_break",
    }
    assert result.search_plan.manifest.body["seed_policy"] == {
        "root_seed": "0",
        "seed_derivation_rule": "deterministic_scope_hash",
        "seed_scopes": ["search", "candidate_generation", "tie_break"],
    }
    assert result.search_plan.manifest.body["parallel_budget"] == {
        "max_worker_count": 1,
        "candidate_batch_size": 1,
        "aggregation_rule": "deterministic_candidate_id_order",
    }
    assert result.search_plan.manifest.body["frontier_policy"]["axes"] == [
        "structure_code_bits",
        "description_gain_bits",
        "inner_primary_score",
    ]
    assert result.search_plan.manifest.body["frontier_policy"]["forbidden_axes"] == [
        "holdout_results",
        "outer_fold_results",
        "null_results",
        "robustness_results",
    ]
    assert result.search_plan.manifest.body["search_time_predictive_policy"] == (
        "fold_local_only"
    )
    assert result.evaluation_plan.manifest.body["outer_fold_count"] == 2
    assert result.evaluation_plan.manifest.body["horizon_weights"] == [
        {"horizon": 1, "weight": "1"}
    ]
    assert result.evaluation_plan.manifest.body["development_segments"][0]["role"] == (
        "development"
    )
    assert (
        result.evaluation_plan.manifest.body["confirmatory_segment"]["role"]
        == "confirmatory_holdout"
    )
    assert result.evaluation_plan.manifest.body["comparison_key"] == {
        "forecast_object_type": "point",
        "horizon_set": [1],
        "scored_origin_set_id": (
            result.evaluation_plan.manifest.body["scored_origin_set_id"]
        ),
    }
    assert result.reference_description.quantized_sequence == (
        20,
        24,
        27,
        30,
        29,
        32,
    )
    assert result.target_transform_object.manifest.ref == (
        result.target_transform.manifest.ref
    )
    assert result.quantization_object.manifest.ref == (
        result.codelength_policy.manifest.ref
    )
    assert result.quantization_object.step_string == "0.5"
    assert result.observation_model_object.manifest.ref == (
        result.observation_model.manifest.ref
    )
    assert result.reference_description_object.reference_bits > 0
    assert result.codelength_policy_object.manifest.ref == (
        result.codelength_policy.manifest.ref
    )
    assert result.snapshot_object.sampling_metadata.raw_row_count == 6
    assert result.snapshot_object.sampling_metadata.coded_row_count == 6
    assert result.snapshot.manifest.body["raw_observation_hash"] == sha256_digest(
        result.snapshot.manifest.body["raw_observations"]
    )
    assert result.snapshot.manifest.body["coded_target_hash"] == sha256_digest(
        result.snapshot.manifest.body["coded_targets"]
    )
    assert set(registry.list_lineage_parents(result.snapshot.manifest.ref)) == {
        item.manifest.ref for item in result.observation_manifests
    }
    assert registry.resolve(result.evaluation_plan.manifest.ref).manifest == (
        result.evaluation_plan.manifest
    )
    assert (
        result.feature_view_object.require_stage_reuse("search")
        is result.feature_view_object
    )
