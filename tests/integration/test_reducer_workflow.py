from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

from euclid.artifacts import FilesystemArtifactStore
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.control_plane import SQLiteMetadataStore
from euclid.manifest_registry import ManifestRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _typed_ref(payload: dict[str, str]) -> TypedRef:
    return TypedRef(
        schema_name=str(payload["schema_name"]),
        object_id=str(payload["object_id"]),
    )


def _load_workflow_module():
    try:
        return import_module("euclid.prototype.workflow")
    except ModuleNotFoundError as exc:
        pytest.fail(f"prototype workflow module is missing: {exc}")


def _build_registry(tmp_path: Path) -> ManifestRegistry:
    catalog = load_contract_catalog(PROJECT_ROOT)
    return ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
    )


def test_run_prototype_reducer_workflow_emits_sealed_candidate_bundle_and_replays(
    tmp_path: Path,
) -> None:
    workflow = _load_workflow_module()
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = _build_registry(tmp_path)

    result = workflow.run_prototype_reducer_workflow(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
    )

    assert {summary.family_id for summary in result.candidate_summaries} == {
        "constant",
        "drift",
        "seasonal_naive",
        "linear_trend",
    }
    assert (
        result.baseline_registry.manifest.body["baseline_declarations"][0][
            "baseline_id"
        ]
        == "constant_baseline"
    )
    assert result.forecast_comparison_policy.manifest.body[
        "required_comparison_key_fields"
    ] == [
        "forecast_object_type",
        "score_policy_ref",
        "horizon_set",
        "scored_origin_set_id",
    ]
    assert result.frontier.manifest.schema_name == "frontier_manifest@1.0.0"
    assert result.frontier.manifest.body["frontier_axes"] == [
        "structure_code_bits",
        "description_gain_bits",
        "inner_primary_score",
    ]
    assert result.rejected_diagnostics.manifest.schema_name == (
        "rejected_diagnostics_manifest@1.0.0"
    )
    assert all(
        "confirmatory_primary_score" not in record
        and "baseline_primary_score" not in record
        for record in result.search_ledger.manifest.body["candidates"]
    )
    assert (
        result.search_ledger.manifest.body["budget_accounting"]["accounting_status"]
        == "within_budget"
    )
    assert result.frozen_shortlist.manifest.body["selection_rule"] == (
        "single_lowest_total_bits_admissible_candidate"
    )
    assert (
        result.freeze_event.manifest.body["holdout_materialized_before_freeze"] is False
    )
    assert (
        result.freeze_event.manifest.body["post_freeze_candidate_mutation_count"] == 0
    )
    assert result.comparison_universe.manifest.body["comparison_class_status"] == (
        "comparable"
    )
    assert (
        result.comparison_universe.manifest.body["candidate_comparison_key"][
            "scored_origin_set_id"
        ]
        == result.intake.evaluation_plan.manifest.body["scored_origin_set_id"]
    )
    assert result.evaluation_governance.manifest.body["comparison_regime_id"] == (
        "unconditional_model_pair"
    )
    assert result.search_ledger.manifest.schema_name == "search_ledger_manifest@1.0.0"
    assert (
        result.prediction_artifact.manifest.body["stage_id"] == "confirmatory_holdout"
    )
    assert result.point_score_result.manifest.body["comparison_status"] == "comparable"
    robustness_body = result.robustness_report.manifest.body
    null_result = registry.resolve(
        _typed_ref(robustness_body["null_result_ref"])
    ).manifest.body
    perturbation_family_results = tuple(
        registry.resolve(_typed_ref(payload)).manifest.body
        for payload in robustness_body["perturbation_family_result_refs"]
    )
    sensitivity_analyses = tuple(
        registry.resolve(_typed_ref(payload)).manifest.body
        for payload in robustness_body["sensitivity_analysis_refs"]
    )
    assert (
        robustness_body["candidate_id"]
        == result.selected_candidate.manifest.body["candidate_id"]
    )
    assert null_result["status"] in {
        "rejected",
        "null_not_rejected",
    }
    assert len(null_result["surrogate_statistics"]) == 19
    assert {item["family_id"] for item in perturbation_family_results} == {
        "recent_history_truncation",
        "quantization_coarsening",
    }
    assert len(robustness_body["aggregate_metric_results"]) == 3
    assert len(robustness_body["leakage_canary_result_refs"]) == 4
    assert robustness_body["required_canary_type_coverage"] == [
        "future_target_level_feature",
        "late_available_target_copy",
        "holdout_membership_feature",
        "post_cutoff_revision_level_feature",
    ]
    assert robustness_body["final_robustness_status"] in {
        "passed",
        "failed",
        "abstained",
    }
    assert len(sensitivity_analyses) == 4
    assert result.reproducibility_bundle.manifest.body[
        "replay_verification_status"
    ] == ("verified")
    assert result.reproducibility_bundle.manifest.body["seed_records"] == [
        {"seed_scope": "search", "seed_value": "0"},
        {"seed_scope": "surrogate_generation", "seed_value": "0"},
        {"seed_scope": "perturbation", "seed_value": "0"},
    ]
    assert (
        "python_version"
        in result.reproducibility_bundle.manifest.body["environment_metadata"]
    )
    assert (
        result.reproducibility_bundle.manifest.body["stage_order_records"][0][
            "stage_id"
        ]
        == "dataset_snapshot_frozen"
    )
    assert result.run_result.manifest.body["result_mode"] == (
        "abstention_only_publication"
    )
    assert (
        result.publication_record.manifest.body["publication_mode"]
        == result.run_result.manifest.body["result_mode"]
    )
    assert result.publication_record.manifest.body["run_result_ref"] == (
        result.run_result.manifest.ref.as_dict()
    )
    assert result.claim_card is None
    assert result.abstention is not None
    assert result.abstention.manifest.body["abstention_type"] == "robustness_failed"

    fresh_registry = _build_registry(tmp_path)
    replay = workflow.replay_prototype_run(
        bundle_ref=result.reproducibility_bundle.manifest.ref,
        catalog=catalog,
        registry=fresh_registry,
    )

    assert replay.bundle_ref == result.reproducibility_bundle.manifest.ref
    assert replay.run_result_ref == result.run_result.manifest.ref
    assert replay.selected_candidate_ref == result.selected_candidate.manifest.ref
    assert replay.replay_verification_status == "verified"
    assert replay.confirmatory_primary_score == pytest.approx(
        result.confirmatory_primary_score
    )


def test_run_workflow_emits_typed_abstention_when_floor_blocks_all_candidates(
    tmp_path: Path,
) -> None:
    workflow = _load_workflow_module()
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = _build_registry(tmp_path)

    result = workflow.run_prototype_reducer_workflow(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
        minimum_description_gain_bits=10_000.0,
    )

    assert result.claim_card is None
    assert result.abstention is not None
    assert result.abstention.manifest.body["abstention_type"] == "no_admissible_reducer"
    assert (
        result.run_result.manifest.body["result_mode"] == "abstention_only_publication"
    )
    assert (
        result.publication_record.manifest.body["publication_mode"]
        == "abstention_only_publication"
    )
    assert result.reproducibility_bundle.manifest.body["bundle_mode"] == (
        "abstention_only_publication"
    )

    fresh_registry = _build_registry(tmp_path)
    replay = workflow.replay_prototype_run(
        bundle_ref=result.reproducibility_bundle.manifest.ref,
        catalog=catalog,
        registry=fresh_registry,
    )

    assert replay.replay_verification_status == "verified"
    assert replay.run_result_ref == result.run_result.manifest.ref


def test_run_workflow_blocks_candidate_publication_when_robustness_fails(
    tmp_path: Path,
) -> None:
    workflow = _load_workflow_module()
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = _build_registry(tmp_path)

    result = workflow.run_prototype_reducer_workflow(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
    )

    assert result.robustness_report.manifest.body["final_robustness_status"] == "failed"
    assert result.scorecard.manifest.body["descriptive_status"] == (
        "blocked_robustness_failed"
    )
    assert result.claim_card is None
    assert result.abstention is not None
    assert result.abstention.manifest.body["abstention_type"] == "robustness_failed"
    assert (
        result.run_result.manifest.body["result_mode"] == "abstention_only_publication"
    )
