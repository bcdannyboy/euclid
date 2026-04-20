from __future__ import annotations

from pathlib import Path

import euclid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_MANIFEST = PROJECT_ROOT / "fixtures/runtime/prototype-demo.yaml"
PROBABILISTIC_MANIFEST = (
    PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-distribution-demo.yaml"
)


def test_load_demo_run_artifact_graph_returns_notebook_friendly_graph(
    phase01_demo_output_root: Path,
) -> None:
    graph = euclid.load_demo_run_artifact_graph(output_root=phase01_demo_output_root)

    assert graph.run_id == "prototype_constant_candidate_run_result_v1"
    assert graph.root_ref == euclid.TypedRef(
        "run_result_manifest@1.1.0",
        "prototype_constant_candidate_run_result_v1",
    )
    assert graph.manifest_count > 5
    assert graph.inspect(graph.root_ref).manifest.body["result_mode"] == (
        "abstention_only_publication"
    )
    assert graph.children_for(graph.root_ref)
    assert graph.parents_for(graph.root_ref)
    assert any(
        parent.schema_name.endswith("manifest@1.0.0")
        or parent.schema_name.endswith("manifest@1.1.0")
        for parent in graph.parents_for(graph.root_ref)
    )


def test_resolve_demo_artifact_returns_relationships_for_typed_ref(
    phase01_demo_output_root: Path,
) -> None:
    record = euclid.resolve_demo_artifact(
        output_root=phase01_demo_output_root,
        ref="run_result_manifest@1.1.0:prototype_constant_candidate_run_result_v1",
    )

    assert record.ref == euclid.TypedRef(
        "run_result_manifest@1.1.0",
        "prototype_constant_candidate_run_result_v1",
    )
    assert record.metadata.run_id == "prototype_constant_candidate_run_result_v1"
    assert record.children
    assert any(
        child.schema_name == "publication_record_manifest@1.1.0"
        for child in record.children
    )


def test_validate_demo_store_returns_validation_report(
    phase01_demo_output_root: Path,
) -> None:
    report = euclid.validate_demo_store(output_root=phase01_demo_output_root)

    assert report.is_valid
    assert report.manifest_count > 5
    assert report.issue_count == 0


def test_inspect_demo_point_prediction_returns_artifact_backed_summary(
    phase01_demo_output_root: Path,
) -> None:
    inspection = euclid.inspect_demo_point_prediction(
        output_root=phase01_demo_output_root
    )

    assert inspection.run_id == "prototype_constant_candidate_run_result_v1"
    assert inspection.prediction_artifact_ref.schema_name == (
        "prediction_artifact_manifest@1.1.0"
    )
    assert inspection.point_score_result_ref.schema_name == (
        "point_score_result_manifest@1.0.0"
    )
    assert inspection.stage_id == "confirmatory_holdout"
    assert inspection.horizon_set == (1,)
    assert inspection.scored_origin_count == inspection.row_count
    assert inspection.timeguard_failure_count == 0
    assert inspection.aggregated_primary_score >= 0.0
    assert inspection.per_horizon_scores


def test_compare_demo_baseline_returns_artifact_backed_delta_summary(
    phase01_demo_output_root: Path,
) -> None:
    comparison = euclid.compare_demo_baseline(output_root=phase01_demo_output_root)

    assert comparison.run_id == "prototype_constant_candidate_run_result_v1"
    assert comparison.comparison_universe_ref.schema_name == (
        "comparison_universe_manifest@1.0.0"
    )
    assert comparison.baseline_id == "constant_baseline"
    assert comparison.comparison_class_status == "comparable"
    assert comparison.score_delta == (
        comparison.baseline_primary_score - comparison.candidate_primary_score
    )
    assert comparison.candidate_beats_baseline == (
        comparison.candidate_primary_score < comparison.baseline_primary_score
    )


def test_inspect_demo_replay_bundle_returns_recorded_refs_hashes_and_seeds(
    phase01_demo_output_root: Path,
) -> None:
    inspection = euclid.inspect_demo_replay_bundle(output_root=phase01_demo_output_root)

    assert inspection.bundle_ref.schema_name == (
        "reproducibility_bundle_manifest@1.0.0"
    )
    assert inspection.run_result_ref.schema_name == "run_result_manifest@1.1.0"
    assert inspection.replay_verification_status == "verified"
    assert inspection.recorded_stage_order[0] == "dataset_snapshot_frozen"
    assert {record.seed_scope for record in inspection.seed_records} == {
        "search",
        "surrogate_generation",
        "perturbation",
    }
    assert "python_version" in inspection.environment_metadata
    assert any(
        record.artifact_role == "run_result"
        for record in inspection.artifact_hash_records
    )


def test_publish_demo_run_to_catalog_writes_and_loads_read_only_catalog(
    phase01_demo_output_root: Path,
) -> None:
    entry = euclid.publish_demo_run_to_catalog(output_root=phase01_demo_output_root)

    assert entry.request_id == "prototype-demo"
    assert entry.publication_mode == "abstention_only_publication"
    assert entry.publication_record_ref.schema_name == (
        "publication_record_manifest@1.1.0"
    )
    assert entry.reproducibility_bundle_ref.schema_name == (
        "reproducibility_bundle_manifest@1.0.0"
    )
    assert entry.abstention_ref is not None
    assert (phase01_demo_output_root / "catalog" / "index.json").is_file()

    catalog = euclid.load_demo_publication_catalog(output_root=phase01_demo_output_root)

    assert catalog.entry_count == 1
    assert catalog.entries[0].publication_id == entry.publication_id


def test_inspect_demo_catalog_entry_returns_published_artifacts(
    phase01_demo_output_root: Path,
) -> None:
    entry = euclid.publish_demo_run_to_catalog(output_root=phase01_demo_output_root)

    inspection = euclid.inspect_demo_catalog_entry(
        output_root=phase01_demo_output_root,
        publication_id=entry.publication_id,
    )

    assert inspection.entry.publication_id == entry.publication_id
    assert inspection.run_result.manifest.schema_name == "run_result_manifest@1.1.0"
    assert inspection.publication_record.manifest.schema_name == (
        "publication_record_manifest@1.1.0"
    )
    assert inspection.replay_bundle.bundle_ref.schema_name == (
        "reproducibility_bundle_manifest@1.0.0"
    )
    assert inspection.replay_bundle.replay_verification_status == "verified"
    assert inspection.claim_card is None
    assert inspection.abstention is not None


def test_inspect_demo_probabilistic_prediction_returns_artifact_backed_summary(
    tmp_path: Path,
) -> None:
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=PROBABILISTIC_MANIFEST,
        output_root=tmp_path / "probabilistic-output",
    )

    inspection = euclid.inspect_demo_probabilistic_prediction(
        output_root=result.paths.output_root
    )

    assert inspection.request_id == result.request.request_id
    assert inspection.prediction_artifact_ref.schema_name == (
        "prediction_artifact_manifest@1.1.0"
    )
    assert inspection.score_result_ref.schema_name == (
        "probabilistic_score_result_manifest@1.0.0"
    )
    assert inspection.forecast_object_type == "distribution"
    assert inspection.aggregated_primary_score >= 0.0
    assert inspection.rows


def test_inspect_demo_calibration_returns_separate_calibration_summary(
    tmp_path: Path,
) -> None:
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=PROBABILISTIC_MANIFEST,
        output_root=tmp_path / "probabilistic-output",
    )

    calibration = euclid.inspect_demo_calibration(output_root=result.paths.output_root)

    assert calibration.request_id == result.request.request_id
    assert calibration.forecast_object_type == "distribution"
    assert calibration.calibration_result_ref.schema_name == (
        "calibration_result_manifest@1.0.0"
    )
    assert calibration.status in {"passed", "failed"}
    assert calibration.diagnostics
