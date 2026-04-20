from __future__ import annotations

import json
from pathlib import Path

import euclid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_MANIFEST = PROJECT_ROOT / "fixtures/runtime/prototype-demo.yaml"
PROBABILISTIC_MANIFEST = (
    PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-distribution-demo.yaml"
)
ALGORITHMIC_MANIFEST = (
    PROJECT_ROOT / "fixtures/runtime/phase06/algorithmic-search-demo.yaml"
)


def test_run_demo_returns_typed_result_and_writes_local_bundle(tmp_path: Path) -> None:
    result = euclid.run_demo(
        manifest_path=SAMPLE_MANIFEST,
        output_root=tmp_path / "demo-output",
    )

    assert result.request.request_id == "prototype-demo"
    assert result.summary.selected_family == "constant"
    assert result.summary.result_mode == "abstention_only_publication"
    assert result.summary.bundle_ref.schema_name == (
        "reproducibility_bundle_manifest@1.0.0"
    )
    assert result.paths.artifact_root.is_dir()
    assert result.paths.metadata_store_path.is_file()
    assert result.paths.run_summary_path.is_file()


def test_replay_demo_reuses_written_summary_when_bundle_ref_is_omitted(
    tmp_path: Path,
) -> None:
    run_result = euclid.run_demo(
        manifest_path=SAMPLE_MANIFEST,
        output_root=tmp_path / "demo-output",
    )

    replay_result = euclid.replay_demo(output_root=run_result.paths.output_root)

    assert replay_result.summary.bundle_ref == run_result.summary.bundle_ref
    assert replay_result.summary.run_result_ref == run_result.summary.run_result_ref
    assert replay_result.summary.selected_candidate_ref == (
        run_result.summary.selected_candidate_ref
    )
    assert replay_result.summary.replay_verification_status == "verified"


def test_run_demo_point_evaluation_returns_artifact_backed_surfaces(
    tmp_path: Path,
) -> None:
    result = euclid.run_demo_point_evaluation(
        manifest_path=SAMPLE_MANIFEST,
        output_root=tmp_path / "demo-output",
    )

    assert result.run.summary.result_mode == "abstention_only_publication"
    assert result.prediction.run_id == result.run.summary.run_result_ref.object_id
    assert result.prediction.point_score_result_ref.schema_name == (
        "point_score_result_manifest@1.0.0"
    )
    assert result.comparison.comparison_universe_ref.schema_name == (
        "comparison_universe_manifest@1.0.0"
    )
    assert result.comparison.candidate_primary_score == (
        result.prediction.aggregated_primary_score
    )


def test_run_demo_probabilistic_evaluation_returns_prediction_and_calibration(
    tmp_path: Path,
) -> None:
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=PROBABILISTIC_MANIFEST,
        output_root=tmp_path / "probabilistic-output",
    )

    assert result.request.request_id == "phase06-probabilistic-distribution-demo"
    assert result.prediction.forecast_object_type == "distribution"
    assert result.prediction.score_result_ref.schema_name == (
        "probabilistic_score_result_manifest@1.0.0"
    )
    assert result.calibration.calibration_result_ref.schema_name == (
        "calibration_result_manifest@1.0.0"
    )
    assert result.calibration.forecast_object_type == "distribution"
    assert result.calibration.status in {"passed", "failed"}
    assert result.prediction.row_count > 0


def test_run_demo_algorithmic_search_returns_algorithmic_frontier_summary(
    tmp_path: Path,
) -> None:
    result = euclid.run_demo_algorithmic_search(
        manifest_path=ALGORITHMIC_MANIFEST,
        output_root=tmp_path / "algorithmic-output",
    )

    assert result.request.request_id == "phase06-algorithmic-search-demo"
    assert result.summary.selected_family == "algorithmic"
    assert result.summary.selected_candidate_id.startswith("algorithmic_")
    assert result.summary.accepted_candidate_ids
    assert all(
        candidate_id.startswith("algorithmic_")
        for candidate_id in result.summary.accepted_candidate_ids
    )
    assert result.summary.coverage_statement


def test_profile_demo_run_returns_telemetry_artifact(tmp_path: Path) -> None:
    result = euclid.profile_demo_run(
        manifest_path=SAMPLE_MANIFEST,
        output_root=tmp_path / "demo-output",
    )

    assert result.run.summary.selected_family == "constant"
    assert result.telemetry_path.is_file()

    payload = json.loads(result.telemetry_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "performance_telemetry"
    assert payload["profile_kind"] == "demo_run"
    assert payload["subject_id"] == "prototype-demo"
    assert payload["artifact_store"]["write_operation_count"] > 0
    assert payload["artifact_store"]["cache_hit_count"] >= 0
    assert payload["seed_records"] == [{"scope": "search", "value": "0"}]
    assert {
        span["category"] for span in payload["spans"]
    } >= {
        "search",
        "fitting",
        "evaluation",
        "portfolio_selection",
        "replay",
        "publication",
    }
