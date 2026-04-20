from __future__ import annotations

from pathlib import Path

from euclid.fmp_smoke import (
    build_benchmark_task_manifest,
    build_csv_rows_from_fmp_history,
    build_operator_manifest,
    extract_fmp_history_entries,
    format_smoke_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_extract_fmp_history_entries_accepts_list_and_mapping_shapes() -> None:
    history = [{"date": "2024-01-02", "close": 101.5}]

    assert extract_fmp_history_entries(history) == history
    assert extract_fmp_history_entries({"historical": history}) == history
    assert extract_fmp_history_entries({"data": history}) == history


def test_build_csv_rows_from_fmp_history_sorts_and_normalizes() -> None:
    rows = build_csv_rows_from_fmp_history(
        [
            {
                "date": "2024-01-03",
                "close": 102.0,
                "open": 100.0,
                "high": 103.0,
                "low": 99.5,
                "volume": 1500,
                "changePercent": 1.4,
                "adjClose": 101.8,
            },
            {
                "date": "2024-01-02",
                "close": 100.0,
                "open": 98.0,
                "high": 101.0,
                "low": 97.5,
                "volume": 1200,
                "changePercent": 0.8,
                "adjClose": 99.7,
            },
        ],
        symbol="SPY",
    )

    assert [row["event_time"] for row in rows] == [
        "2024-01-02T00:00:00Z",
        "2024-01-03T00:00:00Z",
    ]
    assert rows[0]["source_id"] == "financial_modeling_prep"
    assert rows[0]["series_id"] == "SPY"
    assert rows[0]["available_at"] == "2024-01-02T21:00:00Z"
    assert rows[0]["observed_value"] == 100.0
    assert rows[0]["revision_id"] == 0
    assert rows[0]["adj_close"] == 99.7
    assert rows[0]["change_percent"] == 0.8
    assert rows[0]["volume"] == 1200


def test_build_operator_manifest_supports_point_and_probabilistic_modes(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "datasets" / "spy.csv"

    point_manifest = build_operator_manifest(
        project_root=PROJECT_ROOT,
        dataset_csv=dataset_csv,
        manifest_path=tmp_path / "manifests" / "point.yaml",
        request_id="spy-current-release",
        forecast_object_type="point",
        row_count=90,
    )
    distribution_manifest = build_operator_manifest(
        project_root=PROJECT_ROOT,
        dataset_csv=dataset_csv,
        manifest_path=tmp_path / "manifests" / "distribution.yaml",
        request_id="spy-distribution",
        forecast_object_type="distribution",
        row_count=90,
    )

    assert point_manifest["dataset_csv"] == "../datasets/spy.csv"
    assert point_manifest["forecast_object_type"] == "point"
    assert point_manifest["search"]["class"] == "bounded_heuristic"
    assert point_manifest["search"]["family_ids"] == [
        "constant",
        "drift",
        "linear_trend",
        "seasonal_naive",
    ]

    assert distribution_manifest["dataset_csv"] == "../datasets/spy.csv"
    assert distribution_manifest["horizon"] == 21
    assert distribution_manifest["search"]["class"] == "exact_finite_enumeration"
    assert distribution_manifest["probabilistic"]["forecast_object_type"] == (
        "distribution"
    )
    assert distribution_manifest["probabilistic"]["primary_score"] == (
        "continuous_ranked_probability_score"
    )


def test_build_benchmark_task_manifest_uses_repo_relative_dataset_and_existing_baselines(
    tmp_path: Path,
) -> None:
    workspace_root = PROJECT_ROOT / "build" / "tests" / "fmp-smoke"
    dataset_csv = workspace_root / "spy.csv"
    task_manifest = build_benchmark_task_manifest(
        project_root=PROJECT_ROOT,
        dataset_csv=dataset_csv,
        manifest_path=tmp_path / "benchmark.yaml",
        symbol="SPY",
        row_count=120,
    )

    assert task_manifest["track_id"] == "predictive_generalization"
    assert task_manifest["dataset_ref"] == "build/tests/fmp-smoke/spy.csv"
    assert task_manifest["task_id"].startswith("fmp_spy_")
    assert task_manifest["baseline_registry"][0]["manifest_path"] == (
        "benchmarks/baselines/predictive_generalization/naive-last-value.yaml"
    )
    assert "fixture_spec_id" not in task_manifest
    assert "fixture_family_id" not in task_manifest


def test_format_smoke_report_includes_modes_and_artifact_locations() -> None:
    report = format_smoke_report(
        {
            "symbol": "SPY",
            "workspace_root": "/tmp/fmp-smoke",
            "dataset_csv": "/tmp/fmp-smoke/spy.csv",
            "rows": 120,
            "date_range": {
                "start": "2024-01-02T00:00:00Z",
                "end": "2024-06-28T00:00:00Z",
            },
            "results": {
                "current_release": {
                    "run_id": "spy-current-release",
                    "selected_family": "linear_trend",
                    "replay_verification": "verified",
                },
                "benchmark": {
                    "task_id": "fmp_spy_benchmark",
                    "report_path": "/tmp/fmp-smoke/report.md",
                },
            },
        }
    )

    assert "Euclid FMP smoke" in report
    assert "Symbol: SPY" in report
    assert "Mode: current_release" in report
    assert "Replay verification: verified" in report
    assert "Mode: benchmark" in report
    assert "Report path: /tmp/fmp-smoke/report.md" in report
