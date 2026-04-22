from __future__ import annotations

import json
from pathlib import Path

from euclid.benchmarks import profile_benchmark_task
from euclid.benchmarks import runtime as benchmark_runtime
from euclid.benchmarks.manifests import BenchmarkSuiteSurfaceRequirement
from euclid.control_plane import SQLiteExecutionStateStore

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_profile_benchmark_task_emits_telemetry_and_report_artifacts(
    tmp_path: Path,
) -> None:
    result = profile_benchmark_task(
        manifest_path=(
            PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
        ),
        benchmark_root=tmp_path / "benchmarks",
    )

    assert result.report_paths.task_result_path.is_file()
    assert result.telemetry_path.is_file()

    payload = json.loads(result.telemetry_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "performance_telemetry"
    assert payload["profile_kind"] == "benchmark_task"
    assert payload["subject_id"] == "planted_analytic_demo"
    assert payload["artifact_store"]["write_operation_count"] > 0
    assert payload["artifact_store"]["write_throughput_bytes_per_second"] >= 0.0
    assert any(
        record["submitter_id"] == "analytic_backend"
        and record["declared_candidate_limit"] == 128
        for record in payload["budget_records"]
    )
    assert payload["attributes"]["benchmark_budget_report"] == {
        "budget_id": "benchmark_task_budget:planted_analytic_demo",
        "candidate_limit": 128,
        "wall_clock_seconds": 300,
        "parallel_workers": 1,
        "submitter_count": 4,
        "status": "reported",
    }
    assert any(
        record["submitter_id"] == "analytic_backend"
        and record["declared_restarts"] == 3
        for record in payload["restart_records"]
    )
    assert {span["category"] for span in payload["spans"]} >= {
        "benchmark_intake",
        "search",
        "portfolio_selection",
        "reporting",
    }

    task_result = json.loads(
        result.report_paths.task_result_path.read_text(encoding="utf-8")
    )
    semantic_summary = task_result["semantic_summary"]
    assert semantic_summary["run_support_object_ids"] == [
        "observation_model:gaussian_point",
        "quantization:decimal_1e-6",
        "target_transform:identity",
    ]
    assert semantic_summary["claim_lane_ids"] == ["forecast_object:point"]
    assert semantic_summary["replay_ids"] == ["replay:ledger_only"]
    assert semantic_summary["engine_ids"] == [
        "algorithmic_search_backend",
        "analytic_backend",
        "portfolio_orchestrator",
        "recursive_spectral_backend",
    ]
    assert semantic_summary["score_policy_ids"] == ["score:mean_absolute_error"]
    assert semantic_summary["threshold_ids"] == [
        "practical_significance_margin",
        "predictive_adequacy_floor:mean_absolute_error",
    ]


def test_surface_status_fails_when_task_result_file_lacks_semantic_summary(
    tmp_path: Path,
) -> None:
    result = profile_benchmark_task(
        manifest_path=(
            PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
        ),
        benchmark_root=tmp_path / "benchmarks",
    )
    result.report_paths.task_result_path.write_text(
        json.dumps(
            {
                "artifact_type": "benchmark_task_result",
                "status": "completed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    status = benchmark_runtime._surface_status(
        requirement=BenchmarkSuiteSurfaceRequirement(
            surface_id="retained_core_release",
            task_ids=("planted_analytic_demo",),
            replay_required=True,
        ),
        task_results=(result,),
    )

    assert status.benchmark_status == "failed"
    assert status.evidence["semantic_status"] == "missing"


def test_profile_benchmark_task_resume_reuses_cached_context_and_submitters(
    tmp_path: Path,
) -> None:
    benchmark_root = tmp_path / "benchmarks"

    first = profile_benchmark_task(
        manifest_path=(
            PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
        ),
        benchmark_root=benchmark_root,
        parallel_workers=3,
    )
    second = profile_benchmark_task(
        manifest_path=(
            PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
        ),
        benchmark_root=benchmark_root,
        parallel_workers=3,
    )

    assert [
        result.selected_candidate_id for result in first.submitter_results
    ] == [result.selected_candidate_id for result in second.submitter_results]

    payload = json.loads(second.telemetry_path.read_text(encoding="utf-8"))
    assert any(
        measurement["name"] == "resume_checkpoint_hits"
        and measurement["category"] == "checkpoint_resume"
        and measurement["value"] >= 4
        for measurement in payload["measurements"]
    )
    assert any(
        measurement["name"] == "parallel_worker_count"
        and measurement["category"] == "benchmark_runtime"
        and measurement["value"] == 3
        for measurement in payload["measurements"]
    )

    control_plane_path = (
        benchmark_root
        / "results"
        / "rediscovery"
        / "planted_analytic_demo"
        / "_profile_runtime"
        / "active-runs"
        / "planted_analytic_demo"
        / "control-plane"
        / "execution-state.sqlite3"
    )
    snapshot = SQLiteExecutionStateStore(control_plane_path).load_run_snapshot(
        "planted_analytic_demo"
    )
    assert {
        state.step_id: state.status for state in snapshot.step_states
    } == {
        "benchmark.runtime.context": "completed",
        "benchmark.submitter.analytic_backend": "completed",
        "benchmark.submitter.recursive_spectral_backend": "completed",
        "benchmark.submitter.algorithmic_search_backend": "completed",
        "benchmark.submitter.portfolio_orchestrator": "completed",
        "benchmark.runtime.reporting": "completed",
    }
