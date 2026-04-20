from __future__ import annotations

from pathlib import Path

import euclid
from euclid.benchmarks import profile_benchmark_task
from euclid.performance import (
    PerformanceBudget,
    SuitePerformanceBudget,
    collect_performance_suite,
    evaluate_performance_budget,
    evaluate_suite_performance_budget,
    load_performance_telemetry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_MANIFEST = PROJECT_ROOT / "fixtures/runtime/prototype-demo.yaml"
BENCHMARK_MANIFEST = (
    PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
)


def test_phase_a_runtime_smoke_stays_within_budget(tmp_path: Path) -> None:
    demo = euclid.profile_demo_run(
        manifest_path=SAMPLE_MANIFEST,
        output_root=tmp_path / "demo-output",
    )
    benchmark = profile_benchmark_task(
        manifest_path=BENCHMARK_MANIFEST,
        benchmark_root=tmp_path / "benchmark-output",
    )

    demo_telemetry = load_performance_telemetry(demo.telemetry_path)
    benchmark_telemetry = load_performance_telemetry(benchmark.telemetry_path)
    suite = collect_performance_suite(
        suite_id="phase_a_runtime_smoke",
        telemetry_paths=(demo.telemetry_path, benchmark.telemetry_path),
    )

    demo_budget = evaluate_performance_budget(
        demo_telemetry,
        PerformanceBudget(
            budget_id="demo_run_smoke",
            max_wall_time_seconds=5.0,
            max_peak_memory_bytes=256 * 1024 * 1024,
        ),
    )
    benchmark_budget = evaluate_performance_budget(
        benchmark_telemetry,
        PerformanceBudget(
            budget_id="benchmark_task_smoke",
            max_wall_time_seconds=10.0,
            max_peak_memory_bytes=256 * 1024 * 1024,
        ),
    )
    suite_budget = evaluate_suite_performance_budget(
        suite,
        SuitePerformanceBudget(
            budget_id="phase_a_runtime_suite",
            max_total_wall_time_seconds=12.0,
            max_profile_wall_time_seconds=10.0,
        ),
    )

    assert demo_budget.passed
    assert benchmark_budget.passed
    assert suite_budget.passed
    assert suite.profile_count == 2
    assert suite.max_wall_time_seconds >= demo_telemetry.wall_time_seconds
    assert suite.total_wall_time_seconds >= suite.max_wall_time_seconds
