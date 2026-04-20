from __future__ import annotations

from pathlib import Path

import euclid
from euclid.performance import (
    SuitePerformanceBudget,
    collect_performance_suite,
    evaluate_suite_performance_budget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_RELEASE_SUITE = PROJECT_ROOT / "benchmarks/suites/current-release.yaml"


def test_current_release_portfolio_runtime_stays_within_budget(tmp_path: Path) -> None:
    suite_result = euclid.profile_benchmark_suite(
        manifest_path=CURRENT_RELEASE_SUITE,
        benchmark_root=tmp_path / "current-release-suite",
        resume=False,
    )

    suite = collect_performance_suite(
        suite_id="phase_f_current_release_suite",
        telemetry_paths=tuple(
            result.telemetry_path for result in suite_result.task_results
        ),
    )
    budget = evaluate_suite_performance_budget(
        suite,
        SuitePerformanceBudget(
            budget_id="phase_f_current_release_budget",
            max_total_wall_time_seconds=30.0,
            max_profile_wall_time_seconds=10.0,
            max_peak_memory_bytes=512 * 1024 * 1024,
        ),
    )

    assert budget.passed
    assert suite.profile_count == len(suite_result.task_results)
    assert suite.max_wall_time_seconds <= 10.0
