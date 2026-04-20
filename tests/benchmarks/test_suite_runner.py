from __future__ import annotations

import json
from pathlib import Path

import euclid
import euclid.benchmarks.runtime as benchmark_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MVP_SUITE = PROJECT_ROOT / "benchmarks/suites/mvp.yaml"
CURRENT_RELEASE_SUITE = PROJECT_ROOT / "benchmarks/suites/current-release.yaml"


def test_profile_benchmark_suite_runs_declared_mvp_tasks_and_writes_summary(
    tmp_path: Path,
) -> None:
    result = euclid.profile_benchmark_suite(
        manifest_path=MVP_SUITE,
        benchmark_root=tmp_path / "benchmark-suite",
        resume=False,
    )

    assert result.suite_manifest.suite_id == "mvp"
    assert [task.task_manifest.task_id for task in result.task_results] == [
        "planted_analytic_demo",
        "seasonal_trend_demo",
        "leakage_trap_demo",
    ]
    assert result.summary_path.exists()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["suite_id"] == "mvp"
    assert summary["task_count"] == 3
    assert summary["completed_task_count"] == 3
    assert summary["required_tracks"] == [
        "rediscovery",
        "predictive_generalization",
        "adversarial_honesty",
    ]


def test_current_release_suite_uses_canonical_active_scope_name(
    tmp_path: Path,
) -> None:
    result = euclid.profile_benchmark_suite(
        manifest_path=CURRENT_RELEASE_SUITE,
        benchmark_root=tmp_path / "current-release-suite",
        resume=False,
    )

    assert result.suite_manifest.suite_id == "current_release"
    assert result.summary_path.exists()
    assert {surface.surface_id for surface in result.surface_statuses} == {
        "retained_core_release",
        "algorithmic_backend",
        "shared_plus_local_decomposition",
        "mechanistic_lane",
    }
    assert all(
        surface.benchmark_status == "passed" for surface in result.surface_statuses
    )
    assert all(surface.replay_status == "passed" for surface in result.surface_statuses)


def test_profile_benchmark_suite_uses_explicit_project_root_when_installed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_runtime_file = (
        tmp_path
        / "venv"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "euclid"
        / "benchmarks"
        / "runtime.py"
    )
    fake_runtime_file.parent.mkdir(parents=True, exist_ok=True)
    fake_runtime_file.write_text("# installed runtime placeholder\n", encoding="utf-8")
    monkeypatch.setattr(benchmark_runtime, "__file__", str(fake_runtime_file))

    result = euclid.profile_benchmark_suite(
        manifest_path=MVP_SUITE,
        benchmark_root=tmp_path / "installed-suite",
        project_root=PROJECT_ROOT,
        resume=False,
    )

    assert result.summary_path.exists()
    assert all(
        surface.benchmark_status == "passed" for surface in result.surface_statuses
    )
