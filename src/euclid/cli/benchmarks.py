from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from euclid.benchmarks import (
    load_benchmark_task_manifests,
    profile_benchmark_suite,
    profile_benchmark_task,
)
from euclid.release import write_suite_evidence_bundle
from euclid.operator_runtime.resources import (
    resolve_asset_root,
)
from euclid.operator_runtime.resources import (
    resolve_workspace_root as resolve_runtime_workspace_root,
)

benchmarks_app = typer.Typer(
    add_completion=False,
    help="Run named benchmark suites through the certified benchmark harness.",
)


def _project_root() -> Path:
    return resolve_asset_root()


def _suite_manifest_path(*, suite: str, project_root: Path) -> Path | None:
    direct_path = (project_root / suite).resolve()
    if direct_path.is_file():
        return direct_path
    named_suite = (project_root / "benchmarks" / "suites" / suite).resolve()
    if named_suite.is_file():
        return named_suite
    if not named_suite.suffix:
        yaml_named_suite = named_suite.with_suffix(".yaml")
        if yaml_named_suite.is_file():
            return yaml_named_suite
    return None


def _load_suite_manifests(*, suite: str, project_root: Path) -> tuple:
    task_root = project_root / "benchmarks" / "tasks"
    if suite == "release_smoke":
        return load_benchmark_task_manifests(task_root)
    suite_root = task_root / suite
    if suite_root.is_dir():
        return load_benchmark_task_manifests(suite_root)
    raise typer.BadParameter(
        f"unsupported benchmark suite {suite!r}",
        param_hint="suite",
    )


@benchmarks_app.command("run")
def run_suite(
    suite: str = typer.Option(
        ...,
        help="Named suite to run. Supports release_smoke or a benchmark track id.",
    ),
    project_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        exists=True,
        readable=True,
        help="Repository root used to resolve benchmark manifests.",
    ),
    benchmark_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory where benchmark suite artifacts are written.",
    ),
    parallel_workers: int = typer.Option(
        1,
        min=1,
        help="Deterministic worker count for benchmark suite tasks.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Reuse cached benchmark artifacts when available.",
    ),
) -> None:
    """Run a named benchmark suite and print the saved artifact paths."""
    resolved_project_root = resolve_asset_root(project_root)
    resolved_benchmark_root = (
        benchmark_root.resolve()
        if benchmark_root is not None
        else resolve_runtime_workspace_root(project_root)
        / "build"
        / "benchmark-suites"
        / suite
    )
    suite_manifest_path = _suite_manifest_path(
        suite=suite,
        project_root=resolved_project_root,
    )
    if suite_manifest_path is not None:
        suite_result = profile_benchmark_suite(
            manifest_path=suite_manifest_path,
            benchmark_root=resolved_benchmark_root,
            project_root=resolved_project_root,
            parallel_workers=parallel_workers,
            resume=resume,
        )
        write_suite_evidence_bundle(
            suite_result=suite_result,
            workspace_root=resolve_runtime_workspace_root(project_root),
        )
        lines = [
            "Euclid benchmark suite",
            f"Suite: {suite_result.suite_manifest.suite_id}",
            f"Benchmark root: {resolved_benchmark_root}",
            f"Summary: {suite_result.summary_path}",
        ]
        lines.extend(
            (
                f"- {result.task_manifest.track_id}/{result.task_manifest.task_id}: "
                f"task_result={result.report_paths.task_result_path} "
                f"report={result.report_paths.report_path} "
                f"telemetry={result.telemetry_path}"
            )
            for result in suite_result.task_results
        )
        typer.echo("\n".join(lines))
        return
    manifests = _load_suite_manifests(
        suite=suite,
        project_root=resolved_project_root,
    )
    results = tuple(
        profile_benchmark_task(
            manifest_path=manifest.source_path,
            benchmark_root=resolved_benchmark_root,
            project_root=resolved_project_root,
            parallel_workers=parallel_workers,
            resume=resume,
        )
        for manifest in manifests
    )
    lines = [
        "Euclid benchmark suite",
        f"Suite: {suite}",
        f"Benchmark root: {resolved_benchmark_root}",
    ]
    lines.extend(
        (
            f"- {result.task_manifest.track_id}/{result.task_manifest.task_id}: "
            f"task_result={result.report_paths.task_result_path} "
            f"report={result.report_paths.report_path} "
            f"telemetry={result.telemetry_path}"
        )
        for result in results
    )
    typer.echo("\n".join(lines))


__all__ = ["benchmarks_app", "run_suite"]
