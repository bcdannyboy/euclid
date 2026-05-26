from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import yaml

from euclid.artifacts import FilesystemArtifactStore
from euclid.benchmarks.manifests import (
    BenchmarkSuiteManifest,
    BenchmarkSuiteSurfaceRequirement,
    BenchmarkTaskManifest,
    ensure_benchmark_repository_tree,
    load_benchmark_suite_manifest,
    load_benchmark_task_manifest,
)
from euclid.benchmarks.reporting import (
    BenchmarkTaskReportArtifactPaths,
    build_benchmark_suite_task_semantics,
    build_benchmark_surface_diagnostics,
    build_benchmark_task_semantic_summary,
    build_benchmark_task_track_summary,
    write_benchmark_task_report_artifacts,
)
from euclid.benchmarks.submitters import (
    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
    BenchmarkHarnessContext,
    BenchmarkSubmitterResult,
    run_benchmark_submitter,
)
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.control_plane import (
    FileLock,
    RuntimeWorkspace,
    RunWorkspacePaths,
    SQLiteExecutionStateStore,
    SQLiteMetadataStore,
)
from euclid.ingestion import ingest_csv_dataset
from euclid.manifests.base import ManifestEnvelope
from euclid.manifest_registry import ManifestRegistry
from euclid.modules.calibration import (
    build_calibration_contract,
    evaluate_prediction_calibration,
)
from euclid.modules.candidate_fitting import (
    build_candidate_fit_artifacts,
    fit_candidate_window,
)
from euclid.modules.forecast_paths import forecast_path as build_forecast_path
from euclid.modules.probabilistic_evaluation import (
    emit_probabilistic_prediction_artifact,
)
from euclid.modules.replay import verify_portfolio_replay_contract
from euclid.modules.scoring import (
    bind_distribution_row_observation_model,
    score_prediction_artifact,
)
from euclid.modules.split_planning import resolve_scored_origin_target_row
from euclid.operator_runtime.intake_planning import build_operator_intake_plan
from euclid.performance import (
    PerformanceTelemetryArtifact,
    TelemetryRecorder,
    benchmark_budget_report,
    write_performance_telemetry,
)
from euclid.runtime.hashing import sha256_digest
from euclid.search.backends import DescriptiveSearchProposal
from euclid.stochastic.event_definitions import EventDefinition

_BENCHMARK_RUNTIME_CACHE_VERSION = "1.0.1"


@dataclass(frozen=True)
class ProfiledBenchmarkTaskResult:
    task_manifest: BenchmarkTaskManifest
    submitter_results: tuple[BenchmarkSubmitterResult, ...]
    report_paths: BenchmarkTaskReportArtifactPaths
    telemetry: PerformanceTelemetryArtifact
    telemetry_path: Path


@dataclass(frozen=True)
class BenchmarkSuiteSurfaceStatus:
    surface_id: str
    task_ids: tuple[str, ...]
    benchmark_status: str
    replay_status: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class ProfiledBenchmarkSuiteResult:
    suite_manifest: BenchmarkSuiteManifest
    task_results: tuple[ProfiledBenchmarkTaskResult, ...]
    surface_statuses: tuple[BenchmarkSuiteSurfaceStatus, ...]
    summary_path: Path


@dataclass(frozen=True)
class _BenchmarkRuntimeState:
    workspace_paths: RunWorkspacePaths
    context_cache_path: Path
    submitter_cache_root: Path


def profile_benchmark_task(
    *,
    manifest_path: Path | str,
    benchmark_root: Path | str,
    project_root: Path | str | None = None,
    parallel_workers: int = 1,
    resume: bool = True,
) -> ProfiledBenchmarkTaskResult:
    task_manifest = load_benchmark_task_manifest(manifest_path)
    resolved_project_root = _resolve_project_root(project_root)
    benchmark_tree = ensure_benchmark_repository_tree(benchmark_root)
    runtime_state = _benchmark_runtime_state(
        task_manifest=task_manifest,
        benchmark_tree_root=benchmark_tree.root,
    )
    telemetry = TelemetryRecorder()
    telemetry.record_measurement(
        name="parallel_worker_count",
        category="benchmark_runtime",
        value=max(1, int(parallel_workers)),
        unit="workers",
        attributes={"task_id": task_manifest.task_id},
    )
    execution_state = SQLiteExecutionStateStore(
        runtime_state.workspace_paths.control_plane_store_path
    )

    with FileLock(runtime_state.workspace_paths.run_lock_path):
        context, context_checkpoint_hits = _load_or_build_benchmark_harness_context(
            task_manifest=task_manifest,
            benchmark_tree_root=benchmark_tree.root,
            project_root=resolved_project_root,
            runtime_state=runtime_state,
            execution_state=execution_state,
            telemetry=telemetry,
            resume=resume,
        )
        submitter_results, submitter_checkpoint_hits = _load_or_run_submitters(
            context=context,
            runtime_state=runtime_state,
            execution_state=execution_state,
            telemetry=telemetry,
            parallel_workers=parallel_workers,
            resume=resume,
        )
        if task_manifest.metric_thresholds:
            with telemetry.span(
                "benchmark.threshold_metrics",
                category="metric_thresholds",
                attributes={
                    "task_id": task_manifest.task_id,
                    "track_id": task_manifest.track_id,
                    "threshold_count": len(task_manifest.metric_thresholds),
                },
            ):
                with telemetry.allocation_tracing_paused():
                    submitter_results = _submitter_results_with_threshold_metrics(
                        context=context,
                        task_manifest=task_manifest,
                        submitter_results=submitter_results,
                        project_root=resolved_project_root,
                    )
        else:
            submitter_results = _submitter_results_with_threshold_metrics(
                context=context,
                task_manifest=task_manifest,
                submitter_results=submitter_results,
                project_root=resolved_project_root,
            )
        telemetry.record_measurement(
            name="resume_checkpoint_hits",
            category="checkpoint_resume",
            value=context_checkpoint_hits + submitter_checkpoint_hits,
            unit="checkpoints",
            attributes={"task_id": task_manifest.task_id},
        )
        with telemetry.span(
            "benchmark.reporting",
            category="reporting",
            attributes={"task_id": task_manifest.task_id},
        ):
            execution_state.save_step_state(
                run_id=task_manifest.task_id,
                step_id="benchmark.runtime.reporting",
                module_id="benchmark_runtime",
                status="running",
                cursor="reporting",
                details={"report_root": str(benchmark_tree.reports_dir)},
            )
            report_paths = write_benchmark_task_report_artifacts(
                benchmark_root=benchmark_tree.root,
                task_manifest=task_manifest,
                submitter_results=submitter_results,
                task_status="completed",
                track_summary=_task_semantic_summary(task_manifest),
            )
            execution_state.save_step_state(
                run_id=task_manifest.task_id,
                step_id="benchmark.runtime.reporting",
                module_id="benchmark_runtime",
                status="completed",
                cursor="reports_written",
                details={"report_path": str(report_paths.report_path)},
            )

    telemetry_artifact = telemetry.build_artifact(
        profile_kind="benchmark_task",
        subject_id=task_manifest.task_id,
        attributes={
            "track_id": task_manifest.track_id,
            "manifest_path": str(task_manifest.source_path),
            "benchmark_root": str(benchmark_tree.root),
            "resume_enabled": resume,
            "benchmark_budget_report": benchmark_budget_report(
                task_id=task_manifest.task_id,
                candidate_limit=int(
                    task_manifest.frozen_protocol.budget_policy.get(
                        "candidate_limit",
                        0,
                    )
                ),
                wall_clock_seconds=int(
                    task_manifest.frozen_protocol.budget_policy.get(
                        "wall_clock_seconds",
                        0,
                    )
                ),
                parallel_workers=max(1, int(parallel_workers)),
                submitter_count=len(task_manifest.submitter_ids),
            ),
        },
    )
    telemetry_path = write_performance_telemetry(
        benchmark_tree.results_dir
        / task_manifest.track_id
        / task_manifest.task_id
        / "performance-telemetry.json",
        telemetry_artifact,
    )
    return ProfiledBenchmarkTaskResult(
        task_manifest=task_manifest,
        submitter_results=submitter_results,
        report_paths=report_paths,
        telemetry=telemetry_artifact,
        telemetry_path=telemetry_path,
    )


def profile_benchmark_suite(
    *,
    manifest_path: Path | str,
    benchmark_root: Path | str,
    project_root: Path | str | None = None,
    parallel_workers: int = 1,
    resume: bool = True,
) -> ProfiledBenchmarkSuiteResult:
    suite_manifest = load_benchmark_suite_manifest(manifest_path)
    task_results = tuple(
        profile_benchmark_task(
            manifest_path=task_manifest_path,
            benchmark_root=benchmark_root,
            project_root=project_root,
            parallel_workers=parallel_workers,
            resume=resume,
        )
        for task_manifest_path in suite_manifest.task_manifest_paths
    )
    surface_statuses = tuple(
        _surface_status(
            requirement=requirement,
            task_results=task_results,
        )
        for requirement in suite_manifest.surface_requirements
    )
    summary_path = _write_suite_summary(
        suite_manifest=suite_manifest,
        benchmark_root=Path(benchmark_root),
        task_results=task_results,
        surface_statuses=surface_statuses,
    )
    return ProfiledBenchmarkSuiteResult(
        suite_manifest=suite_manifest,
        task_results=task_results,
        surface_statuses=surface_statuses,
        summary_path=summary_path,
    )


def _benchmark_runtime_state(
    *,
    task_manifest: BenchmarkTaskManifest,
    benchmark_tree_root: Path,
) -> _BenchmarkRuntimeState:
    runtime_root = (
        benchmark_tree_root
        / "results"
        / task_manifest.track_id
        / task_manifest.task_id
        / "_profile_runtime"
    )
    workspace = RuntimeWorkspace(runtime_root)
    workspace_paths = workspace.paths_for_run(task_manifest.task_id)
    workspace.materialize(workspace_paths)
    return _BenchmarkRuntimeState(
        workspace_paths=workspace_paths,
        context_cache_path=workspace_paths.cache_root / "benchmark-harness-context.pkl",
        submitter_cache_root=workspace_paths.cache_root / "submitter-results",
    )


def _surface_status(
    *,
    requirement: BenchmarkSuiteSurfaceRequirement,
    task_results: tuple[ProfiledBenchmarkTaskResult, ...],
) -> BenchmarkSuiteSurfaceStatus:
    matched = tuple(
        result
        for result in task_results
        if result.task_manifest.task_id in requirement.task_ids
    )
    task_semantic_status = {
        result.task_manifest.task_id: _task_semantic_status(result)
        for result in matched
    }
    task_semantic_assertion_status = {
        result.task_manifest.task_id: _task_semantic_assertion_status(result)
        for result in matched
    }
    task_replay_verification_status = {
        result.task_manifest.task_id: _task_replay_verification_status(result)
        for result in matched
    }
    task_replay_reason_codes = {
        result.task_manifest.task_id: _task_replay_reason_codes(result)
        for result in matched
    }
    semantic_reason_codes = _semantic_reason_codes(
        requirement=requirement,
        matched=matched,
        task_semantic_status=task_semantic_status,
        task_semantic_assertion_status=task_semantic_assertion_status,
    )
    replay_artifact_paths = [
        str(path)
        for result in matched
        for path in result.report_paths.replay_ref_paths.values()
    ]
    missing_replay_artifacts = [
        str(path)
        for result in matched
        for path in result.report_paths.replay_ref_paths.values()
        if not path.is_file()
    ]
    unverified_replay_artifacts = [
        str(path)
        for result in matched
        for path in result.report_paths.replay_ref_paths.values()
        if path.is_file()
        and _task_replay_verification_status(result) not in {"verified", "missing"}
    ]
    benchmark_passed = (
        len(matched) == len(requirement.task_ids)
        and all(status == "verified" for status in task_semantic_status.values())
        and all(
            status == "passed" for status in task_semantic_assertion_status.values()
        )
    )
    if not requirement.replay_required:
        replay_passed = True
    elif requirement.surface_id == "portfolio_orchestration":
        replay_passed = (
            len(matched) == len(requirement.task_ids)
            and all(
                status == "verified"
                for status in task_replay_verification_status.values()
            )
            and all(
                result.report_paths.portfolio_selection_record_path is not None
                and result.report_paths.portfolio_selection_record_path.is_file()
                for result in matched
            )
        )
    else:
        replay_passed = len(matched) == len(requirement.task_ids) and all(
            status == "verified" for status in task_replay_verification_status.values()
        )
    replay_verification_status = (
        "not_required"
        if not requirement.replay_required
        else _aggregate_replay_verification_status(
            task_replay_verification_status.values()
        )
    )
    return BenchmarkSuiteSurfaceStatus(
        surface_id=requirement.surface_id,
        task_ids=requirement.task_ids,
        benchmark_status="passed" if benchmark_passed else "failed",
        replay_status="passed" if replay_passed else "failed",
        evidence={
            "task_ids": list(requirement.task_ids),
            "matched_task_ids": [result.task_manifest.task_id for result in matched],
            "task_result_paths": [
                str(result.report_paths.task_result_path) for result in matched
            ],
            "semantic_status": (
                "verified"
                if task_semantic_status
                and all(status == "verified" for status in task_semantic_status.values())
                else "missing"
            ),
            "semantic_assertion_status": (
                "passed"
                if task_semantic_assertion_status
                and all(
                    status == "passed"
                    for status in task_semantic_assertion_status.values()
                )
                else (
                    "missing"
                    if any(
                        status == "missing"
                        for status in task_semantic_assertion_status.values()
                    )
                    else "failed"
                )
            ),
            "task_semantic_status": task_semantic_status,
            "task_semantic_assertion_status": task_semantic_assertion_status,
            "task_replay_verification_status": task_replay_verification_status,
            "replay_verification_status": replay_verification_status,
            "replay_artifact_paths": replay_artifact_paths,
            "missing_replay_artifacts": sorted(set(missing_replay_artifacts)),
            "unverified_replay_artifacts": sorted(set(unverified_replay_artifacts)),
            "replay_reason_codes": task_replay_reason_codes,
            "reason_codes": (
                *semantic_reason_codes,
                *_replay_reason_codes(task_replay_verification_status.values()),
            ),
            **build_benchmark_surface_diagnostics(
                tuple(result.task_manifest for result in matched),
                replay_verification_status=replay_verification_status,
            ),
        },
    )


def _write_suite_summary(
    *,
    suite_manifest: BenchmarkSuiteManifest,
    benchmark_root: Path,
    task_results: tuple[ProfiledBenchmarkTaskResult, ...],
    surface_statuses: tuple[BenchmarkSuiteSurfaceStatus, ...],
) -> Path:
    summary_path = (
        benchmark_root
        / "results"
        / "suites"
        / suite_manifest.suite_id
        / "benchmark-suite-summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    completed_task_count = sum(
        1 for result in task_results if result.report_paths.task_result_path.is_file()
    )
    semantic_summary = _suite_semantic_summary(task_results)
    payload = {
        "suite_id": suite_manifest.suite_id,
        "description": suite_manifest.description,
        "authority_snapshot_id": suite_manifest.authority_snapshot_id,
        "fixture_spec_id": suite_manifest.fixture_spec_id,
        "task_count": len(task_results),
        "completed_task_count": completed_task_count,
        "required_tracks": list(suite_manifest.required_tracks),
        "task_results": [
            {
                "track_id": result.task_manifest.track_id,
                "task_id": result.task_manifest.task_id,
                "task_result_path": str(result.report_paths.task_result_path),
                "report_path": str(result.report_paths.report_path),
                "telemetry_path": str(result.telemetry_path),
                "search_class": _search_class(result.task_manifest),
                "composition_operators": list(
                    result.task_manifest.composition_operators
                ),
                **build_benchmark_suite_task_semantics(
                    result.task_manifest,
                    replay_verification_status=_task_replay_verification_status(result),
                ),
                "replay_artifact_paths": [
                    str(path) for path in result.report_paths.replay_ref_paths.values()
                ],
            }
            for result in task_results
        ],
        "surface_statuses": [
            {
                "surface_id": status.surface_id,
                "task_ids": list(status.task_ids),
                "benchmark_status": status.benchmark_status,
                "replay_status": status.replay_status,
                "evidence": dict(status.evidence),
            }
            for status in surface_statuses
        ],
        "search_class_coverage": _search_class_coverage(task_results),
        "composition_operator_coverage": _composition_operator_coverage(task_results),
        "semantic_summary": semantic_summary,
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _load_or_build_benchmark_harness_context(
    *,
    task_manifest: BenchmarkTaskManifest,
    benchmark_tree_root: Path,
    project_root: Path,
    runtime_state: _BenchmarkRuntimeState,
    execution_state: SQLiteExecutionStateStore,
    telemetry: TelemetryRecorder,
    resume: bool,
) -> tuple[BenchmarkHarnessContext, int]:
    execution_state.save_step_state(
        run_id=task_manifest.task_id,
        step_id="benchmark.runtime.context",
        module_id="benchmark_runtime",
        status="running",
        cursor="benchmark_intake",
        details={"resume_enabled": resume},
    )
    dataset_path = _dataset_path(task_manifest, project_root=project_root)
    cache_signature = _cache_signature(
        task_manifest=task_manifest,
        dataset_path=dataset_path,
        suffix="context",
        project_root=project_root,
    )
    with telemetry.span(
        "benchmark.intake",
        category="benchmark_intake",
        attributes={
            "task_id": task_manifest.task_id,
            "track_id": task_manifest.track_id,
            "resume_enabled": resume,
        },
    ):
        cached_context = (
            _load_pickle_cache(
                runtime_state.context_cache_path,
                expected_signature=cache_signature,
            )
            if resume
            else None
        )
        if isinstance(cached_context, BenchmarkHarnessContext):
            execution_state.save_step_state(
                run_id=task_manifest.task_id,
                step_id="benchmark.runtime.context",
                module_id="benchmark_runtime",
                status="completed",
                cursor="context_loaded_from_cache",
                details={"cache_hit": True},
            )
            return cached_context, 1

        context = _build_benchmark_harness_context(
            task_manifest=task_manifest,
            benchmark_tree_root=benchmark_tree_root,
            project_root=project_root,
            runtime_state=runtime_state,
            telemetry=telemetry,
        )
        _write_pickle_cache(
            runtime_state.context_cache_path,
            signature=cache_signature,
            payload=context,
        )
    execution_state.save_step_state(
        run_id=task_manifest.task_id,
        step_id="benchmark.runtime.context",
        module_id="benchmark_runtime",
        status="completed",
        cursor="context_built",
        details={"cache_hit": False},
    )
    return context, 0


def _load_or_run_submitters(
    *,
    context: BenchmarkHarnessContext,
    runtime_state: _BenchmarkRuntimeState,
    execution_state: SQLiteExecutionStateStore,
    telemetry: TelemetryRecorder,
    parallel_workers: int,
    resume: bool,
) -> tuple[tuple[BenchmarkSubmitterResult, ...], int]:
    submitter_ids = context.task_manifest.submitter_ids
    checkpoint_hits = 0
    runtime_state.submitter_cache_root.mkdir(parents=True, exist_ok=True)

    single_submitter_ids = tuple(
        submitter_id
        for submitter_id in submitter_ids
        if submitter_id != PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
    )
    cached_single_results: dict[str, BenchmarkSubmitterResult] = {}
    uncached_single_submitter_ids: list[str] = []
    for submitter_id in single_submitter_ids:
        step_id = f"benchmark.submitter.{submitter_id}"
        cache_path = runtime_state.submitter_cache_root / f"{submitter_id}.pkl"
        cached_result = (
            _load_pickle_cache(
                cache_path,
                expected_signature=_submitter_cache_signature(
                    context=context,
                    submitter_id=submitter_id,
                ),
            )
            if resume
            else None
        )
        if isinstance(cached_result, BenchmarkSubmitterResult):
            cached_single_results[submitter_id] = cached_result
            checkpoint_hits += 1
            execution_state.save_step_state(
                run_id=context.task_manifest.task_id,
                step_id=step_id,
                module_id="benchmark_submitter",
                status="completed",
                cursor="loaded_from_cache",
                details={"cache_hit": True, "submitter_id": submitter_id},
            )
            continue
        execution_state.save_step_state(
            run_id=context.task_manifest.task_id,
            step_id=step_id,
            module_id="benchmark_submitter",
            status="running",
            cursor="executing",
            details={"cache_hit": False, "submitter_id": submitter_id},
        )
        uncached_single_submitter_ids.append(submitter_id)

    single_results = dict(cached_single_results)
    if uncached_single_submitter_ids:
        single_results.update(
            _run_single_submitter_jobs(
                context=context,
                submitter_ids=tuple(uncached_single_submitter_ids),
                telemetry=telemetry,
                parallel_workers=parallel_workers,
            )
        )
        for submitter_id in uncached_single_submitter_ids:
            execution_state.save_step_state(
                run_id=context.task_manifest.task_id,
                step_id=f"benchmark.submitter.{submitter_id}",
                module_id="benchmark_submitter",
                status="completed",
                cursor="submitter_completed",
                details={"cache_hit": False, "submitter_id": submitter_id},
            )
            _write_pickle_cache(
                runtime_state.submitter_cache_root / f"{submitter_id}.pkl",
                signature=_submitter_cache_signature(
                    context=context,
                    submitter_id=submitter_id,
                ),
                payload=single_results[submitter_id],
            )

    portfolio_result: BenchmarkSubmitterResult | None = None
    if PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID in submitter_ids:
        portfolio_cache_path = (
            runtime_state.submitter_cache_root
            / f"{PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID}.pkl"
        )
        cached_portfolio_result = (
            _load_pickle_cache(
                portfolio_cache_path,
                expected_signature=_submitter_cache_signature(
                    context=context,
                    submitter_id=PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
                ),
            )
            if resume
            else None
        )
        if isinstance(cached_portfolio_result, BenchmarkSubmitterResult):
            checkpoint_hits += 1
            execution_state.save_step_state(
                run_id=context.task_manifest.task_id,
                step_id=f"benchmark.submitter.{PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID}",
                module_id="benchmark_submitter",
                status="completed",
                cursor="loaded_from_cache",
                details={
                    "cache_hit": True,
                    "submitter_id": PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
                },
            )
            portfolio_result = cached_portfolio_result
        else:
            execution_state.save_step_state(
                run_id=context.task_manifest.task_id,
                step_id=f"benchmark.submitter.{PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID}",
                module_id="benchmark_submitter",
                status="running",
                cursor="portfolio_selection",
                details={
                    "cache_hit": False,
                    "submitter_id": PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
                },
            )
            portfolio_result = run_benchmark_submitter(
                context=context,
                submitter_id=PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
                telemetry=telemetry,
                child_results=tuple(
                    single_results[submitter_id]
                    for submitter_id in single_submitter_ids
                    if submitter_id in single_results
                ),
            )
            execution_state.save_step_state(
                run_id=context.task_manifest.task_id,
                step_id=f"benchmark.submitter.{PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID}",
                module_id="benchmark_submitter",
                status="completed",
                cursor="portfolio_completed",
                details={
                    "cache_hit": False,
                    "submitter_id": PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
                },
            )
            _write_pickle_cache(
                portfolio_cache_path,
                signature=_submitter_cache_signature(
                    context=context,
                    submitter_id=PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
                ),
                payload=portfolio_result,
            )

    ordered_results: list[BenchmarkSubmitterResult] = []
    for submitter_id in submitter_ids:
        if submitter_id == PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID:
            if portfolio_result is not None:
                ordered_results.append(portfolio_result)
            continue
        ordered_results.append(single_results[submitter_id])
    return tuple(ordered_results), checkpoint_hits


def _run_single_submitter_jobs(
    *,
    context: BenchmarkHarnessContext,
    submitter_ids: tuple[str, ...],
    telemetry: TelemetryRecorder,
    parallel_workers: int,
) -> dict[str, BenchmarkSubmitterResult]:
    resolved_parallel_workers = max(1, int(parallel_workers))
    if resolved_parallel_workers == 1 or len(submitter_ids) <= 1:
        return {
            submitter_id: run_benchmark_submitter(
                context=context,
                submitter_id=submitter_id,
                telemetry=telemetry,
            )
            for submitter_id in submitter_ids
        }

    from concurrent.futures import Future, ThreadPoolExecutor

    with ThreadPoolExecutor(
        max_workers=min(resolved_parallel_workers, len(submitter_ids))
    ) as executor:
        future_by_submitter_id: dict[str, Future[BenchmarkSubmitterResult]] = {
            submitter_id: executor.submit(
                run_benchmark_submitter,
                context=context,
                submitter_id=submitter_id,
                telemetry=telemetry,
            )
            for submitter_id in submitter_ids
        }
        return {
            submitter_id: future_by_submitter_id[submitter_id].result()
            for submitter_id in submitter_ids
        }


def _build_benchmark_harness_context(
    *,
    task_manifest: BenchmarkTaskManifest,
    benchmark_tree_root: Path,
    project_root: Path,
    runtime_state: _BenchmarkRuntimeState,
    telemetry: TelemetryRecorder,
) -> BenchmarkHarnessContext:
    catalog = load_contract_catalog(project_root)
    registry = ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(
            runtime_state.workspace_paths.artifact_root,
            telemetry=telemetry,
        ),
        metadata_store=SQLiteMetadataStore(
            runtime_state.workspace_paths.metadata_store_path
        ),
    )
    dataset_path = _dataset_path(task_manifest, project_root=project_root)
    admitted_data = ingest_csv_dataset(dataset_path)
    row_count = len(admitted_data.coded_observations)
    effective_feature_row_count = max(1, row_count - 2)
    horizon = _resolve_horizon(task_manifest)
    min_train_size = int(
        task_manifest.frozen_protocol.split_policy.get("initial_window", 3)
    )
    min_train_size = min(min_train_size, max(1, effective_feature_row_count - 1))
    horizon = min(horizon, max(1, effective_feature_row_count - min_train_size))
    min_train_size = max(
        1,
        min(min_train_size, effective_feature_row_count - horizon),
    )
    intake = build_operator_intake_plan(
        csv_path=dataset_path,
        catalog=catalog,
        registry=registry,
        cutoff_available_at=str(
            task_manifest.frozen_protocol.snapshot_policy.get("availability_cutoff")
            or ""
        )
        or None,
        quantization_step=_resolve_quantization_step(task_manifest),
        min_train_size=min_train_size,
        horizon=horizon,
        search_seed=str(task_manifest.frozen_protocol.seed_policy.get("seed", "0")),
        search_class=_search_class(task_manifest),
        proposal_limit=int(
            task_manifest.frozen_protocol.budget_policy.get("candidate_limit", 16)
        ),
        minimum_description_gain_bits=_minimum_description_gain_bits(task_manifest),
        seasonal_period=max(2, min(12, horizon)),
    )
    evaluation_plan = replace(
        intake.evaluation_plan_object,
        forecast_object_type=task_manifest.frozen_protocol.forecast_object_type,
    )
    feature_view = _attach_snapshot_side_information(
        feature_view=intake.feature_view_object,
        snapshot=intake.snapshot_object,
    )
    return BenchmarkHarnessContext(
        task_manifest=task_manifest,
        snapshot=intake.snapshot_object,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        canonicalization_policy_ref=intake.canonicalization_policy.manifest.ref,
        codelength_policy_ref=intake.codelength_policy.manifest.ref,
        reference_description_policy_ref=intake.reference_description_policy.manifest.ref,
        observation_model_ref=intake.observation_model.manifest.ref,
        search_class=_search_class(task_manifest),
        seasonal_period=max(2, min(12, horizon)),
        minimum_description_gain_bits=_minimum_description_gain_bits(task_manifest),
        project_root=project_root,
        proposal_specs=_benchmark_proposal_specs(
            task_manifest=task_manifest,
            entity_panel=intake.evaluation_plan_object.entity_panel,
        ),
    )


def _task_semantic_summary(task_manifest: BenchmarkTaskManifest) -> dict[str, Any]:
    return build_benchmark_task_track_summary(task_manifest)


def _suite_semantic_summary(
    task_results: tuple[ProfiledBenchmarkTaskResult, ...],
) -> dict[str, Any]:
    forecast_object_types = sorted(
        {
            result.task_manifest.frozen_protocol.forecast_object_type
            for result in task_results
        }
    )
    score_laws = sorted({result.task_manifest.score_law for result in task_results})
    abstention_modes = sorted(
        {result.task_manifest.abstention_mode for result in task_results}
    )
    replay_obligations = sorted(
        {result.task_manifest.replay_obligation for result in task_results}
    )
    calibration_expectations = {
        result.task_manifest.frozen_protocol.forecast_object_type: expectation
        for result in task_results
        for expectation in (result.task_manifest.calibration_expectation,)
        if expectation is not None
    }
    return {
        "forecast_object_types": forecast_object_types,
        "score_laws": score_laws,
        "abstention_modes": abstention_modes,
        "replay_obligations": replay_obligations,
        "calibration_expectations": calibration_expectations,
        "run_support_object_ids": sorted(
            {
                value
                for result in task_results
                for value in build_benchmark_task_semantic_summary(
                    result.task_manifest
                )["run_support_object_ids"]
            }
        ),
        "claim_lane_ids": sorted(
            {
                value
                for result in task_results
                for value in build_benchmark_task_semantic_summary(
                    result.task_manifest
                )["claim_lane_ids"]
            }
        ),
        "replay_ids": sorted(
            {
                value
                for result in task_results
                for value in build_benchmark_task_semantic_summary(
                    result.task_manifest
                )["replay_ids"]
            }
        ),
        "engine_ids": sorted(
            {
                value
                for result in task_results
                for value in build_benchmark_task_semantic_summary(
                    result.task_manifest
                )["engine_ids"]
            }
        ),
        "score_policy_ids": sorted(
            {
                value
                for result in task_results
                for value in build_benchmark_task_semantic_summary(
                    result.task_manifest
                )["score_policy_ids"]
            }
        ),
        "threshold_ids": sorted(
            {
                value
                for result in task_results
                for value in build_benchmark_task_semantic_summary(
                    result.task_manifest
                )["threshold_ids"]
            }
        ),
    }


def _task_semantic_status(result: ProfiledBenchmarkTaskResult) -> str:
    if not result.report_paths.task_result_path.is_file():
        return "missing"
    try:
        payload = json.loads(
            result.report_paths.task_result_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return "missing"
    if payload.get("status") not in {"completed", "passed"}:
        return "failed"
    semantic_summary = payload.get("semantic_summary")
    if not isinstance(semantic_summary, dict):
        return "missing"
    required_keys = {
        "run_support_object_ids",
        "claim_lane_ids",
        "replay_ids",
        "engine_ids",
        "score_policy_ids",
        "threshold_ids",
    }
    if any(
        not isinstance(semantic_summary.get(key), list) or not semantic_summary[key]
        for key in required_keys
    ):
        return "missing"
    if _semantic_assertion_status(payload) != "passed":
        return "failed"
    return "verified"


def _task_semantic_assertion_status(result: ProfiledBenchmarkTaskResult) -> str:
    if not result.report_paths.task_result_path.is_file():
        return "missing"
    try:
        payload = json.loads(
            result.report_paths.task_result_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return "missing"
    return _semantic_assertion_status(payload)


def _semantic_assertion_status(payload: dict[str, Any]) -> str:
    semantic_assertions = payload.get("semantic_assertions")
    if not isinstance(semantic_assertions, dict):
        return "missing"
    status = semantic_assertions.get("overall_status")
    if status == "passed":
        claim_scope = semantic_assertions.get("claim_scope")
        if isinstance(claim_scope, dict) and claim_scope.get(
            "counts_as_claim_evidence"
        ):
            return "failed"
        return "passed"
    return "failed"


def _search_class_coverage(
    task_results: tuple[ProfiledBenchmarkTaskResult, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for search_class in sorted(
        {
            result.task_manifest.search_class
            for result in task_results
            if result.task_manifest.search_class is not None
            and result.task_manifest.search_class_honesty
        }
    ):
        matching = tuple(
            result
            for result in task_results
            if result.task_manifest.search_class == search_class
            and result.task_manifest.search_class_honesty
        )
        if not matching:
            continue
        contract = dict(matching[0].task_manifest.search_class_honesty)
        rows.append(
            {
                "search_class": search_class,
                **contract,
                "proof_mode": "direct_benchmark_task",
                "covered_task_ids": sorted(
                    result.task_manifest.task_id for result in matching
                ),
                "replay_verified": _tasks_have_replay_artifacts(matching),
            }
        )
    return rows


def _composition_operator_coverage(
    task_results: tuple[ProfiledBenchmarkTaskResult, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for operator_id in sorted(
        {
            operator
            for result in task_results
            for operator in result.task_manifest.composition_operators
        }
    ):
        matching = tuple(
            result
            for result in task_results
            if operator_id in result.task_manifest.composition_operators
        )
        rows.append(
            {
                "composition_operator": operator_id,
                "proof_mode": "direct_benchmark_task",
                "covered_task_ids": sorted(
                    result.task_manifest.task_id for result in matching
                ),
                "replay_verified": _tasks_have_replay_artifacts(matching),
            }
        )
    return rows


def _tasks_have_replay_artifacts(
    task_results: tuple[ProfiledBenchmarkTaskResult, ...],
) -> bool:
    return all(
        _task_replay_verification_status(result) == "verified"
        for result in task_results
    )


def _semantic_reason_codes(
    *,
    requirement: BenchmarkSuiteSurfaceRequirement,
    matched: tuple[ProfiledBenchmarkTaskResult, ...],
    task_semantic_status: Mapping[str, str],
    task_semantic_assertion_status: Mapping[str, str],
) -> tuple[str, ...]:
    reason_codes: list[str] = []
    if len(matched) != len(requirement.task_ids):
        reason_codes.append("task_result_missing")
    if any(status == "missing" for status in task_semantic_status.values()):
        reason_codes.append("semantic_summary_missing")
    if any(status == "missing" for status in task_semantic_assertion_status.values()):
        reason_codes.append("semantic_assertion_missing")
    if any(status == "failed" for status in task_semantic_assertion_status.values()):
        reason_codes.append("semantic_assertion_failed")
    return tuple(dict.fromkeys(reason_codes))


def _submitter_results_with_threshold_metrics(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_results: tuple[BenchmarkSubmitterResult, ...],
    context: BenchmarkHarnessContext | None = None,
    project_root: Path | str | None = None,
) -> tuple[BenchmarkSubmitterResult, ...]:
    if not task_manifest.metric_thresholds:
        return submitter_results
    catalog = (
        load_contract_catalog(_resolve_project_root(project_root))
        if context is not None
        else None
    )
    enriched_results = tuple(
        _submitter_result_with_threshold_metrics(
            task_manifest=task_manifest,
            submitter_result=submitter_result,
            context=context,
            catalog=catalog,
        )
        for submitter_result in submitter_results
    )
    return _submitter_results_with_metric_preferred_portfolio(
        task_manifest=task_manifest,
        submitter_results=enriched_results,
    )


def _submitter_results_with_metric_preferred_portfolio(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_results: tuple[BenchmarkSubmitterResult, ...],
) -> tuple[BenchmarkSubmitterResult, ...]:
    portfolio_index = next(
        (
            index
            for index, result in enumerate(submitter_results)
            if result.submitter_id == PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
        ),
        None,
    )
    if portfolio_index is None:
        return submitter_results
    portfolio_result = submitter_results[portfolio_index]
    child_results = tuple(
        result
        for result in submitter_results
        if result.submitter_id != PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
    )
    selected_submitter_id = _string_or_none(
        portfolio_result.replay_contract.get("selected_submitter_id")
    )
    selected_child = next(
        (
            child
            for child in child_results
            if child.submitter_id == selected_submitter_id
        ),
        None,
    )
    if selected_child is not None and _result_satisfies_metric_thresholds(
        task_manifest=task_manifest,
        submitter_result=selected_child,
    ):
        return submitter_results
    metric_selected = _metric_preferred_child_result(
        task_manifest=task_manifest,
        child_results=child_results,
    )
    if metric_selected is None or metric_selected is selected_child:
        return submitter_results
    updated_portfolio = _portfolio_result_with_metric_selection(
        task_manifest=task_manifest,
        portfolio_result=portfolio_result,
        child_results=child_results,
        selected_child=metric_selected,
        previous_child=selected_child,
    )
    return (
        *submitter_results[:portfolio_index],
        updated_portfolio,
        *submitter_results[portfolio_index + 1 :],
    )


def _metric_preferred_child_result(
    *,
    task_manifest: BenchmarkTaskManifest,
    child_results: tuple[BenchmarkSubmitterResult, ...],
) -> BenchmarkSubmitterResult | None:
    threshold_passing = tuple(
        child
        for child in child_results
        if _result_satisfies_metric_thresholds(
            task_manifest=task_manifest,
            submitter_result=child,
        )
    )
    if not threshold_passing:
        return None
    return min(
        threshold_passing,
        key=lambda child: _metric_preferred_sort_key(
            task_manifest=task_manifest,
            submitter_result=child,
        ),
    )


def _portfolio_result_with_metric_selection(
    *,
    task_manifest: BenchmarkTaskManifest,
    portfolio_result: BenchmarkSubmitterResult,
    child_results: tuple[BenchmarkSubmitterResult, ...],
    selected_child: BenchmarkSubmitterResult,
    previous_child: BenchmarkSubmitterResult | None,
) -> BenchmarkSubmitterResult:
    decision_trace = (
        *portfolio_result.decision_trace,
        {
            "step": "benchmark_metric_threshold_gate",
            "reason_code": "selected_first_threshold_passing_finalist",
            "previous_submitter_id": (
                previous_child.submitter_id if previous_child is not None else None
            ),
            "previous_candidate_id": (
                previous_child.selected_candidate_id
                if previous_child is not None
                else None
            ),
            "selected_submitter_id": selected_child.submitter_id,
            "selected_candidate_id": selected_child.selected_candidate_id,
            "threshold_status_by_submitter": _threshold_status_by_submitter(
                task_manifest=task_manifest,
                child_results=child_results,
            ),
        },
        {
            "step": "select_portfolio_winner",
            "selected_backend_family": _string_or_none(
                (
                    selected_child.selected_candidate.structural_layer.cir_family_id
                    if selected_child.selected_candidate is not None
                    else None
                )
            ),
            "selected_candidate_hash": selected_child.selected_candidate_hash,
            "selected_candidate_id": selected_child.selected_candidate_id,
        },
    )
    replay_contract = dict(portfolio_result.replay_contract)
    replay_contract.update(
        {
            "selected_provenance_id": selected_child.submitter_id,
            "selected_submitter_id": selected_child.submitter_id,
            "selected_candidate_id": selected_child.selected_candidate_id,
            "selected_candidate_hash": selected_child.selected_candidate_hash,
            "decision_trace": decision_trace,
            "compared_finalists": _selected_first_finalists(
                finalists=portfolio_result.compared_finalists,
                selected_child=selected_child,
            ),
        }
    )
    replay_contract.update(
        verify_portfolio_replay_contract(
            replay_contract,
            selected_candidate_id=selected_child.selected_candidate_id,
            selected_candidate_hash=selected_child.selected_candidate_hash,
            compared_finalists=replay_contract["compared_finalists"],
            decision_trace=decision_trace,
        )
    )
    return replace(
        portfolio_result,
        selected_candidate=selected_child.selected_candidate,
        selected_candidate_id=selected_child.selected_candidate_id,
        selected_candidate_hash=selected_child.selected_candidate_hash,
        selected_candidate_metrics=selected_child.selected_candidate_metrics,
        replay_contract=replay_contract,
        child_results=child_results,
        compared_finalists=tuple(replay_contract["compared_finalists"]),
        decision_trace=decision_trace,
    )


def _threshold_status_by_submitter(
    *,
    task_manifest: BenchmarkTaskManifest,
    child_results: tuple[BenchmarkSubmitterResult, ...],
) -> dict[str, dict[str, str]]:
    return {
        child.submitter_id: {
            threshold_id: (
                "passed"
                if _metric_threshold_passes(
                    metrics=child.selected_candidate_metrics,
                    threshold=threshold,
                )
                else "failed"
            )
            for threshold_id, threshold in sorted(task_manifest.metric_thresholds.items())
        }
        for child in child_results
        if child.status == "selected"
    }


def _selected_first_finalists(
    *,
    finalists: tuple[Mapping[str, Any], ...],
    selected_child: BenchmarkSubmitterResult,
) -> tuple[Mapping[str, Any], ...]:
    selected: list[Mapping[str, Any]] = []
    before_selected: list[Mapping[str, Any]] = []
    after_selected: list[Mapping[str, Any]] = []
    found_selected = False
    for finalist in finalists:
        finalist_submitter_id = finalist.get("submitter_id") or finalist.get(
            "provenance_id"
        )
        finalist_candidate_id = finalist.get("candidate_id")
        if (
            finalist_submitter_id == selected_child.submitter_id
            and finalist_candidate_id == selected_child.selected_candidate_id
        ):
            selected.append(finalist)
            found_selected = True
        elif found_selected:
            after_selected.append(finalist)
        else:
            before_selected.append(finalist)
    return tuple((*selected, *before_selected, *after_selected))


def _result_satisfies_metric_thresholds(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_result: BenchmarkSubmitterResult,
) -> bool:
    if submitter_result.status != "selected":
        return False
    if not task_manifest.metric_thresholds:
        return False
    return all(
        _metric_threshold_passes(
            metrics=submitter_result.selected_candidate_metrics,
            threshold=threshold,
        )
        for threshold in task_manifest.metric_thresholds.values()
    )


def _metric_threshold_passes(
    *,
    metrics: Mapping[str, Any] | None,
    threshold: Any,
) -> bool:
    if not isinstance(threshold, Mapping):
        return False
    metric_id = threshold.get("metric_id")
    if not isinstance(metric_id, str) or not metric_id.strip():
        return False
    comparator = threshold.get("comparator")
    if not isinstance(comparator, str) or not comparator.strip():
        return False
    threshold_value = threshold.get("threshold")
    measurement_required = threshold.get("measurement_required", True)
    if not isinstance(measurement_required, bool):
        return False
    observed_value = metrics.get(metric_id.strip()) if isinstance(metrics, Mapping) else None
    if observed_value is None:
        return not measurement_required
    return _compare_metric_threshold(
        observed_value,
        comparator=comparator.strip(),
        threshold_value=threshold_value,
    )


def _compare_metric_threshold(
    observed_value: Any,
    *,
    comparator: str,
    threshold_value: Any,
) -> bool:
    if not isinstance(observed_value, (int, float)) or isinstance(observed_value, bool):
        return False
    if not isinstance(threshold_value, (int, float)) or isinstance(threshold_value, bool):
        return False
    observed = float(observed_value)
    threshold = float(threshold_value)
    if comparator == ">=":
        return observed >= threshold
    if comparator == "<=":
        return observed <= threshold
    if comparator == ">":
        return observed > threshold
    if comparator == "<":
        return observed < threshold
    if comparator == "==":
        return observed == threshold
    return False


def _metric_preferred_sort_key(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_result: BenchmarkSubmitterResult,
) -> tuple[float, float, float, str, str]:
    metrics = submitter_result.selected_candidate_metrics or {}
    return (
        -_numeric_metric(metrics, "practical_significance_margin", default=-math.inf),
        _numeric_metric(metrics, task_manifest.score_law, default=math.inf),
        _numeric_metric(metrics, "total_code_bits", default=math.inf),
        submitter_result.submitter_id,
        submitter_result.selected_candidate_id or "",
    )


def _numeric_metric(
    metrics: Mapping[str, Any],
    metric_id: str,
    *,
    default: float,
) -> float:
    value = metrics.get(metric_id)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _submitter_result_with_threshold_metrics(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_result: BenchmarkSubmitterResult,
    context: BenchmarkHarnessContext | None = None,
    catalog: Any | None = None,
) -> BenchmarkSubmitterResult:
    if submitter_result.selected_candidate_id is None:
        return submitter_result
    metrics = dict(submitter_result.selected_candidate_metrics or {})
    missing_metric_ids: list[str] = []
    for threshold in task_manifest.metric_thresholds.values():
        if not isinstance(threshold, dict):
            continue
        metric_id = threshold.get("metric_id")
        if not isinstance(metric_id, str) or not metric_id.strip():
            continue
        metric_id = metric_id.strip()
        if metric_id not in metrics:
            missing_metric_ids.append(metric_id)
    if not missing_metric_ids:
        return submitter_result
    measured_metrics = _measured_threshold_metrics(
        task_manifest=task_manifest,
        submitter_result=submitter_result,
        context=context,
        catalog=catalog,
    )
    changed = False
    for threshold in task_manifest.metric_thresholds.values():
        if not isinstance(threshold, dict):
            continue
        metric_id = threshold.get("metric_id")
        if not isinstance(metric_id, str) or not metric_id.strip():
            continue
        metric_id = metric_id.strip()
        if metric_id not in missing_metric_ids:
            continue
        value = _observed_threshold_metric_value(
            task_manifest=task_manifest,
            metrics=metrics,
            measured_metrics=measured_metrics,
            metric_id=metric_id,
            threshold=threshold,
        )
        if value is None:
            continue
        metrics[metric_id] = value
        changed = True
    if not changed:
        return submitter_result
    return replace(submitter_result, selected_candidate_metrics=metrics)


def _observed_threshold_metric_value(
    *,
    task_manifest: BenchmarkTaskManifest,
    metrics: dict[str, Any],
    measured_metrics: Mapping[str, Any],
    metric_id: str,
    threshold: dict[str, Any],
) -> float | None:
    measured_value = measured_metrics.get(metric_id)
    if isinstance(measured_value, (int, float)) and not isinstance(
        measured_value,
        bool,
    ):
        return _stable_metric_float(float(measured_value))
    return None


def _measured_threshold_metrics(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_result: BenchmarkSubmitterResult,
    context: BenchmarkHarnessContext | None,
    catalog: Any | None,
) -> dict[str, float]:
    if (
        context is None
        or catalog is None
        or submitter_result.selected_candidate is None
    ):
        return {}
    forecast_object_type = task_manifest.frozen_protocol.forecast_object_type
    if forecast_object_type == "point":
        return _measured_point_threshold_metrics(
            task_manifest=task_manifest,
            submitter_result=submitter_result,
            context=context,
        )
    return _measured_probabilistic_threshold_metrics(
        task_manifest=task_manifest,
        submitter_result=submitter_result,
        context=context,
        catalog=catalog,
        forecast_object_type=forecast_object_type,
    )


def _measured_point_threshold_metrics(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_result: BenchmarkSubmitterResult,
    context: BenchmarkHarnessContext,
) -> dict[str, float]:
    candidate = submitter_result.selected_candidate
    if candidate is None:
        return {}
    search_plan = context.build_search_plan(
        submitter_id=submitter_result.submitter_id,
        candidate_family_ids=(submitter_result.selected_candidate_id or "candidate",),
    )
    feature_view = context.feature_view.require_stage_reuse("evaluation")
    rows_by_entity = _benchmark_rows_by_entity(feature_view)
    errors: list[float] = []
    baseline_errors_by_id: dict[str, list[float]] = {}
    for segment in _benchmark_metric_segments(context):
        fit_result = fit_candidate_window(
            candidate=candidate,
            feature_view=context.feature_view,
            fit_window=segment,
            search_plan=search_plan,
            stage_id="benchmark_metric",
        )
        for scored_origin in _segment_scored_origins(
            context=context,
            segment_id=segment.segment_id,
        ):
            entity = scored_origin.entity or feature_view.series_id
            entity_rows = rows_by_entity.get(entity, ())
            if scored_origin.origin_index >= len(entity_rows):
                continue
            origin_row = entity_rows[scored_origin.origin_index]
            target_row = resolve_scored_origin_target_row(
                feature_view=feature_view,
                scored_origin=scored_origin,
            )
            if target_row is None:
                continue
            forecast_path = build_forecast_path(
                candidate=fit_result.fitted_candidate,
                fit_result=fit_result,
                origin_row=origin_row,
                max_horizon=max(segment.horizon_set),
                entity=scored_origin.entity,
            )
            point_forecast = forecast_path.predictions.get(scored_origin.horizon)
            if point_forecast is None or not math.isfinite(float(point_forecast)):
                continue
            realized = float(target_row["target"])
            if not math.isfinite(realized):
                continue
            errors.append(abs(float(point_forecast) - realized))
            for baseline_id, baseline_value in _point_baseline_predictions(
                task_manifest=task_manifest,
                rows_by_entity=rows_by_entity,
                entity=entity,
                scored_origin=scored_origin,
            ).items():
                baseline_errors_by_id.setdefault(baseline_id, []).append(
                    abs(baseline_value - realized)
                )
    if not errors:
        return {}
    score = _stable_metric_float(fmean(errors))
    metrics = {task_manifest.score_law: score}
    baseline_scores = [
        fmean(baseline_errors)
        for baseline_errors in baseline_errors_by_id.values()
        if baseline_errors
    ]
    if baseline_scores:
        metrics["practical_significance_margin"] = _stable_metric_float(
            min(baseline_scores) - score
        )
    return metrics


def _point_baseline_predictions(
    *,
    task_manifest: BenchmarkTaskManifest,
    rows_by_entity: Mapping[str, tuple[Mapping[str, Any], ...]],
    entity: str,
    scored_origin: Any,
) -> dict[str, float]:
    predictions: dict[str, float] = {}
    entity_rows = rows_by_entity.get(entity, ())
    for baseline in _point_forecast_baseline_specs(task_manifest):
        baseline_id = str(baseline["baseline_id"])
        rule = baseline["rule"]
        if rule == "last_observation_carried_forward":
            value = _point_baseline_value_at_index(
                entity_rows,
                index=int(scored_origin.origin_index),
            )
        elif rule == "last_seasonal_cycle":
            value = _seasonal_baseline_value(
                entity_rows,
                origin_index=int(scored_origin.origin_index),
                target_index=int(scored_origin.target_index),
                seasonal_period=int(baseline["seasonal_period"]),
            )
        else:
            value = None
        if value is not None:
            predictions[baseline_id] = value
    return predictions


def _point_forecast_baseline_specs(
    task_manifest: BenchmarkTaskManifest,
) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for entry in getattr(task_manifest, "baseline_registry", ()):
        payload = _baseline_registry_payload(
            entry,
            task_manifest=task_manifest,
        )
        baseline_id = _string_or_none(payload.get("baseline_id")) or _string_or_none(
            getattr(entry, "entry_id", None)
        )
        if baseline_id is None:
            continue
        policy = payload.get("policy")
        policy = policy if isinstance(policy, Mapping) else {}
        rule = _string_or_none(policy.get("rule"))
        if (
            baseline_id == "naive_last_value"
            or rule == "last_observation_carried_forward"
        ):
            specs.append(
                {
                    "baseline_id": baseline_id,
                    "rule": "last_observation_carried_forward",
                }
            )
            continue
        if baseline_id == "seasonal_naive" or rule == "last_seasonal_cycle":
            seasonal_period = _positive_int_or_none(policy.get("seasonal_period"))
            if seasonal_period is None:
                continue
            specs.append(
                {
                    "baseline_id": baseline_id,
                    "rule": "last_seasonal_cycle",
                    "seasonal_period": seasonal_period,
                }
            )
    return tuple(specs)


def _baseline_registry_payload(
    entry: Any,
    *,
    task_manifest: BenchmarkTaskManifest,
) -> dict[str, Any]:
    payload = getattr(entry, "payload", None)
    payload = dict(payload) if isinstance(payload, Mapping) else {}
    manifest_payload = _baseline_manifest_payload(
        payload.get("manifest_path"),
        task_manifest=task_manifest,
    )
    if not manifest_payload:
        return payload
    merged = {**manifest_payload, **payload}
    manifest_policy = manifest_payload.get("policy")
    payload_policy = payload.get("policy")
    if isinstance(manifest_policy, Mapping) or isinstance(payload_policy, Mapping):
        merged["policy"] = {
            **(dict(manifest_policy) if isinstance(manifest_policy, Mapping) else {}),
            **(dict(payload_policy) if isinstance(payload_policy, Mapping) else {}),
        }
    return merged


def _baseline_manifest_payload(
    manifest_path: Any,
    *,
    task_manifest: BenchmarkTaskManifest,
) -> dict[str, Any]:
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        return {}
    for candidate_path in _baseline_manifest_candidate_paths(
        manifest_path.strip(),
        task_manifest=task_manifest,
    ):
        if not candidate_path.is_file():
            continue
        try:
            loaded = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        return dict(loaded) if isinstance(loaded, Mapping) else {}
    return {}


def _baseline_manifest_candidate_paths(
    manifest_path: str,
    *,
    task_manifest: BenchmarkTaskManifest,
) -> tuple[Path, ...]:
    path = Path(manifest_path)
    if path.is_absolute():
        return (path,)
    candidates: list[Path] = [path]
    source_path = getattr(task_manifest, "source_path", None)
    if isinstance(source_path, Path):
        parts = source_path.resolve().parts
        if "benchmarks" in parts:
            benchmark_index = parts.index("benchmarks")
            project_root = Path(*parts[:benchmark_index])
            candidates.extend(
                (
                    project_root / path,
                    project_root / "src" / "euclid" / "_assets" / path,
                )
            )
        candidates.append(source_path.parent / path)
    return tuple(dict.fromkeys(candidate.resolve() for candidate in candidates))


def _point_baseline_value_at_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    index: int,
) -> float | None:
    if index < 0 or index >= len(rows):
        return None
    value = rows[index].get("target")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _stable_metric_float(float(value))
    return None


def _seasonal_baseline_value(
    rows: Sequence[Mapping[str, Any]],
    *,
    origin_index: int,
    target_index: int,
    seasonal_period: int,
) -> float | None:
    if seasonal_period <= 0:
        return None
    baseline_index = target_index - seasonal_period
    while baseline_index > origin_index:
        baseline_index -= seasonal_period
    return _point_baseline_value_at_index(rows, index=baseline_index)


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _measured_probabilistic_threshold_metrics(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_result: BenchmarkSubmitterResult,
    context: BenchmarkHarnessContext,
    catalog: Any,
    forecast_object_type: str,
) -> dict[str, float]:
    candidate = submitter_result.selected_candidate
    if candidate is None:
        return {}
    score_policy = _benchmark_score_policy_manifest(
        catalog=catalog,
        task_manifest=task_manifest,
        context=context,
    )
    search_plan = context.build_search_plan(
        submitter_id=submitter_result.submitter_id,
        candidate_family_ids=(submitter_result.selected_candidate_id or "candidate",),
    )
    search_plan_manifest = search_plan.to_manifest(catalog)
    prediction_artifacts: list[ManifestEnvelope] = []
    supporting_artifacts: list[ManifestEnvelope] = []
    for segment in _benchmark_metric_segments(context):
        fit_result = fit_candidate_window(
            candidate=candidate,
            feature_view=context.feature_view,
            fit_window=segment,
            search_plan=search_plan,
            stage_id="benchmark_metric",
        )
        fit_artifacts = build_candidate_fit_artifacts(
            catalog=catalog,
            fit_result=fit_result,
            search_plan_ref=search_plan_manifest.ref,
            selection_floor_bits=0.0,
        )
        prediction_artifacts.append(
            emit_probabilistic_prediction_artifact(
                catalog=catalog,
                feature_view=context.feature_view,
                evaluation_plan=context.evaluation_plan,
                evaluation_segment=segment,
                fit_result=fit_result,
                score_policy_manifest=score_policy,
                stage_id="outer_test",
                forecast_object_type=forecast_object_type,
                stochastic_evidence_mode="production",
                stochastic_fit_result=fit_result,
                residual_history_ref=fit_artifacts.residual_history.ref,
                supporting_artifact_sink=supporting_artifacts,
            )
        )
    prediction_artifact = _combined_prediction_artifact(
        catalog=catalog,
        task_manifest=task_manifest,
        submitter_result=submitter_result,
        score_policy=score_policy,
        prediction_artifacts=tuple(prediction_artifacts),
    )
    if prediction_artifact is None:
        return {}
    metrics: dict[str, float] = {}
    score_result = score_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy,
        prediction_artifact_manifest=prediction_artifact,
    )
    if (
        score_result.body.get("comparison_status") == "comparable"
        and isinstance(score_result.body.get("aggregated_primary_score"), (int, float))
    ):
        candidate_score = float(score_result.body["aggregated_primary_score"])
        metrics[task_manifest.score_law] = _stable_metric_float(candidate_score)
        baseline_score = _last_value_probabilistic_baseline_score(
            catalog=catalog,
            task_manifest=task_manifest,
            submitter_result=submitter_result,
            context=context,
            score_policy=score_policy,
            candidate_prediction_artifact=prediction_artifact,
        )
        if baseline_score is not None:
            metrics["practical_significance_margin"] = _stable_metric_float(
                baseline_score - candidate_score
            )
    calibration_policy = dict(task_manifest.frozen_protocol.calibration_policy or {})
    calibration_contract = build_calibration_contract(
        catalog=catalog,
        forecast_object_type=forecast_object_type,
        thresholds=_numeric_policy_values(calibration_policy),
    )
    calibration_result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=calibration_contract,
        prediction_artifact_manifest=prediction_artifact,
    )
    metrics.update(_calibration_threshold_metric_values(calibration_result.body))
    return metrics


def _last_value_probabilistic_baseline_score(
    *,
    catalog: Any,
    task_manifest: BenchmarkTaskManifest,
    submitter_result: BenchmarkSubmitterResult,
    context: BenchmarkHarnessContext,
    score_policy: ManifestEnvelope,
    candidate_prediction_artifact: ManifestEnvelope,
) -> float | None:
    baseline_artifact = _last_value_probabilistic_baseline_artifact(
        catalog=catalog,
        task_manifest=task_manifest,
        submitter_result=submitter_result,
        context=context,
        score_policy=score_policy,
        candidate_prediction_artifact=candidate_prediction_artifact,
    )
    if baseline_artifact is None:
        return None
    score_result = score_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy,
        prediction_artifact_manifest=baseline_artifact,
    )
    if (
        score_result.body.get("comparison_status") != "comparable"
        or not isinstance(
            score_result.body.get("aggregated_primary_score"),
            (int, float),
        )
    ):
        return None
    return _stable_metric_float(float(score_result.body["aggregated_primary_score"]))


def _last_value_probabilistic_baseline_artifact(
    *,
    catalog: Any,
    task_manifest: BenchmarkTaskManifest,
    submitter_result: BenchmarkSubmitterResult,
    context: BenchmarkHarnessContext,
    score_policy: ManifestEnvelope,
    candidate_prediction_artifact: ManifestEnvelope,
) -> ManifestEnvelope | None:
    feature_view = context.feature_view.require_stage_reuse("evaluation")
    rows_by_entity = _benchmark_rows_by_entity(feature_view)
    baseline_rows = []
    for row in candidate_prediction_artifact.body.get("rows", ()):
        origin_value = _last_observation_value_for_prediction_row(
            row=row,
            rows_by_entity=rows_by_entity,
            default_entity=feature_view.series_id,
        )
        if origin_value is None:
            return None
        baseline_scale = _last_value_baseline_scale_for_prediction_row(
            row=row,
            rows_by_entity=rows_by_entity,
            default_entity=feature_view.series_id,
        )
        baseline_rows.append(
            _probabilistic_prediction_row_with_location(
                row=row,
                location=origin_value,
                scale=baseline_scale,
            )
        )
    if not baseline_rows:
        return None
    body = dict(candidate_prediction_artifact.body)
    body.update(
        {
            "prediction_artifact_id": (
                f"{task_manifest.task_id}__naive_last_value__benchmark_metrics"
            ),
            "candidate_id": "naive_last_value",
            "score_policy_ref": score_policy.ref.as_dict(),
            "rows": baseline_rows,
            "comparison_key": {
                **dict(candidate_prediction_artifact.body.get("comparison_key", {})),
                "baseline_id": "naive_last_value",
            },
            "effective_probabilistic_config": {
                **dict(
                    candidate_prediction_artifact.body.get(
                        "effective_probabilistic_config",
                        {},
                    )
                ),
                "baseline_id": "naive_last_value",
                "baseline_location_rule": "last_observed_value_at_origin",
            },
        }
    )
    del submitter_result
    return ManifestEnvelope.build(
        schema_name=candidate_prediction_artifact.schema_name,
        module_id=candidate_prediction_artifact.module_id,
        body=body,
        catalog=catalog,
    )


def _last_observation_value_for_prediction_row(
    *,
    row: Mapping[str, Any],
    rows_by_entity: Mapping[str, tuple[Mapping[str, Any], ...]],
    default_entity: str,
) -> float | None:
    entity = str(row.get("entity") or default_entity)
    origin_time = str(row.get("origin_time", ""))
    for feature_row in rows_by_entity.get(entity, ()):
        if str(feature_row.get("event_time")) == origin_time:
            value = feature_row.get("target")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return _stable_metric_float(float(value))
    return None


def _last_value_baseline_scale_for_prediction_row(
    *,
    row: Mapping[str, Any],
    rows_by_entity: Mapping[str, tuple[Mapping[str, Any], ...]],
    default_entity: str,
) -> float | None:
    entity = str(row.get("entity") or default_entity)
    origin_time = str(row.get("origin_time", ""))
    history = tuple(
        feature_row
        for feature_row in rows_by_entity.get(entity, ())
        if str(feature_row.get("event_time", "")) <= origin_time
        and isinstance(feature_row.get("target"), (int, float))
        and not isinstance(feature_row.get("target"), bool)
    )
    residuals = [
        float(current["target"]) - float(previous["target"])
        for previous, current in zip(history, history[1:])
    ]
    if not residuals:
        return None
    mean_residual = fmean(residuals)
    variance = fmean((residual - mean_residual) ** 2 for residual in residuals)
    return _stable_metric_float(max(math.sqrt(variance), 1e-9))


def _probabilistic_prediction_row_with_location(
    *,
    row: Mapping[str, Any],
    location: float,
    scale: float | None = None,
) -> dict[str, Any]:
    baseline_row = dict(row)
    old_location = _probabilistic_row_location(row)
    old_scale = _probabilistic_row_scale(row)
    scale_ratio = (
        _stable_metric_float(float(scale) / old_scale)
        if scale is not None and old_scale > 0.0
        else 1.0
    )
    baseline_row["distribution_parameters"] = _distribution_parameters_with_location(
        row,
        location=location,
        scale=scale,
    )
    if "location" in baseline_row:
        baseline_row["location"] = _stable_metric_float(float(location))
    if "lower_bound" in baseline_row and "upper_bound" in baseline_row:
        baseline_row["lower_bound"] = _rescale_around_location(
            value=float(baseline_row["lower_bound"]),
            old_location=old_location,
            new_location=location,
            scale_ratio=scale_ratio,
        )
        baseline_row["upper_bound"] = _rescale_around_location(
            value=float(baseline_row["upper_bound"]),
            old_location=old_location,
            new_location=location,
            scale_ratio=scale_ratio,
        )
    if isinstance(baseline_row.get("intervals"), list):
        baseline_row["intervals"] = [
            {
                **dict(interval),
                "lower_bound": _rescale_around_location(
                    value=float(interval["lower_bound"]),
                    old_location=old_location,
                    new_location=location,
                    scale_ratio=scale_ratio,
                ),
                "upper_bound": _rescale_around_location(
                    value=float(interval["upper_bound"]),
                    old_location=old_location,
                    new_location=location,
                    scale_ratio=scale_ratio,
                ),
            }
            for interval in baseline_row["intervals"]
        ]
    if isinstance(baseline_row.get("quantiles"), list):
        baseline_row["quantiles"] = [
            {
                **dict(quantile),
                "value": _rescale_around_location(
                    value=float(quantile["value"]),
                    old_location=old_location,
                    new_location=location,
                    scale_ratio=scale_ratio,
                ),
            }
            for quantile in baseline_row["quantiles"]
        ]
    if "event_probability" in baseline_row and "event_definition" in baseline_row:
        event_definition = EventDefinition.from_manifest(
            baseline_row["event_definition"]
        )
        baseline_row["event_probability"] = _stable_metric_float(
            event_definition.probability(
                bind_distribution_row_observation_model(baseline_row)
            )
        )
    return baseline_row


def _rescale_around_location(
    *,
    value: float,
    old_location: float,
    new_location: float,
    scale_ratio: float,
) -> float:
    return _stable_metric_float(
        float(new_location) + ((float(value) - float(old_location)) * scale_ratio)
    )


def _probabilistic_row_location(row: Mapping[str, Any]) -> float:
    parameters = row.get("distribution_parameters")
    if isinstance(parameters, Mapping) and isinstance(
        parameters.get("location"),
        (int, float),
    ):
        return float(parameters["location"])
    if isinstance(row.get("location"), (int, float)):
        return float(row["location"])
    if isinstance(row.get("lower_bound"), (int, float)) and isinstance(
        row.get("upper_bound"),
        (int, float),
    ):
        return (float(row["lower_bound"]) + float(row["upper_bound"])) / 2.0
    quantiles = row.get("quantiles")
    if isinstance(quantiles, list):
        median = next(
            (
                float(quantile["value"])
                for quantile in quantiles
                if float(quantile.get("level", math.nan)) == 0.5
            ),
            None,
        )
        if median is not None:
            return median
    return float(row.get("realized_observation", 0.0))


def _probabilistic_row_scale(row: Mapping[str, Any]) -> float:
    parameters = row.get("distribution_parameters")
    if isinstance(parameters, Mapping) and isinstance(
        parameters.get("scale"),
        (int, float),
    ):
        return max(float(parameters["scale"]), 1e-9)
    if isinstance(row.get("lower_bound"), (int, float)) and isinstance(
        row.get("upper_bound"),
        (int, float),
    ):
        return max((float(row["upper_bound"]) - float(row["lower_bound"])) / 2.0, 1e-9)
    return 1.0


def _distribution_parameters_with_location(
    row: Mapping[str, Any],
    *,
    location: float,
    scale: float | None = None,
) -> dict[str, float]:
    parameters = dict(row.get("distribution_parameters") or {})
    parameters["location"] = _stable_metric_float(float(location))
    if scale is not None and "scale" in parameters:
        parameters["scale"] = _stable_metric_float(max(float(scale), 1e-9))
    return {
        str(key): _stable_metric_float(float(value))
        for key, value in parameters.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _benchmark_score_policy_manifest(
    *,
    catalog: Any,
    task_manifest: BenchmarkTaskManifest,
    context: BenchmarkHarnessContext,
) -> ManifestEnvelope:
    forecast_object_type = task_manifest.frozen_protocol.forecast_object_type
    horizon_weights = _equal_weight_simplex(context.evaluation_plan.horizon_set)
    if forecast_object_type == "point":
        body: dict[str, Any] = {
            "score_policy_id": f"{task_manifest.task_id}_point_policy_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": "point",
            "point_loss_id": "absolute_error",
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": horizon_weights,
            "entity_aggregation_mode": (
                "per_entity_primary_score_then_declared_entity_weights"
                if len(context.evaluation_plan.entity_panel) > 1
                else "single_entity_only_no_cross_entity_aggregation"
            ),
            "secondary_diagnostic_ids": [],
            "forbidden_primary_metric_ids": [],
            "lower_is_better": True,
            "comparison_class_rule": "identical_score_policy_required",
        }
        return ManifestEnvelope.build(
            schema_name="point_score_policy_manifest@1.0.0",
            module_id="scoring",
            body=body,
            catalog=catalog,
        )
    schema_name = {
        "distribution": "probabilistic_score_policy_manifest@1.0.0",
        "interval": "interval_score_policy_manifest@1.0.0",
        "quantile": "quantile_score_policy_manifest@1.0.0",
        "event_probability": "event_probability_score_policy_manifest@1.0.0",
    }[forecast_object_type]
    body = {
        "score_policy_id": f"{task_manifest.task_id}_{forecast_object_type}_policy_v1",
        "owner_prompt_id": "prompt.scoring-calibration-v1",
        "scope_id": "euclid_v1_binding_scope@1.0.0",
        "forecast_object_type": forecast_object_type,
        "primary_score": task_manifest.score_law,
        "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
        "horizon_weights": horizon_weights,
        "entity_aggregation_mode": "single_entity_only_no_cross_entity_aggregation",
        "secondary_diagnostic_ids": [],
        "forbidden_primary_metric_ids": [],
        "lower_is_better": True,
        "comparison_class_rule": "identical_score_policy_required",
    }
    if forecast_object_type == "event_probability":
        body["event_definition"] = _benchmark_event_definition(task_manifest)
    return ManifestEnvelope.build(
        schema_name=schema_name,
        module_id="scoring",
        body=body,
        catalog=catalog,
    )


def _benchmark_event_definition(task_manifest: BenchmarkTaskManifest) -> dict[str, Any]:
    calibration_policy = task_manifest.frozen_protocol.calibration_policy or {}
    declared = calibration_policy.get("event_definition")
    if isinstance(declared, Mapping):
        return dict(declared)
    return {
        "event_id": f"{task_manifest.task_id}_target_ge_zero",
        "operator": "greater_than_or_equal",
        "threshold": 0.0,
        "threshold_source": "declared_literal",
        "variable": "target",
        "calibration_required": True,
    }


def _combined_prediction_artifact(
    *,
    catalog: Any,
    task_manifest: BenchmarkTaskManifest,
    submitter_result: BenchmarkSubmitterResult,
    score_policy: ManifestEnvelope,
    prediction_artifacts: tuple[ManifestEnvelope, ...],
) -> ManifestEnvelope | None:
    if not prediction_artifacts:
        return None
    first = prediction_artifacts[0].body
    rows = [
        dict(row)
        for artifact in prediction_artifacts
        for row in artifact.body.get("rows", ())
    ]
    if not rows:
        return None
    scored_origins = [
        dict(origin)
        for artifact in prediction_artifacts
        for origin in artifact.body.get("scored_origin_panel", ())
    ]
    body = dict(first)
    body.update(
        {
            "prediction_artifact_id": (
                f"{task_manifest.task_id}__{submitter_result.submitter_id}"
                "__benchmark_metrics"
            ),
            "candidate_id": submitter_result.selected_candidate_id,
            "stage_id": "outer_test",
            "fit_window_id": "benchmark_metric_aggregate",
            "test_window_id": "benchmark_metric_development_segments",
            "score_policy_ref": score_policy.ref.as_dict(),
            "rows": rows,
            "scored_origin_panel": scored_origins,
            "scored_origin_set_id": sha256_digest(scored_origins),
            "missing_scored_origins": [
                dict(item)
                for artifact in prediction_artifacts
                for item in artifact.body.get("missing_scored_origins", ())
            ],
            "timeguard_checks": [
                dict(item)
                for artifact in prediction_artifacts
                for item in artifact.body.get("timeguard_checks", ())
            ],
            "comparison_key": {
                **dict(first.get("comparison_key", {})),
                "scored_origin_set_id": sha256_digest(scored_origins),
            },
        }
    )
    return ManifestEnvelope.build(
        schema_name=prediction_artifacts[0].schema_name,
        module_id=prediction_artifacts[0].module_id,
        body=body,
        catalog=catalog,
    )


def _calibration_threshold_metric_values(
    calibration_body: Mapping[str, Any],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for diagnostic in calibration_body.get("diagnostics", ()):
        if not isinstance(diagnostic, Mapping):
            continue
        for metric_id in (
            "max_ks_distance",
            "max_reliability_gap",
        ):
            value = diagnostic.get(metric_id)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[metric_id] = _stable_metric_float(float(value))
        absolute_gap = diagnostic.get("absolute_gap")
        if isinstance(absolute_gap, (int, float)) and not isinstance(
            absolute_gap,
            bool,
        ):
            metrics["max_abs_coverage_gap"] = _stable_metric_float(
                float(absolute_gap)
            )
        max_abs_gap = diagnostic.get("max_abs_gap")
        if isinstance(max_abs_gap, (int, float)) and not isinstance(
            max_abs_gap,
            bool,
        ):
            metrics["max_abs_hit_balance_gap"] = _stable_metric_float(
                float(max_abs_gap)
            )
    return metrics


def _benchmark_metric_segments(context: BenchmarkHarnessContext):
    return context.evaluation_plan.development_segments or (
        context.evaluation_plan.confirmatory_segment,
    )


def _segment_scored_origins(
    *,
    context: BenchmarkHarnessContext,
    segment_id: str,
):
    return tuple(
        origin
        for origin in context.evaluation_plan.scored_origin_panel
        if origin.segment_id == segment_id
    )


def _benchmark_rows_by_entity(feature_view) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in feature_view.rows:
        entity = str(row.get("entity", feature_view.series_id))
        grouped.setdefault(entity, []).append(row)
    return {entity: tuple(rows) for entity, rows in grouped.items()}


def _numeric_policy_values(policy: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in policy.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _equal_weight_simplex(horizon_set: tuple[int, ...]) -> list[dict[str, str | int]]:
    if not horizon_set:
        return [{"horizon": 1, "weight": "1.0"}]
    scale = 10**12
    base_units = scale // len(horizon_set)

    def _format_weight(units: int) -> str:
        integer, fractional = divmod(units, scale)
        if fractional == 0:
            return f"{integer}.0"
        return f"{integer}.{fractional:012d}".rstrip("0")

    return [
        {
            "horizon": horizon,
            "weight": _format_weight(
                base_units
                if index < len(horizon_set) - 1
                else scale - (base_units * (len(horizon_set) - 1))
            ),
        }
        for index, horizon in enumerate(horizon_set)
    ]


def _stable_metric_float(value: float) -> float:
    if not math.isfinite(value):
        return value
    return float(f"{value:.12g}")


def _task_replay_verification_status(result: ProfiledBenchmarkTaskResult) -> str:
    replay_ref_paths = tuple(result.report_paths.replay_ref_paths.values())
    if not replay_ref_paths or not all(path.is_file() for path in replay_ref_paths):
        return "missing"
    replay_ref_status = _replay_ref_files_verification_status(result)
    if replay_ref_status is not None and replay_ref_status != "verified":
        return replay_ref_status
    bundle_status = _task_reproducibility_bundle_replay_status(result)
    if bundle_status is not None:
        return bundle_status
    return replay_ref_status or "unverified"


def _task_replay_reason_codes(result: ProfiledBenchmarkTaskResult) -> list[str]:
    status = _task_replay_verification_status(result)
    if status == "verified":
        return []
    if status == "missing":
        return ["missing_replay_artifact"]
    return ["unverified_replay_artifact"]


def _replay_ref_files_verification_status(
    result: ProfiledBenchmarkTaskResult,
) -> str | None:
    statuses: list[str] = []
    missing_status = False
    for path in result.report_paths.replay_ref_paths.values():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing_status = True
            continue
        status = _replay_status_from_mapping(payload)
        if status is None:
            missing_status = True
            continue
        if status == "verified" and _replay_ref_cross_check_failure(
            replay_payload=payload,
            replay_path=path,
            result=result,
        ):
            statuses.append("failed")
            continue
        statuses.append(status)
    if any(status == "missing" for status in statuses):
        return "missing"
    non_verified = tuple(status for status in statuses if status != "verified")
    if non_verified:
        return non_verified[0]
    if missing_status:
        return "unverified"
    if statuses:
        return "verified"
    return None


def _replay_ref_cross_check_failure(
    *,
    replay_payload: Mapping[str, Any],
    replay_path: Path,
    result: ProfiledBenchmarkTaskResult,
) -> str | None:
    submitter_id = replay_payload.get("submitter_id")
    if not isinstance(submitter_id, str):
        return "missing_submitter_id"
    submitter_path = result.report_paths.submitter_result_paths.get(submitter_id)
    if not isinstance(submitter_path, Path) or not submitter_path.is_file():
        return "missing_submitter_artifact"
    try:
        submitter_payload = json.loads(submitter_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid_submitter_artifact"
    if not isinstance(submitter_payload, Mapping):
        return "invalid_submitter_artifact"
    replay_contract = replay_payload.get("replay_contract")
    if not isinstance(replay_contract, Mapping):
        return "missing_replay_contract"
    if replay_contract.get("replay_verification_status") != "verified":
        return "replay_contract_not_verified"
    submitter_replay_contract = submitter_payload.get("replay_contract")
    if not isinstance(submitter_replay_contract, Mapping):
        return "missing_submitter_replay_contract"
    if _stable_json_payload(submitter_replay_contract) != _stable_json_payload(
        replay_contract
    ):
        return "replay_contract_mismatch"
    submitter_replay_ref = submitter_payload.get("replay_ref")
    if not isinstance(submitter_replay_ref, Mapping):
        return "missing_submitter_replay_ref"
    expected_replay_digest = submitter_replay_ref.get("sha256")
    if not isinstance(expected_replay_digest, str) or not expected_replay_digest:
        return "missing_submitter_replay_ref_digest"
    if expected_replay_digest != _sha256_file(replay_path):
        return "submitter_replay_ref_digest_mismatch"

    selected_candidate_id = submitter_payload.get("selected_candidate_id")
    selected_candidate_hash = submitter_payload.get("selected_candidate_hash")
    if selected_candidate_id is None and selected_candidate_hash is None:
        if (
            replay_contract.get("candidate_id") is not None
            or replay_contract.get("candidate_hash") is not None
        ):
            return "abstention_replay_candidate_mismatch"
        return None
    contract_candidate_id = replay_contract.get(
        "candidate_id",
        replay_contract.get("selected_candidate_id"),
    )
    contract_candidate_hash = replay_contract.get(
        "candidate_hash",
        replay_contract.get("selected_candidate_hash"),
    )
    if contract_candidate_id != selected_candidate_id:
        return "candidate_id_mismatch"
    if contract_candidate_hash != selected_candidate_hash:
        return "candidate_hash_mismatch"
    if "candidate_id" in replay_contract or "candidate_hash" in replay_contract:
        if not _submitter_ledger_contains_selected_candidate(
            submitter_payload=submitter_payload,
            selected_candidate_id=selected_candidate_id,
            selected_candidate_hash=selected_candidate_hash,
        ):
            return "candidate_ledger_mismatch"
        replay_hooks = replay_contract.get("replay_hooks")
        if not isinstance(replay_hooks, list) or not replay_hooks:
            return "missing_replay_hooks"
    return None


def _stable_json_payload(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _submitter_ledger_contains_selected_candidate(
    *,
    submitter_payload: Mapping[str, Any],
    selected_candidate_id: Any,
    selected_candidate_hash: Any,
) -> bool:
    candidate_ledger = submitter_payload.get("candidate_ledger")
    if not isinstance(candidate_ledger, Sequence) or isinstance(
        candidate_ledger, (str, bytes)
    ):
        return False
    for entry in candidate_ledger:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("ledger_status") != "accepted":
            continue
        if entry.get("candidate_id") != selected_candidate_id:
            continue
        if entry.get("candidate_hash") != selected_candidate_hash:
            continue
        return True
    return False


def _task_reproducibility_bundle_replay_status(
    result: ProfiledBenchmarkTaskResult,
) -> str | None:
    payload = _load_task_result_payload(result)
    if payload is None:
        return None
    for field_name in (
        "reproducibility_bundle_manifest",
        "reproducibility_bundle",
        "reproducibility_bundle_payload",
    ):
        status = _replay_status_from_mapping(payload.get(field_name))
        if status is not None:
            return status
    return _replay_status_from_mapping(payload.get("track_summary"))


def _load_task_result_payload(
    result: ProfiledBenchmarkTaskResult,
) -> dict[str, Any] | None:
    task_result_path = getattr(result.report_paths, "task_result_path", None)
    if not isinstance(task_result_path, Path) or not task_result_path.is_file():
        return None
    try:
        payload = json.loads(task_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _replay_status_from_mapping(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    body_status = _replay_status_from_mapping(value.get("body"))
    if body_status is not None:
        return body_status
    status = value.get("replay_verification_status")
    if isinstance(status, str) and status.strip():
        return status.strip()
    return None


def _aggregate_replay_verification_status(statuses: Iterable[str]) -> str:
    values = tuple(statuses)
    if not values or any(status == "missing" for status in values):
        return "missing"
    if all(status == "verified" for status in values):
        return "verified"
    non_verified = tuple(status for status in values if status != "verified")
    if len(set(non_verified)) == 1:
        return non_verified[0]
    return "unverified"


def _replay_reason_codes(statuses: Iterable[str]) -> tuple[str, ...]:
    values = tuple(statuses)
    reason_codes: list[str] = []
    if not values or any(status == "missing" for status in values):
        reason_codes.append("replay_artifact_missing")
    if any(status not in {"missing", "verified"} for status in values):
        reason_codes.append("replay_unverified")
    return tuple(reason_codes)


def _cache_signature(
    *,
    task_manifest: BenchmarkTaskManifest,
    dataset_path: Path,
    suffix: str,
    project_root: Path | None = None,
) -> str:
    return sha256_digest(
        {
            "cache_version": _BENCHMARK_RUNTIME_CACHE_VERSION,
            "suffix": suffix,
            "task_manifest": _path_signature(task_manifest.source_path),
            "dataset": _path_signature(dataset_path),
            "runtime_sources": _runtime_source_signatures(project_root=project_root),
            "semantic_controls": {
                "task_family": task_manifest.task_family,
                "search_class": task_manifest.search_class,
                "search_class_honesty": task_manifest.search_class_honesty,
                "metric_thresholds": task_manifest.metric_thresholds,
                "expected_claim_ceiling": task_manifest.expected_claim_ceiling,
                "false_claim_expectations": task_manifest.false_claim_expectations,
                "abstention_policy": task_manifest.abstention_policy,
            },
        }
    )


def _submitter_cache_signature(
    *,
    context: BenchmarkHarnessContext,
    submitter_id: str,
) -> str:
    return sha256_digest(
        {
            "cache_version": _BENCHMARK_RUNTIME_CACHE_VERSION,
            "submitter_id": submitter_id,
            "task_id": context.task_manifest.task_id,
            "track_id": context.task_manifest.track_id,
            "protocol_contract": context.protocol_contract,
            "search_class": context.search_class,
            "seasonal_period": context.seasonal_period,
            "task_manifest": _path_signature(context.task_manifest.source_path),
            "proposal_specs": [
                _proposal_spec_signature(proposal)
                for proposal in context.proposal_specs
            ],
            "context_data": _benchmark_context_data_signature(context),
            "runtime_sources": _runtime_source_signatures(
                project_root=context.project_root
            ),
        }
    )


def _benchmark_context_data_signature(context: BenchmarkHarnessContext) -> str:
    snapshot = context.snapshot
    feature_view = context.feature_view
    evaluation_plan = context.evaluation_plan
    materialization_hashes = getattr(snapshot, "materialization_hashes", None)
    materialization_report = getattr(feature_view, "materialization_report", None)
    evaluation_plan_payload = (
        evaluation_plan.as_dict()
        if hasattr(evaluation_plan, "as_dict")
        else repr(evaluation_plan)
    )
    return sha256_digest(
        {
            "snapshot": {
                "series_id": getattr(snapshot, "series_id", None),
                "cutoff_available_at": getattr(snapshot, "cutoff_available_at", None),
                "revision_policy": getattr(snapshot, "revision_policy", None),
                "row_count": getattr(snapshot, "row_count", None),
                "entity_panel": list(getattr(snapshot, "entity_panel", ())),
                "lineage_payload_hashes": list(
                    getattr(snapshot, "lineage_payload_hashes", ())
                ),
                "materialization_hashes": (
                    {
                        "raw_observation_hash": getattr(
                            materialization_hashes,
                            "raw_observation_hash",
                            None,
                        ),
                        "coded_target_hash": getattr(
                            materialization_hashes,
                            "coded_target_hash",
                            None,
                        ),
                        "lineage_payload_hash": getattr(
                            materialization_hashes,
                            "lineage_payload_hash",
                            None,
                        ),
                    }
                    if materialization_hashes is not None
                    else None
                ),
            },
            "feature_view": {
                "series_id": getattr(feature_view, "series_id", None),
                "feature_names": list(getattr(feature_view, "feature_names", ())),
                "entity_panel": list(getattr(feature_view, "entity_panel", ())),
                "row_count": len(tuple(getattr(feature_view, "rows", ()))),
                "rows_sha256": sha256_digest(
                    list(getattr(feature_view, "rows", ()))
                ),
                "materialization_report": (
                    materialization_report.as_dict()
                    if hasattr(materialization_report, "as_dict")
                    else None
                ),
            },
            "evaluation_plan": evaluation_plan_payload,
        }
    )


def _runtime_source_signatures(
    *,
    project_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    package_root = _runtime_source_package_root(project_root)
    source_paths = (
        package_root / "benchmarks" / "runtime.py",
        package_root / "benchmarks" / "manifests.py",
        package_root / "benchmarks" / "reporting.py",
        package_root / "benchmarks" / "submitters.py",
        package_root / "adapters" / "algorithmic_dsl.py",
        package_root / "adapters" / "portfolio.py",
        package_root / "algorithmic_dsl.py",
        package_root / "cir" / "models.py",
        package_root / "cir" / "normalize.py",
        package_root / "manifests" / "runtime_models.py",
        package_root / "math" / "codelength.py",
        package_root / "math" / "observation_models.py",
        package_root / "math" / "quantization.py",
        package_root / "math" / "reference_descriptions.py",
        package_root / "modules" / "calibration.py",
        package_root / "modules" / "candidate_fitting.py",
        package_root / "modules" / "features.py",
        package_root / "modules" / "forecast_paths.py",
        package_root / "modules" / "probabilistic_evaluation.py",
        package_root / "modules" / "replay.py",
        package_root / "modules" / "scoring.py",
        package_root / "modules" / "search_planning.py",
        package_root / "modules" / "snapshotting.py",
        package_root / "modules" / "split_planning.py",
        package_root / "operator_runtime" / "intake_planning.py",
        package_root / "performance.py",
        package_root / "reducers" / "composition.py",
        package_root / "reducers" / "models.py",
        package_root / "search" / "backends.py",
        package_root / "search" / "descriptive_coding.py",
        package_root / "search" / "frontier.py",
        package_root / "search" / "policies.py",
        package_root / "search" / "portfolio.py",
        package_root / "stochastic" / "event_definitions.py",
    )
    return {
        path.relative_to(package_root).as_posix(): _path_signature(path)
        for path in source_paths
    }


def _runtime_source_package_root(project_root: Path | None) -> Path:
    if project_root is not None:
        resolved_project_root = project_root.resolve()
        if (
            resolved_project_root.name == "_assets"
            and resolved_project_root.parent.name == "euclid"
            and (resolved_project_root.parent / "benchmarks" / "runtime.py").exists()
        ):
            return resolved_project_root.parent
        checkout_package_root = (resolved_project_root / "src" / "euclid").resolve()
        if checkout_package_root.exists():
            return checkout_package_root
        direct_package_root = (resolved_project_root / "euclid").resolve()
        if direct_package_root.exists():
            return direct_package_root

    module_package_root = Path(__file__).resolve().parents[1]
    if (module_package_root / "benchmarks" / "submitters.py").exists():
        return module_package_root

    import euclid._version as version_module

    return Path(version_module.__file__).resolve().parent


def _proposal_spec_signature(proposal: DescriptiveSearchProposal) -> dict[str, Any]:
    return {
        "candidate_id": proposal.candidate_id,
        "primitive_family": proposal.primitive_family,
        "form_class": proposal.form_class,
        "feature_dependencies": list(proposal.feature_dependencies),
        "parameter_values": dict(proposal.parameter_values),
        "literal_values": dict(proposal.literal_values),
        "persistent_state": dict(proposal.persistent_state),
        "composition_payload": (
            dict(proposal.composition_payload)
            if proposal.composition_payload is not None
            else None
        ),
        "history_access_mode": proposal.history_access_mode,
        "max_lag": proposal.max_lag,
        "required_observation_model_family": (
            proposal.required_observation_model_family
        ),
    }


def _path_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_pickle_cache(path: Path, *, expected_signature: str) -> Any | None:
    if _load_pickle_cache_signature(path) != expected_signature:
        return None
    try:
        payload = pickle.loads(path.read_bytes())
    except (AttributeError, EOFError, FileNotFoundError, pickle.PickleError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("signature") != expected_signature:
        return None
    return payload.get("payload")


def _pickle_cache_signature_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.signature.json")


def _load_pickle_cache_signature(path: Path) -> str | None:
    try:
        payload = json.loads(
            _pickle_cache_signature_path(path).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    signature = payload.get("signature")
    return signature if isinstance(signature, str) else None


def _write_pickle_cache(
    path: Path,
    *,
    signature: str,
    payload: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_payload = pickle.dumps(
            {"signature": signature, "payload": payload},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    except (AttributeError, pickle.PickleError, TypeError):
        return
    path.write_bytes(cache_payload)
    _pickle_cache_signature_path(path).write_text(
        json.dumps(
            {
                "cache_format": "pickle_payload_v1",
                "signature": signature,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _dataset_path(
    task_manifest: BenchmarkTaskManifest,
    *,
    project_root: Path,
) -> Path:
    dataset_ref = Path(task_manifest.dataset_ref)
    candidate = (project_root / dataset_ref).resolve()
    if candidate.exists():
        return candidate

    packaged_prefix = Path("src") / "euclid" / "_assets"
    try:
        relative_to_asset_root = dataset_ref.relative_to(packaged_prefix)
    except ValueError:
        return candidate

    packaged_candidate = (project_root / relative_to_asset_root).resolve()
    if packaged_candidate.exists():
        return packaged_candidate
    return candidate


def _resolve_project_root(project_root: Path | str | None) -> Path:
    from euclid.operator_runtime.resources import resolve_asset_root

    return resolve_asset_root(project_root)


def _search_class(task_manifest: BenchmarkTaskManifest) -> str:
    if task_manifest.search_class is not None:
        return task_manifest.search_class
    if task_manifest.task_family == "algorithmic_symbolic_regression":
        return "exact_finite_enumeration"
    return (
        "stochastic_heuristic"
        if int(task_manifest.frozen_protocol.seed_policy.get("restarts", 0)) > 0
        else "bounded_heuristic"
    )


def _resolve_horizon(task_manifest: BenchmarkTaskManifest) -> int:
    horizon_policy = getattr(task_manifest, "horizon_policy", None)
    if isinstance(horizon_policy, dict):
        horizons = horizon_policy.get("horizons")
        if isinstance(horizons, list) and horizons:
            return max(int(item) for item in horizons)
    return 1


def _resolve_quantization_step(task_manifest: BenchmarkTaskManifest) -> str:
    lattice = str(task_manifest.frozen_protocol.quantization_policy.get("lattice", ""))
    if lattice.startswith("decimal_"):
        token = lattice.removeprefix("decimal_")
        try:
            return format(float(token), ".12g")
        except ValueError:
            return "0.000001"
    return "0.000001"


def _minimum_description_gain_bits(task_manifest: BenchmarkTaskManifest) -> float:
    if task_manifest.task_family.startswith("shared_local_panel_"):
        return -5.0
    return 0.0


def _benchmark_proposal_specs(
    *,
    task_manifest: BenchmarkTaskManifest,
    entity_panel: tuple[str, ...],
) -> tuple[DescriptiveSearchProposal, ...]:
    if task_manifest.task_family.startswith("shared_local_panel_"):
        return _shared_local_benchmark_proposals(
            task_manifest=task_manifest,
            entity_panel=entity_panel,
        )
    if task_manifest.task_family.startswith("search_class_"):
        return _declared_search_class_benchmark_proposals(
            task_manifest=task_manifest,
        )
    if task_manifest.task_family == "composition_piecewise_surface":
        return (_piecewise_benchmark_proposal(),)
    if task_manifest.task_family == "composition_additive_residual_surface":
        return (_additive_residual_benchmark_proposal(),)
    if task_manifest.task_family == "composition_regime_conditioned_surface":
        return (_regime_conditioned_benchmark_proposal(),)
    return ()


def _declared_search_class_benchmark_proposals(
    *,
    task_manifest: BenchmarkTaskManifest,
) -> tuple[DescriptiveSearchProposal, ...]:
    declared_programs = task_manifest.search_class_honesty.get(
        "declared_candidate_programs",
    )
    if declared_programs is None:
        return ()
    if not isinstance(declared_programs, list):
        raise ContractValidationError(
            code="invalid_benchmark_manifest_field",
            message=(
                "search_class_honesty.declared_candidate_programs must be a list"
            ),
            field_path="search_class_honesty.declared_candidate_programs",
        )

    proposals: list[DescriptiveSearchProposal] = []
    for index, program_spec in enumerate(declared_programs):
        if not isinstance(program_spec, Mapping):
            raise ContractValidationError(
                code="invalid_benchmark_manifest_field",
                message="declared candidate program entries must be mappings",
                field_path=(
                    "search_class_honesty.declared_candidate_programs"
                    f"[{index}]"
                ),
            )
        candidate_id = program_spec.get("candidate_id")
        program_source = program_spec.get("algorithmic_program")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ContractValidationError(
                code="invalid_benchmark_manifest_field",
                message="declared candidate programs require candidate_id",
                field_path=(
                    "search_class_honesty.declared_candidate_programs"
                    f"[{index}].candidate_id"
                ),
            )
        if not isinstance(program_source, str) or not program_source:
            raise ContractValidationError(
                code="invalid_benchmark_manifest_field",
                message="declared candidate programs require algorithmic_program",
                field_path=(
                    "search_class_honesty.declared_candidate_programs"
                    f"[{index}].algorithmic_program"
                ),
            )
        max_lag = int(program_spec.get("max_lag", 0))
        history_access_mode = str(
            program_spec.get(
                "history_access_mode",
                "causal_current_observation"
                if max_lag == 0
                else "bounded_lag_window",
            )
        )
        proposals.append(
            DescriptiveSearchProposal(
                candidate_id=candidate_id,
                primitive_family="algorithmic",
                form_class="bounded_program",
                literal_values={
                    "algorithmic_program": program_source,
                    "algorithmic_state_slot_count": int(
                        program_spec.get("algorithmic_state_slot_count", 1)
                    ),
                    "program_node_count": int(
                        program_spec.get("program_node_count", 8)
                    ),
                    "declaration_source": "benchmark_manifest",
                },
                history_access_mode=history_access_mode,
                max_lag=max_lag,
            )
        )
    return tuple(proposals)


def _shared_local_benchmark_proposals(
    *,
    task_manifest: BenchmarkTaskManifest,
    entity_panel: tuple[str, ...],
) -> tuple[DescriptiveSearchProposal, ...]:
    if len(entity_panel) < 2:
        raise ContractValidationError(
            code="invalid_benchmark_manifest_field",
            message=(
                "shared-local benchmark tasks require a declared multi-entity panel"
            ),
            field_path="task_family",
            details={
                "task_family": task_manifest.task_family,
                "entity_panel": list(entity_panel),
            },
        )

    composition_payload = {
        "operator_id": "shared_plus_local_decomposition",
        "entity_index_set": list(entity_panel),
        "shared_component_ref": "shared_component",
        "local_component_refs": [
            f"local_component_{index}" for index, _ in enumerate(entity_panel, start=1)
        ],
        "sharing_map": ["intercept"],
        "unseen_entity_rule": "panel_entities_only",
    }
    if task_manifest.task_family == "shared_local_panel_negative_case":
        return (
            DescriptiveSearchProposal(
                candidate_id="shared_local_panel_false_generalization",
                primitive_family="analytic",
                form_class="closed_form_expression",
                feature_dependencies=("missing_panel_feature",),
                parameter_values={"shared_intercept": 0.0},
                composition_payload=composition_payload,
            ),
        )
    if task_manifest.task_family == "shared_local_panel_abstention_case":
        return ()
    return (
        DescriptiveSearchProposal(
            candidate_id="shared_local_panel_joint_optimizer",
            primitive_family="analytic",
            form_class="closed_form_expression",
            parameter_values={"shared_intercept": 0.0},
            composition_payload=composition_payload,
        ),
    )


def _equality_saturation_benchmark_proposals() -> tuple[DescriptiveSearchProposal, ...]:
    return (
        DescriptiveSearchProposal(
            candidate_id="analytic_piecewise_complex",
            primitive_family="analytic",
            form_class="closed_form_expression",
            feature_dependencies=("lag_1",),
            parameter_values={"intercept": 14.0},
            literal_values={"upper_cut": 3.0, "lower_cut": 1.0},
            persistent_state={"step_count": 1, "running_total": 0.0},
            composition_payload={
                "operator_id": "piecewise",
                "ordered_partition": [
                    {
                        "start_literal": 0.0,
                        "end_literal": 1.0,
                        "reducer_id": "head",
                    },
                    {
                        "start_literal": 1.0,
                        "end_literal": 3.0,
                        "reducer_id": "tail",
                    },
                ],
            },
            history_access_mode="bounded_lag_window",
            max_lag=1,
        ),
        DescriptiveSearchProposal(
            candidate_id="analytic_intercept_simple",
            primitive_family="analytic",
            form_class="closed_form_expression",
            parameter_values={"intercept": 14.0},
        ),
    )


def _stochastic_benchmark_proposals() -> tuple[DescriptiveSearchProposal, ...]:
    return tuple(
        DescriptiveSearchProposal(
            candidate_id=f"analytic_seed_{index}",
            primitive_family="analytic",
            form_class="closed_form_expression",
            parameter_values={"intercept": 10.0 + index},
        )
        for index in range(4)
    )


def _piecewise_benchmark_proposal() -> DescriptiveSearchProposal:
    return DescriptiveSearchProposal(
        candidate_id="analytic_piecewise_surface",
        primitive_family="analytic",
        form_class="closed_form_expression",
        feature_dependencies=("lag_1",),
        parameter_values={
            "intercept__head": 20.0,
            "lag_coefficient__head": 0.6,
            "intercept__tail": 5.0,
            "lag_coefficient__tail": 1.1,
        },
        literal_values={"upper_cut": 3.0, "lower_cut": 1.0},
        persistent_state={"step_count": 1, "running_total": 0.0},
        composition_payload={
            "operator_id": "piecewise",
            "ordered_partition": [
                {"start_literal": 0.0, "end_literal": 1.0, "reducer_id": "head"},
                {"start_literal": 1.0, "end_literal": 3.0, "reducer_id": "tail"},
            ],
        },
        history_access_mode="bounded_lag_window",
        max_lag=1,
    )


def _additive_residual_benchmark_proposal() -> DescriptiveSearchProposal:
    return DescriptiveSearchProposal(
        candidate_id="analytic_additive_residual_surface",
        primitive_family="analytic",
        form_class="closed_form_expression",
        feature_dependencies=("lag_1",),
        parameter_values={
            "intercept__trend_component": 0.0,
            "lag_coefficient__trend_component": 1.0,
            "intercept__seasonal_component": 0.0,
            "lag_coefficient__seasonal_component": 1.0,
        },
        literal_values={
            "lookup_residual_wrapper_ref": "benchmark_declared_residual_lag_1"
        },
        composition_payload={
            "operator_id": "additive_residual",
            "base_reducer": "trend_component",
            "residual_reducer": "seasonal_component",
            "shared_observation_model": "point_identity",
        },
        history_access_mode="bounded_lag_window",
        max_lag=1,
    )


def _regime_conditioned_benchmark_proposal() -> DescriptiveSearchProposal:
    return DescriptiveSearchProposal(
        candidate_id="analytic_regime_conditioned_surface",
        primitive_family="analytic",
        form_class="closed_form_expression",
        feature_dependencies=("lag_1",),
        parameter_values={
            "intercept__stable_branch": 3.0,
            "lag_coefficient__stable_branch": 1.0,
            "intercept__volatile_branch": 8.0,
            "lag_coefficient__volatile_branch": 1.0,
        },
        composition_payload={
            "operator_id": "regime_conditioned",
            "gating_law": {
                "gating_law_id": "regime_flag_gate",
                "selection_mode": "hard_switch",
            },
            "regime_information_contract": ["regime_flag"],
            "branch_reducers": [
                {"regime_value": "stable", "reducer_id": "stable_branch"},
                {"regime_value": "volatile", "reducer_id": "volatile_branch"},
            ],
        },
        history_access_mode="bounded_lag_window",
        max_lag=1,
    )


def _attach_snapshot_side_information(
    *,
    feature_view,
    snapshot,
):
    side_information_by_coordinate = {
        (row.entity, row.event_time, row.available_at): dict(row.side_information)
        for row in snapshot.raw_rows
        if row.target is not None
    }
    rows = []
    for row in feature_view.rows:
        entity = str(row.get("entity", feature_view.series_id))
        coordinate = (
            entity,
            str(row["event_time"]),
            str(row["available_at"]),
        )
        side_information = side_information_by_coordinate.get(coordinate, {})
        augmented_row = dict(row)
        for field_name, value in side_information.items():
            augmented_row.setdefault(field_name, value)
        rows.append(augmented_row)
    return replace(feature_view, rows=tuple(rows))


__all__ = [
    "BenchmarkSuiteSurfaceStatus",
    "ProfiledBenchmarkSuiteResult",
    "ProfiledBenchmarkTaskResult",
    "profile_benchmark_suite",
    "profile_benchmark_task",
]
