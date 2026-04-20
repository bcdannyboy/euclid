from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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
    build_benchmark_task_track_summary,
    write_benchmark_task_report_artifacts,
)
from euclid.benchmarks.submitters import (
    ALGORITHMIC_SEARCH_SUBMITTER_ID,
    BenchmarkHarnessContext,
    BenchmarkSubmitterResult,
    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
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
from euclid.manifest_registry import ManifestRegistry
from euclid.operator_runtime.intake_planning import build_operator_intake_plan
from euclid.performance import (
    PerformanceTelemetryArtifact,
    TelemetryRecorder,
    write_performance_telemetry,
)
from euclid.runtime.hashing import sha256_digest
from euclid.search.backends import DescriptiveSearchProposal

_BENCHMARK_RUNTIME_CACHE_VERSION = "1.0.0"


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
    benchmark_passed = len(matched) == len(requirement.task_ids) and all(
        result.report_paths.task_result_path.is_file() for result in matched
    )
    if not requirement.replay_required:
        replay_passed = True
    elif requirement.surface_id == "portfolio_orchestration":
        replay_passed = len(matched) == len(requirement.task_ids) and all(
            result.report_paths.portfolio_selection_record_path is not None
            and result.report_paths.portfolio_selection_record_path.is_file()
            for result in matched
        )
    else:
        replay_passed = len(matched) == len(requirement.task_ids) and all(
            all(
                path.is_file() for path in result.report_paths.replay_ref_paths.values()
            )
            for result in matched
        )
    replay_verification_status = "verified" if replay_passed else "missing"
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
    }


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
        all(path.is_file() for path in result.report_paths.replay_ref_paths.values())
        for result in task_results
    )


def _task_replay_verification_status(result: ProfiledBenchmarkTaskResult) -> str:
    if all(path.is_file() for path in result.report_paths.replay_ref_paths.values()):
        return "verified"
    return "missing"


def _cache_signature(
    *,
    task_manifest: BenchmarkTaskManifest,
    dataset_path: Path,
    suffix: str,
) -> str:
    return sha256_digest(
        {
            "cache_version": _BENCHMARK_RUNTIME_CACHE_VERSION,
            "suffix": suffix,
            "task_manifest": _path_signature(task_manifest.source_path),
            "dataset": _path_signature(dataset_path),
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
        }
    )


def _path_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_pickle_cache(path: Path, *, expected_signature: str) -> Any | None:
    try:
        payload = pickle.loads(path.read_bytes())
    except (FileNotFoundError, EOFError, pickle.PickleError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("signature") != expected_signature:
        return None
    return payload.get("payload")


def _write_pickle_cache(
    path: Path,
    *,
    signature: str,
    payload: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        pickle.dumps(
            {"signature": signature, "payload": payload},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )


def _dataset_path(
    task_manifest: BenchmarkTaskManifest,
    *,
    project_root: Path,
) -> Path:
    return (project_root / task_manifest.dataset_ref).resolve()


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
    if task_manifest.task_family == "search_class_equality_saturation_surface":
        return _equality_saturation_benchmark_proposals()
    if task_manifest.task_family == "search_class_stochastic_surface":
        return _stochastic_benchmark_proposals()
    if task_manifest.task_family == "composition_piecewise_surface":
        return (_piecewise_benchmark_proposal(),)
    if task_manifest.task_family == "composition_additive_residual_surface":
        return (_additive_residual_benchmark_proposal(),)
    if task_manifest.task_family == "composition_regime_conditioned_surface":
        return (_regime_conditioned_benchmark_proposal(),)
    return ()


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
        parameter_values={"intercept": 14.0},
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
        parameter_values={"intercept": 14.0},
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
        parameter_values={"intercept": 14.0},
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
