from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from euclid._cli_compat import patch_typer_click_metavar_compatibility
from euclid.benchmarks import profile_benchmark_task
from euclid.bootstrap import smoke_summary
from euclid.cli.benchmarks import benchmarks_app
from euclid.cli.replay import register as register_replay_command
from euclid.cli.run import register as register_run_command
from euclid.cli.workbench import register as register_workbench_command
from euclid.demo import (
    default_algorithmic_demo_manifest_path,
    default_demo_manifest_path,
    default_probabilistic_demo_manifest_path,
    format_algorithmic_search_run_summary,
    format_point_evaluation_run_summary,
    format_probabilistic_evaluation_run_summary,
    format_replay_summary,
    format_run_summary,
    profile_demo_run,
    replay_demo,
    run_demo,
    run_demo_algorithmic_search,
    run_demo_point_evaluation,
    run_demo_probabilistic_evaluation,
)
from euclid.inspection import (
    compare_demo_baseline,
    format_baseline_comparison,
    format_calibration_inspection,
    format_demo_artifact_graph,
    format_demo_catalog_entry,
    format_demo_lineage_graph,
    format_demo_publication_catalog,
    format_demo_store_validation,
    format_point_prediction_inspection,
    format_probabilistic_prediction_inspection,
    format_replay_bundle_inspection,
    format_resolved_artifact,
    inspect_demo_calibration,
    inspect_demo_catalog_entry,
    inspect_demo_point_prediction,
    inspect_demo_probabilistic_prediction,
    inspect_demo_replay_bundle,
    load_demo_publication_catalog,
    load_demo_run_artifact_graph,
    publish_demo_run_to_catalog,
    resolve_demo_artifact,
    validate_demo_store,
)
from euclid.readiness import gate_results_by_id
from euclid.release import (
    certify_research_readiness,
    execute_release_notebook_smoke,
    get_release_status,
    run_repo_test_matrix,
    run_clean_install_certification,
    run_release_benchmark_smoke,
    run_release_determinism_smoke,
    run_release_performance_smoke,
    validate_release_contracts,
    verify_completion_report,
)

patch_typer_click_metavar_compatibility()

app = typer.Typer(
    add_completion=False,
    help=(
        "CLI for Euclid current_release operations and full_vision "
        "certification surfaces. Demo/prototype commands are compatibility-only."
    ),
    no_args_is_help=True,
)
demo_app = typer.Typer(
    add_completion=False,
    help="Compatibility-only demo and prototype commands. Not a certified runtime.",
)
point_app = typer.Typer(
    add_completion=False,
    help="Compatibility-only point-evaluation surfaces for retained demo assets.",
)
probabilistic_app = typer.Typer(
    add_completion=False,
    help=(
        "Compatibility-only probabilistic evaluation surfaces for retained "
        "Phase 06 demo assets."
    ),
)
calibration_app = typer.Typer(
    add_completion=False,
    help="Inspect compatibility-only calibration artifacts for demo runs.",
)
catalog_app = typer.Typer(
    add_completion=False,
    help="Publish compatibility-only demo runs into a local catalog view.",
)
algorithmic_app = typer.Typer(
    add_completion=False,
    help="Run compatibility-only retained Phase 06 algorithmic search demos.",
)
benchmark_app = typer.Typer(
    add_completion=False,
    help="Run benchmark-task profiling surfaces.",
)
release_app = typer.Typer(
    add_completion=False,
    help=(
        "Run schema-driven release, certification, and closure commands for "
        "current_release, full_vision, and shipped_releasable."
    ),
)


@app.callback()
def cli() -> None:
    """Root command group for Euclid bootstrap entrypoints."""


@app.command("smoke")
def smoke() -> None:
    """Print the currently bootstrapped workflow and dependency surface."""
    typer.echo(smoke_summary())


@demo_app.command("run")
def demo_run(
    manifest: Path = typer.Option(
        default_demo_manifest_path(),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the demo request manifest.",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory where the immutable demo artifacts and registry are written.",
    ),
) -> None:
    """Run the local demo from a manifest-backed dataset."""
    typer.echo(
        format_run_summary(run_demo(manifest_path=manifest, output_root=output_root))
    )


@demo_app.command("profile-run")
def demo_profile_run(
    manifest: Path = typer.Option(
        default_demo_manifest_path(),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the demo request manifest.",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory where the profiled demo artifacts are written.",
    ),
) -> None:
    """Run the local demo and emit machine-readable performance telemetry."""
    result = profile_demo_run(manifest_path=manifest, output_root=output_root)
    lines = [
        "Euclid demo profile",
        f"Request id: {result.run.request.request_id}",
        f"Telemetry artifact: {result.telemetry_path}",
        "profile_kind=demo_run",
        f"Selected family: {result.run.summary.selected_family}",
        f"Result mode: {result.run.summary.result_mode}",
    ]
    typer.echo("\n".join(lines))


@demo_app.command("replay")
def demo_replay(
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory containing the prior demo run outputs.",
    ),
    bundle_ref: Optional[str] = typer.Option(
        None,
        help="Optional bundle ref in schema_name:object_id form.",
    ),
) -> None:
    """Replay a previously sealed demo bundle."""
    result = replay_demo(output_root=output_root, bundle_ref=bundle_ref)
    typer.echo(format_replay_summary(result))


@demo_app.command("replay-inspect")
def demo_replay_inspect(
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory containing the prior demo run outputs.",
    ),
    bundle_ref: Optional[str] = typer.Option(
        None,
        help="Optional bundle ref in schema_name:object_id form.",
    ),
) -> None:
    """Inspect the sealed replay bundle for a demo run."""
    typer.echo(
        format_replay_bundle_inspection(
            inspect_demo_replay_bundle(
                output_root=output_root,
                bundle_ref=bundle_ref,
            )
        )
    )


@demo_app.command("inspect")
def demo_inspect(
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory containing the prior demo run outputs.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        help="Optional run_result run_id to inspect instead of the latest summary.",
    ),
) -> None:
    """Inspect the sealed artifact graph for a demo run."""
    graph = load_demo_run_artifact_graph(output_root=output_root, run_id=run_id)
    typer.echo(format_demo_artifact_graph(graph))


@demo_app.command("resolve")
def demo_resolve(
    ref: str = typer.Option(
        ...,
        help="Typed ref in schema_name:object_id form.",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory containing the prior demo run outputs.",
    ),
) -> None:
    """Resolve one sealed manifest and print its lineage and typed-ref edges."""
    artifact = resolve_demo_artifact(output_root=output_root, ref=ref)
    typer.echo(format_resolved_artifact(artifact))


@demo_app.command("lineage")
def demo_lineage(
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory containing the prior demo run outputs.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        help="Optional run_result run_id to inspect instead of the latest summary.",
    ),
) -> None:
    """Print the upstream and downstream lineage around a sealed demo run."""
    graph = load_demo_run_artifact_graph(output_root=output_root, run_id=run_id)
    typer.echo(format_demo_lineage_graph(graph))


@demo_app.command("validate")
def demo_validate(
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory containing the prior demo run outputs.",
    ),
) -> None:
    """Validate sealed artifact integrity and registry edge consistency."""
    report = validate_demo_store(output_root=output_root)
    typer.echo(format_demo_store_validation(report))


@demo_app.command("publish")
def demo_publish(
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory containing the prior demo run outputs.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        help="Optional run_result run_id to publish instead of the latest summary.",
    ),
) -> None:
    """Project a replay-verified sealed demo run into the local catalog view."""
    entry = publish_demo_run_to_catalog(output_root=output_root, run_id=run_id)
    catalog_root = load_demo_publication_catalog(output_root=output_root).catalog_root
    lines = [
        "Euclid demo catalog publication",
        f"Request id: {entry.request_id}",
        f"Publication id: {entry.publication_id}",
        f"Publication mode: {entry.publication_mode}",
        f"Catalog scope: {entry.catalog_scope}",
        f"Published at: {entry.published_at}",
        f"Catalog root: {catalog_root}",
    ]
    typer.echo("\n".join(lines))


@benchmark_app.command("profile-task")
def benchmark_profile_task(
    manifest: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the benchmark task manifest.",
    ),
    benchmark_root: Path = typer.Option(
        ...,
        file_okay=False,
        help="Benchmark root where reports and telemetry are written.",
    ),
    parallel_workers: int = typer.Option(
        1,
        min=1,
        help="Deterministic worker count for submitter execution.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Reuse benchmark checkpoints and cached submitter results when available.",
    ),
) -> None:
    """Run one benchmark task through the retained harness and emit telemetry."""
    result = profile_benchmark_task(
        manifest_path=manifest,
        benchmark_root=benchmark_root,
        parallel_workers=parallel_workers,
        resume=resume,
    )
    lines = [
        "Euclid benchmark profile",
        f"Task id: {result.task_manifest.task_id}",
        f"Track: {result.task_manifest.track_id}",
        f"Telemetry artifact: {result.telemetry_path}",
        f"Report: {result.report_paths.report_path}",
    ]
    typer.echo("\n".join(lines))


@release_app.command("status")
def release_status(
    project_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        exists=True,
        readable=True,
        help=(
            "Workspace root used for report emission. Certified assets resolve "
            "from packaged mirrors by default."
        ),
    ),
) -> None:
    """Print the current package version and release readiness state."""

    def _release_gate_status(gate_id: str) -> str:
        gate = release_gates.get(gate_id)
        return gate.status if gate is not None else "missing"

    status = get_release_status(project_root=project_root)
    current_release = status.policy_judgments["current_release_v1"]
    full_vision = status.policy_judgments["full_vision_v1"]
    shipped_releasable = status.shipped_releasable_judgment
    release_gates = gate_results_by_id(status.readiness_judgment.gate_results)
    release_gate_lines = [
        "Readiness gate statuses:",
        *(
            f"- {gate_id}: {_release_gate_status(gate_id)}"
            for gate_id in (
                "contracts.catalog",
                "notebook.smoke",
                "determinism.same_seed",
                "performance.runtime_smoke",
            )
        ),
    ]
    enhancement_gate_lines = ["Enhancement phase gates:"]
    for phase_gate in status.enhancement_phase_gates:
        status_counts = phase_gate.get("id_gate_status_counts", {})
        if isinstance(status_counts, dict):
            status_summary = ", ".join(
                f"{key}={value}" for key, value in sorted(status_counts.items())
            )
        else:
            status_summary = "status_counts=malformed"
        enhancement_gate_lines.append(
            f"- {phase_gate['phase_id']}: {phase_gate['phase_status']} "
            f"({phase_gate['id_gate_count']} ids; {status_summary})"
        )
    lines = [
        "Euclid release status",
        f"Project root: {status.project_root}",
        f"Current version: {status.current_version}",
        f"Release target: {status.target_version}",
        f"Target ready: {'yes' if status.target_ready else 'no'}",
        (
            "Current release verdict: "
            f"{current_release.final_verdict} (current_release_v1)"
        ),
        "Current release reason codes: "
        + (
            ", ".join(current_release.reason_codes)
            if current_release.reason_codes
            else "none"
        ),
        f"Current release catalog scope: {current_release.catalog_scope}",
        f"Full vision verdict: {full_vision.final_verdict} (full_vision_v1)",
        "Full vision reason codes: "
        + (", ".join(full_vision.reason_codes) if full_vision.reason_codes else "none"),
        f"Full vision catalog scope: {full_vision.catalog_scope}",
        (
            "Shipped or releasable verdict: "
            f"{shipped_releasable.final_verdict} (shipped_releasable_v1)"
        ),
        "Shipped or releasable reason codes: "
        + (
            ", ".join(shipped_releasable.reason_codes)
            if shipped_releasable.reason_codes
            else "none"
        ),
        f"Shipped or releasable catalog scope: {shipped_releasable.catalog_scope}",
        *release_gate_lines,
        *enhancement_gate_lines,
        f"Blocked reason: {status.blocked_reason}",
    ]
    typer.echo("\n".join(lines))


@release_app.command("repo-test-matrix")
def release_repo_test_matrix(
    project_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        exists=True,
        readable=True,
        help="Checkout root used to execute the required certification test matrix.",
    ),
) -> None:
    """Execute the required repo test matrix and emit a machine-readable summary."""
    result = run_repo_test_matrix(project_root=project_root)
    lines = [
        "Euclid repo test matrix",
        f"Project root: {result.project_root}",
        f"Report: {result.report_path}",
        f"Passed: {'yes' if result.passed else 'no'}",
        f"Summary: {result.summary_line}",
    ]
    typer.echo("\n".join(lines))
    if not result.passed:
        raise typer.Exit(code=1)


@release_app.command("validate-contracts")
def release_validate_contracts(
    project_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        exists=True,
        readable=True,
        help="Repository root used to load schemas/contracts.",
    ),
) -> None:
    """Load the contract catalog and print a release-friendly validation summary."""
    result = validate_release_contracts(project_root=project_root)
    lines = [
        "Euclid contract catalog validation",
        f"Project root: {result.project_root}",
        f"Schema count: {result.schema_count}",
        f"Module count: {result.module_count}",
        f"Enum count: {result.enum_count}",
        f"Contract document count: {result.contract_document_count}",
    ]
    typer.echo("\n".join(lines))


@release_app.command("certify-clean-install")
def release_certify_clean_install(
    project_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        exists=True,
        readable=True,
        help="Repository root used to build and certify the wheel install.",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Optional clean-install work directory for certification artifacts.",
    ),
    wheel_dir: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Optional directory where the built certification wheel is written.",
    ),
) -> None:
    """Build the wheel and certify the packaged runtime from an isolated install."""
    result = run_clean_install_certification(
        project_root=project_root,
        output_root=output_root,
        wheel_dir=wheel_dir,
    )
    lines = [
        "Euclid clean-install certification",
        f"Project root: {result.project_root}",
        f"Report: {result.report_path}",
        f"Surface completion: {result.surface_completion:.6f}",
    ]
    lines.extend(
        f"- {surface.surface_id}: {surface.status}" for surface in result.surfaces
    )
    typer.echo("\n".join(lines))
    if any(surface.status != "passed" for surface in result.surfaces):
        raise typer.Exit(code=1)


@release_app.command("benchmark-smoke")
def release_benchmark_smoke(
    project_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        exists=True,
        readable=True,
        help="Repository root used to resolve sample benchmark manifests.",
    ),
    benchmark_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory where benchmark smoke artifacts are written.",
    ),
    parallel_workers: int = typer.Option(
        1,
        min=1,
        help="Deterministic worker count for benchmark smoke tasks.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Reuse cached benchmark smoke artifacts when available.",
    ),
) -> None:
    """Run one representative benchmark task from each Phase 08 track."""
    result = run_release_benchmark_smoke(
        project_root=project_root,
        benchmark_root=benchmark_root,
        parallel_workers=parallel_workers,
        resume=resume,
    )
    lines = [
        "Euclid benchmark smoke",
        f"Project root: {result.project_root}",
        f"Benchmark root: {result.benchmark_root}",
    ]
    lines.extend(
        (
            f"- {case.track_id}/{case.task_id}: "
            f"report={case.report_path} "
            f"telemetry={case.telemetry_path}"
        )
        for case in result.cases
    )
    typer.echo("\n".join(lines))


@release_app.command("determinism-smoke")
def release_determinism_smoke(
    project_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        exists=True,
        readable=True,
        help="Repository root used to resolve the certified operator example.",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory where determinism smoke artifacts are written.",
    ),
) -> None:
    """Run the release determinism smoke against the certified operator example."""
    result = run_release_determinism_smoke(
        project_root=project_root,
        output_root=output_root,
    )
    lines = [
        "Euclid determinism smoke",
        f"Project root: {result.project_root}",
        f"Output root: {result.output_root}",
        f"Summary: {result.summary_path}",
        f"Identical: {'yes' if result.identical else 'no'}",
    ]
    typer.echo("\n".join(lines))
    if not result.identical:
        raise typer.Exit(code=1)


@release_app.command("performance-smoke")
def release_performance_smoke(
    project_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        exists=True,
        readable=True,
        help="Repository root used to resolve the certified runtime surfaces.",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory where performance smoke artifacts are written.",
    ),
) -> None:
    """Run the release performance smoke against operator and benchmark surfaces."""
    result = run_release_performance_smoke(
        project_root=project_root,
        output_root=output_root,
    )
    lines = [
        "Euclid performance smoke",
        f"Project root: {result.project_root}",
        f"Output root: {result.output_root}",
        f"Summary: {result.summary_path}",
        (
            "Operator budget passed: "
            f"{'yes' if result.operator_budget_passed else 'no'}"
        ),
        (
            "Benchmark budget passed: "
            f"{'yes' if result.benchmark_budget_passed else 'no'}"
        ),
        f"Suite budget passed: {'yes' if result.suite_budget_passed else 'no'}",
    ]
    typer.echo("\n".join(lines))
    if not (
        result.operator_budget_passed
        and result.benchmark_budget_passed
        and result.suite_budget_passed
    ):
        raise typer.Exit(code=1)


@release_app.command("notebook-smoke")
def release_notebook_smoke(
    project_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        exists=True,
        readable=True,
        help="Repository root used to resolve the notebook and fixtures.",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory where notebook smoke artifacts are written.",
    ),
    notebook_path: Optional[Path] = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help=(
            "Optional notebook path to execute instead of the default current-release "
            "notebook."
        ),
    ),
) -> None:
    """Execute the checked-in notebook workflow against the public Python API."""
    result = execute_release_notebook_smoke(
        project_root=project_root,
        output_root=output_root,
        notebook_path=notebook_path,
    )
    lines = [
        "Euclid notebook smoke",
        f"Project root: {result.project_root}",
        f"Notebook: {result.notebook_path}",
        f"Summary: {result.summary_path}",
        "Probabilistic cases: "
        + ", ".join(case_id for case_id in result.probabilistic_case_ids),
        f"Catalog entries: {result.catalog_entries}",
        f"Publication mode: {result.publication_mode}",
    ]
    typer.echo("\n".join(lines))


@release_app.command("verify-completion")
def release_verify_completion(
    project_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        exists=True,
        readable=True,
        help="Repository root used to load the regression policy.",
    ),
    report_path: Optional[Path] = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional completion report path to verify.",
    ),
    policy_path: Optional[Path] = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional regression policy path to apply.",
    ),
) -> None:
    """Fail when the completion report regresses below the declared policy floor."""
    result = verify_completion_report(
        project_root=project_root,
        report_path=report_path,
        policy_path=policy_path,
    )
    lines = [
        "Euclid completion verification",
        f"Project root: {result.project_root}",
        f"Report: {result.report_path}",
        f"Policy: {result.policy_path}",
        f"Passed: {'yes' if result.passed else 'no'}",
    ]
    lines.extend(f"- {message}" for message in result.failure_messages)
    typer.echo("\n".join(lines))
    if not result.passed:
        raise typer.Exit(code=1)


@release_app.command("certify-research-readiness")
def release_certify_research_readiness(
    project_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        exists=True,
        readable=True,
        help="Workspace root containing the certification command artifacts.",
    ),
) -> None:
    """Fail closed unless the full research-readiness certification battery is green."""
    result = certify_research_readiness(project_root=project_root)
    lines = [
        "Euclid research-readiness certification",
        f"Project root: {result.project_root}",
        f"Report: {result.report_path}",
        f"Status: {result.status}",
    ]
    lines.extend(f"- {reason}" for reason in result.reason_codes)
    typer.echo("\n".join(lines))
    if result.status != "ready":
        raise typer.Exit(code=1)


@point_app.command("run")
def demo_point_run(
    manifest: Path = typer.Option(
        default_demo_manifest_path(),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the demo request manifest.",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory where the immutable point-evaluation artifacts are written.",
    ),
) -> None:
    """Run the demo and print the sealed point-evaluation surface summary."""
    typer.echo(
        format_point_evaluation_run_summary(
            run_demo_point_evaluation(
                manifest_path=manifest,
                output_root=output_root,
            )
        )
    )


@catalog_app.command("list")
def demo_catalog_list(
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory containing the local demo publication catalog.",
    ),
) -> None:
    """List published entries in the local read-only demo catalog."""
    typer.echo(
        format_demo_publication_catalog(
            load_demo_publication_catalog(output_root=output_root)
        )
    )


@catalog_app.command("inspect")
def demo_catalog_inspect(
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory containing the local demo publication catalog.",
    ),
    publication_id: Optional[str] = typer.Option(
        None,
        help="Optional publication id to inspect.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        help="Optional run_result run_id to inspect.",
    ),
) -> None:
    """Inspect one published entry from the local demo catalog."""
    typer.echo(
        format_demo_catalog_entry(
            inspect_demo_catalog_entry(
                output_root=output_root,
                publication_id=publication_id,
                run_id=run_id,
            )
        )
    )


@point_app.command("inspect")
def demo_point_inspect(
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory containing the prior demo run outputs.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        help="Optional run_result run_id to inspect instead of the latest summary.",
    ),
) -> None:
    """Inspect the sealed point prediction artifact and bound score result."""
    typer.echo(
        format_point_prediction_inspection(
            inspect_demo_point_prediction(output_root=output_root, run_id=run_id)
        )
    )


@point_app.command("compare")
def demo_point_compare(
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory containing the prior demo run outputs.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        help="Optional run_result run_id to inspect instead of the latest summary.",
    ),
) -> None:
    """Inspect the sealed shortlisted-candidate baseline comparison surface."""
    typer.echo(
        format_baseline_comparison(
            compare_demo_baseline(output_root=output_root, run_id=run_id)
        )
    )


@probabilistic_app.command("run")
def demo_probabilistic_run(
    manifest: Path = typer.Option(
        default_probabilistic_demo_manifest_path(),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the probabilistic demo request manifest.",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory where the probabilistic demo artifacts are written.",
    ),
) -> None:
    """Run a retained Phase 06 probabilistic demo manifest."""
    typer.echo(
        format_probabilistic_evaluation_run_summary(
            run_demo_probabilistic_evaluation(
                manifest_path=manifest,
                output_root=output_root,
            )
        )
    )


@probabilistic_app.command("inspect")
def demo_probabilistic_inspect(
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory containing the prior probabilistic demo outputs.",
    ),
) -> None:
    """Inspect the sealed probabilistic prediction artifact and bound score result."""
    typer.echo(
        format_probabilistic_prediction_inspection(
            inspect_demo_probabilistic_prediction(output_root=output_root)
        )
    )


@calibration_app.command("inspect")
def demo_calibration_inspect(
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory containing the prior probabilistic demo outputs.",
    ),
) -> None:
    """Inspect the sealed calibration artifact for a probabilistic demo run."""
    typer.echo(
        format_calibration_inspection(inspect_demo_calibration(output_root=output_root))
    )


@algorithmic_app.command("run")
def demo_algorithmic_run(
    manifest: Path = typer.Option(
        default_algorithmic_demo_manifest_path(),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the algorithmic search demo request manifest.",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory where the algorithmic search demo artifacts are written.",
    ),
) -> None:
    """Run a retained Phase 06 algorithmic search demo manifest."""
    typer.echo(
        format_algorithmic_search_run_summary(
            run_demo_algorithmic_search(
                manifest_path=manifest,
                output_root=output_root,
            )
        )
    )


demo_app.add_typer(point_app, name="point")
demo_app.add_typer(probabilistic_app, name="probabilistic")
demo_app.add_typer(calibration_app, name="calibration")
demo_app.add_typer(catalog_app, name="catalog")
demo_app.add_typer(algorithmic_app, name="algorithmic")
app.add_typer(demo_app, name="demo")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(benchmarks_app, name="benchmarks")
app.add_typer(release_app, name="release")
register_run_command(app)
register_replay_command(app)
register_workbench_command(app)


def main() -> None:
    app()
