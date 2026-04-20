from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from euclid.cli import app

RUNNER = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_RELEASE_MANIFEST = PROJECT_ROOT / "examples/current_release_run.yaml"
PROBABILISTIC_MANIFEST = (
    PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-distribution-demo.yaml"
)
ALGORITHMIC_MANIFEST = (
    PROJECT_ROOT / "fixtures/runtime/phase06/algorithmic-search-demo.yaml"
)


def _write_cli_manifest(
    tmp_path: Path,
    *,
    request_id: str,
    source_manifest: Path = CURRENT_RELEASE_MANIFEST,
) -> Path:
    payload = yaml.safe_load(source_manifest.read_text(encoding="utf-8"))
    payload["request_id"] = request_id
    dataset_csv = Path(payload["dataset_csv"])
    if not dataset_csv.is_absolute():
        dataset_csv = source_manifest.parent / dataset_csv
    payload["dataset_csv"] = str(dataset_csv)
    manifest_path = tmp_path / f"{request_id}.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def test_smoke_command_prints_available_workflows() -> None:
    result = RUNNER.invoke(app, ["smoke"])

    assert result.exit_code == 0
    assert "Available workflows" in result.stdout
    assert "benchmarks.current_release" in result.stdout
    assert "demo.replay" not in result.stdout
    assert "storage: sqlite3" in result.stdout


def test_demo_run_command_prints_readable_summary(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "demo",
            "run",
            "--manifest",
            str(PROJECT_ROOT / "fixtures/runtime/prototype-demo.yaml"),
            "--output-root",
            str(tmp_path / "demo-output"),
        ],
    )

    assert result.exit_code == 0
    assert "Euclid demo run" in result.stdout
    assert "Selected family: constant" in result.stdout
    assert "Result mode: abstention_only_publication" in result.stdout
    assert "prototype_constant_candidate_bundle_v1" in result.stdout


def test_demo_profile_run_command_prints_telemetry_summary(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "demo",
            "profile-run",
            "--manifest",
            str(PROJECT_ROOT / "fixtures/runtime/prototype-demo.yaml"),
            "--output-root",
            str(tmp_path / "demo-output"),
        ],
    )

    assert result.exit_code == 0
    assert "Euclid demo profile" in result.stdout
    assert "Telemetry artifact:" in result.stdout
    assert "profile_kind=demo_run" in result.stdout


def test_benchmark_profile_task_command_prints_telemetry_summary(
    tmp_path: Path,
) -> None:
    result = RUNNER.invoke(
        app,
        [
            "benchmark",
            "profile-task",
            "--manifest",
            str(
                PROJECT_ROOT
                / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
            ),
            "--benchmark-root",
            str(tmp_path / "benchmarks"),
        ],
    )

    assert result.exit_code == 0
    assert "Euclid benchmark profile" in result.stdout
    assert "Telemetry artifact:" in result.stdout
    assert "Task id: planted_analytic_demo" in result.stdout


def test_benchmark_profile_task_command_accepts_parallel_resume_controls(
    tmp_path: Path,
) -> None:
    result = RUNNER.invoke(
        app,
        [
            "benchmark",
            "profile-task",
            "--manifest",
            str(
                PROJECT_ROOT
                / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
            ),
            "--benchmark-root",
            str(tmp_path / "benchmarks"),
            "--parallel-workers",
            "3",
            "--resume",
        ],
    )

    assert result.exit_code == 0
    assert "Euclid benchmark profile" in result.stdout


def test_run_command_prints_operator_facing_summary(tmp_path: Path) -> None:
    manifest = _write_cli_manifest(
        tmp_path,
        request_id=f"cli-run-{tmp_path.name}",
    )

    result = RUNNER.invoke(
        app,
        [
            "run",
            "--config",
            str(manifest),
            "--output-root",
            str(tmp_path / "operator-run"),
        ],
    )

    assert result.exit_code == 0
    assert "Euclid run" in result.stdout
    assert "Run result ref:" in result.stdout
    assert "Bundle ref:" in result.stdout
    assert "Scope ledger ref:" in result.stdout
    assert "Saved summary:" in result.stdout


def test_replay_command_resolves_default_output_root_from_run_id(
    tmp_path: Path,
) -> None:
    request_id = f"cli-replay-{tmp_path.name}"
    manifest = _write_cli_manifest(tmp_path, request_id=request_id)

    run_result = RUNNER.invoke(
        app,
        ["run", "--config", str(manifest)],
    )
    assert run_result.exit_code == 0

    replay_result = RUNNER.invoke(
        app,
        ["replay", "--run-id", request_id],
    )

    assert replay_result.exit_code == 0
    assert "Euclid replay" in replay_result.stdout
    assert "Run id: " + request_id in replay_result.stdout
    assert "Replay verification: verified" in replay_result.stdout


def test_benchmarks_run_command_runs_named_suite(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "benchmarks",
            "run",
            "--suite",
            "rediscovery",
            "--benchmark-root",
            str(tmp_path / "benchmarks"),
        ],
    )

    assert result.exit_code == 0
    assert "Euclid benchmark suite" in result.stdout
    assert "Suite: rediscovery" in result.stdout
    assert "planted_analytic_demo" in result.stdout


def test_demo_replay_command_reads_the_bundle_from_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "demo-output"

    run_result = RUNNER.invoke(
        app,
        [
            "demo",
            "run",
            "--manifest",
            str(PROJECT_ROOT / "fixtures/runtime/prototype-demo.yaml"),
            "--output-root",
            str(output_root),
        ],
    )
    assert run_result.exit_code == 0

    replay_result = RUNNER.invoke(
        app,
        ["demo", "replay", "--output-root", str(output_root)],
    )

    assert replay_result.exit_code == 0
    assert "Euclid demo replay" in replay_result.stdout
    assert "Replay verification: verified" in replay_result.stdout
    assert "Selected family: constant" in replay_result.stdout


def test_demo_replay_inspect_command_prints_bundle_details(
    phase01_demo_output_root: Path,
) -> None:
    result = RUNNER.invoke(
        app,
        ["demo", "replay-inspect", "--output-root", str(phase01_demo_output_root)],
    )

    assert result.exit_code == 0
    assert "Euclid demo replay bundle" in result.stdout
    assert "Seed records:" in result.stdout
    assert "Stage order:" in result.stdout


def test_demo_inspect_command_requires_existing_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "demo-output"
    result = RUNNER.invoke(
        app,
        ["demo", "inspect", "--output-root", str(output_root)],
    )

    assert result.exit_code != 0


def test_demo_inspect_command_prints_run_manifest_graph_from_phase01_outputs(
    phase01_demo_output_root: Path,
) -> None:
    inspect_result = RUNNER.invoke(
        app,
        ["demo", "inspect", "--output-root", str(phase01_demo_output_root)],
    )

    assert inspect_result.exit_code == 0
    assert "Euclid demo artifact inspection" in inspect_result.stdout
    assert "Run id: prototype_constant_candidate_run_result_v1" in inspect_result.stdout
    assert (
        "run_result_manifest@1.1.0:prototype_constant_candidate_run_result_v1"
        in inspect_result.stdout
    )
    assert "search_plan_manifest@1.0.0:" in inspect_result.stdout


def test_demo_resolve_command_prints_manifest_relationships_from_phase01_outputs(
    phase01_demo_output_root: Path,
) -> None:
    resolve_result = RUNNER.invoke(
        app,
        [
            "demo",
            "resolve",
            "--output-root",
            str(phase01_demo_output_root),
            "--ref",
            "run_result_manifest@1.1.0:prototype_constant_candidate_run_result_v1",
        ],
    )

    assert resolve_result.exit_code == 0
    assert "Resolved manifest" in resolve_result.stdout
    assert "Schema: run_result_manifest@1.1.0" in resolve_result.stdout
    assert "Children:" in resolve_result.stdout
    assert "publication_record_manifest@1.1.0:" in resolve_result.stdout
    assert "scope_ledger_manifest@1.0.0:" in resolve_result.stdout


def test_demo_lineage_command_prints_lineage_tree_from_phase01_outputs(
    phase01_demo_output_root: Path,
) -> None:
    lineage_result = RUNNER.invoke(
        app,
        ["demo", "lineage", "--output-root", str(phase01_demo_output_root)],
    )

    assert lineage_result.exit_code == 0
    assert "Euclid demo lineage graph" in lineage_result.stdout
    assert (
        "run_result_manifest@1.1.0:prototype_constant_candidate_run_result_v1"
        in lineage_result.stdout
    )
    assert "publication_record_manifest@1.1.0:" in lineage_result.stdout


def test_demo_validate_command_reports_clean_store_from_phase01_outputs(
    phase01_demo_output_root: Path,
) -> None:
    validate_result = RUNNER.invoke(
        app,
        ["demo", "validate", "--output-root", str(phase01_demo_output_root)],
    )

    assert validate_result.exit_code == 0
    assert "Euclid demo store validation" in validate_result.stdout
    assert "Store valid: yes" in validate_result.stdout
    assert "Issue count: 0" in validate_result.stdout


def test_demo_publish_and_catalog_commands_project_read_only_catalog(
    phase01_demo_output_root: Path,
) -> None:
    publish_result = RUNNER.invoke(
        app,
        ["demo", "publish", "--output-root", str(phase01_demo_output_root)],
    )

    assert publish_result.exit_code == 0
    assert "Euclid demo catalog publication" in publish_result.stdout
    assert "Publication mode: abstention_only_publication" in publish_result.stdout

    list_result = RUNNER.invoke(
        app,
        ["demo", "catalog", "list", "--output-root", str(phase01_demo_output_root)],
    )

    assert list_result.exit_code == 0
    assert "Euclid demo publication catalog" in list_result.stdout
    assert "prototype-demo" in list_result.stdout
    assert "abstention_only_publication" in list_result.stdout


def test_demo_catalog_inspect_command_reads_published_entry(
    phase01_demo_output_root: Path,
) -> None:
    publish_result = RUNNER.invoke(
        app,
        ["demo", "publish", "--output-root", str(phase01_demo_output_root)],
    )
    assert publish_result.exit_code == 0

    inspect_result = RUNNER.invoke(
        app,
        ["demo", "catalog", "inspect", "--output-root", str(phase01_demo_output_root)],
    )

    assert inspect_result.exit_code == 0
    assert "Euclid demo catalog entry" in inspect_result.stdout
    assert "Publication record:" in inspect_result.stdout
    assert "Reproducibility bundle:" in inspect_result.stdout


def test_demo_point_run_command_prints_point_evaluation_summary(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "demo",
            "point",
            "run",
            "--manifest",
            str(PROJECT_ROOT / "fixtures/runtime/prototype-demo.yaml"),
            "--output-root",
            str(tmp_path / "demo-output"),
        ],
    )

    assert result.exit_code == 0
    assert "Euclid demo point evaluation run" in result.stdout
    assert "Prediction artifact:" in result.stdout
    assert "Comparison universe:" in result.stdout


def test_demo_point_inspect_command_reads_prediction_artifact_from_store(
    phase01_demo_output_root: Path,
) -> None:
    result = RUNNER.invoke(
        app,
        [
            "demo",
            "point",
            "inspect",
            "--output-root",
            str(phase01_demo_output_root),
        ],
    )

    assert result.exit_code == 0
    assert "Euclid demo point prediction" in result.stdout
    assert "Stage: confirmatory_holdout" in result.stdout
    assert "Horizon set: 1" in result.stdout


def test_demo_point_compare_command_reads_baseline_comparison_from_store(
    phase01_demo_output_root: Path,
) -> None:
    result = RUNNER.invoke(
        app,
        [
            "demo",
            "point",
            "compare",
            "--output-root",
            str(phase01_demo_output_root),
        ],
    )

    assert result.exit_code == 0
    assert "Euclid demo baseline comparison" in result.stdout
    assert "Baseline id: constant_baseline" in result.stdout
    assert "Candidate beats baseline:" in result.stdout


def test_demo_point_run_help_renders_without_crashing() -> None:
    result = RUNNER.invoke(app, ["demo", "point", "run", "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "Euclid demo point evaluation run" not in result.stdout
    assert "Run the demo and print the sealed point-evaluation surface summary." in (
        result.stdout
    )


def test_demo_probabilistic_run_command_prints_probabilistic_summary(
    tmp_path: Path,
) -> None:
    result = RUNNER.invoke(
        app,
        [
            "demo",
            "probabilistic",
            "run",
            "--manifest",
            str(PROBABILISTIC_MANIFEST),
            "--output-root",
            str(tmp_path / "probabilistic-output"),
        ],
    )

    assert result.exit_code == 0
    assert "Euclid demo probabilistic evaluation run" in result.stdout
    assert "Forecast object type: distribution" in result.stdout
    assert "Calibration result:" in result.stdout


def test_demo_calibration_inspect_command_reads_calibration_artifact(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "probabilistic-output"
    run_result = RUNNER.invoke(
        app,
        [
            "demo",
            "probabilistic",
            "run",
            "--manifest",
            str(PROBABILISTIC_MANIFEST),
            "--output-root",
            str(output_root),
        ],
    )
    assert run_result.exit_code == 0

    inspect_result = RUNNER.invoke(
        app,
        [
            "demo",
            "calibration",
            "inspect",
            "--output-root",
            str(output_root),
        ],
    )

    assert inspect_result.exit_code == 0
    assert "Euclid demo calibration" in inspect_result.stdout
    assert "Forecast object type: distribution" in inspect_result.stdout
    assert "Diagnostics:" in inspect_result.stdout


def test_demo_algorithmic_run_command_prints_search_summary(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "demo",
            "algorithmic",
            "run",
            "--manifest",
            str(ALGORITHMIC_MANIFEST),
            "--output-root",
            str(tmp_path / "algorithmic-output"),
        ],
    )

    assert result.exit_code == 0
    assert "Euclid demo algorithmic search" in result.stdout
    assert "Selected family: algorithmic" in result.stdout
    assert "Accepted candidates:" in result.stdout
