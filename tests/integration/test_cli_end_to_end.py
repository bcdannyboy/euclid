from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from euclid.cli import app

RUNNER = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _write_manifest(tmp_path: Path, *, request_id: str) -> Path:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "examples/current_release_run.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["request_id"] = request_id
    payload["dataset_csv"] = str(PROJECT_ROOT / "examples/minimal_dataset.csv")
    manifest_path = tmp_path / f"{request_id}.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def test_root_run_uses_operator_runtime(tmp_path: Path) -> None:
    run_imports = _imported_modules(PROJECT_ROOT / "src/euclid/cli/run.py")
    request_id = f"cli-run-{tmp_path.name}"
    manifest = _write_manifest(tmp_path, request_id=request_id)

    assert "euclid.operator_runtime.run" in run_imports
    assert "euclid.demo" not in run_imports

    result = RUNNER.invoke(
        app,
        ["run", "--config", str(manifest), "--output-root", str(tmp_path / "operator")],
    )

    assert result.exit_code == 0
    assert "Euclid run" in result.stdout
    assert "Scope ledger ref:" in result.stdout
    assert "Extension lanes:" in result.stdout


def test_root_replay_uses_operator_runtime(tmp_path: Path) -> None:
    replay_imports = _imported_modules(PROJECT_ROOT / "src/euclid/cli/replay.py")
    request_id = f"cli-replay-{tmp_path.name}"
    manifest = _write_manifest(tmp_path, request_id=request_id)
    output_root = tmp_path / "operator"

    run_result = RUNNER.invoke(
        app,
        ["run", "--config", str(manifest), "--output-root", str(output_root)],
    )
    assert run_result.exit_code == 0
    assert "euclid.operator_runtime.replay" in replay_imports
    assert "euclid.demo" not in replay_imports

    replay_result = RUNNER.invoke(
        app,
        ["replay", "--run-id", request_id, "--output-root", str(output_root)],
    )

    assert replay_result.exit_code == 0
    assert "Euclid replay" in replay_result.stdout
    assert "Replay verification: verified" in replay_result.stdout


def test_root_run_overwrites_existing_evidence_report(tmp_path: Path) -> None:
    request_id = f"cli-run-report-{tmp_path.name}"
    manifest = _write_manifest(tmp_path, request_id=request_id)
    report_path = tmp_path / "run-evidence.json"
    report_path.write_text("stale", encoding="utf-8")

    result = RUNNER.invoke(
        app,
        [
            "run",
            "--config",
            str(manifest),
            "--output-root",
            str(tmp_path / "operator"),
            "--evidence-report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_id"] == "operator_run_evidence_v1"
    assert payload["run_id"] == request_id


def test_root_replay_overwrites_existing_evidence_report(tmp_path: Path) -> None:
    request_id = f"cli-replay-report-{tmp_path.name}"
    manifest = _write_manifest(tmp_path, request_id=request_id)
    output_root = tmp_path / "operator"
    run_result = RUNNER.invoke(
        app,
        ["run", "--config", str(manifest), "--output-root", str(output_root)],
    )
    assert run_result.exit_code == 0

    report_path = tmp_path / "replay-evidence.json"
    report_path.write_text("stale", encoding="utf-8")
    replay_result = RUNNER.invoke(
        app,
        [
            "replay",
            "--run-id",
            request_id,
            "--output-root",
            str(output_root),
            "--evidence-report",
            str(report_path),
        ],
    )

    assert replay_result.exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_id"] == "operator_replay_evidence_v1"
    assert payload["run_id"] == request_id


def test_demo_commands_remain_separate_from_certified_runtime() -> None:
    root_help = RUNNER.invoke(app, ["--help"])
    demo_help = RUNNER.invoke(app, ["demo", "--help"])

    assert root_help.exit_code == 0
    assert demo_help.exit_code == 0
    assert "compatibility-only" in root_help.stdout.lower()
    assert "compatibility-only" in demo_help.stdout.lower()
    assert "not a certified runtime" in demo_help.stdout.lower()
