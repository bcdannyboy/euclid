from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from euclid.cli import app
from euclid.operator_runtime.run import run_operator

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = CliRunner()


def _normalized(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split()).lower()


def test_readme_describes_certified_and_compatibility_surfaces_truthfully() -> None:
    readme = _normalized(REPO_ROOT / "README.md")

    assert "compatibility-only tooling" in readme
    assert "`current_release`" in readme
    assert "`full_vision`" in readme
    assert "`shipped_releasable`" in readme
    assert "certified runtime" in readme


def test_cli_help_labels_demo_only_surfaces() -> None:
    root_help = RUNNER.invoke(app, ["--help"])
    demo_help = RUNNER.invoke(app, ["demo", "--help"])

    assert root_help.exit_code == 0
    assert demo_help.exit_code == 0
    assert "compatibility-only" in root_help.stdout.lower()
    assert "compatibility-only" in demo_help.stdout.lower()
    assert "not a certified runtime" in demo_help.stdout.lower()


def test_certified_runtime_path_avoids_prototype_scope_ledger(tmp_path: Path) -> None:
    result = run_operator(
        manifest_path=REPO_ROOT / "examples" / "current_release_run.yaml",
        output_root=tmp_path / "operator-run",
    )

    assert "prototype" not in result.summary.scope_ledger_ref.object_id
    assert "prototype" not in result.paths.run_summary_path.read_text(encoding="utf-8")
