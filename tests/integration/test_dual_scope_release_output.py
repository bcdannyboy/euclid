from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from euclid.cli import app

RUNNER = CliRunner()


def test_release_status_command_emits_dual_scope_verdicts(
    project_root: Path,
) -> None:
    result = RUNNER.invoke(
        app,
        ["release", "status", "--project-root", str(project_root)],
    )

    assert result.exit_code == 0, result.stdout
    assert "Current release verdict:" in result.stdout
    assert "Full vision verdict:" in result.stdout
    assert "Shipped or releasable verdict:" in result.stdout
    assert "current_release_v1" in result.stdout
    assert "full_vision_v1" in result.stdout
    assert "determinism.same_seed" in result.stdout
