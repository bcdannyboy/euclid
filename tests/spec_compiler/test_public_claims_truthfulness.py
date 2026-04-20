from __future__ import annotations

from pathlib import Path
import sys

import pytest
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback for local test envs
    import tomli as tomllib

from typer.testing import CliRunner

if sys.version_info >= (3, 11):
    from euclid.cli import app
else:  # pragma: no cover - local verification env is Python 3.9
    app = None


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = CliRunner()


def _normalized(relative_path: str) -> str:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    return " ".join(text.replace(">", " ").split()).lower()


def test_package_metadata_does_not_overclaim_certification() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    description = pyproject["project"]["description"].lower()

    assert "certified full-program runtime surfaces" not in description
    assert "full-program" not in description
    assert "current_release" in description
    assert "full_vision" in description


def test_cli_help_does_not_overclaim_full_program_certification() -> None:
    if app is None:
        pytest.skip("CLI import requires Python >= 3.11")

    result = RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "certified cli" not in result.stdout.lower()
    assert "full-program runtime" not in result.stdout.lower()
    assert "current_release" in result.stdout
    assert "full_vision" in result.stdout


def test_readme_and_runbook_claims_match_scope_authority() -> None:
    required_scope_terms = ("`current_release`", "`full_vision`", "`shipped_releasable`")
    stale_claims = (
        "current certified release scope",
        "full-program python surface",
        "full-program release workflow",
        "full-program certification example",
        "certified operator target",
    )
    checked_docs = (
        "README.md",
        "docs/reference/runtime-cli.md",
        "docs/reference/benchmarks-readiness.md",
        "docs/reference/testing-truthfulness.md",
    )

    for relative_path in checked_docs:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        normalized = " ".join(text.replace(">", " ").split()).lower()
        for stale_claim in stale_claims:
            assert stale_claim not in normalized, (
                f"{relative_path} still overclaims public scope with: {stale_claim}"
            )

    for relative_path in ("README.md", "docs/reference/benchmarks-readiness.md"):
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for term in required_scope_terms:
            assert term in text, f"{relative_path} must name {term}"

    runtime_cli = (REPO_ROOT / "docs/reference/runtime-cli.md").read_text(
        encoding="utf-8"
    )
    assert "The main way to run Euclid is through `euclid`" in runtime_cli
    assert "## Compatibility-only commands" in runtime_cli

    truthfulness = (REPO_ROOT / "docs/reference/testing-truthfulness.md").read_text(
        encoding="utf-8"
    )
    assert "`tests/spec_compiler/*`" in truthfulness
