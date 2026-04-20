from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
CLI_PATH = REPO_ROOT / "src" / "euclid" / "cli" / "__init__.py"
RELEASE_SMOKE_PATH = REPO_ROOT / "scripts" / "release_smoke.sh"
BENCHMARK_SUITE_SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark_suite.sh"


def test_readme_and_cli_claims_match_runtime_truth() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    cli = CLI_PATH.read_text(encoding="utf-8")

    assert "`current_release`" in readme
    assert "`full_vision`" in readme
    assert "`shipped_releasable`" in readme
    assert "certify-research-readiness" in readme
    assert "compatibility-only tooling" in readme

    assert "compatibility-only" in cli
    assert "certify-research-readiness" in cli
    assert "repo-test-matrix" in cli


def test_release_smoke_covers_public_release_story() -> None:
    script = RELEASE_SMOKE_PATH.read_text(encoding="utf-8")

    assert "release repo-test-matrix" in script
    assert 'benchmarks run --suite "current-release.yaml"' in script
    assert 'benchmarks run --suite "full-vision.yaml"' in script
    assert "run --config" in script
    assert "full_vision_run.yaml" in script
    assert "replay --run-id" in script
    assert "full-vision-run" in script
    assert "release certify-clean-install" in script
    assert "release status" in script
    assert "release verify-completion" in script
    assert "release certify-research-readiness" in script


def test_benchmark_suite_script_defaults_match_certified_suite() -> None:
    script = BENCHMARK_SUITE_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'SUITE_NAME="${SUITE_NAME:-current-release.yaml}"' in script
    assert "--suite" in script
    assert "full-program" not in script
