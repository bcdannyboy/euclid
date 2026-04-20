from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_local_dev_scripts_exist_for_bootstrap_commands() -> None:
    expected_scripts = {
        "scripts/install.sh": "pip install -e \".[dev]\"",
        "scripts/test.sh": (
            "-m pytest -q tests/unit tests/integration tests/golden tests/fixtures"
        ),
        "scripts/lint.sh": (
            "-m ruff check src/euclid tests/conftest.py "
            "tests/unit tests/integration"
        ),
        "scripts/demo.sh": "-m euclid demo run",
        "scripts/package.sh": "-m build --sdist --wheel",
        "scripts/install_smoke.sh": "-m venv",
        "scripts/benchmark_smoke.sh": (
            "-m euclid benchmarks run \\\n"
            '  --suite "${PROJECT_ROOT}/benchmarks/suites/mvp.yaml"'
        ),
        "scripts/release_smoke.sh": "-m euclid release repo-test-matrix",
    }

    for relative_path, required_fragment in expected_scripts.items():
        script_path = PROJECT_ROOT / relative_path
        script_text = script_path.read_text()

        assert script_text.startswith("#!/usr/bin/env bash")
        assert "set -euo pipefail" in script_text
        assert 'PYTHON_BIN="${PYTHON_BIN:-python3.11}"' in script_text
        assert required_fragment in script_text


def test_release_and_install_smoke_scripts_use_packaged_runtime_surfaces() -> None:
    install_smoke = (PROJECT_ROOT / "scripts/install_smoke.sh").read_text(
        encoding="utf-8"
    )
    release_smoke = (PROJECT_ROOT / "scripts/release_smoke.sh").read_text(
        encoding="utf-8"
    )
    benchmark_suite = (PROJECT_ROOT / "scripts/benchmark_suite.sh").read_text(
        encoding="utf-8"
    )
    perf_smoke = (PROJECT_ROOT / "scripts/perf_smoke.sh").read_text(encoding="utf-8")

    assert "EUCLID_PROJECT_ROOT" not in install_smoke
    assert "--project-root" not in install_smoke
    assert "PYTHONPATH" not in install_smoke
    assert "PYTHONPATH" not in benchmark_suite
    assert "PYTHONPATH" not in perf_smoke
    assert "get_release_candidate_workflow" in install_smoke
    assert 'euclid benchmarks run --suite "current-release.yaml"' in install_smoke
    assert 'euclid run --config "${WORKFLOW_MANIFEST}"' in install_smoke
    assert 'euclid replay --run-id "${WORKFLOW_RUN_ID}"' in install_smoke
    assert 'export PYTHONPATH="${PROJECT_ROOT}/src"' in release_smoke
    assert 'euclid release repo-test-matrix --project-root "${PROJECT_ROOT}"' in release_smoke
    assert 'euclid benchmarks run --suite "current-release.yaml"' in release_smoke
    assert 'euclid benchmarks run --suite "full-vision.yaml"' in release_smoke
    assert 'euclid run --config "${PROJECT_ROOT}/examples/full_vision_run.yaml"' in release_smoke
    assert 'euclid replay --run-id "full-vision-run"' in release_smoke
    assert "euclid release certify-clean-install --project-root" in release_smoke
    assert 'euclid release status --project-root "${PROJECT_ROOT}"' in release_smoke
    assert 'euclid release verify-completion --project-root "${PROJECT_ROOT}"' in release_smoke
    assert (
        'euclid release certify-research-readiness --project-root "${PROJECT_ROOT}"'
        in release_smoke
    )
    assert "tests/perf/test_runtime_smoke.py" in perf_smoke
    assert '--suite "${SUITE_NAME}"' in benchmark_suite
    assert "--no-resume" in benchmark_suite
