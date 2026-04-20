from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_RELEASE_SUITE = REPO_ROOT / "benchmarks/suites/current-release.yaml"
PACKAGED_CURRENT_RELEASE_SUITE = (
    REPO_ROOT / "src/euclid/_assets/benchmarks/suites/current-release.yaml"
)
CURRENT_RELEASE_WORKFLOW = REPO_ROOT / "examples/current_release_run.yaml"
PACKAGED_CURRENT_RELEASE_WORKFLOW = (
    REPO_ROOT / "src/euclid/_assets/examples/current_release_run.yaml"
)
REMOVED_ACTIVE_ALIASES = (
    REPO_ROOT / "benchmarks/suites/full-program.yaml",
    REPO_ROOT / "src/euclid/_assets/benchmarks/suites/full-program.yaml",
    REPO_ROOT / "examples/full_program_run.yaml",
    REPO_ROOT / "src/euclid/_assets/examples/full_program_run.yaml",
)
ACTIVE_SCOPE_FILES = (
    CURRENT_RELEASE_SUITE,
    PACKAGED_CURRENT_RELEASE_SUITE,
    CURRENT_RELEASE_WORKFLOW,
    PACKAGED_CURRENT_RELEASE_WORKFLOW,
    REPO_ROOT / "src/euclid/bootstrap.py",
    REPO_ROOT / "src/euclid/_version.py",
    REPO_ROOT / "src/euclid/__init__.py",
    REPO_ROOT / "src/euclid/release.py",
    REPO_ROOT / "schemas/readiness/euclid-readiness.yaml",
    REPO_ROOT / "src/euclid/_assets/schemas/readiness/euclid-readiness.yaml",
    REPO_ROOT / "scripts/install_smoke.sh",
    REPO_ROOT / "scripts/benchmark_suite.sh",
)
DISALLOWED_CANONICAL_ALIASES = (
    "full_program",
    "full-program.yaml",
    "full_program_run.yaml",
    "benchmarks.full_program",
    "suite.full_program",
    "euclid_full_program_release_candidate",
    "full_program_release",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), (
        "missing required file: " f"{path.relative_to(REPO_ROOT).as_posix()}"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_current_release_suite_and_example_use_current_release_name() -> None:
    suite_payload = _load_yaml(CURRENT_RELEASE_SUITE)
    packaged_suite_payload = _load_yaml(PACKAGED_CURRENT_RELEASE_SUITE)
    workflow_payload = _load_yaml(CURRENT_RELEASE_WORKFLOW)
    packaged_workflow_payload = _load_yaml(PACKAGED_CURRENT_RELEASE_WORKFLOW)

    assert suite_payload["suite_id"] == "current_release"
    assert packaged_suite_payload["suite_id"] == "current_release"
    assert workflow_payload["workflow_id"] == "euclid_current_release_candidate"
    assert (
        packaged_workflow_payload["workflow_id"]
        == "euclid_current_release_candidate"
    )
    assert (
        workflow_payload["benchmark_suite"]
        == "benchmarks/suites/current-release.yaml"
    )
    assert packaged_workflow_payload["benchmark_suite"] == (
        "benchmarks/suites/current-release.yaml"
    )

    for removed_alias in REMOVED_ACTIVE_ALIASES:
        assert not removed_alias.exists(), (
            f"active alias must be retired: {removed_alias.relative_to(REPO_ROOT)}"
        )


def test_no_active_scope_bearing_surface_uses_full_program_as_canonical_name() -> None:
    for path in ACTIVE_SCOPE_FILES:
        text = path.read_text(encoding="utf-8")
        for fragment in DISALLOWED_CANONICAL_ALIASES:
            assert fragment not in text, (
                f"{path.relative_to(REPO_ROOT)} still uses "
                f"stale canonical alias {fragment}"
            )
