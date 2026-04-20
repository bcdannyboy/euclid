from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
POLICY_PATH = (
    PROJECT_ROOT / "schemas" / "readiness" / "completion-regression-policy.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing required file: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_completion_regression_policy_declares_required_scores_and_surfaces() -> None:
    payload = _load_yaml(POLICY_PATH)

    assert payload["version"] == 1
    assert payload["kind"] == "completion_regression_policy"
    assert payload["policy_state"] == "full_closure"
    assert set(payload["minimum_completion_values"]) == {
        "full_vision_completion",
        "current_gate_completion",
        "shipped_releasable_completion",
    }
    assert payload["minimum_policy_verdicts"] == {
        "current_release_v1": "ready",
        "full_vision_v1": "ready",
        "shipped_releasable_v1": "ready",
    }
    assert set(payload["required_clean_install_surface_ids"]) == {
        "release_status",
        "operator_run",
        "operator_replay",
        "benchmark_execution",
        "determinism_same_seed",
        "performance_runtime_smoke",
        "packaged_notebook_smoke",
    }
    assert set(payload["policy_states"]) == {"transition", "near_closure", "full_closure"}
    assert payload["required_row_evidence_classes"] == [
        {
            "row_id": "evidence_lane:readiness_and_closure",
            "required_evidence_classes": ["packaging_install"],
        }
    ]


def test_ci_workflow_recomputes_and_verifies_completion_artifacts() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "python3.11 -m euclid release repo-test-matrix" in workflow
    assert "python3.11 -m euclid release certify-clean-install" in workflow
    assert "python3.11 -m euclid release status" in workflow
    assert "python3.11 -m euclid release verify-completion" in workflow
    assert "python3.11 -m euclid release certify-research-readiness" in workflow
    assert "build/reports/repo_test_matrix.json" in workflow
    assert "build/reports/completion-report.json" in workflow
    assert "build/reports/clean-install-certification.json" in workflow
    assert "build/reports/verify-completion.json" in workflow
    assert "build/reports/research-readiness.json" in workflow
