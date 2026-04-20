from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SURVIVING_IMPLEMENTATION_FILES = (
    "docs/implementation/authority-reconciliation.yaml",
    "docs/implementation/authority-snapshot.yaml",
    "docs/implementation/certification-command-contract.yaml",
    "docs/implementation/certification-evidence-contract.yaml",
    "docs/implementation/certification-fixture-spec.yaml",
    "docs/implementation/euclid-closure-map.yaml",
    "docs/implementation/lifecycle-artifact-closure-contract.yaml",
    "docs/implementation/subtask-test-traceability.yaml",
)


def _load_yaml(relative_path: str) -> dict[str, Any]:
    path = REPO_ROOT / relative_path
    assert path.is_file(), f"missing required file: {relative_path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_cleanup_register_lists_all_known_misleading_surfaces() -> None:
    implementation_files = tuple(
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted((REPO_ROOT / "docs/implementation").glob("*.yaml"))
    )
    assert implementation_files == SURVIVING_IMPLEMENTATION_FILES


def test_cleanup_register_entries_have_owner_action_and_closure_fields() -> None:
    snapshot = _load_yaml("docs/implementation/authority-snapshot.yaml")
    reconciliation = _load_yaml("docs/implementation/authority-reconciliation.yaml")
    closure_map = _load_yaml("docs/implementation/euclid-closure-map.yaml")
    traceability = _load_yaml("docs/implementation/subtask-test-traceability.yaml")
    command_contract = _load_yaml("docs/implementation/certification-command-contract.yaml")
    evidence_contract = _load_yaml("docs/implementation/certification-evidence-contract.yaml")

    dependent_contracts = {
        entry["path"] for entry in snapshot["dependent_contracts"]
    }
    assert dependent_contracts >= {
        path
        for path in SURVIVING_IMPLEMENTATION_FILES
        if path != "docs/implementation/authority-snapshot.yaml"
    }

    authority_snapshot_id = snapshot["authority_snapshot_id"]
    assert reconciliation["authority_snapshot_id"] == authority_snapshot_id
    assert closure_map["authority_snapshot_id"] == authority_snapshot_id
    assert command_contract["authority_snapshot_id"] == authority_snapshot_id
    assert evidence_contract["authority_snapshot_id"] == authority_snapshot_id

    traceability_id = traceability["traceability_id"]
    assert closure_map["traceability_id"] == traceability_id
    assert command_contract["traceability_id"] == traceability_id
    assert evidence_contract["traceability_id"] == traceability_id


def test_overview_routes_doc_cleanup_work_to_register() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    contracts = (REPO_ROOT / "docs/reference/contracts-manifests.md").read_text(
        encoding="utf-8"
    )
    readiness = (REPO_ROOT / "docs/reference/benchmarks-readiness.md").read_text(
        encoding="utf-8"
    )

    assert "docs/reference/contracts-manifests.md" in readme
    assert "docs/reference/benchmarks-readiness.md" in readme
    assert "`docs/implementation/*.yaml`" in contracts
    assert "`docs/implementation/*.yaml`" in readiness
    assert "documentation-cleanup-register.md" not in readme
    assert "documentation-cleanup-register.md" not in contracts
    assert "documentation-cleanup-register.md" not in readiness
