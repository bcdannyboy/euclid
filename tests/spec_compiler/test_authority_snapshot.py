from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHORITY_SNAPSHOT_PATH = REPO_ROOT / "docs/implementation/authority-snapshot.yaml"
LIVE_SCOPE_AUTHORITY_DOCS = {
    "README.md",
    "docs/reference/system.md",
    "docs/reference/contracts-manifests.md",
    "docs/reference/benchmarks-readiness.md",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), (
        "missing required file: " f"{path.relative_to(REPO_ROOT).as_posix()}"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_authority_snapshot_hashes_match_authority_docs() -> None:
    payload = _load_yaml(AUTHORITY_SNAPSHOT_PATH)
    for entry in payload["scope_authority_docs"]:
        path = REPO_ROOT / entry["path"]
        assert entry["sha256"] == _sha256(path)
    master_plan = payload["master_plan"]
    master_plan_path = REPO_ROOT / master_plan["path"]
    if master_plan_path.is_file():
        assert master_plan["sha256"] == _sha256(master_plan_path)
    else:
        # The authority snapshot still carries the planning anchor even when the
        # plan document is not present in a trimmed workspace checkout.
        assert master_plan["path"].startswith("docs/plans/")


def test_authority_snapshot_tracks_live_reference_workspace_docs() -> None:
    payload = _load_yaml(AUTHORITY_SNAPSHOT_PATH)
    scope_docs = {entry["path"] for entry in payload["scope_authority_docs"]}

    assert LIVE_SCOPE_AUTHORITY_DOCS <= scope_docs
    assert "EUCLID.md" not in scope_docs
    assert not any(path.startswith("docs/canonical/") for path in scope_docs)


def test_authority_snapshot_references_matrix_policies_traceability_and_fixture_spec(
) -> None:
    payload = _load_yaml(AUTHORITY_SNAPSHOT_PATH)
    dependent_paths = {
        entry["path"] for entry in payload["dependent_contracts"]
    }
    assert {
        "schemas/readiness/full-vision-matrix.yaml",
        "schemas/readiness/current-release-v1.yaml",
        "schemas/readiness/full-vision-v1.yaml",
        "schemas/readiness/shipped-releasable-v1.yaml",
        "schemas/readiness/euclid-readiness.yaml",
        "schemas/readiness/evidence-strength-policy.yaml",
        "schemas/readiness/completion-regression-policy.yaml",
        "schemas/readiness/completion-report.schema.yaml",
        "docs/implementation/euclid-closure-map.yaml",
        "docs/implementation/subtask-test-traceability.yaml",
        "docs/implementation/certification-fixture-spec.yaml",
    } <= dependent_paths


def test_authority_changes_fail_until_snapshot_is_reconciled() -> None:
    payload = _load_yaml(AUTHORITY_SNAPSHOT_PATH)
    stale_payload = dict(payload)
    stale_docs = [dict(entry) for entry in payload["scope_authority_docs"]]
    stale_docs[0]["sha256"] = "0" * 64
    stale_payload["scope_authority_docs"] = stale_docs

    live_hash = _sha256(REPO_ROOT / stale_docs[0]["path"])
    assert stale_docs[0]["sha256"] != live_hash
    assert any("invalidates this snapshot" in rule for rule in payload["rules"])


def test_certification_manifests_reference_authority_snapshot_id() -> None:
    payload = _load_yaml(AUTHORITY_SNAPSHOT_PATH)
    authority_snapshot_id = payload["authority_snapshot_id"]

    for relative_path in (
        "benchmarks/suites/current-release.yaml",
        "benchmarks/suites/full-vision.yaml",
        "examples/current_release_run.yaml",
        "examples/full_vision_run.yaml",
    ):
        manifest = _load_yaml(REPO_ROOT / relative_path)
        assert manifest["authority_snapshot_id"] == authority_snapshot_id

    command_contract = _load_yaml(
        REPO_ROOT / "docs/implementation/certification-command-contract.yaml"
    )
    for command in command_contract["commands"]:
        assert command["authority_snapshot_id"] == authority_snapshot_id
