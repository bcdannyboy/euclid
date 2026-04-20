from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    REPO_ROOT / "docs/implementation/certification-evidence-contract.yaml"
)
AUTHORITY_SNAPSHOT_PATH = REPO_ROOT / "docs/implementation/authority-snapshot.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), (
        "missing required file: " f"{path.relative_to(REPO_ROOT).as_posix()}"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_closing_evidence_bundles_require_full_provenance_fields() -> None:
    payload = _load_yaml(CONTRACT_PATH)
    required_fields = set(payload["required_bundle_fields"])
    assert required_fields >= {
        "authority_snapshot_id",
        "command_contract_id",
        "closure_map_id",
        "traceability_id",
        "fixture_spec_id",
        "producer_command_id",
    }
    for bundle in payload["bundles"]:
        assert required_fields <= set(bundle)


def test_stale_evidence_is_rejected_after_authority_or_manifest_change() -> None:
    payload = _load_yaml(CONTRACT_PATH)
    authority_snapshot = _load_yaml(AUTHORITY_SNAPSHOT_PATH)
    assert any("stale" in rule.lower() for rule in payload["rules"])
    for bundle in payload["bundles"]:
        assert bundle["authority_snapshot_id"] == authority_snapshot["authority_snapshot_id"]
        assert bundle["input_manifest_digests"]


def test_scope_specific_evidence_bundles_cannot_alias_each_other() -> None:
    payload = _load_yaml(CONTRACT_PATH)
    bundles_by_scope = {bundle["scope_id"]: bundle for bundle in payload["bundles"]}

    assert {"current_release", "full_vision", "shipped_releasable"} <= set(
        bundles_by_scope
    )
    assert len({bundle["evidence_bundle_id"] for bundle in payload["bundles"]}) == len(
        payload["bundles"]
    )
    assert (
        bundles_by_scope["current_release"]["evidence_bundle_id"]
        != bundles_by_scope["full_vision"]["evidence_bundle_id"]
    )
    assert (
        bundles_by_scope["full_vision"]["evidence_bundle_id"]
        != bundles_by_scope["shipped_releasable"]["evidence_bundle_id"]
    )
