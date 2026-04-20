from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    REPO_ROOT / "docs/implementation/lifecycle-artifact-closure-contract.yaml"
)
CLOSURE_MAP_PATH = REPO_ROOT / "docs/implementation/euclid-closure-map.yaml"
TRACEABILITY_PATH = REPO_ROOT / "docs/implementation/subtask-test-traceability.yaml"
FULL_VISION_MATRIX_PATH = REPO_ROOT / "schemas/readiness/full-vision-matrix.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), (
        "missing required file: " f"{path.relative_to(REPO_ROOT).as_posix()}"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_every_lifecycle_artifact_family_requires_four_proof_kinds() -> None:
    payload = _load_yaml(CONTRACT_PATH)
    assert set(payload["required_proof_kinds"]) == {
        "producer",
        "downstream_consumer",
        "replay",
        "tamper_or_missing_negative",
    }
    for family in payload["artifact_families"]:
        assert family["producer_test_ids"]
        assert family["downstream_consumer_test_ids"]
        assert family["replay_test_ids"]
        assert family["tamper_or_missing_negative_test_ids"]


def test_no_lifecycle_row_can_close_from_existence_only() -> None:
    payload = _load_yaml(CONTRACT_PATH)
    assert any("Artifact existence alone never closes" in rule for rule in payload["rules"])
    assert any("Schema validation alone never closes" in rule for rule in payload["rules"])
    assert any("Roundtrip-only proof never closes" in rule for rule in payload["rules"])


def test_lifecycle_contract_rows_match_closure_map_and_traceability() -> None:
    contract = _load_yaml(CONTRACT_PATH)
    closure_map = _load_yaml(CLOSURE_MAP_PATH)
    traceability = _load_yaml(TRACEABILITY_PATH)
    matrix = _load_yaml(FULL_VISION_MATRIX_PATH)

    lifecycle_rows = {
        row["row_id"]
        for row in matrix["rows"]
        if row["capability_type"] == "lifecycle_artifact"
    }
    contract_rows = {
        capability_row_id
        for family in contract["artifact_families"]
        for capability_row_id in family["capability_row_ids"]
    }
    closure_rows = {
        row["capability_row_id"]
        for row in closure_map["rows"]
        if row["row_family"] == "lifecycle_artifact"
    }
    traceability_test_ids = {
        test_id
        for row in traceability["rows"]
        for test_ids in row["named_test_ids"].values()
        for test_id in test_ids
    }

    assert lifecycle_rows == contract_rows == closure_rows
    for family in contract["artifact_families"]:
        for test_id in (
            *family["producer_test_ids"],
            *family["downstream_consumer_test_ids"],
            *family["replay_test_ids"],
            *family["tamper_or_missing_negative_test_ids"],
        ):
            assert test_id in traceability_test_ids
