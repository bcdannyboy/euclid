from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "docs/plans/2026-04-21-euclid-enhancement-master-plan.md"
TRACEABILITY_PATH = REPO_ROOT / "docs/implementation/enhancement-traceability.yaml"
PHASE_IDS = tuple(f"P{index:02d}" for index in range(17))
REQUIRED_ROW_FIELDS = {
    "id",
    "phase_id",
    "status",
    "implementation_files",
    "test_files",
    "gate_refs",
    "evidence_refs",
    "edge_cases",
    "redaction_assertions",
    "replay_assertions",
    "claim_scope_assertions",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), (
        "missing required file: " f"{path.relative_to(REPO_ROOT).as_posix()}"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _plan_ids() -> dict[str, set[str]]:
    grouped = {phase_id: set() for phase_id in PHASE_IDS}
    text = PLAN_PATH.read_text(encoding="utf-8")
    for match in re.finditer(r"\b(P\d{2}-T\d{2}(?:-S\d{2})?)\b", text):
        item_id = match.group(1)
        grouped[item_id[:3]].add(item_id)
    return grouped


def test_enhancement_traceability_covers_every_plan_id_once() -> None:
    payload = _load_yaml(TRACEABILITY_PATH)
    rows = payload["rows"]
    rows_by_id = {row["id"]: row for row in rows}
    expected_ids = set().union(*_plan_ids().values())

    assert len(rows_by_id) == len(rows)
    assert set(rows_by_id) == expected_ids


def test_enhancement_traceability_rows_have_non_empty_evidence_spine_fields() -> None:
    payload = _load_yaml(TRACEABILITY_PATH)

    for row in payload["rows"]:
        assert REQUIRED_ROW_FIELDS <= set(row)
        for field in REQUIRED_ROW_FIELDS:
            assert row[field], f"{row['id']} has empty {field}"


def test_gate_manifests_mirror_enhancement_traceability_rows() -> None:
    rows = {row["id"]: row for row in _load_yaml(TRACEABILITY_PATH)["rows"]}

    for phase_id, required_ids in _plan_ids().items():
        manifest = _load_yaml(REPO_ROOT / "tests/gates" / f"{phase_id}.yaml")
        assert set(manifest["covered_ids"]) == required_ids
        assert set(manifest["id_gates"]) == required_ids
        for item_id in required_ids:
            assert manifest["id_gates"][item_id]["status"] == rows[item_id]["status"]
            assert manifest["id_gates"][item_id]["gate_refs"] == rows[item_id][
                "gate_refs"
            ]
