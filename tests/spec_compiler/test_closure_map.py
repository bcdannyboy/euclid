from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CLOSURE_MAP_PATH = REPO_ROOT / "docs/implementation/euclid-closure-map.yaml"
FULL_VISION_MATRIX_PATH = REPO_ROOT / "schemas/readiness/full-vision-matrix.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), (
        "missing required file: " f"{path.relative_to(REPO_ROOT).as_posix()}"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_every_full_vision_row_has_owning_tasks() -> None:
    matrix = _load_yaml(FULL_VISION_MATRIX_PATH)
    closure_map = _load_yaml(CLOSURE_MAP_PATH)
    rows_by_id = {row["capability_row_id"]: row for row in closure_map["rows"]}
    required_row_fields = set(closure_map["required_row_fields"])

    assert set(rows_by_id) == {row["row_id"] for row in matrix["rows"]}
    for row in closure_map["rows"]:
        assert required_row_fields <= set(row)
        assert row["primary_owner_task_id"]
        assert row["row_family"] == row["capability_row_id"].split(":", 1)[0]
        assert row["row_family"]
        assert row["work_kind"] in closure_map["allowed_work_kinds"]
        assert row["closing_evidence_scope_ids"]


def test_no_capability_row_has_overlapping_primary_owners() -> None:
    closure_map = _load_yaml(CLOSURE_MAP_PATH)
    seen: set[str] = set()
    for row in closure_map["rows"]:
        capability_row_id = row["capability_row_id"]
        assert capability_row_id not in seen, f"duplicate primary owner for {capability_row_id}"
        seen.add(capability_row_id)


def test_every_task_in_master_plan_maps_to_at_least_one_capability_row() -> None:
    matrix = _load_yaml(FULL_VISION_MATRIX_PATH)
    closure_map = _load_yaml(CLOSURE_MAP_PATH)

    matrix_row_families = {row["capability_type"] for row in matrix["rows"]}
    closure_row_families = {row["row_family"] for row in closure_map["rows"]}

    assert set(closure_map["must_cover_row_families"]) == matrix_row_families
    assert closure_row_families == matrix_row_families
    assert closure_map["authority_reconciliation_ref"] == "docs/implementation/authority-reconciliation.yaml"
    assert (REPO_ROOT / closure_map["authority_reconciliation_ref"]).is_file()


def test_no_silent_substitution_is_declared_without_shared_provenance() -> None:
    closure_map = _load_yaml(CLOSURE_MAP_PATH)
    allowed_modes = set(closure_map["allowed_shared_provenance_modes"])

    for row in closure_map["rows"]:
        mode = row["shared_provenance_mode"]
        assert mode in allowed_modes
        if mode == "declared_shared":
            assert row.get("shared_provenance_id")
            assert len(set(row["closing_evidence_scope_ids"])) > 1
            continue
        assert not row.get("shared_provenance_id")
