from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACEABILITY_PATH = REPO_ROOT / "docs/implementation/subtask-test-traceability.yaml"
CLOSURE_MAP_PATH = REPO_ROOT / "docs/implementation/euclid-closure-map.yaml"
COMMAND_CONTRACT_PATH = REPO_ROOT / "docs/implementation/certification-command-contract.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), (
        "missing required file: " f"{path.relative_to(REPO_ROOT).as_posix()}"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _rows_by_entry(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["task_id"]), str(row["subtask_id"])): row
        for row in rows
    }


def _strip_backticks(value: str) -> str:
    return value.strip("`")


def test_every_task_and_subtask_has_traceability_rows() -> None:
    payload = _load_yaml(TRACEABILITY_PATH)
    entries = _rows_by_entry(payload["rows"])
    required_row_fields = set(payload["required_row_fields"])

    assert len(entries) == len(payload["rows"])
    for row in payload["rows"]:
        assert required_row_fields <= set(row)
        assert row["capability_row_ids"]

    for task_id in {str(row["task_id"]) for row in payload["rows"]}:
        assert (task_id, "__task__") in entries


def test_every_subtask_declares_all_required_test_classes_or_justified_not_applicable(
) -> None:
    payload = _load_yaml(TRACEABILITY_PATH)
    allowed_reasons = set(payload["allowed_not_applicable_reasons"])
    expected_classes = set(payload["required_test_classes"])

    for row in payload["rows"]:
        declared_classes = set(row["required_test_classes"])
        assert declared_classes == expected_classes
        for test_class in expected_classes:
            status = row["required_test_classes"][test_class]
            if status == "required":
                assert row["named_test_ids"][test_class]
                continue
            assert status == "not_applicable"
            justification = row["not_applicable_justifications"][test_class]
            assert justification["reason_code"] in allowed_reasons
            assert justification["note"]


def test_every_traceability_row_points_to_named_tests_and_blocking_artifacts() -> None:
    payload = _load_yaml(TRACEABILITY_PATH)
    for row in payload["rows"]:
        assert row["blocking_artifact_ids"]
        for artifact_id in row["blocking_artifact_ids"]:
            assert isinstance(artifact_id, str) and artifact_id
        for test_class, test_ids in row["named_test_ids"].items():
            if row["required_test_classes"][test_class] == "required":
                assert test_ids
            for test_id in test_ids:
                normalized = _strip_backticks(test_id)
                relative_path, _, node_id = normalized.partition("::")
                assert relative_path and node_id, f"traceability test id must be file::node: {test_id}"
                assert relative_path.startswith("tests/")
                assert relative_path.endswith(".py")


def test_traceability_ledger_and_closure_map_remain_consistent() -> None:
    traceability = _load_yaml(TRACEABILITY_PATH)
    closure_map = _load_yaml(CLOSURE_MAP_PATH)
    capability_row_ids = {row["capability_row_id"] for row in closure_map["rows"]}
    task_to_rows: dict[str, set[str]] = {}
    for row in closure_map["rows"]:
        task_to_rows.setdefault(str(row["primary_owner_task_id"]), set()).add(
            str(row["capability_row_id"])
        )
        for task_id in row["supporting_task_ids"]:
            task_to_rows.setdefault(str(task_id), set()).add(
                str(row["capability_row_id"])
            )

    for row in traceability["rows"]:
        assert set(row["capability_row_ids"]) <= capability_row_ids
        assert set(row["capability_row_ids"]) <= task_to_rows[str(row["task_id"])]


def test_required_certification_tests_cannot_be_waived_by_skip_or_xfail() -> None:
    traceability = _load_yaml(TRACEABILITY_PATH)
    command_contract = _load_yaml(COMMAND_CONTRACT_PATH)
    command_ids = {row["command_id"] for row in command_contract["commands"]}

    for row in traceability["rows"]:
        if row["required_command_ids"]:
            assert set(row["required_command_ids"]) <= command_ids
        for test_class, status in row["required_test_classes"].items():
            if status == "required":
                assert row["named_test_ids"][test_class]
