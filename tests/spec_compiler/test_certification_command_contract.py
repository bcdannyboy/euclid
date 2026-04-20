from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMAND_CONTRACT_PATH = (
    REPO_ROOT / "docs/implementation/certification-command-contract.yaml"
)
AUTHORITY_SNAPSHOT_PATH = REPO_ROOT / "docs/implementation/authority-snapshot.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), (
        "missing required file: " f"{path.relative_to(REPO_ROOT).as_posix()}"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_every_required_certification_command_has_declared_inputs_outputs_and_scope(
) -> None:
    payload = _load_yaml(COMMAND_CONTRACT_PATH)
    required_fields = set(payload["required_command_fields"])
    authority_snapshot = _load_yaml(AUTHORITY_SNAPSHOT_PATH)
    expected_run_order = 1

    for command in payload["commands"]:
        assert required_fields <= set(command)
        assert command["scope_ids"]
        assert isinstance(command["run_order"], int)
        assert command["run_order"] == expected_run_order
        assert command["authority_snapshot_id"] == authority_snapshot["authority_snapshot_id"]
        expected_run_order += 1


def test_clean_install_certification_declares_self_contained_build_toolchain() -> None:
    payload = _load_yaml(COMMAND_CONTRACT_PATH)
    assert payload["environment"]["clean_install_build_frontend"] == "build"
    assert payload["environment"]["clean_install_build_backend"] == (
        "setuptools.build_meta"
    )
    assert payload["environment"]["clean_install_runtime_dependency_source"] == (
        "local_wheelhouse_no_index"
    )
    clean_install = next(
        command
        for command in payload["commands"]
        if command["command_id"] == "clean_install_certification"
    )
    assert set(clean_install["build_toolchain_requirements"]) >= {
        "python3.11",
        "build",
        "setuptools>=69",
        "wheel",
    }
    assert any("no-index wheelhouse" in rule for rule in payload["rules"])


def test_replay_certification_uses_declared_run_id_derivation() -> None:
    payload = _load_yaml(COMMAND_CONTRACT_PATH)
    replay = next(
        command
        for command in payload["commands"]
        if command["command_id"] == "full_vision_operator_replay"
    )
    derivation = replay["run_id_derivation"]
    assert derivation["source_command_id"] == "full_vision_operator_run"
    assert derivation["input_artifact_id"] == (
        "build/certification/full_vision_run/run-result.json"
    )
    assert derivation["field_path"] == "run_id"


def test_required_certification_commands_cannot_be_skipped_or_xfailed() -> None:
    payload = _load_yaml(COMMAND_CONTRACT_PATH)
    assert payload["environment"]["zero_skip_policy"] == (
        "required_tests_must_not_skip_or_xfail"
    )
    assert any("skipped" in rule.lower() for rule in payload["rules"])
    assert any("xfail" in rule.lower() for rule in payload["rules"])
