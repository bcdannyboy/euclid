from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
COMMAND_CONTRACT_PATH = (
    REPO_ROOT / "docs" / "implementation" / "certification-command-contract.yaml"
)
ACTION_REF_PATTERN = re.compile(r"^\s*-\s+uses:\s+(?P<action>\S+)@(?P<ref>\S+)\s*$")
FULL_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_ci_executes_every_required_certification_command() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    command_contract = _load_yaml(COMMAND_CONTRACT_PATH)

    for command in command_contract["commands"]:
        if command["command_id"] == "full_vision_operator_replay":
            assert "--run-id full-vision-run" not in workflow
            assert "build/certification/full_vision_run/run-result.json" in workflow
            assert "FULL_VISION_RUN_ID" in workflow
            assert (
                'python3.11 -m euclid replay --run-id "$FULL_VISION_RUN_ID"'
                in workflow
            )
            assert (
                "--output-root build/certification/full_vision_run "
                "--evidence-report "
                "build/reports/full_vision_operator_replay_evidence.json"
            ) in workflow
            continue
        assert command["command"] in workflow

    assert 'python3.11 -m pip install -e ".[dev]"' not in workflow
    assert "astral-sh/setup-uv" in workflow
    assert "uv sync --locked" in workflow
    assert "build/reports/completion-report.json" in workflow
    assert "build/reports/clean-install-certification.json" in workflow
    assert "build/reports/research-readiness.json" in workflow


def test_ci_pins_external_actions_to_immutable_shas() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    action_refs = [
        match.groupdict()
        for line in workflow.splitlines()
        if (match := ACTION_REF_PATTERN.match(line))
    ]

    assert action_refs, "CI workflow must declare external action steps"
    mutable_refs = [
        f"{entry['action']}@{entry['ref']}"
        for entry in action_refs
        if not FULL_SHA_PATTERN.fullmatch(entry["ref"])
    ]
    assert mutable_refs == []
