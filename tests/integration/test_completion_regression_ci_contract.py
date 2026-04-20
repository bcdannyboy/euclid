from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
COMMAND_CONTRACT_PATH = (
    REPO_ROOT / "docs" / "implementation" / "certification-command-contract.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_ci_executes_every_required_certification_command() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    command_contract = _load_yaml(COMMAND_CONTRACT_PATH)

    for command in command_contract["commands"]:
        if command["command_id"] == "full_vision_operator_replay":
            assert "python3.11 -m euclid replay --run-id full-vision-run" in workflow
            assert (
                "--output-root build/certification/full_vision_run "
                "--evidence-report "
                "build/reports/full_vision_operator_replay_evidence.json"
            ) in workflow
            continue
        assert command["command"] in workflow

    assert "build/reports/completion-report.json" in workflow
    assert "build/reports/clean-install-certification.json" in workflow
    assert "build/reports/research-readiness.json" in workflow
