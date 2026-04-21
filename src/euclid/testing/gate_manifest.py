from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class GateManifestError(ValueError):
    """Raised when a phase gate manifest is incomplete or malformed."""


@dataclass(frozen=True)
class CommandGate:
    commands: tuple[str, ...]


@dataclass(frozen=True)
class AssertionGate:
    assertions: tuple[str, ...]


@dataclass(frozen=True)
class EdgeCaseGate:
    required: tuple[str, ...]


@dataclass(frozen=True)
class PhaseGateManifest:
    phase_id: str
    covered_ids: tuple[str, ...]
    fixture_unit: CommandGate
    fixture_integration: CommandGate
    fixture_regression: CommandGate
    live_api: CommandGate
    redaction: AssertionGate
    replay: AssertionGate
    claim_scope: AssertionGate
    edge_cases: EdgeCaseGate


def extract_plan_phase_ids(path: Path, *, phase_id: str) -> tuple[str, ...]:
    if not path.is_file():
        raise GateManifestError(f"plan does not exist: {path}")
    task_pattern = re.compile(rf"^### ({re.escape(phase_id)}-T\d+):")
    subtask_pattern = re.compile(rf"`({re.escape(phase_id)}-T\d+-S\d+)`")
    next_phase_pattern = re.compile(r"^## 26\.\d+ P\d+")
    in_phase = False
    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## 26.") and phase_id in line:
            in_phase = True
            continue
        if in_phase and next_phase_pattern.match(line) and phase_id not in line:
            break
        if not in_phase:
            continue
        matches = []
        task_match = task_pattern.match(line)
        if task_match:
            matches.append(task_match.group(1))
        matches.extend(subtask_pattern.findall(line))
        for item in matches:
            if item not in seen:
                seen.add(item)
                ids.append(item)
    if not ids:
        raise GateManifestError(f"no task or subtask ids found for {phase_id} in {path}")
    return tuple(ids)


def load_gate_manifest(
    path: Path,
    *,
    required_ids: Sequence[str] = (),
) -> PhaseGateManifest:
    if not path.is_file():
        raise GateManifestError(f"gate manifest does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise GateManifestError(f"gate manifest must be a mapping: {path}")

    manifest = PhaseGateManifest(
        phase_id=_required_string(payload, "phase_id"),
        covered_ids=_required_string_tuple(payload, "covered_ids"),
        fixture_unit=CommandGate(
            commands=_required_nested_string_tuple(payload, "fixture_unit", "commands")
        ),
        fixture_integration=CommandGate(
            commands=_required_nested_string_tuple(
                payload,
                "fixture_integration",
                "commands",
            )
        ),
        fixture_regression=CommandGate(
            commands=_required_nested_string_tuple(
                payload,
                "fixture_regression",
                "commands",
            )
        ),
        live_api=CommandGate(
            commands=_required_nested_string_tuple(payload, "live_api", "commands")
        ),
        redaction=AssertionGate(
            assertions=_required_nested_string_tuple(
                payload,
                "redaction",
                "assertions",
            )
        ),
        replay=AssertionGate(
            assertions=_required_nested_string_tuple(payload, "replay", "assertions")
        ),
        claim_scope=AssertionGate(
            assertions=_required_nested_string_tuple(
                payload,
                "claim_scope",
                "assertions",
            )
        ),
        edge_cases=EdgeCaseGate(
            required=_required_nested_string_tuple(payload, "edge_cases", "required")
        ),
    )
    missing = sorted(set(required_ids) - set(manifest.covered_ids))
    if missing:
        raise GateManifestError(
            f"{path} is missing required task/subtask ids: {', '.join(missing)}"
        )
    return manifest


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GateManifestError(f"gate manifest missing non-empty {key}")
    return value.strip()


def _required_string_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise GateManifestError(f"gate manifest missing non-empty list {key}")
    items = tuple(str(item).strip() for item in value if str(item).strip())
    if len(items) != len(value):
        raise GateManifestError(f"gate manifest contains blank item in {key}")
    return items


def _required_nested_string_tuple(
    payload: Mapping[str, Any],
    key: str,
    nested_key: str,
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise GateManifestError(f"gate manifest missing section {key}")
    return _required_string_tuple(value, nested_key)


__all__ = [
    "AssertionGate",
    "CommandGate",
    "EdgeCaseGate",
    "extract_plan_phase_ids",
    "GateManifestError",
    "PhaseGateManifest",
    "load_gate_manifest",
]
