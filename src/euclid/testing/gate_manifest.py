from __future__ import annotations

import re
import shlex
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


REQUIRED_ID_GATE_FIELDS = (
    "status",
    "implementation_files",
    "test_files",
    "gate_refs",
    "evidence_refs",
    "edge_cases",
    "redaction_assertions",
    "replay_assertions",
    "claim_scope_assertions",
)


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
    id_gates_schema: tuple[str, ...]
    id_gates: Mapping[str, Mapping[str, Any]]


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
    project_root: Path | None = None,
    validate_references: bool = False,
) -> PhaseGateManifest:
    if not path.is_file():
        raise GateManifestError(f"gate manifest does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise GateManifestError(f"gate manifest must be a mapping: {path}")

    phase_id = _required_string(payload, "phase_id")
    covered_ids = _required_string_tuple(payload, "covered_ids")
    missing = sorted(set(required_ids) - set(covered_ids))
    if missing:
        raise GateManifestError(
            f"{path} is missing required task/subtask ids: {', '.join(missing)}"
        )
    id_gates_schema = _load_id_gates_schema(payload, required_ids=required_ids)
    id_gates = _load_id_gates(
        payload,
        required_ids=required_ids,
        schema=id_gates_schema,
        path=path,
        project_root=project_root,
        validate_references=validate_references,
    )
    if validate_references:
        _validate_manifest_commands(payload, path=path, project_root=project_root)
    manifest = PhaseGateManifest(
        phase_id=phase_id,
        covered_ids=covered_ids,
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
        id_gates_schema=id_gates_schema,
        id_gates=id_gates,
    )
    return manifest


def _load_id_gates_schema(
    payload: Mapping[str, Any],
    *,
    required_ids: Sequence[str],
) -> tuple[str, ...]:
    if not required_ids and "id_gates_schema" not in payload:
        return ()
    schema = _required_string_tuple(payload, "id_gates_schema")
    missing_fields = sorted(set(REQUIRED_ID_GATE_FIELDS) - set(schema))
    if missing_fields:
        raise GateManifestError(
            "gate manifest id_gates_schema missing required fields: "
            + ", ".join(missing_fields)
        )
    return schema


def _load_id_gates(
    payload: Mapping[str, Any],
    *,
    required_ids: Sequence[str],
    schema: Sequence[str],
    path: Path,
    project_root: Path | None,
    validate_references: bool,
) -> Mapping[str, Mapping[str, Any]]:
    if not required_ids and "id_gates" not in payload:
        return {}
    value = payload.get("id_gates")
    if not isinstance(value, Mapping) or not value:
        raise GateManifestError(f"gate manifest missing non-empty id_gates: {path}")
    normalized: dict[str, Mapping[str, Any]] = {}
    for key, row in value.items():
        if not isinstance(row, Mapping):
            raise GateManifestError(f"gate manifest id_gates row must be mapping: {key}")
        row_key = str(key).strip()
        if not row_key:
            raise GateManifestError("gate manifest contains blank id_gates key")
        normalized[row_key] = row
    missing_ids = sorted(set(required_ids) - set(normalized))
    if missing_ids:
        raise GateManifestError(
            f"{path} is missing id_gates rows for: {', '.join(missing_ids)}"
        )
    for item_id in required_ids:
        row = normalized[str(item_id)]
        for field in schema:
            if field not in row:
                raise GateManifestError(f"{path} {item_id} missing id_gates.{field}")
            if _is_blank_gate_value(row[field]):
                raise GateManifestError(f"{path} {item_id} has blank id_gates.{field}")
        _validate_specific_gate_refs(path=path, item_id=str(item_id), row=row)
        if validate_references:
            _validate_row_references(
                path=path,
                item_id=str(item_id),
                row=row,
                project_root=project_root,
            )
    return normalized


def _validate_specific_gate_refs(
    *,
    path: Path,
    item_id: str,
    row: Mapping[str, Any],
) -> None:
    gate_refs = tuple(str(ref).strip() for ref in row.get("gate_refs", ()))
    if not any(ref.endswith(f"#{item_id}") for ref in gate_refs):
        if "-S" in item_id:
            parent_id = item_id.rsplit("-S", 1)[0]
            if any(ref.endswith(f"#{parent_id}") for ref in gate_refs):
                raise GateManifestError(
                    f"{path} {item_id} uses generic parent gate {parent_id}"
                )
        raise GateManifestError(f"{path} {item_id} missing specific gate ref")


def _validate_row_references(
    *,
    path: Path,
    item_id: str,
    row: Mapping[str, Any],
    project_root: Path | None,
) -> None:
    root = _reference_root(path, project_root)
    for field in ("implementation_files", "test_files"):
        for reference in row.get(field, ()):
            _require_existing_reference_path(
                root,
                str(reference),
                context=f"{path} {item_id} {field}",
            )
    for reference in row.get("gate_refs", ()):
        _require_existing_reference_path(
            root,
            str(reference).split("#", 1)[0],
            context=f"{path} {item_id} gate_refs",
        )
    for reference in row.get("evidence_refs", ()):
        prefix, value = _split_evidence_reference(str(reference))
        if prefix in {"test", "gate", "doc"}:
            _require_existing_reference_path(
                root,
                value.split("#", 1)[0],
                context=f"{path} {item_id} evidence_refs",
            )
        elif prefix == "artifact" and not value.startswith("build/"):
            _require_existing_reference_path(
                root,
                value,
                context=f"{path} {item_id} evidence_refs",
            )


def _validate_manifest_commands(
    payload: Mapping[str, Any],
    *,
    path: Path,
    project_root: Path | None,
) -> None:
    root = _reference_root(path, project_root)
    for section in ("fixture_unit", "fixture_integration", "fixture_regression", "live_api"):
        commands = payload.get(section, {}).get("commands", ())
        for command in commands:
            _validate_command_references(
                str(command),
                path=path,
                section=section,
                root=root,
            )


def _validate_command_references(
    command: str,
    *,
    path: Path,
    section: str,
    root: Path,
) -> None:
    tokens = tuple(shlex.split(command))
    if not tokens:
        raise GateManifestError(f"{path} {section} contains blank command")
    for token in tokens:
        if token.startswith("-") or "=" in token and not token.startswith(("tests/", "docs/", "src/", "schemas/", "scripts/", "fixtures/", "examples/")):
            continue
        if token.startswith(("./", "tests/", "docs/", "src/", "schemas/", "scripts/", "fixtures/", "examples/")):
            _require_existing_reference_path(
                root,
                token[2:] if token.startswith("./") else token,
                context=f"{path} {section} command",
            )


def _reference_root(path: Path, project_root: Path | None) -> Path:
    if project_root is not None:
        return project_root
    resolved = path.resolve()
    for parent in resolved.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return path.parent


def _split_evidence_reference(reference: str) -> tuple[str, str]:
    if ":" not in reference:
        return ("path", reference)
    prefix, value = reference.split(":", 1)
    return prefix, value


def _require_existing_reference_path(
    root: Path,
    reference: str,
    *,
    context: str,
) -> None:
    normalized = reference.strip()
    if not normalized:
        raise GateManifestError(f"{context} contains blank reference")
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = root / candidate
    if any(character in normalized for character in "*?["):
        if list(root.glob(normalized)):
            return
        raise GateManifestError(f"{context} references unmatched glob: {normalized}")
    if not candidate.exists():
        raise GateManifestError(f"{context} references missing path: {normalized}")


def _is_blank_gate_value(value: Any) -> bool:
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        return not value
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return len(value) == 0
    return value is None


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
    "REQUIRED_ID_GATE_FIELDS",
    "PhaseGateManifest",
    "load_gate_manifest",
]
