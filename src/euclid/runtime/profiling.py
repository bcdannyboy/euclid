from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from euclid.contracts.refs import TypedRef
from euclid.control_plane import SQLiteMetadataStore
from euclid.modules.replay import inspect_reproducibility_bundle
from euclid.operator_runtime._compat_runtime import (
    _build_registry,
    _load_catalog,
    _resolve_replay_paths_and_summary,
)

SEED_SENSITIVE_ARTIFACT_ROLES = (
    "candidate_or_abstention",
    "robustness_report",
    "run_result",
    "scorecard",
    "search_plan",
    "validation_scope",
)
SEED_SENSITIVE_SEED_SCOPES = (
    "perturbation",
    "search",
    "surrogate_generation",
)


@dataclass(frozen=True)
class RuntimeDeterminismSnapshot:
    output_root: Path
    request_id: str
    run_result_ref: TypedRef
    bundle_ref: TypedRef
    selected_family: str
    result_mode: str
    confirmatory_primary_score: float
    artifact_hashes: Mapping[str, str]
    seed_records: Mapping[str, str]
    manifest_hashes: Mapping[str, str]


@dataclass(frozen=True)
class RuntimeDeterminismComparison:
    identical: bool
    changed_artifact_roles: tuple[str, ...]
    changed_seed_scopes: tuple[str, ...]
    changed_manifest_refs: tuple[str, ...]
    unexpected_artifact_roles: tuple[str, ...]
    unexpected_seed_scopes: tuple[str, ...]
    unexpected_manifest_refs: tuple[str, ...]


def capture_operator_runtime_snapshot(
    *,
    output_root: Path | str,
    request_id: str | None = None,
) -> RuntimeDeterminismSnapshot:
    resolved_output_root = Path(output_root).resolve()
    paths, summary_payload = _resolve_replay_paths_and_summary(
        output_root=resolved_output_root,
        run_id=request_id,
    )
    catalog = _load_catalog()
    registry = _build_registry(catalog=catalog, paths=paths)
    bundle_ref = _typed_ref(summary_payload["bundle_ref"])
    replay_inspection = inspect_reproducibility_bundle(registry.resolve(bundle_ref))
    run_result_ref = _typed_ref(summary_payload["run_result_ref"])
    metadata_store = SQLiteMetadataStore(paths.metadata_store_path)
    manifest_hashes = {
        f"{record.schema_name}:{record.object_id}": record.content_hash
        for record in metadata_store.list_manifests_for_run(run_result_ref.object_id)
    }
    return RuntimeDeterminismSnapshot(
        output_root=resolved_output_root,
        request_id=_resolved_request_id(paths=paths, summary_payload=summary_payload),
        run_result_ref=run_result_ref,
        bundle_ref=replay_inspection.bundle_ref,
        selected_family=str(summary_payload["selected_family"]),
        result_mode=str(summary_payload["result_mode"]),
        confirmatory_primary_score=float(summary_payload["confirmatory_primary_score"]),
        artifact_hashes=_artifact_hashes(replay_inspection),
        seed_records=_seed_records(replay_inspection),
        manifest_hashes=manifest_hashes,
    )


def capture_demo_runtime_snapshot(
    *,
    output_root: Path | str,
    request_id: str | None = None,
) -> RuntimeDeterminismSnapshot:
    return capture_operator_runtime_snapshot(output_root=output_root, request_id=request_id)


def compare_runtime_determinism(
    first: RuntimeDeterminismSnapshot,
    second: RuntimeDeterminismSnapshot,
    *,
    stochastic_artifact_roles: Sequence[str] = (),
    stochastic_seed_scopes: Sequence[str] = (),
    stochastic_manifest_refs: Sequence[str] = (),
) -> RuntimeDeterminismComparison:
    changed_artifact_roles = _changed_keys(
        first.artifact_hashes,
        second.artifact_hashes,
    )
    changed_seed_scopes = _changed_keys(
        first.seed_records,
        second.seed_records,
    )
    changed_manifest_refs = _changed_keys(
        first.manifest_hashes,
        second.manifest_hashes,
    )
    allowed_artifact_roles = set(stochastic_artifact_roles)
    allowed_seed_scopes = set(stochastic_seed_scopes)
    allowed_manifest_refs = set(stochastic_manifest_refs)
    unexpected_artifact_roles = tuple(
        role for role in changed_artifact_roles if role not in allowed_artifact_roles
    )
    unexpected_seed_scopes = tuple(
        scope for scope in changed_seed_scopes if scope not in allowed_seed_scopes
    )
    unexpected_manifest_refs = tuple(
        ref for ref in changed_manifest_refs if ref not in allowed_manifest_refs
    )
    return RuntimeDeterminismComparison(
        identical=not (
            changed_artifact_roles or changed_seed_scopes or changed_manifest_refs
        ),
        changed_artifact_roles=changed_artifact_roles,
        changed_seed_scopes=changed_seed_scopes,
        changed_manifest_refs=changed_manifest_refs,
        unexpected_artifact_roles=unexpected_artifact_roles,
        unexpected_seed_scopes=unexpected_seed_scopes,
        unexpected_manifest_refs=unexpected_manifest_refs,
    )


def _artifact_hashes(inspection) -> dict[str, str]:
    return {
        record.artifact_role: record.sha256
        for record in inspection.artifact_hash_records
    }


def _seed_records(inspection) -> dict[str, str]:
    return {
        record.seed_scope: record.seed_value
        for record in inspection.seed_records
    }


def _changed_keys(
    first: Mapping[str, str],
    second: Mapping[str, str],
) -> tuple[str, ...]:
    keys = sorted(set(first) | set(second))
    return tuple(key for key in keys if first.get(key) != second.get(key))


def _load_summary(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("run summary must deserialize to a mapping")
    return payload


def _typed_ref(payload: object) -> TypedRef:
    if not isinstance(payload, Mapping):
        raise ValueError("typed ref payload must be a mapping")
    schema_name = payload.get("schema_name")
    object_id = payload.get("object_id")
    if not isinstance(schema_name, str) or not isinstance(object_id, str):
        raise ValueError("typed ref payload must include schema_name and object_id")
    return TypedRef(schema_name=schema_name, object_id=object_id)


def _resolved_request_id(*, paths, summary_payload: Mapping[str, object]) -> str:
    request_id = summary_payload.get("request_id")
    if isinstance(request_id, str) and request_id:
        return request_id
    return paths.sealed_run_root.name


__all__ = [
    "RuntimeDeterminismComparison",
    "RuntimeDeterminismSnapshot",
    "SEED_SENSITIVE_ARTIFACT_ROLES",
    "SEED_SENSITIVE_SEED_SCOPES",
    "capture_demo_runtime_snapshot",
    "capture_operator_runtime_snapshot",
    "compare_runtime_determinism",
]
