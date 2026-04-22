from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from euclid.panel.partial_pooling import fit_partial_pooling_baseline


@dataclass(frozen=True)
class SharedStructurePanelResult:
    status: str
    shared_skeleton_terms: tuple[str, ...]
    global_parameters: Mapping[str, float]
    entity_local_parameters: Mapping[str, Mapping[str, float]]
    dispersion_penalty: float
    baseline_evidence_role: str
    claim_lane_ceiling: str
    universal_law_evidence_allowed: bool
    reason_codes: tuple[str, ...]
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "shared_structure_panel_evidence@1.0.0",
            "status": self.status,
            "shared_skeleton_terms": list(self.shared_skeleton_terms),
            "global_parameters": dict(self.global_parameters),
            "entity_local_parameters": {
                entity: dict(parameters)
                for entity, parameters in self.entity_local_parameters.items()
            },
            "dispersion_penalty": self.dispersion_penalty,
            "baseline_evidence_role": self.baseline_evidence_role,
            "claim_lane_ceiling": self.claim_lane_ceiling,
            "universal_law_evidence_allowed": self.universal_law_evidence_allowed,
            "reason_codes": list(self.reason_codes),
            "replay_identity": self.replay_identity,
        }


def discover_shared_structure_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    entity_field: str = "entity",
    target_field: str = "target",
    feature_fields: Sequence[str] = (),
    shared_skeleton_terms: Sequence[str] = (),
    entity_supports: Mapping[str, set[str] | frozenset[str]] | None = None,
    min_entities: int = 2,
) -> SharedStructurePanelResult:
    rows_tuple = tuple(rows)
    feature_tuple = tuple(str(field) for field in feature_fields)
    skeleton_terms = tuple(str(term) for term in shared_skeleton_terms)
    if _has_entity_leakage(
        rows_tuple,
        entity_field=entity_field,
        feature_fields=feature_tuple,
    ):
        return _result(
            status="rejected",
            shared_skeleton_terms=(),
            global_parameters={},
            entity_local_parameters={},
            dispersion_penalty=0.0,
            reason_codes=("entity_leakage_detected",),
        )

    baseline = fit_partial_pooling_baseline(
        rows_tuple,
        entity_field=entity_field,
        target_field=target_field,
        min_entities=min_entities,
    )
    if baseline.status != "baseline_fit":
        return _result(
            status="abstained",
            shared_skeleton_terms=(),
            global_parameters=baseline.global_parameters,
            entity_local_parameters=baseline.entity_local_parameters,
            dispersion_penalty=0.0,
            reason_codes=baseline.reason_codes,
        )
    if not skeleton_terms:
        return _result(
            status="abstained",
            shared_skeleton_terms=(),
            global_parameters=baseline.global_parameters,
            entity_local_parameters=baseline.entity_local_parameters,
            dispersion_penalty=0.0,
            reason_codes=("missing_shared_skeleton",),
        )
    if entity_supports is not None and _is_local_only_overfit(entity_supports):
        return _result(
            status="abstained",
            shared_skeleton_terms=(),
            global_parameters=baseline.global_parameters,
            entity_local_parameters=baseline.entity_local_parameters,
            dispersion_penalty=0.0,
            reason_codes=("local_only_overfit_candidate",),
        )

    dispersion_penalty = float(
        baseline.parameter_dispersion.get("intercept_offset_range", 0.0)
    )
    return _result(
        status="selected",
        shared_skeleton_terms=skeleton_terms,
        global_parameters=baseline.global_parameters,
        entity_local_parameters=baseline.entity_local_parameters,
        dispersion_penalty=dispersion_penalty,
        reason_codes=(),
    )


def _has_entity_leakage(
    rows: Sequence[Mapping[str, Any]],
    *,
    entity_field: str,
    feature_fields: Sequence[str],
) -> bool:
    for feature in feature_fields:
        lowered = feature.lower()
        if lowered == entity_field.lower() or "entity" in lowered:
            return True
    for row in rows:
        entity = str(row.get(entity_field, ""))
        if not entity:
            continue
        for feature in feature_fields:
            value = row.get(feature)
            if isinstance(value, str) and value == entity:
                return True
    return False


def _is_local_only_overfit(
    entity_supports: Mapping[str, set[str] | frozenset[str]],
) -> bool:
    supports = [set(support) for support in entity_supports.values()]
    if len(supports) < 2:
        return True
    shared = set.intersection(*supports)
    return not shared


def _result(
    *,
    status: str,
    shared_skeleton_terms: Sequence[str],
    global_parameters: Mapping[str, float],
    entity_local_parameters: Mapping[str, Mapping[str, float]],
    dispersion_penalty: float,
    reason_codes: tuple[str, ...],
) -> SharedStructurePanelResult:
    claim_lane_ceiling = (
        "predictive_within_declared_scope"
        if status == "selected"
        else "descriptive_structure"
    )
    identity_payload = {
        "claim_lane_ceiling": claim_lane_ceiling,
        "dispersion_penalty": dispersion_penalty,
        "entity_local_parameters": {
            entity: dict(parameters)
            for entity, parameters in entity_local_parameters.items()
        },
        "global_parameters": dict(global_parameters),
        "reason_codes": list(reason_codes),
        "shared_skeleton_terms": list(shared_skeleton_terms),
        "status": status,
    }
    return SharedStructurePanelResult(
        status=status,
        shared_skeleton_terms=tuple(shared_skeleton_terms),
        global_parameters=dict(global_parameters),
        entity_local_parameters={
            entity: dict(parameters)
            for entity, parameters in entity_local_parameters.items()
        },
        dispersion_penalty=float(round(float(dispersion_penalty), 12)),
        baseline_evidence_role="baseline_only",
        claim_lane_ceiling=claim_lane_ceiling,
        universal_law_evidence_allowed=False,
        reason_codes=reason_codes,
        replay_identity=f"shared-structure-panel:{_digest(identity_payload)}",
    )


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "SharedStructurePanelResult",
    "discover_shared_structure_panel",
]
