from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PartialPoolingResult:
    status: str
    global_parameters: Mapping[str, float]
    entity_local_parameters: Mapping[str, Mapping[str, float]]
    pooling_strength: float
    parameter_dispersion: Mapping[str, float]
    evidence_role: str
    claim_lane_ceiling: str
    universal_law_evidence_allowed: bool
    reason_codes: tuple[str, ...]
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "partial_pooling_panel_baseline@1.0.0",
            "status": self.status,
            "global_parameters": dict(self.global_parameters),
            "entity_local_parameters": {
                entity: dict(parameters)
                for entity, parameters in self.entity_local_parameters.items()
            },
            "pooling_strength": self.pooling_strength,
            "parameter_dispersion": dict(self.parameter_dispersion),
            "evidence_role": self.evidence_role,
            "claim_lane_ceiling": self.claim_lane_ceiling,
            "universal_law_evidence_allowed": self.universal_law_evidence_allowed,
            "reason_codes": list(self.reason_codes),
            "replay_identity": self.replay_identity,
        }


def fit_partial_pooling_baseline(
    rows: Sequence[Mapping[str, Any]],
    *,
    entity_field: str = "entity",
    target_field: str = "target",
    min_entities: int = 2,
    min_rows_per_entity: int = 1,
    ridge_strength: float = 1.0,
) -> PartialPoolingResult:
    grouped = _group_targets(
        rows,
        entity_field=entity_field,
        target_field=target_field,
    )
    if len(grouped) < int(min_entities):
        return _result(
            status="abstained",
            global_parameters={},
            entity_local_parameters={},
            pooling_strength=0.0,
            parameter_dispersion={},
            reason_codes=("insufficient_entities",),
        )
    if any(len(values) < int(min_rows_per_entity) for values in grouped.values()):
        return _result(
            status="abstained",
            global_parameters={},
            entity_local_parameters={},
            pooling_strength=0.0,
            parameter_dispersion={},
            reason_codes=("insufficient_entity_rows",),
        )

    all_targets = [value for values in grouped.values() for value in values]
    global_intercept = _stable_float(fmean(all_targets))
    first_count = len(next(iter(grouped.values())))
    pooling_strength = _stable_float(
        first_count / (first_count + max(float(ridge_strength), 0.0))
    )
    entity_local_parameters: dict[str, dict[str, float]] = {}
    offsets: list[float] = []
    for entity, values in sorted(grouped.items()):
        raw_offset = fmean(values) - global_intercept
        offset = _stable_float(raw_offset * pooling_strength)
        offsets.append(offset)
        entity_local_parameters[entity] = {"intercept_offset": offset}
    dispersion = {
        "intercept_offset_range": _stable_float(max(offsets) - min(offsets)),
    }
    return _result(
        status="baseline_fit",
        global_parameters={"global_intercept": global_intercept},
        entity_local_parameters=entity_local_parameters,
        pooling_strength=pooling_strength,
        parameter_dispersion=dispersion,
        reason_codes=(),
    )


def _group_targets(
    rows: Sequence[Mapping[str, Any]],
    *,
    entity_field: str,
    target_field: str,
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        entity = str(row.get(entity_field, "")).strip()
        if not entity:
            continue
        try:
            target = float(row[target_field])
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(entity, []).append(target)
    return grouped


def _result(
    *,
    status: str,
    global_parameters: Mapping[str, float],
    entity_local_parameters: Mapping[str, Mapping[str, float]],
    pooling_strength: float,
    parameter_dispersion: Mapping[str, float],
    reason_codes: tuple[str, ...],
) -> PartialPoolingResult:
    claim_lane_ceiling = (
        "predictive_within_declared_scope"
        if status == "baseline_fit"
        else "descriptive_structure"
    )
    identity_payload = {
        "entity_local_parameters": {
            entity: dict(parameters)
            for entity, parameters in entity_local_parameters.items()
        },
        "global_parameters": dict(global_parameters),
        "parameter_dispersion": dict(parameter_dispersion),
        "pooling_strength": pooling_strength,
        "reason_codes": list(reason_codes),
        "status": status,
    }
    return PartialPoolingResult(
        status=status,
        global_parameters=dict(global_parameters),
        entity_local_parameters={
            entity: dict(parameters)
            for entity, parameters in entity_local_parameters.items()
        },
        pooling_strength=pooling_strength,
        parameter_dispersion=dict(parameter_dispersion),
        evidence_role="baseline_only",
        claim_lane_ceiling=claim_lane_ceiling,
        universal_law_evidence_allowed=False,
        reason_codes=reason_codes,
        replay_identity=f"partial-pooling:{_digest(identity_payload)}",
    )


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


__all__ = [
    "PartialPoolingResult",
    "fit_partial_pooling_baseline",
]
