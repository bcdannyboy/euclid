from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from euclid.cir.models import CandidateIntermediateRepresentation


@dataclass(frozen=True)
class ComparableBackendFinalist:
    candidate_id: str
    candidate_hash: str
    backend_family: str
    adapter_id: str
    adapter_class: str
    forecast_object_type: str
    total_code_bits: float
    description_gain_bits: float
    structure_code_bits: float
    canonical_byte_length: int
    coverage_statement: str | None = None
    exactness_ceiling: str | None = None
    scope_declaration: str | None = None
    provenance_id: str | None = None
    replay_contract: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "backend_family": self.backend_family,
            "family_id": self.backend_family,
            "adapter_id": self.adapter_id,
            "adapter_class": self.adapter_class,
            "forecast_object_type": self.forecast_object_type,
            "total_code_bits": self.total_code_bits,
            "description_gain_bits": self.description_gain_bits,
            "structure_code_bits": self.structure_code_bits,
            "canonical_byte_length": self.canonical_byte_length,
            "replay_contract": dict(self.replay_contract),
        }
        if self.coverage_statement is not None:
            payload["coverage_statement"] = self.coverage_statement
        if self.exactness_ceiling is not None:
            payload["exactness_ceiling"] = self.exactness_ceiling
        if self.scope_declaration is not None:
            payload["scope_declaration"] = self.scope_declaration
        if self.provenance_id is not None:
            payload["provenance_id"] = self.provenance_id
            payload["submitter_id"] = self.provenance_id
        return payload


def normalize_cir_finalist(
    candidate: CandidateIntermediateRepresentation,
    *,
    total_code_bits: float,
    description_gain_bits: float,
    structure_code_bits: float,
    coverage_statement: str | None = None,
    exactness_ceiling: str | None = None,
    scope_declaration: str | None = None,
    provenance_id: str | None = None,
    replay_contract: Mapping[str, Any] | None = None,
) -> ComparableBackendFinalist:
    origin = candidate.evidence_layer.backend_origin_record
    return ComparableBackendFinalist(
        candidate_id=origin.source_candidate_id,
        candidate_hash=candidate.canonical_hash(),
        backend_family=(
            origin.backend_family or candidate.structural_layer.cir_family_id
        ),
        adapter_id=origin.adapter_id,
        adapter_class=origin.adapter_class,
        forecast_object_type=(
            candidate.execution_layer.forecast_operator.forecast_object_type
        ),
        total_code_bits=float(total_code_bits),
        description_gain_bits=float(description_gain_bits),
        structure_code_bits=float(structure_code_bits),
        canonical_byte_length=len(candidate.canonical_bytes()),
        coverage_statement=coverage_statement,
        exactness_ceiling=exactness_ceiling,
        scope_declaration=scope_declaration,
        provenance_id=provenance_id,
        replay_contract=dict(replay_contract or {}),
    )


def normalize_submitter_finalist(
    result: Any,
) -> ComparableBackendFinalist | None:
    if (
        result.status != "selected"
        or result.selected_candidate_metrics is None
        or result.selected_candidate is None
    ):
        return None
    metrics = dict(result.selected_candidate_metrics)
    total_code_bits = _finite_metric(metrics, "total_code_bits")
    description_gain_bits = _finite_metric(metrics, "description_gain_bits")
    structure_code_bits = _finite_metric(metrics, "structure_code_bits")
    canonical_byte_length = _positive_int_metric(metrics, "canonical_byte_length")
    if (
        total_code_bits is None
        or description_gain_bits is None
        or structure_code_bits is None
        or canonical_byte_length is None
    ):
        return None
    candidate = result.selected_candidate
    origin = candidate.evidence_layer.backend_origin_record
    return ComparableBackendFinalist(
        candidate_id=str(result.selected_candidate_id),
        candidate_hash=str(result.selected_candidate_hash),
        backend_family=(
            origin.backend_family or candidate.structural_layer.cir_family_id
        ),
        adapter_id=origin.adapter_id,
        adapter_class=origin.adapter_class,
        forecast_object_type=(
            candidate.execution_layer.forecast_operator.forecast_object_type
        ),
        total_code_bits=total_code_bits,
        description_gain_bits=description_gain_bits,
        structure_code_bits=structure_code_bits,
        canonical_byte_length=canonical_byte_length,
        provenance_id=result.submitter_id,
        replay_contract=dict(result.replay_contract),
    )


def _finite_metric(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _positive_int_metric(metrics: Mapping[str, Any], key: str) -> int | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


__all__ = [
    "ComparableBackendFinalist",
    "normalize_cir_finalist",
    "normalize_submitter_finalist",
]
