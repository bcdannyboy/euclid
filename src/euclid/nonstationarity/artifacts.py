from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from euclid.contracts.errors import ContractValidationError

NONSTATIONARITY_STATUSES = (
    "passed",
    "failed",
    "abstained",
    "adapter_unavailable",
    "not_evaluated",
)

NONSTATIONARITY_REASON_CODES = frozenset(
    {
        "changepoint_detection_not_run",
        "changepoint_min_segment_size_not_met",
        "cusum_instability_detected",
        "insufficient_observations_for_stability_diagnostic",
        "instability_unhandled_by_nonstationary_lane",
        "ruptures_changepoint_backend_unavailable",
        "stability_diagnostic_not_run",
        "statsmodels_stability_backend_unavailable",
    }
)

ADAPTER_UNAVAILABLE_REASON_CODES = frozenset(
    {
        "ruptures_changepoint_backend_unavailable",
        "statsmodels_stability_backend_unavailable",
    }
)

DEFAULT_CLAIM_SCOPE = "diagnostic_evidence_only"


def _reason_codes(value: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_codes = (value,)
    else:
        raw_codes = tuple(value)

    seen: dict[str, None] = {}
    for index, reason_code in enumerate(raw_codes):
        if not isinstance(reason_code, str) or not reason_code.strip():
            raise ContractValidationError(
                code="empty_nonstationarity_reason_code",
                message="nonstationarity reason codes must be non-empty strings",
                field_path=f"reason_codes[{index}]",
            )
        if reason_code not in NONSTATIONARITY_REASON_CODES:
            raise ContractValidationError(
                code="unknown_nonstationarity_reason_code",
                message=f"unknown nonstationarity reason code {reason_code!r}",
                field_path=f"reason_codes[{index}]",
                details={"reason_code": reason_code},
            )
        seen.setdefault(reason_code, None)
    return tuple(seen)


def _stable_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _stable_value(mapping[key]) for key in sorted(mapping)}


def _stable_value(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, Mapping):
        return _stable_mapping(value)
    if isinstance(value, tuple):
        return [_stable_value(item) for item in value]
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    return value


def _artifact_ref(ref: Any) -> Any:
    if hasattr(ref, "as_dict"):
        return ref.as_dict()
    if isinstance(ref, Mapping):
        return _stable_mapping(ref)
    return ref


@dataclass(frozen=True)
class NonstationarityStatus:
    status: str
    reason_codes: tuple[str, ...] = ()
    evidence_refs: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in NONSTATIONARITY_STATUSES:
            raise ContractValidationError(
                code="unknown_nonstationarity_status",
                message=f"unknown nonstationarity status {self.status!r}",
                field_path="status",
                details={"status": self.status},
            )

        normalized_codes = _reason_codes(self.reason_codes)
        if self.status == "passed" and normalized_codes:
            raise ContractValidationError(
                code="passed_nonstationarity_status_has_reason_codes",
                message="passed nonstationarity status must not carry blockers",
                field_path="reason_codes",
            )
        if self.status != "passed" and not normalized_codes:
            raise ContractValidationError(
                code="missing_nonstationarity_reason_code",
                message=(
                    "non-passed nonstationarity statuses require a specific "
                    "reason code"
                ),
                field_path="reason_codes",
            )
        if self.status == "adapter_unavailable":
            unexpected = [
                code
                for code in normalized_codes
                if code not in ADAPTER_UNAVAILABLE_REASON_CODES
            ]
            if unexpected:
                raise ContractValidationError(
                    code="non_adapter_reason_code_for_adapter_unavailable",
                    message=(
                        "adapter_unavailable statuses require adapter backend "
                        "reason codes"
                    ),
                    field_path="reason_codes",
                    details={"reason_code": unexpected[0]},
                )

        object.__setattr__(self, "reason_codes", normalized_codes)
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def passed(
        cls,
        *,
        evidence_refs: Sequence[Any] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "NonstationarityStatus":
        return cls(
            status="passed",
            evidence_refs=tuple(evidence_refs),
            metadata=metadata or {},
        )

    @classmethod
    def failed(
        cls,
        reason_codes: Sequence[str] | str,
        *,
        evidence_refs: Sequence[Any] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "NonstationarityStatus":
        return cls(
            status="failed",
            reason_codes=_reason_codes(reason_codes),
            evidence_refs=tuple(evidence_refs),
            metadata=metadata or {},
        )

    @classmethod
    def abstained(
        cls,
        reason_codes: Sequence[str] | str,
        *,
        evidence_refs: Sequence[Any] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "NonstationarityStatus":
        return cls(
            status="abstained",
            reason_codes=_reason_codes(reason_codes),
            evidence_refs=tuple(evidence_refs),
            metadata=metadata or {},
        )

    @classmethod
    def adapter_unavailable(
        cls,
        reason_codes: Sequence[str] | str,
        *,
        evidence_refs: Sequence[Any] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "NonstationarityStatus":
        return cls(
            status="adapter_unavailable",
            reason_codes=_reason_codes(reason_codes),
            evidence_refs=tuple(evidence_refs),
            metadata=metadata or {},
        )

    @classmethod
    def not_evaluated(
        cls,
        reason_codes: Sequence[str] | str,
        *,
        evidence_refs: Sequence[Any] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "NonstationarityStatus":
        return cls(
            status="not_evaluated",
            reason_codes=_reason_codes(reason_codes),
            evidence_refs=tuple(evidence_refs),
            metadata=metadata or {},
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "evidence_refs": [_artifact_ref(ref) for ref in self.evidence_refs],
            "metadata": _stable_mapping(self.metadata),
        }


@dataclass(frozen=True)
class StabilityDiagnosticArtifact:
    diagnostic_id: str
    series_id: str
    method: str
    statistic_name: str
    statistic_value: float | None
    p_value: float | None
    critical_value: float | None
    status: NonstationarityStatus
    window_start: int | None
    window_end: int | None
    sample_count: int
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    claim_scope: str = DEFAULT_CLAIM_SCOPE
    law_claim_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_manifest(self) -> dict[str, Any]:
        status_manifest = self.status.as_manifest()
        return {
            "schema_name": "stability_diagnostic_artifact@1.0.0",
            "artifact_type": "stability_diagnostic",
            "diagnostic_id": self.diagnostic_id,
            "series_id": self.series_id,
            "method": self.method,
            "statistic_name": self.statistic_name,
            "statistic_value": self.statistic_value,
            "p_value": self.p_value,
            "critical_value": self.critical_value,
            "status": status_manifest["status"],
            "reason_codes": status_manifest["reason_codes"],
            "evidence_refs": status_manifest["evidence_refs"],
            "window_start": self.window_start,
            "window_end": self.window_end,
            "sample_count": self.sample_count,
            "parameters": _stable_mapping(self.parameters),
            "metadata": _stable_mapping(self.metadata),
            "claim_scope": self.claim_scope,
            "law_claim_allowed": self.law_claim_allowed,
        }


@dataclass(frozen=True)
class ChangePointArtifact:
    artifact_id: str
    series_id: str
    method: str
    detected_points: tuple[int, ...]
    penalty: float | None
    min_segment_size: int
    tolerance: int
    status: NonstationarityStatus
    sample_count: int
    cost_model: str = "unknown"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    claim_scope: str = DEFAULT_CLAIM_SCOPE
    law_claim_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "detected_points",
            tuple(sorted({int(point) for point in self.detected_points})),
        )
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_manifest(self) -> dict[str, Any]:
        status_manifest = self.status.as_manifest()
        return {
            "schema_name": "change_point_artifact@1.0.0",
            "artifact_type": "change_point",
            "artifact_id": self.artifact_id,
            "series_id": self.series_id,
            "method": self.method,
            "detected_points": list(self.detected_points),
            "penalty": self.penalty,
            "min_segment_size": self.min_segment_size,
            "tolerance": self.tolerance,
            "status": status_manifest["status"],
            "reason_codes": status_manifest["reason_codes"],
            "evidence_refs": status_manifest["evidence_refs"],
            "sample_count": self.sample_count,
            "cost_model": self.cost_model,
            "parameters": _stable_mapping(self.parameters),
            "metadata": _stable_mapping(self.metadata),
            "claim_scope": self.claim_scope,
            "law_claim_allowed": self.law_claim_allowed,
        }


__all__ = [
    "ADAPTER_UNAVAILABLE_REASON_CODES",
    "DEFAULT_CLAIM_SCOPE",
    "NONSTATIONARITY_REASON_CODES",
    "NONSTATIONARITY_STATUSES",
    "ChangePointArtifact",
    "NonstationarityStatus",
    "StabilityDiagnosticArtifact",
]
