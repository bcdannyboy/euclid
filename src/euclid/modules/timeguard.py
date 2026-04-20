from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.modules.snapshotting import FrozenDatasetSnapshot


@dataclass(frozen=True)
class TimeSafetyCoordinateAudit:
    coordinate_index: int
    event_time: str
    available_at: str
    payload_hash: str
    status: str
    entity: str | None = None
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "available_at": self.available_at,
            "coordinate_index": self.coordinate_index,
            "event_time": self.event_time,
            "payload_hash": self.payload_hash,
            "reason_codes": list(self.reason_codes),
            "status": self.status,
        }
        if self.entity is not None:
            body["entity"] = self.entity
        return body


@dataclass(frozen=True)
class CanaryFailure:
    canary_id: str
    stage_id: str
    coordinate_index: int
    event_time: str
    available_at: str
    reason_code: str
    entity: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "available_at": self.available_at,
            "canary_id": self.canary_id,
            "coordinate_index": self.coordinate_index,
            "details": dict(self.details),
            "event_time": self.event_time,
            "reason_code": self.reason_code,
            "stage_id": self.stage_id,
        }
        if self.entity is not None:
            body["entity"] = self.entity
        return body


@dataclass(frozen=True)
class TimeSafetyAudit:
    status: str
    block_reasons: tuple[str, ...]
    checked_row_count: int
    cutoff_available_at: str
    coordinate_audits: tuple[TimeSafetyCoordinateAudit, ...] = ()
    canary_failures: tuple[CanaryFailure, ...] = ()
    causal_availability_window: Mapping[str, Any] = field(default_factory=dict)

    def to_manifest(
        self,
        catalog: ContractCatalog,
        *,
        snapshot_ref: TypedRef | None = None,
    ) -> ManifestEnvelope:
        body = {
            "time_safety_audit_id": "prototype_time_safety_audit_v1",
            "owner_id": "module.timeguard-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "status": self.status,
            "block_reason_codes": list(self.block_reasons),
            "checked_row_count": self.checked_row_count,
            "cutoff_available_at": self.cutoff_available_at,
            "coordinate_audits": [
                coordinate.as_dict() for coordinate in self.coordinate_audits
            ],
            "canary_failures": [
                canary_failure.as_dict()
                for canary_failure in self.canary_failures
            ],
            "causal_availability_window": dict(self.causal_availability_window),
        }
        if snapshot_ref is not None:
            body["snapshot_ref"] = snapshot_ref.as_dict()
        return ManifestEnvelope.build(
            schema_name="time_safety_audit_manifest@1.0.0",
            module_id="timeguard",
            body=body,
            catalog=catalog,
        )


def audit_snapshot_time_safety(snapshot: FrozenDatasetSnapshot) -> TimeSafetyAudit:
    block_reasons: set[str] = set()
    coordinate_audits: list[TimeSafetyCoordinateAudit] = []
    canary_failures: list[CanaryFailure] = []

    for index, row in enumerate(snapshot.rows):
        reason_codes: set[str] = set()
        if row.available_at > snapshot.cutoff_available_at:
            reason_codes.add("future_availability")
        if row.event_time > snapshot.cutoff_available_at:
            reason_codes.add("future_event_time")

        normalized_reasons = tuple(sorted(reason_codes))
        status = "blocked" if normalized_reasons else "passed"
        coordinate_audits.append(
            TimeSafetyCoordinateAudit(
                coordinate_index=index,
                entity=row.entity,
                event_time=row.event_time,
                available_at=row.available_at,
                payload_hash=row.payload_hash,
                status=status,
                reason_codes=normalized_reasons,
            )
        )
        block_reasons.update(normalized_reasons)
        for reason_code in normalized_reasons:
            canary_failures.append(
                CanaryFailure(
                    canary_id=f"timeguard:{index}:{reason_code}",
                    stage_id="time_safety_audit",
                    coordinate_index=index,
                    entity=row.entity,
                    event_time=row.event_time,
                    available_at=row.available_at,
                    reason_code=reason_code,
                    details={"payload_hash": row.payload_hash},
                )
            )

    deduped_reasons = tuple(sorted(block_reasons))
    available_times = tuple(row.available_at for row in snapshot.rows)
    causal_availability_window = {
        "coordinate_count": snapshot.row_count,
        "cutoff_available_at": snapshot.cutoff_available_at,
        "late_coordinate_count": sum(
            1
            for coordinate in coordinate_audits
            if "future_availability" in coordinate.reason_codes
        ),
        "max_available_at": max(available_times),
        "min_available_at": min(available_times),
    }
    return TimeSafetyAudit(
        status="blocked" if deduped_reasons else "passed",
        block_reasons=deduped_reasons,
        checked_row_count=snapshot.row_count,
        cutoff_available_at=snapshot.cutoff_available_at,
        coordinate_audits=tuple(coordinate_audits),
        canary_failures=tuple(canary_failures),
        causal_availability_window=causal_availability_window,
    )
