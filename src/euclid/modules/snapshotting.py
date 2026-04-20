from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import DatasetSnapshotManifestModel
from euclid.modules._shared import canonical_timestamp
from euclid.modules.ingestion import ObservationRecord
from euclid.runtime.hashing import sha256_digest


@dataclass(frozen=True)
class SnapshotRow:
    event_time: str
    available_at: str
    observed_value: float
    revision_id: int
    payload_hash: str
    entity: str | None = None

    def as_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "event_time": self.event_time,
            "available_at": self.available_at,
            "observed_value": self.observed_value,
            "revision_id": self.revision_id,
            "payload_hash": self.payload_hash,
        }
        if self.entity is not None:
            body["entity"] = self.entity
        return body


@dataclass(frozen=True)
class RawSnapshotRow:
    entity: str
    event_time: str
    available_at: str
    target: float | None
    revision_id: int
    payload_hash: str
    source_id: str | None
    side_information: Mapping[str, Any]

    def as_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "entity": self.entity,
            "event_time": self.event_time,
            "available_at": self.available_at,
            "target": self.target,
            "revision_id": self.revision_id,
            "payload_hash": self.payload_hash,
            "side_information": dict(self.side_information),
        }
        if self.source_id is not None:
            body["source_id"] = self.source_id
        return body


@dataclass(frozen=True)
class SnapshotSamplingMetadata:
    visible_observation_count: int
    raw_row_count: int
    coded_row_count: int
    excluded_missing_target_count: int
    order_relation: str = "entity_then_event_time_then_availability_time"

    def as_dict(self) -> dict[str, object]:
        return {
            "coded_row_count": self.coded_row_count,
            "excluded_missing_target_count": self.excluded_missing_target_count,
            "order_relation": self.order_relation,
            "raw_row_count": self.raw_row_count,
            "visible_observation_count": self.visible_observation_count,
        }


@dataclass(frozen=True)
class SourceProvenanceRecord:
    source_id: str | None
    raw_row_count: int
    coded_row_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "raw_row_count": self.raw_row_count,
            "coded_row_count": self.coded_row_count,
        }


@dataclass(frozen=True)
class SnapshotMaterializationHashes:
    raw_observation_hash: str
    coded_target_hash: str
    lineage_payload_hash: str


@dataclass(frozen=True)
class FrozenDatasetSnapshot:
    series_id: str
    cutoff_available_at: str
    revision_policy: str
    rows: tuple[SnapshotRow, ...]
    entity_panel: tuple[str, ...] = field(default_factory=tuple)
    raw_rows: tuple[RawSnapshotRow, ...] = field(default_factory=tuple)
    sampling_metadata: SnapshotSamplingMetadata | None = None
    source_provenance: tuple[SourceProvenanceRecord, ...] = field(default_factory=tuple)
    materialization_hashes: SnapshotMaterializationHashes | None = None
    lineage_payload_hashes: tuple[str, ...] = field(default_factory=tuple)
    admitted_side_information_fields: tuple[str, ...] = field(default_factory=tuple)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def __post_init__(self) -> None:
        rows = tuple(
            SnapshotRow(
                entity=row.entity or self.series_id,
                event_time=row.event_time,
                available_at=row.available_at,
                observed_value=row.observed_value,
                revision_id=row.revision_id,
                payload_hash=row.payload_hash,
            )
            for row in self.rows
        )
        object.__setattr__(self, "rows", rows)

        raw_rows = self.raw_rows or tuple(
            RawSnapshotRow(
                entity=row.entity or self.series_id,
                event_time=row.event_time,
                available_at=row.available_at,
                target=row.observed_value,
                revision_id=row.revision_id,
                payload_hash=row.payload_hash,
                source_id=None,
                side_information={"revision_id": row.revision_id},
            )
            for row in rows
        )
        object.__setattr__(self, "raw_rows", raw_rows)

        entity_panel = self.entity_panel or tuple(
            sorted({row.entity for row in raw_rows})
        )
        object.__setattr__(self, "entity_panel", entity_panel)

        lineage_payload_hashes = self.lineage_payload_hashes or tuple(
            row.payload_hash for row in raw_rows
        )
        object.__setattr__(self, "lineage_payload_hashes", lineage_payload_hashes)

        admitted_side_information_fields = self.admitted_side_information_fields or (
            _admitted_side_information_fields(raw_rows)
        )
        object.__setattr__(
            self,
            "admitted_side_information_fields",
            admitted_side_information_fields,
        )

        sampling_metadata = self.sampling_metadata or SnapshotSamplingMetadata(
            visible_observation_count=len(raw_rows),
            raw_row_count=len(raw_rows),
            coded_row_count=len(self.rows),
            excluded_missing_target_count=max(0, len(raw_rows) - len(self.rows)),
        )
        object.__setattr__(self, "sampling_metadata", sampling_metadata)

        source_provenance = self.source_provenance or _source_provenance(raw_rows)
        object.__setattr__(self, "source_provenance", source_provenance)

        materialization_hashes = self.materialization_hashes or (
            SnapshotMaterializationHashes(
                raw_observation_hash=sha256_digest(
                    [row.as_dict() for row in raw_rows]
                ),
                coded_target_hash=sha256_digest(
                    [row.as_dict() for row in rows]
                ),
                lineage_payload_hash=sha256_digest(list(lineage_payload_hashes)),
            )
        )
        object.__setattr__(self, "materialization_hashes", materialization_hashes)

    def to_manifest(self, catalog: ContractCatalog) -> ManifestEnvelope:
        return DatasetSnapshotManifestModel(
            series_id=self.series_id,
            entity_panel=self.entity_panel,
            cutoff_available_at=self.cutoff_available_at,
            revision_policy=self.revision_policy,
            rows=tuple(row.as_dict() for row in self.rows),
            raw_observations=tuple(row.as_dict() for row in self.raw_rows),
            sampling_metadata=self.sampling_metadata.as_dict(),
            source_provenance=tuple(
                record.as_dict() for record in self.source_provenance
            ),
            admitted_side_information_fields=self.admitted_side_information_fields,
            raw_observation_hash=self.materialization_hashes.raw_observation_hash,
            coded_target_hash=self.materialization_hashes.coded_target_hash,
            lineage_payload_hash=self.materialization_hashes.lineage_payload_hash,
            lineage_payload_hashes=self.lineage_payload_hashes,
        ).to_manifest(catalog)


def freeze_dataset_snapshot(
    observations: Iterable[ObservationRecord],
    *,
    cutoff_available_at: str | None = None,
) -> FrozenDatasetSnapshot:
    ordered = tuple(observations)
    if not ordered:
        raise ContractValidationError(
            code="empty_dataset",
            message="cannot freeze an empty observation sequence",
            field_path="observations",
        )
    entity_panel = tuple(sorted({observation.series_id for observation in ordered}))

    resolved_cutoff = canonical_timestamp(
        cutoff_available_at or max(observation.available_at for observation in ordered)
    )
    visible = [
        observation
        for observation in ordered
        if observation.available_at <= resolved_cutoff
    ]
    if not visible:
        raise ContractValidationError(
            code="empty_snapshot",
            message="no observations are visible at the requested cutoff",
            field_path="cutoff_available_at",
        )

    latest_by_event_time: dict[tuple[str, str], ObservationRecord] = {}
    for observation in visible:
        key = (observation.entity, observation.event_time)
        current = latest_by_event_time.get(key)
        if current is None or (
            observation.available_at,
            observation.revision_id,
            observation.payload_hash,
        ) > (
            current.available_at,
            current.revision_id,
            current.payload_hash,
        ):
            latest_by_event_time[key] = observation

    selected_records = tuple(
        selected for _, selected in sorted(latest_by_event_time.items())
    )
    raw_rows = tuple(
        RawSnapshotRow(
            entity=selected.entity,
            event_time=selected.event_time,
            available_at=selected.available_at,
            target=selected.target,
            revision_id=selected.revision_id,
            payload_hash=selected.payload_hash,
            source_id=selected.source_id,
            side_information=selected.side_information,
        )
        for selected in selected_records
    )
    rows = tuple(
        SnapshotRow(
            entity=selected.entity,
            event_time=selected.event_time,
            available_at=selected.available_at,
            observed_value=selected.observed_value,
            revision_id=selected.revision_id,
            payload_hash=selected.payload_hash,
        )
        for selected in selected_records
        if selected.target is not None
    )
    if not rows:
        raise ContractValidationError(
            code="empty_coded_snapshot",
            message="no coded targets are visible at the requested cutoff",
            field_path="cutoff_available_at",
        )

    lineage_payload_hashes = tuple(
        observation.payload_hash
        for observation in sorted(
            visible,
            key=lambda observation: (
                observation.entity,
                observation.event_time,
                observation.available_at,
                observation.revision_id,
                observation.payload_hash,
            ),
        )
    )
    sampling_metadata = SnapshotSamplingMetadata(
        visible_observation_count=len(visible),
        raw_row_count=len(raw_rows),
        coded_row_count=len(rows),
        excluded_missing_target_count=len(raw_rows) - len(rows),
    )
    return FrozenDatasetSnapshot(
        series_id=_panel_series_id(entity_panel),
        cutoff_available_at=resolved_cutoff,
        revision_policy="latest_available_revision_per_event_time",
        rows=rows,
        entity_panel=entity_panel,
        raw_rows=raw_rows,
        sampling_metadata=sampling_metadata,
        source_provenance=_source_provenance(raw_rows),
        materialization_hashes=SnapshotMaterializationHashes(
            raw_observation_hash=sha256_digest(
                [row.as_dict() for row in raw_rows]
            ),
            coded_target_hash=sha256_digest([row.as_dict() for row in rows]),
            lineage_payload_hash=sha256_digest(list(lineage_payload_hashes)),
        ),
        lineage_payload_hashes=lineage_payload_hashes,
        admitted_side_information_fields=_admitted_side_information_fields(raw_rows),
    )


def _source_provenance(
    raw_rows: tuple[RawSnapshotRow, ...],
) -> tuple[SourceProvenanceRecord, ...]:
    counts: dict[str | None, list[int]] = {}
    for row in raw_rows:
        source_counts = counts.setdefault(row.source_id, [0, 0])
        source_counts[0] += 1
        if row.target is not None:
            source_counts[1] += 1
    return tuple(
        SourceProvenanceRecord(
            source_id=source_id,
            raw_row_count=raw_count,
            coded_row_count=coded_count,
        )
        for source_id, (raw_count, coded_count) in sorted(
            counts.items(),
            key=lambda item: (item[0] is None, "" if item[0] is None else item[0]),
        )
    )


def _admitted_side_information_fields(
    raw_rows: tuple[RawSnapshotRow, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                key
                for row in raw_rows
                for key in row.side_information
            }
        )
    )


def _panel_series_id(entity_panel: tuple[str, ...]) -> str:
    if len(entity_panel) == 1:
        return entity_panel[0]
    return f"entity_panel__{sha256_digest(list(entity_panel))[7:19]}"
