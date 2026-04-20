from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from euclid.artifacts.artifact_store import StoredArtifact
from euclid.contracts.refs import ManifestBodyRef, TypedRef
from euclid.manifests.base import ManifestEnvelope


@dataclass(frozen=True)
class ManifestMetadataRecord:
    schema_name: str
    object_id: str
    schema_family: str | None
    schema_version: str | None
    module_id: str
    run_id: str | None
    content_hash: str
    artifact_path: Path
    size_bytes: int


@dataclass(frozen=True)
class ManifestReferenceRecord:
    source_ref: TypedRef
    field_path: str
    target_ref: TypedRef


@dataclass(frozen=True)
class LineageEdgeRecord:
    parent_ref: TypedRef
    child_ref: TypedRef


class MetadataStore(Protocol):
    def upsert_manifest(
        self,
        manifest: ManifestEnvelope,
        artifact: StoredArtifact,
        *,
        body_ref_matches: tuple[ManifestBodyRef, ...] = (),
    ) -> ManifestMetadataRecord: ...

    def get_manifest(self, ref: TypedRef) -> ManifestMetadataRecord: ...

    def append_lineage(self, parent_ref: TypedRef, child_ref: TypedRef) -> None: ...

    def list_manifests(self) -> tuple[ManifestMetadataRecord, ...]: ...

    def list_manifests_for_run(
        self, run_id: str
    ) -> tuple[ManifestMetadataRecord, ...]: ...

    def list_lineage_children(self, parent_ref: TypedRef) -> tuple[TypedRef, ...]: ...

    def list_lineage_parents(self, child_ref: TypedRef) -> tuple[TypedRef, ...]: ...

    def list_lineage_edges(self) -> tuple[LineageEdgeRecord, ...]: ...

    def list_referenced_manifests(
        self, ref: TypedRef
    ) -> tuple[ManifestReferenceRecord, ...]: ...

    def list_referrers(
        self, ref: TypedRef
    ) -> tuple[ManifestReferenceRecord, ...]: ...

    def list_reference_edges(self) -> tuple[ManifestReferenceRecord, ...]: ...
