from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from euclid.artifacts.artifact_store import (
    ArtifactIntegrityError,
    ArtifactStore,
    StoredArtifact,
)
from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef, extract_manifest_body_refs
from euclid.control_plane.metadata_store import (
    LineageEdgeRecord,
    ManifestMetadataRecord,
    ManifestReferenceRecord,
    MetadataStore,
)
from euclid.manifests.base import ManifestEnvelope


@dataclass(frozen=True)
class RegisteredManifest:
    manifest: ManifestEnvelope
    artifact: StoredArtifact
    metadata: ManifestMetadataRecord


@dataclass(frozen=True)
class StoreValidationIssue:
    code: str
    message: str
    ref: TypedRef | None = None
    related_ref: TypedRef | None = None
    path: Path | None = None


@dataclass(frozen=True)
class StoreValidationReport:
    manifest_count: int
    issues: tuple[StoreValidationIssue, ...]

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def is_valid(self) -> bool:
        return not self.issues


class ManifestRegistry:
    def __init__(
        self,
        *,
        catalog: ContractCatalog,
        artifact_store: ArtifactStore,
        metadata_store: MetadataStore,
    ) -> None:
        self._catalog = catalog
        self._artifact_store = artifact_store
        self._metadata_store = metadata_store

    def register(
        self,
        manifest: ManifestEnvelope,
        *,
        parent_refs: tuple[TypedRef, ...] = (),
    ) -> RegisteredManifest:
        artifact = self._artifact_store.write_manifest(manifest)
        metadata = self._metadata_store.upsert_manifest(
            manifest,
            artifact,
            body_ref_matches=extract_manifest_body_refs(
                schema_name=manifest.schema_name,
                body=manifest.body,
                catalog=self._catalog,
            ),
        )
        for parent_ref in parent_refs:
            self._metadata_store.append_lineage(parent_ref, manifest.ref)
        return RegisteredManifest(
            manifest=manifest, artifact=artifact, metadata=metadata
        )

    def resolve(self, ref: TypedRef) -> RegisteredManifest:
        metadata = self._metadata_store.get_manifest(ref)
        payload = self._artifact_store.read_json(metadata.content_hash)
        self._validate_payload_identity(payload, metadata=metadata)
        manifest = ManifestEnvelope.build(
            schema_name=payload["schema_name"],
            object_id=payload["object_id"],
            module_id=payload["module_id"],
            body=payload["body"],
            catalog=self._catalog,
        )
        artifact = StoredArtifact(
            content_hash=metadata.content_hash,
            path=metadata.artifact_path,
            size_bytes=metadata.size_bytes,
        )
        return RegisteredManifest(
            manifest=manifest, artifact=artifact, metadata=metadata
        )

    def list_manifests_for_run(self, run_id: str) -> tuple[RegisteredManifest, ...]:
        return tuple(
            self.resolve(
                TypedRef(
                    schema_name=record.schema_name,
                    object_id=record.object_id,
                )
            )
            for record in self._metadata_store.list_manifests_for_run(run_id)
        )

    def list_lineage_children(self, ref: TypedRef) -> tuple[TypedRef, ...]:
        return self._metadata_store.list_lineage_children(ref)

    def list_lineage_parents(self, ref: TypedRef) -> tuple[TypedRef, ...]:
        return self._metadata_store.list_lineage_parents(ref)

    def list_lineage_edges(self) -> tuple[LineageEdgeRecord, ...]:
        return self._metadata_store.list_lineage_edges()

    def list_referenced_manifests(
        self, ref: TypedRef
    ) -> tuple[ManifestReferenceRecord, ...]:
        return self._metadata_store.list_referenced_manifests(ref)

    def list_referrers(self, ref: TypedRef) -> tuple[ManifestReferenceRecord, ...]:
        return self._metadata_store.list_referrers(ref)

    def validate_store(self) -> StoreValidationReport:
        metadata_records = self._metadata_store.list_manifests()
        known_refs = {
            (record.schema_name, record.object_id)
            for record in metadata_records
        }
        issues: list[StoreValidationIssue] = []

        for record in metadata_records:
            ref = TypedRef(schema_name=record.schema_name, object_id=record.object_id)
            try:
                self.resolve(ref)
            except (ArtifactIntegrityError, KeyError, ValueError) as exc:
                issues.append(
                    StoreValidationIssue(
                        code="artifact_integrity_error",
                        message=str(exc),
                        ref=ref,
                        path=record.artifact_path,
                    )
                )

        for edge in self._metadata_store.list_lineage_edges():
            if (
                edge.parent_ref.schema_name,
                edge.parent_ref.object_id,
            ) not in known_refs:
                issues.append(
                    StoreValidationIssue(
                        code="dangling_lineage_parent",
                        message=(
                            "lineage edge points at a missing parent manifest "
                            f"{edge.parent_ref.schema_name}:{edge.parent_ref.object_id}"
                        ),
                        ref=edge.child_ref,
                        related_ref=edge.parent_ref,
                    )
                )
            if (edge.child_ref.schema_name, edge.child_ref.object_id) not in known_refs:
                issues.append(
                    StoreValidationIssue(
                        code="dangling_lineage_child",
                        message=(
                            "lineage edge points at a missing child manifest "
                            f"{edge.child_ref.schema_name}:{edge.child_ref.object_id}"
                        ),
                        ref=edge.parent_ref,
                        related_ref=edge.child_ref,
                    )
                )

        for edge in self._metadata_store.list_reference_edges():
            if (
                edge.source_ref.schema_name,
                edge.source_ref.object_id,
            ) not in known_refs:
                issues.append(
                    StoreValidationIssue(
                        code="dangling_reference_source",
                        message=(
                            "typed-ref edge points at a missing source manifest "
                            f"{edge.source_ref.schema_name}:{edge.source_ref.object_id}"
                        ),
                        ref=edge.source_ref,
                        related_ref=edge.target_ref,
                    )
                )
            if (
                edge.target_ref.schema_name,
                edge.target_ref.object_id,
            ) not in known_refs:
                issues.append(
                    StoreValidationIssue(
                        code="dangling_reference_target",
                        message=(
                            "typed-ref edge points at a missing target manifest "
                            f"{edge.target_ref.schema_name}:{edge.target_ref.object_id}"
                        ),
                        ref=edge.source_ref,
                        related_ref=edge.target_ref,
                    )
                )

        return StoreValidationReport(
            manifest_count=len(metadata_records),
            issues=tuple(issues),
        )

    def _validate_payload_identity(
        self,
        payload: Mapping[str, object],
        *,
        metadata: ManifestMetadataRecord,
    ) -> None:
        expected_pairs = {
            "schema_name": metadata.schema_name,
            "object_id": metadata.object_id,
            "module_id": metadata.module_id,
        }
        for field_name, expected_value in expected_pairs.items():
            actual_value = payload.get(field_name)
            if actual_value != expected_value:
                raise ArtifactIntegrityError(
                    f"manifest payload {field_name} mismatch for "
                    f"{metadata.schema_name}:{metadata.object_id}: expected "
                    f"{expected_value!r}, got {actual_value!r}"
                )
