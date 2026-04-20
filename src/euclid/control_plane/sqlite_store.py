from __future__ import annotations

import sqlite3
from pathlib import Path

from euclid.artifacts.artifact_store import StoredArtifact
from euclid.contracts.loader import parse_schema_name
from euclid.contracts.refs import ManifestBodyRef, TypedRef
from euclid.control_plane.metadata_store import (
    LineageEdgeRecord,
    ManifestMetadataRecord,
    ManifestReferenceRecord,
)
from euclid.manifests.base import ManifestEnvelope


class DuplicateObjectIdError(ValueError):
    pass


class SQLiteMetadataStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path)
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def upsert_manifest(
        self,
        manifest: ManifestEnvelope,
        artifact: StoredArtifact,
        *,
        body_ref_matches: tuple[ManifestBodyRef, ...] = (),
    ) -> ManifestMetadataRecord:
        schema_family, schema_version, _ = parse_schema_name(manifest.schema_name)
        run_id = _extract_run_id(manifest)
        existing = self._connection.execute(
            """
            SELECT
                schema_name,
                object_id,
                schema_family,
                schema_version,
                module_id,
                run_id,
                content_hash,
                artifact_path,
                size_bytes
            FROM manifests
            WHERE schema_name = ? AND object_id = ?
            """,
            (manifest.schema_name, manifest.object_id),
        ).fetchone()

        if existing is not None and existing["content_hash"] != manifest.content_hash:
            raise DuplicateObjectIdError(
                f"{manifest.schema_name}:{manifest.object_id} already exists "
                "with a different content hash"
            )

        if existing is None:
            self._connection.execute(
                """
                INSERT INTO manifests (
                    schema_name,
                    object_id,
                    schema_family,
                    schema_version,
                    module_id,
                    run_id,
                    content_hash,
                    artifact_path,
                    size_bytes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.schema_name,
                    manifest.object_id,
                    schema_family,
                    schema_version,
                    manifest.module_id,
                    run_id,
                    manifest.content_hash,
                    str(artifact.path),
                    artifact.size_bytes,
                ),
            )
        else:
            self._connection.execute(
                """
                UPDATE manifests
                SET
                    schema_family = ?,
                    schema_version = ?,
                    module_id = ?,
                    run_id = ?,
                    content_hash = ?,
                    artifact_path = ?,
                    size_bytes = ?
                WHERE schema_name = ? AND object_id = ?
                """,
                (
                    schema_family,
                    schema_version,
                    manifest.module_id,
                    run_id,
                    manifest.content_hash,
                    str(artifact.path),
                    artifact.size_bytes,
                    manifest.schema_name,
                    manifest.object_id,
                ),
            )

        self._replace_reference_edges(manifest.ref, body_ref_matches)
        self._connection.commit()
        return ManifestMetadataRecord(
            schema_name=manifest.schema_name,
            object_id=manifest.object_id,
            schema_family=schema_family,
            schema_version=schema_version,
            module_id=manifest.module_id,
            run_id=run_id,
            content_hash=manifest.content_hash,
            artifact_path=artifact.path,
            size_bytes=artifact.size_bytes,
        )

    def get_manifest(self, ref: TypedRef) -> ManifestMetadataRecord:
        row = self._connection.execute(
            """
            SELECT
                schema_name,
                object_id,
                schema_family,
                schema_version,
                module_id,
                run_id,
                content_hash,
                artifact_path,
                size_bytes
            FROM manifests
            WHERE schema_name = ? AND object_id = ?
            """,
            (ref.schema_name, ref.object_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown manifest ref: {ref.schema_name}:{ref.object_id}")
        return _row_to_record(row)

    def append_lineage(self, parent_ref: TypedRef, child_ref: TypedRef) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO lineage_edges (
                parent_schema_name,
                parent_object_id,
                child_schema_name,
                child_object_id
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                parent_ref.schema_name,
                parent_ref.object_id,
                child_ref.schema_name,
                child_ref.object_id,
            ),
        )
        self._connection.commit()

    def list_manifests(self) -> tuple[ManifestMetadataRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                schema_name,
                object_id,
                schema_family,
                schema_version,
                module_id,
                run_id,
                content_hash,
                artifact_path,
                size_bytes
            FROM manifests
            ORDER BY schema_name, object_id
            """
        ).fetchall()
        return tuple(_row_to_record(row) for row in rows)

    def list_manifests_for_run(self, run_id: str) -> tuple[ManifestMetadataRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                schema_name,
                object_id,
                schema_family,
                schema_version,
                module_id,
                run_id,
                content_hash,
                artifact_path,
                size_bytes
            FROM manifests
            WHERE run_id = ?
            ORDER BY schema_name, object_id
            """,
            (run_id,),
        ).fetchall()
        return tuple(_row_to_record(row) for row in rows)

    def list_lineage_children(self, parent_ref: TypedRef) -> tuple[TypedRef, ...]:
        rows = self._connection.execute(
            """
            SELECT child_schema_name, child_object_id
            FROM lineage_edges
            WHERE parent_schema_name = ? AND parent_object_id = ?
            ORDER BY child_schema_name, child_object_id
            """,
            (parent_ref.schema_name, parent_ref.object_id),
        ).fetchall()
        return tuple(
            TypedRef(
                schema_name=row["child_schema_name"],
                object_id=row["child_object_id"],
            )
            for row in rows
        )

    def list_lineage_parents(self, child_ref: TypedRef) -> tuple[TypedRef, ...]:
        rows = self._connection.execute(
            """
            SELECT parent_schema_name, parent_object_id
            FROM lineage_edges
            WHERE child_schema_name = ? AND child_object_id = ?
            ORDER BY parent_schema_name, parent_object_id
            """,
            (child_ref.schema_name, child_ref.object_id),
        ).fetchall()
        return tuple(
            TypedRef(
                schema_name=row["parent_schema_name"],
                object_id=row["parent_object_id"],
            )
            for row in rows
        )

    def list_lineage_edges(self) -> tuple[LineageEdgeRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                parent_schema_name,
                parent_object_id,
                child_schema_name,
                child_object_id
            FROM lineage_edges
            ORDER BY
                parent_schema_name,
                parent_object_id,
                child_schema_name,
                child_object_id
            """
        ).fetchall()
        return tuple(_row_to_lineage_edge(row) for row in rows)

    def list_referenced_manifests(
        self, ref: TypedRef
    ) -> tuple[ManifestReferenceRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                source_schema_name,
                source_object_id,
                field_path,
                target_schema_name,
                target_object_id
            FROM typed_ref_edges
            WHERE source_schema_name = ? AND source_object_id = ?
            ORDER BY field_path, target_schema_name, target_object_id
            """,
            (ref.schema_name, ref.object_id),
        ).fetchall()
        return tuple(_row_to_reference_record(row) for row in rows)

    def list_referrers(self, ref: TypedRef) -> tuple[ManifestReferenceRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                source_schema_name,
                source_object_id,
                field_path,
                target_schema_name,
                target_object_id
            FROM typed_ref_edges
            WHERE target_schema_name = ? AND target_object_id = ?
            ORDER BY source_schema_name, source_object_id, field_path
            """,
            (ref.schema_name, ref.object_id),
        ).fetchall()
        return tuple(_row_to_reference_record(row) for row in rows)

    def list_reference_edges(self) -> tuple[ManifestReferenceRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                source_schema_name,
                source_object_id,
                field_path,
                target_schema_name,
                target_object_id
            FROM typed_ref_edges
            ORDER BY
                source_schema_name,
                source_object_id,
                field_path,
                target_schema_name,
                target_object_id
            """
        ).fetchall()
        return tuple(_row_to_reference_record(row) for row in rows)

    def _ensure_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS manifests (
                schema_name TEXT NOT NULL,
                object_id TEXT NOT NULL,
                schema_family TEXT,
                schema_version TEXT,
                module_id TEXT NOT NULL,
                run_id TEXT,
                content_hash TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                PRIMARY KEY (schema_name, object_id)
            )
            """
        )
        self._ensure_manifest_column("schema_family", "TEXT")
        self._ensure_manifest_column("schema_version", "TEXT")
        self._ensure_manifest_column("run_id", "TEXT")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lineage_edges (
                parent_schema_name TEXT NOT NULL,
                parent_object_id TEXT NOT NULL,
                child_schema_name TEXT NOT NULL,
                child_object_id TEXT NOT NULL,
                PRIMARY KEY (
                    parent_schema_name,
                    parent_object_id,
                    child_schema_name,
                    child_object_id
                )
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS typed_ref_edges (
                source_schema_name TEXT NOT NULL,
                source_object_id TEXT NOT NULL,
                field_path TEXT NOT NULL,
                target_schema_name TEXT NOT NULL,
                target_object_id TEXT NOT NULL,
                PRIMARY KEY (
                    source_schema_name,
                    source_object_id,
                    field_path,
                    target_schema_name,
                    target_object_id
                )
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS manifests_content_hash_idx
            ON manifests(content_hash)
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS manifests_run_id_idx
            ON manifests(run_id)
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS manifests_schema_family_idx
            ON manifests(schema_family)
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS lineage_edges_child_idx
            ON lineage_edges(child_schema_name, child_object_id)
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS typed_ref_edges_target_idx
            ON typed_ref_edges(target_schema_name, target_object_id)
            """
        )
        self._connection.commit()

    def _ensure_manifest_column(self, name: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(manifests)")
        }
        if name not in columns:
            self._connection.execute(
                f"ALTER TABLE manifests ADD COLUMN {name} {definition}"
            )

    def _replace_reference_edges(
        self,
        source_ref: TypedRef,
        body_ref_matches: tuple[ManifestBodyRef, ...],
    ) -> None:
        self._connection.execute(
            """
            DELETE FROM typed_ref_edges
            WHERE source_schema_name = ? AND source_object_id = ?
            """,
            (source_ref.schema_name, source_ref.object_id),
        )
        self._connection.executemany(
            """
            INSERT INTO typed_ref_edges (
                source_schema_name,
                source_object_id,
                field_path,
                target_schema_name,
                target_object_id
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    source_ref.schema_name,
                    source_ref.object_id,
                    match.field_path,
                    match.ref.schema_name,
                    match.ref.object_id,
                )
                for match in body_ref_matches
            ],
        )


def _extract_run_id(manifest: ManifestEnvelope) -> str | None:
    run_id = manifest.body.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return run_id
    return None


def _row_to_record(row: sqlite3.Row) -> ManifestMetadataRecord:
    return ManifestMetadataRecord(
        schema_name=row["schema_name"],
        object_id=row["object_id"],
        schema_family=row["schema_family"],
        schema_version=row["schema_version"],
        module_id=row["module_id"],
        run_id=row["run_id"],
        content_hash=row["content_hash"],
        artifact_path=Path(row["artifact_path"]),
        size_bytes=row["size_bytes"],
    )


def _row_to_lineage_edge(row: sqlite3.Row) -> LineageEdgeRecord:
    return LineageEdgeRecord(
        parent_ref=TypedRef(
            schema_name=row["parent_schema_name"],
            object_id=row["parent_object_id"],
        ),
        child_ref=TypedRef(
            schema_name=row["child_schema_name"],
            object_id=row["child_object_id"],
        ),
    )


def _row_to_reference_record(row: sqlite3.Row) -> ManifestReferenceRecord:
    return ManifestReferenceRecord(
        source_ref=TypedRef(
            schema_name=row["source_schema_name"],
            object_id=row["source_object_id"],
        ),
        field_path=row["field_path"],
        target_ref=TypedRef(
            schema_name=row["target_schema_name"],
            object_id=row["target_object_id"],
        ),
    )
