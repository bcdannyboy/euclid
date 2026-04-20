from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from euclid.artifacts.artifact_store import ArtifactIntegrityError, StoredArtifact
from euclid.manifests.base import ManifestEnvelope
from euclid.runtime.hashing import canonicalize_json, sha256_digest

if TYPE_CHECKING:
    from euclid.performance import TelemetryRecorder


class FilesystemArtifactStore:
    def __init__(
        self,
        root: Path,
        *,
        telemetry: "TelemetryRecorder | None" = None,
    ) -> None:
        self._root = root
        self._objects_root = self._root / "objects"
        self._telemetry = telemetry
        self._objects_root.mkdir(parents=True, exist_ok=True)

    def write_json(self, value: object) -> StoredArtifact:
        canonical = canonicalize_json(value)
        return self._write_bytes(
            canonical.encode("utf-8"), sha256_digest(value), suffix=".json"
        )

    def write_manifest(self, manifest: ManifestEnvelope) -> StoredArtifact:
        return self._write_bytes(
            manifest.canonical_json.encode("utf-8"),
            manifest.content_hash,
            suffix=".json",
        )

    def read_json(self, content_hash: str) -> dict:
        artifact_path = self._payload_path(content_hash, suffix=".json")
        start = time.perf_counter()
        payload = self._load_json_verified(artifact_path, expected_hash=content_hash)
        self._record_io(
            operation="read",
            size_bytes=artifact_path.stat().st_size,
            elapsed_seconds=time.perf_counter() - start,
        )
        return payload

    def _write_bytes(
        self, payload: bytes, content_hash: str, *, suffix: str
    ) -> StoredArtifact:
        start = time.perf_counter()
        artifact_path = self._payload_path(content_hash, suffix=suffix)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path.exists():
            self._load_json_verified(artifact_path, expected_hash=content_hash)
            stored = StoredArtifact(
                content_hash=content_hash,
                path=artifact_path,
                size_bytes=artifact_path.stat().st_size,
            )
            self._record_io(
                operation="write",
                size_bytes=stored.size_bytes,
                elapsed_seconds=time.perf_counter() - start,
                cache_hit=True,
            )
            return stored

        temp_fd, temp_name = tempfile.mkstemp(
            dir=artifact_path.parent,
            prefix=".artifact-",
            suffix=suffix,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(temp_fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp_path, artifact_path)
            except FileExistsError:
                self._load_json_verified(artifact_path, expected_hash=content_hash)
            else:
                self._fsync_directory(artifact_path.parent)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        stored = StoredArtifact(
            content_hash=content_hash,
            path=artifact_path,
            size_bytes=artifact_path.stat().st_size,
        )
        self._record_io(
            operation="write",
            size_bytes=stored.size_bytes,
            elapsed_seconds=time.perf_counter() - start,
            cache_hit=False,
        )
        return stored

    def _payload_path(self, content_hash: str, *, suffix: str) -> Path:
        algorithm, digest = content_hash.split(":", 1)
        return (
            self._objects_root
            / algorithm
            / digest[:2]
            / digest[2:]
            / f"payload{suffix}"
        )

    def _load_json_verified(
        self,
        artifact_path: Path,
        *,
        expected_hash: str,
    ) -> dict:
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ArtifactIntegrityError(
                f"artifact is missing for {expected_hash}: {artifact_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ArtifactIntegrityError(
                f"artifact is not valid JSON for {expected_hash}: {artifact_path}"
            ) from exc

        if not isinstance(payload, Mapping):
            raise ArtifactIntegrityError(
                f"artifact payload must decode to a mapping for {expected_hash}"
            )

        actual_hash = self._semantic_hash_for_payload(payload)
        if actual_hash != expected_hash:
            raise ArtifactIntegrityError(
                "content hash mismatch for "
                f"{artifact_path}: expected {expected_hash}, got {actual_hash}"
            )
        return dict(payload)

    def _semantic_hash_for_payload(self, payload: Mapping[str, Any]) -> str:
        if self._looks_like_manifest_envelope(payload):
            return sha256_digest(
                {
                    "body": payload["body"],
                    "module_id": payload["module_id"],
                    "schema_name": payload["schema_name"],
                }
            )
        return sha256_digest(payload)

    def _looks_like_manifest_envelope(self, payload: Mapping[str, Any]) -> bool:
        return (
            isinstance(payload.get("body"), Mapping)
            and isinstance(payload.get("module_id"), str)
            and isinstance(payload.get("object_id"), str)
            and isinstance(payload.get("schema_name"), str)
        )

    def _fsync_directory(self, path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _record_io(
        self,
        *,
        operation: str,
        size_bytes: int,
        elapsed_seconds: float,
        cache_hit: bool = False,
    ) -> None:
        if self._telemetry is None:
            return
        self._telemetry.record_artifact_io(
            operation=operation,
            size_bytes=size_bytes,
            elapsed_seconds=elapsed_seconds,
            cache_hit=cache_hit,
        )
