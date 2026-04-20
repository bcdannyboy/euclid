from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from euclid.manifests.base import ManifestEnvelope


class ArtifactIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class StoredArtifact:
    content_hash: str
    path: Path
    size_bytes: int


class ArtifactStore(Protocol):
    def write_json(self, value: object) -> StoredArtifact: ...

    def write_manifest(self, manifest: ManifestEnvelope) -> StoredArtifact: ...

    def read_json(self, content_hash: str) -> dict: ...
