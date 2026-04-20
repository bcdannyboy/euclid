from __future__ import annotations

import json

import pytest

from euclid.artifacts import ArtifactIntegrityError, FilesystemArtifactStore


def test_filesystem_artifact_store_is_content_addressed_and_idempotent(
    tmp_path,
) -> None:
    store = FilesystemArtifactStore(tmp_path / "artifacts")

    left = store.write_json({"b": 2, "a": 1})
    right = store.write_json({"a": 1, "b": 2})

    assert left.content_hash == right.content_hash
    assert left.path == right.path
    assert json.loads(left.path.read_text()) == {"a": 1, "b": 2}


def test_filesystem_artifact_store_detects_corrupted_payload_on_read(
    tmp_path,
) -> None:
    store = FilesystemArtifactStore(tmp_path / "artifacts")

    artifact = store.write_json({"a": 1})
    artifact.path.write_text('{"a": 2}\n', encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="content hash mismatch"):
        store.read_json(artifact.content_hash)
