from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from euclid.artifacts import FilesystemArtifactStore


def test_filesystem_artifact_store_handles_concurrent_content_addressed_writes(
    tmp_path,
) -> None:
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    payload = {"cutoff": "2026-04-12T00:00:00Z", "rows": 5}

    def write_payload() -> tuple[str, str]:
        artifact = store.write_json(payload)
        return artifact.content_hash, str(artifact.path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: write_payload(), range(16)))

    assert len({content_hash for content_hash, _ in results}) == 1
    assert len({path for _, path in results}) == 1
    assert not any(store._objects_root.rglob(".artifact-*.json"))
