from __future__ import annotations

from euclid.runtime.hashing import canonicalize_json, sha256_digest
from euclid.runtime.ids import make_content_addressed_id


def test_content_hashing_is_insensitive_to_dict_insertion_order() -> None:
    left = {"schema_name": "dataset_snapshot_manifest@1.0.0", "body": {"a": 1, "b": 2}}
    right = {"body": {"b": 2, "a": 1}, "schema_name": "dataset_snapshot_manifest@1.0.0"}

    assert canonicalize_json(left) == canonicalize_json(right)
    assert sha256_digest(left) == sha256_digest(right)
    assert make_content_addressed_id("dataset_snapshot_manifest@1.0.0", left) == (
        make_content_addressed_id("dataset_snapshot_manifest@1.0.0", right)
    )

