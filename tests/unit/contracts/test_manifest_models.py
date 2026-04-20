from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.manifests.base import ManifestEnvelope

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_manifest_envelope_builds_stable_ids_and_hashes() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    left = ManifestEnvelope.build(
        schema_name="dataset_snapshot_manifest@1.0.0",
        module_id="snapshotting",
        body={"cutoff": "2026-04-12T00:00:00Z", "rows": 3},
        catalog=catalog,
    )
    right = ManifestEnvelope.build(
        schema_name="dataset_snapshot_manifest@1.0.0",
        module_id="snapshotting",
        body={"rows": 3, "cutoff": "2026-04-12T00:00:00Z"},
        catalog=catalog,
    )

    assert left.object_id == right.object_id
    assert left.content_hash == right.content_hash
    assert left.canonical_json == right.canonical_json


def test_manifest_envelope_rejects_unknown_schema_names() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    with pytest.raises(ContractValidationError) as excinfo:
        ManifestEnvelope.build(
            schema_name="not_a_real_manifest@9.9.9",
            module_id="snapshotting",
            body={"rows": 1},
            catalog=catalog,
        )

    assert excinfo.value.code == "unknown_schema_name"


def test_manifest_envelope_serializes_keys_in_canonical_order() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    manifest = ManifestEnvelope.build(
        schema_name="dataset_snapshot_manifest@1.0.0",
        module_id="snapshotting",
        object_id="snapshot_serialization_fixture",
        body={"rows": 3, "cutoff": "2026-04-12T00:00:00Z"},
        catalog=catalog,
    )

    assert (
        manifest.canonical_json
        == '{"body":{"cutoff":"2026-04-12T00:00:00Z","rows":3},'
        '"module_id":"snapshotting",'
        '"object_id":"snapshot_serialization_fixture",'
        '"schema_name":"dataset_snapshot_manifest@1.0.0"}'
    )
