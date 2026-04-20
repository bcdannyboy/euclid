from __future__ import annotations

from pathlib import Path

import pytest

from euclid.artifacts import FilesystemArtifactStore
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef, extract_manifest_body_refs
from euclid.control_plane import DuplicateObjectIdError, SQLiteMetadataStore
from euclid.manifests import RunDeclarationManifest, SearchPlanManifest
from euclid.manifests.base import ManifestEnvelope

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_metadata_store_rejects_duplicate_object_ids_with_different_hashes(
    tmp_path,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    metadata = SQLiteMetadataStore(tmp_path / "registry.sqlite3")
    artifacts = FilesystemArtifactStore(tmp_path / "artifacts")

    first = ManifestEnvelope.build(
        schema_name="dataset_snapshot_manifest@1.0.0",
        module_id="snapshotting",
        body={"rows": 1},
        catalog=catalog,
        object_id="snapshot_duplicate_guard",
    )
    second = ManifestEnvelope.build(
        schema_name="dataset_snapshot_manifest@1.0.0",
        module_id="snapshotting",
        body={"rows": 2},
        catalog=catalog,
        object_id="snapshot_duplicate_guard",
    )

    metadata.upsert_manifest(first, artifacts.write_manifest(first))

    with pytest.raises(DuplicateObjectIdError):
        metadata.upsert_manifest(second, artifacts.write_manifest(second))


def test_metadata_store_indexes_run_ids_typed_refs_and_lineage_queries(
    tmp_path,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    metadata = SQLiteMetadataStore(tmp_path / "registry.sqlite3")
    artifacts = FilesystemArtifactStore(tmp_path / "artifacts")

    run_manifest = RunDeclarationManifest(
        run_id="demo_run",
        entrypoint_id="demo.run",
        requested_at="2026-04-12T00:00:00Z",
    ).to_manifest(catalog)
    search_manifest = SearchPlanManifest(
        search_plan_id="demo_search_plan",
        canonicalization_policy_ref=TypedRef(
            "canonicalization_policy_manifest@1.0.0",
            "canonicalization_default",
        ),
        codelength_policy_ref=TypedRef(
            "codelength_policy_manifest@1.1.0",
            "mdl_policy_default",
        ),
        reference_description_policy_ref=TypedRef(
            "reference_description_policy_manifest@1.1.0",
            "reference_description_default",
        ),
        observation_model_ref=TypedRef(
            "observation_model_manifest@1.1.0",
            "observation_model_default",
        ),
        predictive_mode="point_forecast",
        candidate_family_ids=("constant",),
    ).to_manifest(catalog)

    metadata.upsert_manifest(
        run_manifest,
        artifacts.write_manifest(run_manifest),
        body_ref_matches=extract_manifest_body_refs(
            schema_name=run_manifest.schema_name,
            body=run_manifest.body,
            catalog=catalog,
        ),
    )
    metadata.upsert_manifest(
        search_manifest,
        artifacts.write_manifest(search_manifest),
        body_ref_matches=extract_manifest_body_refs(
            schema_name=search_manifest.schema_name,
            body=search_manifest.body,
            catalog=catalog,
        ),
    )
    metadata.append_lineage(run_manifest.ref, search_manifest.ref)

    run_records = metadata.list_manifests_for_run("demo_run")
    children = metadata.list_lineage_children(run_manifest.ref)
    references = metadata.list_referenced_manifests(search_manifest.ref)
    referrers = metadata.list_referrers(
        TypedRef("codelength_policy_manifest@1.1.0", "mdl_policy_default")
    )

    assert [(record.schema_name, record.object_id) for record in run_records] == [
        (run_manifest.schema_name, run_manifest.object_id)
    ]
    assert children == (search_manifest.ref,)
    assert {reference.field_path for reference in references} == {
        "body.canonicalization_policy_ref",
        "body.codelength_policy_ref",
        "body.reference_description_policy_ref",
        "body.observation_model_ref",
    }
    assert {reference.target_ref for reference in references} == {
        TypedRef("canonicalization_policy_manifest@1.0.0", "canonicalization_default"),
        TypedRef("codelength_policy_manifest@1.1.0", "mdl_policy_default"),
        TypedRef(
            "reference_description_policy_manifest@1.1.0",
            "reference_description_default",
        ),
        TypedRef("observation_model_manifest@1.1.0", "observation_model_default"),
    }
    assert len(referrers) == 1
    assert referrers[0].source_ref == search_manifest.ref
