from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from euclid.artifacts import ArtifactIntegrityError, FilesystemArtifactStore
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.control_plane import SQLiteMetadataStore
from euclid.manifest_registry import ManifestRegistry
from euclid.manifests import RunDeclarationManifest, SearchPlanManifest
from euclid.manifests.base import ManifestEnvelope
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ADMITTED_FORECAST_OBJECT_TYPES = (
    "point",
    "distribution",
    "interval",
    "quantile",
    "event_probability",
)


def test_manifest_registry_roundtrip_is_stable(tmp_path) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
    )

    manifest = ManifestEnvelope.build(
        schema_name="dataset_snapshot_manifest@1.0.0",
        module_id="snapshotting",
        body={"cutoff": "2026-04-12T00:00:00Z", "rows": 5},
        catalog=catalog,
    )

    first = registry.register(manifest)
    second = registry.register(manifest)
    resolved = registry.resolve(first.manifest.ref)

    assert second.artifact.content_hash == first.artifact.content_hash
    assert resolved.manifest == manifest
    assert resolved.manifest.ref == manifest.ref


def test_manifest_registry_exposes_index_queries_for_runs_refs_and_lineage(
    tmp_path,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
    )

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

    registered_run = registry.register(run_manifest)
    registry.register(search_manifest, parent_refs=(registered_run.manifest.ref,))

    run_records = registry.list_manifests_for_run("demo_run")
    children = registry.list_lineage_children(registered_run.manifest.ref)
    references = registry.list_referenced_manifests(search_manifest.ref)

    assert [record.manifest.ref for record in run_records] == [
        registered_run.manifest.ref
    ]
    assert children == (search_manifest.ref,)
    assert {reference.target_ref for reference in references} == {
        TypedRef("canonicalization_policy_manifest@1.0.0", "canonicalization_default"),
        TypedRef("codelength_policy_manifest@1.1.0", "mdl_policy_default"),
        TypedRef(
            "reference_description_policy_manifest@1.1.0",
            "reference_description_default",
        ),
        TypedRef("observation_model_manifest@1.1.0", "observation_model_default"),
    }


def test_manifest_registry_validate_store_reports_corrupted_artifacts(
    tmp_path,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
    )

    manifest = ManifestEnvelope.build(
        schema_name="dataset_snapshot_manifest@1.0.0",
        module_id="snapshotting",
        body={"cutoff": "2026-04-12T00:00:00Z", "rows": 5},
        catalog=catalog,
    )
    registered = registry.register(manifest)
    registered.artifact.path.write_text(
        (
            '{"body":{"cutoff":"2026-04-12T00:00:00Z","rows":6},'
            f'"module_id":"snapshotting","object_id":"{manifest.object_id}",'
            '"schema_name":"dataset_snapshot_manifest@1.0.0"}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError):
        registry.resolve(manifest.ref)

    report = registry.validate_store()

    assert report.manifest_count == 1
    assert not report.is_valid
    assert report.issue_count == 1
    assert report.issues[0].ref == manifest.ref
    assert report.issues[0].code == "artifact_integrity_error"


@pytest.mark.parametrize(
    "forecast_object_type",
    _ADMITTED_FORECAST_OBJECT_TYPES,
)
def test_run_declaration_roundtrip_preserves_admitted_forecast_object_type(
    forecast_object_type: str,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    manifest = ManifestEnvelope.build(
        schema_name="run_manifest@1.0.0",
        module_id="manifest_registry",
        body={
            "run_id": f"{forecast_object_type}_demo_run",
            "entrypoint_id": "demo.run",
            "requested_at": "2026-04-12T00:00:00Z",
            "lifecycle_state": "run_declared",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": forecast_object_type,
            "requested_manifest_refs": [],
            "seed_records": [],
        },
        catalog=catalog,
    )

    restored = RunDeclarationManifest.from_manifest(manifest)
    roundtrip = restored.to_manifest(catalog)

    assert roundtrip.body.get("forecast_object_type") == forecast_object_type


@pytest.mark.parametrize(
    "forecast_object_type",
    _ADMITTED_FORECAST_OBJECT_TYPES,
)
def test_evaluation_plan_manifest_preserves_admitted_forecast_object_type(
    forecast_object_type: str,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    evaluation_plan, audit = _evaluation_plan(forecast_object_type)

    manifest = evaluation_plan.to_manifest(
        catalog,
        time_safety_audit_ref=audit.to_manifest(catalog).ref,
    )

    assert manifest.body["forecast_object_type"] == forecast_object_type
    assert (
        manifest.body["comparison_key"]["forecast_object_type"]
        == forecast_object_type
    )


@pytest.mark.parametrize(
    "forecast_object_type",
    _ADMITTED_FORECAST_OBJECT_TYPES,
)
def test_search_plan_manifest_preserves_admitted_forecast_object_type(
    forecast_object_type: str,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    evaluation_plan, _ = _evaluation_plan(forecast_object_type)
    canonicalization_policy = build_canonicalization_policy()

    search_plan = build_search_plan(
        evaluation_plan=evaluation_plan,
        canonicalization_policy_ref=TypedRef(
            "canonicalization_policy_manifest@1.0.0",
            canonicalization_policy.canonicalization_policy_id,
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
        candidate_family_ids=("constant",),
    )

    manifest = search_plan.to_manifest(catalog)

    assert manifest.body.get("forecast_object_type") == forecast_object_type


def _evaluation_plan(forecast_object_type: str):
    snapshot = FrozenDatasetSnapshot(
        series_id="manifest-roundtrip-series",
        cutoff_available_at="2026-01-06T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=(
            SnapshotRow(
                event_time="2026-01-01T00:00:00Z",
                available_at="2026-01-01T00:00:00Z",
                observed_value=10.0,
                revision_id=0,
                payload_hash="sha256:a",
            ),
            SnapshotRow(
                event_time="2026-01-02T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                observed_value=12.0,
                revision_id=0,
                payload_hash="sha256:b",
            ),
            SnapshotRow(
                event_time="2026-01-03T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                observed_value=13.0,
                revision_id=0,
                payload_hash="sha256:c",
            ),
            SnapshotRow(
                event_time="2026-01-04T00:00:00Z",
                available_at="2026-01-04T00:00:00Z",
                observed_value=15.0,
                revision_id=0,
                payload_hash="sha256:d",
            ),
            SnapshotRow(
                event_time="2026-01-05T00:00:00Z",
                available_at="2026-01-05T00:00:00Z",
                observed_value=16.0,
                revision_id=0,
                payload_hash="sha256:e",
            ),
            SnapshotRow(
                event_time="2026-01-06T00:00:00Z",
                available_at="2026-01-06T00:00:00Z",
                observed_value=18.0,
                revision_id=0,
                payload_hash="sha256:f",
            ),
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    return replace(evaluation_plan, forecast_object_type=forecast_object_type), audit
