from __future__ import annotations

from importlib import import_module

import pytest

from euclid.operator_runtime.resources import (
    EuclidAssetError,
    resolve_asset_root,
    resolve_example_path,
)


def test_canonical_runtime_modules_are_importable_from_top_level_packages() -> None:
    expected_exports = {
        "euclid.manifest_registry": ("ManifestRegistry", "RegisteredManifest"),
        "euclid.ingestion": (
            "AdmittedOrderedNumericData",
            "ObservationRecord",
            "ingest_csv_dataset",
            "ingest_csv_observations",
            "ingest_dataframe_dataset",
            "ingest_parquet_dataset",
        ),
        "euclid.snapshotting": ("FrozenDatasetSnapshot", "freeze_dataset_snapshot"),
        "euclid.timeguard": ("TimeSafetyAudit", "audit_snapshot_time_safety"),
        "euclid.features": (
            "FeatureSpec",
            "FeatureView",
            "default_feature_spec",
            "materialize_feature_view",
        ),
        "euclid.split_planning": ("EvaluationPlan", "build_evaluation_plan"),
    }

    for module_name, names in expected_exports.items():
        module = import_module(module_name)
        for name in names:
            assert hasattr(module, name), f"{module_name} is missing {name}"


def test_control_plane_and_artifact_namespaces_are_separated() -> None:
    artifacts = import_module("euclid.artifacts")
    control_plane = import_module("euclid.control_plane")

    assert hasattr(artifacts, "ArtifactStore")
    assert hasattr(artifacts, "FilesystemArtifactStore")
    assert hasattr(artifacts, "StoredArtifact")
    assert not hasattr(artifacts, "SQLiteMetadataStore")

    assert hasattr(control_plane, "ManifestMetadataRecord")
    assert hasattr(control_plane, "MetadataStore")
    assert hasattr(control_plane, "RuntimeWorkspace")
    assert hasattr(control_plane, "SQLiteExecutionStateStore")
    assert hasattr(control_plane, "FileLock")
    assert hasattr(control_plane, "SQLiteMetadataStore")
    assert not hasattr(control_plane, "FilesystemArtifactStore")


def test_runtime_assets_resolve_from_packaged_asset_root() -> None:
    asset_root = resolve_asset_root()
    example_path = resolve_example_path("current_release_run.yaml")

    assert asset_root.name == "_assets"
    assert example_path.is_file()
    assert asset_root in example_path.parents


def test_missing_runtime_asset_fails_with_typed_error_code() -> None:
    with pytest.raises(EuclidAssetError) as exc_info:
        resolve_example_path("does-not-exist.yaml")

    assert exc_info.value.code == "euclid_asset_missing"
    assert "does-not-exist.yaml" in str(exc_info.value)
