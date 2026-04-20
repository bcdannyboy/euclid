from __future__ import annotations

from pathlib import Path

from euclid.artifacts import FilesystemArtifactStore
from euclid.contracts.loader import load_contract_catalog
from euclid.control_plane import SQLiteMetadataStore
from euclid.manifest_registry import ManifestRegistry
from euclid.manifests import RunDeclarationManifest
from euclid.modules.external_evidence_ingestion import register_external_evidence_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_external_evidence_registration_writes_lineage_and_typed_ref_edges(
    tmp_path: Path,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
    )

    run_manifest = registry.register(
        RunDeclarationManifest(
            run_id="mechanistic_demo_run",
            entrypoint_id="demo.mechanistic",
            requested_at="2026-04-14T00:00:00Z",
        ).to_manifest(catalog)
    )

    bundle = register_external_evidence_bundle(
        catalog=catalog,
        registry=registry,
        bundle_id="glucose_mechanism_bundle",
        domain_id="glucose_regulation",
        acquisition_window={
            "start": "2026-02-01T00:00:00Z",
            "end": "2026-02-04T00:00:00Z",
        },
        raw_sources=(
            {
                "source_id": "paper-b",
                "citation": "doi:10.1000/beta",
                "evidence_kind": "intervention",
                "acquired_at": "2026-02-03T09:00:00Z",
                "content": {"finding": "glucose decreases after perturbation"},
                "provenance": {"publisher": "lab-beta"},
                "independence_mode": "external_domain_source",
            },
            {
                "source_id": "paper-a",
                "citation": "doi:10.1000/alpha",
                "evidence_kind": "measurement",
                "acquired_at": "2026-02-01T09:00:00Z",
                "content": {"finding": "glucose recovers toward baseline"},
                "provenance": {"publisher": "lab-alpha"},
                "independence_mode": "external_domain_source",
            },
        ),
        parent_refs=(run_manifest.manifest.ref,),
    )

    assert set(registry.list_lineage_parents(bundle.bundle.manifest.ref)) == {
        run_manifest.manifest.ref,
        *(record.manifest.ref for record in bundle.records),
        *(digest.manifest.ref for digest in bundle.source_digests),
    }

    for digest, record in zip(bundle.source_digests, bundle.records, strict=True):
        assert registry.list_lineage_parents(record.manifest.ref) == (
            digest.manifest.ref,
        )

    referenced_targets = {
        edge.target_ref
        for edge in registry.list_referenced_manifests(bundle.bundle.manifest.ref)
    }
    assert referenced_targets == {
        *(record.manifest.ref for record in bundle.records),
        *(digest.manifest.ref for digest in bundle.source_digests),
    }
