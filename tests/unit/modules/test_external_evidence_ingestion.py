from __future__ import annotations

from pathlib import Path

import pytest

from euclid.artifacts import FilesystemArtifactStore
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.control_plane import SQLiteMetadataStore
from euclid.manifest_registry import ManifestRegistry
from euclid.modules.external_evidence_ingestion import (
    build_external_evidence_bundle,
    register_external_evidence_bundle,
    verify_external_evidence_bundle_integrity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _raw_sources() -> tuple[dict[str, object], ...]:
    return (
        {
            "source_id": "paper-b",
            "citation": "doi:10.1000/beta",
            "evidence_kind": "intervention",
            "acquired_at": "2026-02-03T09:00:00Z",
            "content": {
                "finding": "glucose decreases after perturbation",
                "effect_direction": "negative",
            },
            "provenance": {"publisher": "lab-beta"},
            "independence_mode": "external_domain_source",
        },
        {
            "source_id": "paper-a",
            "citation": "doi:10.1000/alpha",
            "evidence_kind": "measurement",
            "acquired_at": "2026-02-01T09:00:00Z",
            "content": {
                "finding": "glucose recovers toward baseline",
                "effect_direction": "positive",
            },
            "provenance": {"publisher": "lab-alpha"},
            "independence_mode": "external_domain_source",
        },
    )


def test_build_external_evidence_bundle_sorts_sources_and_emits_stable_refs() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    result = build_external_evidence_bundle(
        bundle_id="glucose_mechanism_bundle",
        domain_id="glucose_regulation",
        acquisition_window={
            "start": "2026-02-01T00:00:00Z",
            "end": "2026-02-04T00:00:00Z",
        },
        raw_sources=_raw_sources(),
    )

    assert tuple(record.source_id for record in result.evidence_records) == (
        "paper-a",
        "paper-b",
    )
    assert tuple(digest.source_id for digest in result.source_digests) == (
        "paper-a",
        "paper-b",
    )

    manifest = result.external_evidence_manifest.to_manifest(catalog)
    assert manifest.body["ordered_source_ids"] == ["paper-a", "paper-b"]
    assert manifest.body["source_count"] == 2
    assert manifest.body["record_refs"] == [
        record.ref.as_dict() for record in result.evidence_records
    ]
    assert manifest.body["source_digest_refs"] == [
        digest.ref.as_dict() for digest in result.source_digests
    ]


def test_external_evidence_bundle_is_content_stable_under_input_permutation() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    sources = _raw_sources()

    first = build_external_evidence_bundle(
        bundle_id="glucose_mechanism_bundle",
        domain_id="glucose_regulation",
        acquisition_window={
            "start": "2026-02-01T00:00:00Z",
            "end": "2026-02-04T00:00:00Z",
        },
        raw_sources=sources,
    )
    second = build_external_evidence_bundle(
        bundle_id="glucose_mechanism_bundle",
        domain_id="glucose_regulation",
        acquisition_window={
            "start": "2026-02-01T00:00:00Z",
            "end": "2026-02-04T00:00:00Z",
        },
        raw_sources=tuple(reversed(sources)),
    )

    assert tuple(
        digest.digest_sha256 for digest in first.source_digests
    ) == tuple(digest.digest_sha256 for digest in second.source_digests)
    first_manifest = first.external_evidence_manifest.to_manifest(catalog)
    second_manifest = second.external_evidence_manifest.to_manifest(catalog)

    assert first_manifest.content_hash == second_manifest.content_hash


def test_build_external_evidence_bundle_rejects_predictive_overlap_sources() -> None:
    with pytest.raises(ContractValidationError):
        build_external_evidence_bundle(
            bundle_id="glucose_mechanism_bundle",
            domain_id="glucose_regulation",
            acquisition_window={
                "start": "2026-02-01T00:00:00Z",
                "end": "2026-02-04T00:00:00Z",
            },
            raw_sources=(
                {
                    "source_id": "paper-overlap",
                    "citation": "doi:10.1000/overlap",
                    "evidence_kind": "derived_prediction",
                    "acquired_at": "2026-02-02T09:00:00Z",
                    "content": {
                        "finding": "glucose response copied from forecast",
                    },
                    "provenance": {"publisher": "lab-overlap"},
                    "independence_mode": "derived_from_predictive_output",
                    "derived_from_predictive_ref": {
                        "schema_name": "point_score_result_manifest@1.0.0",
                        "object_id": "score_result",
                    },
                },
            ),
        )


def test_registered_external_evidence_bundle_integrity_verifies_sorted_sources(
    tmp_path: Path,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
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
        raw_sources=_raw_sources(),
    )

    integrity = verify_external_evidence_bundle_integrity(
        bundle=bundle.bundle,
        registry=registry,
    )

    assert integrity.status == "verified"
    assert integrity.failure_reason_codes == ()
    assert integrity.independence_modes == (
        "external_domain_source",
        "external_domain_source",
    )


def test_integrity_check_rejects_missing_provenance_or_digest_mismatch(
    tmp_path: Path,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
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
        raw_sources=_raw_sources(),
    )
    tampered_record = registry.register(
        type(bundle.records[0].manifest).build(
            schema_name=bundle.records[0].manifest.schema_name,
            object_id="glucose_mechanism_bundle__paper-a__record_tampered",
            module_id=bundle.records[0].manifest.module_id,
            body={
                **bundle.records[0].manifest.body,
                "external_evidence_record_id": (
                    "glucose_mechanism_bundle__paper-a__record_tampered"
                ),
                "provenance": {},
            },
            catalog=catalog,
        )
    )
    tampered_bundle = registry.register(
        type(bundle.bundle.manifest).build(
            schema_name=bundle.bundle.manifest.schema_name,
            object_id="glucose_mechanism_bundle_tampered",
            module_id=bundle.bundle.manifest.module_id,
            body={
                **bundle.bundle.manifest.body,
                "external_evidence_id": "glucose_mechanism_bundle_tampered",
                "bundle_id": "glucose_mechanism_bundle_tampered",
                "ordered_source_ids": ["paper-a", "paper-b"],
                "record_refs": [
                    tampered_record.manifest.ref.as_dict(),
                    bundle.records[1].manifest.ref.as_dict(),
                ],
            },
            catalog=catalog,
        )
    )

    integrity = verify_external_evidence_bundle_integrity(
        bundle=tampered_bundle,
        registry=registry,
    )

    assert integrity.status == "failed"
    assert set(integrity.failure_reason_codes) >= {
        "external_evidence_provenance_missing",
        "external_evidence_digest_mismatch",
    }
