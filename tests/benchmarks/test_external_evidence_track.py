from __future__ import annotations

from pathlib import Path

import euclid
import yaml

from euclid.artifacts import FilesystemArtifactStore
from euclid.contracts.loader import load_contract_catalog
from euclid.control_plane import SQLiteMetadataStore
from euclid.manifest_registry import ManifestRegistry
from euclid.modules.external_evidence_ingestion import (
    register_external_evidence_bundle,
    verify_external_evidence_bundle_integrity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSITIVE_TASK = (
    PROJECT_ROOT / "benchmarks/tasks/mechanistic/mechanistic-lane-positive.yaml"
)
NEGATIVE_TASK = (
    PROJECT_ROOT / "benchmarks/tasks/mechanistic/mechanistic-lane-negative.yaml"
)
POSITIVE_FIXTURE = (
    PROJECT_ROOT / "fixtures/runtime/mechanistic/mechanistic-positive-evidence.yaml"
)


def test_positive_case_requires_external_evidence_semantics(tmp_path: Path) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=POSITIVE_TASK,
        benchmark_root=tmp_path / "external-evidence-positive-benchmark",
        resume=False,
    )
    fixture = _fixture(POSITIVE_FIXTURE)
    registry = _build_registry(tmp_path / "registry")
    bundle = register_external_evidence_bundle(
        catalog=load_contract_catalog(PROJECT_ROOT),
        registry=registry,
        bundle_id=str(fixture["bundle_id"]),
        domain_id=str(fixture["domain_id"]),
        acquisition_window=dict(fixture["acquisition_window"]),
        raw_sources=tuple(fixture["raw_sources"]),
    )
    integrity = verify_external_evidence_bundle_integrity(
        bundle=bundle.bundle,
        registry=registry,
    )

    assert result.task_manifest.regime_tags[:2] == (
        "mechanistic_lane",
        "external_evidence_required",
    )
    assert result.submitter_results[0].status == "selected"
    assert integrity.status == "verified"
    assert integrity.failure_reason_codes == ()


def test_insufficient_provenance_or_independence_lowers_outcome(
    tmp_path: Path,
) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=NEGATIVE_TASK,
        benchmark_root=tmp_path / "external-evidence-insufficient-benchmark",
        resume=False,
    )
    fixture = _fixture(POSITIVE_FIXTURE)
    registry = _build_registry(tmp_path / "registry")
    bundle = register_external_evidence_bundle(
        catalog=load_contract_catalog(PROJECT_ROOT),
        registry=registry,
        bundle_id=str(fixture["bundle_id"]),
        domain_id=str(fixture["domain_id"]),
        acquisition_window=dict(fixture["acquisition_window"]),
        raw_sources=tuple(fixture["raw_sources"]),
    )
    catalog = load_contract_catalog(PROJECT_ROOT)
    tampered_record = registry.register(
        type(bundle.records[0].manifest).build(
            schema_name=bundle.records[0].manifest.schema_name,
            object_id=f"{fixture['bundle_id']}__paper-a__insufficient_record",
            module_id=bundle.records[0].manifest.module_id,
            body={
                **bundle.records[0].manifest.body,
                "external_evidence_record_id": (
                    f"{fixture['bundle_id']}__paper-a__insufficient_record"
                ),
                "provenance": {},
                "independence_mode": "derived_from_predictive_output",
            },
            catalog=catalog,
        )
    )
    insufficient_bundle = registry.register(
        type(bundle.bundle.manifest).build(
            schema_name=bundle.bundle.manifest.schema_name,
            object_id=f"{fixture['bundle_id']}__insufficient",
            module_id=bundle.bundle.manifest.module_id,
            body={
                **bundle.bundle.manifest.body,
                "external_evidence_id": f"{fixture['bundle_id']}__insufficient",
                "bundle_id": f"{fixture['bundle_id']}__insufficient",
                "record_refs": [
                    tampered_record.manifest.ref.as_dict(),
                    bundle.records[1].manifest.ref.as_dict(),
                ],
            },
            catalog=catalog,
        )
    )
    integrity = verify_external_evidence_bundle_integrity(
        bundle=insufficient_bundle,
        registry=registry,
    )

    assert result.submitter_results[0].status == "selected"
    assert integrity.status == "failed"
    assert set(integrity.failure_reason_codes) >= {
        "external_evidence_independence_violation",
        "external_evidence_provenance_missing",
        "external_evidence_digest_mismatch",
    }


def test_tampered_bundle_fails_track(tmp_path: Path) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=POSITIVE_TASK,
        benchmark_root=tmp_path / "external-evidence-tamper-benchmark",
        resume=False,
    )
    fixture = _fixture(POSITIVE_FIXTURE)
    registry = _build_registry(tmp_path / "registry")
    bundle = register_external_evidence_bundle(
        catalog=load_contract_catalog(PROJECT_ROOT),
        registry=registry,
        bundle_id=str(fixture["bundle_id"]),
        domain_id=str(fixture["domain_id"]),
        acquisition_window=dict(fixture["acquisition_window"]),
        raw_sources=tuple(fixture["raw_sources"]),
    )
    catalog = load_contract_catalog(PROJECT_ROOT)
    tampered_digest = registry.register(
        type(bundle.source_digests[0].manifest).build(
            schema_name=bundle.source_digests[0].manifest.schema_name,
            object_id=f"{fixture['bundle_id']}__paper-a__tampered_digest",
            module_id=bundle.source_digests[0].manifest.module_id,
            body={
                **bundle.source_digests[0].manifest.body,
                "source_digest_id": f"{fixture['bundle_id']}__paper-a__tampered_digest",
                "digest_sha256": "0" * 64,
            },
            catalog=catalog,
        )
    )
    tampered_record = registry.register(
        type(bundle.records[0].manifest).build(
            schema_name=bundle.records[0].manifest.schema_name,
            object_id=f"{fixture['bundle_id']}__paper-a__tampered_record",
            module_id=bundle.records[0].manifest.module_id,
            body={
                **bundle.records[0].manifest.body,
                "external_evidence_record_id": (
                    f"{fixture['bundle_id']}__paper-a__tampered_record"
                ),
                "source_digest_ref": tampered_digest.manifest.ref.as_dict(),
            },
            catalog=catalog,
        )
    )
    tampered_bundle = registry.register(
        type(bundle.bundle.manifest).build(
            schema_name=bundle.bundle.manifest.schema_name,
            object_id=f"{fixture['bundle_id']}__tampered",
            module_id=bundle.bundle.manifest.module_id,
            body={
                **bundle.bundle.manifest.body,
                "external_evidence_id": f"{fixture['bundle_id']}__tampered",
                "bundle_id": f"{fixture['bundle_id']}__tampered",
                "record_refs": [
                    tampered_record.manifest.ref.as_dict(),
                    bundle.records[1].manifest.ref.as_dict(),
                ],
                "source_digest_refs": [
                    tampered_digest.manifest.ref.as_dict(),
                    bundle.source_digests[1].manifest.ref.as_dict(),
                ],
            },
            catalog=catalog,
        )
    )
    integrity = verify_external_evidence_bundle_integrity(
        bundle=tampered_bundle,
        registry=registry,
    )

    assert result.submitter_results[0].status == "selected"
    assert integrity.status == "failed"
    assert "external_evidence_digest_mismatch" in integrity.failure_reason_codes


def _build_registry(tmp_path: Path) -> ManifestRegistry:
    catalog = load_contract_catalog(PROJECT_ROOT)
    return ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
    )


def _fixture(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
