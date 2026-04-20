from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifest_registry import ManifestRegistry, RegisteredManifest
from euclid.manifests.runtime_models import (
    ExternalEvidenceManifest,
    ExternalEvidenceRecordManifest,
    SourceDigestManifest,
)
from euclid.runtime.hashing import normalize_json_value, sha256_digest


@dataclass(frozen=True)
class ExternalEvidenceBundleResult:
    source_digests: tuple[SourceDigestManifest, ...]
    evidence_records: tuple[ExternalEvidenceRecordManifest, ...]
    external_evidence_manifest: ExternalEvidenceManifest


@dataclass(frozen=True)
class RegisteredExternalEvidenceBundle:
    source_digests: tuple[RegisteredManifest, ...]
    records: tuple[RegisteredManifest, ...]
    bundle: RegisteredManifest


@dataclass(frozen=True)
class ExternalEvidenceIntegrityCheck:
    status: str
    failure_reason_codes: tuple[str, ...]
    record_refs: tuple[TypedRef, ...]
    source_digest_refs: tuple[TypedRef, ...]
    independence_modes: tuple[str, ...]


def build_external_evidence_bundle(
    *,
    bundle_id: str,
    domain_id: str,
    acquisition_window: Mapping[str, Any],
    raw_sources: Sequence[Mapping[str, Any]],
) -> ExternalEvidenceBundleResult:
    if not bundle_id:
        raise ContractValidationError(
            code="invalid_external_evidence_bundle",
            message="bundle_id must be a non-empty string",
            field_path="bundle_id",
        )
    if not domain_id:
        raise ContractValidationError(
            code="invalid_external_evidence_bundle",
            message="domain_id must be a non-empty string",
            field_path="domain_id",
        )
    if not isinstance(acquisition_window, Mapping):
        raise ContractValidationError(
            code="invalid_external_evidence_bundle",
            message="acquisition_window must be a mapping",
            field_path="acquisition_window",
        )
    ordered_sources = tuple(
        sorted(
            (
                _normalize_source_payload(source, index=index)
                for index, source in enumerate(raw_sources)
            ),
            key=lambda item: (item["acquired_at"], item["source_id"]),
        )
    )
    if not ordered_sources:
        raise ContractValidationError(
            code="invalid_external_evidence_bundle",
            message="raw_sources must not be empty",
            field_path="raw_sources",
        )

    source_digests = tuple(
        SourceDigestManifest(
            object_id=f"{bundle_id}__{source['source_id']}__digest",
            source_digest_id=f"{bundle_id}__{source['source_id']}__digest",
            source_id=str(source["source_id"]),
            domain_id=domain_id,
            acquired_at=str(source["acquired_at"]),
            evidence_kind=str(source["evidence_kind"]),
            digest_sha256=_source_digest_sha256(source),
        )
        for source in ordered_sources
    )
    evidence_records = tuple(
        ExternalEvidenceRecordManifest(
            object_id=f"{bundle_id}__{source['source_id']}__record",
            external_evidence_record_id=f"{bundle_id}__{source['source_id']}__record",
            bundle_id=bundle_id,
            source_id=str(source["source_id"]),
            domain_id=domain_id,
            acquired_at=str(source["acquired_at"]),
            evidence_kind=str(source["evidence_kind"]),
            source_digest_ref=digest.ref,
            citation=str(source["citation"]),
            content=dict(source["content"]),
            provenance=dict(source["provenance"]),
            independence_mode=str(source["independence_mode"]),
        )
        for source, digest in zip(ordered_sources, source_digests, strict=True)
    )
    external_evidence_manifest = ExternalEvidenceManifest(
        object_id=bundle_id,
        external_evidence_id=bundle_id,
        bundle_id=bundle_id,
        domain_id=domain_id,
        acquisition_window=dict(acquisition_window),
        ordered_source_ids=tuple(
            str(source["source_id"]) for source in ordered_sources
        ),
        record_refs=tuple(record.ref for record in evidence_records),
        source_digest_refs=tuple(digest.ref for digest in source_digests),
        source_count=len(ordered_sources),
    )
    return ExternalEvidenceBundleResult(
        source_digests=source_digests,
        evidence_records=evidence_records,
        external_evidence_manifest=external_evidence_manifest,
    )


def register_external_evidence_bundle(
    *,
    catalog: ContractCatalog,
    registry: ManifestRegistry,
    bundle_id: str,
    domain_id: str,
    acquisition_window: Mapping[str, Any],
    raw_sources: Sequence[Mapping[str, Any]],
    parent_refs: Sequence[TypedRef] = (),
) -> RegisteredExternalEvidenceBundle:
    built = build_external_evidence_bundle(
        bundle_id=bundle_id,
        domain_id=domain_id,
        acquisition_window=acquisition_window,
        raw_sources=raw_sources,
    )
    registered_digests = tuple(
        registry.register(digest.to_manifest(catalog))
        for digest in built.source_digests
    )
    registered_records = tuple(
        registry.register(
            record.to_manifest(catalog),
            parent_refs=(digest.manifest.ref,),
        )
        for digest, record in zip(
            registered_digests, built.evidence_records, strict=True
        )
    )
    bundle = registry.register(
        built.external_evidence_manifest.to_manifest(catalog),
        parent_refs=tuple(parent_refs)
        + tuple(digest.manifest.ref for digest in registered_digests)
        + tuple(record.manifest.ref for record in registered_records),
    )
    return RegisteredExternalEvidenceBundle(
        source_digests=registered_digests,
        records=registered_records,
        bundle=bundle,
    )


def verify_external_evidence_bundle_integrity(
    *,
    bundle: RegisteredManifest,
    registry: ManifestRegistry,
) -> ExternalEvidenceIntegrityCheck:
    model = ExternalEvidenceManifest.from_manifest(bundle.manifest)
    failure_reason_codes: list[str] = []
    record_refs = tuple(model.record_refs)
    source_digest_refs = tuple(model.source_digest_refs)
    if model.source_count != len(record_refs) or model.source_count != len(
        source_digest_refs
    ):
        failure_reason_codes.append("external_evidence_source_count_mismatch")

    records = tuple(registry.resolve(ref) for ref in record_refs)
    digest_by_ref = {
        ref: registry.resolve(ref) for ref in source_digest_refs
    }
    ordered_record_ids = tuple(
        str(record.manifest.body["source_id"]) for record in records
    )
    ordered_digest_ids = tuple(
        str(digest_by_ref[ref].manifest.body["source_id"]) for ref in source_digest_refs
    )
    if ordered_record_ids != tuple(model.ordered_source_ids):
        failure_reason_codes.append("external_evidence_order_mismatch")
    if ordered_digest_ids != tuple(model.ordered_source_ids):
        failure_reason_codes.append("external_evidence_digest_order_mismatch")

    independence_modes: list[str] = []
    for record in records:
        record_model = ExternalEvidenceRecordManifest.from_manifest(record.manifest)
        independence_modes.append(record_model.independence_mode)
        if record_model.independence_mode != "external_domain_source":
            failure_reason_codes.append("external_evidence_independence_violation")
        if not record_model.provenance:
            failure_reason_codes.append("external_evidence_provenance_missing")
        if record_model.source_digest_ref not in digest_by_ref:
            failure_reason_codes.append("external_evidence_missing_source_digest")
            continue
        digest_model = SourceDigestManifest.from_manifest(
            digest_by_ref[record_model.source_digest_ref].manifest
        )
        if digest_model.source_id != record_model.source_id:
            failure_reason_codes.append("external_evidence_digest_source_mismatch")
        expected_digest = _source_digest_sha256(
            {
                "source_id": record_model.source_id,
                "citation": record_model.citation,
                "evidence_kind": record_model.evidence_kind,
                "acquired_at": record_model.acquired_at,
                "content": record_model.content,
                "provenance": record_model.provenance,
                "independence_mode": record_model.independence_mode,
            }
        )
        if digest_model.digest_sha256 != expected_digest:
            failure_reason_codes.append("external_evidence_digest_mismatch")

    normalized_failure_codes = tuple(dict.fromkeys(failure_reason_codes))
    return ExternalEvidenceIntegrityCheck(
        status="failed" if normalized_failure_codes else "verified",
        failure_reason_codes=normalized_failure_codes,
        record_refs=record_refs,
        source_digest_refs=source_digest_refs,
        independence_modes=tuple(independence_modes),
    )


def _normalize_source_payload(
    source: Mapping[str, Any], *, index: int
) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise ContractValidationError(
            code="invalid_external_evidence_source",
            message=f"raw_sources[{index}] must be a mapping",
            field_path=f"raw_sources[{index}]",
        )
    normalized = {
        "source_id": _required_string(
            source.get("source_id"), field_path=f"raw_sources[{index}].source_id"
        ),
        "citation": _required_string(
            source.get("citation"), field_path=f"raw_sources[{index}].citation"
        ),
        "evidence_kind": _required_string(
            source.get("evidence_kind"),
            field_path=f"raw_sources[{index}].evidence_kind",
        ),
        "acquired_at": _required_string(
            source.get("acquired_at"), field_path=f"raw_sources[{index}].acquired_at"
        ),
        "content": _required_mapping(
            source.get("content"), field_path=f"raw_sources[{index}].content"
        ),
        "provenance": _required_mapping(
            source.get("provenance"), field_path=f"raw_sources[{index}].provenance"
        ),
        "independence_mode": _required_string(
            source.get("independence_mode", "external_domain_source"),
            field_path=f"raw_sources[{index}].independence_mode",
        ),
    }
    _validate_source_independence(
        source,
        normalized=normalized,
        index=index,
    )
    return normalized


def _required_string(value: Any, *, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(
            code="invalid_external_evidence_source",
            message=f"{field_path} must be a non-empty string",
            field_path=field_path,
        )
    return value.strip()


def _required_mapping(value: Any, *, field_path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(
            code="invalid_external_evidence_source",
            message=f"{field_path} must be a mapping",
            field_path=field_path,
        )
    return dict(normalize_json_value(value))


def _validate_source_independence(
    source: Mapping[str, Any],
    *,
    normalized: Mapping[str, Any],
    index: int,
) -> None:
    independence_mode = str(normalized["independence_mode"])
    if independence_mode != "external_domain_source":
        raise ContractValidationError(
            code="invalid_external_evidence_source",
            message=(
                "external evidence sources must be independent external-domain "
                f"inputs; got independence_mode={independence_mode!r}"
            ),
            field_path=f"raw_sources[{index}].independence_mode",
        )
    if source.get("derived_from_predictive_ref") is not None:
        raise ContractValidationError(
            code="invalid_external_evidence_source",
            message=(
                "external evidence sources must not declare predictive-output "
                "lineage at ingestion time"
            ),
            field_path=f"raw_sources[{index}].derived_from_predictive_ref",
        )


def _source_digest_sha256(source: Mapping[str, Any]) -> str:
    return sha256_digest(
        {
            "acquired_at": source["acquired_at"],
            "citation": source["citation"],
            "content": normalize_json_value(source["content"]),
            "evidence_kind": source["evidence_kind"],
            "independence_mode": source["independence_mode"],
            "provenance": normalize_json_value(source["provenance"]),
            "source_id": source["source_id"],
        }
    )


__all__ = [
    "ExternalEvidenceIntegrityCheck",
    "ExternalEvidenceBundleResult",
    "RegisteredExternalEvidenceBundle",
    "build_external_evidence_bundle",
    "register_external_evidence_bundle",
    "verify_external_evidence_bundle_integrity",
]
