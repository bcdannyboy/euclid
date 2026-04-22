from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifest_registry import ManifestRegistry, RegisteredManifest
from euclid.manifests.runtime_models import (
    DomainSpecificMechanismMappingManifest,
    EvidenceIndependenceProtocolManifest,
    InvarianceTestManifest,
    MechanisticEvidenceDossierManifest,
    UnitsCheckManifest,
)
from euclid.modules.claims import (
    CLAIM_LANE_MECHANISTIC,
    CLAIM_LANE_PREDICTIVE,
    normalize_claim_lane,
)


@dataclass(frozen=True)
class MechanisticEvidenceEvaluation:
    mechanism_mapping: DomainSpecificMechanismMappingManifest
    units_check: UnitsCheckManifest
    invariance_test: InvarianceTestManifest
    evidence_independence: EvidenceIndependenceProtocolManifest
    dossier: MechanisticEvidenceDossierManifest


@dataclass(frozen=True)
class RegisteredMechanisticEvidenceEvaluation:
    mechanism_mapping: RegisteredManifest
    units_check: RegisteredManifest
    invariance_test: RegisteredManifest
    evidence_independence: RegisteredManifest
    dossier: RegisteredManifest


def evaluate_mechanistic_evidence(
    *,
    mechanistic_evidence_id: str,
    candidate_ref: TypedRef,
    prediction_artifact_ref: TypedRef,
    external_evidence_ref: TypedRef,
    lower_claim_ceiling: str,
    term_bindings: Sequence[Mapping[str, Any]],
    term_units: Sequence[Mapping[str, Any]],
    invariance_checks: Sequence[Mapping[str, Any]],
    external_evidence_records: Sequence[Mapping[str, Any]],
    predictive_evidence_refs: Sequence[TypedRef] = (),
) -> MechanisticEvidenceEvaluation:
    normalized_lower_claim_ceiling = normalize_claim_lane(
        lower_claim_ceiling,
        allow_legacy=True,
    )
    if not mechanistic_evidence_id:
        raise ContractValidationError(
            code="invalid_mechanistic_evidence",
            message="mechanistic_evidence_id must be a non-empty string",
            field_path="mechanistic_evidence_id",
        )

    mapping_status, mapping_reason_codes = _mapping_status(term_bindings)
    mechanism_mapping = DomainSpecificMechanismMappingManifest(
        object_id=f"{mechanistic_evidence_id}__mapping",
        mechanism_mapping_id=f"{mechanistic_evidence_id}__mapping",
        candidate_ref=candidate_ref,
        prediction_artifact_ref=prediction_artifact_ref,
        external_evidence_ref=external_evidence_ref,
        status=mapping_status,
        term_bindings=tuple(dict(item) for item in term_bindings),
        reason_codes=mapping_reason_codes,
    )

    units_status, units_reason_codes = _units_status(
        term_bindings=term_bindings,
        term_units=term_units,
    )
    units_check = UnitsCheckManifest(
        object_id=f"{mechanistic_evidence_id}__units",
        units_check_id=f"{mechanistic_evidence_id}__units",
        mechanism_mapping_ref=mechanism_mapping.ref,
        status=units_status,
        term_units=tuple(dict(item) for item in term_units),
        reason_codes=units_reason_codes,
    )

    invariance_status, invariance_reason_codes = _invariance_status(
        invariance_checks=invariance_checks
    )
    invariance_test = InvarianceTestManifest(
        object_id=f"{mechanistic_evidence_id}__invariance",
        invariance_test_id=f"{mechanistic_evidence_id}__invariance",
        mechanism_mapping_ref=mechanism_mapping.ref,
        external_evidence_ref=external_evidence_ref,
        status=invariance_status,
        checks=tuple(dict(item) for item in invariance_checks),
        reason_codes=invariance_reason_codes,
    )

    (
        independence_status,
        independence_reason_codes,
        overlap_refs,
    ) = _evidence_independence_status(
        external_evidence_records=external_evidence_records,
        predictive_evidence_refs=predictive_evidence_refs,
    )
    evidence_independence = EvidenceIndependenceProtocolManifest(
        object_id=f"{mechanistic_evidence_id}__independence",
        evidence_independence_id=f"{mechanistic_evidence_id}__independence",
        external_evidence_ref=external_evidence_ref,
        prediction_artifact_ref=prediction_artifact_ref,
        status=independence_status,
        predictive_evidence_refs=tuple(predictive_evidence_refs),
        overlap_refs=overlap_refs,
        reason_codes=independence_reason_codes,
    )

    dossier_status, resolved_claim_ceiling, dossier_reason_codes = _dossier_status(
        lower_claim_ceiling=normalized_lower_claim_ceiling,
        mapping_status=mechanism_mapping.status,
        units_status=units_check.status,
        invariance_status=invariance_test.status,
        independence_status=evidence_independence.status,
        mapping_reason_codes=mechanism_mapping.reason_codes,
        units_reason_codes=units_check.reason_codes,
        invariance_reason_codes=invariance_test.reason_codes,
        independence_reason_codes=evidence_independence.reason_codes,
    )
    dossier = MechanisticEvidenceDossierManifest(
        object_id=mechanistic_evidence_id,
        mechanistic_evidence_id=mechanistic_evidence_id,
        candidate_ref=candidate_ref,
        prediction_artifact_ref=prediction_artifact_ref,
        external_evidence_ref=external_evidence_ref,
        mechanism_mapping_ref=mechanism_mapping.ref,
        units_check_ref=units_check.ref,
        invariance_test_ref=invariance_test.ref,
        evidence_independence_ref=evidence_independence.ref,
        status=dossier_status,
        lower_claim_ceiling=normalized_lower_claim_ceiling,
        resolved_claim_ceiling=resolved_claim_ceiling,
        reason_codes=dossier_reason_codes,
    )
    return MechanisticEvidenceEvaluation(
        mechanism_mapping=mechanism_mapping,
        units_check=units_check,
        invariance_test=invariance_test,
        evidence_independence=evidence_independence,
        dossier=dossier,
    )


def register_mechanistic_evidence(
    *,
    catalog: ContractCatalog,
    registry: ManifestRegistry,
    mechanistic_evidence_id: str,
    candidate_ref: TypedRef,
    prediction_artifact_ref: TypedRef,
    external_evidence_ref: TypedRef,
    lower_claim_ceiling: str,
    term_bindings: Sequence[Mapping[str, Any]],
    term_units: Sequence[Mapping[str, Any]],
    invariance_checks: Sequence[Mapping[str, Any]],
    external_evidence_records: Sequence[Mapping[str, Any]],
    predictive_evidence_refs: Sequence[TypedRef] = (),
) -> RegisteredMechanisticEvidenceEvaluation:
    evaluation = evaluate_mechanistic_evidence(
        mechanistic_evidence_id=mechanistic_evidence_id,
        candidate_ref=candidate_ref,
        prediction_artifact_ref=prediction_artifact_ref,
        external_evidence_ref=external_evidence_ref,
        lower_claim_ceiling=lower_claim_ceiling,
        term_bindings=term_bindings,
        term_units=term_units,
        invariance_checks=invariance_checks,
        external_evidence_records=external_evidence_records,
        predictive_evidence_refs=predictive_evidence_refs,
    )
    mechanism_mapping = registry.register(
        evaluation.mechanism_mapping.to_manifest(catalog),
        parent_refs=(candidate_ref, prediction_artifact_ref, external_evidence_ref),
    )
    units_check = registry.register(
        evaluation.units_check.to_manifest(catalog),
        parent_refs=(mechanism_mapping.manifest.ref,),
    )
    invariance_test = registry.register(
        evaluation.invariance_test.to_manifest(catalog),
        parent_refs=(mechanism_mapping.manifest.ref, external_evidence_ref),
    )
    evidence_independence = registry.register(
        evaluation.evidence_independence.to_manifest(catalog),
        parent_refs=(external_evidence_ref, prediction_artifact_ref),
    )
    dossier = registry.register(
        evaluation.dossier.to_manifest(catalog),
        parent_refs=(
            external_evidence_ref,
            mechanism_mapping.manifest.ref,
            units_check.manifest.ref,
            invariance_test.manifest.ref,
            evidence_independence.manifest.ref,
        ),
    )
    return RegisteredMechanisticEvidenceEvaluation(
        mechanism_mapping=mechanism_mapping,
        units_check=units_check,
        invariance_test=invariance_test,
        evidence_independence=evidence_independence,
        dossier=dossier,
    )


def _mapping_status(
    term_bindings: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, ...]]:
    if not term_bindings:
        return "failed", ("mechanism_mapping_missing",)
    required_keys = {"term_id", "domain_entity", "activity"}
    for index, binding in enumerate(term_bindings):
        if not isinstance(binding, Mapping):
            return "failed", (f"mechanism_mapping_invalid_{index}",)
        if any(not str(binding.get(key, "")).strip() for key in required_keys):
            return "failed", ("mechanism_mapping_incomplete",)
    return "passed", ()


def _units_status(
    *,
    term_bindings: Sequence[Mapping[str, Any]],
    term_units: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, ...]]:
    if not term_units:
        return "failed", ("units_check_missing",)
    declared_term_ids = {
        str(binding.get("term_id", ""))
        for binding in term_bindings
        if isinstance(binding, Mapping)
    }
    units_by_term = {
        str(item.get("term_id", "")): item
        for item in term_units
        if isinstance(item, Mapping)
    }
    if any(term_id not in units_by_term for term_id in declared_term_ids):
        return "failed", ("units_check_incomplete",)
    if any(item.get("compatible") is False for item in units_by_term.values()):
        return "failed", ("units_check_incompatible",)
    return "passed", ()


def _invariance_status(
    *,
    invariance_checks: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, ...]]:
    if not invariance_checks:
        return "failed", ("invariance_evidence_missing",)
    if not any(str(check.get("status", "")) == "passed" for check in invariance_checks):
        return "failed", ("invariance_check_failed",)
    return "passed", ()


def _evidence_independence_status(
    *,
    external_evidence_records: Sequence[Mapping[str, Any]],
    predictive_evidence_refs: Sequence[TypedRef],
) -> tuple[str, tuple[str, ...], tuple[TypedRef, ...]]:
    overlap_refs: list[TypedRef] = []
    predictive_ref_keys = {
        (ref.schema_name, ref.object_id) for ref in predictive_evidence_refs
    }
    for record in external_evidence_records:
        if not isinstance(record, Mapping):
            continue
        independence_mode = str(record.get("independence_mode", ""))
        if independence_mode != "external_domain_source":
            payload = record.get("derived_from_predictive_ref")
            if isinstance(payload, Mapping):
                derived_ref = TypedRef(
                    schema_name=str(payload["schema_name"]),
                    object_id=str(payload["object_id"]),
                )
                if (
                    derived_ref.schema_name,
                    derived_ref.object_id,
                ) in predictive_ref_keys:
                    overlap_refs.append(derived_ref)
            else:
                overlap_refs.append(
                    TypedRef(
                        schema_name="prediction_artifact_manifest@1.1.0",
                        object_id="predictive_overlap",
                    )
                )
    if overlap_refs:
        return "failed", ("predictive_evidence_overlap",), tuple(overlap_refs)
    return "passed", (), ()


def _dossier_status(
    *,
    lower_claim_ceiling: str,
    mapping_status: str,
    units_status: str,
    invariance_status: str,
    independence_status: str,
    mapping_reason_codes: Sequence[str],
    units_reason_codes: Sequence[str],
    invariance_reason_codes: Sequence[str],
    independence_reason_codes: Sequence[str],
) -> tuple[str, str, tuple[str, ...]]:
    if lower_claim_ceiling != CLAIM_LANE_PREDICTIVE:
        return (
            "blocked_predictive_floor",
            lower_claim_ceiling,
            ("predictive_floor_required",),
        )
    if {
        mapping_status,
        units_status,
        invariance_status,
        independence_status,
    } == {"passed"}:
        return "passed", CLAIM_LANE_MECHANISTIC, ()
    return (
        "downgraded_to_predictive_within_declared_scope",
        CLAIM_LANE_PREDICTIVE,
        _unique_codes(
            (
                *mapping_reason_codes,
                *units_reason_codes,
                *invariance_reason_codes,
                *independence_reason_codes,
            )
        ),
    )


def _unique_codes(codes: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for code in codes:
        text = str(code)
        if text:
            seen.setdefault(text, None)
    return tuple(seen)


__all__ = [
    "MechanisticEvidenceEvaluation",
    "RegisteredMechanisticEvidenceEvaluation",
    "evaluate_mechanistic_evidence",
    "register_mechanistic_evidence",
]
