from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    PublicationRecordManifest,
    RunResultManifest,
)


@dataclass(frozen=True)
class LocalPublicationCatalogEntry:
    request_id: str
    publication_id: str
    publication_record_ref: TypedRef
    run_result_ref: TypedRef
    reproducibility_bundle_ref: TypedRef
    publication_mode: str
    catalog_scope: str
    published_at: str
    replay_verification_status: str
    comparator_exposure_status: str
    forecast_object_type: str | None = None
    validation_scope_ref: TypedRef | None = None
    scorecard_ref: TypedRef | None = None
    claim_card_ref: TypedRef | None = None
    abstention_ref: TypedRef | None = None
    primary_score_result_ref: TypedRef | None = None
    primary_calibration_result_ref: TypedRef | None = None
    mechanistic_evidence_ref: TypedRef | None = None

    @property
    def run_id(self) -> str:
        return self.run_result_ref.object_id

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "publication_id": self.publication_id,
            "publication_record_ref": self.publication_record_ref.as_dict(),
            "run_result_ref": self.run_result_ref.as_dict(),
            "reproducibility_bundle_ref": self.reproducibility_bundle_ref.as_dict(),
            "publication_mode": self.publication_mode,
            "catalog_scope": self.catalog_scope,
            "published_at": self.published_at,
            "replay_verification_status": self.replay_verification_status,
            "comparator_exposure_status": self.comparator_exposure_status,
        }
        if self.forecast_object_type is not None:
            payload["forecast_object_type"] = self.forecast_object_type
        if self.validation_scope_ref is not None:
            payload["validation_scope_ref"] = self.validation_scope_ref.as_dict()
        if self.scorecard_ref is not None:
            payload["scorecard_ref"] = self.scorecard_ref.as_dict()
        if self.claim_card_ref is not None:
            payload["claim_card_ref"] = self.claim_card_ref.as_dict()
        if self.abstention_ref is not None:
            payload["abstention_ref"] = self.abstention_ref.as_dict()
        if self.primary_score_result_ref is not None:
            payload["primary_score_result_ref"] = (
                self.primary_score_result_ref.as_dict()
            )
        if self.primary_calibration_result_ref is not None:
            payload["primary_calibration_result_ref"] = (
                self.primary_calibration_result_ref.as_dict()
            )
        if self.mechanistic_evidence_ref is not None:
            payload["mechanistic_evidence_ref"] = (
                self.mechanistic_evidence_ref.as_dict()
            )
        return payload

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, object]
    ) -> "LocalPublicationCatalogEntry":
        return cls(
            request_id=_required_string(payload, "request_id"),
            publication_id=_required_string(payload, "publication_id"),
            publication_record_ref=_required_typed_ref(
                payload, "publication_record_ref"
            ),
            run_result_ref=_required_typed_ref(payload, "run_result_ref"),
            reproducibility_bundle_ref=_required_typed_ref(
                payload, "reproducibility_bundle_ref"
            ),
            publication_mode=_required_string(payload, "publication_mode"),
            catalog_scope=_required_string(payload, "catalog_scope"),
            published_at=_required_string(payload, "published_at"),
            replay_verification_status=_required_string(
                payload, "replay_verification_status"
            ),
            comparator_exposure_status=_required_string(
                payload, "comparator_exposure_status"
            ),
            forecast_object_type=_optional_string(payload, "forecast_object_type"),
            validation_scope_ref=_optional_typed_ref(payload, "validation_scope_ref"),
            scorecard_ref=_optional_typed_ref(payload, "scorecard_ref"),
            claim_card_ref=_optional_typed_ref(payload, "claim_card_ref"),
            abstention_ref=_optional_typed_ref(payload, "abstention_ref"),
            primary_score_result_ref=_optional_typed_ref(
                payload, "primary_score_result_ref"
            ),
            primary_calibration_result_ref=_optional_typed_ref(
                payload, "primary_calibration_result_ref"
            ),
            mechanistic_evidence_ref=_optional_typed_ref(
                payload, "mechanistic_evidence_ref"
            ),
        )


def build_run_result_manifest(
    *,
    object_id: str,
    run_id: str,
    scope_ledger_ref: TypedRef,
    search_plan_ref: TypedRef,
    evaluation_plan_ref: TypedRef,
    comparison_universe_ref: TypedRef,
    evaluation_event_log_ref: TypedRef,
    evaluation_governance_ref: TypedRef,
    reproducibility_bundle_ref: TypedRef,
    forecast_object_type: str = "point",
    primary_validation_scope_ref: TypedRef | None = None,
    publication_mode: str,
    selected_candidate_ref: TypedRef | None = None,
    scorecard_ref: TypedRef | None = None,
    claim_card_ref: TypedRef | None = None,
    abstention_ref: TypedRef | None = None,
    prediction_artifact_refs: Sequence[TypedRef] = (),
    primary_score_result_ref: TypedRef | None = None,
    primary_calibration_result_ref: TypedRef | None = None,
    primary_external_evidence_ref: TypedRef | None = None,
    primary_mechanistic_evidence_ref: TypedRef | None = None,
    robustness_report_refs: Sequence[TypedRef] = (),
    deferred_scope_policy_refs: Sequence[TypedRef] = (),
) -> RunResultManifest:
    if publication_mode == "candidate_publication":
        if (
            selected_candidate_ref is None
            or scorecard_ref is None
            or claim_card_ref is None
        ):
            raise ContractValidationError(
                code="candidate_publication_missing_required_refs",
                message=(
                    "candidate publication requires reducer, scorecard, and claim refs"
                ),
                field_path="publication_mode",
            )
        if abstention_ref is not None:
            raise ContractValidationError(
                code="candidate_publication_forbids_abstention_ref",
                message="candidate publication may not include an abstention ref",
                field_path="abstention_ref",
            )
    elif publication_mode == "abstention_only_publication":
        if abstention_ref is None:
            raise ContractValidationError(
                code="abstention_publication_missing_required_ref",
                message="abstention-only publication requires an abstention ref",
                field_path="abstention_ref",
            )
        if any(
            ref is not None
            for ref in (selected_candidate_ref, scorecard_ref, claim_card_ref)
        ):
            raise ContractValidationError(
                code="invalid_result_mode_payload",
                message=(
                    "abstention-only publication may not carry hidden candidate refs"
                ),
                field_path="publication_mode",
            )
    else:
        raise ContractValidationError(
            code="invalid_publication_mode",
            message=f"unsupported publication_mode {publication_mode!r}",
            field_path="publication_mode",
        )

    return RunResultManifest(
        object_id=object_id,
        run_id=run_id,
        scope_ledger_ref=scope_ledger_ref,
        search_plan_ref=search_plan_ref,
        evaluation_plan_ref=evaluation_plan_ref,
        comparison_universe_ref=comparison_universe_ref,
        evaluation_event_log_ref=evaluation_event_log_ref,
        evaluation_governance_ref=evaluation_governance_ref,
        result_mode=publication_mode,
        forecast_object_type=forecast_object_type,
        primary_validation_scope_ref=primary_validation_scope_ref,
        primary_reducer_artifact_ref=selected_candidate_ref,
        primary_scorecard_ref=scorecard_ref,
        primary_claim_card_ref=claim_card_ref,
        primary_abstention_ref=abstention_ref,
        prediction_artifact_refs=tuple(prediction_artifact_refs),
        primary_score_result_ref=primary_score_result_ref,
        primary_calibration_result_ref=primary_calibration_result_ref,
        primary_external_evidence_ref=primary_external_evidence_ref,
        primary_mechanistic_evidence_ref=primary_mechanistic_evidence_ref,
        robustness_report_refs=tuple(robustness_report_refs),
        reproducibility_bundle_ref=reproducibility_bundle_ref,
        deferred_scope_policy_refs=tuple(deferred_scope_policy_refs),
    )


def build_publication_record_manifest(
    *,
    object_id: str,
    publication_id: str,
    run_result_manifest: ManifestEnvelope,
    comparison_universe_manifest: ManifestEnvelope,
    reproducibility_bundle_manifest: ManifestEnvelope,
    readiness_judgment_manifest: ManifestEnvelope,
    schema_lifecycle_integration_closure_ref: TypedRef,
    catalog_scope: str,
    published_at: str,
) -> PublicationRecordManifest:
    publication_mode = str(run_result_manifest.body["result_mode"])
    replay_verification_status = str(
        reproducibility_bundle_manifest.body["replay_verification_status"]
    )
    if replay_verification_status != "verified":
        raise ContractValidationError(
            code="publication_requires_verified_replay",
            message="publication is blocked unless replay verification is verified",
            field_path="reproducibility_bundle_manifest.replay_verification_status",
        )

    readiness_final_verdict = str(readiness_judgment_manifest.body["final_verdict"])
    if catalog_scope == "public" and readiness_final_verdict != "ready":
        raise ContractValidationError(
            code="public_catalog_requires_ready_judgment",
            message="public catalog publication requires a ready readiness judgment",
            field_path="catalog_scope",
        )

    comparator_exposure_status = _resolve_comparator_exposure_status(
        publication_mode=publication_mode,
        run_result_manifest=run_result_manifest,
        comparison_universe_manifest=comparison_universe_manifest,
    )
    return PublicationRecordManifest(
        object_id=object_id,
        publication_id=publication_id,
        run_result_ref=run_result_manifest.ref,
        catalog_scope=catalog_scope,
        publication_mode=publication_mode,
        replay_verification_status=replay_verification_status,
        comparator_exposure_status=comparator_exposure_status,
        reproducibility_bundle_ref=reproducibility_bundle_manifest.ref,
        readiness_judgment_ref=readiness_judgment_manifest.ref,
        schema_lifecycle_integration_closure_ref=(
            schema_lifecycle_integration_closure_ref
        ),
        published_at=published_at,
    )


def build_local_publication_catalog_entry(
    *,
    request_id: str,
    publication_record_manifest: ManifestEnvelope,
    run_result_manifest: ManifestEnvelope,
    reproducibility_bundle_manifest: ManifestEnvelope,
) -> LocalPublicationCatalogEntry:
    publication_record_ref = _required_typed_ref(
        publication_record_manifest.body, "run_result_ref"
    )
    if publication_record_ref != run_result_manifest.ref:
        raise ContractValidationError(
            code="publication_record_run_result_mismatch",
            message="publication record must reference the projected run result",
            field_path="publication_record_manifest.body.run_result_ref",
        )

    bundle_ref = _required_typed_ref(
        publication_record_manifest.body, "reproducibility_bundle_ref"
    )
    if bundle_ref != reproducibility_bundle_manifest.ref:
        raise ContractValidationError(
            code="publication_record_bundle_ref_mismatch",
            message=(
                "publication record must reference the projected reproducibility bundle"
            ),
            field_path="publication_record_manifest.body.reproducibility_bundle_ref",
        )

    publication_mode = _required_string(
        publication_record_manifest.body,
        "publication_mode",
    )
    if publication_mode != str(run_result_manifest.body["result_mode"]):
        raise ContractValidationError(
            code="publication_record_result_mode_mismatch",
            message="publication record mode must match the run result mode",
            field_path="publication_record_manifest.body.publication_mode",
        )

    replay_verification_status = str(
        reproducibility_bundle_manifest.body["replay_verification_status"]
    )
    if replay_verification_status != "verified":
        raise ContractValidationError(
            code="local_catalog_requires_verified_replay",
            message="local catalog entries require a replay-verified bundle",
            field_path="reproducibility_bundle_manifest.body.replay_verification_status",
        )

    scorecard_ref: TypedRef | None = None
    claim_card_ref: TypedRef | None = None
    abstention_ref: TypedRef | None = None
    if publication_mode == "candidate_publication":
        scorecard_ref = _required_typed_ref(
            run_result_manifest.body,
            "primary_scorecard_ref",
        )
        claim_card_ref = _required_typed_ref(
            run_result_manifest.body,
            "primary_claim_card_ref",
        )
    elif publication_mode == "abstention_only_publication":
        abstention_ref = _required_typed_ref(
            run_result_manifest.body,
            "primary_abstention_ref",
        )
    else:
        raise ContractValidationError(
            code="invalid_publication_mode",
            message=f"unsupported publication_mode {publication_mode!r}",
            field_path="publication_record_manifest.body.publication_mode",
        )

    return LocalPublicationCatalogEntry(
        request_id=request_id,
        publication_id=_required_string(
            publication_record_manifest.body,
            "publication_id",
        ),
        publication_record_ref=publication_record_manifest.ref,
        run_result_ref=run_result_manifest.ref,
        reproducibility_bundle_ref=reproducibility_bundle_manifest.ref,
        publication_mode=publication_mode,
        catalog_scope=_required_string(
            publication_record_manifest.body,
            "catalog_scope",
        ),
        published_at=_required_string(publication_record_manifest.body, "published_at"),
        replay_verification_status=replay_verification_status,
        comparator_exposure_status=_required_string(
            publication_record_manifest.body,
            "comparator_exposure_status",
        ),
        forecast_object_type=str(
            run_result_manifest.body.get("forecast_object_type", "point")
        ),
        validation_scope_ref=_optional_typed_ref(
            run_result_manifest.body,
            "primary_validation_scope_ref",
        ),
        scorecard_ref=scorecard_ref,
        claim_card_ref=claim_card_ref,
        abstention_ref=abstention_ref,
        primary_score_result_ref=_optional_typed_ref(
            run_result_manifest.body,
            "primary_score_result_ref",
        ),
        primary_calibration_result_ref=_optional_typed_ref(
            run_result_manifest.body,
            "primary_calibration_result_ref",
        ),
        mechanistic_evidence_ref=_optional_typed_ref(
            run_result_manifest.body,
            "primary_mechanistic_evidence_ref",
        ),
    )


def _resolve_comparator_exposure_status(
    *,
    publication_mode: str,
    run_result_manifest: ManifestEnvelope,
    comparison_universe_manifest: ManifestEnvelope,
) -> str:
    run_result_body = run_result_manifest.body
    if publication_mode == "abstention_only_publication":
        if any(
            field in run_result_body
            for field in (
                "primary_reducer_artifact_ref",
                "primary_scorecard_ref",
                "primary_claim_card_ref",
            )
        ):
            raise ContractValidationError(
                code="abstention_publication_has_hidden_candidate_refs",
                message=(
                    "abstention-only publication may not retain hidden candidate refs"
                ),
                field_path="run_result_manifest",
            )
        return "not_applicable_abstention_only"

    comparison_body = comparison_universe_manifest.body
    has_candidate_score = "candidate_score_result_ref" in comparison_body
    has_baseline_score = "baseline_score_result_ref" in comparison_body
    paired_records = comparison_body.get("paired_comparison_records", ())
    comparator_refs = comparison_body.get("comparator_score_result_refs", ())
    if (
        str(comparison_body.get("comparison_class_status")) != "comparable"
        or not has_candidate_score
        or not has_baseline_score
        or not paired_records
        or not comparator_refs
    ):
        raise ContractValidationError(
            code="candidate_publication_requires_comparator_exposure",
            message=(
                "candidate publication requires sealed comparable candidate and "
                "comparator exposure"
            ),
            field_path="comparison_universe_manifest",
        )
    return "satisfied"


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"expected non-empty string for {key!r}")


def _required_typed_ref(payload: Mapping[str, object], key: str) -> TypedRef:
    value = payload.get(key)
    if isinstance(value, Mapping):
        return _typed_ref(value, field_path=key)
    raise ValueError(f"expected typed ref mapping for {key!r}")


def _optional_typed_ref(payload: Mapping[str, object], key: str) -> TypedRef | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, Mapping):
        return _typed_ref(value, field_path=key)
    raise ValueError(f"expected typed ref mapping for {key!r}")


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"expected non-empty string for {key!r}")


def _typed_ref(payload: Mapping[str, object], *, field_path: str) -> TypedRef:
    schema_name = payload.get("schema_name")
    object_id = payload.get("object_id")
    if isinstance(schema_name, str) and isinstance(object_id, str):
        return TypedRef(schema_name=schema_name, object_id=object_id)
    raise ValueError(f"{field_path} must contain schema_name and object_id")


__all__ = [
    "LocalPublicationCatalogEntry",
    "build_local_publication_catalog_entry",
    "build_publication_record_manifest",
    "build_run_result_manifest",
]
