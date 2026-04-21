from __future__ import annotations

import math
import platform
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from euclid.adapters.portfolio import ComparableBackendFinalist
from euclid.contracts.refs import TypedRef
from euclid.manifest_registry import ManifestRegistry, RegisteredManifest
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    ArtifactHashRecord,
    ReplayStageRecord,
    ReproducibilityBundleManifest,
    SeedRecord,
)
from euclid.modules.external_evidence_ingestion import (
    verify_external_evidence_bundle_integrity,
)
from euclid.runtime.numerical_environment import flatten_numerical_environment

_DEFAULT_REPLAY_ENTRYPOINT = "retained_scope_replay_v1"
_DEFAULT_REQUIRED_SEED_SCOPES = (
    "search",
    "surrogate_generation",
    "perturbation",
)


@dataclass(frozen=True)
class ReplayInspection:
    bundle_ref: TypedRef
    run_result_ref: TypedRef
    bundle_mode: str
    required_manifest_refs: tuple[TypedRef, ...]
    artifact_hash_records: tuple[ArtifactHashRecord, ...]
    seed_records: tuple[SeedRecord, ...]
    environment_metadata: Mapping[str, str]
    stage_order_records: tuple[ReplayStageRecord, ...]
    replay_verification_status: str
    failure_reason_codes: tuple[str, ...]

    @property
    def recorded_stage_order(self) -> tuple[str, ...]:
        return tuple(record.stage_id for record in self.stage_order_records)


@dataclass(frozen=True)
class ReplayedOutcome:
    selected_candidate_id: str
    confirmatory_primary_score: float
    publication_mode: str
    descriptive_status: str
    descriptive_reason_codes: tuple[str, ...]
    predictive_status: str
    predictive_reason_codes: tuple[str, ...]
    replayed_stage_order: tuple[str, ...]
    mechanistic_status: str = "not_requested"
    mechanistic_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayVerificationResult:
    bundle_ref: TypedRef
    run_result_ref: TypedRef
    selected_candidate_ref: TypedRef
    replay_verification_status: str
    failure_reason_codes: tuple[str, ...]
    confirmatory_primary_score: float
    required_manifest_refs: tuple[TypedRef, ...]
    artifact_hash_records: tuple[ArtifactHashRecord, ...]
    actual_hashes: Mapping[str, str]
    seed_records: tuple[SeedRecord, ...]
    environment_metadata: Mapping[str, str]
    stage_order_records: tuple[ReplayStageRecord, ...]
    recorded_stage_order: tuple[str, ...]
    replayed_stage_order: tuple[str, ...]


def build_runtime_environment_metadata() -> dict[str, str]:
    metadata = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": sys.platform,
        "platform_release": platform.release(),
        "machine": platform.machine(),
    }
    metadata.update(flatten_numerical_environment())
    return metadata


def build_replay_seed_records(seed_value: str) -> tuple[SeedRecord, ...]:
    return tuple(
        SeedRecord(seed_scope=scope, seed_value=seed_value)
        for scope in _DEFAULT_REQUIRED_SEED_SCOPES
    )


def build_replay_stage_order(
    *,
    dataset_snapshot_ref: TypedRef,
    feature_view_ref: TypedRef,
    search_plan_ref: TypedRef,
    evaluation_plan_ref: TypedRef,
    comparison_universe_ref: TypedRef,
    evaluation_event_log_ref: TypedRef,
    evaluation_governance_ref: TypedRef,
    scorecard_ref: TypedRef,
    candidate_or_abstention_ref: TypedRef,
    run_result_ref: TypedRef,
    external_evidence_ref: TypedRef | None = None,
    mechanistic_evidence_ref: TypedRef | None = None,
) -> tuple[ReplayStageRecord, ...]:
    records = [
        ReplayStageRecord(
            stage_id="dataset_snapshot_frozen",
            manifest_ref=dataset_snapshot_ref,
        ),
        ReplayStageRecord(
            stage_id="feature_view_materialized",
            manifest_ref=feature_view_ref,
        ),
        ReplayStageRecord(stage_id="search_plan_frozen", manifest_ref=search_plan_ref),
        ReplayStageRecord(
            stage_id="evaluation_plan_frozen",
            manifest_ref=evaluation_plan_ref,
        ),
        ReplayStageRecord(
            stage_id="comparison_universe_resolved",
            manifest_ref=comparison_universe_ref,
        ),
        ReplayStageRecord(
            stage_id="evaluation_event_log_written",
            manifest_ref=evaluation_event_log_ref,
        ),
        ReplayStageRecord(
            stage_id="evaluation_governance_resolved",
            manifest_ref=evaluation_governance_ref,
        ),
        ReplayStageRecord(stage_id="scorecard_resolved", manifest_ref=scorecard_ref),
    ]
    if external_evidence_ref is not None:
        records.append(
            ReplayStageRecord(
                stage_id="external_evidence_resolved",
                manifest_ref=external_evidence_ref,
            )
        )
    if mechanistic_evidence_ref is not None:
        records.append(
            ReplayStageRecord(
                stage_id="mechanistic_evidence_resolved",
                manifest_ref=mechanistic_evidence_ref,
            )
        )
    records.extend(
        (
            ReplayStageRecord(
                stage_id="publication_decision_resolved",
                manifest_ref=candidate_or_abstention_ref,
            ),
            ReplayStageRecord(
                stage_id="run_result_assembled", manifest_ref=run_result_ref
            ),
        )
    )
    return tuple(records)


def build_reproducibility_bundle_manifest(
    *,
    object_id: str,
    bundle_id: str,
    bundle_mode: str,
    dataset_snapshot_ref: TypedRef,
    feature_view_ref: TypedRef,
    search_plan_ref: TypedRef,
    evaluation_plan_ref: TypedRef,
    comparison_universe_ref: TypedRef,
    evaluation_event_log_ref: TypedRef,
    evaluation_governance_ref: TypedRef,
    run_result_ref: TypedRef,
    required_manifest_refs: Sequence[TypedRef],
    artifact_hash_records: Sequence[ArtifactHashRecord],
    seed_records: Sequence[SeedRecord] | None = None,
    environment_metadata: Mapping[str, str] | None = None,
    stage_order_records: Sequence[ReplayStageRecord] = (),
    replay_verification_status: str = "verified",
    failure_reason_codes: Sequence[str] = (),
    scope_id: str = "euclid_v1_binding_scope@1.0.0",
    replay_entrypoint_id: str = _DEFAULT_REPLAY_ENTRYPOINT,
) -> ReproducibilityBundleManifest:
    return ReproducibilityBundleManifest(
        object_id=object_id,
        bundle_id=bundle_id,
        scope_id=scope_id,
        bundle_mode=bundle_mode,
        dataset_snapshot_ref=dataset_snapshot_ref,
        feature_view_ref=feature_view_ref,
        search_plan_ref=search_plan_ref,
        evaluation_plan_ref=evaluation_plan_ref,
        comparison_universe_ref=comparison_universe_ref,
        evaluation_event_log_ref=evaluation_event_log_ref,
        evaluation_governance_ref=evaluation_governance_ref,
        run_result_ref=run_result_ref,
        required_manifest_refs=tuple(required_manifest_refs),
        artifact_hash_records=tuple(artifact_hash_records),
        seed_records=tuple(seed_records or build_replay_seed_records("0")),
        environment_metadata=dict(
            environment_metadata or build_runtime_environment_metadata()
        ),
        stage_order_records=tuple(stage_order_records),
        replay_entrypoint_id=replay_entrypoint_id,
        replay_verification_status=replay_verification_status,
        failure_reason_codes=tuple(str(code) for code in failure_reason_codes),
    )


def build_artifact_hash_records(
    *,
    snapshot: RegisteredManifest,
    feature_view: RegisteredManifest,
    search_plan: RegisteredManifest,
    evaluation_plan: RegisteredManifest,
    run_result_manifest: ManifestEnvelope,
    candidate_or_abstention: RegisteredManifest,
    scorecard: RegisteredManifest,
    prediction_artifact: RegisteredManifest,
    robustness_report: RegisteredManifest,
    validation_scope: RegisteredManifest | None = None,
    score_result: RegisteredManifest | None = None,
    calibration_result: RegisteredManifest | None = None,
    external_evidence: RegisteredManifest | None = None,
    mechanistic_evidence: RegisteredManifest | None = None,
) -> tuple[ArtifactHashRecord, ...]:
    records = [
        ArtifactHashRecord(
            artifact_role="dataset_snapshot",
            sha256=snapshot.manifest.content_hash,
        ),
        ArtifactHashRecord(
            artifact_role="feature_view",
            sha256=feature_view.manifest.content_hash,
        ),
        ArtifactHashRecord(
            artifact_role="search_plan",
            sha256=search_plan.manifest.content_hash,
        ),
        ArtifactHashRecord(
            artifact_role="evaluation_plan",
            sha256=evaluation_plan.manifest.content_hash,
        ),
        ArtifactHashRecord(
            artifact_role="run_result",
            sha256=run_result_manifest.content_hash,
        ),
        ArtifactHashRecord(
            artifact_role="candidate_or_abstention",
            sha256=candidate_or_abstention.manifest.content_hash,
        ),
        ArtifactHashRecord(
            artifact_role="scorecard",
            sha256=scorecard.manifest.content_hash,
        ),
        ArtifactHashRecord(
            artifact_role="prediction_artifact",
            sha256=prediction_artifact.manifest.content_hash,
        ),
        *(
            ()
            if validation_scope is None
            else (
                ArtifactHashRecord(
                    artifact_role="validation_scope",
                    sha256=validation_scope.manifest.content_hash,
                ),
            )
        ),
        ArtifactHashRecord(
            artifact_role="robustness_report",
            sha256=robustness_report.manifest.content_hash,
        ),
    ]
    if score_result is not None:
        records.append(
            ArtifactHashRecord(
                artifact_role="score_result",
                sha256=score_result.manifest.content_hash,
            )
        )
    if calibration_result is not None:
        records.append(
            ArtifactHashRecord(
                artifact_role="calibration_result",
                sha256=calibration_result.manifest.content_hash,
            )
        )
    if external_evidence is not None:
        records.append(
            ArtifactHashRecord(
                artifact_role="external_evidence",
                sha256=external_evidence.manifest.content_hash,
            )
        )
    if mechanistic_evidence is not None:
        records.append(
            ArtifactHashRecord(
                artifact_role="mechanistic_evidence",
                sha256=mechanistic_evidence.manifest.content_hash,
            )
        )
    return tuple(records)


def build_portfolio_replay_contract(
    *,
    selection_record_id: str,
    selection_scope: str,
    selection_rule: str,
    selected_provenance_id: str | None,
    selected_candidate_id: str | None,
    selected_candidate_hash: str | None,
    compared_finalists: Sequence[ComparableBackendFinalist | Mapping[str, Any]],
    decision_trace: Sequence[Mapping[str, Any]],
    replay_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "selection_record_id": selection_record_id,
        "selection_scope": selection_scope,
        "selection_rule": selection_rule,
        "selected_provenance_id": selected_provenance_id,
        "selected_candidate_id": selected_candidate_id,
        "selected_candidate_hash": selected_candidate_hash,
        "compared_finalists": [
            (
                item.as_dict()
                if isinstance(item, ComparableBackendFinalist)
                else dict(item)
            )
            for item in compared_finalists
        ],
        "decision_trace": [dict(item) for item in decision_trace],
    }
    if replay_policy is not None:
        payload["replay_policy"] = dict(replay_policy)
    return payload


def verify_portfolio_replay_contract(
    replay_contract: Mapping[str, Any],
    *,
    selected_candidate_id: str | None,
    selected_candidate_hash: str | None,
    compared_finalists: Sequence[ComparableBackendFinalist | Mapping[str, Any]],
    decision_trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_finalists = tuple(
        item.as_dict() if isinstance(item, ComparableBackendFinalist) else dict(item)
        for item in compared_finalists
    )
    expected_trace = tuple(dict(item) for item in decision_trace)
    failure_reason_codes: list[str] = []
    if replay_contract.get("selected_candidate_id") != selected_candidate_id:
        failure_reason_codes.append("selected_candidate_id_mismatch")
    if replay_contract.get("selected_candidate_hash") != selected_candidate_hash:
        failure_reason_codes.append("selected_candidate_hash_mismatch")
    if tuple(replay_contract.get("compared_finalists", ())) != expected_finalists:
        failure_reason_codes.append("compared_finalists_mismatch")
    if tuple(replay_contract.get("decision_trace", ())) != expected_trace:
        failure_reason_codes.append("decision_trace_mismatch")
    return {
        "replay_verification_status": (
            "failed" if failure_reason_codes else "verified"
        ),
        "failure_reason_codes": failure_reason_codes,
    }


def required_manifest_refs_for_publication(
    *,
    publication_mode: str,
    candidate_ref: TypedRef,
    scorecard_ref: TypedRef,
    claim_ref: TypedRef | None,
    abstention_ref: TypedRef | None,
    supporting_refs: Sequence[TypedRef] = (),
) -> tuple[TypedRef, ...]:
    if publication_mode == "candidate_publication":
        if claim_ref is None:
            raise ValueError("candidate publication requires claim_ref")
        return (candidate_ref, scorecard_ref, claim_ref, *tuple(supporting_refs))
    if abstention_ref is None:
        raise ValueError("abstention-only publication requires abstention_ref")
    return (abstention_ref, *tuple(supporting_refs))


def inspect_reproducibility_bundle(
    bundle: RegisteredManifest | ManifestEnvelope,
) -> ReplayInspection:
    bundle_manifest = _bundle_manifest(bundle)
    model = ReproducibilityBundleManifest.from_manifest(bundle_manifest)
    return ReplayInspection(
        bundle_ref=bundle_manifest.ref,
        run_result_ref=model.run_result_ref,
        bundle_mode=model.bundle_mode,
        required_manifest_refs=model.required_manifest_refs,
        artifact_hash_records=model.artifact_hash_records,
        seed_records=model.seed_records,
        environment_metadata=model.environment_metadata,
        stage_order_records=model.stage_order_records,
        replay_verification_status=model.replay_verification_status,
        failure_reason_codes=model.failure_reason_codes,
    )


def verify_replayed_outcome(
    *,
    bundle: RegisteredManifest,
    registry: ManifestRegistry,
    outcome: ReplayedOutcome,
) -> ReplayVerificationResult:
    inspection = inspect_reproducibility_bundle(bundle)
    run_result = registry.resolve(inspection.run_result_ref)
    scorecard = resolve_scorecard(run_result.manifest.body, registry)
    primary_score_result = resolve_primary_score_result(run_result, registry)
    resolved_selected_candidate_ref = selected_candidate_ref(
        run_result.manifest.body,
        registry,
    )
    selected_candidate = registry.resolve(resolved_selected_candidate_ref)
    actual_hashes = actual_artifact_hashes(
        bundle=bundle,
        run_result=run_result,
        registry=registry,
    )
    actual_required_refs = required_manifest_refs_from_run_result(
        run_result.manifest.body
    )
    recorded_hashes = {
        record.artifact_role: record.sha256
        for record in inspection.artifact_hash_records
    }

    failure_codes: list[str] = []

    if inspection.bundle_mode != str(run_result.manifest.body["result_mode"]):
        failure_codes.append("nondeterministic_replay")
    if inspection.required_manifest_refs != actual_required_refs:
        failure_codes.append("missing_required_ref")

    required_hash_roles = tuple(actual_hashes)
    if any(role not in recorded_hashes for role in required_hash_roles):
        failure_codes.append("missing_required_hash")
    elif any(
        recorded_hashes[role] != actual_hashes[role] for role in required_hash_roles
    ):
        failure_codes.append("artifact_hash_mismatch")

    recorded_seed_scopes = {record.seed_scope for record in inspection.seed_records}
    if any(
        scope not in recorded_seed_scopes for scope in _DEFAULT_REQUIRED_SEED_SCOPES
    ):
        failure_codes.append("missing_seed_record")

    if (
        not inspection.recorded_stage_order
        or inspection.recorded_stage_order != outcome.replayed_stage_order
    ):
        failure_codes.append("nondeterministic_replay")

    primary_external_evidence_ref = run_result.manifest.body.get(
        "primary_external_evidence_ref"
    )
    if isinstance(primary_external_evidence_ref, Mapping):
        integrity = verify_external_evidence_bundle_integrity(
            bundle=registry.resolve(_typed_ref(primary_external_evidence_ref)),
            registry=registry,
        )
        failure_codes.extend(integrity.failure_reason_codes)

    if (
        str(scorecard.manifest.body["descriptive_status"]) != outcome.descriptive_status
        or tuple(
            str(item) for item in scorecard.manifest.body["descriptive_reason_codes"]
        )
        != outcome.descriptive_reason_codes
        or str(scorecard.manifest.body["predictive_status"])
        != outcome.predictive_status
        or tuple(
            str(item) for item in scorecard.manifest.body["predictive_reason_codes"]
        )
        != outcome.predictive_reason_codes
        or str(scorecard.manifest.body.get("mechanistic_status", "not_requested"))
        != outcome.mechanistic_status
        or tuple(
            str(item)
            for item in scorecard.manifest.body.get("mechanistic_reason_codes", ())
        )
        != outcome.mechanistic_reason_codes
        or str(run_result.manifest.body["result_mode"]) != outcome.publication_mode
        or str(selected_candidate.manifest.body["candidate_id"])
        != outcome.selected_candidate_id
        or not math.isclose(
            float(primary_score_result.manifest.body["aggregated_primary_score"]),
            outcome.confirmatory_primary_score,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        failure_codes.append("nondeterministic_replay")

    normalized_failure_codes = _unique_codes(failure_codes)
    return ReplayVerificationResult(
        bundle_ref=inspection.bundle_ref,
        run_result_ref=inspection.run_result_ref,
        selected_candidate_ref=resolved_selected_candidate_ref,
        replay_verification_status=(
            "failed" if normalized_failure_codes else "verified"
        ),
        failure_reason_codes=normalized_failure_codes,
        confirmatory_primary_score=float(
            primary_score_result.manifest.body["aggregated_primary_score"]
        ),
        required_manifest_refs=inspection.required_manifest_refs,
        artifact_hash_records=inspection.artifact_hash_records,
        actual_hashes=actual_hashes,
        seed_records=inspection.seed_records,
        environment_metadata=inspection.environment_metadata,
        stage_order_records=inspection.stage_order_records,
        recorded_stage_order=inspection.recorded_stage_order,
        replayed_stage_order=tuple(outcome.replayed_stage_order),
    )


def resolve_scorecard(
    run_result_body: Mapping[str, Any], registry: ManifestRegistry
) -> RegisteredManifest:
    if run_result_body["result_mode"] == "candidate_publication":
        return registry.resolve(_typed_ref(run_result_body["primary_scorecard_ref"]))
    abstention = registry.resolve(_typed_ref(run_result_body["primary_abstention_ref"]))
    return registry.resolve(_typed_ref(abstention.manifest.body["governing_refs"][0]))


def resolve_candidate_or_abstention(
    run_result_body: Mapping[str, Any], registry: ManifestRegistry
) -> RegisteredManifest:
    if run_result_body["result_mode"] == "candidate_publication":
        return registry.resolve(_typed_ref(run_result_body["primary_claim_card_ref"]))
    return registry.resolve(_typed_ref(run_result_body["primary_abstention_ref"]))


def resolve_prediction_artifact(
    run_result_body: Mapping[str, Any], registry: ManifestRegistry
) -> RegisteredManifest:
    return registry.resolve(_typed_ref(run_result_body["prediction_artifact_refs"][0]))


def resolve_point_score_result(
    run_result: RegisteredManifest, registry: ManifestRegistry
) -> RegisteredManifest:
    return resolve_primary_score_result(run_result, registry)


def resolve_primary_score_result(
    run_result: RegisteredManifest, registry: ManifestRegistry
) -> RegisteredManifest:
    run_result_body = run_result.manifest.body
    primary_score_result_ref = run_result_body.get("primary_score_result_ref")
    if isinstance(primary_score_result_ref, Mapping):
        return registry.resolve(_typed_ref(primary_score_result_ref))
    scorecard = resolve_scorecard(run_result.manifest.body, registry)
    return registry.resolve(
        _typed_ref(scorecard.manifest.body["point_score_result_ref"])
    )


def resolve_primary_calibration_result(
    run_result: RegisteredManifest, registry: ManifestRegistry
) -> RegisteredManifest | None:
    run_result_body = run_result.manifest.body
    calibration_result_ref = run_result_body.get("primary_calibration_result_ref")
    if isinstance(calibration_result_ref, Mapping):
        return registry.resolve(_typed_ref(calibration_result_ref))
    scorecard = resolve_scorecard(run_result.manifest.body, registry)
    calibration_ref = scorecard.manifest.body.get("calibration_result_ref")
    if isinstance(calibration_ref, Mapping):
        return registry.resolve(_typed_ref(calibration_ref))
    return None


def selected_candidate_ref(
    run_result_body: Mapping[str, Any], registry: ManifestRegistry
) -> TypedRef:
    scorecard = resolve_scorecard(run_result_body, registry)
    return _typed_ref(scorecard.manifest.body["candidate_ref"])


def required_manifest_refs_from_run_result(
    run_result_body: Mapping[str, Any],
) -> tuple[TypedRef, ...]:
    probabilistic_support_refs_enabled = (
        str(run_result_body.get("forecast_object_type", "point")) != "point"
        or isinstance(run_result_body.get("primary_score_result_ref"), Mapping)
        or isinstance(run_result_body.get("primary_calibration_result_ref"), Mapping)
    )
    supporting_refs: tuple[TypedRef, ...] = ()
    if probabilistic_support_refs_enabled:
        supporting_refs = tuple(
            _typed_ref(ref_payload)
            for ref_payload in run_result_body.get("prediction_artifact_refs", ())
            if isinstance(ref_payload, Mapping)
        )
    if isinstance(run_result_body.get("primary_validation_scope_ref"), Mapping):
        supporting_refs = (
            *supporting_refs,
            _typed_ref(run_result_body["primary_validation_scope_ref"]),
        )
    if probabilistic_support_refs_enabled and isinstance(
        run_result_body.get("primary_score_result_ref"), Mapping
    ):
        supporting_refs = (
            *supporting_refs,
            _typed_ref(run_result_body["primary_score_result_ref"]),
        )
    if probabilistic_support_refs_enabled and isinstance(
        run_result_body.get("primary_calibration_result_ref"), Mapping
    ):
        supporting_refs = (
            *supporting_refs,
            _typed_ref(run_result_body["primary_calibration_result_ref"]),
        )
    if isinstance(run_result_body.get("primary_external_evidence_ref"), Mapping):
        supporting_refs = (
            *supporting_refs,
            _typed_ref(run_result_body["primary_external_evidence_ref"]),
        )
    if isinstance(run_result_body.get("primary_mechanistic_evidence_ref"), Mapping):
        supporting_refs = (
            *supporting_refs,
            _typed_ref(run_result_body["primary_mechanistic_evidence_ref"]),
        )
    if run_result_body["result_mode"] == "candidate_publication":
        return (
            _typed_ref(run_result_body["primary_reducer_artifact_ref"]),
            _typed_ref(run_result_body["primary_scorecard_ref"]),
            _typed_ref(run_result_body["primary_claim_card_ref"]),
            *supporting_refs,
        )
    return (_typed_ref(run_result_body["primary_abstention_ref"]), *supporting_refs)


def actual_artifact_hashes(
    *,
    bundle: RegisteredManifest,
    run_result: RegisteredManifest,
    registry: ManifestRegistry,
) -> dict[str, str]:
    run_result_body = run_result.manifest.body
    bundle_model = ReproducibilityBundleManifest.from_manifest(bundle.manifest)
    snapshot = registry.resolve(bundle_model.dataset_snapshot_ref)
    feature_view = registry.resolve(bundle_model.feature_view_ref)
    search_plan = registry.resolve(bundle_model.search_plan_ref)
    evaluation_plan = registry.resolve(bundle_model.evaluation_plan_ref)
    scorecard = resolve_scorecard(run_result_body, registry)
    candidate_or_abstention = resolve_candidate_or_abstention(run_result_body, registry)
    prediction_artifact = resolve_prediction_artifact(run_result_body, registry)
    primary_score_result = resolve_primary_score_result(run_result, registry)
    primary_calibration_result = resolve_primary_calibration_result(
        run_result, registry
    )
    primary_validation_scope_ref = run_result_body.get("primary_validation_scope_ref")
    primary_validation_scope = (
        registry.resolve(_typed_ref(primary_validation_scope_ref))
        if isinstance(primary_validation_scope_ref, Mapping)
        else None
    )
    primary_external_evidence_ref = run_result_body.get(
        "primary_external_evidence_ref"
    )
    primary_external_evidence = (
        registry.resolve(_typed_ref(primary_external_evidence_ref))
        if isinstance(primary_external_evidence_ref, Mapping)
        else None
    )
    primary_mechanistic_evidence_ref = run_result_body.get(
        "primary_mechanistic_evidence_ref"
    )
    primary_mechanistic_evidence = (
        registry.resolve(_typed_ref(primary_mechanistic_evidence_ref))
        if isinstance(primary_mechanistic_evidence_ref, Mapping)
        else None
    )
    robustness_report = registry.resolve(
        _typed_ref(run_result_body["robustness_report_refs"][0])
    )
    probabilistic_support_hashes_enabled = (
        str(run_result_body.get("forecast_object_type", "point")) != "point"
        or isinstance(run_result_body.get("primary_score_result_ref"), Mapping)
        or isinstance(run_result_body.get("primary_calibration_result_ref"), Mapping)
    )
    hashes = {
        "dataset_snapshot": snapshot.manifest.content_hash,
        "feature_view": feature_view.manifest.content_hash,
        "search_plan": search_plan.manifest.content_hash,
        "evaluation_plan": evaluation_plan.manifest.content_hash,
        "run_result": run_result.manifest.content_hash,
        "candidate_or_abstention": candidate_or_abstention.manifest.content_hash,
        "scorecard": scorecard.manifest.content_hash,
        "prediction_artifact": prediction_artifact.manifest.content_hash,
        "robustness_report": robustness_report.manifest.content_hash,
    }
    if primary_validation_scope is not None:
        hashes["validation_scope"] = primary_validation_scope.manifest.content_hash
    if (
        probabilistic_support_hashes_enabled
        and primary_score_result.manifest.schema_name
        != "point_score_result_manifest@1.0.0"
    ):
        hashes["score_result"] = primary_score_result.manifest.content_hash
    if probabilistic_support_hashes_enabled and primary_calibration_result is not None:
        hashes["calibration_result"] = primary_calibration_result.manifest.content_hash
    if primary_external_evidence is not None:
        hashes["external_evidence"] = primary_external_evidence.manifest.content_hash
    if primary_mechanistic_evidence is not None:
        hashes["mechanistic_evidence"] = (
            primary_mechanistic_evidence.manifest.content_hash
        )
    return hashes


def _typed_ref(payload: Mapping[str, Any]) -> TypedRef:
    return TypedRef(
        schema_name=str(payload["schema_name"]),
        object_id=str(payload["object_id"]),
    )


def _bundle_manifest(bundle: RegisteredManifest | ManifestEnvelope) -> ManifestEnvelope:
    if isinstance(bundle, ManifestEnvelope):
        return bundle
    return bundle.manifest


def _unique_codes(codes: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for code in codes:
        if code:
            seen.setdefault(str(code), None)
    return tuple(seen)


__all__ = [
    "ReplayInspection",
    "ReplayVerificationResult",
    "ReplayedOutcome",
    "actual_artifact_hashes",
    "build_artifact_hash_records",
    "build_portfolio_replay_contract",
    "build_replay_seed_records",
    "build_replay_stage_order",
    "build_reproducibility_bundle_manifest",
    "build_runtime_environment_metadata",
    "inspect_reproducibility_bundle",
    "required_manifest_refs_for_publication",
    "required_manifest_refs_from_run_result",
    "resolve_candidate_or_abstention",
    "resolve_primary_calibration_result",
    "resolve_primary_score_result",
    "resolve_point_score_result",
    "resolve_prediction_artifact",
    "resolve_scorecard",
    "selected_candidate_ref",
    "verify_portfolio_replay_contract",
    "verify_replayed_outcome",
]
