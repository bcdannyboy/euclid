from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.features import (
    FeatureSpec,
    FeatureView,
    default_feature_spec,
    materialize_feature_view,
)
from euclid.ingestion import (
    AdmittedOrderedNumericData,
    ObservationRecord,
    ingest_csv_dataset,
)
from euclid.manifest_registry import ManifestRegistry, RegisteredManifest
from euclid.manifests.runtime_models import (
    CanonicalizationPolicyManifest,
    SearchPlanManifest,
)
from euclid.math.operator_support import (
    CodelengthPolicyObject,
    ObservationModelObject,
    OperatorSupportBundle,
    QuantizationObject,
    ReferenceDescriptionObject,
    TargetTransformObject,
    build_operator_support_bundle,
)
from euclid.modules.external_evidence_ingestion import (
    RegisteredExternalEvidenceBundle,
    register_external_evidence_bundle,
)
from euclid.search_planning import build_canonicalization_policy, build_search_plan
from euclid.snapshotting import FrozenDatasetSnapshot, freeze_dataset_snapshot
from euclid.split_planning import EvaluationPlan, build_evaluation_plan
from euclid.timeguard import TimeSafetyAudit, audit_snapshot_time_safety


@dataclass(frozen=True)
class OperatorIntakePlanningResult:
    admitted_data: AdmittedOrderedNumericData
    observation_records: tuple[ObservationRecord, ...]
    observation_manifests: tuple[RegisteredManifest, ...]
    snapshot_object: FrozenDatasetSnapshot
    snapshot: RegisteredManifest
    time_safety_audit_object: TimeSafetyAudit
    time_safety_audit: RegisteredManifest
    feature_spec_object: FeatureSpec
    feature_spec: RegisteredManifest
    feature_view_object: FeatureView
    feature_view: RegisteredManifest
    evaluation_plan_object: EvaluationPlan
    evaluation_plan: RegisteredManifest
    canonicalization_policy_object: CanonicalizationPolicyManifest
    canonicalization_policy: RegisteredManifest
    search_plan_object: SearchPlanManifest
    search_plan: RegisteredManifest
    external_evidence_bundle: RegisteredExternalEvidenceBundle | None
    support_bundle: OperatorSupportBundle
    target_transform: RegisteredManifest
    base_measure_policy: RegisteredManifest
    reference_description_policy: RegisteredManifest
    observation_model: RegisteredManifest
    codelength_policy: RegisteredManifest

    @property
    def reference_description(self):
        return self.support_bundle.reference_description

    @property
    def target_transform_object(self) -> TargetTransformObject:
        return self.support_bundle.target_transform_object

    @property
    def quantization_object(self) -> QuantizationObject:
        return self.support_bundle.quantization_object

    @property
    def observation_model_object(self) -> ObservationModelObject:
        return self.support_bundle.observation_model_object

    @property
    def reference_description_object(self) -> ReferenceDescriptionObject:
        return self.support_bundle.reference_description_object

    @property
    def codelength_policy_object(self) -> CodelengthPolicyObject:
        return self.support_bundle.codelength_policy_object


def build_operator_intake_plan(
    *,
    csv_path: Path,
    catalog: ContractCatalog,
    registry: ManifestRegistry,
    cutoff_available_at: str | None = None,
    quantization_step: str = "0.5",
    min_train_size: int = 3,
    horizon: int = 1,
    predictive_mode: str = "predictive_requested",
    search_family_ids: tuple[str, ...] = (
        "constant",
        "drift",
        "linear_trend",
        "seasonal_naive",
    ),
    search_class: str = "bounded_heuristic",
    search_seed: str = "0",
    proposal_limit: int | None = None,
    minimum_description_gain_bits: float = 0.0,
    seasonal_period: int = 2,
    forecast_object_type: str = "point",
    external_evidence_payload: dict[str, object] | None = None,
    declared_entity_panel: tuple[str, ...] = (),
) -> OperatorIntakePlanningResult:
    admitted_data = ingest_csv_dataset(csv_path)
    observation_records = admitted_data.observations
    observation_manifests = tuple(
        registry.register(observation.to_manifest(catalog))
        for observation in observation_records
    )

    support_bundle = build_operator_support_bundle(
        catalog=catalog,
        observed_values=admitted_data.coded_targets,
        quantization_step=quantization_step,
    )
    target_transform = registry.register(support_bundle.target_transform_manifest)
    base_measure_policy = registry.register(support_bundle.base_measure_policy_manifest)
    reference_description_policy = registry.register(
        support_bundle.reference_description_policy_manifest
    )
    observation_model = registry.register(
        support_bundle.observation_model_manifest,
        parent_refs=(
            target_transform.manifest.ref,
            base_measure_policy.manifest.ref,
        ),
    )
    codelength_policy = registry.register(
        support_bundle.codelength_policy_manifest,
        parent_refs=(
            target_transform.manifest.ref,
            base_measure_policy.manifest.ref,
            reference_description_policy.manifest.ref,
        ),
    )

    snapshot_object = freeze_dataset_snapshot(
        admitted_data.observations,
        cutoff_available_at=cutoff_available_at,
    )
    observation_manifest_by_hash = {
        item.manifest.body["payload_hash"]: item.manifest.ref
        for item in observation_manifests
    }
    snapshot = registry.register(
        snapshot_object.to_manifest(catalog),
        parent_refs=tuple(
            observation_manifest_by_hash[payload_hash]
            for payload_hash in snapshot_object.lineage_payload_hashes
        ),
    )
    time_safety_audit_object = audit_snapshot_time_safety(snapshot_object)
    time_safety_audit = registry.register(
        time_safety_audit_object.to_manifest(
            catalog,
            snapshot_ref=snapshot.manifest.ref,
        ),
        parent_refs=(snapshot.manifest.ref,),
    )
    feature_spec_object = default_feature_spec()
    feature_spec = registry.register(feature_spec_object.to_manifest(catalog))
    feature_view_object = materialize_feature_view(
        snapshot=snapshot_object,
        audit=time_safety_audit_object,
        feature_spec=feature_spec_object,
    )
    if declared_entity_panel:
        observed_entity_panel = tuple(feature_view_object.entity_panel)
        if tuple(declared_entity_panel) != observed_entity_panel:
            raise ContractValidationError(
                code="unseen_entity_rule_violation",
                message=(
                    "operator intake detected entities outside the declared "
                    "shared-local validation panel"
                ),
                field_path="declared_entity_panel",
                details={
                    "declared_entity_panel": list(declared_entity_panel),
                    "observed_entity_panel": list(observed_entity_panel),
                },
            )
    feature_view = registry.register(
        feature_view_object.to_manifest(
            catalog,
            snapshot_ref=snapshot.manifest.ref,
            feature_spec_ref=feature_spec.manifest.ref,
            time_safety_audit_ref=time_safety_audit.manifest.ref,
        ),
        parent_refs=(
            snapshot.manifest.ref,
            time_safety_audit.manifest.ref,
            feature_spec.manifest.ref,
        ),
    )
    evaluation_plan_object = build_evaluation_plan(
        feature_view=feature_view_object,
        audit=time_safety_audit_object,
        min_train_size=min_train_size,
        horizon=horizon,
        forecast_object_type=forecast_object_type,
    )
    evaluation_plan = registry.register(
        evaluation_plan_object.to_manifest(
            catalog,
            time_safety_audit_ref=time_safety_audit.manifest.ref,
            feature_view_ref=feature_view.manifest.ref,
        ),
        parent_refs=(
            time_safety_audit.manifest.ref,
            feature_view.manifest.ref,
        ),
    )
    canonicalization_policy_object = build_canonicalization_policy()
    canonicalization_policy = registry.register(
        canonicalization_policy_object.to_manifest(catalog)
    )
    search_plan_object = build_search_plan(
        evaluation_plan=evaluation_plan_object,
        canonicalization_policy_ref=canonicalization_policy.manifest.ref,
        codelength_policy_ref=codelength_policy.manifest.ref,
        reference_description_policy_ref=reference_description_policy.manifest.ref,
        observation_model_ref=observation_model.manifest.ref,
        candidate_family_ids=search_family_ids,
        search_class=search_class,
        predictive_mode=predictive_mode,
        random_seed=search_seed,
        proposal_limit=proposal_limit,
        minimum_description_gain_bits=minimum_description_gain_bits,
        seasonal_period=seasonal_period,
    )
    search_plan = registry.register(
        search_plan_object.to_manifest(catalog),
        parent_refs=(
            canonicalization_policy.manifest.ref,
            codelength_policy.manifest.ref,
            reference_description_policy.manifest.ref,
            observation_model.manifest.ref,
            evaluation_plan.manifest.ref,
        ),
    )
    external_evidence_bundle = None
    if external_evidence_payload is not None:
        external_evidence_bundle = register_external_evidence_bundle(
            catalog=catalog,
            registry=registry,
            bundle_id=str(external_evidence_payload["bundle_id"]),
            domain_id=str(external_evidence_payload["domain_id"]),
            acquisition_window=dict(external_evidence_payload["acquisition_window"]),
            raw_sources=tuple(external_evidence_payload["raw_sources"]),
        )
    return OperatorIntakePlanningResult(
        admitted_data=admitted_data,
        observation_records=observation_records,
        observation_manifests=observation_manifests,
        snapshot_object=snapshot_object,
        snapshot=snapshot,
        time_safety_audit_object=time_safety_audit_object,
        time_safety_audit=time_safety_audit,
        feature_spec_object=feature_spec_object,
        feature_spec=feature_spec,
        feature_view_object=feature_view_object,
        feature_view=feature_view,
        evaluation_plan_object=evaluation_plan_object,
        evaluation_plan=evaluation_plan,
        canonicalization_policy_object=canonicalization_policy_object,
        canonicalization_policy=canonicalization_policy,
        search_plan_object=search_plan_object,
        search_plan=search_plan,
        external_evidence_bundle=external_evidence_bundle,
        support_bundle=support_bundle,
        target_transform=target_transform,
        base_measure_policy=base_measure_policy,
        reference_description_policy=reference_description_policy,
        observation_model=observation_model,
        codelength_policy=codelength_policy,
    )


__all__ = [
    "OperatorIntakePlanningResult",
    "build_operator_intake_plan",
]
