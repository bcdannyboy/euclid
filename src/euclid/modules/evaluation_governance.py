from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.modules.split_planning import EvaluationPlan

_SCOPE_ID = "euclid_v1_binding_scope@1.0.0"
_OWNER_PROMPT_ID = "prompt.predictive-validation-v1"


@dataclass(frozen=True)
class ComparatorDeclaration:
    comparator_declaration_id: str
    baseline_id: str
    comparator_kind: str
    forecast_object_type: str
    family_id: str
    freeze_rule: str

    def as_dict(self) -> dict[str, str]:
        return {
            "baseline_id": self.baseline_id,
            "comparator_declaration_id": self.comparator_declaration_id,
            "comparator_kind": self.comparator_kind,
            "family_id": self.family_id,
            "forecast_object_type": self.forecast_object_type,
            "freeze_rule": self.freeze_rule,
        }


@dataclass(frozen=True)
class BaselineRegistry:
    compatible_point_score_policy_ref: TypedRef
    declarations: tuple[ComparatorDeclaration, ...]
    baseline_registry_id: str = "prototype_baseline_registry_v1"
    owner_prompt_id: str = _OWNER_PROMPT_ID
    scope_id: str = _SCOPE_ID

    @property
    def primary_baseline_id(self) -> str:
        return self.declarations[0].baseline_id

    @property
    def baseline_ids(self) -> tuple[str, ...]:
        return tuple(declaration.baseline_id for declaration in self.declarations)

    def to_manifest(self, catalog: ContractCatalog) -> ManifestEnvelope:
        return ManifestEnvelope.build(
            schema_name="baseline_registry_manifest@1.1.0",
            module_id="evaluation_governance",
            body={
                "baseline_registry_id": self.baseline_registry_id,
                "owner_prompt_id": self.owner_prompt_id,
                "scope_id": self.scope_id,
                "primary_baseline_id": self.primary_baseline_id,
                "baseline_ids": list(self.baseline_ids),
                "baseline_declarations": [
                    declaration.as_dict() for declaration in self.declarations
                ],
                "compatible_point_score_policy_ref": (
                    self.compatible_point_score_policy_ref.as_dict()
                ),
            },
            catalog=catalog,
        )


@dataclass(frozen=True)
class ComparisonKey:
    forecast_object_type: str
    score_policy_ref: TypedRef
    horizon_set: tuple[int, ...]
    scored_origin_set_id: str
    entity_panel: tuple[str, ...] = ()
    entity_weights: tuple[Mapping[str, Any], ...] = ()
    composition_signature: str | None = None

    def as_dict(self) -> dict[str, Any]:
        body = {
            "forecast_object_type": self.forecast_object_type,
            "horizon_set": list(self.horizon_set),
            "score_policy_ref": self.score_policy_ref.as_dict(),
            "scored_origin_set_id": self.scored_origin_set_id,
        }
        if self.entity_panel:
            body["entity_panel"] = list(self.entity_panel)
        if self.entity_weights:
            body["entity_weights"] = [dict(item) for item in self.entity_weights]
        if self.composition_signature is not None:
            body["composition_signature"] = self.composition_signature
        return body


@dataclass(frozen=True)
class ForecastComparisonPolicy:
    primary_score_policy_ref: TypedRef
    primary_baseline_id: str
    comparison_policy_id: str = "prototype_primary_baseline_comparison_v1"
    owner_prompt_id: str = _OWNER_PROMPT_ID
    scope_id: str = _SCOPE_ID
    comparison_regime_id: str = "unconditional_model_pair"
    required_comparison_key_fields: tuple[str, ...] = (
        "forecast_object_type",
        "score_policy_ref",
        "horizon_set",
        "scored_origin_set_id",
    )
    candidate_must_strictly_beat_baseline: bool = True

    def to_manifest(self, catalog: ContractCatalog) -> ManifestEnvelope:
        return ManifestEnvelope.build(
            schema_name="forecast_comparison_policy_manifest@1.1.0",
            module_id="evaluation_governance",
            body={
                "comparison_policy_id": self.comparison_policy_id,
                "owner_prompt_id": self.owner_prompt_id,
                "scope_id": self.scope_id,
                "comparison_regime_id": self.comparison_regime_id,
                "primary_score_policy_ref": self.primary_score_policy_ref.as_dict(),
                "primary_baseline_id": self.primary_baseline_id,
                "required_comparison_key_fields": list(
                    self.required_comparison_key_fields
                ),
                "candidate_must_strictly_beat_baseline": (
                    self.candidate_must_strictly_beat_baseline
                ),
            },
            catalog=catalog,
        )


@dataclass(frozen=True)
class ComparisonUniverse:
    selected_candidate_id: str
    baseline_id: str
    candidate_primary_score: float
    baseline_primary_score: float
    candidate_comparison_key: ComparisonKey
    baseline_comparison_key: ComparisonKey
    comparison_universe_id: str = "prototype_comparison_universe_v1"
    owner_prompt_id: str = _OWNER_PROMPT_ID
    scope_id: str = _SCOPE_ID
    comparison_class_status: str = "comparable"
    candidate_score_result_ref: TypedRef | None = None
    baseline_score_result_ref: TypedRef | None = None
    comparator_score_result_refs: tuple[TypedRef, ...] = ()
    paired_comparison_records: tuple[Mapping[str, Any], ...] = ()
    practical_significance_margin: float | None = None

    @property
    def candidate_beats_baseline(self) -> bool:
        return self.candidate_primary_score < self.baseline_primary_score

    def to_manifest(self, catalog: ContractCatalog) -> ManifestEnvelope:
        body = {
            "comparison_universe_id": self.comparison_universe_id,
            "owner_prompt_id": self.owner_prompt_id,
            "scope_id": self.scope_id,
            "selected_candidate_id": self.selected_candidate_id,
            "baseline_id": self.baseline_id,
            "candidate_primary_score": self.candidate_primary_score,
            "baseline_primary_score": self.baseline_primary_score,
            "candidate_beats_baseline": self.candidate_beats_baseline,
            "comparison_class_status": self.comparison_class_status,
            "candidate_comparison_key": self.candidate_comparison_key.as_dict(),
            "baseline_comparison_key": self.baseline_comparison_key.as_dict(),
        }
        if self.candidate_score_result_ref is not None:
            body["candidate_score_result_ref"] = (
                self.candidate_score_result_ref.as_dict()
            )
        if self.baseline_score_result_ref is not None:
            body["baseline_score_result_ref"] = self.baseline_score_result_ref.as_dict()
        if self.comparator_score_result_refs:
            body["comparator_score_result_refs"] = [
                ref.as_dict() for ref in self.comparator_score_result_refs
            ]
        if self.paired_comparison_records:
            body["paired_comparison_records"] = [
                dict(record) for record in self.paired_comparison_records
            ]
        if self.practical_significance_margin is not None:
            body["practical_significance_margin"] = self.practical_significance_margin
        return ManifestEnvelope.build(
            schema_name="comparison_universe_manifest@1.0.0",
            module_id="evaluation_governance",
            body=body,
            catalog=catalog,
        )


@dataclass(frozen=True)
class EvaluationEvent:
    event_id: str
    ref_kind: str
    related_object_ref: TypedRef
    event_type: str | None = None
    stage_id: str | None = None
    segment_id: str | None = None
    access_count: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "ref_kind": self.ref_kind,
            "related_object_ref": self.related_object_ref.as_dict(),
        }
        if self.event_type is not None:
            payload["event_type"] = self.event_type
        if self.stage_id is not None:
            payload["stage_id"] = self.stage_id
        if self.segment_id is not None:
            payload["segment_id"] = self.segment_id
        if self.access_count is not None:
            payload["access_count"] = self.access_count
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class EvaluationEventLog:
    events: tuple[EvaluationEvent, ...]
    stage_bookkeeping: tuple[Mapping[str, Any], ...] = ()
    search_isolation_evidence: Mapping[str, Any] = field(default_factory=dict)
    confirmatory_access_count: int = 0
    post_holdout_mutation_count: int = 0
    confirmatory_segment_status: str = "sealed_pre_access"
    evaluation_event_log_id: str = "prototype_evaluation_event_log_v1"
    owner_prompt_id: str = _OWNER_PROMPT_ID
    scope_id: str = _SCOPE_ID

    def to_manifest(self, catalog: ContractCatalog) -> ManifestEnvelope:
        return ManifestEnvelope.build(
            schema_name="evaluation_event_log_manifest@1.0.0",
            module_id="evaluation_governance",
            body={
                "evaluation_event_log_id": self.evaluation_event_log_id,
                "owner_prompt_id": self.owner_prompt_id,
                "scope_id": self.scope_id,
                "events": [event.as_dict() for event in self.events],
                "stage_bookkeeping": [
                    dict(record) for record in self.stage_bookkeeping
                ],
                "search_isolation_evidence": dict(self.search_isolation_evidence),
                "confirmatory_access_count": self.confirmatory_access_count,
                "post_holdout_mutation_count": self.post_holdout_mutation_count,
                "confirmatory_segment_status": self.confirmatory_segment_status,
            },
            catalog=catalog,
        )


@dataclass(frozen=True)
class EvaluationGovernance:
    comparison_universe_ref: TypedRef
    event_log_ref: TypedRef
    freeze_event_ref: TypedRef
    frozen_shortlist_ref: TypedRef
    confirmatory_promotion_allowed: bool
    evaluation_governance_id: str = "prototype_evaluation_governance_v1"
    owner_prompt_id: str = _OWNER_PROMPT_ID
    scope_id: str = _SCOPE_ID
    comparison_regime_id: str = "unconditional_model_pair"
    many_model_adjustment_id: str = "none"
    holdout_reuse_policy: str = "single_use_without_reusable_holdout_mechanism"
    post_holdout_mutation_rule: str = "forbidden_after_holdout_exposure"

    def to_manifest(self, catalog: ContractCatalog) -> ManifestEnvelope:
        return ManifestEnvelope.build(
            schema_name="evaluation_governance_manifest@1.1.0",
            module_id="evaluation_governance",
            body={
                "evaluation_governance_id": self.evaluation_governance_id,
                "owner_prompt_id": self.owner_prompt_id,
                "scope_id": self.scope_id,
                "comparison_universe_ref": self.comparison_universe_ref.as_dict(),
                "event_log_ref": self.event_log_ref.as_dict(),
                "freeze_event_ref": self.freeze_event_ref.as_dict(),
                "frozen_shortlist_ref": self.frozen_shortlist_ref.as_dict(),
                "comparison_regime_id": self.comparison_regime_id,
                "many_model_adjustment_id": self.many_model_adjustment_id,
                "holdout_reuse_policy": self.holdout_reuse_policy,
                "post_holdout_mutation_rule": self.post_holdout_mutation_rule,
                "confirmatory_promotion_allowed": (self.confirmatory_promotion_allowed),
            },
            catalog=catalog,
        )


@dataclass(frozen=True)
class PredictiveGatePolicy:
    allowed_forecast_object_types: tuple[str, ...]
    policy_id: str = "prototype_point_predictive_gate_v1"
    owner_prompt_id: str = _OWNER_PROMPT_ID
    scope_id: str = _SCOPE_ID
    required_time_safety_statuses: tuple[str, ...] = ("verified",)
    required_confirmatory_statuses: tuple[str, ...] = ("clean", "replicated")
    minimum_outer_improvement_rule: str = "nonnegative_mean_primary_score_improvement"
    minimum_holdout_improvement_rule: str = "nonnegative_mean_primary_score_improvement"
    pairwise_confirmation_rule: str = "declared_pairwise_test_passes"
    requires_calibration_pass: bool = False

    @property
    def forecast_object_type(self) -> str:
        if len(self.allowed_forecast_object_types) != 1:
            raise ContractValidationError(
                code="ambiguous_forecast_object_type",
                message="policy must declare exactly one forecast object type",
                field_path="allowed_forecast_object_types",
            )
        return self.allowed_forecast_object_types[0]

    def to_manifest(self, catalog: ContractCatalog) -> ManifestEnvelope:
        return ManifestEnvelope.build(
            schema_name="predictive_gate_policy_manifest@1.1.0",
            module_id="evaluation_governance",
            body={
                "policy_id": self.policy_id,
                "owner_prompt_id": self.owner_prompt_id,
                "scope_id": self.scope_id,
                "allowed_forecast_object_types": list(
                    self.allowed_forecast_object_types
                ),
                "required_time_safety_statuses": list(
                    self.required_time_safety_statuses
                ),
                "required_confirmatory_statuses": list(
                    self.required_confirmatory_statuses
                ),
                "minimum_outer_improvement_rule": (self.minimum_outer_improvement_rule),
                "minimum_holdout_improvement_rule": (
                    self.minimum_holdout_improvement_rule
                ),
                "pairwise_confirmation_rule": self.pairwise_confirmation_rule,
                "requires_calibration_pass": self.requires_calibration_pass,
            },
            catalog=catalog,
        )


def build_baseline_registry(
    *,
    compatible_point_score_policy_ref: TypedRef,
    primary_baseline_id: str = "constant_baseline",
) -> BaselineRegistry:
    declaration = ComparatorDeclaration(
        comparator_declaration_id=f"{primary_baseline_id}_declaration",
        baseline_id=primary_baseline_id,
        comparator_kind="baseline",
        forecast_object_type="point",
        family_id="constant",
        freeze_rule="frozen_before_confirmatory_access",
    )
    return BaselineRegistry(
        compatible_point_score_policy_ref=compatible_point_score_policy_ref,
        declarations=(declaration,),
    )


def build_comparison_key(
    *,
    evaluation_plan: EvaluationPlan,
    score_policy_ref: TypedRef,
    forecast_object_type: str = "point",
    composition_signature: str | None = None,
) -> ComparisonKey:
    return ComparisonKey(
        forecast_object_type=forecast_object_type,
        score_policy_ref=score_policy_ref,
        horizon_set=evaluation_plan.horizon_set,
        scored_origin_set_id=evaluation_plan.scored_origin_set_id,
        entity_panel=(
            evaluation_plan.entity_panel
            if len(evaluation_plan.entity_panel) > 1
            else ()
        ),
        composition_signature=composition_signature,
    )


def build_forecast_comparison_policy(
    *,
    primary_score_policy_ref: TypedRef,
    primary_baseline_id: str,
) -> ForecastComparisonPolicy:
    return ForecastComparisonPolicy(
        primary_score_policy_ref=primary_score_policy_ref,
        primary_baseline_id=primary_baseline_id,
    )


def build_comparison_universe(
    *,
    selected_candidate_id: str,
    baseline_id: str,
    candidate_primary_score: float,
    baseline_primary_score: float,
    candidate_comparison_key: ComparisonKey,
    baseline_comparison_key: ComparisonKey,
    candidate_score_result_ref: TypedRef | None = None,
    baseline_score_result_ref: TypedRef | None = None,
    comparator_score_result_refs: tuple[TypedRef, ...] = (),
    paired_comparison_records: tuple[Mapping[str, Any], ...] = (),
    practical_significance_margin: float | None = None,
) -> ComparisonUniverse:
    _require_matching_comparison_keys(
        candidate_key=candidate_comparison_key,
        baseline_key=baseline_comparison_key,
    )
    return ComparisonUniverse(
        selected_candidate_id=selected_candidate_id,
        baseline_id=baseline_id,
        candidate_primary_score=candidate_primary_score,
        baseline_primary_score=baseline_primary_score,
        candidate_comparison_key=candidate_comparison_key,
        baseline_comparison_key=baseline_comparison_key,
        candidate_score_result_ref=candidate_score_result_ref,
        baseline_score_result_ref=baseline_score_result_ref,
        comparator_score_result_refs=comparator_score_result_refs,
        paired_comparison_records=paired_comparison_records,
        practical_significance_margin=practical_significance_margin,
    )


def build_evaluation_event_log(
    *,
    search_plan_ref: TypedRef,
    frozen_shortlist_ref: TypedRef,
    freeze_event_ref: TypedRef,
    freeze_event_manifest: ManifestEnvelope | None = None,
    comparison_universe_ref: TypedRef | None = None,
    prediction_artifact_ref: TypedRef | None = None,
    run_result_ref: TypedRef | None = None,
    search_local_segment_ids: tuple[str, ...] = (),
    confirmatory_segment_id: str = "confirmatory_holdout",
    holdout_access_count: int = 0,
    pre_holdout_analytic_rerun_segment_ids: tuple[str, ...] = (),
    nonanalytic_retry_segment_ids: tuple[str, ...] = (),
    replication_segment_ids: tuple[str, ...] = (),
    post_holdout_mutation_count: int = 0,
) -> EvaluationEventLog:
    effective_holdout_access_count = _resolve_holdout_access_count(
        holdout_access_count=holdout_access_count,
        prediction_artifact_ref=prediction_artifact_ref,
    )
    freeze_evidence = _freeze_event_evidence(
        freeze_event_ref=freeze_event_ref,
        freeze_event_manifest=freeze_event_manifest,
    )
    _validate_confirmatory_event_log_request(
        freeze_evidence=freeze_evidence,
        effective_holdout_access_count=effective_holdout_access_count,
        replication_segment_ids=replication_segment_ids,
        post_holdout_mutation_count=post_holdout_mutation_count,
    )

    events: list[EvaluationEvent] = [
        EvaluationEvent(
            event_id="search_plan_frozen",
            ref_kind="search_plan",
            related_object_ref=search_plan_ref,
            stage_id="plan_freeze",
        ),
        EvaluationEvent(
            event_id="shortlist_frozen",
            ref_kind="frozen_shortlist",
            related_object_ref=frozen_shortlist_ref,
            stage_id="global_pair_freeze_pre_holdout",
            segment_id="global_pair_freeze_pre_holdout",
        ),
        EvaluationEvent(
            event_id="candidate_frozen",
            ref_kind="freeze_event",
            related_object_ref=freeze_event_ref,
            stage_id="global_pair_freeze_pre_holdout",
            segment_id="global_pair_freeze_pre_holdout",
        ),
    ]
    for index, segment_id in enumerate(pre_holdout_analytic_rerun_segment_ids):
        events.append(
            EvaluationEvent(
                event_id=f"pre_holdout_analytic_rerun_{index}",
                event_type="pre_holdout_analytic_rerun",
                ref_kind="search_plan",
                related_object_ref=search_plan_ref,
                stage_id="inner_search",
                segment_id=segment_id,
                details={"rerun_scope": "analytic_only"},
            )
        )
    for index, segment_id in enumerate(nonanalytic_retry_segment_ids):
        events.append(
            EvaluationEvent(
                event_id=f"nonanalytic_retry_{index}",
                event_type="nonanalytic_retry",
                ref_kind="search_plan",
                related_object_ref=search_plan_ref,
                stage_id="inner_search",
                segment_id=segment_id,
                details={"retry_scope": "nonanalytic_backend"},
            )
        )
    if comparison_universe_ref is not None:
        events.append(
            EvaluationEvent(
                event_id="comparison_universe_bound",
                ref_kind="comparison_universe",
                related_object_ref=comparison_universe_ref,
                stage_id="global_pair_freeze_pre_holdout",
                segment_id="global_pair_freeze_pre_holdout",
            )
        )
    if effective_holdout_access_count > 0:
        events.append(
            EvaluationEvent(
                event_id="confirmatory_holdout_materialized",
                event_type="holdout_materialized",
                ref_kind="freeze_event",
                related_object_ref=freeze_event_ref,
                stage_id="confirmatory_holdout",
                segment_id=confirmatory_segment_id,
                access_count=effective_holdout_access_count,
                details={
                    "confirmatory_holdout_influenced_search": False,
                    "holdout_materialized_before_freeze": freeze_evidence[
                        "holdout_materialized_before_freeze"
                    ],
                },
            )
        )
    if prediction_artifact_ref is not None:
        events.append(
            EvaluationEvent(
                event_id="confirmatory_prediction_emitted",
                ref_kind="prediction_artifact",
                related_object_ref=prediction_artifact_ref,
                stage_id="confirmatory_holdout",
                segment_id=confirmatory_segment_id,
            )
        )
    if post_holdout_mutation_count > 0:
        events.append(
            EvaluationEvent(
                event_id="post_holdout_mutation_detected",
                event_type="post_holdout_mutation",
                ref_kind="freeze_event",
                related_object_ref=freeze_event_ref,
                stage_id="confirmatory_holdout",
                segment_id=confirmatory_segment_id,
                details={"post_holdout_mutation_count": post_holdout_mutation_count},
            )
        )
    for index, segment_id in enumerate(replication_segment_ids):
        events.append(
            EvaluationEvent(
                event_id=f"fresh_replication_started_{index}",
                event_type="fresh_replication_started",
                ref_kind="search_plan",
                related_object_ref=search_plan_ref,
                stage_id="fresh_replication",
                segment_id=segment_id,
            )
        )
        events.append(
            EvaluationEvent(
                event_id=f"fresh_replication_completed_{index}",
                event_type="fresh_replication_completed",
                ref_kind="freeze_event",
                related_object_ref=freeze_event_ref,
                stage_id="fresh_replication",
                segment_id=segment_id,
            )
        )
    if run_result_ref is not None:
        events.append(
            EvaluationEvent(
                event_id="run_result_published",
                ref_kind="run_result",
                related_object_ref=run_result_ref,
                stage_id=(
                    "fresh_replication"
                    if replication_segment_ids
                    else "confirmatory_holdout"
                ),
                segment_id=(
                    replication_segment_ids[-1]
                    if replication_segment_ids
                    else confirmatory_segment_id
                ),
            )
        )
    return EvaluationEventLog(
        events=tuple(events),
        stage_bookkeeping=_build_stage_bookkeeping(
            search_local_segment_ids=search_local_segment_ids,
            confirmatory_segment_id=confirmatory_segment_id,
            effective_holdout_access_count=effective_holdout_access_count,
            replication_segment_ids=replication_segment_ids,
        ),
        search_isolation_evidence={
            "search_plan_ref": search_plan_ref.as_dict(),
            "frozen_shortlist_ref": frozen_shortlist_ref.as_dict(),
            "freeze_event_ref": freeze_event_ref.as_dict(),
            "search_local_segment_ids": list(search_local_segment_ids),
            "holdout_materialized_before_freeze": freeze_evidence[
                "holdout_materialized_before_freeze"
            ],
            "post_freeze_candidate_mutation_count": freeze_evidence[
                "post_freeze_candidate_mutation_count"
            ],
            "confirmatory_holdout_influenced_search": False,
        },
        confirmatory_access_count=effective_holdout_access_count,
        post_holdout_mutation_count=post_holdout_mutation_count,
        confirmatory_segment_status=_confirmatory_segment_status(
            effective_holdout_access_count=effective_holdout_access_count,
            replication_segment_ids=replication_segment_ids,
            post_holdout_mutation_count=post_holdout_mutation_count,
        ),
    )


def build_evaluation_governance(
    *,
    comparison_universe_ref: TypedRef,
    event_log_ref: TypedRef,
    freeze_event_ref: TypedRef,
    frozen_shortlist_ref: TypedRef,
    confirmatory_promotion_allowed: bool,
) -> EvaluationGovernance:
    return EvaluationGovernance(
        comparison_universe_ref=comparison_universe_ref,
        event_log_ref=event_log_ref,
        freeze_event_ref=freeze_event_ref,
        frozen_shortlist_ref=frozen_shortlist_ref,
        confirmatory_promotion_allowed=confirmatory_promotion_allowed,
    )


def build_predictive_gate_policy(
    *,
    allowed_forecast_object_types: tuple[str, ...] = ("point",),
) -> PredictiveGatePolicy:
    unique_types = tuple(dict.fromkeys(allowed_forecast_object_types))
    if not unique_types:
        raise ContractValidationError(
            code="missing_forecast_object_type",
            message=(
                "predictive gate policies must declare at least one forecast "
                "object type"
            ),
            field_path="allowed_forecast_object_types",
        )
    requires_calibration_pass = any(
        forecast_object_type != "point" for forecast_object_type in unique_types
    )
    policy_id = (
        "probabilistic_predictive_gate_v1"
        if requires_calibration_pass
        else "prototype_point_predictive_gate_v1"
    )
    return PredictiveGatePolicy(
        allowed_forecast_object_types=unique_types,
        policy_id=policy_id,
        requires_calibration_pass=requires_calibration_pass,
    )


def resolve_confirmatory_promotion_allowed(
    *,
    candidate_beats_baseline: bool,
    predictive_gate_policy_manifest: ManifestEnvelope,
    calibration_result_manifest: ManifestEnvelope | None = None,
) -> bool:
    if not candidate_beats_baseline:
        return False
    requires_calibration_pass = bool(
        predictive_gate_policy_manifest.body.get("requires_calibration_pass", False)
    )
    if not requires_calibration_pass:
        return True
    if calibration_result_manifest is None:
        return False
    return bool(calibration_result_manifest.body.get("pass") is True)


def _require_matching_comparison_keys(
    *,
    candidate_key: ComparisonKey,
    baseline_key: ComparisonKey,
) -> None:
    mismatches: dict[str, Any] = {}
    if candidate_key.forecast_object_type != baseline_key.forecast_object_type:
        mismatches["forecast_object_type"] = {
            "candidate": candidate_key.forecast_object_type,
            "baseline": baseline_key.forecast_object_type,
        }
    if candidate_key.score_policy_ref != baseline_key.score_policy_ref:
        mismatches["score_policy_ref"] = {
            "candidate": candidate_key.score_policy_ref.as_dict(),
            "baseline": baseline_key.score_policy_ref.as_dict(),
        }
    if candidate_key.horizon_set != baseline_key.horizon_set:
        mismatches["horizon_set"] = {
            "candidate": list(candidate_key.horizon_set),
            "baseline": list(baseline_key.horizon_set),
        }
    if candidate_key.scored_origin_set_id != baseline_key.scored_origin_set_id:
        mismatches["scored_origin_set_id"] = {
            "candidate": candidate_key.scored_origin_set_id,
            "baseline": baseline_key.scored_origin_set_id,
        }
    if candidate_key.entity_panel != baseline_key.entity_panel:
        mismatches["entity_panel"] = {
            "candidate": list(candidate_key.entity_panel),
            "baseline": list(baseline_key.entity_panel),
        }
    if candidate_key.entity_weights != baseline_key.entity_weights:
        mismatches["entity_weights"] = {
            "candidate": [dict(item) for item in candidate_key.entity_weights],
            "baseline": [dict(item) for item in baseline_key.entity_weights],
        }
    if (
        candidate_key.composition_signature is not None
        and baseline_key.composition_signature is not None
        and candidate_key.composition_signature != baseline_key.composition_signature
    ):
        mismatches["composition_signature"] = {
            "candidate": candidate_key.composition_signature,
            "baseline": baseline_key.composition_signature,
        }
    if mismatches:
        raise ContractValidationError(
            code="comparison_key_mismatch",
            message=(
                "candidate and baseline are comparable only when forecast "
                "object type, score policy, horizon set, scored-origin set, "
                "declared entity panel geometry, and any explicit composition "
                "signature all match; cross-object comparison requires an "
                "explicit reduction contract"
            ),
            field_path="comparison_key",
            details=mismatches,
        )


def _resolve_holdout_access_count(
    *,
    holdout_access_count: int,
    prediction_artifact_ref: TypedRef | None,
) -> int:
    if holdout_access_count < 0:
        raise ContractValidationError(
            code="invalid_confirmatory_access_count",
            message="holdout_access_count must be >= 0",
            field_path="holdout_access_count",
        )
    if holdout_access_count == 0 and prediction_artifact_ref is not None:
        return 1
    return holdout_access_count


def _freeze_event_evidence(
    *,
    freeze_event_ref: TypedRef,
    freeze_event_manifest: ManifestEnvelope | None,
) -> dict[str, Any]:
    evidence = {
        "holdout_materialized_before_freeze": False,
        "post_freeze_candidate_mutation_count": 0,
    }
    if freeze_event_manifest is None:
        return evidence
    if freeze_event_manifest.ref != freeze_event_ref:
        raise ContractValidationError(
            code="freeze_event_ref_mismatch",
            message="freeze_event_manifest must match freeze_event_ref",
            field_path="freeze_event_manifest.ref",
            details={
                "freeze_event_ref": freeze_event_ref.as_dict(),
                "freeze_event_manifest_ref": freeze_event_manifest.ref.as_dict(),
            },
        )
    if freeze_event_manifest.schema_name != "freeze_event_manifest@1.0.0":
        raise ContractValidationError(
            code="invalid_freeze_event_manifest",
            message="freeze_event_manifest must use freeze_event_manifest@1.0.0",
            field_path="freeze_event_manifest.schema_name",
            details={"schema_name": freeze_event_manifest.schema_name},
        )
    return {
        "holdout_materialized_before_freeze": bool(
            freeze_event_manifest.body["holdout_materialized_before_freeze"]
        ),
        "post_freeze_candidate_mutation_count": int(
            freeze_event_manifest.body["post_freeze_candidate_mutation_count"]
        ),
    }


def _validate_confirmatory_event_log_request(
    *,
    freeze_evidence: Mapping[str, Any],
    effective_holdout_access_count: int,
    replication_segment_ids: tuple[str, ...],
    post_holdout_mutation_count: int,
) -> None:
    if freeze_evidence["holdout_materialized_before_freeze"]:
        raise ContractValidationError(
            code="holdout_not_sealed_pre_search",
            message="confirmatory holdout must remain sealed until the pair freeze",
            field_path="freeze_event_manifest.body.holdout_materialized_before_freeze",
        )
    if (
        effective_holdout_access_count > 0
        and freeze_evidence["post_freeze_candidate_mutation_count"] > 0
    ):
        raise ContractValidationError(
            code="confirmatory_pair_not_frozen_pre_holdout",
            message="confirmatory access requires a pair frozen before holdout use",
            field_path="freeze_event_manifest.body.post_freeze_candidate_mutation_count",
            details={
                "post_freeze_candidate_mutation_count": freeze_evidence[
                    "post_freeze_candidate_mutation_count"
                ]
            },
        )
    if effective_holdout_access_count > 1 and not replication_segment_ids:
        raise ContractValidationError(
            code="holdout_exhausted",
            message=(
                "confirmatory holdout is single-use unless fresh replication is sealed"
            ),
            field_path="holdout_access_count",
            details={"holdout_access_count": effective_holdout_access_count},
        )
    if post_holdout_mutation_count < 0:
        raise ContractValidationError(
            code="invalid_post_holdout_mutation_count",
            message="post_holdout_mutation_count must be >= 0",
            field_path="post_holdout_mutation_count",
        )
    if post_holdout_mutation_count > 0 and not replication_segment_ids:
        raise ContractValidationError(
            code="post_holdout_mutation_detected",
            message=(
                "post-holdout mutation cannot reuse the same confirmatory segment "
                "without a fresh replication segment"
            ),
            field_path="post_holdout_mutation_count",
            details={"post_holdout_mutation_count": post_holdout_mutation_count},
        )


def _build_stage_bookkeeping(
    *,
    search_local_segment_ids: tuple[str, ...],
    confirmatory_segment_id: str,
    effective_holdout_access_count: int,
    replication_segment_ids: tuple[str, ...],
) -> tuple[Mapping[str, Any], ...]:
    bookkeeping: list[Mapping[str, Any]] = [
        {
            "stage_id": "inner_search",
            "stage_kind": "search_local_fitting",
            "segment_ids": list(search_local_segment_ids),
            "status": "completed",
            "uses_confirmatory_holdout": False,
        },
        {
            "stage_id": "global_pair_freeze_pre_holdout",
            "stage_kind": "shortlist_freeze",
            "segment_ids": ["global_pair_freeze_pre_holdout"],
            "status": "completed",
            "uses_confirmatory_holdout": False,
        },
        {
            "stage_id": "confirmatory_holdout",
            "stage_kind": "confirmatory_access",
            "segment_ids": [confirmatory_segment_id],
            "status": (
                "materialized" if effective_holdout_access_count > 0 else "sealed"
            ),
            "access_count": effective_holdout_access_count,
            "uses_confirmatory_holdout": True,
        },
    ]
    for segment_id in replication_segment_ids:
        bookkeeping.append(
            {
                "stage_id": "fresh_replication",
                "stage_kind": "replication_segment",
                "segment_ids": [segment_id],
                "status": "completed",
                "uses_confirmatory_holdout": True,
            }
        )
    return tuple(bookkeeping)


def _confirmatory_segment_status(
    *,
    effective_holdout_access_count: int,
    replication_segment_ids: tuple[str, ...],
    post_holdout_mutation_count: int,
) -> str:
    if replication_segment_ids:
        return "replicated_after_mutation"
    if effective_holdout_access_count > 0:
        return "accessed_once"
    if post_holdout_mutation_count > 0:
        return "mutation_detected"
    return "sealed_pre_access"
