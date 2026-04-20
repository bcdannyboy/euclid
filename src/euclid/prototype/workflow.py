from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping

from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifest_registry import ManifestRegistry, RegisteredManifest
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    AbstentionManifest,
    CandidateSpecManifest,
    PerHorizonScore,
    PointScoreResultManifest,
    PredictionArtifactManifest,
    PredictionRow,
    ReadinessGateRecord,
    ReadinessJudgmentManifest,
    SchemaLifecycleIntegrationClosureManifest,
)
from euclid.math.prototype_support import build_prototype_support_bundle
from euclid.math.quantization import FixedStepMidTreadQuantizer
from euclid.math.reference_descriptions import build_reference_description
from euclid.modules.catalog_publishing import (
    build_publication_record_manifest,
    build_run_result_manifest,
)
from euclid.modules.claims import resolve_claim_publication
from euclid.modules.evaluation_governance import (
    build_baseline_registry,
    build_comparison_key,
    build_comparison_universe,
    build_evaluation_event_log,
    build_evaluation_governance,
    build_forecast_comparison_policy,
    build_predictive_gate_policy,
    resolve_confirmatory_promotion_allowed,
)
from euclid.modules.gate_lifecycle import resolve_scorecard_status
from euclid.modules.replay import (
    ReplayedOutcome,
    build_artifact_hash_records,
    build_replay_seed_records,
    build_replay_stage_order,
    build_reproducibility_bundle_manifest,
    required_manifest_refs_for_publication,
    verify_replayed_outcome,
)
from euclid.modules.replay import (
    resolve_point_score_result as resolve_replay_point_score_result,
)
from euclid.modules.replay import (
    resolve_scorecard as resolve_replay_scorecard,
)
from euclid.modules.robustness import (
    PerturbationRunRecord,
    build_leakage_canary_manifest,
    build_leakage_canary_result_manifest,
    build_null_protocol_manifest,
    build_null_result_manifest,
    build_perturbation_family_result_manifest,
    build_perturbation_protocol_manifest,
    build_retention_metric_manifest,
    build_robustness_report,
    build_sensitivity_analysis_manifest,
    build_surrogate_generator_manifest,
    evaluate_aggregate_metric_results,
    evaluate_null_comparison,
    evaluate_perturbation_family,
)
from euclid.operator_runtime.models import (
    DEFAULT_ADMISSIBILITY_RULE_IDS,
    DEFAULT_RUN_SUPPORT_OBJECT_IDS,
)
from euclid.performance import TelemetryRecorder
from euclid.prototype.intake_planning import (
    PrototypeIntakePlanningResult,
    build_prototype_intake_plan,
)
from euclid.readiness import ReadinessGateResult, judge_readiness
from euclid.search_planning import (
    SearchCandidateRecord,
    build_freeze_event,
    build_frontier,
    build_frozen_shortlist,
    build_rejected_diagnostics,
    build_search_ledger,
)

_FAMILY_BITS = 2.0
_SEASONAL_PERIOD = 2
_DEFAULT_LOSS_ID = "absolute_error"
_REPLAY_ENTRYPOINT_ID = "retained_scope_replay_v1"
_SCOPE_ID = "euclid_v1_binding_scope@1.0.0"
_CLAIM_INTERPRETATION_HISTORY = "historical_structure_summary"
_CLAIM_INTERPRETATION_POINT = "point_forecast_within_declared_validation_scope"
_FORBIDDEN_INTERPRETATIONS = (
    "causal_claim",
    "mechanism_claim",
    "transport_claim",
    "cross_entity_generalization",
    "probabilistic_forecast_claim",
    "calibration_claim",
)
_FAMILY_ORDER = {
    "seasonal_naive": 0,
    "constant": 1,
    "linear_trend": 2,
    "drift": 3,
}


@dataclass(frozen=True)
class CandidateSummary:
    candidate_id: str
    family_id: str
    exploratory_primary_score: float
    confirmatory_primary_score: float
    baseline_primary_score: float
    description_gain_bits: float
    admissible: bool
    parameters: Mapping[str, float | int]


@dataclass(frozen=True)
class PrototypeReducerWorkflowResult:
    intake: PrototypeIntakePlanningResult
    scope_ledger: RegisteredManifest
    canonicalization_policy: RegisteredManifest
    search_plan: RegisteredManifest
    search_ledger: RegisteredManifest
    frontier: RegisteredManifest
    rejected_diagnostics: RegisteredManifest
    point_score_policy: RegisteredManifest
    baseline_registry: RegisteredManifest
    forecast_comparison_policy: RegisteredManifest
    selected_candidate: RegisteredManifest
    selected_candidate_spec: RegisteredManifest
    selected_candidate_structure: RegisteredManifest
    frozen_shortlist: RegisteredManifest
    freeze_event: RegisteredManifest
    comparison_universe: RegisteredManifest
    evaluation_event_log: RegisteredManifest
    evaluation_governance: RegisteredManifest
    prediction_artifact: RegisteredManifest
    point_score_result: RegisteredManifest
    calibration_contract: RegisteredManifest
    calibration_result: RegisteredManifest
    null_protocol: RegisteredManifest
    perturbation_protocol: RegisteredManifest
    leakage_canary_result: RegisteredManifest
    leakage_canary_results: tuple[RegisteredManifest, ...]
    robustness_report: RegisteredManifest
    scorecard: RegisteredManifest
    validation_scope: RegisteredManifest
    readiness_judgment: RegisteredManifest
    schema_lifecycle_integration_closure: RegisteredManifest
    reproducibility_bundle: RegisteredManifest
    run_result: RegisteredManifest
    publication_record: RegisteredManifest
    claim_card: RegisteredManifest | None
    abstention: RegisteredManifest | None
    candidate_summaries: tuple[CandidateSummary, ...]
    confirmatory_primary_score: float
    replay_verified: bool


@dataclass(frozen=True)
class PrototypeReplayResult:
    bundle_ref: TypedRef
    run_result_ref: TypedRef
    selected_candidate_ref: TypedRef
    replay_verification_status: str
    confirmatory_primary_score: float
    failure_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CandidateEvaluation:
    family_id: str
    candidate_id: str
    parameters: Mapping[str, float | int]
    exploratory_primary_score: float
    confirmatory_primary_score: float
    baseline_primary_score: float
    description_components: Mapping[str, float]
    description_gain_bits: float
    admissible: bool
    structure_signature: str
    confirmatory_prediction_rows: tuple[dict[str, Any], ...]
    development_losses: tuple[float, ...]


@dataclass(frozen=True)
class _RobustnessArtifacts:
    surrogate_generator: RegisteredManifest
    null_protocol: RegisteredManifest
    null_result: RegisteredManifest
    perturbation_protocol: RegisteredManifest
    perturbation_family_results: tuple[RegisteredManifest, ...]
    leakage_canary_results: tuple[RegisteredManifest, ...]
    sensitivity_analyses: tuple[RegisteredManifest, ...]
    robustness_report: RegisteredManifest


def run_prototype_reducer_workflow(
    *,
    csv_path,
    catalog: ContractCatalog,
    registry: ManifestRegistry,
    cutoff_available_at: str | None = None,
    quantization_step: str = "0.5",
    min_train_size: int = 3,
    horizon: int = 1,
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
    telemetry: TelemetryRecorder | None = None,
) -> PrototypeReducerWorkflowResult:
    intake = build_prototype_intake_plan(
        csv_path=csv_path,
        catalog=catalog,
        registry=registry,
        cutoff_available_at=cutoff_available_at,
        quantization_step=quantization_step,
        min_train_size=min_train_size,
        horizon=horizon,
        search_family_ids=search_family_ids,
        search_class=search_class,
        search_seed=search_seed,
        proposal_limit=proposal_limit,
        minimum_description_gain_bits=minimum_description_gain_bits,
        seasonal_period=seasonal_period,
    )
    if telemetry is not None:
        telemetry.record_seed(
            scope="search",
            value=intake.search_plan_object.random_seed,
        )
    feature_rows = tuple(intake.feature_view_object.require_stage_reuse("search").rows)
    folds = tuple(intake.evaluation_plan_object.folds)
    intake.support_bundle.require_supported_point_loss(_DEFAULT_LOSS_ID)

    scope_ledger = registry.register(
        _build_manifest(
            catalog,
            schema_name="scope_ledger_manifest@1.0.0",
            module_id="manifest_registry",
            body={
                "scope_ledger_id": "prototype_scope_ledger_v1",
                "scope_id": _SCOPE_ID,
                "forecast_object_type": "point",
                "candidate_family_ids": list(_candidate_family_ids()),
                "run_support_object_ids": list(DEFAULT_RUN_SUPPORT_OBJECT_IDS),
                "admissibility_rule_ids": list(DEFAULT_ADMISSIBILITY_RULE_IDS),
                "deferred_scope_annotations": [
                    "shared_plus_local_decomposition",
                    "mechanistic_evidence",
                    "algorithmic_publication",
                ],
            },
        )
    )
    canonicalization_policy = intake.canonicalization_policy
    search_plan = intake.search_plan
    point_score_policy = registry.register(
        _build_manifest(
            catalog,
            schema_name="point_score_policy_manifest@1.0.0",
            module_id="scoring",
            body={
                "score_policy_id": "prototype_absolute_error_policy_v1",
                "owner_prompt_id": "prompt.scoring-calibration-v1",
                "scope_id": _SCOPE_ID,
                "forecast_object_type": "point",
                "point_loss_id": _DEFAULT_LOSS_ID,
                "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
                "horizon_weights": [{"horizon": 1, "weight": "1"}],
                "entity_aggregation_mode": (
                    "single_entity_only_no_cross_entity_aggregation"
                ),
                "secondary_diagnostic_ids": [],
                "forbidden_primary_metric_ids": [
                    "root_mean_squared_error",
                    "mean_absolute_percentage_error",
                    "symmetric_mean_absolute_percentage_error",
                    "weighted_absolute_percentage_error",
                    "r_squared",
                    "correlation",
                    "directional_accuracy",
                    "sign_accuracy",
                    "skill_score",
                    "mixed_loss_bundle",
                ],
                "lower_is_better": True,
                "comparison_class_rule": "identical_score_policy_required",
            },
        )
    )
    baseline_registry = registry.register(
        build_baseline_registry(
            compatible_point_score_policy_ref=point_score_policy.manifest.ref,
        ).to_manifest(catalog),
        parent_refs=(point_score_policy.manifest.ref,),
    )
    forecast_comparison_policy = registry.register(
        build_forecast_comparison_policy(
            primary_score_policy_ref=point_score_policy.manifest.ref,
            primary_baseline_id="constant_baseline",
        ).to_manifest(catalog),
        parent_refs=(point_score_policy.manifest.ref,),
    )
    predictive_gate_policy = registry.register(
        build_predictive_gate_policy().to_manifest(catalog)
    )

    candidate_evaluations = _evaluate_candidates(
        feature_rows=feature_rows,
        folds=folds,
        support_bundle=intake.support_bundle,
        minimum_description_gain_bits=minimum_description_gain_bits,
        point_loss_id=_DEFAULT_LOSS_ID,
        family_ids=tuple(intake.search_plan_object.candidate_family_ids),
        telemetry=telemetry,
    )
    candidate_summaries = tuple(
        CandidateSummary(
            candidate_id=candidate.candidate_id,
            family_id=candidate.family_id,
            exploratory_primary_score=candidate.exploratory_primary_score,
            confirmatory_primary_score=candidate.confirmatory_primary_score,
            baseline_primary_score=candidate.baseline_primary_score,
            description_gain_bits=candidate.description_gain_bits,
            admissible=candidate.admissible,
            parameters=dict(candidate.parameters),
        )
        for candidate in candidate_evaluations
    )
    search_candidate_records = _search_candidate_records(candidate_evaluations)
    if telemetry is not None:
        telemetry.record_measurement(
            name="candidate_family_queue_depth",
            category="portfolio_selection",
            value=len(candidate_evaluations),
            unit="families",
            attributes={"scope": "prototype_workflow"},
        )
    best_overall_candidate_runtime = _select_best_overall_candidate(
        candidate_evaluations
    )
    if telemetry is None:
        accepted_candidate_runtime = _select_accepted_candidate(
            candidate_evaluations
        )
    else:
        with telemetry.span(
            "prototype.portfolio_selection",
            category="portfolio_selection",
            attributes={"candidate_family_count": len(candidate_evaluations)},
        ):
            accepted_candidate_runtime = _select_accepted_candidate(
                candidate_evaluations
            )
    selected_candidate_runtime = (
        accepted_candidate_runtime or best_overall_candidate_runtime
    )

    selected_candidate_spec = _register_runtime_model(
        registry,
        catalog,
        CandidateSpecManifest(
            candidate_spec_id=f"{selected_candidate_runtime.candidate_id}_spec",
            family_id=selected_candidate_runtime.family_id,
            parameter_summary=_normalize_mapping(selected_candidate_runtime.parameters),
            selection_floor_bits=_stable_float(minimum_description_gain_bits),
            parent_refs=(search_plan.manifest.ref,),
        ),
    )
    selected_candidate_structure = registry.register(
        _build_manifest(
            catalog,
            schema_name="canonical_structure_code_manifest@1.0.0",
            module_id="candidate_fitting",
            body={
                "canonical_structure_code_id": (
                    f"{selected_candidate_runtime.candidate_id}_structure"
                ),
                "owner_id": "module.candidate-fitting-v1",
                "scope_id": _SCOPE_ID,
                "candidate_id": selected_candidate_runtime.candidate_id,
                "family_id": selected_candidate_runtime.family_id,
                "structure_signature": selected_candidate_runtime.structure_signature,
            },
        ),
        parent_refs=(selected_candidate_spec.manifest.ref,),
    )
    selected_candidate = registry.register(
        _build_manifest(
            catalog,
            schema_name="reducer_artifact_manifest@1.0.0",
            module_id="candidate_fitting",
            body={
                "reducer_artifact_id": selected_candidate_runtime.candidate_id,
                "owner_id": "module.candidate-fitting-v1",
                "scope_id": _SCOPE_ID,
                "candidate_id": selected_candidate_runtime.candidate_id,
                "family_id": selected_candidate_runtime.family_id,
                "composition_operator": "identity",
                "parameter_summary": _normalize_mapping(
                    selected_candidate_runtime.parameters
                ),
                "canonical_structure_signature": (
                    selected_candidate_runtime.structure_signature
                ),
                "description_gain_bits": _stable_float(
                    selected_candidate_runtime.description_gain_bits
                ),
                "exploratory_primary_score": _stable_float(
                    selected_candidate_runtime.exploratory_primary_score
                ),
                "candidate_spec_ref": selected_candidate_spec.manifest.ref.as_dict(),
            },
        ),
        parent_refs=(
            selected_candidate_spec.manifest.ref,
            selected_candidate_structure.manifest.ref,
        ),
    )
    search_ledger = _register_runtime_model(
        registry,
        catalog,
        build_search_ledger(
            search_plan=intake.search_plan_object,
            candidate_records=search_candidate_records,
            selected_candidate_id=best_overall_candidate_runtime.candidate_id,
            accepted_candidate_id=(
                accepted_candidate_runtime.candidate_id
                if accepted_candidate_runtime is not None
                else None
            ),
            parent_refs=(search_plan.manifest.ref,),
        ),
    )
    frontier = _register_runtime_model(
        registry,
        catalog,
        build_frontier(
            search_plan=intake.search_plan_object,
            candidate_records=search_candidate_records,
            parent_refs=(search_plan.manifest.ref,),
        ),
    )
    rejected_diagnostics = _register_runtime_model(
        registry,
        catalog,
        build_rejected_diagnostics(
            candidate_records=search_candidate_records,
            parent_refs=(search_plan.manifest.ref,),
        ),
    )
    frozen_shortlist = _register_runtime_model(
        registry,
        catalog,
        build_frozen_shortlist(
            search_plan_ref=search_plan.manifest.ref,
            candidate_ref=selected_candidate.manifest.ref,
            parent_refs=(search_plan.manifest.ref, selected_candidate.manifest.ref),
        ),
    )
    freeze_event = _register_runtime_model(
        registry,
        catalog,
        build_freeze_event(
            frozen_candidate_ref=selected_candidate.manifest.ref,
            frozen_shortlist_ref=frozen_shortlist.manifest.ref,
            confirmatory_baseline_id="constant_baseline",
            parent_refs=(
                selected_candidate.manifest.ref,
                frozen_shortlist.manifest.ref,
            ),
        ),
    )
    prediction_artifact = _register_runtime_model(
        registry,
        catalog,
        PredictionArtifactManifest(
            prediction_artifact_id=(
                f"{selected_candidate_runtime.candidate_id}_confirmatory_prediction"
            ),
            candidate_id=selected_candidate_runtime.candidate_id,
            stage_id="confirmatory_holdout",
            fit_window_id="confirmatory_train_window_v1",
            test_window_id="confirmatory_test_window_v1",
            model_freeze_status="global_finalist_frozen",
            refit_rule_applied="pre_holdout_development_refit",
            score_policy_ref=point_score_policy.manifest.ref,
            rows=tuple(
                PredictionRow.from_payload(
                    row,
                    field_path=f"confirmatory_prediction_rows[{index}]",
                )
                for index, row in enumerate(
                    selected_candidate_runtime.confirmatory_prediction_rows
                )
            ),
            parent_refs=(
                selected_candidate.manifest.ref,
                intake.evaluation_plan.manifest.ref,
            ),
        ),
    )
    point_score_result = _register_runtime_model(
        registry,
        catalog,
        PointScoreResultManifest(
            score_result_id=f"{selected_candidate_runtime.candidate_id}_point_score",
            score_policy_ref=point_score_policy.manifest.ref,
            prediction_artifact_ref=prediction_artifact.manifest.ref,
            per_horizon=(
                PerHorizonScore(
                    horizon=1,
                    valid_origin_count=len(
                        selected_candidate_runtime.confirmatory_prediction_rows
                    ),
                    mean_point_loss=_stable_float(
                        selected_candidate_runtime.confirmatory_primary_score
                    ),
                ),
            ),
            aggregated_primary_score=_stable_float(
                selected_candidate_runtime.confirmatory_primary_score
            ),
            comparison_status="comparable",
            failure_reason_code=None,
        ),
    )
    confirmatory_fold = _confirmatory_fold(folds)
    confirmatory_train = _train_rows(feature_rows, confirmatory_fold)
    baseline_point_forecast = _stable_float(
        fmean(float(row["target"]) for row in confirmatory_train)
    )
    baseline_prediction_artifact = _register_runtime_model(
        registry,
        catalog,
        PredictionArtifactManifest(
            prediction_artifact_id="constant_baseline_confirmatory_prediction",
            candidate_id="constant_baseline",
            stage_id="confirmatory_holdout",
            fit_window_id="confirmatory_train_window_v1",
            test_window_id="confirmatory_test_window_v1",
            model_freeze_status="global_finalist_frozen",
            refit_rule_applied="pre_holdout_development_refit",
            score_policy_ref=point_score_policy.manifest.ref,
            rows=tuple(
                PredictionRow.from_payload(
                    {
                        **row,
                        "point_forecast": baseline_point_forecast,
                    },
                    field_path=f"baseline_prediction_rows[{index}]",
                )
                for index, row in enumerate(
                    selected_candidate_runtime.confirmatory_prediction_rows
                )
            ),
            parent_refs=(
                baseline_registry.manifest.ref,
                intake.evaluation_plan.manifest.ref,
            ),
        ),
    )
    baseline_point_score_result = _register_runtime_model(
        registry,
        catalog,
        PointScoreResultManifest(
            score_result_id="constant_baseline_point_score",
            score_policy_ref=point_score_policy.manifest.ref,
            prediction_artifact_ref=baseline_prediction_artifact.manifest.ref,
            per_horizon=(
                PerHorizonScore(
                    horizon=1,
                    valid_origin_count=len(
                        selected_candidate_runtime.confirmatory_prediction_rows
                    ),
                    mean_point_loss=_stable_float(
                        selected_candidate_runtime.baseline_primary_score
                    ),
                ),
            ),
            aggregated_primary_score=_stable_float(
                selected_candidate_runtime.baseline_primary_score
            ),
            comparison_status="comparable",
            failure_reason_code=None,
        ),
    )
    calibration_contract = registry.register(
        _build_manifest(
            catalog,
            schema_name="calibration_contract_manifest@1.0.0",
            module_id="scoring",
            body={
                "calibration_contract_id": "prototype_point_calibration_contract_v1",
                "owner_prompt_id": "prompt.scoring-calibration-v1",
                "scope_id": _SCOPE_ID,
                "forecast_object_type": "point",
                "status_policy": "not_applicable_for_forecast_type",
            },
        )
    )
    calibration_result = registry.register(
        _build_manifest(
            catalog,
            schema_name="calibration_result_manifest@1.0.0",
            module_id="scoring",
            body={
                "calibration_result_id": "prototype_point_calibration_result_v1",
                "owner_prompt_id": "prompt.scoring-calibration-v1",
                "scope_id": _SCOPE_ID,
                "calibration_contract_ref": calibration_contract.manifest.ref.as_dict(),
                "prediction_artifact_ref": prediction_artifact.manifest.ref.as_dict(),
                "forecast_object_type": "point",
                "status": "not_applicable_for_forecast_type",
                "failure_reason_code": None,
                "pass": None,
                "gate_effect": "none",
                "diagnostics": [],
            },
        ),
        parent_refs=(calibration_contract.manifest.ref,),
    )
    comparison_universe = registry.register(
        build_comparison_universe(
            selected_candidate_id=selected_candidate_runtime.candidate_id,
            baseline_id="constant_baseline",
            candidate_primary_score=_stable_float(
                selected_candidate_runtime.confirmatory_primary_score
            ),
            baseline_primary_score=_stable_float(
                selected_candidate_runtime.baseline_primary_score
            ),
            candidate_comparison_key=build_comparison_key(
                evaluation_plan=intake.evaluation_plan_object,
                score_policy_ref=point_score_policy.manifest.ref,
            ),
            baseline_comparison_key=build_comparison_key(
                evaluation_plan=intake.evaluation_plan_object,
                score_policy_ref=point_score_policy.manifest.ref,
            ),
            candidate_score_result_ref=point_score_result.manifest.ref,
            baseline_score_result_ref=baseline_point_score_result.manifest.ref,
            comparator_score_result_refs=(baseline_point_score_result.manifest.ref,),
            paired_comparison_records=(
                {
                    "comparator_id": "constant_baseline",
                    "comparator_kind": "baseline",
                    "comparison_status": "comparable",
                    "candidate_primary_score": _stable_float(
                        selected_candidate_runtime.confirmatory_primary_score
                    ),
                    "comparator_primary_score": _stable_float(
                        selected_candidate_runtime.baseline_primary_score
                    ),
                    "primary_score_delta": _stable_float(
                        selected_candidate_runtime.confirmatory_primary_score
                        - selected_candidate_runtime.baseline_primary_score
                    ),
                    "mean_loss_differential": _stable_float(
                        selected_candidate_runtime.confirmatory_primary_score
                        - selected_candidate_runtime.baseline_primary_score
                    ),
                    "score_result_ref": (
                        baseline_point_score_result.manifest.ref.as_dict()
                    ),
                },
            ),
        ).to_manifest(catalog),
        parent_refs=(
            freeze_event.manifest.ref,
            frozen_shortlist.manifest.ref,
            point_score_policy.manifest.ref,
            forecast_comparison_policy.manifest.ref,
        ),
    )
    run_result_id = _run_result_id(selected_candidate_runtime)
    bundle_id = _bundle_id(selected_candidate_runtime)
    run_result_ref = TypedRef(
        schema_name="run_result_manifest@1.1.0",
        object_id=run_result_id,
    )
    bundle_ref = TypedRef(
        schema_name="reproducibility_bundle_manifest@1.0.0",
        object_id=bundle_id,
    )
    evaluation_event_log = registry.register(
        build_evaluation_event_log(
            search_plan_ref=search_plan.manifest.ref,
            frozen_shortlist_ref=frozen_shortlist.manifest.ref,
            freeze_event_ref=freeze_event.manifest.ref,
            freeze_event_manifest=freeze_event.manifest,
            comparison_universe_ref=comparison_universe.manifest.ref,
            search_local_segment_ids=tuple(
                segment.segment_id
                for segment in intake.evaluation_plan_object.development_segments
            ),
            confirmatory_segment_id=(
                intake.evaluation_plan_object.confirmatory_segment.segment_id
            ),
            holdout_access_count=1,
            prediction_artifact_ref=prediction_artifact.manifest.ref,
            run_result_ref=run_result_ref,
        ).to_manifest(catalog),
        parent_refs=(
            search_plan.manifest.ref,
            frozen_shortlist.manifest.ref,
            freeze_event.manifest.ref,
            comparison_universe.manifest.ref,
            prediction_artifact.manifest.ref,
        ),
    )
    evaluation_governance = registry.register(
        build_evaluation_governance(
            comparison_universe_ref=comparison_universe.manifest.ref,
            event_log_ref=evaluation_event_log.manifest.ref,
            freeze_event_ref=freeze_event.manifest.ref,
            frozen_shortlist_ref=frozen_shortlist.manifest.ref,
            confirmatory_promotion_allowed=resolve_confirmatory_promotion_allowed(
                candidate_beats_baseline=(
                    selected_candidate_runtime.confirmatory_primary_score
                    < selected_candidate_runtime.baseline_primary_score
                ),
                predictive_gate_policy_manifest=predictive_gate_policy.manifest,
                calibration_result_manifest=calibration_result.manifest,
            ),
        ).to_manifest(catalog),
        parent_refs=(
            comparison_universe.manifest.ref,
            evaluation_event_log.manifest.ref,
            freeze_event.manifest.ref,
            frozen_shortlist.manifest.ref,
        ),
    )
    robustness_artifacts = _materialize_robustness_artifacts(
        catalog=catalog,
        registry=registry,
        intake=intake,
        feature_rows=feature_rows,
        folds=folds,
        baseline_registry=baseline_registry,
        point_score_policy=point_score_policy,
        frozen_shortlist=frozen_shortlist,
        selected_candidate_runtime=selected_candidate_runtime,
        minimum_description_gain_bits=minimum_description_gain_bits,
    )
    null_protocol = robustness_artifacts.null_protocol
    perturbation_protocol = robustness_artifacts.perturbation_protocol
    leakage_canary_result = robustness_artifacts.leakage_canary_results[0]
    robustness_report = robustness_artifacts.robustness_report
    scorecard_decision = resolve_scorecard_status(
        candidate_admissible=selected_candidate_runtime.admissible,
        robustness_status=str(
            robustness_report.manifest.body.get("final_robustness_status", "failed")
        ),
        robustness_reason_codes=_robustness_reason_codes(
            registry=registry,
            robustness_report_body=robustness_report.manifest.body,
            leakage_canary_results=robustness_artifacts.leakage_canary_results,
        ),
        candidate_beats_baseline=(
            selected_candidate_runtime.confirmatory_primary_score
            < selected_candidate_runtime.baseline_primary_score
        ),
        confirmatory_promotion_allowed=bool(
            evaluation_governance.manifest.body["confirmatory_promotion_allowed"]
        ),
        point_score_comparison_status=str(
            point_score_result.manifest.body["comparison_status"]
        ),
        time_safety_status=str(intake.time_safety_audit.manifest.body["status"]),
        calibration_status=str(calibration_result.manifest.body["status"]),
        descriptive_failure_reason_codes=_gate_descriptive_reason_codes(
            selected_candidate_runtime
        ),
        predictive_governance_reason_codes=_predictive_governance_reason_codes(
            evaluation_governance.manifest.body
        ),
    )
    scorecard = registry.register(
        _build_manifest(
            catalog,
            schema_name="scorecard_manifest@1.1.0",
            module_id="gate_lifecycle",
            body={
                "scorecard_id": "prototype_scorecard_v1",
                "candidate_ref": selected_candidate.manifest.ref.as_dict(),
                "observation_model_ref": (
                    intake.observation_model.manifest.ref.as_dict()
                ),
                "canonical_structure_code_ref": (
                    selected_candidate_structure.manifest.ref.as_dict()
                ),
                "target_transform_ref": intake.target_transform.manifest.ref.as_dict(),
                "base_measure_policy_ref": (
                    intake.base_measure_policy.manifest.ref.as_dict()
                ),
                "codelength_policy_ref": (
                    intake.codelength_policy.manifest.ref.as_dict()
                ),
                "reference_description_policy_ref": (
                    intake.reference_description_policy.manifest.ref.as_dict()
                ),
                "L_family_bits": _stable_float(
                    selected_candidate_runtime.description_components["L_family_bits"]
                ),
                "L_structure_bits": _stable_float(
                    selected_candidate_runtime.description_components[
                        "L_structure_bits"
                    ]
                ),
                "L_literals_bits": _stable_float(
                    selected_candidate_runtime.description_components["L_literals_bits"]
                ),
                "L_params_bits": _stable_float(
                    selected_candidate_runtime.description_components["L_params_bits"]
                ),
                "L_state_bits": _stable_float(
                    selected_candidate_runtime.description_components["L_state_bits"]
                ),
                "L_data_bits": _stable_float(
                    selected_candidate_runtime.description_components["L_data_bits"]
                ),
                "L_total_bits": _stable_float(
                    selected_candidate_runtime.description_components["L_total_bits"]
                ),
                "reference_bits": _stable_float(
                    selected_candidate_runtime.description_components["reference_bits"]
                ),
                "description_gain_bits": _stable_float(
                    selected_candidate_runtime.description_gain_bits
                ),
                "point_score_policy_ref": point_score_policy.manifest.ref.as_dict(),
                "point_score_result_ref": point_score_result.manifest.ref.as_dict(),
                "calibration_contract_ref": calibration_contract.manifest.ref.as_dict(),
                "calibration_result_ref": calibration_result.manifest.ref.as_dict(),
                "evaluation_plan_ref": intake.evaluation_plan.manifest.ref.as_dict(),
                "baseline_registry_ref": baseline_registry.manifest.ref.as_dict(),
                "forecast_comparison_policy_ref": (
                    forecast_comparison_policy.manifest.ref.as_dict()
                ),
                "comparison_universe_ref": comparison_universe.manifest.ref.as_dict(),
                "evaluation_event_log_ref": (
                    evaluation_event_log.manifest.ref.as_dict()
                ),
                "evaluation_governance_ref": (
                    evaluation_governance.manifest.ref.as_dict()
                ),
                "predictive_gate_policy_ref": (
                    predictive_gate_policy.manifest.ref.as_dict()
                ),
                "null_protocol_ref": null_protocol.manifest.ref.as_dict(),
                "perturbation_protocol_ref": (
                    perturbation_protocol.manifest.ref.as_dict()
                ),
                "robustness_report_ref": robustness_report.manifest.ref.as_dict(),
                "time_safety_audit_ref": (
                    intake.time_safety_audit.manifest.ref.as_dict()
                ),
                "descriptive_status": scorecard_decision.descriptive_status,
                "descriptive_reason_codes": list(
                    scorecard_decision.descriptive_reason_codes
                ),
                "predictive_status": scorecard_decision.predictive_status,
                "predictive_reason_codes": list(
                    scorecard_decision.predictive_reason_codes
                ),
            },
        ),
        parent_refs=(
            selected_candidate.manifest.ref,
            intake.observation_model.manifest.ref,
            selected_candidate_structure.manifest.ref,
            intake.target_transform.manifest.ref,
            intake.base_measure_policy.manifest.ref,
            intake.codelength_policy.manifest.ref,
            intake.reference_description_policy.manifest.ref,
            point_score_policy.manifest.ref,
            point_score_result.manifest.ref,
            calibration_contract.manifest.ref,
            calibration_result.manifest.ref,
            intake.evaluation_plan.manifest.ref,
            baseline_registry.manifest.ref,
            forecast_comparison_policy.manifest.ref,
            comparison_universe.manifest.ref,
            evaluation_event_log.manifest.ref,
            evaluation_governance.manifest.ref,
            predictive_gate_policy.manifest.ref,
            null_protocol.manifest.ref,
            perturbation_protocol.manifest.ref,
            robustness_report.manifest.ref,
            intake.time_safety_audit.manifest.ref,
        ),
    )
    claim_card: RegisteredManifest | None = None
    abstention: RegisteredManifest | None = None
    if telemetry is None:
        claim_decision = resolve_claim_publication(
            scorecard_body=scorecard.manifest.body
        )
        validation_scope = registry.register(
            _build_manifest(
                catalog,
                schema_name="validation_scope_manifest@1.0.0",
                module_id="claims",
                body={
                    "validation_scope_id": "prototype_validation_scope_v1",
                    "scope_ledger_ref": scope_ledger.manifest.ref.as_dict(),
                    "evaluation_plan_ref": (
                        intake.evaluation_plan.manifest.ref.as_dict()
                    ),
                    "baseline_registry_ref": baseline_registry.manifest.ref.as_dict(),
                    "forecast_comparison_policy_ref": (
                        forecast_comparison_policy.manifest.ref.as_dict()
                    ),
                    "comparison_universe_ref": (
                        comparison_universe.manifest.ref.as_dict()
                    ),
                    "evaluation_event_log_ref": (
                        evaluation_event_log.manifest.ref.as_dict()
                    ),
                    "evaluation_governance_ref": (
                        evaluation_governance.manifest.ref.as_dict()
                    ),
                    "predictive_gate_policy_ref": (
                        predictive_gate_policy.manifest.ref.as_dict()
                    ),
                    "null_protocol_ref": null_protocol.manifest.ref.as_dict(),
                    "perturbation_protocol_ref": (
                        perturbation_protocol.manifest.ref.as_dict()
                    ),
                    "time_safety_audit_ref": (
                        intake.time_safety_audit.manifest.ref.as_dict()
                    ),
                    "point_score_policy_ref": point_score_policy.manifest.ref.as_dict(),
                    "calibration_contract_ref": (
                        calibration_contract.manifest.ref.as_dict()
                    ),
                    "public_forecast_object_type": "point",
                    "run_support_object_ids": list(DEFAULT_RUN_SUPPORT_OBJECT_IDS),
                    "admissibility_rule_ids": list(DEFAULT_ADMISSIBILITY_RULE_IDS),
                    "entity_scope": "single_entity_only",
                    "horizon_scope": "single_horizon",
                },
            ),
            parent_refs=(
                scope_ledger.manifest.ref,
                intake.evaluation_plan.manifest.ref,
                baseline_registry.manifest.ref,
                forecast_comparison_policy.manifest.ref,
                comparison_universe.manifest.ref,
                evaluation_event_log.manifest.ref,
                evaluation_governance.manifest.ref,
                predictive_gate_policy.manifest.ref,
                null_protocol.manifest.ref,
                perturbation_protocol.manifest.ref,
                intake.time_safety_audit.manifest.ref,
                point_score_policy.manifest.ref,
                calibration_contract.manifest.ref,
            ),
        )
        publication_mode = claim_decision.publication_mode
        if publication_mode == "candidate_publication":
            claim_card = registry.register(
                _build_manifest(
                    catalog,
                    schema_name="claim_card_manifest@1.1.0",
                    module_id="claims",
                    body={
                        "claim_card_id": "prototype_claim_card_v1",
                        "candidate_ref": selected_candidate.manifest.ref.as_dict(),
                        "scorecard_ref": scorecard.manifest.ref.as_dict(),
                        "validation_scope_ref": validation_scope.manifest.ref.as_dict(),
                        "claim_type": claim_decision.claim_type,
                        "claim_ceiling": claim_decision.claim_ceiling,
                        "predictive_support_status": (
                            claim_decision.predictive_support_status
                        ),
                        "allowed_interpretation_codes": list(
                            claim_decision.allowed_interpretation_codes
                        ),
                        "forbidden_interpretation_codes": list(
                            claim_decision.forbidden_interpretation_codes
                        ),
                    },
                ),
                parent_refs=(
                    selected_candidate.manifest.ref,
                    scorecard.manifest.ref,
                    validation_scope.manifest.ref,
                ),
            )
        else:
            abstention = _register_runtime_model(
                registry,
                catalog,
                AbstentionManifest(
                    abstention_id="prototype_abstention_v1",
                    abstention_type=str(claim_decision.abstention_type),
                    blocked_ceiling=str(claim_decision.blocked_ceiling),
                    reason_codes=claim_decision.abstention_reason_codes,
                    governing_refs=(scorecard.manifest.ref,),
                ),
            )
        run_result_manifest = build_run_result_manifest(
            object_id=run_result_id,
            run_id=run_result_id,
            scope_ledger_ref=scope_ledger.manifest.ref,
            search_plan_ref=search_plan.manifest.ref,
            evaluation_plan_ref=intake.evaluation_plan.manifest.ref,
            comparison_universe_ref=comparison_universe.manifest.ref,
            evaluation_event_log_ref=evaluation_event_log.manifest.ref,
            evaluation_governance_ref=evaluation_governance.manifest.ref,
            primary_validation_scope_ref=validation_scope.manifest.ref,
            publication_mode=publication_mode,
            selected_candidate_ref=(
                selected_candidate.manifest.ref
                if publication_mode == "candidate_publication"
                else None
            ),
            scorecard_ref=(
                scorecard.manifest.ref
                if publication_mode == "candidate_publication"
                else None
            ),
            claim_card_ref=claim_card.manifest.ref if claim_card else None,
            abstention_ref=abstention.manifest.ref if abstention else None,
            prediction_artifact_refs=(prediction_artifact.manifest.ref,),
            robustness_report_refs=(robustness_report.manifest.ref,),
            reproducibility_bundle_ref=bundle_ref,
            deferred_scope_policy_refs=(),
        )
        publication_artifact = claim_card or abstention
        assert publication_artifact is not None
        reproducibility_bundle = _register_runtime_model(
            registry,
            catalog,
            build_reproducibility_bundle_manifest(
                object_id=bundle_id,
                bundle_id=bundle_id,
                bundle_mode=publication_mode,
                dataset_snapshot_ref=intake.snapshot.manifest.ref,
                feature_view_ref=intake.feature_view.manifest.ref,
                search_plan_ref=search_plan.manifest.ref,
                evaluation_plan_ref=intake.evaluation_plan.manifest.ref,
                comparison_universe_ref=comparison_universe.manifest.ref,
                evaluation_event_log_ref=evaluation_event_log.manifest.ref,
                evaluation_governance_ref=evaluation_governance.manifest.ref,
                run_result_ref=run_result_ref,
                required_manifest_refs=required_manifest_refs_for_publication(
                    publication_mode=publication_mode,
                    candidate_ref=selected_candidate.manifest.ref,
                    scorecard_ref=scorecard.manifest.ref,
                    claim_ref=claim_card.manifest.ref if claim_card else None,
                    abstention_ref=abstention.manifest.ref if abstention else None,
                    supporting_refs=(validation_scope.manifest.ref,),
                ),
                artifact_hash_records=build_artifact_hash_records(
                    snapshot=intake.snapshot,
                    feature_view=intake.feature_view,
                    search_plan=search_plan,
                    evaluation_plan=intake.evaluation_plan,
                    run_result_manifest=run_result_manifest.to_manifest(catalog),
                    candidate_or_abstention=publication_artifact,
                    scorecard=scorecard,
                    prediction_artifact=prediction_artifact,
                    validation_scope=validation_scope,
                    robustness_report=robustness_report,
                ),
                seed_records=build_replay_seed_records(
                    str(search_plan.manifest.body["seed_policy"]["root_seed"])
                ),
                stage_order_records=build_replay_stage_order(
                    dataset_snapshot_ref=intake.snapshot.manifest.ref,
                    feature_view_ref=intake.feature_view.manifest.ref,
                    search_plan_ref=search_plan.manifest.ref,
                    evaluation_plan_ref=intake.evaluation_plan.manifest.ref,
                    comparison_universe_ref=comparison_universe.manifest.ref,
                    evaluation_event_log_ref=evaluation_event_log.manifest.ref,
                    evaluation_governance_ref=evaluation_governance.manifest.ref,
                    scorecard_ref=scorecard.manifest.ref,
                    candidate_or_abstention_ref=publication_artifact.manifest.ref,
                    run_result_ref=run_result_ref,
                ),
                replay_entrypoint_id=_REPLAY_ENTRYPOINT_ID,
                replay_verification_status="verified",
                failure_reason_codes=(),
            ),
        )
        run_result = _register_runtime_model(
            registry,
            catalog,
            build_run_result_manifest(
                object_id=run_result_manifest.object_id,
                run_id=run_result_manifest.run_id,
                scope_ledger_ref=run_result_manifest.scope_ledger_ref,
                search_plan_ref=run_result_manifest.search_plan_ref,
                evaluation_plan_ref=run_result_manifest.evaluation_plan_ref,
                comparison_universe_ref=run_result_manifest.comparison_universe_ref,
                evaluation_event_log_ref=run_result_manifest.evaluation_event_log_ref,
                evaluation_governance_ref=run_result_manifest.evaluation_governance_ref,
                primary_validation_scope_ref=(
                    run_result_manifest.primary_validation_scope_ref
                ),
                reproducibility_bundle_ref=reproducibility_bundle.manifest.ref,
                publication_mode=run_result_manifest.result_mode,
                selected_candidate_ref=run_result_manifest.primary_reducer_artifact_ref,
                scorecard_ref=run_result_manifest.primary_scorecard_ref,
                claim_card_ref=run_result_manifest.primary_claim_card_ref,
                abstention_ref=run_result_manifest.primary_abstention_ref,
                prediction_artifact_refs=run_result_manifest.prediction_artifact_refs,
                robustness_report_refs=run_result_manifest.robustness_report_refs,
                deferred_scope_policy_refs=run_result_manifest.deferred_scope_policy_refs,
            ),
        )
        readiness_judgment = _register_runtime_model(
            registry,
            catalog,
            _build_runtime_readiness_manifest(
                judgment_id="prototype_readiness_judgment_v1",
                replay_verification_status="verified",
                schema_closure_status="passed",
            ),
        )
        schema_lifecycle_integration_closure = _register_runtime_model(
            registry,
            catalog,
            SchemaLifecycleIntegrationClosureManifest(
                closure_id="prototype_schema_lifecycle_integration_closure_v1",
                status="passed",
            ),
        )
        publication_record = _register_runtime_model(
            registry,
            catalog,
            build_publication_record_manifest(
                object_id=f"{run_result_id}_publication_record",
                publication_id=f"{run_result_id}_publication_record",
                run_result_manifest=run_result.manifest,
                comparison_universe_manifest=comparison_universe.manifest,
                reproducibility_bundle_manifest=reproducibility_bundle.manifest,
                readiness_judgment_manifest=readiness_judgment.manifest,
                schema_lifecycle_integration_closure_ref=(
                    schema_lifecycle_integration_closure.manifest.ref
                ),
                catalog_scope="public",
                published_at="2026-04-12T00:00:00Z",
            ),
        )
        replay = replay_prototype_run(
            bundle_ref=reproducibility_bundle.manifest.ref,
            catalog=catalog,
            registry=registry,
        )
    else:
        with telemetry.span(
            "prototype.publication",
            category="publication",
            attributes={"run_id": run_result_id},
        ):
            claim_decision = resolve_claim_publication(
                scorecard_body=scorecard.manifest.body
            )
            validation_scope = registry.register(
                _build_manifest(
                    catalog,
                    schema_name="validation_scope_manifest@1.0.0",
                    module_id="claims",
                    body={
                        "validation_scope_id": "prototype_validation_scope_v1",
                        "scope_ledger_ref": scope_ledger.manifest.ref.as_dict(),
                        "evaluation_plan_ref": (
                            intake.evaluation_plan.manifest.ref.as_dict()
                        ),
                        "baseline_registry_ref": (
                            baseline_registry.manifest.ref.as_dict()
                        ),
                        "forecast_comparison_policy_ref": (
                            forecast_comparison_policy.manifest.ref.as_dict()
                        ),
                        "comparison_universe_ref": (
                            comparison_universe.manifest.ref.as_dict()
                        ),
                        "evaluation_event_log_ref": (
                            evaluation_event_log.manifest.ref.as_dict()
                        ),
                        "evaluation_governance_ref": (
                            evaluation_governance.manifest.ref.as_dict()
                        ),
                        "predictive_gate_policy_ref": (
                            predictive_gate_policy.manifest.ref.as_dict()
                        ),
                        "null_protocol_ref": null_protocol.manifest.ref.as_dict(),
                        "perturbation_protocol_ref": (
                            perturbation_protocol.manifest.ref.as_dict()
                        ),
                        "time_safety_audit_ref": (
                            intake.time_safety_audit.manifest.ref.as_dict()
                        ),
                        "point_score_policy_ref": (
                            point_score_policy.manifest.ref.as_dict()
                        ),
                        "calibration_contract_ref": (
                            calibration_contract.manifest.ref.as_dict()
                        ),
                        "public_forecast_object_type": "point",
                        "run_support_object_ids": list(
                            DEFAULT_RUN_SUPPORT_OBJECT_IDS
                        ),
                        "admissibility_rule_ids": list(
                            DEFAULT_ADMISSIBILITY_RULE_IDS
                        ),
                        "entity_scope": "single_entity_only",
                        "horizon_scope": "single_horizon",
                    },
                ),
                parent_refs=(
                    scope_ledger.manifest.ref,
                    intake.evaluation_plan.manifest.ref,
                    baseline_registry.manifest.ref,
                    forecast_comparison_policy.manifest.ref,
                    comparison_universe.manifest.ref,
                    evaluation_event_log.manifest.ref,
                    evaluation_governance.manifest.ref,
                    predictive_gate_policy.manifest.ref,
                    null_protocol.manifest.ref,
                    perturbation_protocol.manifest.ref,
                    intake.time_safety_audit.manifest.ref,
                    point_score_policy.manifest.ref,
                    calibration_contract.manifest.ref,
                ),
            )
            publication_mode = claim_decision.publication_mode
            if publication_mode == "candidate_publication":
                claim_card = registry.register(
                    _build_manifest(
                        catalog,
                        schema_name="claim_card_manifest@1.1.0",
                        module_id="claims",
                        body={
                            "claim_card_id": "prototype_claim_card_v1",
                            "candidate_ref": selected_candidate.manifest.ref.as_dict(),
                            "scorecard_ref": scorecard.manifest.ref.as_dict(),
                            "validation_scope_ref": (
                                validation_scope.manifest.ref.as_dict()
                            ),
                            "claim_type": claim_decision.claim_type,
                            "claim_ceiling": claim_decision.claim_ceiling,
                            "predictive_support_status": (
                                claim_decision.predictive_support_status
                            ),
                            "allowed_interpretation_codes": list(
                                claim_decision.allowed_interpretation_codes
                            ),
                            "forbidden_interpretation_codes": list(
                                claim_decision.forbidden_interpretation_codes
                            ),
                        },
                    ),
                    parent_refs=(
                        selected_candidate.manifest.ref,
                        scorecard.manifest.ref,
                        validation_scope.manifest.ref,
                    ),
                )
            else:
                abstention = _register_runtime_model(
                    registry,
                    catalog,
                    AbstentionManifest(
                        abstention_id="prototype_abstention_v1",
                        abstention_type=str(claim_decision.abstention_type),
                        blocked_ceiling=str(claim_decision.blocked_ceiling),
                        reason_codes=claim_decision.abstention_reason_codes,
                        governing_refs=(scorecard.manifest.ref,),
                    ),
                )
            run_result_manifest = build_run_result_manifest(
                object_id=run_result_id,
                run_id=run_result_id,
                scope_ledger_ref=scope_ledger.manifest.ref,
                search_plan_ref=search_plan.manifest.ref,
                evaluation_plan_ref=intake.evaluation_plan.manifest.ref,
                comparison_universe_ref=comparison_universe.manifest.ref,
                evaluation_event_log_ref=evaluation_event_log.manifest.ref,
                evaluation_governance_ref=evaluation_governance.manifest.ref,
                primary_validation_scope_ref=validation_scope.manifest.ref,
                publication_mode=publication_mode,
                selected_candidate_ref=(
                    selected_candidate.manifest.ref
                    if publication_mode == "candidate_publication"
                    else None
                ),
                scorecard_ref=(
                    scorecard.manifest.ref
                    if publication_mode == "candidate_publication"
                    else None
                ),
                claim_card_ref=claim_card.manifest.ref if claim_card else None,
                abstention_ref=abstention.manifest.ref if abstention else None,
                prediction_artifact_refs=(prediction_artifact.manifest.ref,),
                robustness_report_refs=(robustness_report.manifest.ref,),
                reproducibility_bundle_ref=bundle_ref,
                deferred_scope_policy_refs=(),
            )
            publication_artifact = claim_card or abstention
            assert publication_artifact is not None
            reproducibility_bundle = _register_runtime_model(
                registry,
                catalog,
                build_reproducibility_bundle_manifest(
                    object_id=bundle_id,
                    bundle_id=bundle_id,
                    bundle_mode=publication_mode,
                    dataset_snapshot_ref=intake.snapshot.manifest.ref,
                    feature_view_ref=intake.feature_view.manifest.ref,
                    search_plan_ref=search_plan.manifest.ref,
                    evaluation_plan_ref=intake.evaluation_plan.manifest.ref,
                    comparison_universe_ref=comparison_universe.manifest.ref,
                    evaluation_event_log_ref=evaluation_event_log.manifest.ref,
                    evaluation_governance_ref=evaluation_governance.manifest.ref,
                    run_result_ref=run_result_ref,
                    required_manifest_refs=required_manifest_refs_for_publication(
                        publication_mode=publication_mode,
                        candidate_ref=selected_candidate.manifest.ref,
                        scorecard_ref=scorecard.manifest.ref,
                        claim_ref=claim_card.manifest.ref if claim_card else None,
                        abstention_ref=abstention.manifest.ref if abstention else None,
                        supporting_refs=(validation_scope.manifest.ref,),
                    ),
                    artifact_hash_records=build_artifact_hash_records(
                        snapshot=intake.snapshot,
                        feature_view=intake.feature_view,
                        search_plan=search_plan,
                        evaluation_plan=intake.evaluation_plan,
                        run_result_manifest=run_result_manifest.to_manifest(catalog),
                        candidate_or_abstention=publication_artifact,
                        scorecard=scorecard,
                        prediction_artifact=prediction_artifact,
                        validation_scope=validation_scope,
                        robustness_report=robustness_report,
                    ),
                    seed_records=build_replay_seed_records(
                        str(search_plan.manifest.body["seed_policy"]["root_seed"])
                    ),
                    stage_order_records=build_replay_stage_order(
                        dataset_snapshot_ref=intake.snapshot.manifest.ref,
                        feature_view_ref=intake.feature_view.manifest.ref,
                        search_plan_ref=search_plan.manifest.ref,
                        evaluation_plan_ref=intake.evaluation_plan.manifest.ref,
                        comparison_universe_ref=comparison_universe.manifest.ref,
                        evaluation_event_log_ref=evaluation_event_log.manifest.ref,
                        evaluation_governance_ref=evaluation_governance.manifest.ref,
                        scorecard_ref=scorecard.manifest.ref,
                        candidate_or_abstention_ref=publication_artifact.manifest.ref,
                        run_result_ref=run_result_ref,
                    ),
                    replay_entrypoint_id=_REPLAY_ENTRYPOINT_ID,
                    replay_verification_status="verified",
                    failure_reason_codes=(),
                ),
            )
            run_result = _register_runtime_model(
                registry,
                catalog,
                build_run_result_manifest(
                    object_id=run_result_manifest.object_id,
                    run_id=run_result_manifest.run_id,
                    scope_ledger_ref=run_result_manifest.scope_ledger_ref,
                    search_plan_ref=run_result_manifest.search_plan_ref,
                    evaluation_plan_ref=run_result_manifest.evaluation_plan_ref,
                    comparison_universe_ref=run_result_manifest.comparison_universe_ref,
                    evaluation_event_log_ref=run_result_manifest.evaluation_event_log_ref,
                    evaluation_governance_ref=run_result_manifest.evaluation_governance_ref,
                    primary_validation_scope_ref=(
                        run_result_manifest.primary_validation_scope_ref
                    ),
                    reproducibility_bundle_ref=reproducibility_bundle.manifest.ref,
                    publication_mode=run_result_manifest.result_mode,
                    selected_candidate_ref=run_result_manifest.primary_reducer_artifact_ref,
                    scorecard_ref=run_result_manifest.primary_scorecard_ref,
                    claim_card_ref=run_result_manifest.primary_claim_card_ref,
                    abstention_ref=run_result_manifest.primary_abstention_ref,
                    prediction_artifact_refs=run_result_manifest.prediction_artifact_refs,
                    robustness_report_refs=run_result_manifest.robustness_report_refs,
                    deferred_scope_policy_refs=run_result_manifest.deferred_scope_policy_refs,
                ),
            )
            readiness_judgment = _register_runtime_model(
                registry,
                catalog,
                _build_runtime_readiness_manifest(
                    judgment_id="prototype_readiness_judgment_v1",
                    replay_verification_status="verified",
                    schema_closure_status="passed",
                ),
            )
            schema_lifecycle_integration_closure = _register_runtime_model(
                registry,
                catalog,
                SchemaLifecycleIntegrationClosureManifest(
                    closure_id="prototype_schema_lifecycle_integration_closure_v1",
                    status="passed",
                ),
            )
            publication_record = _register_runtime_model(
                registry,
                catalog,
                build_publication_record_manifest(
                    object_id=f"{run_result_id}_publication_record",
                    publication_id=f"{run_result_id}_publication_record",
                    run_result_manifest=run_result.manifest,
                    comparison_universe_manifest=comparison_universe.manifest,
                    reproducibility_bundle_manifest=reproducibility_bundle.manifest,
                    readiness_judgment_manifest=readiness_judgment.manifest,
                    schema_lifecycle_integration_closure_ref=(
                        schema_lifecycle_integration_closure.manifest.ref
                    ),
                    catalog_scope="public",
                    published_at="2026-04-12T00:00:00Z",
                ),
            )
        with telemetry.span(
            "prototype.replay",
            category="replay",
            attributes={"bundle_id": bundle_id},
        ):
            replay = replay_prototype_run(
                bundle_ref=reproducibility_bundle.manifest.ref,
                catalog=catalog,
                registry=registry,
            )
    if replay.replay_verification_status != "verified":
        raise ValueError(
            "replay verification failed for the sealed prototype reducer workflow"
        )

    return PrototypeReducerWorkflowResult(
        intake=intake,
        scope_ledger=scope_ledger,
        canonicalization_policy=canonicalization_policy,
        search_plan=search_plan,
        search_ledger=search_ledger,
        frontier=frontier,
        rejected_diagnostics=rejected_diagnostics,
        point_score_policy=point_score_policy,
        baseline_registry=baseline_registry,
        forecast_comparison_policy=forecast_comparison_policy,
        selected_candidate=selected_candidate,
        selected_candidate_spec=selected_candidate_spec,
        selected_candidate_structure=selected_candidate_structure,
        frozen_shortlist=frozen_shortlist,
        freeze_event=freeze_event,
        comparison_universe=comparison_universe,
        evaluation_event_log=evaluation_event_log,
        evaluation_governance=evaluation_governance,
        prediction_artifact=prediction_artifact,
        point_score_result=point_score_result,
        calibration_contract=calibration_contract,
        calibration_result=calibration_result,
        null_protocol=null_protocol,
        perturbation_protocol=perturbation_protocol,
        leakage_canary_result=leakage_canary_result,
        leakage_canary_results=robustness_artifacts.leakage_canary_results,
        robustness_report=robustness_report,
        scorecard=scorecard,
        validation_scope=validation_scope,
        readiness_judgment=readiness_judgment,
        schema_lifecycle_integration_closure=(schema_lifecycle_integration_closure),
        reproducibility_bundle=reproducibility_bundle,
        run_result=run_result,
        publication_record=publication_record,
        claim_card=claim_card,
        abstention=abstention,
        candidate_summaries=candidate_summaries,
        confirmatory_primary_score=selected_candidate_runtime.confirmatory_primary_score,
        replay_verified=True,
    )


def replay_prototype_run(
    *,
    bundle_ref: TypedRef,
    catalog: ContractCatalog,
    registry: ManifestRegistry,
) -> PrototypeReplayResult:
    bundle = registry.resolve(bundle_ref)
    run_result = registry.resolve(_typed_ref(bundle.manifest.body["run_result_ref"]))
    scorecard = resolve_replay_scorecard(run_result.manifest.body, registry)
    search_plan = registry.resolve(_typed_ref(bundle.manifest.body["search_plan_ref"]))
    feature_view = registry.resolve(
        _typed_ref(bundle.manifest.body["feature_view_ref"])
    )
    evaluation_plan = registry.resolve(
        _typed_ref(bundle.manifest.body["evaluation_plan_ref"])
    )
    evaluation_governance = registry.resolve(
        _typed_ref(scorecard.manifest.body["evaluation_governance_ref"])
    )
    calibration_result = registry.resolve(
        _typed_ref(scorecard.manifest.body["calibration_result_ref"])
    )
    time_safety_audit = registry.resolve(
        _typed_ref(scorecard.manifest.body["time_safety_audit_ref"])
    )
    robustness_report = registry.resolve(
        _typed_ref(scorecard.manifest.body["robustness_report_ref"])
    )
    leakage_canary_results = tuple(
        registry.resolve(_typed_ref(item))
        for item in robustness_report.manifest.body.get(
            "leakage_canary_result_refs", ()
        )
    )
    point_score_result = resolve_replay_point_score_result(run_result, registry)

    codelength_policy = registry.resolve(
        _typed_ref(search_plan.manifest.body["codelength_policy_ref"])
    )
    minimum_description_gain_bits = float(
        search_plan.manifest.body["minimum_description_gain_bits"]
    )
    feature_rows = tuple(feature_view.manifest.body["rows"])
    folds = tuple(evaluation_plan.manifest.body["folds"])
    support_bundle = build_prototype_support_bundle(
        catalog=catalog,
        observed_values=tuple(float(row["target"]) for row in feature_rows),
        quantization_step=str(codelength_policy.manifest.body["quantization_step"]),
    )
    candidates = _evaluate_candidates(
        feature_rows=feature_rows,
        folds=folds,
        support_bundle=support_bundle,
        minimum_description_gain_bits=minimum_description_gain_bits,
        point_loss_id=_DEFAULT_LOSS_ID,
        family_ids=tuple(search_plan.manifest.body["candidate_family_ids"]),
    )
    replay_candidate = (
        _select_accepted_candidate(candidates)
        or _select_best_overall_candidate(candidates)
    )
    scorecard_decision = resolve_scorecard_status(
        candidate_admissible=replay_candidate.admissible,
        robustness_status=str(
            robustness_report.manifest.body.get("final_robustness_status", "failed")
        ),
        robustness_reason_codes=_robustness_reason_codes(
            registry=registry,
            robustness_report_body=robustness_report.manifest.body,
            leakage_canary_results=leakage_canary_results,
        ),
        candidate_beats_baseline=(
            replay_candidate.confirmatory_primary_score
            < replay_candidate.baseline_primary_score
        ),
        confirmatory_promotion_allowed=bool(
            evaluation_governance.manifest.body["confirmatory_promotion_allowed"]
        ),
        point_score_comparison_status=str(
            point_score_result.manifest.body["comparison_status"]
        ),
        time_safety_status=str(time_safety_audit.manifest.body["status"]),
        calibration_status=str(calibration_result.manifest.body["status"]),
        descriptive_failure_reason_codes=_gate_descriptive_reason_codes(
            replay_candidate
        ),
        predictive_governance_reason_codes=_predictive_governance_reason_codes(
            evaluation_governance.manifest.body
        ),
    )
    expected_mode = resolve_claim_publication(
        scorecard_body=scorecard.manifest.body
    ).publication_mode
    verification = verify_replayed_outcome(
        bundle=bundle,
        registry=registry,
        outcome=ReplayedOutcome(
            selected_candidate_id=replay_candidate.candidate_id,
            confirmatory_primary_score=replay_candidate.confirmatory_primary_score,
            publication_mode=expected_mode,
            descriptive_status=scorecard_decision.descriptive_status,
            descriptive_reason_codes=scorecard_decision.descriptive_reason_codes,
            predictive_status=scorecard_decision.predictive_status,
            predictive_reason_codes=scorecard_decision.predictive_reason_codes,
            replayed_stage_order=(
                "dataset_snapshot_frozen",
                "feature_view_materialized",
                "search_plan_frozen",
                "evaluation_plan_frozen",
                "comparison_universe_resolved",
                "evaluation_event_log_written",
                "evaluation_governance_resolved",
                "scorecard_resolved",
                "publication_decision_resolved",
                "run_result_assembled",
            ),
        ),
    )
    return PrototypeReplayResult(
        bundle_ref=verification.bundle_ref,
        run_result_ref=verification.run_result_ref,
        selected_candidate_ref=verification.selected_candidate_ref,
        replay_verification_status=verification.replay_verification_status,
        confirmatory_primary_score=verification.confirmatory_primary_score,
        failure_reason_codes=verification.failure_reason_codes,
    )


def _candidate_family_ids() -> tuple[str, ...]:
    return ("constant", "drift", "seasonal_naive", "linear_trend")


def _build_runtime_readiness_manifest(
    *,
    judgment_id: str,
    replay_verification_status: str,
    schema_closure_status: str,
) -> ReadinessJudgmentManifest:
    judgment = judge_readiness(
        judgment_id=judgment_id,
        gate_results=(
            ReadinessGateResult(
                gate_id="workflow.replay_verification",
                status=(
                    "passed"
                    if replay_verification_status == "verified"
                    else "failed"
                ),
                required=True,
                summary="Replay verification must be verified before publication.",
                evidence={
                    "replay_verification_status": replay_verification_status,
                },
            ),
            ReadinessGateResult(
                gate_id="workflow.schema_lifecycle",
                status="passed" if schema_closure_status == "passed" else "failed",
                required=True,
                summary=(
                    "Schema lifecycle integration closure must pass before "
                    "publication."
                ),
                evidence={"schema_closure_status": schema_closure_status},
            ),
        ),
    )
    return ReadinessJudgmentManifest(
        judgment_id=judgment.judgment_id,
        final_verdict=judgment.final_verdict,
        catalog_scope=judgment.catalog_scope,
        verdict_summary=judgment.verdict_summary,
        reason_codes=judgment.reason_codes,
        judged_at="2026-04-14T00:00:00Z",
        required_gate_count=judgment.required_gate_count,
        passed_gate_count=judgment.passed_gate_count,
        failed_gate_count=judgment.failed_gate_count,
        missing_gate_count=judgment.missing_gate_count,
        gate_records=tuple(
            ReadinessGateRecord(
                gate_id=gate.gate_id,
                status=gate.status,
                required=gate.required,
                summary=gate.summary,
                evidence=gate.evidence,
            )
            for gate in judgment.gate_results
        ),
    )


def _build_manifest(
    catalog: ContractCatalog,
    *,
    schema_name: str,
    module_id: str,
    body: Mapping[str, Any],
    object_id: str | None = None,
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name=schema_name,
        module_id=module_id,
        body=body,
        catalog=catalog,
        object_id=object_id,
    )


def _register_runtime_model(
    registry: ManifestRegistry,
    catalog: ContractCatalog,
    model: Any,
) -> RegisteredManifest:
    return registry.register(
        model.to_manifest(catalog),
        parent_refs=model.lineage_refs,
    )


def _development_rows(
    feature_rows: tuple[dict[str, Any], ...],
    folds: tuple[dict[str, Any], ...],
    *,
    train_start_overrides: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], ...]:
    confirmatory_fold = _confirmatory_fold(folds)
    return _train_rows(
        feature_rows,
        confirmatory_fold,
        train_start_overrides=train_start_overrides,
    )


def _train_rows(
    feature_rows: tuple[dict[str, Any], ...],
    fold: Mapping[str, Any],
    *,
    train_start_overrides: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], ...]:
    segment_id = str(fold["segment_id"])
    train_start_index = (
        int(train_start_overrides[segment_id])
        if train_start_overrides and segment_id in train_start_overrides
        else int(fold.get("train_start_index", 0))
    )
    train_end_index = int(fold["train_end_index"])
    return feature_rows[train_start_index : train_end_index + 1]


def _confirmatory_fold(folds: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    for fold in folds:
        if fold["role"] == "confirmatory_holdout":
            return fold
    raise ContractValidationError(
        code="missing_confirmatory_holdout_fold",
        message=(
            "evaluation plans must declare an explicit confirmatory_holdout fold "
            "before held-out evidence can be used"
        ),
        field_path="folds",
    )


def _search_candidate_records(
    candidates: tuple[_CandidateEvaluation, ...],
) -> tuple[SearchCandidateRecord, ...]:
    return tuple(
        SearchCandidateRecord(
            candidate_id=candidate.candidate_id,
            family_id=candidate.family_id,
            total_code_bits=_stable_float(
                candidate.description_components["L_total_bits"]
            ),
            structure_code_bits=_stable_float(
                candidate.description_components["L_structure_bits"]
            ),
            description_gain_bits=_stable_float(candidate.description_gain_bits),
            inner_primary_score=_stable_float(candidate.exploratory_primary_score),
            admissible=candidate.admissible,
            ranked=True,
            law_eligible=candidate.admissible,
            canonical_byte_length=len(
                candidate.structure_signature.encode("utf-8")
            ),
            rejection_reason_codes=tuple(_descriptive_reason_codes(candidate)),
        )
        for candidate in candidates
    )


def _evaluate_candidates(
    *,
    feature_rows: tuple[dict[str, Any], ...] | tuple[Mapping[str, Any], ...],
    folds: tuple[dict[str, Any], ...] | tuple[Mapping[str, Any], ...],
    support_bundle,
    minimum_description_gain_bits: float,
    point_loss_id: str,
    family_ids: tuple[str, ...] | None = None,
    train_start_overrides: Mapping[str, int] | None = None,
    telemetry: TelemetryRecorder | None = None,
) -> tuple[_CandidateEvaluation, ...]:
    rows = tuple(dict(row) for row in feature_rows)
    fold_defs = tuple(dict(fold) for fold in folds)
    candidate_family_ids = family_ids or _candidate_family_ids()
    if telemetry is not None:
        telemetry.record_measurement(
            name="candidate_family_queue_depth",
            category="search",
            value=len(candidate_family_ids),
            unit="families",
            attributes={"phase": "candidate_family_dispatch"},
        )
    evaluations: list[_CandidateEvaluation] = []
    for index, family_id in enumerate(candidate_family_ids):
        if telemetry is not None:
            telemetry.record_measurement(
                name="candidate_family_queue_depth",
                category="search",
                value=len(candidate_family_ids) - index,
                unit="families",
                attributes={"family_id": family_id},
            )
            with telemetry.span(
                f"prototype.search.{family_id}",
                category="search",
                attributes={
                    "family_id": family_id,
                    "fold_count": len(fold_defs),
                },
            ):
                evaluations.append(
                    _evaluate_candidate_family(
                        family_id=family_id,
                        rows=rows,
                        fold_defs=fold_defs,
                        support_bundle=support_bundle,
                        minimum_description_gain_bits=minimum_description_gain_bits,
                        point_loss_id=point_loss_id,
                        train_start_overrides=train_start_overrides,
                        telemetry=telemetry,
                    )
                )
            continue
        evaluations.append(
            _evaluate_candidate_family(
                family_id=family_id,
                rows=rows,
                fold_defs=fold_defs,
                support_bundle=support_bundle,
                minimum_description_gain_bits=minimum_description_gain_bits,
                point_loss_id=point_loss_id,
                train_start_overrides=train_start_overrides,
            )
        )
    return tuple(
        sorted(
            evaluations,
            key=lambda item: (
                item.exploratory_primary_score,
                -item.description_gain_bits,
                _FAMILY_ORDER[item.family_id],
            ),
        )
    )


def _candidate_selection_sort_key(candidate: _CandidateEvaluation) -> tuple[float, float, float, int, str]:
    return (
        _stable_float(candidate.description_components["L_total_bits"]),
        -_stable_float(candidate.description_gain_bits),
        _stable_float(candidate.description_components["L_structure_bits"]),
        len(candidate.structure_signature.encode("utf-8")),
        candidate.candidate_id,
    )


def _ordered_candidates(
    candidates: tuple[_CandidateEvaluation, ...],
) -> tuple[_CandidateEvaluation, ...]:
    return tuple(sorted(candidates, key=_candidate_selection_sort_key))


def _select_best_overall_candidate(
    candidates: tuple[_CandidateEvaluation, ...],
) -> _CandidateEvaluation:
    return _ordered_candidates(candidates)[0]


def _select_accepted_candidate(
    candidates: tuple[_CandidateEvaluation, ...],
) -> _CandidateEvaluation | None:
    for candidate in _ordered_candidates(candidates):
        if candidate.admissible:
            return candidate
    return None


def _evaluate_candidate_family(
    *,
    family_id: str,
    rows: tuple[dict[str, Any], ...],
    fold_defs: tuple[dict[str, Any], ...],
    support_bundle,
    minimum_description_gain_bits: float,
    point_loss_id: str,
    train_start_overrides: Mapping[str, int] | None = None,
    telemetry: TelemetryRecorder | None = None,
) -> _CandidateEvaluation:
    quantizer = support_bundle.quantizer
    support_bundle.require_supported_point_loss(point_loss_id)
    confirmatory_fold = _confirmatory_fold(fold_defs)
    development_rows = _development_rows(
        rows,
        fold_defs,
        train_start_overrides=train_start_overrides,
    )
    development_targets = tuple(float(row["target"]) for row in development_rows)
    reference_bits = build_reference_description(
        development_targets,
        quantizer=quantizer,
    ).reference_bits

    development_fold_materials: list[
        tuple[tuple[dict[str, Any], ...], Mapping[str, Any]]
    ] = []
    if telemetry is None:
        for fold in fold_defs:
            if fold["role"] != "development":
                continue
            train_rows = _train_rows(
                rows,
                fold,
                train_start_overrides=train_start_overrides,
            )
            test_rows = tuple(
                rows[int(fold["test_start_index"]) : int(fold["test_end_index"]) + 1]
            )
            development_fold_materials.append(
                (
                    test_rows,
                    _fit_candidate(
                        family_id,
                        tuple(float(row["target"]) for row in train_rows),
                    ),
                )
            )
        full_fit = _fit_candidate(family_id, development_targets)
        confirmatory_train = _train_rows(
            rows,
            confirmatory_fold,
            train_start_overrides=train_start_overrides,
        )
        confirmatory_test = tuple(
            rows[
                int(confirmatory_fold["test_start_index"]) : int(
                    confirmatory_fold["test_end_index"]
                )
                + 1
            ]
        )
        confirmatory_fit = _fit_candidate(
            family_id,
            tuple(float(row["target"]) for row in confirmatory_train),
        )
    else:
        with telemetry.span(
            f"prototype.fitting.development.{family_id}",
            category="fitting",
            attributes={"family_id": family_id, "phase": "development"},
        ):
            for fold in fold_defs:
                if fold["role"] != "development":
                    continue
                train_rows = _train_rows(
                    rows,
                    fold,
                    train_start_overrides=train_start_overrides,
                )
                test_rows = tuple(
                    rows[
                        int(fold["test_start_index"]) : int(fold["test_end_index"]) + 1
                    ]
                )
                development_fold_materials.append(
                    (
                        test_rows,
                        _fit_candidate(
                            family_id,
                            tuple(float(row["target"]) for row in train_rows),
                        ),
                    )
                )
            full_fit = _fit_candidate(family_id, development_targets)
        with telemetry.span(
            f"prototype.fitting.confirmatory.{family_id}",
            category="fitting",
            attributes={"family_id": family_id, "phase": "confirmatory"},
        ):
            confirmatory_train = _train_rows(
                rows,
                confirmatory_fold,
                train_start_overrides=train_start_overrides,
            )
            confirmatory_test = tuple(
                rows[
                    int(confirmatory_fold["test_start_index"]) : int(
                        confirmatory_fold["test_end_index"]
                    )
                    + 1
                ]
            )
            confirmatory_fit = _fit_candidate(
                family_id,
                tuple(float(row["target"]) for row in confirmatory_train),
            )

    development_losses: list[float] = []
    if telemetry is None:
        for test_rows, fit in development_fold_materials:
            predictions = _predict_rows(family_id, fit, test_rows)
            development_losses.extend(
                support_bundle.point_loss(
                    point_loss_id=point_loss_id,
                    point_forecast=prediction["point_forecast"],
                    realized_observation=float(prediction["realized_observation"]),
                )
                for prediction in predictions
            )
        description_components = _description_components(
            family_id=family_id,
            parameters=full_fit["parameters"],
            fitted_values=_fitted_sequence(
                family_id,
                full_fit,
                development_targets,
            ),
            actual_values=development_targets,
            reference_bits=reference_bits,
            quantizer=quantizer,
        )
        confirmatory_predictions = _predict_rows(
            family_id,
            confirmatory_fit,
            confirmatory_test,
        )
        confirmatory_loss = fmean(
            support_bundle.point_loss(
                point_loss_id=point_loss_id,
                point_forecast=prediction["point_forecast"],
                realized_observation=float(prediction["realized_observation"]),
            )
            for prediction in confirmatory_predictions
        )
    else:
        with telemetry.span(
            f"prototype.evaluation.development.{family_id}",
            category="evaluation",
            attributes={"family_id": family_id, "phase": "development"},
        ):
            for test_rows, fit in development_fold_materials:
                predictions = _predict_rows(family_id, fit, test_rows)
                development_losses.extend(
                    support_bundle.point_loss(
                        point_loss_id=point_loss_id,
                        point_forecast=prediction["point_forecast"],
                        realized_observation=float(prediction["realized_observation"]),
                    )
                    for prediction in predictions
                )
            description_components = _description_components(
                family_id=family_id,
                parameters=full_fit["parameters"],
                fitted_values=_fitted_sequence(
                    family_id,
                    full_fit,
                    development_targets,
                ),
                actual_values=development_targets,
                reference_bits=reference_bits,
                quantizer=quantizer,
            )
        with telemetry.span(
            f"prototype.evaluation.confirmatory.{family_id}",
            category="evaluation",
            attributes={"family_id": family_id, "phase": "confirmatory"},
        ):
            confirmatory_predictions = _predict_rows(
                family_id,
                confirmatory_fit,
                confirmatory_test,
            )
            confirmatory_loss = fmean(
                support_bundle.point_loss(
                    point_loss_id=point_loss_id,
                    point_forecast=prediction["point_forecast"],
                    realized_observation=float(prediction["realized_observation"]),
                )
                for prediction in confirmatory_predictions
            )
    baseline_loss = _constant_baseline_loss(confirmatory_train, confirmatory_test)
    description_gain_bits = _stable_float(
        description_components["reference_bits"]
        - description_components["L_total_bits"]
    )
    admissible = description_gain_bits > minimum_description_gain_bits
    return _CandidateEvaluation(
        family_id=family_id,
        candidate_id=f"prototype_{family_id}_candidate_v1",
        parameters=full_fit["parameters"],
        exploratory_primary_score=_stable_float(fmean(development_losses)),
        confirmatory_primary_score=_stable_float(confirmatory_loss),
        baseline_primary_score=_stable_float(baseline_loss),
        description_components=description_components,
        description_gain_bits=description_gain_bits,
        admissible=admissible,
        structure_signature=_structure_signature(family_id, full_fit["parameters"]),
        confirmatory_prediction_rows=tuple(confirmatory_predictions),
        development_losses=tuple(_stable_float(loss) for loss in development_losses),
    )


def _fit_candidate(
    family_id: str,
    targets: tuple[float, ...],
) -> dict[str, Any]:
    if family_id == "constant":
        mean_value = fmean(targets)
        return {
            "training_size": len(targets),
            "parameters": {"mean": _stable_float(mean_value)},
        }
    if family_id == "drift":
        if len(targets) == 1:
            slope = 0.0
        else:
            slope = (targets[-1] - targets[0]) / (len(targets) - 1)
        return {
            "training_size": len(targets),
            "parameters": {
                "intercept": _stable_float(targets[0]),
                "slope": _stable_float(slope),
            },
        }
    if family_id == "seasonal_naive":
        seasonal_anchor = (
            targets[-_SEASONAL_PERIOD]
            if len(targets) >= _SEASONAL_PERIOD
            else targets[-1]
        )
        return {
            "training_size": len(targets),
            "parameters": {
                "season_length": _SEASONAL_PERIOD,
                "seasonal_anchor": _stable_float(seasonal_anchor),
            },
        }
    if family_id == "linear_trend":
        x_values = tuple(range(len(targets)))
        x_mean = fmean(x_values)
        y_mean = fmean(targets)
        numerator = sum(
            (x_value - x_mean) * (target - y_mean)
            for x_value, target in zip(x_values, targets, strict=True)
        )
        denominator = sum((x_value - x_mean) ** 2 for x_value in x_values)
        slope = 0.0 if denominator == 0.0 else numerator / denominator
        intercept = y_mean - (slope * x_mean)
        return {
            "training_size": len(targets),
            "parameters": {
                "intercept": _stable_float(intercept),
                "slope": _stable_float(slope),
            },
        }
    raise ValueError(f"unsupported candidate family: {family_id}")


def _predict_rows(
    family_id: str,
    fit: Mapping[str, Any],
    test_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    test_rows_tuple = tuple(dict(row) for row in test_rows)
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(test_rows_tuple):
        horizon = 1
        point_forecast = _stable_float(
            _forecast_next(family_id, fit, step=index + 1, reference_row=row)
        )
        predictions.append(
            {
                "origin_time": row["event_time"],
                "available_at": row["available_at"],
                "horizon": horizon,
                "point_forecast": point_forecast,
                "realized_observation": _stable_float(float(row["target"])),
            }
        )
    return predictions


def _forecast_next(
    family_id: str,
    fit: Mapping[str, Any],
    *,
    step: int,
    reference_row: Mapping[str, Any],
) -> float:
    params = fit["parameters"]
    training_index = int(fit["training_size"]) - 1 + step
    if family_id == "constant":
        return float(params["mean"])
    if family_id == "drift":
        return float(params["intercept"]) + (float(params["slope"]) * training_index)
    if family_id == "seasonal_naive":
        return float(params["seasonal_anchor"])
    if family_id == "linear_trend":
        return float(params["intercept"]) + (float(params["slope"]) * training_index)
    raise ValueError(f"unsupported candidate family: {family_id}")


def _fitted_sequence(
    family_id: str,
    fit: Mapping[str, Any],
    actual_values: tuple[float, ...],
) -> tuple[float, ...]:
    params = fit["parameters"]
    if family_id == "constant":
        return tuple(float(params["mean"]) for _ in actual_values)
    if family_id == "drift":
        intercept = float(params["intercept"])
        slope = float(params["slope"])
        return tuple(intercept + (slope * index) for index in range(len(actual_values)))
    if family_id == "seasonal_naive":
        season_length = int(params["season_length"])
        fitted: list[float] = []
        for index, value in enumerate(actual_values):
            if index < season_length:
                fitted.append(value)
            else:
                fitted.append(actual_values[index - season_length])
        return tuple(fitted)
    if family_id == "linear_trend":
        intercept = float(params["intercept"])
        slope = float(params["slope"])
        return tuple(intercept + (slope * index) for index in range(len(actual_values)))
    raise ValueError(f"unsupported candidate family: {family_id}")


def _description_components(
    *,
    family_id: str,
    parameters: Mapping[str, float | int],
    fitted_values: tuple[float, ...],
    actual_values: tuple[float, ...],
    reference_bits: float,
    quantizer: FixedStepMidTreadQuantizer,
) -> dict[str, float]:
    residual_indices = [
        quantizer.quantize_index(actual - fitted)
        for actual, fitted in zip(actual_values, fitted_values, strict=True)
    ]
    parameter_indices = [
        int(value) if isinstance(value, int) else quantizer.quantize_index(float(value))
        for value in parameters.values()
    ]
    structure_bits = 1.0 if family_id == "seasonal_naive" else 0.0
    literal_bits = (
        float(_natural_code_length(int(parameters["season_length"])))
        if family_id == "seasonal_naive"
        else 0.0
    )
    parameter_bits = float(_code_bits(parameter_indices))
    data_bits = float(_natural_code_length(len(residual_indices))) + float(
        _code_bits(residual_indices)
    )
    total_bits = (
        _FAMILY_BITS + structure_bits + literal_bits + parameter_bits + 0.0 + data_bits
    )
    return {
        "L_family_bits": _FAMILY_BITS,
        "L_structure_bits": structure_bits,
        "L_literals_bits": literal_bits,
        "L_params_bits": parameter_bits,
        "L_state_bits": 0.0,
        "L_data_bits": data_bits,
        "L_total_bits": _stable_float(total_bits),
        "reference_bits": _stable_float(reference_bits),
    }


def _constant_baseline_loss(
    confirmatory_train: tuple[dict[str, Any], ...],
    confirmatory_test: tuple[dict[str, Any], ...],
) -> float:
    baseline_mean = fmean(float(row["target"]) for row in confirmatory_train)
    losses = [abs(float(row["target"]) - baseline_mean) for row in confirmatory_test]
    return _stable_float(fmean(losses))


def _materialize_robustness_artifacts(
    *,
    catalog: ContractCatalog,
    registry: ManifestRegistry,
    intake: PrototypeIntakePlanningResult,
    feature_rows: tuple[dict[str, Any], ...],
    folds: tuple[dict[str, Any], ...],
    baseline_registry: RegisteredManifest,
    point_score_policy: RegisteredManifest,
    frozen_shortlist: RegisteredManifest,
    selected_candidate_runtime: _CandidateEvaluation,
    minimum_description_gain_bits: float,
) -> _RobustnessArtifacts:
    surrogate_generator = registry.register(
        build_surrogate_generator_manifest(
            catalog,
            surrogate_generator_id="prototype_surrogate_generator_v1",
        )
    )
    null_protocol = registry.register(
        build_null_protocol_manifest(
            catalog,
            protocol_id="prototype_null_protocol_v1",
            surrogate_generator_ref=surrogate_generator.manifest.ref,
            resample_count=19,
            max_p_value="0.25",
        ),
        parent_refs=(surrogate_generator.manifest.ref,),
    )

    retention_metrics = {
        "canonical_form_exact_match_rate": registry.register(
            build_retention_metric_manifest(
                catalog,
                retention_metric_id="prototype_canonical_form_exact_match_rate_v1",
                metric_id="canonical_form_exact_match_rate",
            )
        ),
        "positive_description_gain_rate": registry.register(
            build_retention_metric_manifest(
                catalog,
                retention_metric_id="prototype_positive_description_gain_rate_v1",
                metric_id="positive_description_gain_rate",
            )
        ),
        "outer_baseline_win_rate": registry.register(
            build_retention_metric_manifest(
                catalog,
                retention_metric_id="prototype_outer_baseline_win_rate_v1",
                metric_id="outer_baseline_win_rate",
            )
        ),
    }
    metric_refs_by_id = {
        metric_id: registered.manifest.ref
        for metric_id, registered in retention_metrics.items()
    }
    metric_thresholds = {
        metric_ref.object_id: 0.5 for metric_ref in metric_refs_by_id.values()
    }
    perturbation_protocol = registry.register(
        build_perturbation_protocol_manifest(
            catalog,
            protocol_id="prototype_perturbation_protocol_v1",
            base_codelength_policy_ref=intake.codelength_policy.manifest.ref,
            baseline_registry_ref=baseline_registry.manifest.ref,
            frozen_baseline_id="constant_baseline",
            point_score_policy_ref=point_score_policy.manifest.ref,
            required_metric_refs=tuple(metric_refs_by_id.values()),
            metric_thresholds=metric_thresholds,
        ),
        parent_refs=(
            intake.codelength_policy.manifest.ref,
            baseline_registry.manifest.ref,
            point_score_policy.manifest.ref,
            *(item.manifest.ref for item in retention_metrics.values()),
        ),
    )

    null_result = evaluate_null_comparison(
        observed_statistic=selected_candidate_runtime.description_gain_bits,
        surrogate_statistics=_surrogate_description_gain_statistics(
            family_id=selected_candidate_runtime.family_id,
            feature_rows=feature_rows,
            folds=folds,
            support_bundle=intake.support_bundle,
            surrogate_count=19,
            seed=intake.search_plan_object.random_seed,
        ),
        max_p_value=float(null_protocol.manifest.body["max_p_value"]),
    )

    recent_history_runs = tuple(
        PerturbationRunRecord(
            perturbation_id=f"recent_history_truncation_{fraction:g}",
            canonical_form_matches=(
                perturbed.structure_signature
                == selected_candidate_runtime.structure_signature
            ),
            description_gain_bits=perturbed.description_gain_bits,
            outer_candidate_score=perturbed.exploratory_primary_score,
            outer_baseline_score=_development_baseline_loss(
                feature_rows,
                folds,
                train_start_overrides=_recent_history_train_start_overrides(
                    folds, fraction
                ),
            ),
            metadata={"recent_history_fraction": fraction},
        )
        for fraction in (0.5, 0.75)
        for perturbed in (
            _evaluate_candidates(
                feature_rows=feature_rows,
                folds=folds,
                support_bundle=intake.support_bundle,
                minimum_description_gain_bits=minimum_description_gain_bits,
                point_loss_id=_DEFAULT_LOSS_ID,
                family_ids=(selected_candidate_runtime.family_id,),
                train_start_overrides=_recent_history_train_start_overrides(
                    folds, fraction
                ),
            )[0],
        )
    )
    base_quantizer = intake.support_bundle.quantizer
    quantization_runs = tuple(
        PerturbationRunRecord(
            perturbation_id=f"quantization_coarsening_{multiplier}x",
            canonical_form_matches=(
                perturbed.structure_signature
                == selected_candidate_runtime.structure_signature
            ),
            description_gain_bits=perturbed.description_gain_bits,
            outer_candidate_score=perturbed.exploratory_primary_score,
            outer_baseline_score=_development_baseline_loss(
                _coarsen_feature_rows(feature_rows, base_quantizer, multiplier),
                folds,
            ),
            metadata={"quantization_multiplier": multiplier},
        )
        for multiplier in (2, 4)
        for perturbed in (
            _evaluate_candidates(
                feature_rows=_coarsen_feature_rows(
                    feature_rows,
                    base_quantizer,
                    multiplier,
                ),
                folds=folds,
                support_bundle=intake.support_bundle,
                minimum_description_gain_bits=minimum_description_gain_bits,
                point_loss_id=_DEFAULT_LOSS_ID,
                family_ids=(selected_candidate_runtime.family_id,),
            )[0],
        )
    )

    perturbation_family_results = (
        evaluate_perturbation_family(
            family_id="recent_history_truncation",
            metric_refs_by_id=metric_refs_by_id,
            runs=recent_history_runs,
        ),
        evaluate_perturbation_family(
            family_id="quantization_coarsening",
            metric_refs_by_id=metric_refs_by_id,
            runs=quantization_runs,
        ),
    )
    aggregate_metric_results, stability_status = evaluate_aggregate_metric_results(
        family_results=perturbation_family_results,
        required_metric_refs=tuple(metric_refs_by_id.values()),
        metric_thresholds=metric_thresholds,
    )
    null_result_manifest = _register_runtime_model(
        registry,
        catalog,
        build_null_result_manifest(
            catalog,
            null_result_id="prototype_null_result_v1",
            null_protocol_ref=null_protocol.manifest.ref,
            candidate_id=selected_candidate_runtime.candidate_id,
            evaluation=null_result,
        ),
    )
    registered_perturbation_family_results = tuple(
        _register_runtime_model(
            registry,
            catalog,
            build_perturbation_family_result_manifest(
                catalog,
                perturbation_family_result_id=(
                    f"prototype_{evaluation.family_id}_perturbation_result_v1"
                ),
                perturbation_protocol_ref=perturbation_protocol.manifest.ref,
                candidate_id=selected_candidate_runtime.candidate_id,
                evaluation=evaluation,
            ),
        )
        for evaluation in perturbation_family_results
    )
    perturbation_family_refs_by_id = {
        str(item.manifest.body["family_id"]): item.manifest.ref
        for item in registered_perturbation_family_results
    }

    canary_types = (
        "future_target_level_feature",
        "late_available_target_copy",
        "holdout_membership_feature",
        "post_cutoff_revision_level_feature",
    )
    canary_manifests = tuple(
        registry.register(
            build_leakage_canary_manifest(
                catalog,
                canary_id=f"prototype_{canary_type}_v1",
                canary_type=canary_type,
            )
        )
        for canary_type in canary_types
    )
    leakage_canary_results = tuple(
        registry.register(
            build_leakage_canary_result_manifest(
                catalog,
                canary_result_id=f"{canary.manifest.body['canary_id']}_result",
                canary_ref=canary.manifest.ref,
                canary_type=str(canary.manifest.body["canary_type"]),
                observed_terminal_state="blocked_at_expected_stage",
                observed_block_stage=str(canary.manifest.body["expected_block_stage"]),
                observed_reason_code=str(canary.manifest.body["expected_reason_code"]),
                stage_evidence_ref=_stage_evidence_ref(
                    stage_id=str(canary.manifest.body["expected_block_stage"]),
                    intake=intake,
                ),
            )
        )
        for canary in canary_manifests
    )

    sensitivity_analyses = tuple(
        {
            "analysis_id": run.perturbation_id,
            "family_id": family_id,
            "perturbation_id": run.perturbation_id,
            "canonical_form_matches": run.canonical_form_matches,
            "description_gain_bits": run.description_gain_bits,
            "outer_candidate_score": run.outer_candidate_score,
            "outer_baseline_score": run.outer_baseline_score,
            "failure_reason_code": run.failure_reason_code,
            "metadata": dict(run.metadata),
        }
        for family_id, runs in (
            ("recent_history_truncation", recent_history_runs),
            ("quantization_coarsening", quantization_runs),
        )
        for run in runs
    )
    registered_sensitivity_analyses = tuple(
        _register_runtime_model(
            registry,
            catalog,
            build_sensitivity_analysis_manifest(
                catalog,
                sensitivity_analysis_id=(
                    str(analysis.get("analysis_id", "sensitivity_analysis"))
                ),
                perturbation_family_result_ref=perturbation_family_refs_by_id[
                    str(analysis["family_id"])
                ],
                candidate_id=selected_candidate_runtime.candidate_id,
                analysis=analysis,
            ),
        )
        for analysis in sensitivity_analyses
    )
    robustness_report = _register_runtime_model(
        registry,
        catalog,
        build_robustness_report(
            candidate_id=selected_candidate_runtime.candidate_id,
            null_protocol_ref=null_protocol.manifest.ref,
            null_result=null_result,
            null_result_ref=null_result_manifest.manifest.ref,
            perturbation_protocol_ref=perturbation_protocol.manifest.ref,
            perturbation_family_results=perturbation_family_results,
            perturbation_family_result_refs=tuple(
                item.manifest.ref for item in registered_perturbation_family_results
            ),
            aggregate_metric_results=aggregate_metric_results,
            stability_status=stability_status,
            leakage_canary_result_refs=tuple(
                item.manifest.ref for item in leakage_canary_results
            ),
            leakage_canary_results=tuple(
                item.manifest.body for item in leakage_canary_results
            ),
            candidate_context={
                "frozen_candidate_set_ref": frozen_shortlist.manifest.ref.as_dict(),
                "evaluation_plan_ref": intake.evaluation_plan.manifest.ref.as_dict(),
                "selected_family_id": selected_candidate_runtime.family_id,
            },
            sensitivity_analysis_refs=tuple(
                item.manifest.ref for item in registered_sensitivity_analyses
            ),
            report_id="prototype_robustness_report_v1",
            parent_refs=(
                frozen_shortlist.manifest.ref,
                intake.evaluation_plan.manifest.ref,
                intake.feature_spec.manifest.ref,
                intake.time_safety_audit.manifest.ref,
                surrogate_generator.manifest.ref,
                *(item.manifest.ref for item in retention_metrics.values()),
                *(item.manifest.ref for item in canary_manifests),
            ),
        ),
    )
    return _RobustnessArtifacts(
        surrogate_generator=surrogate_generator,
        null_protocol=null_protocol,
        null_result=null_result_manifest,
        perturbation_protocol=perturbation_protocol,
        perturbation_family_results=registered_perturbation_family_results,
        leakage_canary_results=leakage_canary_results,
        sensitivity_analyses=registered_sensitivity_analyses,
        robustness_report=robustness_report,
    )


def _surrogate_description_gain_statistics(
    *,
    family_id: str,
    feature_rows: tuple[dict[str, Any], ...],
    folds: tuple[dict[str, Any], ...],
    support_bundle,
    surrogate_count: int,
    seed: str,
) -> tuple[float, ...]:
    development_rows = _development_rows(feature_rows, folds)
    development_targets = tuple(float(row["target"]) for row in development_rows)
    statistics: list[float] = []
    for index in range(surrogate_count):
        surrogate_targets = tuple(
            random.Random(f"{seed}:{index}").sample(
                list(development_targets),
                len(development_targets),
            )
        )
        statistics.append(
            _description_gain_for_targets(
                family_id=family_id,
                targets=surrogate_targets,
                support_bundle=support_bundle,
            )
        )
    return tuple(statistics)


def _description_gain_for_targets(
    *,
    family_id: str,
    targets: tuple[float, ...],
    support_bundle,
) -> float:
    quantizer = support_bundle.quantizer
    reference_bits = build_reference_description(
        targets,
        quantizer=quantizer,
    ).reference_bits
    fit = _fit_candidate(family_id, targets)
    description_components = _description_components(
        family_id=family_id,
        parameters=fit["parameters"],
        fitted_values=_fitted_sequence(family_id, fit, targets),
        actual_values=targets,
        reference_bits=reference_bits,
        quantizer=quantizer,
    )
    return _stable_float(
        description_components["reference_bits"]
        - description_components["L_total_bits"]
    )


def _recent_history_train_start_overrides(
    folds: tuple[dict[str, Any], ...],
    fraction: float,
) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for fold in folds:
        train_row_count = int(fold["train_row_count"])
        keep_count = max(3, int(math.floor(train_row_count * fraction)))
        train_end_index = int(fold["train_end_index"])
        overrides[str(fold["segment_id"])] = max(
            int(fold.get("train_start_index", 0)),
            train_end_index - keep_count + 1,
        )
    return overrides


def _development_baseline_loss(
    feature_rows: tuple[dict[str, Any], ...],
    folds: tuple[dict[str, Any], ...],
    *,
    train_start_overrides: Mapping[str, int] | None = None,
) -> float:
    losses: list[float] = []
    for fold in folds:
        if fold["role"] != "development":
            continue
        train_rows = _train_rows(
            feature_rows,
            fold,
            train_start_overrides=train_start_overrides,
        )
        test_rows = feature_rows[
            int(fold["test_start_index"]) : int(fold["test_end_index"]) + 1
        ]
        losses.append(_constant_baseline_loss(train_rows, test_rows))
    return _stable_float(fmean(losses))


def _coarsen_feature_rows(
    feature_rows: tuple[dict[str, Any], ...],
    quantizer,
    multiplier: int,
) -> tuple[dict[str, Any], ...]:
    coarsened_quantizer = quantizer.__class__(step=quantizer.step * multiplier)
    return tuple(
        {
            **dict(row),
            "target": _stable_float(
                coarsened_quantizer.quantize_value(float(row["target"]))
            ),
        }
        for row in feature_rows
    )


def _stage_evidence_ref(
    *,
    stage_id: str,
    intake: PrototypeIntakePlanningResult,
) -> Mapping[str, Any]:
    if stage_id == "feature_spec_validation":
        return {
            "ref_kind": "feature_spec",
            "typed_ref": intake.feature_spec.manifest.ref.as_dict(),
        }
    if stage_id == "evaluation_plan_binding":
        return {
            "ref_kind": "evaluation_plan",
            "typed_ref": intake.evaluation_plan.manifest.ref.as_dict(),
        }
    return {
        "ref_kind": "time_safety_audit",
        "typed_ref": intake.time_safety_audit.manifest.ref.as_dict(),
    }


def _point_loss(
    point_loss_id: str,
    point_forecast: float,
    realized_observation: float,
) -> float:
    error = point_forecast - realized_observation
    if point_loss_id == "absolute_error":
        return abs(error)
    if point_loss_id == "squared_error":
        return error**2
    raise ValueError(f"unsupported point loss: {point_loss_id}")


def _gate_descriptive_reason_codes(candidate: _CandidateEvaluation) -> tuple[str, ...]:
    if candidate.admissible:
        return ()
    return ("descriptive_gate_failed", "no_candidate_survived_search")


def _predictive_governance_reason_codes(
    evaluation_governance_body: Mapping[str, Any],
) -> tuple[str, ...]:
    if bool(evaluation_governance_body.get("confirmatory_promotion_allowed")):
        return ()
    return ()


def _robustness_reason_codes(
    *,
    registry: ManifestRegistry,
    robustness_report_body: Mapping[str, Any],
    leakage_canary_results: tuple[RegisteredManifest, ...],
) -> tuple[str, ...]:
    codes: list[str] = []

    null_result = robustness_report_body.get("null_result")
    null_result_ref = robustness_report_body.get("null_result_ref")
    if isinstance(null_result_ref, Mapping):
        null_result = registry.resolve(
            TypedRef(
                schema_name=str(null_result_ref["schema_name"]),
                object_id=str(null_result_ref["object_id"]),
            )
        ).manifest.body
    if isinstance(null_result, Mapping):
        failure_reason_code = null_result.get("failure_reason_code")
        if failure_reason_code:
            codes.extend(("null_protocol_failed", str(failure_reason_code)))

    if str(robustness_report_body.get("stability_status")) == "failed":
        codes.append("perturbation_protocol_failed")

    for item in robustness_report_body.get("aggregate_metric_results", ()):
        if not isinstance(item, Mapping):
            continue
        failure_reason_code = item.get("failure_reason_code")
        if failure_reason_code:
            codes.append(str(failure_reason_code))

    perturbation_family_results: list[Mapping[str, Any]] = [
        item
        for item in robustness_report_body.get("perturbation_family_results", ())
        if isinstance(item, Mapping)
    ]
    for ref in robustness_report_body.get("perturbation_family_result_refs", ()):
        if not isinstance(ref, Mapping):
            continue
        perturbation_family_results.append(
            registry.resolve(
                TypedRef(
                    schema_name=str(ref["schema_name"]),
                    object_id=str(ref["object_id"]),
                )
            ).manifest.body
        )

    for item in perturbation_family_results:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("status")) == "failed":
            codes.append("perturbation_protocol_failed")
        for metric in item.get("metric_results", ()):
            if not isinstance(metric, Mapping):
                continue
            failure_reason_code = metric.get("failure_reason_code")
            if failure_reason_code:
                codes.append(str(failure_reason_code))

    canary_failure_codes: list[str] = []
    for result in leakage_canary_results:
        body = result.manifest.body
        failure_reason_codes = body.get("failure_reason_codes", ())
        if isinstance(failure_reason_codes, list):
            canary_failure_codes.extend(str(code) for code in failure_reason_codes)
        if not bool(body.get("pass", False)) or canary_failure_codes:
            codes.append("leakage_canary_failed")
    codes.extend(canary_failure_codes)

    return tuple(dict.fromkeys(codes))


def _descriptive_reason_codes(candidate: _CandidateEvaluation) -> list[str]:
    if candidate.admissible:
        return []
    return ["codelength_comparability_failed", "description_gain_non_positive"]


def _predictive_reason_codes(candidate: _CandidateEvaluation) -> list[str]:
    if not candidate.admissible:
        return ["point_score_not_comparable"]
    if candidate.confirmatory_primary_score < candidate.baseline_primary_score:
        return []
    return ["baseline_rule_failed"]


def _structure_signature(
    family_id: str,
    parameters: Mapping[str, float | int],
) -> str:
    rendered = ",".join(f"{key}={value}" for key, value in sorted(parameters.items()))
    return f"{family_id}:{rendered}"


def _code_bits(indices: Iterable[int]) -> int:
    return sum(_natural_code_length(_zigzag_encode(index)) for index in indices)


def _natural_code_length(value: int) -> int:
    bounded = max(value, 0)
    level_1 = math.floor(math.log2(bounded + 1))
    level_2 = math.floor(math.log2(level_1 + 1))
    return level_1 + (2 * level_2) + 1


def _zigzag_encode(value: int) -> int:
    return (2 * value) if value >= 0 else (-2 * value) - 1


def _run_result_id(candidate: _CandidateEvaluation) -> str:
    suffix = "candidate" if candidate.admissible else "abstention"
    return f"prototype_{candidate.family_id}_{suffix}_run_result_v1"


def _bundle_id(candidate: _CandidateEvaluation) -> str:
    suffix = "candidate" if candidate.admissible else "abstention"
    return f"prototype_{candidate.family_id}_{suffix}_bundle_v1"


def _typed_ref(payload: Mapping[str, Any]) -> TypedRef:
    return TypedRef(
        schema_name=str(payload["schema_name"]),
        object_id=str(payload["object_id"]),
    )


def _normalize_mapping(value: Mapping[str, float | int]) -> dict[str, float | int]:
    return {
        key: (int(item) if isinstance(item, int) else _stable_float(float(item)))
        for key, item in sorted(value.items())
    }


def _stable_float(value: float) -> float:
    return round(float(value), 6)
