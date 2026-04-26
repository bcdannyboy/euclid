from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np

from euclid.algorithmic_dsl import (
    canonicalize_fraction,
    evaluate_algorithmic_program,
    parse_algorithmic_program,
)
from euclid.cir.models import (
    CandidateIntermediateRepresentation,
    CIRLiteral,
    CIRLiteralBlock,
    CIRReplayHook,
    CIRReplayHooks,
)
from euclid.cir.normalize import (
    build_cir_candidate_from_reducer,
    require_full_cir_closure,
)
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.expr.evaluators import evaluate_expression
from euclid.expr.serialization import expression_from_dict
from euclid.fit.multi_horizon import (
    FitStrategySpec,
    resolve_fit_strategy,
    training_origin_set_id,
)
from euclid.fit.parameterization import ParameterDeclaration
from euclid.fit.refit import FitDataSplit, fit_cir_candidate
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    CandidateSpecManifest,
    CandidateStateManifest,
    ResidualHistoryManifest,
    SearchPlanManifest,
)
from euclid.modules.features import FeatureView
from euclid.modules.residual_history import (
    ForecastResidualRecord,
    ResidualHistorySummary,
    ResidualHistoryValidationResult,
    summarize_residual_history,
    validate_residual_history,
)
from euclid.modules.scoring import _aggregate_primary_scores, _point_loss
from euclid.modules.shared_plus_local_decomposition import (
    fit_shared_plus_local_decomposition,
)
from euclid.modules.split_planning import (
    EvaluationPlan,
    EvaluationSegment,
    TrainingOriginPanelRow,
    build_legal_training_origin_panel,
    segment_training_rows,
)
from euclid.reducers.composition import (
    AdditiveResidualComposition,
    PiecewiseComposition,
    RegimeConditionedComposition,
    extract_component_mapping,
    merge_component_mapping,
    resolve_regime_weights,
    select_piecewise_segment,
)
from euclid.reducers.models import (
    ReducerAdmissibilityObject,
    ReducerFamilyId,
    ReducerObject,
    ReducerParameter,
    ReducerParameterObject,
    ReducerStateObject,
    ReducerStateSemantics,
    ReducerStateSlot,
    ReducerStateUpdateContext,
    ReducerStateUpdateRule,
    parse_reducer_composition,
)
from euclid.runtime.hashing import canonicalize_json

_SCOPE_ID = "euclid_v1_binding_scope@1.0.0"
_LEGACY_FIT_STRATEGY_IDS = frozenset({"legacy_one_step"})
_RECURSIVE_ROLLOUT_STRATEGY_IDS = frozenset({"recursive", "recursive_rollout"})
_DIRECT_ANALYTIC_STRATEGY_IDS = frozenset({"direct", "direct_analytic"})
_JOINT_ANALYTIC_STRATEGY_IDS = frozenset({"joint", "joint_analytic"})
_RECTIFY_ANALYTIC_STRATEGY_IDS = frozenset({"rectify", "rectify_analytic"})


@dataclass(frozen=True)
class PersistentStateTransition:
    transition_index: int
    observation_index: int
    event_time: str
    available_at: str
    observed_value: float
    state_before: Mapping[str, Any]
    state_after: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "transition_index": self.transition_index,
            "observation_index": self.observation_index,
            "event_time": self.event_time,
            "available_at": self.available_at,
            "observed_value": self.observed_value,
            "state_before": dict(self.state_before),
            "state_after": dict(self.state_after),
        }


@dataclass(frozen=True)
class CandidateWindowFitResult:
    candidate_id: str
    family_id: str
    candidate_hash: str
    fit_window_id: str
    stage_id: str
    training_row_count: int
    backend_id: str
    objective_id: str
    parameter_summary: Mapping[str, float | int]
    initial_state: Mapping[str, Any]
    final_state: Mapping[str, Any]
    state_transitions: tuple[PersistentStateTransition, ...]
    optimizer_diagnostics: Mapping[str, Any]
    fitted_candidate: CandidateIntermediateRepresentation
    residual_history: tuple[ForecastResidualRecord, ...]
    residual_history_summary: ResidualHistorySummary
    residual_history_validation: ResidualHistoryValidationResult

    @property
    def state_transition_count(self) -> int:
        return len(self.state_transitions)


@dataclass(frozen=True)
class CandidateFitArtifactBundle:
    candidate_spec: ManifestEnvelope
    candidate_state: ManifestEnvelope
    reducer_artifact: ManifestEnvelope
    residual_history: ManifestEnvelope


@dataclass(frozen=True)
class _FamilyFitResult:
    backend_id: str
    objective_id: str
    parameter_summary: Mapping[str, float | int]
    updated_literals: Mapping[str, float | int]
    initial_state: Mapping[str, Any]
    final_state: Mapping[str, Any]
    state_transitions: tuple[PersistentStateTransition, ...]
    converged: bool
    iteration_count: int
    final_loss: float
    component_diagnostics: Mapping[str, Any] | None = None
    residual_row_weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class _ResidualForecastTrace:
    point_forecast: float
    replay_state: Mapping[str, Any]
    component_alignment: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _ResidualHistoryCapture:
    records: tuple[ForecastResidualRecord, ...]
    summary: ResidualHistorySummary
    validation: ResidualHistoryValidationResult
    diagnostics: Mapping[str, Any]


def fit_candidate_window(
    *,
    candidate: CandidateIntermediateRepresentation,
    feature_view: FeatureView,
    fit_window: EvaluationSegment,
    search_plan: SearchPlanManifest,
    stage_id: str = "inner_search",
    shared_local_backend_preference: str | None = None,
    fit_strategy: FitStrategySpec | None = None,
) -> CandidateWindowFitResult:
    closed_candidate = require_full_cir_closure(
        candidate,
        consumer="candidate_fitting",
    )
    legal_feature_view = feature_view.require_stage_reuse("candidate_fitting")
    training_rows = segment_training_rows(
        feature_view=legal_feature_view,
        evaluation_segment=fit_window,
    )
    if not training_rows:
        raise ContractValidationError(
            code="invalid_fit_window",
            message="fit windows must contain at least one training row",
            field_path="fit_window.train_row_count",
        )

    resolved_fit_strategy = fit_strategy or resolve_fit_strategy(
        horizon_set=fit_window.horizon_set,
    )
    family_fit = _fit_family(
        candidate=closed_candidate,
        feature_view=legal_feature_view,
        fit_window=fit_window,
        training_rows=training_rows,
        random_seed=search_plan.random_seed,
        fit_strategy=resolved_fit_strategy,
        shared_local_backend_preference=shared_local_backend_preference,
    )
    optimizer_diagnostics = {
        "backend_id": family_fit.backend_id,
        "objective_id": family_fit.objective_id,
        "seed_value": search_plan.random_seed,
        "converged": family_fit.converged,
        "iteration_count": family_fit.iteration_count,
        "final_loss": family_fit.final_loss,
        "fit_strategy": resolved_fit_strategy.as_dict(),
        "fit_strategy_id": resolved_fit_strategy.strategy_id,
        "fit_strategy_identity": resolved_fit_strategy.identity_hash,
        "objective_horizon_set": list(resolved_fit_strategy.horizon_set),
        "objective_horizon_weights": [
            dict(item) for item in resolved_fit_strategy.horizon_weights
        ],
        "objective_point_loss_id": resolved_fit_strategy.point_loss_id,
        "objective_entity_aggregation_mode": (
            resolved_fit_strategy.entity_aggregation_mode
        ),
    }
    if family_fit.objective_id == "least_squares_one_step_residual_v1":
        optimizer_diagnostics["fit_geometry"] = "legacy_one_step"
    if family_fit.component_diagnostics:
        if family_fit.backend_id == "euclid_unified_fit_layer_v1":
            optimizer_diagnostics.update(dict(family_fit.component_diagnostics))
        elif "fit_strategy_id" in family_fit.component_diagnostics:
            optimizer_diagnostics.update(dict(family_fit.component_diagnostics))
        elif any(
            key in family_fit.component_diagnostics
            for key in ("analytic_feature_terms", "spectral_harmonic_group")
        ):
            optimizer_diagnostics.update(dict(family_fit.component_diagnostics))
        else:
            optimizer_diagnostics["composition_runtime"] = dict(
                family_fit.component_diagnostics
            )
    candidate_id = (
        closed_candidate.evidence_layer.backend_origin_record.source_candidate_id
    )
    residual_capture = _capture_residual_history(
        candidate=closed_candidate,
        family_fit=family_fit,
        feature_view=legal_feature_view,
        fit_window=fit_window,
        training_rows=training_rows,
        candidate_id=candidate_id,
        fit_strategy=resolved_fit_strategy,
    )
    optimizer_diagnostics["residual_history"] = dict(residual_capture.diagnostics)
    optimizer_diagnostics["training_scored_origin_set_id"] = (
        residual_capture.diagnostics["training_scored_origin_set_id"]
    )
    fitted_candidate = _build_fitted_candidate(
        candidate=closed_candidate,
        parameter_summary=family_fit.parameter_summary,
        updated_literals=family_fit.updated_literals,
        final_state=family_fit.final_state,
        backend_id=family_fit.backend_id,
        fit_window_id=fit_window.segment_id,
        optimizer_diagnostics=optimizer_diagnostics,
        component_diagnostics=family_fit.component_diagnostics,
    )
    return CandidateWindowFitResult(
        candidate_id=candidate_id,
        family_id=closed_candidate.structural_layer.cir_family_id,
        candidate_hash=closed_candidate.canonical_hash(),
        fit_window_id=fit_window.segment_id,
        stage_id=stage_id,
        training_row_count=len(training_rows),
        backend_id=family_fit.backend_id,
        objective_id=family_fit.objective_id,
        parameter_summary=dict(family_fit.parameter_summary),
        initial_state=dict(family_fit.initial_state),
        final_state=dict(family_fit.final_state),
        state_transitions=family_fit.state_transitions,
        optimizer_diagnostics=optimizer_diagnostics,
        fitted_candidate=fitted_candidate,
        residual_history=residual_capture.records,
        residual_history_summary=residual_capture.summary,
        residual_history_validation=residual_capture.validation,
    )


def fit_candidate_development_windows(
    *,
    candidates: Sequence[CandidateIntermediateRepresentation],
    feature_view: FeatureView,
    evaluation_plan: EvaluationPlan,
    search_plan: SearchPlanManifest,
) -> tuple[CandidateWindowFitResult, ...]:
    results: list[CandidateWindowFitResult] = []
    for candidate in candidates:
        for segment in evaluation_plan.development_segments:
            results.append(
                fit_candidate_window(
                    candidate=candidate,
                    feature_view=feature_view,
                    fit_window=segment,
                    search_plan=search_plan,
                    stage_id="inner_search",
                )
            )
    return tuple(results)


def build_candidate_fit_artifacts(
    *,
    catalog: ContractCatalog,
    fit_result: CandidateWindowFitResult,
    search_plan_ref: TypedRef,
    selection_floor_bits: float,
    description_gain_bits: float | None = None,
    exploratory_primary_score: float | None = None,
) -> CandidateFitArtifactBundle:
    closed_fitted_candidate = require_full_cir_closure(
        fit_result.fitted_candidate,
        consumer="candidate_fit_artifacts",
    )
    candidate_spec_model = CandidateSpecManifest(
        candidate_spec_id=f"{fit_result.candidate_id}__{fit_result.fit_window_id}__spec",
        candidate_id=fit_result.candidate_id,
        fit_window_id=fit_result.fit_window_id,
        family_id=fit_result.family_id,
        parameter_summary=_normalize_parameter_summary(fit_result.parameter_summary),
        selection_floor_bits=_stable_float(selection_floor_bits),
        optimizer_backend_id=fit_result.backend_id,
        parent_refs=(search_plan_ref,),
    )
    candidate_spec = candidate_spec_model.to_manifest(catalog)
    candidate_state_model = CandidateStateManifest(
        candidate_state_id=(
            f"{fit_result.candidate_id}__{fit_result.fit_window_id}__state"
        ),
        candidate_id=fit_result.candidate_id,
        fit_window_id=fit_result.fit_window_id,
        stage_id=fit_result.stage_id,
        lifecycle_state="fit",
        optimizer_backend_id=fit_result.backend_id,
        optimizer_objective_id=fit_result.objective_id,
        optimizer_seed=str(fit_result.optimizer_diagnostics["seed_value"]),
        converged=bool(fit_result.optimizer_diagnostics["converged"]),
        iteration_count=int(fit_result.optimizer_diagnostics["iteration_count"]),
        final_loss=_stable_float(float(fit_result.optimizer_diagnostics["final_loss"])),
        training_row_count=fit_result.training_row_count,
        initial_state=dict(fit_result.initial_state),
        final_state=dict(fit_result.final_state),
        state_transitions=tuple(
            transition.as_dict() for transition in fit_result.state_transitions
        ),
        candidate_spec_ref=candidate_spec.ref,
        search_plan_ref=search_plan_ref,
        parent_refs=(search_plan_ref, candidate_spec.ref),
    )
    candidate_state = candidate_state_model.to_manifest(catalog)
    residual_history_model = ResidualHistoryManifest(
        residual_history_id=(
            f"{fit_result.candidate_id}__{fit_result.fit_window_id}"
            "__residual_history"
        ),
        candidate_id=fit_result.candidate_id,
        fit_window_id=fit_result.fit_window_id,
        residual_rows=tuple(record.as_dict() for record in fit_result.residual_history),
        residual_history_digest=(
            fit_result.residual_history_summary.residual_history_digest
        ),
        residual_basis=fit_result.residual_history_summary.residual_basis,
        construction_policy="legal_complete_training_origin_panel_v1",
        replay_identity=fit_result.residual_history_summary.replay_identity,
        source_row_set_digest=fit_result.residual_history_summary.source_row_set_digest,
    )
    residual_history = residual_history_model.to_manifest(catalog)
    reducer_body: dict[str, Any] = {
        "reducer_artifact_id": (
            f"{fit_result.candidate_id}__{fit_result.fit_window_id}__reducer"
        ),
        "owner_id": "module.candidate-fitting-v1",
        "scope_id": _SCOPE_ID,
        "candidate_id": fit_result.candidate_id,
        "family_id": fit_result.family_id,
        "fit_window_id": fit_result.fit_window_id,
        "fit_backend_id": fit_result.backend_id,
        "parameter_summary": _normalize_parameter_summary(fit_result.parameter_summary),
        "canonical_structure_signature": closed_fitted_candidate.canonical_hash(),
        "candidate_spec_ref": candidate_spec.ref.as_dict(),
        "candidate_state_ref": candidate_state.ref.as_dict(),
        "search_plan_ref": search_plan_ref.as_dict(),
        "optimizer_diagnostics": dict(fit_result.optimizer_diagnostics),
    }
    composition_graph = closed_fitted_candidate.structural_layer.composition_graph
    if composition_graph.operator_id is not None:
        reducer_body["composition_graph"] = composition_graph.as_dict()
    if description_gain_bits is not None:
        reducer_body["description_gain_bits"] = _stable_float(description_gain_bits)
    if exploratory_primary_score is not None:
        reducer_body["exploratory_primary_score"] = _stable_float(
            exploratory_primary_score
        )
    reducer_artifact = ManifestEnvelope.build(
        schema_name="reducer_artifact_manifest@1.0.0",
        module_id="candidate_fitting",
        body=reducer_body,
        catalog=catalog,
    )
    return CandidateFitArtifactBundle(
        candidate_spec=candidate_spec,
        candidate_state=candidate_state,
        reducer_artifact=reducer_artifact,
        residual_history=residual_history,
    )


def _capture_residual_history(
    *,
    candidate: CandidateIntermediateRepresentation,
    family_fit: _FamilyFitResult,
    feature_view: FeatureView,
    fit_window: EvaluationSegment,
    training_rows: Sequence[Mapping[str, Any]],
    candidate_id: str,
    fit_strategy: FitStrategySpec,
) -> _ResidualHistoryCapture:
    training_origin_panel = build_legal_training_origin_panel(
        feature_view=feature_view,
        evaluation_segment=fit_window,
        horizon_set=fit_strategy.horizon_set,
    )
    if training_origin_panel.status != "passed" or not training_origin_panel.records:
        raise ContractValidationError(
            code="incomplete_residual_history_panel",
            message="candidate fitting requires a complete training-origin panel",
            field_path="fit_window",
            details={
                "diagnostics": [
                    diagnostic.as_dict()
                    for diagnostic in training_origin_panel.diagnostics
                ],
            },
        )

    rows_by_entity = _feature_rows_by_entity(feature_view)
    training_positions = _training_row_positions(
        training_rows=training_rows,
        fit_window=fit_window,
    )
    records: list[ForecastResidualRecord] = []
    component_alignment: list[Mapping[str, Any]] = []
    for panel_row in training_origin_panel.records:
        target_row = rows_by_entity[panel_row.entity][panel_row.target_index]
        origin_row = rows_by_entity[panel_row.entity][panel_row.origin_index]
        row_position = training_positions[(panel_row.entity, panel_row.target_index)]
        trace = _forecast_training_panel_row(
            candidate=candidate,
            family_fit=family_fit,
            training_rows=training_rows,
            training_positions=training_positions,
            panel_row=panel_row,
            origin_row=origin_row,
            target_row=target_row,
            row_position=row_position,
        )
        if trace.component_alignment is not None:
            component_alignment.append(trace.component_alignment)
        realized = _stable_float(float(target_row["target"]))
        residual = _stable_float(realized - trace.point_forecast)
        records.append(
            ForecastResidualRecord(
                candidate_id=candidate_id,
                fit_window_id=fit_window.segment_id,
                entity=panel_row.entity,
                origin_index=panel_row.origin_index,
                origin_time=panel_row.origin_time,
                origin_available_at=panel_row.origin_available_at,
                target_index=panel_row.target_index,
                target_event_time=panel_row.target_event_time,
                target_available_at=panel_row.target_available_at,
                horizon=panel_row.horizon,
                point_forecast=_stable_float(trace.point_forecast),
                realized_observation=realized,
                residual=residual,
                split_role=fit_window.role,
                residual_basis="observation_minus_point_forecast",
                time_safety_status="passed",
                replay_identity=_residual_replay_identity(
                    candidate_id=candidate_id,
                    fit_window=fit_window,
                    panel_row=panel_row,
                    family_fit=family_fit,
                    replay_state=trace.replay_state,
                    fit_strategy=fit_strategy,
                ),
                weight=_residual_row_weight(
                    family_fit=family_fit,
                    row_position=row_position,
                ),
            )
        )

    residual_records = tuple(records)
    summary = summarize_residual_history(residual_records)
    validation = validate_residual_history(residual_records)
    diagnostics: dict[str, Any] = summary.as_dict()
    diagnostics["status"] = validation.status
    diagnostics["validation_reason_codes"] = list(validation.reason_codes)
    diagnostics["fit_strategy"] = fit_strategy.as_dict()
    diagnostics["fit_strategy_identity"] = fit_strategy.identity_hash
    diagnostics["training_scored_origin_set_id"] = training_origin_set_id(
        training_origin_panel
    )
    if component_alignment:
        diagnostics["component_alignment"] = [
            dict(item) for item in component_alignment
        ]
    return _ResidualHistoryCapture(
        records=residual_records,
        summary=summary,
        validation=validation,
        diagnostics=diagnostics,
    )


def _forecast_training_panel_row(
    *,
    candidate: CandidateIntermediateRepresentation,
    family_fit: _FamilyFitResult,
    training_rows: Sequence[Mapping[str, Any]],
    training_positions: Mapping[tuple[str, int], int],
    panel_row: TrainingOriginPanelRow,
    origin_row: Mapping[str, Any],
    target_row: Mapping[str, Any],
    row_position: int,
) -> _ResidualForecastTrace:
    if candidate.structural_layer.expression_payload is not None:
        return _forecast_expression_row(
            candidate=candidate,
            family_fit=family_fit,
            target_row=target_row,
        )

    operator_id = candidate.structural_layer.composition_graph.operator_id
    if operator_id == "additive_residual":
        return _forecast_additive_residual_row(
            candidate=candidate,
            family_fit=family_fit,
            training_rows=training_rows,
            panel_row=panel_row,
            target_row=target_row,
            row_position=row_position,
        )
    if operator_id == "shared_plus_local_decomposition":
        return _forecast_shared_local_row(
            family_fit=family_fit,
            target_row=target_row,
        )
    if operator_id == "piecewise":
        return _forecast_piecewise_row(
            candidate=candidate,
            family_fit=family_fit,
            target_row=target_row,
        )
    if operator_id == "regime_conditioned":
        return _forecast_regime_conditioned_row(
            candidate=candidate,
            family_fit=family_fit,
            target_row=target_row,
        )
    return _forecast_uncomposed_row(
        candidate=candidate,
        family_fit=family_fit,
        panel_row=panel_row,
        origin_row=origin_row,
        target_row=target_row,
        training_positions=training_positions,
        row_position=training_positions[(panel_row.entity, panel_row.target_index)],
    )


def _forecast_expression_row(
    *,
    candidate: CandidateIntermediateRepresentation,
    family_fit: _FamilyFitResult,
    target_row: Mapping[str, Any],
) -> _ResidualForecastTrace:
    payload = candidate.structural_layer.expression_payload
    if payload is None:
        raise ContractValidationError(
            code="missing_expression_payload",
            message="expression residual capture requires CIR expression_payload",
            field_path="candidate.structural_layer.expression_payload",
        )
    expression = expression_from_dict(payload.expression_tree)
    values: dict[str, float] = {
        key: float(value) for key, value in family_fit.parameter_summary.items()
    }
    for key, value in target_row.items():
        if key in {"event_time", "available_at", "entity"}:
            continue
        if isinstance(value, (int, float)):
            values[key] = float(value)
    point_forecast = float(evaluate_expression(expression, values))
    return _ResidualForecastTrace(
        point_forecast=_stable_float(point_forecast),
        replay_state={
            "family_id": "expression_ir",
            "parameters": _normalize_parameter_summary(family_fit.parameter_summary),
            "expression_hash": payload.expression_canonical_hash,
        },
    )


def _forecast_uncomposed_row(
    *,
    candidate: CandidateIntermediateRepresentation,
    family_fit: _FamilyFitResult,
    panel_row: TrainingOriginPanelRow,
    origin_row: Mapping[str, Any],
    target_row: Mapping[str, Any],
    training_positions: Mapping[tuple[str, int], int],
    row_position: int,
) -> _ResidualForecastTrace:
    family_id = candidate.structural_layer.cir_family_id
    if family_id == "analytic":
        replay_state: dict[str, Any] = {
            "family_id": "analytic",
            "horizon": panel_row.horizon,
            "parameters": _normalize_parameter_summary(
                family_fit.parameter_summary
            ),
        }
        feature_terms = _analytic_replay_feature_terms(
            parameters=family_fit.parameter_summary,
            origin_row=origin_row,
            target_row=target_row,
            horizon=panel_row.horizon,
        )
        if feature_terms:
            replay_state["feature_terms"] = feature_terms
        return _ResidualForecastTrace(
            point_forecast=_predict_analytic_panel_row(
                parameters=family_fit.parameter_summary,
                origin_row=origin_row,
                target_row=target_row,
                horizon=panel_row.horizon,
            ),
            replay_state=replay_state,
        )
    if family_id == "recursive":
        origin_position = training_positions[(panel_row.entity, panel_row.origin_index)]
        transition = family_fit.state_transitions[origin_position]
        state_before = dict(transition.state_after)
        if "level" in state_before:
            point_forecast = float(state_before["level"])
        elif "running_mean" in state_before:
            point_forecast = float(state_before["running_mean"])
        else:
            raise ContractValidationError(
                code="invalid_recursive_candidate_state",
                message="recursive residual capture requires a forecastable state",
                field_path="state_before",
            )
        return _ResidualForecastTrace(
            point_forecast=_stable_float(point_forecast),
            replay_state={"family_id": "recursive", **state_before},
        )
    if family_id == "spectral":
        transition = family_fit.state_transitions[row_position]
        state_before = dict(transition.state_before)
        literals = _literal_mapping(candidate)
        phase_index = int(state_before["phase_index"])
        return _ResidualForecastTrace(
            point_forecast=_spectral_prediction(
                parameters=family_fit.parameter_summary,
                literals=literals,
                phase_index=phase_index,
            ),
            replay_state={
                "family_id": "spectral",
                "harmonic_group": list(_spectral_harmonics(literals)),
                **state_before,
            },
        )
    if family_id == "algorithmic":
        transition = family_fit.state_transitions[row_position]
        state_before = dict(transition.state_before)
        literals = _literal_mapping(candidate)
        max_lag = int(candidate.execution_layer.history_access_contract.max_lag or 0)
        program = parse_algorithmic_program(
            str(literals["algorithmic_program"]),
            state_slot_count=int(literals.get("algorithmic_state_slot_count", 1)),
            max_program_nodes=int(literals.get("program_node_count", 8)),
            allowed_observation_lags=_algorithmic_allowed_observation_lags(
                literals=literals,
                max_lag=max_lag,
            ),
        )
        state = tuple(
            Fraction(str(state_before[f"state_{index}"]))
            for index in range(program.state_slot_count)
        )
        step = evaluate_algorithmic_program(
            program,
            state=state,
            observation=_algorithmic_observation_window(
                row=target_row,
                max_lag=max_lag,
                current_value=float(target_row["target"]),
            ),
        )
        return _ResidualForecastTrace(
            point_forecast=_stable_float(float(step.emit_value)),
            replay_state={"family_id": "algorithmic", **state_before},
        )
    raise ContractValidationError(
        code="unsupported_candidate_family",
        message=f"{family_id!r} residual capture is not supported",
        field_path="candidate.structural_layer.cir_family_id",
    )


def _forecast_additive_residual_row(
    *,
    candidate: CandidateIntermediateRepresentation,
    family_fit: _FamilyFitResult,
    training_rows: Sequence[Mapping[str, Any]],
    panel_row: TrainingOriginPanelRow,
    target_row: Mapping[str, Any],
    row_position: int,
) -> _ResidualForecastTrace:
    composition = candidate.structural_layer.composition_graph.composition
    assert isinstance(composition, AdditiveResidualComposition)
    base_parameters = extract_component_mapping(
        family_fit.parameter_summary,
        composition.base_reducer,
    )
    residual_parameters = extract_component_mapping(
        family_fit.parameter_summary,
        composition.residual_reducer,
    )
    base_training_predictions = _analytic_fitted_values_from_rows(
        parameters=base_parameters,
        rows=training_rows,
    )
    residual_targets = tuple(
        _stable_float(float(row["target"]) - base_prediction)
        for row, base_prediction in zip(
            training_rows,
            base_training_predictions,
            strict=True,
        )
    )
    residual_row = dict(target_row)
    residual_row["target"] = residual_targets[row_position]
    residual_row["lag_1"] = (
        0.0 if row_position == 0 else residual_targets[row_position - 1]
    )
    base_prediction = _predict_analytic_row(
        parameters=base_parameters,
        row=target_row,
    )
    residual_prediction = _predict_analytic_row(
        parameters=residual_parameters,
        row=residual_row,
    )
    point_forecast = _stable_float(base_prediction + residual_prediction)
    alignment = {
        "operator_id": "additive_residual",
        "origin_index": panel_row.origin_index,
        "target_index": panel_row.target_index,
        "horizon": panel_row.horizon,
        "base_component_id": composition.base_reducer,
        "residual_component_id": composition.residual_reducer,
        "base_prediction": _stable_float(base_prediction),
        "residual_prediction": _stable_float(residual_prediction),
        "point_forecast": point_forecast,
    }
    return _ResidualForecastTrace(
        point_forecast=point_forecast,
        replay_state={
            "family_id": "analytic",
            "operator_id": "additive_residual",
            "parameters": _normalize_parameter_summary(family_fit.parameter_summary),
            "residual_lag_1": residual_row["lag_1"],
        },
        component_alignment=alignment,
    )


def _forecast_shared_local_row(
    *,
    family_fit: _FamilyFitResult,
    target_row: Mapping[str, Any],
) -> _ResidualForecastTrace:
    entity = str(target_row["entity"])
    parameters = family_fit.parameter_summary
    lag_value = float(target_row.get("lag_1", 0.0))
    point_forecast = float(parameters.get("shared_intercept", 0.0))
    point_forecast += float(parameters.get("shared_lag_coefficient", 0.0)) * lag_value
    point_forecast += float(parameters.get(f"local_adjustment__{entity}", 0.0))
    point_forecast += (
        float(parameters.get(f"local_lag_adjustment__{entity}", 0.0)) * lag_value
    )
    return _ResidualForecastTrace(
        point_forecast=_stable_float(point_forecast),
        replay_state={
            "family_id": "analytic",
            "operator_id": "shared_plus_local_decomposition",
            "entity": entity,
            "parameters": _normalize_parameter_summary(parameters),
        },
    )


def _forecast_piecewise_row(
    *,
    candidate: CandidateIntermediateRepresentation,
    family_fit: _FamilyFitResult,
    target_row: Mapping[str, Any],
) -> _ResidualForecastTrace:
    segment, evidence = select_piecewise_segment(
        candidate.structural_layer.composition_graph,
        row=target_row,
    )
    parameters = extract_component_mapping(
        family_fit.parameter_summary,
        segment.reducer_id,
    )
    return _ResidualForecastTrace(
        point_forecast=_predict_analytic_row(parameters=parameters, row=target_row),
        replay_state={
            "family_id": "analytic",
            "operator_id": "piecewise",
            "selected_branch_id": segment.reducer_id,
            "evidence": dict(evidence),
            "parameters": _normalize_parameter_summary(parameters),
        },
    )


def _forecast_regime_conditioned_row(
    *,
    candidate: CandidateIntermediateRepresentation,
    family_fit: _FamilyFitResult,
    target_row: Mapping[str, Any],
) -> _ResidualForecastTrace:
    branch_weights, evidence = resolve_regime_weights(
        candidate.structural_layer.composition_graph,
        row=target_row,
    )
    branch_predictions: dict[str, float] = {}
    point_forecast = 0.0
    for branch_id, weight in branch_weights.items():
        parameters = extract_component_mapping(family_fit.parameter_summary, branch_id)
        branch_prediction = _predict_analytic_row(
            parameters=parameters,
            row=target_row,
        )
        branch_predictions[branch_id] = _stable_float(branch_prediction)
        point_forecast += float(weight) * branch_prediction
    return _ResidualForecastTrace(
        point_forecast=_stable_float(point_forecast),
        replay_state={
            "family_id": "analytic",
            "operator_id": "regime_conditioned",
            "branch_weights": dict(branch_weights),
            "branch_predictions": branch_predictions,
            "evidence": dict(evidence),
        },
    )


def _predict_analytic_row(
    *,
    parameters: Mapping[str, float | int],
    row: Mapping[str, Any],
) -> float:
    intercept = float(parameters.get("intercept", 0.0))
    feature_terms = _analytic_feature_terms_from_parameters(parameters)
    if not feature_terms:
        return _stable_float(intercept)
    return _stable_float(
        intercept
        + sum(
            float(parameters[parameter_name])
            * _analytic_feature_value(row=row, feature_name=feature_name)
            for feature_name, parameter_name in feature_terms
        )
    )


def _predict_analytic_panel_row(
    *,
    parameters: Mapping[str, float | int],
    origin_row: Mapping[str, Any],
    target_row: Mapping[str, Any] | None = None,
    horizon: int,
) -> float:
    return _stable_float(
        _predict_analytic_from_origin(
            parameters=parameters,
            origin_row=origin_row,
            target_row=target_row,
            horizon=horizon,
        )
    )


def _predict_analytic_from_origin(
    *,
    parameters: Mapping[str, float | int],
    origin_row: Mapping[str, Any],
    target_row: Mapping[str, Any] | None = None,
    horizon: int,
) -> float:
    if f"horizon_{horizon}__intercept" in parameters:
        intercept = float(parameters.get(f"horizon_{horizon}__intercept", 0.0))
        slope_key = f"horizon_{horizon}__lag_coefficient"
        if slope_key not in parameters:
            return _stable_float(intercept)
        return _stable_float(intercept + (float(parameters[slope_key]) * float(origin_row["target"])))
    if "rectify_base__intercept" in parameters:
        base_parameters = {
            key.removeprefix("rectify_base__"): value
            for key, value in parameters.items()
            if key.startswith("rectify_base__")
        }
        base_prediction = _predict_recursive_analytic_from_origin(
            parameters=base_parameters,
            origin_row=origin_row,
            horizon=horizon,
        )
        correction = float(parameters.get(f"rectify_horizon_{horizon}__intercept", 0.0))
        return _stable_float(base_prediction + correction)
    feature_terms = _analytic_feature_terms_from_parameters(parameters)
    if feature_terms and tuple(feature_terms) != (("lag_1", "lag_coefficient"),):
        if horizon != 1 or target_row is None:
            raise ContractValidationError(
                code="unsupported_candidate_family",
                message=(
                    "selected-feature analytic candidates are admitted for "
                    "one-step rows with materialized target features only"
                ),
                field_path="candidate.structural_layer.parameter_block",
            )
        return _predict_analytic_row(parameters=parameters, row=target_row)
    return _predict_recursive_analytic_from_origin(
        parameters=parameters,
        origin_row=origin_row,
        horizon=horizon,
    )


def _analytic_replay_feature_terms(
    *,
    parameters: Mapping[str, float | int],
    origin_row: Mapping[str, Any],
    target_row: Mapping[str, Any],
    horizon: int,
) -> dict[str, float]:
    terms = _analytic_feature_terms_from_parameters(parameters)
    if not terms or tuple(terms) == (("lag_1", "lag_coefficient"),):
        return {}
    row = target_row if horizon == 1 else origin_row
    return {
        feature_name: _stable_float(_analytic_feature_value(row=row, feature_name=feature_name))
        for feature_name, _ in terms
        if feature_name in row
    }


def _predict_recursive_analytic_from_origin(
    *,
    parameters: Mapping[str, float | int],
    origin_row: Mapping[str, Any],
    horizon: int,
) -> float:
    intercept = float(parameters.get("intercept", 0.0))
    if "lag_coefficient" not in parameters:
        return _stable_float(intercept)
    previous_value = float(origin_row["target"])
    forecast = intercept
    for _ in range(1, horizon + 1):
        forecast = intercept + (float(parameters["lag_coefficient"]) * previous_value)
        previous_value = forecast
    return _stable_float(forecast)


def _feature_rows_by_entity(
    feature_view: FeatureView,
) -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in feature_view.rows:
        entity = str(row.get("entity", feature_view.series_id))
        grouped.setdefault(entity, []).append(dict(row))
    return {entity: tuple(rows) for entity, rows in grouped.items()}


def _training_row_positions(
    *,
    training_rows: Sequence[Mapping[str, Any]],
    fit_window: EvaluationSegment,
) -> dict[tuple[str, int], int]:
    entity_offsets: dict[str, int] = {}
    positions: dict[tuple[str, int], int] = {}
    for position, row in enumerate(training_rows):
        entity = str(row.get("entity", ""))
        local_index = fit_window.train_start_index + entity_offsets.get(entity, 0)
        positions[(entity, local_index)] = position
        entity_offsets[entity] = entity_offsets.get(entity, 0) + 1
    return positions


def _residual_row_weight(
    *,
    family_fit: _FamilyFitResult,
    row_position: int,
) -> float:
    if len(family_fit.residual_row_weights) == 0:
        return 1.0
    if row_position >= len(family_fit.residual_row_weights):
        return 1.0
    return _stable_float(float(family_fit.residual_row_weights[row_position]))


def _residual_replay_identity(
    *,
    candidate_id: str,
    fit_window: EvaluationSegment,
    panel_row: TrainingOriginPanelRow,
    family_fit: _FamilyFitResult,
    replay_state: Mapping[str, Any],
    fit_strategy: FitStrategySpec,
) -> str:
    replay_payload = {
        "backend_id": family_fit.backend_id,
        "candidate_id": candidate_id,
        "entity": panel_row.entity,
        "entity_aggregation_mode": fit_strategy.entity_aggregation_mode,
        "fit_strategy_identity": fit_strategy.identity_hash,
        "fit_strategy_id": fit_strategy.strategy_id,
        "fit_window_id": fit_window.segment_id,
        "horizon": panel_row.horizon,
        "horizon_weights": [dict(item) for item in fit_strategy.horizon_weights],
        "origin_index": panel_row.origin_index,
        "point_loss_id": fit_strategy.point_loss_id,
        "replay_state": dict(replay_state),
        "target_index": panel_row.target_index,
    }
    return (
        f"{candidate_id}:{fit_window.segment_id}:{panel_row.entity}:"
        f"o{panel_row.origin_index}:h{panel_row.horizon}:"
        f"{fit_strategy.identity_hash}:"
        f"{canonicalize_json(dict(replay_state))}:"
        f"{canonicalize_json(replay_payload)}"
    )


def _fit_family(
    *,
    candidate: CandidateIntermediateRepresentation,
    feature_view: FeatureView,
    fit_window: EvaluationSegment,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
    fit_strategy: FitStrategySpec,
    shared_local_backend_preference: str | None = None,
) -> _FamilyFitResult:
    if not _is_legacy_fit_strategy(fit_strategy):
        return _fit_multi_horizon_strategy(
            candidate=candidate,
            feature_view=feature_view,
            fit_window=fit_window,
            training_rows=training_rows,
            random_seed=random_seed,
            fit_strategy=fit_strategy,
        )
    composition_graph = candidate.structural_layer.composition_graph
    if candidate.structural_layer.expression_payload is not None:
        return _fit_expression_payload(
            candidate=candidate,
            training_rows=training_rows,
            random_seed=random_seed,
        )
    operator_id = composition_graph.operator_id
    if operator_id == "shared_plus_local_decomposition":
        return _fit_shared_local(
            candidate=candidate,
            training_rows=training_rows,
            random_seed=random_seed,
            backend_preference=shared_local_backend_preference,
        )
    if operator_id == "piecewise":
        return _fit_piecewise(
            candidate=candidate,
            training_rows=training_rows,
            random_seed=random_seed,
        )
    if operator_id == "additive_residual":
        return _fit_additive_residual(
            candidate=candidate,
            training_rows=training_rows,
            random_seed=random_seed,
        )
    if operator_id == "regime_conditioned":
        return _fit_regime_conditioned(
            candidate=candidate,
            training_rows=training_rows,
            random_seed=random_seed,
        )
    return _fit_uncomposed_family(
        candidate=candidate,
        training_rows=training_rows,
        random_seed=random_seed,
    )


def _is_legacy_fit_strategy(fit_strategy: FitStrategySpec) -> bool:
    return fit_strategy.strategy_id in _LEGACY_FIT_STRATEGY_IDS


def _fit_multi_horizon_strategy(
    *,
    candidate: CandidateIntermediateRepresentation,
    feature_view: FeatureView,
    fit_window: EvaluationSegment,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
    fit_strategy: FitStrategySpec,
) -> _FamilyFitResult:
    family_id = candidate.structural_layer.cir_family_id
    operator_id = candidate.structural_layer.composition_graph.operator_id
    if operator_id is not None or candidate.structural_layer.expression_payload is not None:
        _raise_incompatible_fit_strategy(
            candidate=candidate,
            fit_strategy=fit_strategy,
            reason_code="composition_or_expression_strategy_not_supported",
        )
    if family_id == "analytic" and not _analytic_is_recursive_rollout_compatible(
        candidate
    ):
        _raise_incompatible_fit_strategy(
            candidate=candidate,
            fit_strategy=fit_strategy,
            reason_code="selected_feature_analytic_rollout_not_supported",
        )
    if fit_strategy.strategy_id in _RECURSIVE_ROLLOUT_STRATEGY_IDS:
        if family_id == "analytic":
            return _fit_joint_rollout_analytic(
                candidate=candidate,
                feature_view=feature_view,
                fit_window=fit_window,
                training_rows=training_rows,
                random_seed=random_seed,
                fit_strategy=fit_strategy,
                objective_id="least_squares_recursive_rollout_v1",
            )
        if family_id == "recursive":
            return _fit_recursive_rollout(
                candidate=candidate,
                feature_view=feature_view,
                fit_window=fit_window,
                training_rows=training_rows,
                fit_strategy=fit_strategy,
            )
        _raise_incompatible_fit_strategy(
            candidate=candidate,
            fit_strategy=fit_strategy,
            reason_code="recursive_rollout_requires_analytic_or_recursive_candidate",
        )
    if fit_strategy.strategy_id in _DIRECT_ANALYTIC_STRATEGY_IDS:
        if family_id == "analytic":
            return _fit_direct_analytic(
                candidate=candidate,
                feature_view=feature_view,
                fit_window=fit_window,
                training_rows=training_rows,
                random_seed=random_seed,
                fit_strategy=fit_strategy,
            )
        _raise_incompatible_fit_strategy(
            candidate=candidate,
            fit_strategy=fit_strategy,
            reason_code="direct_strategy_requires_analytic_candidate",
        )
    if fit_strategy.strategy_id in _JOINT_ANALYTIC_STRATEGY_IDS:
        if family_id == "analytic":
            return _fit_joint_rollout_analytic(
                candidate=candidate,
                feature_view=feature_view,
                fit_window=fit_window,
                training_rows=training_rows,
                random_seed=random_seed,
                fit_strategy=fit_strategy,
                objective_id="least_squares_joint_rollout_v1",
            )
        _raise_incompatible_fit_strategy(
            candidate=candidate,
            fit_strategy=fit_strategy,
            reason_code="joint_strategy_requires_analytic_candidate",
        )
    if fit_strategy.strategy_id in _RECTIFY_ANALYTIC_STRATEGY_IDS:
        if family_id == "analytic":
            return _fit_rectify_analytic(
                candidate=candidate,
                feature_view=feature_view,
                fit_window=fit_window,
                training_rows=training_rows,
                random_seed=random_seed,
                fit_strategy=fit_strategy,
            )
        _raise_incompatible_fit_strategy(
            candidate=candidate,
            fit_strategy=fit_strategy,
            reason_code="rectify_strategy_requires_analytic_candidate",
        )
    _raise_incompatible_fit_strategy(
        candidate=candidate,
        fit_strategy=fit_strategy,
        reason_code="unknown_fit_strategy",
    )


def _raise_incompatible_fit_strategy(
    *,
    candidate: CandidateIntermediateRepresentation,
    fit_strategy: FitStrategySpec,
    reason_code: str,
) -> None:
    raise ContractValidationError(
        code="incompatible_fit_strategy",
        message="fit strategy is not compatible with this candidate",
        field_path="fit_strategy.strategy_id",
        details={
            "candidate_family_id": candidate.structural_layer.cir_family_id,
            "fit_strategy_id": fit_strategy.strategy_id,
            "operator_id": candidate.structural_layer.composition_graph.operator_id,
            "reason_code": reason_code,
        },
    )


def _fit_expression_payload(
    *,
    candidate: CandidateIntermediateRepresentation,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
) -> _FamilyFitResult:
    payload = candidate.structural_layer.expression_payload
    if payload is None:
        raise ContractValidationError(
            code="missing_expression_payload",
            message="expression fitting requires CIR expression_payload",
            field_path="candidate.structural_layer.expression_payload",
        )
    declared_parameters = _parameter_mapping(candidate)
    parameter_declarations = tuple(
        ParameterDeclaration(
            name=name,
            initial_value=float(declared_parameters.get(name, 0.0)),
        )
        for name in payload.parameter_declarations
    )
    fit_result = fit_cir_candidate(
        candidate=candidate,
        data=FitDataSplit(train_rows=tuple(training_rows)),
        fit_window_id="candidate_fit_window",
        parameter_declarations=parameter_declarations,
        objective_id="squared_error",
        seed=int(random_seed),
    )
    initial_state = _state_mapping(candidate)
    return _FamilyFitResult(
        backend_id="euclid_unified_fit_layer_v1",
        objective_id=fit_result.objective_id,
        parameter_summary=dict(fit_result.parameter_estimates),
        updated_literals={},
        initial_state=initial_state,
        final_state=initial_state,
        state_transitions=_identity_transitions(
            initial_state=initial_state,
            training_rows=training_rows,
        ),
        converged=fit_result.status == "converged",
        iteration_count=int(
            fit_result.optimizer_diagnostics.get("function_evaluations", 0)
        ),
        final_loss=_stable_float(fit_result.loss),
        component_diagnostics={
            "fit_layer": "src/euclid/fit",
            "claim_boundary": dict(fit_result.claim_boundary),
            "replay_identity": fit_result.replay_identity,
            "failure_reasons": list(fit_result.failure_reasons),
            "unified_optimizer_diagnostics": dict(fit_result.optimizer_diagnostics),
        },
    )


def _fit_uncomposed_family(
    *,
    candidate: CandidateIntermediateRepresentation,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
) -> _FamilyFitResult:
    family_id = candidate.structural_layer.cir_family_id
    if family_id == "analytic":
        return _fit_analytic(
            candidate=candidate,
            training_rows=training_rows,
            random_seed=random_seed,
        )
    if family_id == "recursive":
        return _fit_recursive(
            candidate=candidate,
            training_rows=training_rows,
            random_seed=random_seed,
        )
    if family_id == "spectral":
        return _fit_spectral(
            candidate=candidate,
            training_rows=training_rows,
            random_seed=random_seed,
        )
    if family_id == "algorithmic":
        return _fit_algorithmic(
            candidate=candidate,
            training_rows=training_rows,
            random_seed=random_seed,
        )
    raise ContractValidationError(
        code="unsupported_candidate_family",
        message=(
            "candidate fitting supports analytic, recursive, spectral, and "
            "algorithmic CIR families only"
        ),
        field_path="candidate.structural_layer.cir_family_id",
        details={"family_id": family_id},
    )


def _fit_piecewise(
    *,
    candidate: CandidateIntermediateRepresentation,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
) -> _FamilyFitResult:
    _require_analytic_composition_family(candidate, operator_id="piecewise")
    composition = candidate.structural_layer.composition_graph.composition
    assert isinstance(composition, PiecewiseComposition)

    rows_by_branch = {
        segment.reducer_id: [] for segment in composition.ordered_partition
    }
    branch_trace: list[dict[str, Any]] = []
    for index, row in enumerate(training_rows):
        segment, evidence = select_piecewise_segment(
            candidate.structural_layer.composition_graph,
            row=row,
        )
        rows_by_branch[segment.reducer_id].append(row)
        branch_trace.append(
            {
                "row_index": index,
                "event_time": str(row["event_time"]),
                "selected_branch_id": segment.reducer_id,
                "partition_value": evidence["partition_value"],
            }
        )

    merged_parameters: dict[str, Any] = {}
    merged_literals: dict[str, Any] = {}
    merged_state: dict[str, Any] = {}
    final_loss = 0.0
    branch_summaries: list[dict[str, Any]] = []
    for segment in composition.ordered_partition:
        branch_candidate = _component_candidate(
            candidate,
            component_id=segment.reducer_id,
        )
        branch_rows = tuple(rows_by_branch[segment.reducer_id])
        branch_fit = (
            _fit_uncomposed_family(
                candidate=branch_candidate,
                training_rows=branch_rows,
                random_seed=random_seed,
            )
            if branch_rows
            else _preserve_declared_component(branch_candidate)
        )
        merge_component_mapping(
            merged_parameters,
            component_id=segment.reducer_id,
            mapping=branch_fit.parameter_summary,
        )
        merge_component_mapping(
            merged_literals,
            component_id=segment.reducer_id,
            mapping=branch_fit.updated_literals,
        )
        merge_component_mapping(
            merged_state,
            component_id=segment.reducer_id,
            mapping=branch_fit.final_state,
        )
        final_loss += float(branch_fit.final_loss)
        branch_summaries.append(
            {
                "branch_id": segment.reducer_id,
                "row_count": len(branch_rows),
                "final_loss": _stable_float(branch_fit.final_loss),
                "parameter_summary": _normalize_parameter_summary(
                    branch_fit.parameter_summary
                ),
            }
        )
    initial_state = _state_mapping(candidate)
    return _FamilyFitResult(
        backend_id="piecewise_branch_partition_v1",
        objective_id="least_squares_piecewise_branch_partition_v1",
        parameter_summary=merged_parameters,
        updated_literals=merged_literals,
        initial_state=initial_state,
        final_state=merged_state,
        state_transitions=_identity_transitions(
            initial_state=initial_state,
            training_rows=training_rows,
        ),
        converged=True,
        iteration_count=sum(summary["row_count"] for summary in branch_summaries),
        final_loss=_stable_float(final_loss),
        component_diagnostics={
            "operator_id": "piecewise",
            "branch_summaries": branch_summaries,
            "branch_trace": branch_trace,
        },
    )


def _fit_additive_residual(
    *,
    candidate: CandidateIntermediateRepresentation,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
) -> _FamilyFitResult:
    _require_analytic_composition_family(candidate, operator_id="additive_residual")
    composition = candidate.structural_layer.composition_graph.composition
    assert isinstance(composition, AdditiveResidualComposition)

    base_candidate = _component_candidate(
        candidate,
        component_id=composition.base_reducer,
    )
    base_fit = _fit_uncomposed_family(
        candidate=base_candidate,
        training_rows=training_rows,
        random_seed=random_seed,
    )
    base_fitted_values = _analytic_fitted_values_from_rows(
        parameters=base_fit.parameter_summary,
        rows=training_rows,
    )
    residual_targets = tuple(
        _stable_float(float(row["target"]) - fitted)
        for row, fitted in zip(training_rows, base_fitted_values, strict=True)
    )
    residual_rows = _rows_with_targets(
        training_rows,
        targets=residual_targets,
        lag_values=(0.0, *residual_targets[:-1]),
    )
    residual_candidate = _component_candidate(
        candidate,
        component_id=composition.residual_reducer,
    )
    residual_fit = _fit_uncomposed_family(
        candidate=residual_candidate,
        training_rows=residual_rows,
        random_seed=random_seed,
    )

    merged_parameters: dict[str, Any] = {}
    merged_literals: dict[str, Any] = {}
    merged_state: dict[str, Any] = {}
    merge_component_mapping(
        merged_parameters,
        component_id=composition.base_reducer,
        mapping=base_fit.parameter_summary,
    )
    merge_component_mapping(
        merged_parameters,
        component_id=composition.residual_reducer,
        mapping=residual_fit.parameter_summary,
    )
    merge_component_mapping(
        merged_literals,
        component_id=composition.base_reducer,
        mapping=base_fit.updated_literals,
    )
    merge_component_mapping(
        merged_literals,
        component_id=composition.residual_reducer,
        mapping=residual_fit.updated_literals,
    )
    merge_component_mapping(
        merged_state,
        component_id=composition.base_reducer,
        mapping=base_fit.final_state,
    )
    merge_component_mapping(
        merged_state,
        component_id=composition.residual_reducer,
        mapping=residual_fit.final_state,
    )
    initial_state = _state_mapping(candidate)
    return _FamilyFitResult(
        backend_id="additive_residual_composition_v1",
        objective_id="least_squares_additive_residual_v1",
        parameter_summary=merged_parameters,
        updated_literals=merged_literals,
        initial_state=initial_state,
        final_state=merged_state,
        state_transitions=_identity_transitions(
            initial_state=initial_state,
            training_rows=training_rows,
        ),
        converged=True,
        iteration_count=2,
        final_loss=_stable_float(base_fit.final_loss + residual_fit.final_loss),
        component_diagnostics={
            "operator_id": "additive_residual",
            "base_component": {
                "component_id": composition.base_reducer,
                "parameter_summary": _normalize_parameter_summary(
                    base_fit.parameter_summary
                ),
                "final_loss": _stable_float(base_fit.final_loss),
            },
            "residual_component": {
                "component_id": composition.residual_reducer,
                "parameter_summary": _normalize_parameter_summary(
                    residual_fit.parameter_summary
                ),
                "final_loss": _stable_float(residual_fit.final_loss),
            },
        },
    )


def _fit_regime_conditioned(
    *,
    candidate: CandidateIntermediateRepresentation,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
) -> _FamilyFitResult:
    _require_analytic_composition_family(candidate, operator_id="regime_conditioned")
    composition = candidate.structural_layer.composition_graph.composition
    assert isinstance(composition, RegimeConditionedComposition)

    merged_parameters: dict[str, Any] = {}
    merged_literals: dict[str, Any] = {}
    merged_state: dict[str, Any] = {}
    branch_trace: list[dict[str, Any]] = []
    branch_summaries: list[dict[str, Any]] = []
    final_loss = 0.0

    if composition.gating_law.selection_mode == "hard_switch":
        rows_by_branch = {
            branch.reducer_id: [] for branch in composition.branch_reducers
        }
        for index, row in enumerate(training_rows):
            branch_weights, evidence = resolve_regime_weights(
                candidate.structural_layer.composition_graph,
                row=row,
            )
            branch_id = next(iter(branch_weights))
            rows_by_branch[branch_id].append(row)
            branch_trace.append(
                {
                    "row_index": index,
                    "event_time": str(row["event_time"]),
                    "selected_branch_id": branch_id,
                    "selection_mode": "hard_switch",
                }
            )
        for branch in composition.branch_reducers:
            branch_candidate = _component_candidate(
                candidate,
                component_id=branch.reducer_id,
            )
            branch_rows = tuple(rows_by_branch[branch.reducer_id])
            branch_fit = (
                _fit_uncomposed_family(
                    candidate=branch_candidate,
                    training_rows=branch_rows,
                    random_seed=random_seed,
                )
                if branch_rows
                else _preserve_declared_component(branch_candidate)
            )
            merge_component_mapping(
                merged_parameters,
                component_id=branch.reducer_id,
                mapping=branch_fit.parameter_summary,
            )
            merge_component_mapping(
                merged_literals,
                component_id=branch.reducer_id,
                mapping=branch_fit.updated_literals,
            )
            merge_component_mapping(
                merged_state,
                component_id=branch.reducer_id,
                mapping=branch_fit.final_state,
            )
            final_loss += float(branch_fit.final_loss)
            branch_summaries.append(
                {
                    "branch_id": branch.reducer_id,
                    "row_count": len(branch_rows),
                    "final_loss": _stable_float(branch_fit.final_loss),
                    "parameter_summary": _normalize_parameter_summary(
                        branch_fit.parameter_summary
                    ),
                }
            )
    else:
        branch_weights_by_id = {
            branch.reducer_id: [] for branch in composition.branch_reducers
        }
        for index, row in enumerate(training_rows):
            branch_weights, evidence = resolve_regime_weights(
                candidate.structural_layer.composition_graph,
                row=row,
            )
            for branch_id in branch_weights_by_id:
                branch_weights_by_id[branch_id].append(
                    float(branch_weights.get(branch_id, 0.0))
                )
            branch_trace.append(
                {
                    "row_index": index,
                    "event_time": str(row["event_time"]),
                    "selection_mode": "convex_weighting",
                    "branch_weights": dict(branch_weights),
                }
            )
        for branch in composition.branch_reducers:
            branch_candidate = _component_candidate(
                candidate,
                component_id=branch.reducer_id,
            )
            branch_fit = _fit_weighted_analytic(
                candidate=branch_candidate,
                training_rows=training_rows,
                weights=tuple(branch_weights_by_id[branch.reducer_id]),
            )
            merge_component_mapping(
                merged_parameters,
                component_id=branch.reducer_id,
                mapping=branch_fit.parameter_summary,
            )
            merge_component_mapping(
                merged_literals,
                component_id=branch.reducer_id,
                mapping=branch_fit.updated_literals,
            )
            merge_component_mapping(
                merged_state,
                component_id=branch.reducer_id,
                mapping=branch_fit.final_state,
            )
            final_loss += float(branch_fit.final_loss)
            branch_summaries.append(
                {
                    "branch_id": branch.reducer_id,
                    "row_count": len(training_rows),
                    "weighted_final_loss": _stable_float(branch_fit.final_loss),
                    "parameter_summary": _normalize_parameter_summary(
                        branch_fit.parameter_summary
                    ),
                }
            )
    initial_state = _state_mapping(candidate)
    return _FamilyFitResult(
        backend_id="regime_conditioned_composition_v1",
        objective_id="least_squares_regime_conditioned_v1",
        parameter_summary=merged_parameters,
        updated_literals=merged_literals,
        initial_state=initial_state,
        final_state=merged_state,
        state_transitions=_identity_transitions(
            initial_state=initial_state,
            training_rows=training_rows,
        ),
        converged=True,
        iteration_count=max(1, len(branch_summaries)),
        final_loss=_stable_float(final_loss),
        component_diagnostics={
            "operator_id": "regime_conditioned",
            "selection_mode": composition.gating_law.selection_mode,
            "branch_summaries": branch_summaries,
            "branch_trace": branch_trace,
        },
    )


def _fit_direct_analytic(
    *,
    candidate: CandidateIntermediateRepresentation,
    feature_view: FeatureView,
    fit_window: EvaluationSegment,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
    fit_strategy: FitStrategySpec,
) -> _FamilyFitResult:
    del random_seed
    panel = _complete_strategy_training_panel(
        feature_view=feature_view,
        fit_window=fit_window,
        fit_strategy=fit_strategy,
    )
    rows_by_entity = _feature_rows_by_entity(feature_view)
    parameter_summary: dict[str, float | int] = {}
    horizon_summaries: list[dict[str, Any]] = []
    for horizon in fit_strategy.horizon_set:
        horizon_records = tuple(
            record for record in panel.records if record.horizon == horizon
        )
        origin_rows = tuple(
            rows_by_entity[record.entity][record.origin_index]
            for record in horizon_records
        )
        targets = tuple(
            float(rows_by_entity[record.entity][record.target_index]["target"])
            for record in horizon_records
        )
        direct_rows = _rows_with_targets(
            origin_rows,
            targets=targets,
            lag_values=tuple(float(row["target"]) for row in origin_rows),
        )
        horizon_fit = _fit_weighted_analytic(
            candidate=candidate,
            training_rows=direct_rows,
            weights=tuple(1.0 for _ in direct_rows),
        )
        for name, value in horizon_fit.parameter_summary.items():
            parameter_summary[f"horizon_{horizon}__{name}"] = value
        horizon_summaries.append(
            {
                "horizon": horizon,
                "row_count": len(direct_rows),
                "parameter_summary": _normalize_parameter_summary(
                    horizon_fit.parameter_summary
                ),
            }
        )
    initial_state = _state_mapping(candidate)
    state_transitions = _identity_transitions(
        initial_state=initial_state,
        training_rows=training_rows,
    )
    family_fit = _FamilyFitResult(
        backend_id="deterministic_direct_analytic_v1",
        objective_id="least_squares_direct_analytic_v1",
        parameter_summary=parameter_summary,
        updated_literals={},
        initial_state=initial_state,
        final_state=(
            initial_state if not state_transitions else state_transitions[-1].state_after
        ),
        state_transitions=state_transitions,
        converged=True,
        iteration_count=len(fit_strategy.horizon_set),
        final_loss=0.0,
        component_diagnostics={
            "direct": {"horizon_summaries": horizon_summaries},
        },
    )
    return _with_rollout_objective_diagnostics(
        candidate=candidate,
        family_fit=family_fit,
        feature_view=feature_view,
        fit_window=fit_window,
        training_rows=training_rows,
        fit_strategy=fit_strategy,
        training_origin_panel=panel,
    )


def _fit_joint_rollout_analytic(
    *,
    candidate: CandidateIntermediateRepresentation,
    feature_view: FeatureView,
    fit_window: EvaluationSegment,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
    fit_strategy: FitStrategySpec,
    objective_id: str,
) -> _FamilyFitResult:
    del random_seed
    panel = _complete_strategy_training_panel(
        feature_view=feature_view,
        fit_window=fit_window,
        fit_strategy=fit_strategy,
    )
    initial_fit = _fit_analytic(
        candidate=candidate,
        training_rows=training_rows,
        random_seed="0",
    )
    if "lag_coefficient" not in _parameter_mapping(candidate):
        parameter_summary = _fit_joint_intercept_only_parameters(
            feature_view=feature_view,
            training_origin_panel=panel,
            fit_strategy=fit_strategy,
        )
    else:
        parameter_summary = _optimize_joint_affine_parameters(
            candidate=candidate,
            feature_view=feature_view,
            training_origin_panel=panel,
            fit_strategy=fit_strategy,
            initial_parameters=initial_fit.parameter_summary,
        )
    initial_state = _state_mapping(candidate)
    state_transitions = _identity_transitions(
        initial_state=initial_state,
        training_rows=training_rows,
    )
    family_fit = _FamilyFitResult(
        backend_id="deterministic_rollout_coordinate_search_v1",
        objective_id=objective_id,
        parameter_summary=parameter_summary,
        updated_literals={},
        initial_state=initial_state,
        final_state=(
            initial_state if not state_transitions else state_transitions[-1].state_after
        ),
        state_transitions=state_transitions,
        converged=True,
        iteration_count=1,
        final_loss=0.0,
        component_diagnostics={},
    )
    return _with_rollout_objective_diagnostics(
        candidate=candidate,
        family_fit=family_fit,
        feature_view=feature_view,
        fit_window=fit_window,
        training_rows=training_rows,
        fit_strategy=fit_strategy,
        training_origin_panel=panel,
    )


def _fit_rectify_analytic(
    *,
    candidate: CandidateIntermediateRepresentation,
    feature_view: FeatureView,
    fit_window: EvaluationSegment,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
    fit_strategy: FitStrategySpec,
) -> _FamilyFitResult:
    panel = _complete_strategy_training_panel(
        feature_view=feature_view,
        fit_window=fit_window,
        fit_strategy=fit_strategy,
    )
    base_fit = _fit_analytic(
        candidate=candidate,
        training_rows=training_rows,
        random_seed=random_seed,
    )
    rows_by_entity = _feature_rows_by_entity(feature_view)
    parameter_summary: dict[str, float | int] = {
        f"rectify_base__{name}": value
        for name, value in base_fit.parameter_summary.items()
    }
    correction_rows: list[dict[str, Any]] = []
    corrections: dict[int, float] = {}
    for horizon in fit_strategy.horizon_set:
        residuals: list[float] = []
        for record in panel.records:
            if record.horizon != horizon:
                continue
            origin_row = rows_by_entity[record.entity][record.origin_index]
            target_row = rows_by_entity[record.entity][record.target_index]
            base_prediction = _predict_recursive_analytic_from_origin(
                parameters=base_fit.parameter_summary,
                origin_row=origin_row,
                horizon=horizon,
            )
            residual = _stable_float(float(target_row["target"]) - base_prediction)
            residuals.append(residual)
            correction_rows.append(
                {
                    **record.as_dict(),
                    "base_prediction": _stable_float(base_prediction),
                    "correction_residual": residual,
                }
            )
        correction = sum(residuals) / len(residuals) if residuals else 0.0
        corrections[horizon] = _stable_float(correction)
        parameter_summary[f"rectify_horizon_{horizon}__intercept"] = _stable_float(
            correction
        )
    initial_state = _state_mapping(candidate)
    state_transitions = _identity_transitions(
        initial_state=initial_state,
        training_rows=training_rows,
    )
    family_fit = _FamilyFitResult(
        backend_id="deterministic_rectify_analytic_v1",
        objective_id="least_squares_rectify_analytic_v1",
        parameter_summary=parameter_summary,
        updated_literals={},
        initial_state=initial_state,
        final_state=(
            initial_state if not state_transitions else state_transitions[-1].state_after
        ),
        state_transitions=state_transitions,
        converged=True,
        iteration_count=len(fit_strategy.horizon_set) + 1,
        final_loss=0.0,
        component_diagnostics={
            "rectify": {
                "base_parameter_summary": _normalize_parameter_summary(
                    base_fit.parameter_summary
                ),
                "corrections": {
                    f"horizon_{horizon}": value
                    for horizon, value in sorted(corrections.items())
                },
                "correction_training_origin_set_id": training_origin_set_id(panel),
                "correction_training_rows": correction_rows,
            },
        },
    )
    return _with_rollout_objective_diagnostics(
        candidate=candidate,
        family_fit=family_fit,
        feature_view=feature_view,
        fit_window=fit_window,
        training_rows=training_rows,
        fit_strategy=fit_strategy,
        training_origin_panel=panel,
    )


def _fit_recursive_rollout(
    *,
    candidate: CandidateIntermediateRepresentation,
    feature_view: FeatureView,
    fit_window: EvaluationSegment,
    training_rows: Sequence[Mapping[str, Any]],
    fit_strategy: FitStrategySpec,
) -> _FamilyFitResult:
    panel = _complete_strategy_training_panel(
        feature_view=feature_view,
        fit_window=fit_window,
        fit_strategy=fit_strategy,
    )
    initial_state = _state_mapping(candidate)
    if "level" not in initial_state:
        if "running_mean" in initial_state and "step_count" in initial_state:
            base_fit = _fit_recursive(
                candidate=candidate,
                training_rows=training_rows,
                random_seed="0",
            )
            return _with_rollout_objective_diagnostics(
                candidate=candidate,
                family_fit=replace(
                    base_fit,
                    objective_id="least_squares_recursive_rollout_v1",
                ),
                feature_view=feature_view,
                fit_window=fit_window,
                training_rows=training_rows,
                fit_strategy=fit_strategy,
                training_origin_panel=panel,
            )
        raise ContractValidationError(
            code="invalid_recursive_candidate_state",
            message="recursive rollout requires supported recursive state slots",
            field_path="candidate.structural_layer.state_signature.persistent_state",
        )

    best_fit: _FamilyFitResult | None = None
    best_score = math.inf
    alpha_grid = tuple(round(index / 20.0, 2) for index in range(1, 20))
    for alpha in alpha_grid:
        candidate_fit = _recursive_level_fit_for_alpha(
            initial_state=initial_state,
            training_rows=training_rows,
            alpha=alpha,
        )
        scored_fit = _with_rollout_objective_diagnostics(
            candidate=candidate,
            family_fit=candidate_fit,
            feature_view=feature_view,
            fit_window=fit_window,
            training_rows=training_rows,
            fit_strategy=fit_strategy,
            training_origin_panel=panel,
        )
        score = float(
            scored_fit.component_diagnostics["rollout_primary_objective"][
                "aggregated_primary_score"
            ]
        )
        tie_breaks_current = (
            best_fit is None
            or (
                math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12)
                and alpha < float(best_fit.parameter_summary["alpha"])
            )
        )
        if score < best_score - 1e-12 or tie_breaks_current:
            best_score = score
            best_fit = scored_fit
    assert best_fit is not None
    return replace(
        best_fit,
        objective_id="least_squares_recursive_rollout_v1",
        iteration_count=len(alpha_grid),
    )


def _fit_analytic(
    *,
    candidate: CandidateIntermediateRepresentation,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
) -> _FamilyFitResult:
    del random_seed
    return _fit_weighted_analytic(
        candidate=candidate,
        training_rows=training_rows,
        weights=tuple(1.0 for _ in training_rows),
    )


def _analytic_feature_terms(
    *,
    candidate: CandidateIntermediateRepresentation,
    parameters: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    parameter_terms = dict(_analytic_feature_terms_from_parameters(parameters))
    if not parameter_terms:
        return ()
    declared_features = tuple(
        candidate.execution_layer.history_access_contract.allowed_side_information
    )
    ordered_features = [
        feature_name
        for feature_name in declared_features
        if feature_name in parameter_terms
    ]
    ordered_features.extend(
        feature_name
        for feature_name in sorted(parameter_terms)
        if feature_name not in ordered_features
    )
    return tuple(
        (feature_name, parameter_terms[feature_name])
        for feature_name in ordered_features
    )


def _analytic_feature_terms_from_parameters(
    parameters: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    if "lag_coefficient" in parameters:
        return (("lag_1", "lag_coefficient"),)
    terms = []
    for parameter_name in sorted(parameters):
        if not parameter_name.endswith("__coefficient"):
            continue
        if parameter_name.startswith(("horizon_", "rectify_")):
            continue
        terms.append(
            (
                parameter_name[: -len("__coefficient")],
                parameter_name,
            )
        )
    return tuple(terms)


def _analytic_feature_value(
    *,
    row: Mapping[str, Any],
    feature_name: str,
) -> float:
    if feature_name not in row:
        raise ContractValidationError(
            code="unsupported_candidate_family",
            message="analytic affine fits require every declared feature column",
            field_path=f"rows.{feature_name}",
            details={"missing_feature": feature_name},
        )
    return float(row[feature_name])


def _analytic_is_recursive_rollout_compatible(
    candidate: CandidateIntermediateRepresentation,
) -> bool:
    terms = _analytic_feature_terms(
        candidate=candidate,
        parameters=_parameter_mapping(candidate),
    )
    return not terms or tuple(terms) == (("lag_1", "lag_coefficient"),)


def _fit_weighted_analytic(
    *,
    candidate: CandidateIntermediateRepresentation,
    training_rows: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
) -> _FamilyFitResult:
    if len(training_rows) != len(weights):
        raise ContractValidationError(
            code="invalid_weighted_fit_request",
            message="weighted analytic fitting requires one weight per training row",
            field_path="weights",
            details={
                "training_row_count": len(training_rows),
                "weight_count": len(weights),
            },
        )
    total_weight = sum(float(weight) for weight in weights)
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ContractValidationError(
            code="invalid_weighted_fit_request",
            message="weighted analytic fitting requires a positive finite total weight",
            field_path="weights",
        )

    targets = tuple(float(row["target"]) for row in training_rows)
    weight_values = tuple(float(weight) for weight in weights)
    parameters = _parameter_mapping(candidate)
    initial_state = _state_mapping(candidate)
    state_transitions = _identity_transitions(
        initial_state=initial_state,
        training_rows=training_rows,
    )
    final_state = (
        initial_state if not state_transitions else state_transitions[-1].state_after
    )
    feature_terms = _analytic_feature_terms(candidate=candidate, parameters=parameters)
    if feature_terms:
        if tuple(feature_terms) == (("lag_1", "lag_coefficient"),):
            x_values = tuple(float(row["lag_1"]) for row in training_rows)
            x_mean = (
                sum(
                    weight * x_value
                    for weight, x_value in zip(weight_values, x_values, strict=True)
                )
                / total_weight
            )
            y_mean = (
                sum(
                    weight * target
                    for weight, target in zip(weight_values, targets, strict=True)
                )
                / total_weight
            )
            numerator = sum(
                weight * (x_value - x_mean) * (target - y_mean)
                for weight, x_value, target in zip(
                    weight_values,
                    x_values,
                    targets,
                    strict=True,
                )
            )
            denominator = sum(
                weight * (x_value - x_mean) ** 2
                for weight, x_value in zip(weight_values, x_values, strict=True)
            )
            slope = 0.0 if denominator == 0.0 else numerator / denominator
            intercept = y_mean - (slope * x_mean)
            fitted = tuple(intercept + (slope * x_value) for x_value in x_values)
            final_loss = sum(
                weight * (target - estimate) ** 2
                for weight, target, estimate in zip(
                    weight_values,
                    targets,
                    fitted,
                    strict=True,
                )
            )
            return _FamilyFitResult(
                backend_id="deterministic_closed_form_affine_v1",
                objective_id="least_squares_one_step_residual_v1",
                parameter_summary={
                    "intercept": _stable_float(intercept),
                    "lag_coefficient": _stable_float(slope),
                },
                updated_literals={},
                initial_state=initial_state,
                final_state=final_state,
                state_transitions=state_transitions,
                converged=True,
                iteration_count=1,
                final_loss=_stable_float(final_loss),
                component_diagnostics={},
                residual_row_weights=weight_values,
            )
        design_matrix = np.asarray(
            [
                [1.0]
                + [
                    _analytic_feature_value(row=row, feature_name=feature_name)
                    for feature_name, _ in feature_terms
                ]
                for row in training_rows
            ],
            dtype=float,
        )
        target_vector = np.asarray(targets, dtype=float)
        sqrt_weights = np.sqrt(np.asarray(weight_values, dtype=float))
        weighted_design = design_matrix * sqrt_weights[:, None]
        weighted_targets = target_vector * sqrt_weights
        coefficients, *_ = np.linalg.lstsq(
            weighted_design,
            weighted_targets,
            rcond=None,
        )
        fitted_vector = design_matrix @ coefficients
        final_loss = float(
            np.sum(np.asarray(weight_values, dtype=float) * (target_vector - fitted_vector) ** 2)
        )
        parameter_summary = {"intercept": _stable_float(float(coefficients[0]))}
        for coefficient, (_, parameter_name) in zip(
            coefficients[1:],
            feature_terms,
            strict=True,
        ):
            parameter_summary[parameter_name] = _stable_float(float(coefficient))
        return _FamilyFitResult(
            backend_id="deterministic_closed_form_selected_feature_affine_v1",
            objective_id="least_squares_selected_feature_affine_v1",
            parameter_summary=parameter_summary,
            updated_literals={},
            initial_state=initial_state,
            final_state=final_state,
            state_transitions=state_transitions,
            converged=True,
            iteration_count=1,
            final_loss=_stable_float(final_loss),
            component_diagnostics={
                "analytic_feature_terms": [
                    feature_name for feature_name, _ in feature_terms
                ]
            },
            residual_row_weights=weight_values,
        )
    mean_value = (
        sum(
            weight * target
            for weight, target in zip(weight_values, targets, strict=True)
        )
        / total_weight
    )
    final_loss = sum(
        weight * (target - mean_value) ** 2
        for weight, target in zip(weight_values, targets, strict=True)
    )
    return _FamilyFitResult(
        backend_id="deterministic_closed_form_mean_v1",
        objective_id="least_squares_constant_mean_v1",
        parameter_summary={"intercept": _stable_float(mean_value)},
        updated_literals={},
        initial_state=initial_state,
        final_state=final_state,
        state_transitions=state_transitions,
        converged=True,
        iteration_count=1,
        final_loss=_stable_float(final_loss),
        component_diagnostics={},
        residual_row_weights=weight_values,
    )


def _analytic_fitted_values_from_rows(
    *,
    parameters: Mapping[str, float | int],
    rows: Sequence[Mapping[str, Any]],
    lag_key: str = "lag_1",
) -> tuple[float, ...]:
    intercept = float(parameters.get("intercept", 0.0))
    feature_terms = _analytic_feature_terms_from_parameters(parameters)
    if not feature_terms:
        return tuple(_stable_float(intercept) for _ in rows)
    return tuple(
        _stable_float(
            intercept
            + sum(
                float(parameters[parameter_name])
                * _analytic_feature_value(
                    row=row,
                    feature_name=lag_key if parameter_name == "lag_coefficient" else feature_name,
                )
                for feature_name, parameter_name in feature_terms
            )
        )
        for row in rows
    )


def _rows_with_targets(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[float],
    lag_values: Sequence[float] | None = None,
) -> tuple[dict[str, Any], ...]:
    if len(rows) != len(targets):
        raise ContractValidationError(
            code="invalid_synthetic_training_rows",
            message="synthetic targets must align one-for-one with source rows",
            field_path="targets",
        )
    if lag_values is not None and len(lag_values) != len(rows):
        raise ContractValidationError(
            code="invalid_synthetic_training_rows",
            message="synthetic lag values must align one-for-one with source rows",
            field_path="lag_values",
        )
    rewritten: list[dict[str, Any]] = []
    for index, (row, target) in enumerate(zip(rows, targets, strict=True)):
        rewritten_row = dict(row)
        rewritten_row["target"] = _stable_float(target)
        if lag_values is not None:
            rewritten_row["lag_1"] = _stable_float(lag_values[index])
        rewritten.append(rewritten_row)
    return tuple(rewritten)


def _complete_strategy_training_panel(
    *,
    feature_view: FeatureView,
    fit_window: EvaluationSegment,
    fit_strategy: FitStrategySpec,
):
    panel = build_legal_training_origin_panel(
        feature_view=feature_view,
        evaluation_segment=fit_window,
        horizon_set=fit_strategy.horizon_set,
    )
    if panel.status != "passed" or not panel.records:
        raise ContractValidationError(
            code="incomplete_rollout_objective_panel",
            message="fit strategy requires a complete legal training-origin panel",
            field_path="fit_window",
            details={
                "fit_strategy_id": fit_strategy.strategy_id,
                "diagnostics": [
                    diagnostic.as_dict() for diagnostic in panel.diagnostics
                ],
            },
        )
    return panel


def _with_rollout_objective_diagnostics(
    *,
    candidate: CandidateIntermediateRepresentation,
    family_fit: _FamilyFitResult,
    feature_view: FeatureView,
    fit_window: EvaluationSegment,
    training_rows: Sequence[Mapping[str, Any]],
    fit_strategy: FitStrategySpec,
    training_origin_panel,
) -> _FamilyFitResult:
    rows = _rollout_objective_rows(
        candidate=candidate,
        family_fit=family_fit,
        feature_view=feature_view,
        fit_window=fit_window,
        training_rows=training_rows,
        training_origin_panel=training_origin_panel,
    )
    diagnostics = _rollout_objective_diagnostics(
        rows=rows,
        fit_strategy=fit_strategy,
        training_origin_panel=training_origin_panel,
    )
    component_diagnostics = dict(family_fit.component_diagnostics or {})
    component_diagnostics.update(
        {
            "fit_strategy_id": fit_strategy.strategy_id,
            "fit_strategy_identity": fit_strategy.identity_hash,
            "objective_geometry": "legal_training_origin_rollout_panel",
            "rollout_primary_objective": diagnostics,
        }
    )
    return replace(
        family_fit,
        final_loss=diagnostics["aggregated_primary_score"],
        component_diagnostics=component_diagnostics,
    )


def _rollout_objective_rows(
    *,
    candidate: CandidateIntermediateRepresentation,
    family_fit: _FamilyFitResult,
    feature_view: FeatureView,
    fit_window: EvaluationSegment,
    training_rows: Sequence[Mapping[str, Any]],
    training_origin_panel,
) -> tuple[dict[str, Any], ...]:
    rows_by_entity = _feature_rows_by_entity(feature_view)
    training_positions = _training_row_positions(
        training_rows=training_rows,
        fit_window=fit_window,
    )
    include_entity = len(training_origin_panel.entity_panel) > 1
    rows: list[dict[str, Any]] = []
    for panel_row in training_origin_panel.records:
        origin_row = rows_by_entity[panel_row.entity][panel_row.origin_index]
        target_row = rows_by_entity[panel_row.entity][panel_row.target_index]
        trace = _forecast_training_panel_row(
            candidate=candidate,
            family_fit=family_fit,
            training_rows=training_rows,
            training_positions=training_positions,
            panel_row=panel_row,
            origin_row=origin_row,
            target_row=target_row,
            row_position=training_positions[(panel_row.entity, panel_row.target_index)],
        )
        row = {
            "available_at": panel_row.target_available_at,
            "horizon": panel_row.horizon,
            "origin_index": panel_row.origin_index,
            "origin_time": panel_row.origin_time,
            "point_forecast": _stable_float(trace.point_forecast),
            "realized_observation": _stable_float(float(target_row["target"])),
            "target_index": panel_row.target_index,
            "training_origin_id": panel_row.training_origin_id,
        }
        if include_entity:
            row["entity"] = panel_row.entity
        rows.append(row)
    return tuple(rows)


def _rollout_objective_diagnostics(
    *,
    rows: tuple[Mapping[str, Any], ...],
    fit_strategy: FitStrategySpec,
    training_origin_panel,
) -> dict[str, Any]:
    horizon_weights = tuple(
        (int(item["horizon"]), float(item["weight"]))
        for item in fit_strategy.horizon_weights
    )
    entity_weights = _rollout_entity_weights(
        training_origin_panel=training_origin_panel,
        fit_strategy=fit_strategy,
    )
    per_horizon, aggregated_primary_score = _aggregate_primary_scores(
        rows=rows,
        horizon_weights=horizon_weights,
        entity_aggregation_mode=fit_strategy.entity_aggregation_mode,
        entity_weights=tuple(
            (str(item["entity"]), float(item["weight"])) for item in entity_weights
        ),
        row_score=lambda row: _point_loss(
            point_loss_id=fit_strategy.point_loss_id,
            point_forecast=float(row["point_forecast"]),
            realized_observation=float(row["realized_observation"]),
        ),
    )
    return {
        "aggregated_primary_score": _stable_float(aggregated_primary_score),
        "entity_aggregation_mode": fit_strategy.entity_aggregation_mode,
        "entity_weights": [dict(item) for item in entity_weights],
        "fit_strategy_id": fit_strategy.strategy_id,
        "fit_strategy_identity": fit_strategy.identity_hash,
        "horizon_set": list(fit_strategy.horizon_set),
        "horizon_weights": [dict(item) for item in fit_strategy.horizon_weights],
        "per_horizon": [
            {
                "horizon": horizon,
                "valid_origin_count": valid_origin_count,
                "mean_point_loss": mean_point_loss,
            }
            for horizon, valid_origin_count, mean_point_loss in per_horizon
        ],
        "point_loss_id": fit_strategy.point_loss_id,
        "row_count": len(rows),
        "rows": [dict(row) for row in rows],
        "training_origin_set_id": training_origin_set_id(training_origin_panel),
    }


def _rollout_entity_weights(
    *,
    training_origin_panel,
    fit_strategy: FitStrategySpec,
) -> tuple[dict[str, Any], ...]:
    if (
        fit_strategy.entity_aggregation_mode
        == "single_entity_only_no_cross_entity_aggregation"
    ):
        return ()
    entity_panel = tuple(str(entity) for entity in training_origin_panel.entity_panel)
    if not entity_panel:
        return ()
    remaining = 1.0
    weights: list[dict[str, Any]] = []
    for index, entity in enumerate(entity_panel):
        if index == len(entity_panel) - 1:
            weight = remaining
        else:
            weight = _stable_float(1.0 / len(entity_panel))
            remaining = _stable_float(remaining - weight)
        weights.append({"entity": entity, "weight": _stable_float(weight)})
    return tuple(weights)


def _fit_joint_intercept_only_parameters(
    *,
    feature_view: FeatureView,
    training_origin_panel,
    fit_strategy: FitStrategySpec,
) -> dict[str, float]:
    rows_by_entity = _feature_rows_by_entity(feature_view)
    weighted_sum = 0.0
    total_weight = 0.0
    horizon_weights = _strategy_horizon_weights(fit_strategy)
    for record in training_origin_panel.records:
        weight = horizon_weights[record.horizon]
        target = float(rows_by_entity[record.entity][record.target_index]["target"])
        weighted_sum += weight * target
        total_weight += weight
    return {"intercept": _stable_float(weighted_sum / total_weight)}


def _optimize_joint_affine_parameters(
    *,
    candidate: CandidateIntermediateRepresentation,
    feature_view: FeatureView,
    training_origin_panel,
    fit_strategy: FitStrategySpec,
    initial_parameters: Mapping[str, float | int],
) -> dict[str, float]:
    del candidate
    rows_by_entity = _feature_rows_by_entity(feature_view)
    target_values = [
        float(rows_by_entity[record.entity][record.target_index]["target"])
        for record in training_origin_panel.records
    ]
    target_scale = max(max(target_values) - min(target_values), 1.0)
    current = (
        float(initial_parameters.get("intercept", 0.0)),
        float(initial_parameters.get("lag_coefficient", 0.0)),
    )
    best_loss = _joint_rollout_loss(
        parameters={"intercept": current[0], "lag_coefficient": current[1]},
        rows_by_entity=rows_by_entity,
        training_origin_panel=training_origin_panel,
        fit_strategy=fit_strategy,
    )
    intercept_step = target_scale / 2.0
    slope_step = 1.0
    for _ in range(18):
        improved = False
        candidates = []
        for intercept_delta in (-intercept_step, 0.0, intercept_step):
            for slope_delta in (-slope_step, 0.0, slope_step):
                candidates.append(
                    (
                        current[0] + intercept_delta,
                        current[1] + slope_delta,
                    )
                )
        for intercept, slope in candidates:
            loss = _joint_rollout_loss(
                parameters={"intercept": intercept, "lag_coefficient": slope},
                rows_by_entity=rows_by_entity,
                training_origin_panel=training_origin_panel,
                fit_strategy=fit_strategy,
            )
            if loss < best_loss - 1e-12:
                best_loss = loss
                current = (intercept, slope)
                improved = True
        if not improved:
            intercept_step /= 2.0
            slope_step /= 2.0
    return {
        "intercept": _stable_float(current[0]),
        "lag_coefficient": _stable_float(current[1]),
    }


def _joint_rollout_loss(
    *,
    parameters: Mapping[str, float | int],
    rows_by_entity: Mapping[str, Sequence[Mapping[str, Any]]],
    training_origin_panel,
    fit_strategy: FitStrategySpec,
) -> float:
    horizon_weights = _strategy_horizon_weights(fit_strategy)
    losses_by_horizon: dict[int, list[float]] = {
        horizon: [] for horizon in fit_strategy.horizon_set
    }
    for record in training_origin_panel.records:
        origin_row = rows_by_entity[record.entity][record.origin_index]
        target_row = rows_by_entity[record.entity][record.target_index]
        forecast = _predict_recursive_analytic_from_origin(
            parameters=parameters,
            origin_row=origin_row,
            horizon=record.horizon,
        )
        loss = _point_loss(
            point_loss_id=fit_strategy.point_loss_id,
            point_forecast=forecast,
            realized_observation=float(target_row["target"]),
        )
        losses_by_horizon[record.horizon].append(loss)
    return _stable_float(
        sum(
            horizon_weights[horizon]
            * (sum(losses_by_horizon[horizon]) / len(losses_by_horizon[horizon]))
            for horizon in fit_strategy.horizon_set
        )
    )


def _strategy_horizon_weights(fit_strategy: FitStrategySpec) -> dict[int, float]:
    return {
        int(item["horizon"]): float(item["weight"])
        for item in fit_strategy.horizon_weights
    }


def _recursive_level_fit_for_alpha(
    *,
    initial_state: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    alpha: float,
) -> _FamilyFitResult:
    level = float(initial_state["level"])
    step_count = int(initial_state.get("step_count", 0))
    transitions: list[PersistentStateTransition] = []
    for transition_index, row in enumerate(training_rows):
        observed = float(row["target"])
        state_before = {
            "level": _stable_float(level),
            "step_count": step_count,
        }
        level = (alpha * observed) + ((1.0 - alpha) * level)
        step_count += 1
        state_after = {
            "level": _stable_float(level),
            "step_count": step_count,
        }
        transitions.append(
            PersistentStateTransition(
                transition_index=transition_index,
                observation_index=transition_index,
                event_time=str(row["event_time"]),
                available_at=str(row["available_at"]),
                observed_value=_stable_float(observed),
                state_before=state_before,
                state_after=state_after,
            )
        )
    return _FamilyFitResult(
        backend_id="deterministic_recursive_rollout_alpha_grid_v1",
        objective_id="least_squares_recursive_rollout_v1",
        parameter_summary={"alpha": _stable_float(alpha)},
        updated_literals={"alpha": _stable_float(alpha)},
        initial_state=initial_state,
        final_state=dict(transitions[-1].state_after) if transitions else initial_state,
        state_transitions=tuple(transitions),
        converged=True,
        iteration_count=1,
        final_loss=0.0,
        component_diagnostics={},
    )


def _component_candidate(
    candidate: CandidateIntermediateRepresentation,
    *,
    component_id: str,
) -> CandidateIntermediateRepresentation:
    parameters = extract_component_mapping(_parameter_mapping(candidate), component_id)
    literals = extract_component_mapping(_literal_mapping(candidate), component_id)
    final_state = extract_component_mapping(_state_mapping(candidate), component_id)
    reducer = ReducerObject(
        family=ReducerFamilyId(candidate.structural_layer.cir_family_id),
        composition_object=parse_reducer_composition({}),
        fitted_parameters=ReducerParameterObject(
            parameters=tuple(
                ReducerParameter(name=name, value=value)
                for name, value in sorted(parameters.items())
            )
        ),
        state_semantics=ReducerStateSemantics(
            persistent_state=ReducerStateObject(
                slots=tuple(
                    ReducerStateSlot(name=name, value=value)
                    for name, value in sorted(final_state.items())
                )
            ),
            update_rule=_update_rule(
                family_id=candidate.structural_layer.cir_family_id,
                parameter_summary=parameters,
                literals=literals,
                max_lag=int(
                    candidate.execution_layer.history_access_contract.max_lag or 0
                ),
            ),
        ),
        observation_model=candidate.execution_layer.observation_model_binding,
        admissibility=ReducerAdmissibilityObject(
            family_membership=True,
            composition_closure=True,
            observation_model_compatibility=True,
            valid_state_semantics=True,
            codelength_comparability=True,
        ),
    )
    return build_cir_candidate_from_reducer(
        reducer=reducer,
        cir_form_class=candidate.structural_layer.cir_form_class,
        input_signature=candidate.structural_layer.input_signature,
        history_access_contract=candidate.execution_layer.history_access_contract,
        literal_block=CIRLiteralBlock(
            literals=tuple(
                CIRLiteral(name=name, value=value)
                for name, value in sorted(literals.items())
            )
        ),
        forecast_operator=candidate.execution_layer.forecast_operator,
        model_code_decomposition=candidate.evidence_layer.model_code_decomposition,
        backend_origin_record=candidate.evidence_layer.backend_origin_record,
        replay_hooks=candidate.evidence_layer.replay_hooks,
        transient_diagnostics=dict(candidate.evidence_layer.transient_diagnostics),
    )


def _preserve_declared_component(
    candidate: CandidateIntermediateRepresentation,
) -> _FamilyFitResult:
    initial_state = _state_mapping(candidate)
    return _FamilyFitResult(
        backend_id="declared_component_passthrough_v1",
        objective_id="no_training_rows_component_passthrough",
        parameter_summary=_parameter_mapping(candidate),
        updated_literals={},
        initial_state=initial_state,
        final_state=initial_state,
        state_transitions=(),
        converged=True,
        iteration_count=0,
        final_loss=0.0,
        component_diagnostics={},
    )


def _require_analytic_composition_family(
    candidate: CandidateIntermediateRepresentation,
    *,
    operator_id: str,
) -> None:
    family_id = candidate.structural_layer.cir_family_id
    if family_id != "analytic":
        raise ContractValidationError(
            code="unsupported_candidate_family",
            message=(
                f"{operator_id} runtime semantics are currently implemented "
                "for analytic candidates only in the retained bounded v1 slice"
            ),
            field_path="candidate.structural_layer.cir_family_id",
            details={"family_id": family_id, "operator_id": operator_id},
        )


def _fit_recursive(
    *,
    candidate: CandidateIntermediateRepresentation,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
) -> _FamilyFitResult:
    del random_seed
    initial_state = _state_mapping(candidate)
    if "level" in initial_state:
        best_alpha = 0.5
        best_loss = math.inf
        best_final_state: Mapping[str, Any] = initial_state
        best_transitions: tuple[PersistentStateTransition, ...] = ()
        alpha_grid = tuple(round(index / 20.0, 2) for index in range(1, 20))
        for alpha in alpha_grid:
            level = float(initial_state["level"])
            step_count = int(initial_state.get("step_count", 0))
            loss = 0.0
            transitions: list[PersistentStateTransition] = []
            for transition_index, row in enumerate(training_rows):
                observed = float(row["target"])
                state_before = {
                    "level": _stable_float(level),
                    "step_count": step_count,
                }
                loss += (observed - level) ** 2
                level = (alpha * observed) + ((1.0 - alpha) * level)
                step_count += 1
                state_after = {
                    "level": _stable_float(level),
                    "step_count": step_count,
                }
                transitions.append(
                    PersistentStateTransition(
                        transition_index=transition_index,
                        observation_index=transition_index,
                        event_time=str(row["event_time"]),
                        available_at=str(row["available_at"]),
                        observed_value=_stable_float(observed),
                        state_before=state_before,
                        state_after=state_after,
                    )
                )
            if loss < best_loss - 1e-12 or (
                math.isclose(loss, best_loss, rel_tol=0.0, abs_tol=1e-12)
                and alpha < best_alpha
            ):
                best_alpha = alpha
                best_loss = loss
                best_transitions = tuple(transitions)
                best_final_state = (
                    dict(transitions[-1].state_after) if transitions else initial_state
                )
        return _FamilyFitResult(
            backend_id="deterministic_alpha_grid_v1",
            objective_id="least_squares_recursive_level_v1",
            parameter_summary={"alpha": _stable_float(best_alpha)},
            updated_literals={"alpha": _stable_float(best_alpha)},
            initial_state=initial_state,
            final_state=best_final_state,
            state_transitions=best_transitions,
            converged=True,
            iteration_count=len(alpha_grid),
            final_loss=_stable_float(best_loss),
            component_diagnostics={},
        )

    if "running_mean" in initial_state and "step_count" in initial_state:
        running_mean = float(initial_state["running_mean"])
        step_count = int(initial_state["step_count"])
        loss = 0.0
        transitions: list[PersistentStateTransition] = []
        for transition_index, row in enumerate(training_rows):
            observed = float(row["target"])
            state_before = {
                "running_mean": _stable_float(running_mean),
                "step_count": step_count,
            }
            loss += (observed - running_mean) ** 2
            next_step_count = step_count + 1
            running_mean = ((running_mean * step_count) + observed) / next_step_count
            step_count = next_step_count
            state_after = {
                "running_mean": _stable_float(running_mean),
                "step_count": step_count,
            }
            transitions.append(
                PersistentStateTransition(
                    transition_index=transition_index,
                    observation_index=transition_index,
                    event_time=str(row["event_time"]),
                    available_at=str(row["available_at"]),
                    observed_value=_stable_float(observed),
                    state_before=state_before,
                    state_after=state_after,
                )
            )
        final_state = (
            dict(transitions[-1].state_after) if transitions else initial_state
        )
        return _FamilyFitResult(
            backend_id="deterministic_running_mean_rollforward_v1",
            objective_id="least_squares_running_mean_v1",
            parameter_summary={},
            updated_literals={},
            initial_state=initial_state,
            final_state=final_state,
            state_transitions=tuple(transitions),
            converged=True,
            iteration_count=1,
            final_loss=_stable_float(loss),
            component_diagnostics={},
        )

    raise ContractValidationError(
        code="invalid_recursive_candidate_state",
        message="recursive candidates require supported persistent state slots",
        field_path="candidate.structural_layer.state_signature.persistent_state",
    )


def _fit_spectral(
    *,
    candidate: CandidateIntermediateRepresentation,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
) -> _FamilyFitResult:
    del random_seed
    initial_state = _state_mapping(candidate)
    literals = _literal_mapping(candidate)
    harmonics = _spectral_harmonics(literals)
    harmonic_group = len(harmonics) > 1 or "harmonic_group" in literals
    season_length = int(literals["season_length"])
    phase_index = int(initial_state.get("phase_index", 0))
    design_rows: list[list[float]] = []
    targets = tuple(float(row["target"]) for row in training_rows)
    for _ in training_rows:
        design_row: list[float] = []
        for harmonic in harmonics:
            angle = (2.0 * math.pi * harmonic * phase_index) / season_length
            design_row.extend((math.cos(angle), math.sin(angle)))
        design_rows.append(design_row)
        phase_index = (phase_index + 1) % season_length

    design_matrix = np.asarray(design_rows, dtype=float)
    target_vector = np.asarray(targets, dtype=float)
    coefficients, residuals, rank, _ = np.linalg.lstsq(
        design_matrix,
        target_vector,
        rcond=None,
    )
    converged = bool(rank == design_matrix.shape[1])
    parameter_summary = _spectral_parameter_summary(
        harmonics=harmonics,
        coefficients=tuple(float(value) for value in coefficients.tolist()),
        harmonic_group=harmonic_group,
    )

    last_basis_value = float(initial_state.get("last_basis_value", 0.0))
    phase_index = int(initial_state.get("phase_index", 0))
    transitions: list[PersistentStateTransition] = []
    fitted_values: list[float] = []
    for transition_index, row in enumerate(training_rows):
        fitted = _spectral_prediction(
            parameters=parameter_summary,
            literals=literals,
            phase_index=phase_index,
        )
        fitted_values.append(fitted)
        state_before = {
            "last_basis_value": _stable_float(last_basis_value),
            "phase_index": phase_index,
        }
        phase_index = (phase_index + 1) % season_length
        next_angle = (2.0 * math.pi * harmonics[0] * phase_index) / season_length
        last_basis_value = math.sin(next_angle)
        state_after = {
            "last_basis_value": _stable_float(last_basis_value),
            "phase_index": phase_index,
        }
        transitions.append(
            PersistentStateTransition(
                transition_index=transition_index,
                observation_index=transition_index,
                event_time=str(row["event_time"]),
                available_at=str(row["available_at"]),
                observed_value=_stable_float(float(row["target"])),
                state_before=state_before,
                state_after=state_after,
            )
        )
    final_state = dict(transitions[-1].state_after) if transitions else initial_state
    final_loss = sum(
        (actual - fitted) ** 2
        for actual, fitted in zip(targets, fitted_values, strict=True)
    )
    return _FamilyFitResult(
        backend_id="deterministic_harmonic_least_squares_v1",
        objective_id=(
            "least_squares_harmonic_group_basis_v1"
            if harmonic_group
            else "least_squares_harmonic_basis_v1"
        ),
        parameter_summary=parameter_summary,
        updated_literals={},
        initial_state=initial_state,
        final_state=final_state,
        state_transitions=tuple(transitions),
        converged=converged,
        iteration_count=1,
        final_loss=_stable_float(final_loss),
        component_diagnostics=(
            {"spectral_harmonic_group": list(harmonics)}
            if harmonic_group
            else {}
        ),
    )


def _spectral_harmonics(literals: Mapping[str, Any]) -> tuple[int, ...]:
    raw_group = literals.get("harmonic_group")
    if isinstance(raw_group, str) and raw_group.strip():
        return tuple(int(token) for token in raw_group.split(",") if token.strip())
    raw_harmonics = literals.get("harmonics")
    if isinstance(raw_harmonics, str) and raw_harmonics.strip():
        return tuple(int(token) for token in raw_harmonics.split(",") if token.strip())
    return (int(literals["harmonic"]),)


def _spectral_parameter_summary(
    *,
    harmonics: tuple[int, ...],
    coefficients: tuple[float, ...],
    harmonic_group: bool,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for index, harmonic in enumerate(harmonics):
        cosine = _stable_float(coefficients[index * 2])
        sine = _stable_float(coefficients[(index * 2) + 1])
        if not harmonic_group and len(harmonics) == 1:
            summary["cosine_coefficient"] = cosine
            summary["sine_coefficient"] = sine
        else:
            summary[f"cosine_{harmonic}_coefficient"] = cosine
            summary[f"sine_{harmonic}_coefficient"] = sine
    return summary


def _spectral_prediction(
    *,
    parameters: Mapping[str, float | int],
    literals: Mapping[str, Any],
    phase_index: int,
) -> float:
    season_length = int(literals["season_length"])
    harmonics = _spectral_harmonics(literals)
    harmonic_group = len(harmonics) > 1 or "harmonic_group" in literals
    forecast = 0.0
    for harmonic in harmonics:
        angle = (2.0 * math.pi * harmonic * phase_index) / season_length
        if harmonic_group:
            cosine = float(parameters.get(f"cosine_{harmonic}_coefficient", 0.0))
            sine = float(parameters.get(f"sine_{harmonic}_coefficient", 0.0))
        else:
            cosine = float(parameters.get("cosine_coefficient", 0.0))
            sine = float(parameters.get("sine_coefficient", 0.0))
        forecast += (cosine * math.cos(angle)) + (sine * math.sin(angle))
    return _stable_float(forecast)


def _fit_algorithmic(
    *,
    candidate: CandidateIntermediateRepresentation,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
) -> _FamilyFitResult:
    del random_seed
    literals = _literal_mapping(candidate)
    max_lag = int(candidate.execution_layer.history_access_contract.max_lag or 0)
    program = parse_algorithmic_program(
        str(literals["algorithmic_program"]),
        state_slot_count=int(literals.get("algorithmic_state_slot_count", 1)),
        max_program_nodes=int(literals.get("program_node_count", 8)),
        allowed_observation_lags=_algorithmic_allowed_observation_lags(
            literals=literals,
            max_lag=max_lag,
        ),
    )
    initial_state = _state_mapping(candidate)
    state = tuple(
        Fraction(str(initial_state[f"state_{index}"]))
        for index in range(program.state_slot_count)
    )
    transitions: list[PersistentStateTransition] = []
    loss = 0.0
    for transition_index, row in enumerate(training_rows):
        observed = Fraction(str(float(row["target"])))
        observation_window = _algorithmic_observation_window(
            row=row,
            max_lag=max_lag,
            current_value=float(row["target"]),
        )
        step = evaluate_algorithmic_program(
            program,
            state=state,
            observation=observation_window,
        )
        state_before = {
            f"state_{index}": canonicalize_fraction(value)
            for index, value in enumerate(state)
        }
        state_after = {
            f"state_{index}": canonicalize_fraction(value)
            for index, value in enumerate(step.next_state)
        }
        loss += (float(step.emit_value) - float(observed)) ** 2
        transitions.append(
            PersistentStateTransition(
                transition_index=transition_index,
                observation_index=transition_index,
                event_time=str(row["event_time"]),
                available_at=str(row["available_at"]),
                observed_value=_stable_float(float(observed)),
                state_before=state_before,
                state_after=state_after,
            )
        )
        state = step.next_state
    final_state = (
        dict(transitions[-1].state_after) if transitions else dict(initial_state)
    )
    return _FamilyFitResult(
        backend_id="deterministic_algorithmic_replay_v1",
        objective_id="least_squares_algorithmic_emit_v1",
        parameter_summary={},
        updated_literals={},
        initial_state=initial_state,
        final_state=final_state,
        state_transitions=tuple(transitions),
        converged=True,
        iteration_count=1,
        final_loss=_stable_float(loss),
        component_diagnostics={},
    )


def _fit_shared_local(
    *,
    candidate: CandidateIntermediateRepresentation,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
    backend_preference: str | None = None,
) -> _FamilyFitResult:
    summary = fit_shared_plus_local_decomposition(
        candidate=candidate,
        training_rows=training_rows,
        random_seed=random_seed,
        backend_preference=backend_preference or "best",
    )
    initial_state = _state_mapping(candidate)
    state_transitions = _identity_transitions(
        initial_state=initial_state,
        training_rows=training_rows,
    )
    return _FamilyFitResult(
        backend_id=summary.backend_id,
        objective_id=summary.objective_id,
        parameter_summary=dict(summary.parameter_summary),
        updated_literals={},
        initial_state=initial_state,
        final_state=initial_state,
        state_transitions=state_transitions,
        converged=True,
        iteration_count=1,
        final_loss=summary.final_loss,
        component_diagnostics={"shared_local": summary.as_diagnostics()},
    )


def _algorithmic_allowed_observation_lags(
    *,
    literals: Mapping[str, Any],
    max_lag: int,
) -> tuple[int, ...]:
    raw_lags = literals.get("algorithmic_allowed_observation_lags")
    if isinstance(raw_lags, str) and raw_lags.strip():
        return tuple(int(token) for token in raw_lags.split(",") if token.strip())
    return tuple(range(max_lag + 1))


def _algorithmic_observation_window(
    *,
    row: Mapping[str, Any],
    max_lag: int,
    current_value: float,
) -> tuple[Fraction, ...]:
    values: list[Fraction] = [Fraction(str(float(current_value)))]
    for lag in range(1, max_lag + 1):
        field_name = f"lag_{lag}"
        if field_name not in row:
            raise ContractValidationError(
                code="unsupported_candidate_family",
                message=(
                    "algorithmic lagged observation access requires explicit "
                    f"{field_name} features on every training row"
                ),
                field_path=f"rows.{field_name}",
            )
        values.append(Fraction(str(float(row[field_name]))))
    return tuple(values)


def _build_fitted_candidate(
    *,
    candidate: CandidateIntermediateRepresentation,
    parameter_summary: Mapping[str, float | int],
    updated_literals: Mapping[str, float | int],
    final_state: Mapping[str, Any],
    backend_id: str,
    fit_window_id: str,
    optimizer_diagnostics: Mapping[str, Any],
    component_diagnostics: Mapping[str, Any] | None = None,
) -> CandidateIntermediateRepresentation:
    family_id = candidate.structural_layer.cir_family_id
    literals = _literal_mapping(candidate)
    literals.update(updated_literals)
    reducer = ReducerObject(
        family=ReducerFamilyId(family_id),
        composition_object=candidate.structural_layer.composition_graph,
        fitted_parameters=ReducerParameterObject(
            parameters=tuple(
                ReducerParameter(name=name, value=value)
                for name, value in sorted(parameter_summary.items())
            )
        ),
        state_semantics=ReducerStateSemantics(
            persistent_state=ReducerStateObject(
                slots=tuple(
                    ReducerStateSlot(name=name, value=value)
                    for name, value in sorted(final_state.items())
                )
            ),
            update_rule=_update_rule(
                family_id=family_id,
                parameter_summary=parameter_summary,
                literals=literals,
                max_lag=int(
                    candidate.execution_layer.history_access_contract.max_lag or 0
                ),
            ),
        ),
        observation_model=candidate.execution_layer.observation_model_binding,
        admissibility=ReducerAdmissibilityObject(
            family_membership=True,
            composition_closure=True,
            observation_model_compatibility=True,
            valid_state_semantics=True,
            codelength_comparability=True,
        ),
    )
    replay_hooks = CIRReplayHooks(
        hooks=(
            *candidate.evidence_layer.replay_hooks.hooks,
            CIRReplayHook(hook_name="fit_backend", hook_ref=f"backend:{backend_id}"),
            CIRReplayHook(hook_name="fit_window", hook_ref=f"window:{fit_window_id}"),
        )
    )
    transient_diagnostics = dict(candidate.evidence_layer.transient_diagnostics)
    transient_diagnostics.update({"fitting": dict(optimizer_diagnostics)})
    if component_diagnostics:
        transient_diagnostics.update(component_diagnostics)
    return build_cir_candidate_from_reducer(
        reducer=reducer,
        cir_form_class=candidate.structural_layer.cir_form_class,
        input_signature=candidate.structural_layer.input_signature,
        history_access_contract=candidate.execution_layer.history_access_contract,
        literal_block=CIRLiteralBlock(
            literals=tuple(
                CIRLiteral(name=name, value=value)
                for name, value in sorted(literals.items())
            )
        ),
        forecast_operator=candidate.execution_layer.forecast_operator,
        model_code_decomposition=candidate.evidence_layer.model_code_decomposition,
        backend_origin_record=candidate.evidence_layer.backend_origin_record,
        replay_hooks=replay_hooks,
        transient_diagnostics=transient_diagnostics,
    )


def _update_rule(
    *,
    family_id: str,
    parameter_summary: Mapping[str, float | int],
    literals: Mapping[str, float | int],
    max_lag: int = 0,
) -> ReducerStateUpdateRule:
    if family_id == "analytic":
        return ReducerStateUpdateRule(
            update_rule_id="analytic_identity_update",
            implementation=lambda state, context: state,
        )
    if family_id == "recursive":
        if "alpha" in literals:
            alpha = float(literals["alpha"])

            def update(
                state: ReducerStateObject,
                context: ReducerStateUpdateContext,
            ) -> ReducerStateObject:
                previous_level = float(state.get("level"))
                observed = (
                    previous_level
                    if not context.history
                    else float(context.history[-1])
                )
                next_level = (alpha * observed) + ((1.0 - alpha) * previous_level)
                return ReducerStateObject(
                    slots=(
                        ReducerStateSlot(name="level", value=_stable_float(next_level)),
                        ReducerStateSlot(
                            name="step_count",
                            value=int(state.get("step_count")) + 1,
                        ),
                    )
                )

            return ReducerStateUpdateRule(
                update_rule_id="recursive_level_smoother_update",
                implementation=update,
            )

        def update(
            state: ReducerStateObject,
            context: ReducerStateUpdateContext,
        ) -> ReducerStateObject:
            previous_mean = float(state.get("running_mean"))
            previous_steps = int(state.get("step_count"))
            observed = (
                previous_mean if not context.history else float(context.history[-1])
            )
            next_steps = previous_steps + 1
            next_mean = ((previous_mean * previous_steps) + observed) / next_steps
            return ReducerStateObject(
                slots=(
                    ReducerStateSlot(
                        name="running_mean",
                        value=_stable_float(next_mean),
                    ),
                    ReducerStateSlot(name="step_count", value=next_steps),
                )
            )

        return ReducerStateUpdateRule(
            update_rule_id="recursive_running_mean_update",
            implementation=update,
        )
    if family_id == "spectral":
        season_length = int(literals["season_length"])
        harmonic = _spectral_harmonics(literals)[0]

        def update(
            state: ReducerStateObject,
            context: ReducerStateUpdateContext,
        ) -> ReducerStateObject:
            del context
            phase_index = (int(state.get("phase_index")) + 1) % season_length
            angle = (2.0 * math.pi * harmonic * phase_index) / season_length
            basis_value = math.sin(angle)
            return ReducerStateObject(
                slots=(
                    ReducerStateSlot(
                        name="last_basis_value",
                        value=_stable_float(basis_value),
                    ),
                    ReducerStateSlot(name="phase_index", value=phase_index),
                )
            )

        return ReducerStateUpdateRule(
            update_rule_id="spectral_harmonic_basis_update",
            implementation=update,
        )
    if family_id == "algorithmic":
        program = parse_algorithmic_program(
            str(literals["algorithmic_program"]),
            state_slot_count=int(literals.get("algorithmic_state_slot_count", 1)),
            max_program_nodes=int(literals.get("program_node_count", 8)),
            allowed_observation_lags=_algorithmic_allowed_observation_lags(
                literals=literals,
                max_lag=max_lag,
            ),
        )

        def update(
            state: ReducerStateObject,
            context: ReducerStateUpdateContext,
        ) -> ReducerStateObject:
            history = tuple(context.history)
            observation_window = tuple(
                (
                    Fraction(str(float(history[-1 - lag])))
                    if len(history) > lag
                    else Fraction(0, 1)
                )
                for lag in range(max_lag + 1)
            )
            state_tuple = tuple(
                Fraction(str(state.get(f"state_{index}")))
                for index in range(program.state_slot_count)
            )
            step = evaluate_algorithmic_program(
                program,
                state=state_tuple,
                observation=observation_window,
            )
            return ReducerStateObject(
                slots=tuple(
                    ReducerStateSlot(
                        name=f"state_{index}",
                        value=canonicalize_fraction(value),
                    )
                    for index, value in enumerate(step.next_state)
                )
            )

        return ReducerStateUpdateRule(
            update_rule_id=f"algorithmic_program::{program.canonical_source}",
            implementation=update,
        )
    raise ContractValidationError(
        code="unsupported_candidate_family",
        message=f"{family_id!r} is not supported",
        field_path="family_id",
    )


def _identity_transitions(
    *,
    initial_state: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
) -> tuple[PersistentStateTransition, ...]:
    transitions: list[PersistentStateTransition] = []
    for transition_index, row in enumerate(training_rows):
        transitions.append(
            PersistentStateTransition(
                transition_index=transition_index,
                observation_index=transition_index,
                event_time=str(row["event_time"]),
                available_at=str(row["available_at"]),
                observed_value=_stable_float(float(row["target"])),
                state_before=dict(initial_state),
                state_after=dict(initial_state),
            )
        )
    return tuple(transitions)


def _parameter_mapping(
    candidate: CandidateIntermediateRepresentation,
) -> dict[str, float | int]:
    return {
        parameter.name: parameter.value
        for parameter in candidate.structural_layer.parameter_block.parameters
    }


def _literal_mapping(
    candidate: CandidateIntermediateRepresentation,
) -> dict[str, Any]:
    return {
        literal.name: literal.value
        for literal in candidate.structural_layer.literal_block.literals
    }


def _state_mapping(candidate: CandidateIntermediateRepresentation) -> dict[str, Any]:
    return {
        slot.name: slot.value
        for slot in candidate.structural_layer.state_signature.persistent_state.slots
    }


def _normalize_parameter_summary(
    parameter_summary: Mapping[str, float | int],
) -> dict[str, float | int]:
    normalized: dict[str, float | int] = {}
    for name, value in sorted(parameter_summary.items()):
        if isinstance(value, float):
            normalized[name] = _stable_float(value)
        else:
            normalized[name] = value
    return normalized


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


__all__ = [
    "CandidateFitArtifactBundle",
    "CandidateWindowFitResult",
    "PersistentStateTransition",
    "build_candidate_fit_artifacts",
    "fit_candidate_development_windows",
    "fit_candidate_window",
]
