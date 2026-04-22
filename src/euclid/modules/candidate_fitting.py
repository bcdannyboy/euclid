from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

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
from euclid.fit.parameterization import ParameterDeclaration
from euclid.fit.refit import FitDataSplit, fit_cir_candidate
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    CandidateSpecManifest,
    CandidateStateManifest,
    SearchPlanManifest,
)
from euclid.modules.features import FeatureView
from euclid.modules.shared_plus_local_decomposition import (
    fit_shared_plus_local_decomposition,
)
from euclid.modules.split_planning import (
    EvaluationPlan,
    EvaluationSegment,
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

_SCOPE_ID = "euclid_v1_binding_scope@1.0.0"


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

    @property
    def state_transition_count(self) -> int:
        return len(self.state_transitions)


@dataclass(frozen=True)
class CandidateFitArtifactBundle:
    candidate_spec: ManifestEnvelope
    candidate_state: ManifestEnvelope
    reducer_artifact: ManifestEnvelope


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


def fit_candidate_window(
    *,
    candidate: CandidateIntermediateRepresentation,
    feature_view: FeatureView,
    fit_window: EvaluationSegment,
    search_plan: SearchPlanManifest,
    stage_id: str = "inner_search",
    shared_local_backend_preference: str | None = None,
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

    family_fit = _fit_family(
        candidate=closed_candidate,
        training_rows=training_rows,
        random_seed=search_plan.random_seed,
        shared_local_backend_preference=shared_local_backend_preference,
    )
    fitted_candidate = _build_fitted_candidate(
        candidate=closed_candidate,
        parameter_summary=family_fit.parameter_summary,
        updated_literals=family_fit.updated_literals,
        final_state=family_fit.final_state,
        backend_id=family_fit.backend_id,
        fit_window_id=fit_window.segment_id,
        optimizer_diagnostics={
            "backend_id": family_fit.backend_id,
            "objective_id": family_fit.objective_id,
            "seed_value": search_plan.random_seed,
            "converged": family_fit.converged,
            "iteration_count": family_fit.iteration_count,
            "final_loss": family_fit.final_loss,
        },
        component_diagnostics=family_fit.component_diagnostics,
    )
    optimizer_diagnostics = {
        "backend_id": family_fit.backend_id,
        "objective_id": family_fit.objective_id,
        "seed_value": search_plan.random_seed,
        "converged": family_fit.converged,
        "iteration_count": family_fit.iteration_count,
        "final_loss": family_fit.final_loss,
    }
    if family_fit.component_diagnostics:
        if family_fit.backend_id == "euclid_unified_fit_layer_v1":
            optimizer_diagnostics.update(dict(family_fit.component_diagnostics))
        else:
            optimizer_diagnostics["composition_runtime"] = dict(
                family_fit.component_diagnostics
            )
    candidate_id = (
        closed_candidate.evidence_layer.backend_origin_record.source_candidate_id
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
    )


def _fit_family(
    *,
    candidate: CandidateIntermediateRepresentation,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
    shared_local_backend_preference: str | None = None,
) -> _FamilyFitResult:
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
    if "lag_coefficient" in parameters:
        x_values = tuple(float(row["lag_1"]) for row in training_rows)
        x_mean = sum(
            weight * x_value
            for weight, x_value in zip(weight_values, x_values, strict=True)
        ) / total_weight
        y_mean = sum(
            weight * target
            for weight, target in zip(weight_values, targets, strict=True)
        ) / total_weight
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
        )
    mean_value = sum(
        weight * target
        for weight, target in zip(weight_values, targets, strict=True)
    ) / total_weight
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
    )


def _analytic_fitted_values_from_rows(
    *,
    parameters: Mapping[str, float | int],
    rows: Sequence[Mapping[str, Any]],
    lag_key: str = "lag_1",
) -> tuple[float, ...]:
    intercept = float(parameters.get("intercept", 0.0))
    if "lag_coefficient" not in parameters:
        return tuple(_stable_float(intercept) for _ in rows)
    if any(lag_key not in row for row in rows):
        raise ContractValidationError(
            code="unsupported_candidate_family",
            message="analytic affine fits require lag_1 on every training row",
            field_path=f"rows.{lag_key}",
        )
    lag_coefficient = float(parameters["lag_coefficient"])
    return tuple(
        _stable_float(intercept + (lag_coefficient * float(row[lag_key])))
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
    harmonic = int(literals["harmonic"])
    season_length = int(literals["season_length"])
    phase_index = int(initial_state.get("phase_index", 0))
    design_rows: list[tuple[float, float]] = []
    targets = tuple(float(row["target"]) for row in training_rows)
    for _ in training_rows:
        angle = (2.0 * math.pi * harmonic * phase_index) / season_length
        design_rows.append((math.cos(angle), math.sin(angle)))
        phase_index = (phase_index + 1) % season_length

    sum_cc = sum(cosine * cosine for cosine, _ in design_rows)
    sum_ss = sum(sine * sine for _, sine in design_rows)
    sum_cs = sum(cosine * sine for cosine, sine in design_rows)
    sum_cy = sum(
        cosine * target
        for (cosine, _), target in zip(design_rows, targets, strict=True)
    )
    sum_sy = sum(
        sine * target for (_, sine), target in zip(design_rows, targets, strict=True)
    )
    determinant = (sum_cc * sum_ss) - (sum_cs * sum_cs)
    if math.isclose(determinant, 0.0, rel_tol=0.0, abs_tol=1e-12):
        cosine_coefficient = 0.0
        sine_coefficient = 0.0
        converged = False
    else:
        cosine_coefficient = ((sum_cy * sum_ss) - (sum_sy * sum_cs)) / determinant
        sine_coefficient = ((sum_sy * sum_cc) - (sum_cy * sum_cs)) / determinant
        converged = True

    last_basis_value = float(initial_state.get("last_basis_value", 0.0))
    phase_index = int(initial_state.get("phase_index", 0))
    transitions: list[PersistentStateTransition] = []
    fitted_values: list[float] = []
    for transition_index, row in enumerate(training_rows):
        angle = (2.0 * math.pi * harmonic * phase_index) / season_length
        fitted = (cosine_coefficient * math.cos(angle)) + (
            sine_coefficient * math.sin(angle)
        )
        fitted_values.append(fitted)
        state_before = {
            "last_basis_value": _stable_float(last_basis_value),
            "phase_index": phase_index,
        }
        phase_index = (phase_index + 1) % season_length
        next_angle = (2.0 * math.pi * harmonic * phase_index) / season_length
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
        objective_id="least_squares_harmonic_basis_v1",
        parameter_summary={
            "cosine_coefficient": _stable_float(cosine_coefficient),
            "sine_coefficient": _stable_float(sine_coefficient),
        },
        updated_literals={},
        initial_state=initial_state,
        final_state=final_state,
        state_transitions=tuple(transitions),
        converged=converged,
        iteration_count=1,
        final_loss=_stable_float(final_loss),
        component_diagnostics={},
    )


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
        return tuple(
            int(token)
            for token in raw_lags.split(",")
            if token.strip()
        )
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
        harmonic = int(literals["harmonic"])

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
                Fraction(str(float(history[-1 - lag])))
                if len(history) > lag
                else Fraction(0, 1)
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
