from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from fractions import Fraction
from typing import Any, Mapping

from euclid.algorithmic_dsl import (
    evaluate_algorithmic_program,
    parse_algorithmic_program,
)
from euclid.cir.models import CandidateIntermediateRepresentation
from euclid.cir.normalize import require_full_cir_closure
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import PredictionArtifactManifest, PredictionRow
from euclid.modules.candidate_fitting import CandidateWindowFitResult
from euclid.modules.features import FeatureView
from euclid.modules.split_planning import (
    EvaluationPlan,
    EvaluationSegment,
    ScoredOrigin,
    resolve_scored_origin_target_row,
)
from euclid.reducers.composition import (
    AdditiveResidualComposition,
    PiecewiseComposition,
    RegimeConditionedComposition,
    SharedPlusLocalComposition,
    composition_runtime_signature,
    extract_component_mapping,
    resolve_regime_weights,
    select_piecewise_segment,
)
from euclid.runtime.hashing import sha256_digest


@dataclass(frozen=True)
class _StageRules:
    model_freeze_status: str
    refit_rule_applied: str
    segment_role: str | None
    include_outer_fold_id: bool


@dataclass(frozen=True)
class _ForecastPath:
    predictions: dict[int, float]
    runtime_evidence: dict[str, Any] | None = None


_STAGE_RULES = {
    "outer_test": _StageRules(
        model_freeze_status="fold_local_finalist_frozen",
        refit_rule_applied="outer_train_full_refit",
        segment_role="development",
        include_outer_fold_id=True,
    ),
    "confirmatory_holdout": _StageRules(
        model_freeze_status="global_finalist_frozen",
        refit_rule_applied="pre_holdout_development_refit",
        segment_role="confirmatory_holdout",
        include_outer_fold_id=False,
    ),
    "replication_holdout": _StageRules(
        model_freeze_status="replication_pair_frozen",
        refit_rule_applied="pre_replication_development_refit",
        segment_role=None,
        include_outer_fold_id=False,
    ),
}


def emit_point_prediction_artifact(
    *,
    catalog: ContractCatalog,
    feature_view: FeatureView,
    evaluation_plan: EvaluationPlan,
    evaluation_segment: EvaluationSegment,
    fit_result: CandidateWindowFitResult,
    score_policy_manifest: ManifestEnvelope,
    stage_id: str,
) -> ManifestEnvelope:
    closed_candidate = require_full_cir_closure(
        fit_result.fitted_candidate,
        consumer="evaluation",
    )
    legal_feature_view = feature_view.require_stage_reuse("evaluation")
    stage_rules = _resolve_stage_rules(stage_id, evaluation_segment)
    score_law_id = _score_law_id(score_policy_manifest)
    horizon_weights = _horizon_weights(score_policy_manifest)
    _require_matching_horizon_set(
        score_policy_manifest=score_policy_manifest,
        evaluation_segment=evaluation_segment,
    )
    scored_origin_panel = _segment_scored_origin_panel(
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_segment,
    )
    scored_origin_set_id = sha256_digest(
        [origin.as_dict() for origin in scored_origin_panel]
    )
    entity_panel = (
        evaluation_plan.entity_panel if len(evaluation_plan.entity_panel) > 1 else ()
    )
    entity_weights = _default_entity_weights(entity_panel)
    prediction_cache: dict[tuple[str, int], _ForecastPath] = {}

    rows: list[PredictionRow] = []
    missing_scored_origins: list[dict[str, Any]] = []
    timeguard_checks: list[dict[str, Any]] = []
    composition_runtime_rows: list[dict[str, Any]] = []
    for scored_origin in scored_origin_panel:
        origin_row = _origin_row_for_scored_origin(
            feature_view=legal_feature_view,
            scored_origin=scored_origin,
        )
        if origin_row is None:
            missing_scored_origins.append(
                {
                    "scored_origin_id": scored_origin.scored_origin_id,
                    "horizon": scored_origin.horizon,
                    "reason_code": "missing_origin_row",
                }
            )
            continue
        prediction_key = (
            str(origin_row.get("entity", legal_feature_view.series_id)),
            scored_origin.origin_index,
        )
        forecast_path = prediction_cache.get(prediction_key)
        if forecast_path is None:
            forecast_path = _forecast_path(
                candidate=closed_candidate,
                fit_result=fit_result,
                origin_row=origin_row,
                max_horizon=max(evaluation_segment.horizon_set),
                entity=scored_origin.entity,
            )
            prediction_cache[prediction_key] = forecast_path
        if forecast_path.runtime_evidence is not None:
            composition_runtime_rows.append(
                {
                    "scored_origin_id": scored_origin.scored_origin_id,
                    **forecast_path.runtime_evidence,
                }
            )
        target_row = _target_row_for_origin(
            feature_view=legal_feature_view,
            scored_origin=scored_origin,
        )
        if target_row is None:
            missing_scored_origins.append(
                {
                    "scored_origin_id": scored_origin.scored_origin_id,
                    "horizon": scored_origin.horizon,
                    "reason_code": "missing_target_row",
                }
            )
            timeguard_checks.append(
                {
                    "scored_origin_id": scored_origin.scored_origin_id,
                    "expected_available_at": scored_origin.available_at,
                    "observed_available_at": None,
                    "status": "failed",
                }
            )
            continue

        observed_available_at = str(target_row["available_at"])
        timeguard_ok = observed_available_at == scored_origin.available_at
        timeguard_checks.append(
            {
                "scored_origin_id": scored_origin.scored_origin_id,
                "expected_available_at": scored_origin.available_at,
                "observed_available_at": observed_available_at,
                "status": "passed" if timeguard_ok else "failed",
            }
        )
        if not timeguard_ok:
            missing_scored_origins.append(
                {
                    "scored_origin_id": scored_origin.scored_origin_id,
                    "horizon": scored_origin.horizon,
                    "reason_code": "non_time_safe_prediction",
                    "expected_available_at": scored_origin.available_at,
                    "observed_available_at": observed_available_at,
                }
            )
            continue

        point_forecast = forecast_path.predictions.get(scored_origin.horizon)
        if point_forecast is None:
            missing_scored_origins.append(
                {
                    "scored_origin_id": scored_origin.scored_origin_id,
                    "horizon": scored_origin.horizon,
                    "reason_code": "missing_declared_horizon_forecast",
                }
            )
            continue
        if not math.isfinite(point_forecast):
            missing_scored_origins.append(
                {
                    "scored_origin_id": scored_origin.scored_origin_id,
                    "horizon": scored_origin.horizon,
                    "reason_code": "nonfinite_point_forecast",
                }
            )
            continue

        realized_observation = float(target_row["target"])
        if not math.isfinite(realized_observation):
            missing_scored_origins.append(
                {
                    "scored_origin_id": scored_origin.scored_origin_id,
                    "horizon": scored_origin.horizon,
                    "reason_code": "nonfinite_observation",
                }
            )
            continue

        rows.append(
            PredictionRow(
                entity=scored_origin.entity,
                origin_time=scored_origin.origin_time,
                available_at=scored_origin.available_at,
                horizon=scored_origin.horizon,
                point_forecast=_stable_float(point_forecast),
                realized_observation=_stable_float(realized_observation),
            )
        )

    comparison_key = {
        "forecast_object_type": "point",
        "horizon_set": list(evaluation_segment.horizon_set),
        "score_law_id": score_law_id,
        "scored_origin_set_id": scored_origin_set_id,
    }
    if entity_panel:
        comparison_key["entity_panel"] = list(entity_panel)
        comparison_key["entity_weights"] = [dict(item) for item in entity_weights]
    composition_signature = composition_runtime_signature(
        closed_candidate.structural_layer.composition_graph
    )
    if composition_signature is not None:
        comparison_key["composition_signature"] = composition_signature
    prediction_artifact = PredictionArtifactManifest(
        prediction_artifact_id=(
            f"{fit_result.candidate_id}__{stage_id}__{evaluation_segment.segment_id}"
        ),
        candidate_id=fit_result.candidate_id,
        stage_id=stage_id,
        outer_fold_id=(
            evaluation_segment.outer_fold_id
            if stage_rules.include_outer_fold_id
            else None
        ),
        fit_window_id=fit_result.fit_window_id,
        test_window_id=evaluation_segment.segment_id,
        model_freeze_status=stage_rules.model_freeze_status,
        refit_rule_applied=stage_rules.refit_rule_applied,
        score_policy_ref=score_policy_manifest.ref,
        rows=tuple(rows),
        score_law_id=score_law_id,
        horizon_weights=tuple(horizon_weights),
        entity_panel=entity_panel,
        entity_weights=entity_weights,
        scored_origin_panel=tuple(origin.as_dict() for origin in scored_origin_panel),
        scored_origin_set_id=scored_origin_set_id,
        comparison_key=comparison_key,
        missing_scored_origins=tuple(missing_scored_origins),
        timeguard_checks=tuple(timeguard_checks),
        composition_graph=(
            closed_candidate.structural_layer.composition_graph.as_dict()
            if closed_candidate.structural_layer.composition_graph.operator_id
            is not None
            else None
        ),
        composition_runtime_evidence=(
            {"scored_origins": composition_runtime_rows}
            if composition_runtime_rows
            else None
        ),
    )
    return prediction_artifact.to_manifest(catalog)


def _resolve_stage_rules(
    stage_id: str, evaluation_segment: EvaluationSegment
) -> _StageRules:
    try:
        stage_rules = _STAGE_RULES[stage_id]
    except KeyError as exc:
        raise ContractValidationError(
            code="invalid_prediction_stage",
            message=f"{stage_id!r} is not a legal prediction stage",
            field_path="stage_id",
        ) from exc
    if (
        stage_rules.segment_role is not None
        and evaluation_segment.role != stage_rules.segment_role
    ):
        raise ContractValidationError(
            code="invalid_prediction_stage_segment",
            message=(
                "evaluation segment role does not match the requested prediction stage"
            ),
            field_path="evaluation_segment.role",
            details={
                "expected_role": stage_rules.segment_role,
                "actual_role": evaluation_segment.role,
                "stage_id": stage_id,
            },
        )
    return stage_rules


def _score_law_id(score_policy_manifest: ManifestEnvelope) -> str:
    if score_policy_manifest.schema_name != "point_score_policy_manifest@1.0.0":
        raise ContractValidationError(
            code="malformed_score_policy_ref",
            message="prediction artifacts require a point-score policy manifest",
            field_path="score_policy_manifest.schema_name",
            details={"schema_name": score_policy_manifest.schema_name},
        )
    return str(score_policy_manifest.body["point_loss_id"])


def _horizon_weights(
    score_policy_manifest: ManifestEnvelope,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(weight) for weight in score_policy_manifest.body["horizon_weights"]
    )


def _require_matching_horizon_set(
    *,
    score_policy_manifest: ManifestEnvelope,
    evaluation_segment: EvaluationSegment,
) -> None:
    declared_horizon_set = tuple(
        int(weight["horizon"])
        for weight in score_policy_manifest.body["horizon_weights"]
    )
    if declared_horizon_set != evaluation_segment.horizon_set:
        raise ContractValidationError(
            code="prediction_horizon_mismatch",
            message=(
                "prediction artifacts require the score-policy horizon set to "
                "match the frozen evaluation segment"
            ),
            field_path="score_policy_manifest.body.horizon_weights",
            details={
                "score_policy_horizon_set": list(declared_horizon_set),
                "evaluation_segment_horizon_set": list(evaluation_segment.horizon_set),
            },
        )


def _segment_scored_origin_panel(
    *,
    evaluation_plan: EvaluationPlan,
    evaluation_segment: EvaluationSegment,
) -> tuple[ScoredOrigin, ...]:
    origins_by_id = {
        origin.scored_origin_id: origin
        for origin in evaluation_plan.scored_origin_panel
    }
    return tuple(
        origins_by_id[origin_id] for origin_id in evaluation_segment.scored_origin_ids
    )


def _target_row_for_origin(
    *,
    feature_view: FeatureView,
    scored_origin: ScoredOrigin,
) -> Mapping[str, Any] | None:
    return resolve_scored_origin_target_row(
        feature_view=feature_view,
        scored_origin=scored_origin,
    )


def _origin_row_for_scored_origin(
    *,
    feature_view: FeatureView,
    scored_origin: ScoredOrigin,
) -> Mapping[str, Any] | None:
    entity = scored_origin.entity or feature_view.series_id
    entity_rows = _rows_by_entity(feature_view).get(entity)
    if entity_rows is None or scored_origin.origin_index >= len(entity_rows):
        return None
    return entity_rows[scored_origin.origin_index]


def _rows_by_entity(
    feature_view: FeatureView,
) -> dict[str, tuple[dict[str, Any], ...]]:
    rows_by_entity: dict[str, list[dict[str, Any]]] = {}
    for row in feature_view.rows:
        entity = str(row.get("entity", feature_view.series_id))
        rows_by_entity.setdefault(entity, []).append(dict(row))
    return {entity: tuple(rows) for entity, rows in rows_by_entity.items()}


def _forecast_path(
    *,
    candidate: CandidateIntermediateRepresentation,
    fit_result: CandidateWindowFitResult,
    origin_row: Mapping[str, Any],
    max_horizon: int,
    entity: str | None = None,
) -> _ForecastPath:
    family_id = candidate.structural_layer.cir_family_id
    parameters = {
        parameter.name: parameter.value
        for parameter in candidate.structural_layer.parameter_block.parameters
    }
    literals = {
        literal.name: literal.value
        for literal in candidate.structural_layer.literal_block.literals
    }
    state = dict(fit_result.final_state)
    operator_id = candidate.structural_layer.composition_graph.operator_id
    if operator_id == "shared_plus_local_decomposition":
        return _ForecastPath(
            predictions=_shared_local_forecast_path(
                candidate=candidate,
                parameters=parameters,
                entity=entity,
                origin_row=origin_row,
                max_horizon=max_horizon,
            )
        )
    if operator_id == "piecewise":
        return _piecewise_forecast_path(
            candidate=candidate,
            parameters=parameters,
            origin_row=origin_row,
            max_horizon=max_horizon,
        )
    if operator_id == "additive_residual":
        return _additive_residual_forecast_path(
            candidate=candidate,
            parameters=parameters,
            origin_row=origin_row,
            max_horizon=max_horizon,
        )
    if operator_id == "regime_conditioned":
        return _regime_conditioned_forecast_path(
            candidate=candidate,
            parameters=parameters,
            origin_row=origin_row,
            max_horizon=max_horizon,
        )

    if family_id == "analytic":
        return _ForecastPath(
            predictions=_analytic_forecast_path(
                parameters=parameters,
                origin_row=origin_row,
                max_horizon=max_horizon,
            )
        )
    if family_id == "recursive":
        return _ForecastPath(
            predictions=_recursive_forecast_path(state=state, max_horizon=max_horizon)
        )
    if family_id == "spectral":
        return _ForecastPath(
            predictions=_spectral_forecast_path(
                parameters=parameters,
                literals=literals,
                state=state,
                max_horizon=max_horizon,
            )
        )
    if family_id == "algorithmic":
        return _ForecastPath(
            predictions=_algorithmic_forecast_path(
                candidate=candidate,
                literals=literals,
                state=state,
                origin_row=origin_row,
                max_horizon=max_horizon,
            )
        )
    raise ContractValidationError(
        code="unsupported_prediction_candidate",
        message=(
            "point prediction artifacts support analytic, recursive, spectral, "
            "and algorithmic fitted candidates only"
        ),
        field_path="candidate.structural_layer.cir_family_id",
        details={"family_id": family_id},
    )


def _piecewise_forecast_path(
    *,
    candidate: CandidateIntermediateRepresentation,
    parameters: Mapping[str, float | int],
    origin_row: Mapping[str, Any],
    max_horizon: int,
) -> _ForecastPath:
    composition = candidate.structural_layer.composition_graph.composition
    assert isinstance(composition, PiecewiseComposition)
    previous_value = float(origin_row["target"])
    predictions: dict[int, float] = {}
    horizon_trace: list[dict[str, Any]] = []
    for horizon in range(1, max_horizon + 1):
        segment, evidence = select_piecewise_segment(
            candidate.structural_layer.composition_graph,
            row=origin_row,
        )
        branch_parameters = extract_component_mapping(parameters, segment.reducer_id)
        forecast = _analytic_next_forecast(
            parameters=branch_parameters,
            previous_value=previous_value,
        )
        predictions[horizon] = forecast
        horizon_trace.append(
            {
                "horizon": horizon,
                "selected_branch_id": segment.reducer_id,
                "partition_value": evidence["partition_value"],
                "point_forecast": forecast,
            }
        )
        previous_value = forecast
    return _ForecastPath(
        predictions=predictions,
        runtime_evidence={
            "operator_id": "piecewise",
            "signal_field": "piecewise_partition_value",
            "horizon_trace": horizon_trace,
        },
    )


def _additive_residual_forecast_path(
    *,
    candidate: CandidateIntermediateRepresentation,
    parameters: Mapping[str, float | int],
    origin_row: Mapping[str, Any],
    max_horizon: int,
) -> _ForecastPath:
    composition = candidate.structural_layer.composition_graph.composition
    assert isinstance(composition, AdditiveResidualComposition)
    base_predictions = _analytic_forecast_path(
        parameters=extract_component_mapping(parameters, composition.base_reducer),
        origin_row=origin_row,
        max_horizon=max_horizon,
    )
    residual_origin = dict(origin_row)
    residual_origin["target"] = float(origin_row.get("residual_lag_1", 0.0))
    residual_predictions = _analytic_forecast_path(
        parameters=extract_component_mapping(
            parameters,
            composition.residual_reducer,
        ),
        origin_row=residual_origin,
        max_horizon=max_horizon,
    )
    combined_predictions = {
        horizon: _stable_float(
            base_predictions[horizon] + residual_predictions[horizon]
        )
        for horizon in range(1, max_horizon + 1)
    }
    return _ForecastPath(
        predictions=combined_predictions,
        runtime_evidence={
            "operator_id": "additive_residual",
            "base_reducer": composition.base_reducer,
            "residual_reducer": composition.residual_reducer,
            "horizon_trace": [
                {
                    "horizon": horizon,
                    "base_prediction": base_predictions[horizon],
                    "residual_prediction": residual_predictions[horizon],
                    "point_forecast": combined_predictions[horizon],
                }
                for horizon in range(1, max_horizon + 1)
            ],
        },
    )


def _regime_conditioned_forecast_path(
    *,
    candidate: CandidateIntermediateRepresentation,
    parameters: Mapping[str, float | int],
    origin_row: Mapping[str, Any],
    max_horizon: int,
) -> _ForecastPath:
    composition = candidate.structural_layer.composition_graph.composition
    assert isinstance(composition, RegimeConditionedComposition)
    branch_weights, evidence = resolve_regime_weights(
        candidate.structural_layer.composition_graph,
        row=origin_row,
    )
    branch_predictions = {
        branch.reducer_id: _analytic_forecast_path(
            parameters=extract_component_mapping(parameters, branch.reducer_id),
            origin_row=origin_row,
            max_horizon=max_horizon,
        )
        for branch in composition.branch_reducers
        if branch.reducer_id in branch_weights
    }
    predictions = {
        horizon: _stable_float(
            sum(
                weight * branch_predictions[branch_id][horizon]
                for branch_id, weight in branch_weights.items()
            )
        )
        for horizon in range(1, max_horizon + 1)
    }
    return _ForecastPath(
        predictions=predictions,
        runtime_evidence={
            **evidence,
            "horizon_trace": [
                {
                    "horizon": horizon,
                    "point_forecast": predictions[horizon],
                    "branch_predictions": {
                        branch_id: branch_predictions[branch_id][horizon]
                        for branch_id in branch_weights
                    },
                }
                for horizon in range(1, max_horizon + 1)
            ],
        },
    )


def _analytic_next_forecast(
    *,
    parameters: Mapping[str, float | int],
    previous_value: float,
) -> float:
    intercept = float(parameters.get("intercept", 0.0))
    if "lag_coefficient" not in parameters:
        return _stable_float(intercept)
    return _stable_float(
        intercept + (float(parameters["lag_coefficient"]) * previous_value)
    )


def _shared_local_forecast_path(
    *,
    candidate: CandidateIntermediateRepresentation,
    parameters: Mapping[str, float | int],
    entity: str | None,
    origin_row: Mapping[str, Any],
    max_horizon: int,
) -> dict[int, float]:
    if entity is None:
        raise ContractValidationError(
            code="invalid_shared_local_entity",
            message=(
                "shared-plus-local forecasts require an entity-scoped "
                "scored origin"
            ),
            field_path="scored_origin.entity",
        )
    composition = candidate.structural_layer.composition_graph.composition
    if not isinstance(composition, SharedPlusLocalComposition):
        raise ContractValidationError(
            code="invalid_shared_local_entity",
            message="shared-plus-local forecasts require declared entity panel metadata",
            field_path="candidate.structural_layer.composition_graph",
        )
    if entity not in composition.entity_index_set:
        raise ContractValidationError(
            code="unseen_entity_rule_violation",
            message=(
                "shared-plus-local forecasts may not score entities outside the "
                "declared panel"
            ),
            field_path="scored_origin.entity",
            details={
                "entity": entity,
                "entity_panel": list(composition.entity_index_set),
                "unseen_entity_rule": getattr(composition, "unseen_entity_rule", None),
            },
        )
    shared_intercept = float(parameters.get("shared_intercept", 0.0))
    adjustment_key = f"local_adjustment__{entity}"
    if adjustment_key not in parameters:
        raise ContractValidationError(
            code="unseen_entity_rule_violation",
            message=(
                "shared-plus-local forecasts require fitted local parameters for "
                "every declared entity"
            ),
            field_path="fit_result.parameter_summary",
            details={"missing_parameter": adjustment_key, "entity": entity},
        )
    local_adjustment = float(parameters[adjustment_key])
    shared_lag_key = "shared_lag_coefficient"
    local_lag_adjustment_key = f"local_lag_adjustment__{entity}"
    lag_key = f"local_lag_coefficient__{entity}"
    if (
        shared_lag_key not in parameters
        and local_lag_adjustment_key not in parameters
        and lag_key not in parameters
    ):
        forecast = _stable_float(shared_intercept + local_adjustment)
        return {horizon: forecast for horizon in range(1, max_horizon + 1)}

    lag_coefficient = float(parameters.get(shared_lag_key, 0.0)) + float(
        parameters.get(local_lag_adjustment_key, 0.0)
    )
    if lag_key in parameters:
        lag_coefficient = float(parameters[lag_key])
    previous_value = float(origin_row["target"])
    predictions: dict[int, float] = {}
    for horizon in range(1, max_horizon + 1):
        forecast = (shared_intercept + local_adjustment) + (
            lag_coefficient * previous_value
        )
        predictions[horizon] = _stable_float(forecast)
        previous_value = forecast
    return predictions


def _analytic_forecast_path(
    *,
    parameters: Mapping[str, float | int],
    origin_row: Mapping[str, Any],
    max_horizon: int,
) -> dict[int, float]:
    intercept = float(parameters.get("intercept", 0.0))
    if "lag_coefficient" not in parameters:
        return {
            horizon: _stable_float(intercept) for horizon in range(1, max_horizon + 1)
        }
    lag_coefficient = float(parameters["lag_coefficient"])
    previous_value = float(origin_row["target"])
    predictions: dict[int, float] = {}
    for horizon in range(1, max_horizon + 1):
        forecast = intercept + (lag_coefficient * previous_value)
        predictions[horizon] = _stable_float(forecast)
        previous_value = forecast
    return predictions


def _recursive_forecast_path(
    *,
    state: Mapping[str, Any],
    max_horizon: int,
) -> dict[int, float]:
    if "level" in state:
        forecast = _stable_float(float(state["level"]))
        return {horizon: forecast for horizon in range(1, max_horizon + 1)}
    if "running_mean" in state:
        forecast = _stable_float(float(state["running_mean"]))
        return {horizon: forecast for horizon in range(1, max_horizon + 1)}
    raise ContractValidationError(
        code="invalid_recursive_candidate_state",
        message="recursive point prediction requires supported fitted state slots",
        field_path="fit_result.final_state",
        details={"state_topology": sorted(state)},
    )


def _spectral_forecast_path(
    *,
    parameters: Mapping[str, float | int],
    literals: Mapping[str, float | int],
    state: Mapping[str, Any],
    max_horizon: int,
) -> dict[int, float]:
    harmonic = int(literals["harmonic"])
    season_length = int(literals["season_length"])
    phase_index = int(state.get("phase_index", 0))
    cosine = float(parameters.get("cosine_coefficient", 0.0))
    sine = float(parameters.get("sine_coefficient", 0.0))
    predictions: dict[int, float] = {}
    for horizon in range(1, max_horizon + 1):
        angle = (2.0 * math.pi * harmonic * phase_index) / season_length
        predictions[horizon] = _stable_float(
            (cosine * math.cos(angle)) + (sine * math.sin(angle))
        )
        phase_index = (phase_index + 1) % season_length
    return predictions


def _algorithmic_forecast_path(
    *,
    candidate: CandidateIntermediateRepresentation,
    literals: Mapping[str, Any],
    state: Mapping[str, Any],
    origin_row: Mapping[str, Any],
    max_horizon: int,
) -> dict[int, float]:
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
    state_tuple = tuple(
        Fraction(str(state[f"state_{index}"]))
        for index in range(program.state_slot_count)
    )
    predictions: dict[int, float] = {}
    current_state = state_tuple
    observation_window = _algorithmic_observation_window(
        origin_row=origin_row,
        max_lag=max_lag,
        current_value=float(origin_row["target"]),
    )
    for horizon in range(1, max_horizon + 1):
        step = evaluate_algorithmic_program(
            program,
            state=current_state,
            observation=observation_window,
        )
        predictions[horizon] = _stable_float(float(step.emit_value))
        current_state = step.next_state
        observation_window = _algorithmic_shifted_observation_window(
            observation_window,
            next_observation=predictions[horizon],
        )
    return predictions


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
    origin_row: Mapping[str, Any],
    max_lag: int,
    current_value: float,
) -> tuple[Fraction, ...]:
    values: list[Fraction] = [Fraction(str(float(current_value)))]
    for lag in range(1, max_lag + 1):
        field_name = f"lag_{lag}"
        if field_name not in origin_row:
            raise ContractValidationError(
                code="unsupported_prediction_candidate",
                message=(
                    "algorithmic lagged observation access requires explicit "
                    f"{field_name} features on the forecast origin"
                ),
                field_path=f"origin_row.{field_name}",
            )
        values.append(Fraction(str(float(origin_row[field_name]))))
    return tuple(values)


def _algorithmic_shifted_observation_window(
    observation_window: tuple[Fraction, ...],
    *,
    next_observation: float,
) -> tuple[Fraction, ...]:
    if len(observation_window) == 1:
        return (Fraction(str(float(next_observation))),)
    return (
        Fraction(str(float(next_observation))),
        *observation_window[:-1],
    )


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


def _default_entity_weights(
    entity_panel: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    if not entity_panel:
        return ()
    scale = Decimal("0.000000000001")
    count = Decimal(len(entity_panel))
    base_weight = (Decimal("1") / count).quantize(scale, rounding=ROUND_DOWN)
    remaining = Decimal("1")
    weights: list[dict[str, str]] = []
    for index, entity in enumerate(entity_panel):
        if index == len(entity_panel) - 1:
            weight = remaining
        else:
            weight = base_weight
            remaining -= weight
        weights.append({"entity": entity, "weight": _decimal_string(weight)})
    return tuple(weights)


def _decimal_string(value: Decimal) -> str:
    rendered = format(value.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


__all__ = ["emit_point_prediction_artifact"]
