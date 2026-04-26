from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from scipy import stats

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    DistributionPredictionRow,
    EventProbabilityPredictionRow,
    IntervalPredictionRow,
    PredictionArtifactManifest,
    QuantilePredictionRow,
    QuantileValue,
    StochasticModelManifest,
)
from euclid.modules.candidate_fitting import CandidateWindowFitResult
from euclid.modules.evaluation import (
    _default_entity_weights,
    _horizon_weights,
    _origin_row_for_scored_origin,
    _require_matching_horizon_set,
    _resolve_stage_rules,
    _segment_scored_origin_panel,
    _stable_float,
    _target_row_for_origin,
)
from euclid.modules.features import FeatureView
from euclid.modules.forecast_paths import forecast_path as build_forecast_path
from euclid.modules.split_planning import EvaluationPlan, EvaluationSegment
from euclid.runtime.hashing import sha256_digest
from euclid.stochastic.event_definitions import EventDefinition
from euclid.stochastic.observation_models import get_observation_model
from euclid.stochastic.process_models import (
    StochasticPredictiveSupport,
    fit_residual_stochastic_model,
)

_DISTRIBUTION_POLICY_SCHEMA = "probabilistic_score_policy_manifest@1.0.0"
_INTERVAL_POLICY_SCHEMA = "interval_score_policy_manifest@1.0.0"
_QUANTILE_POLICY_SCHEMA = "quantile_score_policy_manifest@1.0.0"
_EVENT_POLICY_SCHEMA = "event_probability_score_policy_manifest@1.0.0"

_POLICY_SCHEMAS = {
    "distribution": _DISTRIBUTION_POLICY_SCHEMA,
    "interval": _INTERVAL_POLICY_SCHEMA,
    "quantile": _QUANTILE_POLICY_SCHEMA,
    "event_probability": _EVENT_POLICY_SCHEMA,
}

_QUANTILE_LEVELS = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class _ProbabilisticScorePolicy:
    forecast_object_type: str
    score_law_id: str
    horizon_weights: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _StochasticPredictiveSupport:
    location: float
    scale: float
    distribution_family: str = "gaussian_location_scale"
    support_kind: str = "all_real"
    distribution_parameters: Mapping[str, float] | None = None


@dataclass(frozen=True)
class _ProbabilisticSupportBundle:
    support_path: dict[int, _StochasticPredictiveSupport]
    stochastic_support_status: str
    stochastic_support_reason_codes: tuple[str, ...]
    residual_history_refs: tuple[Any, ...] = ()
    stochastic_model_refs: tuple[Any, ...] = ()


def emit_probabilistic_prediction_artifact(
    *,
    catalog: ContractCatalog,
    feature_view: FeatureView,
    evaluation_plan: EvaluationPlan,
    evaluation_segment: EvaluationSegment,
    fit_result: CandidateWindowFitResult,
    score_policy_manifest: ManifestEnvelope,
    stage_id: str,
    forecast_object_type: str,
    stochastic_evidence_mode: str = "compatibility",
    stochastic_fit_result: CandidateWindowFitResult | None = None,
    residual_history_ref: TypedRef | None = None,
    stochastic_family_id: str = "gaussian",
    student_t_degrees_of_freedom: float | None = None,
    interval_levels: Sequence[float] = (0.8,),
    quantile_levels: Sequence[float] = _QUANTILE_LEVELS,
    supporting_artifact_sink: list[ManifestEnvelope] | None = None,
) -> ManifestEnvelope:
    legal_feature_view = feature_view.require_stage_reuse("evaluation")
    stage_rules = _resolve_stage_rules(stage_id, evaluation_segment)
    score_policy = _resolve_probabilistic_score_policy(
        score_policy_manifest=score_policy_manifest,
        forecast_object_type=forecast_object_type,
    )
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
    resolved_interval_levels = _resolve_probability_levels(
        interval_levels,
        default=(0.8,),
        field_path="interval_levels",
    )
    resolved_quantile_levels = _resolve_probability_levels(
        quantile_levels,
        default=_QUANTILE_LEVELS,
        field_path="quantile_levels",
    )
    support_cache: dict[tuple[str, int], _ProbabilisticSupportBundle] = {}

    rows = []
    missing_scored_origins: list[dict[str, Any]] = []
    timeguard_checks: list[dict[str, Any]] = []
    stochastic_support_statuses: list[str] = []
    stochastic_support_reason_codes: list[str] = []
    residual_history_refs = []
    stochastic_model_refs = []
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
        support_key = (
            str(origin_row.get("entity", legal_feature_view.series_id)),
            scored_origin.origin_index,
        )
        support_bundle = support_cache.get(support_key)
        if support_bundle is None:
            forecast_path = build_forecast_path(
                candidate=fit_result.fitted_candidate,
                fit_result=fit_result,
                origin_row=origin_row,
                max_horizon=max(evaluation_segment.horizon_set),
                entity=scored_origin.entity,
            )
            support_bundle = _probabilistic_support_path(
                catalog=catalog,
                fit_result=fit_result,
                point_path=forecast_path.predictions,
                stochastic_evidence_mode=stochastic_evidence_mode,
                stochastic_fit_result=stochastic_fit_result,
                residual_history_ref=residual_history_ref,
                stochastic_family_id=stochastic_family_id,
                student_t_degrees_of_freedom=student_t_degrees_of_freedom,
                supporting_artifact_sink=supporting_artifact_sink,
            )
            support_cache[support_key] = support_bundle
            stochastic_support_statuses.append(
                support_bundle.stochastic_support_status
            )
            stochastic_support_reason_codes.extend(
                support_bundle.stochastic_support_reason_codes
            )
            residual_history_refs.extend(support_bundle.residual_history_refs)
            stochastic_model_refs.extend(support_bundle.stochastic_model_refs)
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

        support = support_bundle.support_path.get(scored_origin.horizon)
        if support is None:
            missing_scored_origins.append(
                {
                    "scored_origin_id": scored_origin.scored_origin_id,
                    "horizon": scored_origin.horizon,
                    "reason_code": "missing_declared_horizon_forecast",
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
            _build_row(
                forecast_object_type=forecast_object_type,
                scored_origin=scored_origin,
                support=support,
                realized_observation=_stable_float(realized_observation),
                origin_row=origin_row,
                entity=scored_origin.entity,
                interval_levels=resolved_interval_levels,
                quantile_levels=resolved_quantile_levels,
            )
        )

    prediction_artifact = PredictionArtifactManifest(
        prediction_artifact_id=(
            f"{fit_result.candidate_id}__{stage_id}__{evaluation_segment.segment_id}"
            f"__{forecast_object_type}"
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
        forecast_object_type=forecast_object_type,
        score_law_id=score_policy.score_law_id,
        horizon_weights=score_policy.horizon_weights,
        entity_panel=entity_panel,
        entity_weights=entity_weights,
        scored_origin_panel=tuple(origin.as_dict() for origin in scored_origin_panel),
        scored_origin_set_id=scored_origin_set_id,
        comparison_key={
            "forecast_object_type": forecast_object_type,
            "horizon_set": list(evaluation_segment.horizon_set),
            "score_law_id": score_policy.score_law_id,
            "scored_origin_set_id": scored_origin_set_id,
            **(
                {
                    "entity_panel": list(entity_panel),
                    "entity_weights": [dict(item) for item in entity_weights],
                }
                if entity_panel
                else {}
            ),
        },
        missing_scored_origins=tuple(missing_scored_origins),
        timeguard_checks=tuple(timeguard_checks),
        stochastic_support_status=_combined_stochastic_support_status(
            stochastic_support_statuses
        ),
        stochastic_support_reason_codes=_unique_strings(
            stochastic_support_reason_codes
        ),
        residual_history_refs=_unique_refs(residual_history_refs),
        stochastic_model_refs=_unique_refs(stochastic_model_refs),
        effective_probabilistic_config={
            "distribution_family": _distribution_family_from_stochastic_family(
                stochastic_family_id
            ),
            "interval_levels": list(resolved_interval_levels),
            "quantile_levels": list(resolved_quantile_levels),
        },
    )
    return prediction_artifact.to_manifest(catalog)


def _resolve_probabilistic_score_policy(
    *,
    score_policy_manifest: ManifestEnvelope,
    forecast_object_type: str,
) -> _ProbabilisticScorePolicy:
    try:
        expected_schema_name = _POLICY_SCHEMAS[forecast_object_type]
    except KeyError as exc:
        raise ContractValidationError(
            code="unsupported_forecast_object_type",
            message=(
                "probabilistic evaluation requires a supported forecast " "object type"
            ),
            field_path="forecast_object_type",
            details={"forecast_object_type": forecast_object_type},
        ) from exc

    if score_policy_manifest.schema_name != expected_schema_name:
        raise ContractValidationError(
            code="probabilistic_policy_forecast_object_type_mismatch",
            message=(
                "probabilistic prediction artifacts require the matching score "
                "policy schema for the selected forecast object type"
            ),
            field_path="score_policy_manifest.schema_name",
            details={
                "expected_schema_name": expected_schema_name,
                "schema_name": score_policy_manifest.schema_name,
                "forecast_object_type": forecast_object_type,
            },
        )
    if (
        str(score_policy_manifest.body.get("forecast_object_type"))
        != forecast_object_type
    ):
        raise ContractValidationError(
            code="probabilistic_policy_forecast_object_type_mismatch",
            message="score policy body must declare the same forecast object type",
            field_path="score_policy_manifest.body.forecast_object_type",
            details={
                "expected_forecast_object_type": forecast_object_type,
                "forecast_object_type": score_policy_manifest.body.get(
                    "forecast_object_type"
                ),
            },
        )
    return _ProbabilisticScorePolicy(
        forecast_object_type=forecast_object_type,
        score_law_id=str(score_policy_manifest.body["primary_score"]),
        horizon_weights=_horizon_weights(score_policy_manifest),
    )


def _probabilistic_support_path(
    *,
    catalog: ContractCatalog,
    fit_result: CandidateWindowFitResult,
    point_path: Mapping[int, float],
    stochastic_evidence_mode: str,
    stochastic_fit_result: CandidateWindowFitResult | None,
    residual_history_ref: TypedRef | None,
    stochastic_family_id: str,
    student_t_degrees_of_freedom: float | None,
    supporting_artifact_sink: list[ManifestEnvelope] | None,
) -> _ProbabilisticSupportBundle:
    family_id = fit_result.fitted_candidate.structural_layer.cir_family_id
    if family_id not in {"analytic", "recursive", "spectral", "algorithmic"}:
        raise ContractValidationError(
            code="unsupported_prediction_candidate",
            message=(
                "probabilistic prediction artifacts support analytic, recursive, "
                "spectral, and algorithmic fitted candidates only"
            ),
            field_path="candidate.structural_layer.cir_family_id",
            details={"family_id": family_id},
        )
    if stochastic_evidence_mode not in {"production", "compatibility"}:
        raise ContractValidationError(
            code="invalid_stochastic_evidence_status",
            message="stochastic evidence mode must be production or compatibility",
            field_path="stochastic_evidence_mode",
            details={"stochastic_evidence_mode": stochastic_evidence_mode},
        )
    if stochastic_evidence_mode == "production":
        evidence_fit_result = stochastic_fit_result or fit_result
        if evidence_fit_result.candidate_id != fit_result.candidate_id:
            raise ContractValidationError(
                code="stochastic_evidence_candidate_mismatch",
                message=(
                    "production stochastic evidence must belong to the predicted "
                    "candidate"
                ),
                field_path="stochastic_fit_result.candidate_id",
                details={
                    "prediction_candidate_id": fit_result.candidate_id,
                    "stochastic_candidate_id": evidence_fit_result.candidate_id,
                },
            )
        if residual_history_ref is None or not evidence_fit_result.residual_history:
            raise ContractValidationError(
                code="missing_stochastic_model_evidence",
                message=(
                    "production probabilistic artifacts require residual-history "
                    "backed stochastic model evidence"
                ),
                field_path="residual_history_ref",
                details={
                    "missing": [
                        "residual_history_ref",
                        "stochastic_model_ref",
                    ]
                },
            )
        if evidence_fit_result.residual_history_validation.status != "passed":
            raise ContractValidationError(
                code="invalid_residual_history_evidence",
                message="production stochastic support requires valid residual history",
                field_path="stochastic_fit_result.residual_history",
                details=evidence_fit_result.residual_history_validation.as_dict(),
            )
        stochastic_model = fit_residual_stochastic_model(
            candidate_id=fit_result.candidate_id,
            residual_history=evidence_fit_result.residual_history,
            point_path=point_path,
            family_id=stochastic_family_id,
            horizon_scale_law="sqrt_horizon",
            required_horizon_set=tuple(point_path),
            residual_history_ref=residual_history_ref,
            evidence_status="production",
            student_t_degrees_of_freedom=student_t_degrees_of_freedom,
        )
        stochastic_model_manifest = _stochastic_model_manifest(
            catalog=catalog,
            stochastic_model=stochastic_model,
        )
        if supporting_artifact_sink is not None:
            supporting_artifact_sink.append(stochastic_model_manifest)
    else:
        stochastic_model = fit_residual_stochastic_model(
            candidate_id=fit_result.candidate_id,
            residuals=_stochastic_residual_proxy(fit_result),
            point_path=point_path,
            family_id="gaussian",
            horizon_scale_law="sqrt_horizon",
        )
        stochastic_model_manifest = None
    reason_codes = (
        ("heuristic_gaussian_support_not_production",)
        if stochastic_model.heuristic_gaussian_support
        else ()
    )
    return _ProbabilisticSupportBundle(
        support_path={
            horizon: _StochasticPredictiveSupport(
                location=support.location,
                scale=support.scale,
                distribution_family=support.distribution_family,
                support_kind=support.support_kind,
                distribution_parameters=_distribution_parameters_for_support(
                    support=support,
                    residual_parameters=stochastic_model.residual_parameter_summary,
                ),
            )
            for horizon, support in stochastic_model.support_path().items()
        },
        stochastic_support_status=stochastic_model.evidence_status,
        stochastic_support_reason_codes=reason_codes,
        residual_history_refs=(
            ()
            if stochastic_model.residual_history_ref is None
            else (stochastic_model.residual_history_ref,)
        ),
        stochastic_model_refs=(
            ()
            if stochastic_model_manifest is None
            else (stochastic_model_manifest.ref,)
        ),
    )


def _stochastic_model_manifest(
    *,
    catalog: ContractCatalog,
    stochastic_model,
) -> ManifestEnvelope:
    model = StochasticModelManifest(
        object_id=_stochastic_model_object_id(stochastic_model.replay_identity),
        stochastic_model_id=stochastic_model.replay_identity,
        candidate_id=stochastic_model.candidate_id,
        residual_history_ref=stochastic_model.residual_history_ref,
        observation_family=stochastic_model.observation_family,
        residual_family=stochastic_model.residual_family,
        support_kind=stochastic_model.support_kind,
        horizon_scale_law=stochastic_model.horizon_scale_law,
        fitted_parameters=stochastic_model.residual_parameter_summary,
        residual_count=stochastic_model.residual_count,
        min_count_policy=stochastic_model.min_count_policy,
        evidence_status=stochastic_model.evidence_status,
        heuristic_gaussian_support=stochastic_model.heuristic_gaussian_support,
        replay_identity=stochastic_model.replay_identity,
        residual_location=stochastic_model.residual_location,
        residual_scale=stochastic_model.residual_scale,
    )
    return model.to_manifest(catalog)


def _stochastic_model_object_id(replay_identity: str) -> str:
    return str(replay_identity).replace(":", "_")


def _distribution_parameters_for_support(
    *,
    support: StochasticPredictiveSupport,
    residual_parameters: Mapping[str, float],
) -> dict[str, float]:
    parameters = {
        "location": _stable_float(support.location),
        "scale": _stable_float(support.scale),
    }
    if "df" in residual_parameters:
        parameters["df"] = _stable_float(float(residual_parameters["df"]))
    return parameters


def _build_row(
    *,
    forecast_object_type: str,
    scored_origin,
    support: _StochasticPredictiveSupport,
    realized_observation: float,
    origin_row: Mapping[str, Any],
    entity: str | None = None,
    interval_levels: Sequence[float] = (0.8,),
    quantile_levels: Sequence[float] = _QUANTILE_LEVELS,
):
    row_kwargs = {
        "origin_time": scored_origin.origin_time,
        "available_at": scored_origin.available_at,
        "horizon": scored_origin.horizon,
        "realized_observation": realized_observation,
    }
    if entity is not None:
        row_kwargs["entity"] = entity
    if forecast_object_type == "distribution":
        return DistributionPredictionRow(
            distribution_family=support.distribution_family,
            location=support.location,
            scale=support.scale,
            support_kind=support.support_kind,
            distribution_parameters=_support_parameters(support),
            **row_kwargs,
        )
    if forecast_object_type == "interval":
        intervals = tuple(
            _interval_payload(support=support, nominal_coverage=level)
            for level in interval_levels
        )
        primary_interval = intervals[0]
        lower_bound, upper_bound = _central_interval(
            support=support,
            nominal_coverage=float(primary_interval["nominal_coverage"]),
        )
        return IntervalPredictionRow(
            nominal_coverage=float(primary_interval["nominal_coverage"]),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            intervals=intervals,
            distribution_family=support.distribution_family,
            distribution_parameters=_support_parameters(support),
            **row_kwargs,
        )
    if forecast_object_type == "quantile":
        return QuantilePredictionRow(
            quantiles=tuple(
                QuantileValue(
                    level=level,
                    value=_distribution_quantile(support=support, level=level),
                )
                for level in quantile_levels
            ),
            distribution_family=support.distribution_family,
            distribution_parameters=_support_parameters(support),
            **row_kwargs,
        )
    if forecast_object_type == "event_probability":
        threshold = _stable_float(float(origin_row["target"]))
        event_definition = EventDefinition.from_manifest(
            {
                "event_id": "declared_target_threshold",
                "operator": "greater_than_or_equal",
                "threshold": threshold,
                "threshold_source": "declared_literal",
                "variable": "target",
                "calibration_required": True,
            }
        )
        observation_model = get_observation_model(
            _observation_family_id(support.distribution_family)
        ).bind(
            _support_parameters(support),
        )
        event_probability = _stable_float(event_definition.probability(observation_model))
        realized_event = event_definition.evaluate(realized_observation)
        return EventProbabilityPredictionRow(
            event_definition=event_definition.as_manifest(),
            event_probability=event_probability,
            realized_event=realized_event,
            distribution_family=support.distribution_family,
            distribution_parameters=_support_parameters(support),
            **row_kwargs,
        )
    raise ContractValidationError(
        code="unsupported_forecast_object_type",
        message="unsupported probabilistic forecast object type",
        field_path="forecast_object_type",
        details={"forecast_object_type": forecast_object_type},
    )


def _support_parameters(support: _StochasticPredictiveSupport) -> dict[str, float]:
    if support.distribution_parameters is not None:
        return {
            str(key): _stable_float(float(value))
            for key, value in support.distribution_parameters.items()
        }
    return {
        "location": _stable_float(support.location),
        "scale": _stable_float(support.scale),
    }


def _interval_payload(
    *,
    support: _StochasticPredictiveSupport,
    nominal_coverage: float,
) -> dict[str, float]:
    lower_bound, upper_bound = _central_interval(
        support=support,
        nominal_coverage=nominal_coverage,
    )
    return {
        "nominal_coverage": _stable_float(float(nominal_coverage)),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


def _resolve_probability_levels(
    levels: Sequence[float],
    *,
    default: tuple[float, ...],
    field_path: str,
) -> tuple[float, ...]:
    resolved = tuple(float(level) for level in (levels or default))
    if not resolved:
        return default
    for level in resolved:
        if not 0.0 < level < 1.0:
            raise ContractValidationError(
                code="invalid_probability_level",
                message="probability levels must be strictly between 0 and 1",
                field_path=field_path,
                details={"level": level},
            )
    return resolved


def _distribution_family_from_stochastic_family(stochastic_family_id: str) -> str:
    family = str(stochastic_family_id)
    aliases = {
        "gaussian": "gaussian_location_scale",
        "gaussian_location_scale": "gaussian_location_scale",
        "student_t": "student_t_location_scale",
        "student_t_location_scale": "student_t_location_scale",
        "laplace": "laplace_location_scale",
        "laplace_location_scale": "laplace_location_scale",
    }
    try:
        return aliases[family]
    except KeyError as exc:
        raise ContractValidationError(
            code="unsupported_stochastic_process_family",
            message="unsupported stochastic prediction distribution family",
            field_path="stochastic_family_id",
            details={"stochastic_family_id": stochastic_family_id},
        ) from exc


def _central_interval(
    *,
    support: _StochasticPredictiveSupport,
    nominal_coverage: float,
) -> tuple[float, float]:
    tail_mass = (1.0 - float(nominal_coverage)) / 2.0
    return (
        _distribution_quantile(support=support, level=tail_mass),
        _distribution_quantile(support=support, level=1.0 - tail_mass),
    )


def _distribution_quantile(
    *,
    support: _StochasticPredictiveSupport,
    level: float,
) -> float:
    probability = float(level)
    if not 0.0 < probability < 1.0:
        raise ContractValidationError(
            code="invalid_quantile_level",
            message="quantile levels must be strictly between 0 and 1",
            field_path="quantile.level",
            details={"level": level},
        )
    distribution = _scipy_distribution(support)
    return _stable_float(float(distribution.ppf(probability)))


def _scipy_distribution(support: _StochasticPredictiveSupport):
    parameters = _support_parameters(support)
    family_id = _observation_family_id(support.distribution_family)
    if family_id == "gaussian":
        return stats.norm(
            loc=parameters["location"],
            scale=parameters["scale"],
        )
    if family_id == "student_t":
        return stats.t(
            df=parameters["df"],
            loc=parameters["location"],
            scale=parameters["scale"],
        )
    if family_id == "laplace":
        return stats.laplace(
            loc=parameters["location"],
            scale=parameters["scale"],
        )
    raise ContractValidationError(
        code="unsupported_stochastic_process_family",
        message="unsupported stochastic prediction distribution family",
        field_path="distribution_family",
        details={"distribution_family": support.distribution_family},
    )


def _observation_family_id(distribution_family: str) -> str:
    family_map = {
        "gaussian_location_scale": "gaussian",
        "student_t_location_scale": "student_t",
        "laplace_location_scale": "laplace",
    }
    try:
        return family_map[str(distribution_family)]
    except KeyError as exc:
        raise ContractValidationError(
            code="unsupported_stochastic_process_family",
            message="unsupported stochastic prediction distribution family",
            field_path="distribution_family",
            details={"distribution_family": distribution_family},
        ) from exc


def _base_scale(fit_result: CandidateWindowFitResult) -> float:
    final_loss = max(
        float(fit_result.optimizer_diagnostics.get("final_loss", 0.0)), 0.0
    )
    training_count = max(int(fit_result.training_row_count), 1)
    rmse = math.sqrt(final_loss / training_count)
    parameter_scale = max(
        (
            abs(float(value))
            for value in fit_result.parameter_summary.values()
            if isinstance(value, int | float)
        ),
        default=0.0,
    )
    return _stable_float(max(rmse, 0.25 + (0.05 * parameter_scale)))


def _stochastic_residual_proxy(fit_result: CandidateWindowFitResult) -> tuple[float, ...]:
    base_scale = _base_scale(fit_result)
    return (-base_scale, base_scale)


def _combined_stochastic_support_status(statuses: list[str]) -> str:
    if any(status == "production" for status in statuses):
        return "production"
    return "compatibility"


def _unique_strings(values: list[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(str(value), None)
    return tuple(seen)


def _unique_refs(refs):
    seen: set[tuple[str, str]] = set()
    ordered = []
    for ref in refs:
        key = (ref.schema_name, ref.object_id)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(ref)
    return tuple(ordered)


__all__ = ["emit_probabilistic_prediction_artifact"]
