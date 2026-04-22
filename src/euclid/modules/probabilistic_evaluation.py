from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    DistributionPredictionRow,
    EventProbabilityPredictionRow,
    IntervalPredictionRow,
    PredictionArtifactManifest,
    QuantilePredictionRow,
    QuantileValue,
)
from euclid.modules.candidate_fitting import CandidateWindowFitResult
from euclid.modules.evaluation import (
    _default_entity_weights,
    _forecast_path,
    _horizon_weights,
    _origin_row_for_scored_origin,
    _require_matching_horizon_set,
    _resolve_stage_rules,
    _segment_scored_origin_panel,
    _stable_float,
    _target_row_for_origin,
)
from euclid.modules.features import FeatureView
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

_INTERVAL_Z = 1.281551565545
_QUANTILE_LEVELS = (
    (0.1, -1.281551565545),
    (0.5, 0.0),
    (0.9, 1.281551565545),
)


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
    support_cache: dict[tuple[str, int], dict[int, _StochasticPredictiveSupport]] = {}

    rows = []
    missing_scored_origins: list[dict[str, Any]] = []
    timeguard_checks: list[dict[str, Any]] = []
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
        support_path = support_cache.get(support_key)
        if support_path is None:
            forecast_path = _forecast_path(
                candidate=fit_result.fitted_candidate,
                fit_result=fit_result,
                origin_row=origin_row,
                max_horizon=max(evaluation_segment.horizon_set),
                entity=scored_origin.entity,
            )
            support_path = _probabilistic_support_path(
                fit_result=fit_result,
                point_path=forecast_path.predictions,
            )
            support_cache[support_key] = support_path
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

        support = support_path.get(scored_origin.horizon)
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
    fit_result: CandidateWindowFitResult,
    point_path: Mapping[int, float],
) -> dict[int, _StochasticPredictiveSupport]:
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
    stochastic_model = fit_residual_stochastic_model(
        candidate_id=fit_result.candidate_id,
        residuals=_stochastic_residual_proxy(fit_result),
        point_path=point_path,
        family_id="gaussian",
        horizon_scale_law="sqrt_horizon",
    )
    return {
        horizon: _StochasticPredictiveSupport(
            location=support.location,
            scale=support.scale,
            distribution_family=support.distribution_family,
            support_kind=support.support_kind,
        )
        for horizon, support in stochastic_model.support_path().items()
    }


def _build_row(
    *,
    forecast_object_type: str,
    scored_origin,
    support: _StochasticPredictiveSupport,
    realized_observation: float,
    origin_row: Mapping[str, Any],
    entity: str | None = None,
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
            **row_kwargs,
        )
    if forecast_object_type == "interval":
        lower_bound = _stable_float(support.location - (_INTERVAL_Z * support.scale))
        upper_bound = _stable_float(support.location + (_INTERVAL_Z * support.scale))
        return IntervalPredictionRow(
            nominal_coverage=0.8,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            **row_kwargs,
        )
    if forecast_object_type == "quantile":
        return QuantilePredictionRow(
            quantiles=tuple(
                QuantileValue(
                    level=level,
                    value=_stable_float(support.location + (z_score * support.scale)),
                )
                for level, z_score in _QUANTILE_LEVELS
            ),
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
        observation_model = get_observation_model("gaussian").bind(
            {"location": support.location, "scale": support.scale}
        )
        event_probability = _stable_float(event_definition.probability(observation_model))
        realized_event = event_definition.evaluate(realized_observation)
        return EventProbabilityPredictionRow(
            event_definition=event_definition.as_manifest(),
            event_probability=event_probability,
            realized_event=realized_event,
            **row_kwargs,
        )
    raise ContractValidationError(
        code="unsupported_forecast_object_type",
        message="unsupported probabilistic forecast object type",
        field_path="forecast_object_type",
        details={"forecast_object_type": forecast_object_type},
    )


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


__all__ = ["emit_probabilistic_prediction_artifact"]
