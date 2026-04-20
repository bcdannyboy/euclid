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
class _GaussianPredictiveSupport:
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
    support_cache: dict[tuple[str, int], dict[int, _GaussianPredictiveSupport]] = {}

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
) -> dict[int, _GaussianPredictiveSupport]:
    family_id = fit_result.fitted_candidate.structural_layer.cir_family_id
    base_scale = _base_scale(fit_result)
    if family_id == "analytic":
        return _analytic_support_path(point_path=point_path, base_scale=base_scale)
    if family_id == "recursive":
        return _recursive_support_path(point_path=point_path, base_scale=base_scale)
    if family_id == "spectral":
        parameters = {
            parameter.name: parameter.value
            for parameter in (
                fit_result.fitted_candidate.structural_layer.parameter_block.parameters
            )
        }
        return _spectral_support_path(
            point_path=point_path,
            base_scale=base_scale,
            parameters=parameters,
        )
    if family_id == "algorithmic":
        return _algorithmic_support_path(point_path=point_path, base_scale=base_scale)
    raise ContractValidationError(
        code="unsupported_prediction_candidate",
        message=(
            "probabilistic prediction artifacts support analytic, recursive, "
            "spectral, and algorithmic fitted candidates only"
        ),
        field_path="candidate.structural_layer.cir_family_id",
        details={"family_id": family_id},
    )


def _analytic_support_path(
    *,
    point_path: Mapping[int, float],
    base_scale: float,
) -> dict[int, _GaussianPredictiveSupport]:
    return {
        horizon: _GaussianPredictiveSupport(
            location=_stable_float(point_path[horizon]),
            scale=_stable_float(base_scale * math.sqrt(horizon)),
        )
        for horizon in sorted(point_path)
    }


def _recursive_support_path(
    *,
    point_path: Mapping[int, float],
    base_scale: float,
) -> dict[int, _GaussianPredictiveSupport]:
    return {
        horizon: _GaussianPredictiveSupport(
            location=_stable_float(point_path[horizon]),
            scale=_stable_float(base_scale * (1.0 + (0.15 * (horizon - 1)))),
        )
        for horizon in sorted(point_path)
    }


def _spectral_support_path(
    *,
    point_path: Mapping[int, float],
    base_scale: float,
    parameters: Mapping[str, float | int],
) -> dict[int, _GaussianPredictiveSupport]:
    amplitude = max(
        abs(float(parameters.get("cosine_coefficient", 0.0))),
        abs(float(parameters.get("sine_coefficient", 0.0))),
        1.0,
    )
    return {
        horizon: _GaussianPredictiveSupport(
            location=_stable_float(point_path[horizon]),
            scale=_stable_float(base_scale * (1.0 + ((amplitude / 10.0) * horizon))),
        )
        for horizon in sorted(point_path)
    }


def _algorithmic_support_path(
    *,
    point_path: Mapping[int, float],
    base_scale: float,
) -> dict[int, _GaussianPredictiveSupport]:
    return {
        horizon: _GaussianPredictiveSupport(
            location=_stable_float(point_path[horizon]),
            scale=_stable_float(base_scale * (1.0 + (0.2 * (horizon - 1)))),
        )
        for horizon in sorted(point_path)
    }


def _build_row(
    *,
    forecast_object_type: str,
    scored_origin,
    support: _GaussianPredictiveSupport,
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
        event_probability = _stable_float(
            1.0 - _standard_normal_cdf((threshold - support.location) / support.scale)
        )
        return EventProbabilityPredictionRow(
            event_definition={
                "event_id": "target_ge_origin_target",
                "operator": "greater_than_or_equal",
                "threshold": threshold,
                "threshold_source": "origin_target",
            },
            event_probability=event_probability,
            realized_event=realized_observation >= threshold,
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


def _standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


__all__ = ["emit_probabilistic_prediction_artifact"]
