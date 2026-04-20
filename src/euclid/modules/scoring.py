from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from statistics import NormalDist, fmean
from typing import Any, Callable, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    PerHorizonPrimaryScore,
    PerHorizonScore,
    PointScoreResultManifest,
    ProbabilisticScoreResultManifest,
)
from euclid.modules.evaluation_governance import (
    ComparisonKey,
    build_comparison_universe,
)

_STANDARD_NORMAL = NormalDist()
_PROBABILISTIC_POLICY_SCHEMAS = {
    "distribution": "probabilistic_score_policy_manifest@1.0.0",
    "interval": "interval_score_policy_manifest@1.0.0",
    "quantile": "quantile_score_policy_manifest@1.0.0",
    "event_probability": "event_probability_score_policy_manifest@1.0.0",
}


@dataclass(frozen=True)
class PointComparatorEvaluationResult:
    candidate_score_result: ManifestEnvelope
    comparator_score_results: tuple[ManifestEnvelope, ...]
    comparison_universe: ManifestEnvelope


def score_point_prediction_artifact(
    *,
    catalog: ContractCatalog,
    score_policy_manifest: ManifestEnvelope,
    prediction_artifact_manifest: ManifestEnvelope,
    requested_primary_metric_id: str | None = None,
) -> ManifestEnvelope:
    _require_score_policy_manifest(score_policy_manifest)
    _require_prediction_artifact_manifest(prediction_artifact_manifest)
    score_result_id = (
        f"{prediction_artifact_manifest.body['prediction_artifact_id']}__point_score"
    )

    failure_reason = _score_failure_reason(
        score_policy_manifest=score_policy_manifest,
        prediction_artifact_manifest=prediction_artifact_manifest,
        requested_primary_metric_id=requested_primary_metric_id,
    )
    if failure_reason is not None:
        return _point_score_result_manifest(
            catalog=catalog,
            score_result_id=score_result_id,
            score_policy_ref=score_policy_manifest.ref,
            prediction_artifact_ref=prediction_artifact_manifest.ref,
            per_horizon=(),
            aggregated_primary_score=0.0,
            comparison_status="not_comparable",
            failure_reason_code=failure_reason,
        )

    horizon_weights = _resolve_horizon_weights(score_policy_manifest)
    entity_aggregation_mode = str(score_policy_manifest.body["entity_aggregation_mode"])
    rows = tuple(prediction_artifact_manifest.body["rows"])
    entity_weights = _resolve_declared_entity_weights(
        prediction_artifact_manifest=prediction_artifact_manifest,
        entity_aggregation_mode=entity_aggregation_mode,
    )
    per_horizon_metrics, aggregated_primary_score = _aggregate_primary_scores(
        rows=rows,
        horizon_weights=horizon_weights,
        entity_aggregation_mode=entity_aggregation_mode,
        entity_weights=entity_weights,
        row_score=lambda row: _point_loss(
            point_loss_id=str(score_policy_manifest.body["point_loss_id"]),
            point_forecast=float(row["point_forecast"]),
            realized_observation=float(row["realized_observation"]),
        ),
    )
    per_horizon = tuple(
        PerHorizonScore(
            horizon=horizon,
            valid_origin_count=valid_origin_count,
            mean_point_loss=mean_primary_score,
        )
        for horizon, valid_origin_count, mean_primary_score in per_horizon_metrics
    )

    if not math.isfinite(aggregated_primary_score):
        return _point_score_result_manifest(
            catalog=catalog,
            score_result_id=score_result_id,
            score_policy_ref=score_policy_manifest.ref,
            prediction_artifact_ref=prediction_artifact_manifest.ref,
            per_horizon=(),
            aggregated_primary_score=0.0,
            comparison_status="not_comparable",
            failure_reason_code="nonfinite_score_value",
        )

    return _point_score_result_manifest(
        catalog=catalog,
        score_result_id=score_result_id,
        score_policy_ref=score_policy_manifest.ref,
        prediction_artifact_ref=prediction_artifact_manifest.ref,
        per_horizon=per_horizon,
        aggregated_primary_score=_stable_float(aggregated_primary_score),
        comparison_status="comparable",
        failure_reason_code=None,
    )


def evaluate_point_comparators(
    *,
    catalog: ContractCatalog,
    score_policy_manifest: ManifestEnvelope,
    candidate_prediction_artifact: ManifestEnvelope,
    baseline_registry_manifest: ManifestEnvelope,
    comparator_prediction_artifacts: Mapping[str, ManifestEnvelope],
    practical_significance_margin: float = 0.0,
) -> PointComparatorEvaluationResult:
    _require_baseline_registry_manifest(baseline_registry_manifest)
    margin = _resolve_practical_significance_margin(practical_significance_margin)
    compatible_policy_ref = _typed_ref(
        baseline_registry_manifest.body["compatible_point_score_policy_ref"]
    )
    if compatible_policy_ref != score_policy_manifest.ref:
        raise ContractValidationError(
            code="baseline_registry_score_policy_mismatch",
            message=(
                "baseline registry must bind the same point-score policy as the "
                "candidate/comparator evaluation"
            ),
            field_path="baseline_registry_manifest.body.compatible_point_score_policy_ref",
            details={
                "baseline_registry_policy_ref": compatible_policy_ref.as_dict(),
                "score_policy_ref": score_policy_manifest.ref.as_dict(),
            },
        )

    candidate_score_result = score_point_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy_manifest,
        prediction_artifact_manifest=candidate_prediction_artifact,
    )
    if candidate_score_result.body["comparison_status"] != "comparable":
        raise ContractValidationError(
            code="candidate_point_score_not_comparable",
            message=(
                "candidate prediction artifact is not comparable under the score law"
            ),
            field_path="candidate_prediction_artifact",
            details={
                "failure_reason_code": candidate_score_result.body[
                    "failure_reason_code"
                ]
            },
        )

    candidate_primary_score = float(
        candidate_score_result.body["aggregated_primary_score"]
    )
    comparator_score_results: list[ManifestEnvelope] = []
    paired_comparison_records: list[dict[str, Any]] = []
    primary_baseline_id = str(baseline_registry_manifest.body["primary_baseline_id"])
    primary_baseline_score_result: ManifestEnvelope | None = None
    primary_baseline_artifact: ManifestEnvelope | None = None

    for declaration in baseline_registry_manifest.body["baseline_declarations"]:
        comparator_id = str(declaration["baseline_id"])
        comparator_artifact = comparator_prediction_artifacts.get(comparator_id)
        if comparator_artifact is None:
            raise ContractValidationError(
                code="missing_required_comparator_prediction_artifact",
                message=(
                    "all declared comparators must provide a prediction artifact "
                    "under the frozen evaluation geometry"
                ),
                field_path="comparator_prediction_artifacts",
                details={"missing_comparator_id": comparator_id},
            )

        comparator_score_result = score_point_prediction_artifact(
            catalog=catalog,
            score_policy_manifest=score_policy_manifest,
            prediction_artifact_manifest=comparator_artifact,
        )
        comparator_score_results.append(comparator_score_result)
        record = _build_paired_comparison_record(
            candidate_prediction_artifact=candidate_prediction_artifact,
            candidate_score_result=candidate_score_result,
            comparator_prediction_artifact=comparator_artifact,
            comparator_score_result=comparator_score_result,
            comparator_id=comparator_id,
            comparator_kind=str(declaration["comparator_kind"]),
            practical_significance_margin=margin,
            point_loss_id=str(score_policy_manifest.body["point_loss_id"]),
        )
        paired_comparison_records.append(record)

        if comparator_id == primary_baseline_id:
            primary_baseline_score_result = comparator_score_result
            primary_baseline_artifact = comparator_artifact

    if primary_baseline_score_result is None or primary_baseline_artifact is None:
        raise ContractValidationError(
            code="primary_comparator_missing",
            message=(
                "baseline registry primary comparator must appear in the comparator set"
            ),
            field_path="baseline_registry_manifest.body.primary_baseline_id",
            details={"primary_baseline_id": primary_baseline_id},
        )

    primary_record = next(
        record
        for record in paired_comparison_records
        if record["comparator_id"] == primary_baseline_id
    )
    if primary_record["comparison_status"] != "comparable":
        raise ContractValidationError(
            code="primary_comparator_not_comparable",
            message=(
                "primary comparator must share forecast object type, score policy, "
                "horizon set, and scored-origin set with the candidate"
            ),
            field_path="primary_comparator",
            details=primary_record,
        )

    comparison_universe = build_comparison_universe(
        selected_candidate_id=str(candidate_prediction_artifact.body["candidate_id"]),
        baseline_id=primary_baseline_id,
        candidate_primary_score=candidate_primary_score,
        baseline_primary_score=float(
            primary_baseline_score_result.body["aggregated_primary_score"]
        ),
        candidate_comparison_key=_comparison_key_from_prediction_artifact(
            candidate_prediction_artifact
        ),
        baseline_comparison_key=_comparison_key_from_prediction_artifact(
            primary_baseline_artifact
        ),
        candidate_score_result_ref=candidate_score_result.ref,
        baseline_score_result_ref=primary_baseline_score_result.ref,
        comparator_score_result_refs=tuple(
            score_result.ref for score_result in comparator_score_results
        ),
        paired_comparison_records=tuple(paired_comparison_records),
        practical_significance_margin=margin,
    ).to_manifest(catalog)
    return PointComparatorEvaluationResult(
        candidate_score_result=candidate_score_result,
        comparator_score_results=tuple(comparator_score_results),
        comparison_universe=comparison_universe,
    )


def score_probabilistic_prediction_artifact(
    *,
    catalog: ContractCatalog,
    score_policy_manifest: ManifestEnvelope,
    prediction_artifact_manifest: ManifestEnvelope,
    requested_primary_metric_id: str | None = None,
) -> ManifestEnvelope:
    _require_prediction_artifact_manifest(prediction_artifact_manifest)
    score_result_id = (
        f"{prediction_artifact_manifest.body['prediction_artifact_id']}"
        "__probabilistic_score"
    )

    failure_reason = _probabilistic_score_failure_reason(
        score_policy_manifest=score_policy_manifest,
        prediction_artifact_manifest=prediction_artifact_manifest,
        requested_primary_metric_id=requested_primary_metric_id,
    )
    forecast_object_type = str(
        prediction_artifact_manifest.body["forecast_object_type"]
    )
    if failure_reason is not None:
        return _probabilistic_score_result_manifest(
            catalog=catalog,
            score_result_id=score_result_id,
            score_policy_ref=score_policy_manifest.ref,
            prediction_artifact_ref=prediction_artifact_manifest.ref,
            forecast_object_type=forecast_object_type,
            per_horizon=(),
            aggregated_primary_score=0.0,
            comparison_status="not_comparable",
            failure_reason_code=failure_reason,
        )

    primary_score_id = str(score_policy_manifest.body["primary_score"])
    horizon_weights = _resolve_horizon_weights(score_policy_manifest)
    entity_aggregation_mode = str(score_policy_manifest.body["entity_aggregation_mode"])
    rows = tuple(prediction_artifact_manifest.body["rows"])
    entity_weights = _resolve_declared_entity_weights(
        prediction_artifact_manifest=prediction_artifact_manifest,
        entity_aggregation_mode=entity_aggregation_mode,
    )
    per_horizon_metrics, aggregated_primary_score = _aggregate_primary_scores(
        rows=rows,
        horizon_weights=horizon_weights,
        entity_aggregation_mode=entity_aggregation_mode,
        entity_weights=entity_weights,
        row_score=lambda row: _probabilistic_primary_score(
            forecast_object_type=forecast_object_type,
            primary_score_id=primary_score_id,
            row=row,
        ),
    )
    per_horizon = tuple(
        PerHorizonPrimaryScore(
            horizon=horizon,
            valid_origin_count=valid_origin_count,
            mean_primary_score=mean_primary_score,
        )
        for horizon, valid_origin_count, mean_primary_score in per_horizon_metrics
    )

    if not math.isfinite(aggregated_primary_score):
        return _probabilistic_score_result_manifest(
            catalog=catalog,
            score_result_id=score_result_id,
            score_policy_ref=score_policy_manifest.ref,
            prediction_artifact_ref=prediction_artifact_manifest.ref,
            forecast_object_type=forecast_object_type,
            per_horizon=(),
            aggregated_primary_score=0.0,
            comparison_status="not_comparable",
            failure_reason_code="nonfinite_score_value",
        )

    return _probabilistic_score_result_manifest(
        catalog=catalog,
        score_result_id=score_result_id,
        score_policy_ref=score_policy_manifest.ref,
        prediction_artifact_ref=prediction_artifact_manifest.ref,
        forecast_object_type=forecast_object_type,
        per_horizon=per_horizon,
        aggregated_primary_score=_stable_float(aggregated_primary_score),
        comparison_status="comparable",
        failure_reason_code=None,
    )


def score_prediction_artifact(
    *,
    catalog: ContractCatalog,
    score_policy_manifest: ManifestEnvelope,
    prediction_artifact_manifest: ManifestEnvelope,
    requested_primary_metric_id: str | None = None,
) -> ManifestEnvelope:
    forecast_object_type = str(
        prediction_artifact_manifest.body["forecast_object_type"]
    )
    if forecast_object_type == "point":
        return score_point_prediction_artifact(
            catalog=catalog,
            score_policy_manifest=score_policy_manifest,
            prediction_artifact_manifest=prediction_artifact_manifest,
            requested_primary_metric_id=requested_primary_metric_id,
        )
    return score_probabilistic_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy_manifest,
        prediction_artifact_manifest=prediction_artifact_manifest,
        requested_primary_metric_id=requested_primary_metric_id,
    )


def _point_score_result_manifest(
    *,
    catalog: ContractCatalog,
    score_result_id: str,
    score_policy_ref: TypedRef,
    prediction_artifact_ref: TypedRef,
    per_horizon: tuple[PerHorizonScore, ...],
    aggregated_primary_score: float,
    comparison_status: str,
    failure_reason_code: str | None,
) -> ManifestEnvelope:
    return PointScoreResultManifest(
        score_result_id=score_result_id,
        score_policy_ref=score_policy_ref,
        prediction_artifact_ref=prediction_artifact_ref,
        per_horizon=per_horizon,
        aggregated_primary_score=_stable_float(aggregated_primary_score),
        comparison_status=comparison_status,
        failure_reason_code=failure_reason_code,
    ).to_manifest(catalog)


def _probabilistic_score_result_manifest(
    *,
    catalog: ContractCatalog,
    score_result_id: str,
    score_policy_ref: TypedRef,
    prediction_artifact_ref: TypedRef,
    forecast_object_type: str,
    per_horizon: tuple[PerHorizonPrimaryScore, ...],
    aggregated_primary_score: float,
    comparison_status: str,
    failure_reason_code: str | None,
) -> ManifestEnvelope:
    return ProbabilisticScoreResultManifest(
        score_result_id=score_result_id,
        score_policy_ref=score_policy_ref,
        prediction_artifact_ref=prediction_artifact_ref,
        forecast_object_type=forecast_object_type,
        per_horizon=per_horizon,
        aggregated_primary_score=_stable_float(aggregated_primary_score),
        comparison_status=comparison_status,
        failure_reason_code=failure_reason_code,
    ).to_manifest(catalog)


def _score_failure_reason(
    *,
    score_policy_manifest: ManifestEnvelope,
    prediction_artifact_manifest: ManifestEnvelope,
    requested_primary_metric_id: str | None,
) -> str | None:
    if str(score_policy_manifest.body["forecast_object_type"]) != "point":
        return "unsupported_forecast_object_type"
    if str(prediction_artifact_manifest.body["forecast_object_type"]) != "point":
        return "unsupported_forecast_object_type"
    entity_aggregation_mode = str(score_policy_manifest.body["entity_aggregation_mode"])
    if entity_aggregation_mode not in {
        "single_entity_only_no_cross_entity_aggregation",
        "per_entity_primary_score_then_declared_entity_weights",
    }:
        return "entity_aggregation_out_of_scope"
    entity_weights = _resolve_declared_entity_weights(
        prediction_artifact_manifest=prediction_artifact_manifest,
        entity_aggregation_mode=entity_aggregation_mode,
        failure_only=True,
    )
    if isinstance(entity_weights, str):
        return entity_weights
    if requested_primary_metric_id is not None and requested_primary_metric_id in set(
        score_policy_manifest.body["forbidden_primary_metric_ids"]
    ):
        return "forbidden_primary_metric_requested"
    if (
        _typed_ref(prediction_artifact_manifest.body["score_policy_ref"])
        != score_policy_manifest.ref
    ):
        return "mixed_score_policy_within_comparison_class"

    if prediction_artifact_manifest.body.get("missing_scored_origins"):
        reason_codes = {
            str(item["reason_code"])
            for item in prediction_artifact_manifest.body["missing_scored_origins"]
        }
        if "nonfinite_point_forecast" in reason_codes:
            return "nonfinite_point_forecast"
        if "nonfinite_observation" in reason_codes:
            return "nonfinite_observation"
        return "incomplete_declared_horizon_panel"

    horizon_weights = _resolve_horizon_weights(score_policy_manifest)
    declared_horizons = tuple(horizon for horizon, _ in horizon_weights)
    rows = tuple(prediction_artifact_manifest.body["rows"])
    if not rows:
        return "empty_valid_origin_set_for_horizon"

    if prediction_artifact_manifest.body.get("scored_origin_panel"):
        expected_pairs = {
            _row_panel_key(origin)
            for origin in prediction_artifact_manifest.body["scored_origin_panel"]
        }
        observed_pairs = {_row_panel_key(row) for row in rows}
        if expected_pairs != observed_pairs:
            return "incomplete_declared_horizon_panel"

    origin_horizon_pairs: set[tuple[str | None, str, int]] = set()
    origin_horizons: dict[tuple[str | None, str], list[int]] = {}
    for row in rows:
        point_forecast = float(row["point_forecast"])
        realized_observation = float(row["realized_observation"])
        if not math.isfinite(point_forecast):
            return "nonfinite_point_forecast"
        if not math.isfinite(realized_observation):
            return "nonfinite_observation"
        origin_key = _row_origin_key(row)
        pair_key = _row_panel_key(row)
        if pair_key in origin_horizon_pairs:
            return "incomplete_declared_horizon_panel"
        origin_horizon_pairs.add(pair_key)
        origin_horizons.setdefault(origin_key, []).append(int(row["horizon"]))

    for origin_key, horizons in origin_horizons.items():
        if tuple(sorted(horizons)) != declared_horizons:
            return "incomplete_declared_horizon_panel"
        if not origin_key[1]:
            return "incomplete_declared_horizon_panel"
        if (
            entity_aggregation_mode
            == "per_entity_primary_score_then_declared_entity_weights"
            and origin_key[0] is None
        ):
            return "entity_panel_mismatch"

    common_origin_count = len(origin_horizons)
    if common_origin_count == 0:
        return "empty_valid_origin_set_for_horizon"
    for horizon in declared_horizons:
        horizon_count = sum(1 for row in rows if int(row["horizon"]) == horizon)
        if horizon_count == 0:
            return "empty_valid_origin_set_for_horizon"
        if horizon_count != common_origin_count:
            return "incomplete_declared_horizon_panel"
    return None


def _probabilistic_score_failure_reason(
    *,
    score_policy_manifest: ManifestEnvelope,
    prediction_artifact_manifest: ManifestEnvelope,
    requested_primary_metric_id: str | None,
) -> str | None:
    forecast_object_type = str(
        prediction_artifact_manifest.body["forecast_object_type"]
    )
    expected_schema_name = _PROBABILISTIC_POLICY_SCHEMAS.get(forecast_object_type)
    if expected_schema_name is None:
        return "unsupported_forecast_object_type"
    if score_policy_manifest.schema_name != expected_schema_name:
        return "unsupported_forecast_object_type"
    if (
        str(score_policy_manifest.body.get("forecast_object_type"))
        != forecast_object_type
    ):
        return "unsupported_forecast_object_type"
    entity_aggregation_mode = str(score_policy_manifest.body["entity_aggregation_mode"])
    if entity_aggregation_mode not in {
        "single_entity_only_no_cross_entity_aggregation",
        "per_entity_primary_score_then_declared_entity_weights",
    }:
        return "entity_aggregation_out_of_scope"
    entity_weights = _resolve_declared_entity_weights(
        prediction_artifact_manifest=prediction_artifact_manifest,
        entity_aggregation_mode=entity_aggregation_mode,
        failure_only=True,
    )
    if isinstance(entity_weights, str):
        return entity_weights
    if requested_primary_metric_id is not None and requested_primary_metric_id in set(
        score_policy_manifest.body.get("forbidden_primary_metric_ids", [])
    ):
        return "forbidden_primary_metric_requested"
    if (
        _typed_ref(prediction_artifact_manifest.body["score_policy_ref"])
        != score_policy_manifest.ref
    ):
        return "mixed_score_policy_within_comparison_class"

    if prediction_artifact_manifest.body.get("missing_scored_origins"):
        reason_codes = {
            str(item.get("reason_code"))
            for item in prediction_artifact_manifest.body["missing_scored_origins"]
        }
        if "nonfinite_observation" in reason_codes:
            return "nonfinite_observation"
        return "incomplete_declared_horizon_panel"

    horizon_weights = _resolve_horizon_weights(score_policy_manifest)
    declared_horizons = tuple(horizon for horizon, _ in horizon_weights)
    rows = tuple(prediction_artifact_manifest.body["rows"])
    if not rows:
        return "empty_valid_origin_set_for_horizon"

    if prediction_artifact_manifest.body.get("scored_origin_panel"):
        expected_pairs = {
            _row_panel_key(origin)
            for origin in prediction_artifact_manifest.body["scored_origin_panel"]
        }
        observed_pairs = {_row_panel_key(row) for row in rows}
        if expected_pairs != observed_pairs:
            return "incomplete_declared_horizon_panel"

    origin_horizon_pairs: set[tuple[str | None, str, int]] = set()
    origin_horizons: dict[tuple[str | None, str], list[int]] = {}
    for row in rows:
        row_failure = _probabilistic_row_failure_reason(
            forecast_object_type=forecast_object_type,
            row=row,
        )
        if row_failure is not None:
            return row_failure
        origin_key = _row_origin_key(row)
        pair_key = _row_panel_key(row)
        if pair_key in origin_horizon_pairs:
            return "incomplete_declared_horizon_panel"
        origin_horizon_pairs.add(pair_key)
        origin_horizons.setdefault(origin_key, []).append(int(row["horizon"]))

    for origin_key, horizons in origin_horizons.items():
        if tuple(sorted(horizons)) != declared_horizons:
            return "incomplete_declared_horizon_panel"
        if not origin_key[1]:
            return "incomplete_declared_horizon_panel"
        if (
            entity_aggregation_mode
            == "per_entity_primary_score_then_declared_entity_weights"
            and origin_key[0] is None
        ):
            return "entity_panel_mismatch"

    common_origin_count = len(origin_horizons)
    if common_origin_count == 0:
        return "empty_valid_origin_set_for_horizon"
    for horizon in declared_horizons:
        horizon_count = sum(1 for row in rows if int(row["horizon"]) == horizon)
        if horizon_count == 0:
            return "empty_valid_origin_set_for_horizon"
        if horizon_count != common_origin_count:
            return "incomplete_declared_horizon_panel"
    return None


def _probabilistic_row_failure_reason(
    *,
    forecast_object_type: str,
    row: Mapping[str, Any],
) -> str | None:
    realized_observation = float(row["realized_observation"])
    if not math.isfinite(realized_observation):
        return "nonfinite_observation"
    if forecast_object_type == "distribution":
        if str(row["distribution_family"]) != "gaussian_location_scale":
            return "unsupported_forecast_object_type"
        location = float(row["location"])
        scale = float(row["scale"])
        if not math.isfinite(location) or not math.isfinite(scale) or scale <= 0:
            return "nonfinite_score_value"
        return None
    if forecast_object_type == "interval":
        nominal_coverage = float(row["nominal_coverage"])
        lower_bound = float(row["lower_bound"])
        upper_bound = float(row["upper_bound"])
        if not (0 < nominal_coverage < 1):
            return "nonfinite_score_value"
        if not all(math.isfinite(value) for value in (lower_bound, upper_bound)):
            return "nonfinite_score_value"
        if lower_bound > upper_bound:
            return "nonfinite_score_value"
        return None
    if forecast_object_type == "quantile":
        quantiles = row.get("quantiles", [])
        if not isinstance(quantiles, list) or not quantiles:
            return "nonfinite_score_value"
        seen_levels: set[float] = set()
        for item in quantiles:
            level = float(item["level"])
            value = float(item["value"])
            if not (0 < level < 1):
                return "nonfinite_score_value"
            if level in seen_levels or not math.isfinite(value):
                return "nonfinite_score_value"
            seen_levels.add(level)
        return None
    if forecast_object_type == "event_probability":
        probability = float(row["event_probability"])
        if not math.isfinite(probability) or probability < 0 or probability > 1:
            return "nonfinite_score_value"
        return None
    return "unsupported_forecast_object_type"


def _build_paired_comparison_record(
    *,
    candidate_prediction_artifact: ManifestEnvelope,
    candidate_score_result: ManifestEnvelope,
    comparator_prediction_artifact: ManifestEnvelope,
    comparator_score_result: ManifestEnvelope,
    comparator_id: str,
    comparator_kind: str,
    practical_significance_margin: float,
    point_loss_id: str,
) -> dict[str, Any]:
    if comparator_score_result.body["comparison_status"] != "comparable":
        return {
            "comparator_id": comparator_id,
            "comparator_kind": comparator_kind,
            "comparison_status": "not_comparable",
            "failure_reason_code": comparator_score_result.body["failure_reason_code"],
            "score_result_ref": comparator_score_result.ref.as_dict(),
        }

    candidate_key = _comparison_key_from_prediction_artifact(
        candidate_prediction_artifact
    )
    comparator_key = _comparison_key_from_prediction_artifact(
        comparator_prediction_artifact
    )
    if not _comparison_keys_match(candidate_key, comparator_key):
        return {
            "comparator_id": comparator_id,
            "comparator_kind": comparator_kind,
            "comparison_status": "not_comparable",
            "failure_reason_code": "comparison_key_mismatch",
            "score_result_ref": comparator_score_result.ref.as_dict(),
            "candidate_comparison_key": candidate_key.as_dict(),
            "comparator_comparison_key": comparator_key.as_dict(),
        }

    horizon_weights = _resolve_horizon_weights_from_artifact(
        candidate_prediction_artifact
    )
    candidate_losses = _losses_by_origin_and_horizon(
        prediction_artifact_manifest=candidate_prediction_artifact,
        point_loss_id=point_loss_id,
    )
    comparator_losses = _losses_by_origin_and_horizon(
        prediction_artifact_manifest=comparator_prediction_artifact,
        point_loss_id=point_loss_id,
    )
    origin_keys = tuple(sorted(candidate_losses))

    per_horizon_mean_loss_differentials: list[dict[str, Any]] = []
    for horizon, _ in horizon_weights:
        per_horizon_mean_loss_differentials.append(
            {
                "horizon": horizon,
                "mean_loss_differential": _stable_float(
                    fmean(
                        comparator_losses[origin_key][horizon]
                        - candidate_losses[origin_key][horizon]
                        for origin_key in origin_keys
                    )
                ),
            }
        )

    mean_loss_differential = _stable_float(
        fmean(
            sum(
                weight
                * (
                    comparator_losses[origin_key][horizon]
                    - candidate_losses[origin_key][horizon]
                )
                for horizon, weight in horizon_weights
            )
            for origin_key in origin_keys
        )
    )
    practical_significance_status = _practical_significance_status(
        mean_loss_differential=mean_loss_differential,
        margin=practical_significance_margin,
    )
    return {
        "comparator_id": comparator_id,
        "comparator_kind": comparator_kind,
        "comparison_status": "comparable",
        "failure_reason_code": None,
        "candidate_primary_score": _stable_float(
            float(candidate_score_result.body["aggregated_primary_score"])
        ),
        "comparator_primary_score": _stable_float(
            float(comparator_score_result.body["aggregated_primary_score"])
        ),
        "primary_score_delta": mean_loss_differential,
        "paired_origin_count": len(origin_keys),
        "mean_loss_differential": mean_loss_differential,
        "per_horizon_mean_loss_differentials": per_horizon_mean_loss_differentials,
        "practical_significance_margin": _stable_float(practical_significance_margin),
        "practical_significance_status": practical_significance_status,
        "score_result_ref": comparator_score_result.ref.as_dict(),
    }


def _losses_by_origin_and_horizon(
    *,
    prediction_artifact_manifest: ManifestEnvelope,
    point_loss_id: str,
) -> dict[tuple[str | None, str], dict[int, float]]:
    losses: dict[tuple[str | None, str], dict[int, float]] = {}
    for row in prediction_artifact_manifest.body["rows"]:
        losses.setdefault(_row_origin_key(row), {})[int(row["horizon"])] = _point_loss(
            point_loss_id=point_loss_id,
            point_forecast=float(row["point_forecast"]),
            realized_observation=float(row["realized_observation"]),
        )
    return losses


def _comparison_key_from_prediction_artifact(
    prediction_artifact_manifest: ManifestEnvelope,
) -> ComparisonKey:
    return ComparisonKey(
        forecast_object_type=str(
            prediction_artifact_manifest.body["forecast_object_type"]
        ),
        score_policy_ref=_typed_ref(
            prediction_artifact_manifest.body["score_policy_ref"]
        ),
        horizon_set=tuple(
            int(weight["horizon"])
            for weight in prediction_artifact_manifest.body["horizon_weights"]
        ),
        scored_origin_set_id=str(
            prediction_artifact_manifest.body["scored_origin_set_id"]
        ),
        entity_panel=tuple(
            str(entity)
            for entity in prediction_artifact_manifest.body.get("entity_panel", ())
        ),
        entity_weights=tuple(
            dict(item)
            for item in prediction_artifact_manifest.body.get("entity_weights", ())
        ),
        composition_signature=(
            str(prediction_artifact_manifest.body["comparison_key"]["composition_signature"])
            if prediction_artifact_manifest.body.get("comparison_key", {}).get(
                "composition_signature"
            )
            is not None
            else None
        ),
    )


def _comparison_keys_match(
    candidate_key: ComparisonKey,
    comparator_key: ComparisonKey,
) -> bool:
    return (
        candidate_key.forecast_object_type == comparator_key.forecast_object_type
        and candidate_key.score_policy_ref == comparator_key.score_policy_ref
        and candidate_key.horizon_set == comparator_key.horizon_set
        and candidate_key.scored_origin_set_id == comparator_key.scored_origin_set_id
        and candidate_key.entity_panel == comparator_key.entity_panel
        and candidate_key.entity_weights == comparator_key.entity_weights
        and (
            candidate_key.composition_signature is None
            or comparator_key.composition_signature is None
            or candidate_key.composition_signature
            == comparator_key.composition_signature
        )
    )


def _resolve_horizon_weights(
    score_policy_manifest: ManifestEnvelope,
) -> tuple[tuple[int, float], ...]:
    raw_weights = score_policy_manifest.body["horizon_weights"]
    if not isinstance(raw_weights, list) or not raw_weights:
        raise ContractValidationError(
            code="invalid_horizon_weight_simplex",
            message="point-score policies require a non-empty horizon simplex",
            field_path="score_policy_manifest.body.horizon_weights",
        )

    horizon_weights: list[tuple[int, float]] = []
    seen_horizons: set[int] = set()
    total_weight = Decimal("0")
    for index, item in enumerate(raw_weights):
        horizon = int(item["horizon"])
        if horizon <= 0 or horizon in seen_horizons:
            raise ContractValidationError(
                code="invalid_horizon_weight_simplex",
                message="point-score policies require unique positive horizons",
                field_path=f"score_policy_manifest.body.horizon_weights[{index}].horizon",
            )
        seen_horizons.add(horizon)
        try:
            weight_decimal = Decimal(str(item["weight"]))
        except (InvalidOperation, ValueError) as exc:
            raise ContractValidationError(
                code="invalid_horizon_weight_simplex",
                message="horizon weights must be finite decimal strings",
                field_path=f"score_policy_manifest.body.horizon_weights[{index}].weight",
            ) from exc
        if not weight_decimal.is_finite() or weight_decimal < 0:
            raise ContractValidationError(
                code="invalid_horizon_weight_simplex",
                message="horizon weights must be finite decimal strings >= 0",
                field_path=f"score_policy_manifest.body.horizon_weights[{index}].weight",
            )
        total_weight += weight_decimal
        horizon_weights.append((horizon, float(weight_decimal)))

    if total_weight != Decimal("1") or not math.isfinite(float(total_weight)):
        raise ContractValidationError(
            code="invalid_horizon_weight_simplex",
            message="horizon weights must sum exactly to 1",
            field_path="score_policy_manifest.body.horizon_weights",
        )
    return tuple(sorted(horizon_weights, key=lambda item: item[0]))


def _resolve_horizon_weights_from_artifact(
    prediction_artifact_manifest: ManifestEnvelope,
) -> tuple[tuple[int, float], ...]:
    weights: list[tuple[int, float]] = []
    for item in prediction_artifact_manifest.body["horizon_weights"]:
        weights.append((int(item["horizon"]), float(item["weight"])))
    return tuple(sorted(weights, key=lambda item: item[0]))


def _aggregate_primary_scores(
    *,
    rows: tuple[Mapping[str, Any], ...],
    horizon_weights: tuple[tuple[int, float], ...],
    entity_aggregation_mode: str,
    entity_weights: tuple[tuple[str, float], ...],
    row_score: Callable[[Mapping[str, Any]], float],
) -> tuple[tuple[tuple[int, int, float], ...], float]:
    scores_by_origin_and_horizon: dict[tuple[str | None, str], dict[int, float]] = {}
    for row in rows:
        scores_by_origin_and_horizon.setdefault(_row_origin_key(row), {})[
            int(row["horizon"])
        ] = row_score(row)

    origin_keys = tuple(sorted(scores_by_origin_and_horizon))
    valid_origin_count = len(origin_keys)
    if entity_aggregation_mode == "single_entity_only_no_cross_entity_aggregation":
        per_horizon = tuple(
            (
                horizon,
                valid_origin_count,
                _stable_float(
                    fmean(
                        scores_by_origin_and_horizon[origin_key][horizon]
                        for origin_key in origin_keys
                    )
                ),
            )
            for horizon, _ in horizon_weights
        )
        aggregated_primary_score = _stable_float(
            sum(
                weight * mean_primary_score
                for (_, weight), (_, _, mean_primary_score) in zip(
                    horizon_weights, per_horizon, strict=True
                )
            )
        )
        return per_horizon, aggregated_primary_score

    origin_keys_by_entity = {
        entity: tuple(
            origin_key for origin_key in origin_keys if origin_key[0] == entity
        )
        for entity, _ in entity_weights
    }
    per_horizon_metrics: list[tuple[int, int, float]] = []
    for horizon, _ in horizon_weights:
        weighted_mean = 0.0
        for entity, weight in entity_weights:
            entity_origin_keys = origin_keys_by_entity[entity]
            entity_mean = fmean(
                scores_by_origin_and_horizon[origin_key][horizon]
                for origin_key in entity_origin_keys
            )
            weighted_mean += weight * entity_mean
        per_horizon_metrics.append(
            (horizon, valid_origin_count, _stable_float(weighted_mean))
        )

    aggregated_primary_score = 0.0
    for entity, entity_weight in entity_weights:
        entity_origin_keys = origin_keys_by_entity[entity]
        entity_primary_score = sum(
            horizon_weight
            * fmean(
                scores_by_origin_and_horizon[origin_key][horizon]
                for origin_key in entity_origin_keys
            )
            for horizon, horizon_weight in horizon_weights
        )
        aggregated_primary_score += entity_weight * entity_primary_score
    return tuple(per_horizon_metrics), _stable_float(aggregated_primary_score)


def _resolve_declared_entity_weights(
    *,
    prediction_artifact_manifest: ManifestEnvelope,
    entity_aggregation_mode: str,
    failure_only: bool = False,
) -> tuple[tuple[str, float], ...] | str:
    if entity_aggregation_mode == "single_entity_only_no_cross_entity_aggregation":
        return ()
    raw_entity_panel = prediction_artifact_manifest.body.get("entity_panel")
    if not isinstance(raw_entity_panel, list) or not raw_entity_panel:
        return "entity_panel_mismatch" if failure_only else ()
    entity_panel = tuple(str(entity) for entity in raw_entity_panel)
    if len(set(entity_panel)) != len(entity_panel) or any(
        not entity for entity in entity_panel
    ):
        return "entity_panel_mismatch" if failure_only else ()

    raw_entity_weights = prediction_artifact_manifest.body.get("entity_weights")
    if not isinstance(raw_entity_weights, list) or len(raw_entity_weights) != len(
        entity_panel
    ):
        return "invalid_entity_weight_simplex" if failure_only else ()

    entity_weights: list[tuple[str, float]] = []
    total_weight = Decimal("0")
    seen_entities: set[str] = set()
    for index, item in enumerate(raw_entity_weights):
        if not isinstance(item, Mapping):
            return "invalid_entity_weight_simplex" if failure_only else ()
        entity = str(item.get("entity", ""))
        if not entity or entity in seen_entities:
            return "invalid_entity_weight_simplex" if failure_only else ()
        seen_entities.add(entity)
        try:
            weight_decimal = Decimal(str(item.get("weight")))
        except (InvalidOperation, ValueError):
            return "invalid_entity_weight_simplex" if failure_only else ()
        if not weight_decimal.is_finite() or weight_decimal < 0:
            return "invalid_entity_weight_simplex" if failure_only else ()
        total_weight += weight_decimal
        entity_weights.append((entity, float(weight_decimal)))

    if total_weight != Decimal("1"):
        return "invalid_entity_weight_simplex" if failure_only else ()
    if tuple(entity for entity, _ in entity_weights) != entity_panel:
        return "entity_panel_mismatch" if failure_only else ()

    row_entities = {
        entity
        for entity in (
            _row_entity(row) for row in prediction_artifact_manifest.body["rows"]
        )
        if entity is not None
    }
    scored_origin_entities = {
        entity
        for entity in (
            _row_entity(origin)
            for origin in prediction_artifact_manifest.body.get(
                "scored_origin_panel", ()
            )
        )
        if entity is not None
    }
    if row_entities - set(entity_panel):
        return "entity_panel_mismatch" if failure_only else ()
    if scored_origin_entities and scored_origin_entities != set(entity_panel):
        return "entity_panel_mismatch" if failure_only else ()
    return tuple(entity_weights)


def _row_entity(row: Mapping[str, Any]) -> str | None:
    value = row.get("entity")
    if value is None:
        return None
    entity = str(value)
    return entity or None


def _row_origin_key(row: Mapping[str, Any]) -> tuple[str | None, str]:
    return (_row_entity(row), str(row["origin_time"]))


def _row_panel_key(row: Mapping[str, Any]) -> tuple[str | None, str, int]:
    return (*_row_origin_key(row), int(row["horizon"]))


def _resolve_practical_significance_margin(margin: float) -> float:
    numeric = float(margin)
    if not math.isfinite(numeric) or numeric < 0:
        raise ContractValidationError(
            code="invalid_practical_significance_margin",
            message="practical significance margins must be finite and >= 0",
            field_path="practical_significance_margin",
        )
    return _stable_float(numeric)


def _practical_significance_status(
    *,
    mean_loss_differential: float,
    margin: float,
) -> str:
    if mean_loss_differential > margin:
        return "candidate_better_than_margin"
    if mean_loss_differential < (-margin):
        return "candidate_worse_than_margin"
    return "within_margin"


def _point_loss(
    *,
    point_loss_id: str,
    point_forecast: float,
    realized_observation: float,
) -> float:
    loss_function = _point_loss_function(point_loss_id)
    return _stable_float(loss_function(point_forecast, realized_observation))


def _point_loss_function(point_loss_id: str) -> Callable[[float, float], float]:
    if point_loss_id == "squared_error":
        return lambda point_forecast, realized_observation: (
            (point_forecast - realized_observation) ** 2
        )
    if point_loss_id == "absolute_error":
        return lambda point_forecast, realized_observation: abs(
            point_forecast - realized_observation
        )
    raise ContractValidationError(
        code="unsupported_point_loss_id",
        message="retained point scoring supports squared_error and absolute_error only",
        field_path="score_policy_manifest.body.point_loss_id",
        details={"point_loss_id": point_loss_id},
    )


def _probabilistic_primary_score(
    *,
    forecast_object_type: str,
    primary_score_id: str,
    row: Mapping[str, Any],
) -> float:
    realized_observation = float(row["realized_observation"])
    if forecast_object_type == "distribution":
        location = float(row["location"])
        scale = float(row["scale"])
        if primary_score_id == "continuous_ranked_probability_score":
            return _stable_float(
                _gaussian_crps(
                    location=location,
                    scale=scale,
                    realized_observation=realized_observation,
                )
            )
        if primary_score_id == "log_score":
            return _stable_float(
                _gaussian_log_score(
                    location=location,
                    scale=scale,
                    realized_observation=realized_observation,
                )
            )
    if forecast_object_type == "interval" and primary_score_id == "interval_score":
        return _stable_float(
            _interval_score(
                nominal_coverage=float(row["nominal_coverage"]),
                lower_bound=float(row["lower_bound"]),
                upper_bound=float(row["upper_bound"]),
                realized_observation=realized_observation,
            )
        )
    if forecast_object_type == "quantile" and primary_score_id == "pinball_loss":
        return _stable_float(
            _quantile_pinball_loss(
                quantiles=tuple(row["quantiles"]),
                realized_observation=realized_observation,
            )
        )
    if forecast_object_type == "event_probability":
        probability = float(row["event_probability"])
        realized_event = bool(row["realized_event"])
        if primary_score_id == "brier_score":
            return _stable_float((probability - (1.0 if realized_event else 0.0)) ** 2)
        if primary_score_id == "log_score":
            return _stable_float(
                _event_log_score(
                    probability=probability,
                    realized_event=realized_event,
                )
            )
    raise ContractValidationError(
        code="unsupported_primary_score_id",
        message=(
            "probabilistic scoring requires an admitted score law for the "
            "selected forecast object type"
        ),
        field_path="score_policy_manifest.body.primary_score",
        details={
            "forecast_object_type": forecast_object_type,
            "primary_score_id": primary_score_id,
        },
    )


def _gaussian_crps(
    *,
    location: float,
    scale: float,
    realized_observation: float,
) -> float:
    z = (realized_observation - location) / scale
    return scale * (
        z * (2 * _STANDARD_NORMAL.cdf(z) - 1)
        + 2 * _STANDARD_NORMAL.pdf(z)
        - (1 / math.sqrt(math.pi))
    )


def _gaussian_log_score(
    *,
    location: float,
    scale: float,
    realized_observation: float,
) -> float:
    squared_residual = (realized_observation - location) ** 2
    return 0.5 * math.log(2 * math.pi * (scale**2)) + (
        squared_residual / (2 * (scale**2))
    )


def _interval_score(
    *,
    nominal_coverage: float,
    lower_bound: float,
    upper_bound: float,
    realized_observation: float,
) -> float:
    alpha = 1.0 - nominal_coverage
    score = upper_bound - lower_bound
    if realized_observation < lower_bound:
        score += (2.0 / alpha) * (lower_bound - realized_observation)
    elif realized_observation > upper_bound:
        score += (2.0 / alpha) * (realized_observation - upper_bound)
    return score


def _quantile_pinball_loss(
    *,
    quantiles: tuple[Mapping[str, Any], ...],
    realized_observation: float,
) -> float:
    losses = []
    for item in quantiles:
        level = float(item["level"])
        value = float(item["value"])
        residual = realized_observation - value
        if residual >= 0:
            losses.append(level * residual)
        else:
            losses.append((level - 1.0) * residual)
    return fmean(losses)


def _event_log_score(*, probability: float, realized_event: bool) -> float:
    if realized_event:
        if probability <= 0:
            return math.inf
        return -math.log(probability)
    if probability >= 1:
        return math.inf
    return -math.log1p(-probability)


def _typed_ref(payload: Mapping[str, Any]) -> TypedRef:
    return TypedRef(
        schema_name=str(payload["schema_name"]),
        object_id=str(payload["object_id"]),
    )


def _require_score_policy_manifest(score_policy_manifest: ManifestEnvelope) -> None:
    if score_policy_manifest.schema_name != "point_score_policy_manifest@1.0.0":
        raise ContractValidationError(
            code="malformed_score_policy_ref",
            message="scoring requires a point-score policy manifest",
            field_path="score_policy_manifest.schema_name",
            details={"schema_name": score_policy_manifest.schema_name},
        )


def _require_prediction_artifact_manifest(
    prediction_artifact_manifest: ManifestEnvelope,
) -> None:
    if prediction_artifact_manifest.schema_name != "prediction_artifact_manifest@1.1.0":
        raise ContractValidationError(
            code="malformed_prediction_artifact_ref",
            message="scoring requires a point prediction artifact manifest",
            field_path="prediction_artifact_manifest.schema_name",
            details={"schema_name": prediction_artifact_manifest.schema_name},
        )


def _require_baseline_registry_manifest(
    baseline_registry_manifest: ManifestEnvelope,
) -> None:
    if baseline_registry_manifest.schema_name != "baseline_registry_manifest@1.1.0":
        raise ContractValidationError(
            code="malformed_baseline_registry_ref",
            message="comparator evaluation requires a baseline registry manifest",
            field_path="baseline_registry_manifest.schema_name",
            details={"schema_name": baseline_registry_manifest.schema_name},
        )


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


__all__ = [
    "PointComparatorEvaluationResult",
    "score_prediction_artifact",
    "score_probabilistic_prediction_artifact",
    "evaluate_point_comparators",
    "score_point_prediction_artifact",
]
