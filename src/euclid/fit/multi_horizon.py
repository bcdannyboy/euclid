from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from euclid.cir.models import CandidateIntermediateRepresentation
from euclid.contracts.errors import ContractValidationError
from euclid.runtime.hashing import sha256_digest

if TYPE_CHECKING:
    from euclid.modules.features import FeatureView
    from euclid.modules.split_planning import (
        EvaluationSegment,
        TrainingOriginPanel,
        TrainingOriginPanelRow,
    )

_DEFAULT_ENTITY_AGGREGATION_MODE = "single_entity_only_no_cross_entity_aggregation"
_ENTITY_WEIGHTED_AGGREGATION_MODE = (
    "per_entity_primary_score_then_declared_entity_weights"
)
_SUPPORTED_ENTITY_AGGREGATION_MODES = (
    _DEFAULT_ENTITY_AGGREGATION_MODE,
    _ENTITY_WEIGHTED_AGGREGATION_MODE,
)
_SUPPORTED_POINT_LOSSES = ("squared_error", "absolute_error")


@dataclass(frozen=True)
class FitStrategySpec:
    strategy_id: str = "legacy_one_step"
    horizon_set: tuple[int, ...] = (1,)
    horizon_weights: tuple[dict[str, Any], ...] = field(
        default_factory=lambda: ({"horizon": 1, "weight": "1"},)
    )
    point_loss_id: str = "squared_error"
    entity_aggregation_mode: str = _DEFAULT_ENTITY_AGGREGATION_MODE

    def __post_init__(self) -> None:
        resolved_horizon_set = _resolve_horizon_set(self.horizon_set)
        resolved_horizon_weights = _resolve_horizon_weights(
            horizon_set=resolved_horizon_set,
            horizon_weights=self.horizon_weights,
        )
        resolved_strategy_id = str(self.strategy_id)
        if not resolved_strategy_id:
            raise ContractValidationError(
                code="invalid_fit_strategy",
                message="fit strategies require a non-empty strategy_id",
                field_path="fit_strategy.strategy_id",
            )
        if self.point_loss_id not in _SUPPORTED_POINT_LOSSES:
            raise ContractValidationError(
                code="unsupported_point_loss_id",
                message="fit strategy point_loss_id is not supported",
                field_path="fit_strategy.point_loss_id",
                details={"point_loss_id": self.point_loss_id},
            )
        if self.entity_aggregation_mode not in _SUPPORTED_ENTITY_AGGREGATION_MODES:
            raise ContractValidationError(
                code="entity_aggregation_out_of_scope",
                message="fit strategy entity aggregation mode is not supported",
                field_path="fit_strategy.entity_aggregation_mode",
                details={"entity_aggregation_mode": self.entity_aggregation_mode},
            )
        object.__setattr__(self, "strategy_id", resolved_strategy_id)
        object.__setattr__(self, "horizon_set", resolved_horizon_set)
        object.__setattr__(self, "horizon_weights", resolved_horizon_weights)

    @property
    def identity_components(self) -> dict[str, Any]:
        return {
            "entity_aggregation_mode": self.entity_aggregation_mode,
            "horizon_set": list(self.horizon_set),
            "horizon_weights": [dict(item) for item in self.horizon_weights],
            "point_loss_id": self.point_loss_id,
            "strategy_id": self.strategy_id,
        }

    @property
    def identity_hash(self) -> str:
        return sha256_digest(self.identity_components)

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.identity_components,
            "identity_hash": self.identity_hash,
        }


@dataclass(frozen=True)
class RolloutObjectiveResult:
    fit_window_id: str
    fit_strategy: FitStrategySpec
    training_origin_panel: TrainingOriginPanel
    training_origin_set_id: str
    rows: tuple[dict[str, Any], ...]
    per_horizon: tuple[tuple[int, int, float], ...]
    aggregated_primary_score: float
    entity_weights: tuple[dict[str, Any], ...] = ()

    @property
    def fit_strategy_identity(self) -> str:
        return self.fit_strategy.identity_hash

    @property
    def horizon_set(self) -> tuple[int, ...]:
        return self.fit_strategy.horizon_set

    @property
    def horizon_weights(self) -> tuple[dict[str, Any], ...]:
        return self.fit_strategy.horizon_weights

    @property
    def point_loss_id(self) -> str:
        return self.fit_strategy.point_loss_id

    @property
    def entity_aggregation_mode(self) -> str:
        return self.fit_strategy.entity_aggregation_mode

    def as_diagnostics(self) -> dict[str, Any]:
        return {
            "aggregated_primary_score": self.aggregated_primary_score,
            "entity_aggregation_mode": self.entity_aggregation_mode,
            "entity_weights": [dict(item) for item in self.entity_weights],
            "fit_strategy_identity": self.fit_strategy_identity,
            "horizon_set": list(self.horizon_set),
            "horizon_weights": [dict(item) for item in self.horizon_weights],
            "per_horizon": [
                {
                    "horizon": horizon,
                    "valid_origin_count": valid_origin_count,
                    "mean_point_loss": mean_point_loss,
                }
                for horizon, valid_origin_count, mean_point_loss in self.per_horizon
            ],
            "point_loss_id": self.point_loss_id,
            "row_count": len(self.rows),
            "training_origin_set_id": self.training_origin_set_id,
        }


def resolve_fit_strategy(
    *,
    strategy_id: str = "legacy_one_step",
    horizon_set: Sequence[int] | None = None,
    horizon_weights: Sequence[Mapping[str, Any] | tuple[int, Any]]
    | Mapping[int, Any]
    | None = None,
    point_loss_id: str = "squared_error",
    entity_aggregation_mode: str = _DEFAULT_ENTITY_AGGREGATION_MODE,
) -> FitStrategySpec:
    resolved_horizon_set = _resolve_horizon_set(
        (1,) if horizon_set is None else horizon_set
    )
    resolved_horizon_weights = _resolve_horizon_weights(
        horizon_set=resolved_horizon_set,
        horizon_weights=horizon_weights,
    )
    return FitStrategySpec(
        strategy_id=strategy_id,
        horizon_set=resolved_horizon_set,
        horizon_weights=resolved_horizon_weights,
        point_loss_id=point_loss_id,
        entity_aggregation_mode=entity_aggregation_mode,
    )


def evaluate_rollout_objective(
    *,
    candidate: CandidateIntermediateRepresentation,
    fit_result: Any,
    feature_view: FeatureView,
    fit_window: EvaluationSegment,
    fit_strategy: FitStrategySpec | None = None,
) -> RolloutObjectiveResult:
    from euclid.modules.forecast_paths import forecast_path
    from euclid.modules.scoring import _aggregate_primary_scores, _point_loss
    from euclid.modules.split_planning import build_legal_training_origin_panel

    strategy = fit_strategy or resolve_fit_strategy()
    legal_feature_view = feature_view.require_stage_reuse("candidate_fitting")
    training_origin_panel = build_legal_training_origin_panel(
        feature_view=legal_feature_view,
        evaluation_segment=fit_window,
        horizon_set=strategy.horizon_set,
    )
    if training_origin_panel.status != "passed":
        raise ContractValidationError(
            code="incomplete_rollout_objective_panel",
            message="rollout objectives require a complete legal training-origin panel",
            field_path="fit_window",
            details={
                "diagnostics": [
                    diagnostic.as_dict()
                    for diagnostic in training_origin_panel.diagnostics
                ],
            },
        )

    rows_by_entity = _rows_by_entity(legal_feature_view)
    rows: list[dict[str, Any]] = []
    path_cache: dict[tuple[str, int], Mapping[int, float]] = {}
    max_horizon = max(strategy.horizon_set)
    for panel_row in training_origin_panel.records:
        entity_rows = rows_by_entity[panel_row.entity]
        origin_row = entity_rows[panel_row.origin_index]
        target_row = entity_rows[panel_row.target_index]
        cache_key = (panel_row.entity, panel_row.origin_index)
        predictions = path_cache.get(cache_key)
        if predictions is None:
            path = forecast_path(
                candidate=candidate,
                fit_result=fit_result,
                origin_row=origin_row,
                max_horizon=max_horizon,
                entity=panel_row.entity,
            )
            predictions = path.predictions
            path_cache[cache_key] = predictions
        point_forecast = predictions.get(panel_row.horizon)
        if point_forecast is None or not math.isfinite(float(point_forecast)):
            raise ContractValidationError(
                code="missing_declared_horizon_forecast",
                message="rollout objective could not produce a declared horizon forecast",
                field_path="fit_strategy.horizon_set",
                details={
                    "horizon": panel_row.horizon,
                    "origin_index": panel_row.origin_index,
                },
            )
        realized = float(target_row["target"])
        if not math.isfinite(realized):
            raise ContractValidationError(
                code="nonfinite_observation",
                message="rollout objective target observations must be finite",
                field_path="feature_view.rows.target",
                details={"target_index": panel_row.target_index},
            )
        rows.append(
            _objective_row(
                panel_row=panel_row,
                point_forecast=float(point_forecast),
                realized_observation=realized,
                include_entity=len(training_origin_panel.entity_panel) > 1,
            )
        )

    objective_rows = tuple(rows)
    entity_weights = _default_entity_weights(
        training_origin_panel.entity_panel,
        strategy.entity_aggregation_mode,
    )
    per_horizon, aggregated_primary_score = _aggregate_primary_scores(
        rows=objective_rows,
        horizon_weights=tuple(
            (int(item["horizon"]), float(item["weight"]))
            for item in strategy.horizon_weights
        ),
        entity_aggregation_mode=strategy.entity_aggregation_mode,
        entity_weights=tuple(
            (str(item["entity"]), float(item["weight"])) for item in entity_weights
        ),
        row_score=lambda row: _point_loss(
            point_loss_id=strategy.point_loss_id,
            point_forecast=float(row["point_forecast"]),
            realized_observation=float(row["realized_observation"]),
        ),
    )
    training_origin_set_id = sha256_digest(
        [record.as_dict() for record in training_origin_panel.records]
    )
    return RolloutObjectiveResult(
        fit_window_id=fit_window.segment_id,
        fit_strategy=strategy,
        training_origin_panel=training_origin_panel,
        training_origin_set_id=training_origin_set_id,
        rows=objective_rows,
        per_horizon=per_horizon,
        aggregated_primary_score=_stable_float(aggregated_primary_score),
        entity_weights=entity_weights,
    )


def training_origin_set_id(training_origin_panel: TrainingOriginPanel) -> str:
    return sha256_digest([record.as_dict() for record in training_origin_panel.records])


def _resolve_horizon_set(horizon_set: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_horizon in horizon_set:
        horizon = int(raw_horizon)
        if horizon <= 0:
            raise ContractValidationError(
                code="invalid_fit_strategy",
                message="fit strategy horizon_set values must be positive integers",
                field_path="fit_strategy.horizon_set",
            )
        if horizon not in seen:
            seen.add(horizon)
            normalized.append(horizon)
    if not normalized:
        raise ContractValidationError(
            code="invalid_fit_strategy",
            message="fit strategies require at least one horizon",
            field_path="fit_strategy.horizon_set",
        )
    return tuple(sorted(normalized))


def _resolve_horizon_weights(
    *,
    horizon_set: tuple[int, ...],
    horizon_weights: Sequence[Mapping[str, Any] | tuple[int, Any]]
    | Mapping[int, Any]
    | None,
) -> tuple[dict[str, Any], ...]:
    raw_weights = (
        _default_horizon_weights(horizon_set)
        if horizon_weights is None
        else _normalize_raw_horizon_weights(horizon_weights)
    )
    weights_by_horizon: dict[int, Decimal] = {}
    total = Decimal("0")
    for index, item in enumerate(raw_weights):
        horizon = int(item["horizon"])
        if horizon <= 0 or horizon in weights_by_horizon:
            raise ContractValidationError(
                code="invalid_horizon_weight_simplex",
                message="fit strategy horizon weights require unique positive horizons",
                field_path=f"fit_strategy.horizon_weights[{index}].horizon",
            )
        try:
            weight = Decimal(str(item["weight"]))
        except (InvalidOperation, ValueError) as exc:
            raise ContractValidationError(
                code="invalid_horizon_weight_simplex",
                message="fit strategy horizon weights must be finite decimals",
                field_path=f"fit_strategy.horizon_weights[{index}].weight",
            ) from exc
        if not weight.is_finite() or weight < 0:
            raise ContractValidationError(
                code="invalid_horizon_weight_simplex",
                message="fit strategy horizon weights must be finite decimals >= 0",
                field_path=f"fit_strategy.horizon_weights[{index}].weight",
            )
        weights_by_horizon[horizon] = weight
        total += weight

    if tuple(sorted(weights_by_horizon)) != horizon_set:
        raise ContractValidationError(
            code="invalid_horizon_weight_simplex",
            message="fit strategy horizon weights must cover horizon_set exactly",
            field_path="fit_strategy.horizon_weights",
            details={
                "horizon_set": list(horizon_set),
                "weight_horizons": sorted(weights_by_horizon),
            },
        )
    if total != Decimal("1") or not math.isfinite(float(total)):
        raise ContractValidationError(
            code="invalid_horizon_weight_simplex",
            message="fit strategy horizon weights must sum exactly to 1",
            field_path="fit_strategy.horizon_weights",
        )
    return tuple(
        {"horizon": horizon, "weight": _decimal_string(weights_by_horizon[horizon])}
        for horizon in horizon_set
    )


def _normalize_raw_horizon_weights(
    horizon_weights: Sequence[Mapping[str, Any] | tuple[int, Any]] | Mapping[int, Any],
) -> tuple[dict[str, Any], ...]:
    if isinstance(horizon_weights, Mapping):
        return tuple(
            {"horizon": horizon, "weight": weight}
            for horizon, weight in horizon_weights.items()
        )
    normalized = []
    for item in horizon_weights:
        if isinstance(item, Mapping):
            normalized.append(
                {
                    "horizon": item["horizon"],
                    "weight": item["weight"],
                }
            )
        else:
            horizon, weight = item
            normalized.append({"horizon": horizon, "weight": weight})
    return tuple(normalized)


def _default_horizon_weights(horizon_set: tuple[int, ...]) -> tuple[dict[str, Any], ...]:
    if len(horizon_set) == 1:
        return ({"horizon": horizon_set[0], "weight": "1"},)
    scale = Decimal("0.000000000001")
    count = Decimal(len(horizon_set))
    base_weight = (Decimal("1") / count).quantize(scale, rounding=ROUND_DOWN)
    remaining = Decimal("1")
    weights: list[dict[str, Any]] = []
    for index, horizon in enumerate(horizon_set):
        if index == len(horizon_set) - 1:
            weight = remaining
        else:
            weight = base_weight
            remaining -= weight
        weights.append({"horizon": horizon, "weight": _decimal_string(weight)})
    return tuple(weights)


def _default_entity_weights(
    entity_panel: tuple[str, ...],
    entity_aggregation_mode: str,
) -> tuple[dict[str, Any], ...]:
    if entity_aggregation_mode == _DEFAULT_ENTITY_AGGREGATION_MODE:
        return ()
    if not entity_panel:
        return ()
    scale = Decimal("0.000000000001")
    count = Decimal(len(entity_panel))
    base_weight = (Decimal("1") / count).quantize(scale, rounding=ROUND_DOWN)
    remaining = Decimal("1")
    weights: list[dict[str, Any]] = []
    for index, entity in enumerate(entity_panel):
        if index == len(entity_panel) - 1:
            weight = remaining
        else:
            weight = base_weight
            remaining -= weight
        weights.append({"entity": entity, "weight": _decimal_string(weight)})
    return tuple(weights)


def _objective_row(
    *,
    panel_row: "TrainingOriginPanelRow",
    point_forecast: float,
    realized_observation: float,
    include_entity: bool,
) -> dict[str, Any]:
    row = {
        "available_at": panel_row.target_available_at,
        "horizon": panel_row.horizon,
        "origin_index": panel_row.origin_index,
        "origin_time": panel_row.origin_time,
        "point_forecast": _stable_float(point_forecast),
        "realized_observation": _stable_float(realized_observation),
        "target_index": panel_row.target_index,
        "training_origin_id": panel_row.training_origin_id,
    }
    if include_entity:
        row["entity"] = panel_row.entity
    return row


def _rows_by_entity(
    feature_view: FeatureView,
) -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in feature_view.rows:
        entity = str(row.get("entity", feature_view.series_id))
        grouped.setdefault(entity, []).append(dict(row))
    return {entity: tuple(rows) for entity, rows in grouped.items()}


def _decimal_string(value: Decimal) -> str:
    rendered = format(value.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


__all__ = [
    "FitStrategySpec",
    "RolloutObjectiveResult",
    "evaluate_rollout_objective",
    "resolve_fit_strategy",
    "training_origin_set_id",
]
