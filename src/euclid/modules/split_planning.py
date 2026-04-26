from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.modules.features import FeatureView
from euclid.modules.timeguard import TimeSafetyAudit
from euclid.runtime.hashing import sha256_digest

_ADMITTED_FORECAST_OBJECT_TYPES = (
    "point",
    "distribution",
    "interval",
    "quantile",
    "event_probability",
)


@dataclass(frozen=True)
class HorizonWeight:
    horizon: int
    weight: str

    def as_dict(self) -> dict[str, Any]:
        return {"horizon": self.horizon, "weight": self.weight}


@dataclass(frozen=True)
class ScoredOrigin:
    scored_origin_id: str
    segment_id: str
    outer_fold_id: str
    role: str
    origin_index: int
    origin_time: str
    available_at: str
    horizon: int
    target_index: int
    target_event_time: str
    entity: str | None = None

    def as_dict(self) -> dict[str, Any]:
        body = {
            "available_at": self.available_at,
            "horizon": self.horizon,
            "origin_index": self.origin_index,
            "origin_time": self.origin_time,
            "outer_fold_id": self.outer_fold_id,
            "role": self.role,
            "scored_origin_id": self.scored_origin_id,
            "segment_id": self.segment_id,
            "target_event_time": self.target_event_time,
            "target_index": self.target_index,
        }
        if self.entity is not None:
            body["entity"] = self.entity
        return body


@dataclass(frozen=True)
class TrainingOriginPanelRow:
    training_origin_id: str
    segment_id: str
    outer_fold_id: str
    split_role: str
    entity: str
    origin_index: int
    origin_time: str
    origin_available_at: str
    horizon: int
    target_index: int
    target_event_time: str
    target_available_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "horizon": self.horizon,
            "origin_available_at": self.origin_available_at,
            "origin_index": self.origin_index,
            "origin_time": self.origin_time,
            "outer_fold_id": self.outer_fold_id,
            "segment_id": self.segment_id,
            "split_role": self.split_role,
            "target_available_at": self.target_available_at,
            "target_event_time": self.target_event_time,
            "target_index": self.target_index,
            "training_origin_id": self.training_origin_id,
        }


@dataclass(frozen=True)
class TrainingOriginPanelDiagnostic:
    code: str
    message: str
    entity: str | None = None
    origin_index: int | None = None
    horizon: int | None = None
    target_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.entity is not None:
            body["entity"] = self.entity
        if self.origin_index is not None:
            body["origin_index"] = self.origin_index
        if self.horizon is not None:
            body["horizon"] = self.horizon
        if self.target_index is not None:
            body["target_index"] = self.target_index
        return body


@dataclass(frozen=True)
class TrainingOriginPanel:
    segment_id: str
    split_role: str
    horizon_set: tuple[int, ...]
    entity_panel: tuple[str, ...]
    records: tuple[TrainingOriginPanelRow, ...]
    diagnostics: tuple[TrainingOriginPanelDiagnostic, ...] = ()

    @property
    def status(self) -> str:
        return "failed" if self.diagnostics else "passed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "diagnostics": [diagnostic.as_dict() for diagnostic in self.diagnostics],
            "entity_panel": list(self.entity_panel),
            "horizon_set": list(self.horizon_set),
            "records": [record.as_dict() for record in self.records],
            "segment_id": self.segment_id,
            "split_role": self.split_role,
            "status": self.status,
        }


@dataclass(frozen=True)
class EvaluationSegment:
    segment_id: str
    outer_fold_id: str
    role: str
    train_start_index: int
    train_end_index: int
    test_start_index: int
    test_end_index: int
    train_row_count: int
    test_row_count: int
    origin_index: int
    origin_time: str
    train_end_event_time: str
    test_end_event_time: str
    horizon_set: tuple[int, ...]
    scored_origin_ids: tuple[str, ...]
    entity_panel: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        body = {
            "horizon_set": list(self.horizon_set),
            "origin_index": self.origin_index,
            "origin_time": self.origin_time,
            "outer_fold_id": self.outer_fold_id,
            "role": self.role,
            "scored_origin_ids": list(self.scored_origin_ids),
            "segment_id": self.segment_id,
            "test_end_event_time": self.test_end_event_time,
            "test_end_index": self.test_end_index,
            "test_row_count": self.test_row_count,
            "test_start_index": self.test_start_index,
            "train_end_event_time": self.train_end_event_time,
            "train_end_index": self.train_end_index,
            "train_row_count": self.train_row_count,
            "train_start_index": self.train_start_index,
        }
        if self.entity_panel:
            body["entity_panel"] = list(self.entity_panel)
        return body


@dataclass(frozen=True)
class EvaluationPlan:
    folds: tuple[dict[str, Any], ...]
    horizon_set: tuple[int, ...]
    horizon_weights: tuple[HorizonWeight, ...]
    development_segments: tuple[EvaluationSegment, ...]
    confirmatory_segment: EvaluationSegment
    scored_origin_panel: tuple[ScoredOrigin, ...]
    scored_origin_set_id: str
    entity_panel: tuple[str, ...] = ()
    forecast_object_type: str = "point"

    @property
    def outer_fold_count(self) -> int:
        return len(self.folds)

    @property
    def comparison_key(self) -> dict[str, Any]:
        body = {
            "forecast_object_type": self.forecast_object_type,
            "horizon_set": list(self.horizon_set),
            "scored_origin_set_id": self.scored_origin_set_id,
        }
        if len(self.entity_panel) > 1:
            body["entity_panel"] = list(self.entity_panel)
        return body

    def to_manifest(
        self,
        catalog: ContractCatalog,
        *,
        time_safety_audit_ref: TypedRef,
        feature_view_ref: TypedRef | None = None,
    ) -> ManifestEnvelope:
        body: dict[str, Any] = {
            "evaluation_plan_id": "prototype_nested_walk_forward_plan_v1",
            "owner_prompt_id": "prompt.predictive-validation-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": self.forecast_object_type,
            "split_strategy": "nested_walk_forward_only",
            "inner_search_policy": "fold_local_only",
            "outer_fold_count": self.outer_fold_count,
            "confirmatory_holdout_policy": "single_use_sealed",
            "max_confirmatory_accesses": 1,
            "reusable_holdout_policy": "not_supported",
            "time_safety_audit_ref": time_safety_audit_ref.as_dict(),
            "folds": list(self.folds),
            "horizon_weights": [weight.as_dict() for weight in self.horizon_weights],
            "development_segments": [
                segment.as_dict() for segment in self.development_segments
            ],
            "confirmatory_segment": self.confirmatory_segment.as_dict(),
            "scored_origin_panel": [
                origin.as_dict() for origin in self.scored_origin_panel
            ],
            "scored_origin_set_id": self.scored_origin_set_id,
            "comparison_key": self.comparison_key,
        }
        if self.entity_panel:
            body["entity_panel"] = list(self.entity_panel)
        if feature_view_ref is not None:
            body["feature_view_ref"] = feature_view_ref.as_dict()
        return ManifestEnvelope.build(
            schema_name="evaluation_plan_manifest@1.1.0",
            module_id="split_planning",
            body=body,
            catalog=catalog,
        )


def build_evaluation_plan(
    *,
    feature_view: FeatureView,
    audit: TimeSafetyAudit,
    min_train_size: int,
    horizon: int,
    horizon_weight_strings: tuple[str, ...] | None = None,
    forecast_object_type: str = "point",
) -> EvaluationPlan:
    if audit.status != "passed":
        raise ContractValidationError(
            code="time_safety_blocked",
            message="cannot bind an evaluation plan against a blocked audit",
            field_path="audit.status",
        )
    _require_forecast_object_type(forecast_object_type)
    feature_view.require_stage_reuse("evaluation")
    if min_train_size < 1 or horizon < 1:
        raise ContractValidationError(
            code="invalid_fold_geometry",
            message="min_train_size and horizon must be positive integers",
            field_path="fold_geometry",
        )

    horizon_set = tuple(range(1, horizon + 1))
    horizon_weights = _resolve_horizon_weights(
        horizon_set=horizon_set,
        horizon_weight_strings=horizon_weight_strings,
    )

    folds: list[dict[str, Any]] = []
    segments: list[EvaluationSegment] = []
    scored_origin_panel: list[ScoredOrigin] = []
    entity_panel, rows_by_entity = _rows_by_entity(feature_view)
    aligned_row_count = min(len(rows) for rows in rows_by_entity.values())
    for train_end in range(min_train_size, aligned_row_count):
        test_end = train_end + max(horizon_set)
        if test_end > aligned_row_count:
            break
        fold_index = len(folds)
        outer_fold_id = f"outer_fold_{fold_index}"
        role = (
            "confirmatory_holdout" if test_end == aligned_row_count else "development"
        )
        segment_id = (
            "confirmatory_holdout" if role == "confirmatory_holdout" else outer_fold_id
        )
        origin_index = train_end - 1
        first_entity = entity_panel[0]
        first_train_rows = rows_by_entity[first_entity][:train_end]
        first_test_rows = rows_by_entity[first_entity][train_end:test_end]
        origin_time = str(first_train_rows[-1]["event_time"])
        scored_origin_ids: list[str] = []
        panel_train_row_count = 0
        panel_test_row_count = 0
        for entity in entity_panel:
            entity_train_rows = rows_by_entity[entity][:train_end]
            entity_test_rows = rows_by_entity[entity][train_end:test_end]
            panel_train_row_count += len(entity_train_rows)
            panel_test_row_count += len(entity_test_rows)
            entity_origin_time = str(entity_train_rows[-1]["event_time"])
            for offset, test_row in enumerate(entity_test_rows, start=1):
                scored_origin_id = (
                    f"{outer_fold_id}__{entity}__h{offset}"
                    if len(entity_panel) > 1
                    else f"{outer_fold_id}_h{offset}"
                )
                scored_origin = ScoredOrigin(
                    scored_origin_id=scored_origin_id,
                    segment_id=segment_id,
                    outer_fold_id=outer_fold_id,
                    role=role,
                    origin_index=origin_index,
                    origin_time=entity_origin_time,
                    available_at=str(test_row["available_at"]),
                    horizon=offset,
                    target_index=train_end + offset - 1,
                    target_event_time=str(test_row["event_time"]),
                    entity=entity if len(entity_panel) > 1 else None,
                )
                scored_origin_ids.append(scored_origin.scored_origin_id)
                scored_origin_panel.append(scored_origin)
        segment = EvaluationSegment(
            segment_id=segment_id,
            outer_fold_id=outer_fold_id,
            role=role,
            train_start_index=0,
            train_end_index=train_end - 1,
            test_start_index=train_end,
            test_end_index=test_end - 1,
            train_row_count=panel_train_row_count,
            test_row_count=panel_test_row_count,
            origin_index=origin_index,
            origin_time=origin_time,
            train_end_event_time=str(first_train_rows[-1]["event_time"]),
            test_end_event_time=str(first_test_rows[-1]["event_time"]),
            horizon_set=horizon_set,
            scored_origin_ids=tuple(scored_origin_ids),
            entity_panel=entity_panel if len(entity_panel) > 1 else (),
        )
        segments.append(segment)
        fold = {
            "fold_index": fold_index,
            "fold_id": outer_fold_id,
            "outer_fold_id": outer_fold_id,
            "segment_id": segment_id,
            "role": role,
            "train_start_index": 0,
            "train_end_index": train_end - 1,
            "test_start_index": train_end,
            "test_end_index": test_end - 1,
            "train_row_count": panel_train_row_count,
            "test_row_count": panel_test_row_count,
            "train_end_event_time": first_train_rows[-1]["event_time"],
            "test_end_event_time": first_test_rows[-1]["event_time"],
            "origin_index": origin_index,
            "origin_time": origin_time,
            "horizon_set": list(segment.horizon_set),
            "horizon_weights": [weight.as_dict() for weight in horizon_weights],
            "scored_origin_ids": list(segment.scored_origin_ids),
        }
        if len(entity_panel) > 1:
            fold["entity_panel"] = list(entity_panel)
        folds.append(fold)

    if not folds:
        raise ContractValidationError(
            code="invalid_fold_geometry",
            message="not enough rows for a walk-forward evaluation plan",
            field_path="feature_view.rows",
        )

    development_segments = tuple(
        segment for segment in segments if segment.role == "development"
    )
    confirmatory_segment = next(
        segment for segment in segments if segment.role == "confirmatory_holdout"
    )
    scored_origin_set_id = sha256_digest(
        [origin.as_dict() for origin in scored_origin_panel]
    )
    return EvaluationPlan(
        folds=tuple(folds),
        horizon_set=horizon_set,
        horizon_weights=horizon_weights,
        development_segments=development_segments,
        confirmatory_segment=confirmatory_segment,
        scored_origin_panel=tuple(scored_origin_panel),
        scored_origin_set_id=scored_origin_set_id,
        entity_panel=entity_panel,
        forecast_object_type=forecast_object_type,
    )


def segment_training_rows(
    *,
    feature_view: FeatureView,
    evaluation_segment: EvaluationSegment,
) -> tuple[dict[str, Any], ...]:
    rows_by_entity = _rows_by_entity(feature_view)[1]
    return tuple(
        dict(row)
        for entity in _segment_entity_panel(feature_view, evaluation_segment)
        for row in rows_by_entity[entity][
            evaluation_segment.train_start_index : evaluation_segment.train_end_index
            + 1
        ]
    )


def resolve_scored_origin_target_row(
    *,
    feature_view: FeatureView,
    scored_origin: ScoredOrigin,
) -> dict[str, Any] | None:
    entity = scored_origin.entity or feature_view.series_id
    rows_by_entity = _rows_by_entity(feature_view)[1]
    entity_rows = rows_by_entity.get(entity)
    if entity_rows is None or scored_origin.target_index >= len(entity_rows):
        return None
    return dict(entity_rows[scored_origin.target_index])


def build_legal_training_origin_panel(
    *,
    feature_view: FeatureView,
    evaluation_segment: EvaluationSegment,
    horizon_set: tuple[int, ...] | None = None,
) -> TrainingOriginPanel:
    feature_view.require_stage_reuse("candidate_fitting")
    resolved_horizon_set = _resolve_training_panel_horizon_set(
        evaluation_segment.horizon_set if horizon_set is None else horizon_set
    )
    entity_panel = _segment_entity_panel(feature_view, evaluation_segment)
    rows_by_entity = _rows_by_entity(feature_view)[1]
    diagnostics: list[TrainingOriginPanelDiagnostic] = []
    records: list[TrainingOriginPanelRow] = []
    max_origin_index = evaluation_segment.train_end_index - max(resolved_horizon_set)
    if max_origin_index < evaluation_segment.train_start_index:
        target_index = evaluation_segment.train_start_index + max(resolved_horizon_set)
        diagnostics.extend(
            (
                TrainingOriginPanelDiagnostic(
                    code="missing_horizon_target",
                    message=(
                        "no complete training-origin panel exists for the declared "
                        "horizon set inside the training slice"
                    ),
                    origin_index=evaluation_segment.train_start_index,
                    horizon=max(resolved_horizon_set),
                    target_index=target_index,
                ),
                TrainingOriginPanelDiagnostic(
                    code="target_outside_training_slice",
                    message="target row falls outside the training slice",
                    origin_index=evaluation_segment.train_start_index,
                    horizon=max(resolved_horizon_set),
                    target_index=target_index,
                ),
            )
        )
        return TrainingOriginPanel(
            segment_id=evaluation_segment.segment_id,
            split_role=evaluation_segment.role,
            horizon_set=resolved_horizon_set,
            entity_panel=entity_panel,
            records=(),
            diagnostics=tuple(diagnostics),
        )

    for origin_index in range(
        evaluation_segment.train_start_index,
        max_origin_index + 1,
    ):
        origin_records: list[TrainingOriginPanelRow] = []
        complete_origin = True
        for entity in entity_panel:
            entity_rows = rows_by_entity.get(entity, ())
            if origin_index >= len(entity_rows):
                diagnostics.append(
                    TrainingOriginPanelDiagnostic(
                        code="missing_entity_origin",
                        message="entity is missing the requested origin row",
                        entity=entity,
                        origin_index=origin_index,
                    )
                )
                complete_origin = False
                continue
            origin_row = entity_rows[origin_index]
            for horizon in resolved_horizon_set:
                target_index = origin_index + horizon
                if target_index > evaluation_segment.train_end_index:
                    diagnostics.append(
                        TrainingOriginPanelDiagnostic(
                            code="target_outside_training_slice",
                            message="target row falls outside the training slice",
                            entity=entity,
                            origin_index=origin_index,
                            horizon=horizon,
                            target_index=target_index,
                        )
                    )
                    complete_origin = False
                    continue
                if target_index >= len(entity_rows):
                    diagnostics.append(
                        TrainingOriginPanelDiagnostic(
                            code="missing_entity_target",
                            message="entity is missing the requested horizon target",
                            entity=entity,
                            origin_index=origin_index,
                            horizon=horizon,
                            target_index=target_index,
                        )
                    )
                    complete_origin = False
                    continue
                target_row = entity_rows[target_index]
                row_diagnostic = _training_panel_row_diagnostic(
                    entity=entity,
                    origin_index=origin_index,
                    horizon=horizon,
                    origin_row=origin_row,
                    target_row=target_row,
                )
                if row_diagnostic is not None:
                    diagnostics.append(row_diagnostic)
                    complete_origin = False
                    continue
                origin_records.append(
                    TrainingOriginPanelRow(
                        training_origin_id=(
                            f"{evaluation_segment.segment_id}__{entity}"
                            f"__o{origin_index}__h{horizon}"
                        ),
                        segment_id=evaluation_segment.segment_id,
                        outer_fold_id=evaluation_segment.outer_fold_id,
                        split_role=evaluation_segment.role,
                        entity=entity,
                        origin_index=origin_index,
                        origin_time=str(origin_row["event_time"]),
                        origin_available_at=str(origin_row["available_at"]),
                        horizon=horizon,
                        target_index=target_index,
                        target_event_time=str(target_row["event_time"]),
                        target_available_at=str(target_row["available_at"]),
                    )
                )
        if complete_origin:
            records.extend(origin_records)

    if diagnostics:
        records = []
    return TrainingOriginPanel(
        segment_id=evaluation_segment.segment_id,
        split_role=evaluation_segment.role,
        horizon_set=resolved_horizon_set,
        entity_panel=entity_panel,
        records=tuple(records),
        diagnostics=tuple(diagnostics),
    )


def _segment_entity_panel(
    feature_view: FeatureView,
    evaluation_segment: EvaluationSegment,
) -> tuple[str, ...]:
    if evaluation_segment.entity_panel:
        return evaluation_segment.entity_panel
    return _rows_by_entity(feature_view)[0]


def _rows_by_entity(
    feature_view: FeatureView,
) -> tuple[tuple[str, ...], dict[str, tuple[dict[str, Any], ...]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in feature_view.rows:
        entity = str(row.get("entity", feature_view.series_id))
        grouped.setdefault(entity, []).append(dict(row))
    if feature_view.entity_panel:
        entity_panel = feature_view.entity_panel
    else:
        entity_panel = tuple(sorted(grouped))
    return entity_panel, {
        entity: tuple(grouped.get(entity, ())) for entity in entity_panel
    }


def _resolve_training_panel_horizon_set(
    horizon_set: tuple[int, ...],
) -> tuple[int, ...]:
    normalized: list[int] = []
    for horizon in horizon_set:
        resolved = int(horizon)
        if resolved < 1:
            raise ContractValidationError(
                code="invalid_training_origin_panel",
                message="training-origin horizon values must be positive integers",
                field_path="horizon_set",
            )
        if resolved not in normalized:
            normalized.append(resolved)
    if not normalized:
        raise ContractValidationError(
            code="invalid_training_origin_panel",
            message="training-origin panels require at least one horizon",
            field_path="horizon_set",
        )
    return tuple(normalized)


def _training_panel_row_diagnostic(
    *,
    entity: str,
    origin_index: int,
    horizon: int,
    origin_row: dict[str, Any],
    target_row: dict[str, Any],
) -> TrainingOriginPanelDiagnostic | None:
    for field_name, row in (
        ("origin_available_at", origin_row),
        ("target_available_at", target_row),
    ):
        source_field = "available_at"
        if source_field not in row:
            return TrainingOriginPanelDiagnostic(
                code="missing_origin_or_target_availability",
                message=f"{field_name} cannot be derived from the feature row",
                entity=entity,
                origin_index=origin_index,
                horizon=horizon,
            )
    if "target" not in target_row or target_row["target"] is None:
        return TrainingOriginPanelDiagnostic(
            code="missing_entity_target",
            message="target row is missing an observed target value",
            entity=entity,
            origin_index=origin_index,
            horizon=horizon,
            target_index=origin_index + horizon,
        )
    return None


def _resolve_horizon_weights(
    *,
    horizon_set: tuple[int, ...],
    horizon_weight_strings: tuple[str, ...] | None,
) -> tuple[HorizonWeight, ...]:
    if horizon_weight_strings is None:
        horizon_weight_strings = _default_horizon_weight_strings(len(horizon_set))
    if len(horizon_weight_strings) != len(horizon_set):
        raise ContractValidationError(
            code="invalid_horizon_weight_simplex",
            message=(
                "horizon_weight_strings must contain exactly one weight per "
                "declared horizon"
            ),
            field_path="horizon_weight_strings",
            details={
                "expected_count": len(horizon_set),
                "actual_count": len(horizon_weight_strings),
            },
        )

    parsed_weights: list[Decimal] = []
    for index, weight_string in enumerate(horizon_weight_strings):
        try:
            weight = Decimal(weight_string)
        except (InvalidOperation, ValueError) as exc:
            raise ContractValidationError(
                code="invalid_horizon_weight_simplex",
                message="horizon weights must be finite decimal strings",
                field_path=f"horizon_weight_strings[{index}]",
            ) from exc
        if not weight.is_finite() or weight < 0:
            raise ContractValidationError(
                code="invalid_horizon_weight_simplex",
                message="horizon weights must be finite decimals >= 0",
                field_path=f"horizon_weight_strings[{index}]",
            )
        parsed_weights.append(weight)

    if sum(parsed_weights) != Decimal("1"):
        raise ContractValidationError(
            code="invalid_horizon_weight_simplex",
            message="horizon weights must sum exactly to 1",
            field_path="horizon_weight_strings",
            details={"weights": list(horizon_weight_strings)},
        )

    return tuple(
        HorizonWeight(horizon=horizon, weight=weight_string)
        for horizon, weight_string in zip(
            horizon_set,
            horizon_weight_strings,
            strict=True,
        )
    )


def _default_horizon_weight_strings(horizon_count: int) -> tuple[str, ...]:
    if horizon_count == 1:
        return ("1",)
    precision = Decimal("0.000000000001")
    base = (Decimal("1") / Decimal(horizon_count)).quantize(precision)
    weights = [base for _ in range(horizon_count - 1)]
    weights.append(Decimal("1") - sum(weights))
    return tuple(_decimal_to_string(weight) for weight in weights)


def _decimal_to_string(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _require_forecast_object_type(forecast_object_type: str) -> None:
    if forecast_object_type in _ADMITTED_FORECAST_OBJECT_TYPES:
        return
    raise ContractValidationError(
        code="unsupported_forecast_object_type",
        message=(
            "forecast_object_type must be one of "
            f"{', '.join(_ADMITTED_FORECAST_OBJECT_TYPES)}"
        ),
        field_path="forecast_object_type",
        details={"forecast_object_type": forecast_object_type},
    )
