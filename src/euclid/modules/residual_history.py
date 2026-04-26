from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from euclid.contracts.errors import ContractValidationError
from euclid.runtime.hashing import canonicalize_json, sha256_digest

_RESIDUAL_BASIS = "observation_minus_point_forecast"
_PRODUCTION_FORBIDDEN_SPLIT_ROLES = frozenset({"confirmatory", "confirmatory_holdout"})


@dataclass(frozen=True)
class ForecastResidualRecord:
    candidate_id: str
    fit_window_id: str
    entity: str
    origin_index: int
    origin_time: str
    origin_available_at: str
    target_index: int
    target_event_time: str
    target_available_at: str
    horizon: int
    point_forecast: float
    realized_observation: float
    residual: float
    split_role: str
    residual_basis: str
    time_safety_status: str
    replay_identity: str
    weight: float | None = None
    component_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_id",
            "fit_window_id",
            "entity",
            "origin_time",
            "origin_available_at",
            "target_event_time",
            "target_available_at",
            "split_role",
            "residual_basis",
            "time_safety_status",
            "replay_identity",
        ):
            value = getattr(self, field_name)
            if not str(value).strip():
                raise ContractValidationError(
                    code="invalid_residual_record",
                    message=f"{field_name} must be a non-empty string",
                    field_path=field_name,
                )
        if self.origin_index < 0 or self.target_index < 0:
            raise ContractValidationError(
                code="invalid_residual_geometry",
                message="origin_index and target_index must be non-negative",
                field_path="origin_index",
            )
        if self.horizon < 1:
            raise ContractValidationError(
                code="invalid_residual_geometry",
                message="horizon must be a positive integer",
                field_path="horizon",
            )
        if self.target_index != self.origin_index + self.horizon:
            raise ContractValidationError(
                code="invalid_residual_geometry",
                message="target_index must equal origin_index + horizon",
                field_path="target_index",
                details={
                    "origin_index": self.origin_index,
                    "horizon": self.horizon,
                    "target_index": self.target_index,
                },
            )
        for field_name in ("point_forecast", "realized_observation", "residual"):
            _require_finite_float(getattr(self, field_name), field_name=field_name)
        if self.weight is not None:
            _require_finite_float(self.weight, field_name="weight")

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "fit_window_id": self.fit_window_id,
            "entity": self.entity,
            "origin_index": self.origin_index,
            "origin_time": self.origin_time,
            "origin_available_at": self.origin_available_at,
            "target_index": self.target_index,
            "target_event_time": self.target_event_time,
            "target_available_at": self.target_available_at,
            "horizon": self.horizon,
            "point_forecast": float(self.point_forecast),
            "realized_observation": float(self.realized_observation),
            "realized_value": float(self.realized_observation),
            "residual": float(self.residual),
            "split_role": self.split_role,
            "residual_basis": self.residual_basis,
            "time_safety_status": self.time_safety_status,
            "replay_identity": self.replay_identity,
        }
        if self.weight is not None:
            body["weight"] = float(self.weight)
        if self.component_id is not None:
            body["component_id"] = self.component_id
        return body


ResidualRecordLike = ForecastResidualRecord | Mapping[str, Any]


@dataclass(frozen=True)
class ResidualHistorySummary:
    candidate_id: str
    fit_window_id: str
    residual_count: int
    horizon_set: tuple[int, ...]
    entity_count: int
    split_roles: tuple[str, ...]
    residual_mean: float
    residual_rmse: float
    residual_basis: str
    residual_history_digest: str
    source_row_set_digest: str
    replay_identity: str
    weighted_residual_mean: float | None = None
    weighted_residual_rmse: float | None = None

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "fit_window_id": self.fit_window_id,
            "residual_count": self.residual_count,
            "horizon_set": list(self.horizon_set),
            "entity_count": self.entity_count,
            "split_roles": list(self.split_roles),
            "residual_mean": self.residual_mean,
            "residual_rmse": self.residual_rmse,
            "residual_basis": self.residual_basis,
            "residual_history_digest": self.residual_history_digest,
            "source_row_set_digest": self.source_row_set_digest,
            "replay_identity": self.replay_identity,
        }
        if self.weighted_residual_mean is not None:
            body["weighted_residual_mean"] = self.weighted_residual_mean
        if self.weighted_residual_rmse is not None:
            body["weighted_residual_rmse"] = self.weighted_residual_rmse
        return body


@dataclass(frozen=True)
class ResidualHistoryValidationIssue:
    code: str
    message: str
    field_path: str | None = None
    row_index: int | None = None
    entity: str | None = None
    origin_index: int | None = None
    horizon: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }
        if self.field_path is not None:
            body["field_path"] = self.field_path
        if self.row_index is not None:
            body["row_index"] = self.row_index
        if self.entity is not None:
            body["entity"] = self.entity
        if self.origin_index is not None:
            body["origin_index"] = self.origin_index
        if self.horizon is not None:
            body["horizon"] = self.horizon
        return body


@dataclass(frozen=True)
class ResidualHistoryValidationResult:
    row_count: int
    issues: tuple[ResidualHistoryValidationIssue, ...]

    @property
    def status(self) -> str:
        return "failed" if self.issues else "passed"

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(sorted({issue.code for issue in self.issues}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "issues": [issue.as_dict() for issue in self.issues],
        }


def residual_history_digest(records: Sequence[ResidualRecordLike]) -> str:
    rows = sorted(
        (_canonical_residual_row(record) for record in records),
        key=canonicalize_json,
    )
    return sha256_digest(rows)


def summarize_residual_history(
    records: Sequence[ResidualRecordLike],
) -> ResidualHistorySummary:
    if not records:
        raise ContractValidationError(
            code="empty_residual_history",
            message="residual histories must contain at least one residual row",
            field_path="residual_rows",
        )
    rows = tuple(_canonical_residual_row(record) for record in records)
    first = rows[0]
    candidate_id = str(first["candidate_id"])
    fit_window_id = str(first["fit_window_id"])
    residual_basis = str(first.get("residual_basis", _RESIDUAL_BASIS))
    for index, row in enumerate(rows):
        if row["candidate_id"] != candidate_id:
            raise ContractValidationError(
                code="inconsistent_residual_history",
                message="all residual rows must have the same candidate_id",
                field_path=f"residual_rows[{index}].candidate_id",
            )
        if row["fit_window_id"] != fit_window_id:
            raise ContractValidationError(
                code="inconsistent_residual_history",
                message="all residual rows must have the same fit_window_id",
                field_path=f"residual_rows[{index}].fit_window_id",
            )
        if row.get("residual_basis", _RESIDUAL_BASIS) != residual_basis:
            raise ContractValidationError(
                code="inconsistent_residual_history",
                message="all residual rows must have the same residual_basis",
                field_path=f"residual_rows[{index}].residual_basis",
            )

    residuals = [float(row["residual"]) for row in rows]
    residual_mean = sum(residuals) / len(residuals)
    residual_rmse = math.sqrt(
        sum(value * value for value in residuals) / len(residuals)
    )
    weights = [float(row["weight"]) for row in rows if row.get("weight") is not None]
    weighted_residual_mean: float | None = None
    weighted_residual_rmse: float | None = None
    if len(weights) == len(rows):
        total_weight = sum(weights)
        if math.isfinite(total_weight) and total_weight > 0.0:
            weighted_residual_mean = (
                sum(
                    weight * residual
                    for weight, residual in zip(weights, residuals, strict=True)
                )
                / total_weight
            )
            weighted_residual_rmse = math.sqrt(
                sum(
                    weight * residual * residual
                    for weight, residual in zip(weights, residuals, strict=True)
                )
                / total_weight
            )
    digest = residual_history_digest(records)
    return ResidualHistorySummary(
        candidate_id=candidate_id,
        fit_window_id=fit_window_id,
        residual_count=len(rows),
        horizon_set=tuple(sorted({int(row["horizon"]) for row in rows})),
        entity_count=len({str(row["entity"]) for row in rows}),
        split_roles=tuple(sorted({str(row["split_role"]) for row in rows})),
        residual_mean=residual_mean,
        residual_rmse=residual_rmse,
        residual_basis=residual_basis,
        residual_history_digest=digest,
        source_row_set_digest=_source_row_set_digest(rows),
        replay_identity=f"{candidate_id}:{fit_window_id}:{digest}",
        weighted_residual_mean=weighted_residual_mean,
        weighted_residual_rmse=weighted_residual_rmse,
    )


def validate_residual_history(
    records: Sequence[ResidualRecordLike],
    *,
    production: bool = True,
) -> ResidualHistoryValidationResult:
    issues: list[ResidualHistoryValidationIssue] = []
    for index, record in enumerate(records):
        row = _record_payload(record)
        _validate_required_residual_fields(row, index=index, issues=issues)
        _validate_residual_geometry(row, index=index, issues=issues)
        if production:
            split_role = str(row.get("split_role", ""))
            if split_role in _PRODUCTION_FORBIDDEN_SPLIT_ROLES:
                issues.append(
                    ResidualHistoryValidationIssue(
                        code="confirmatory_residual_history_not_production_evidence",
                        message=(
                            "confirmatory residual rows cannot be used as "
                            "production stochastic evidence"
                        ),
                        field_path=f"residual_rows[{index}].split_role",
                        row_index=index,
                        entity=_optional_str(row.get("entity")),
                        origin_index=_optional_int(row.get("origin_index")),
                        horizon=_optional_int(row.get("horizon")),
                    )
                )
    return ResidualHistoryValidationResult(row_count=len(records), issues=tuple(issues))


def _validate_required_residual_fields(
    row: Mapping[str, Any],
    *,
    index: int,
    issues: list[ResidualHistoryValidationIssue],
) -> None:
    for field_name in (
        "candidate_id",
        "fit_window_id",
        "entity",
        "origin_index",
        "origin_time",
        "target_index",
        "target_event_time",
        "horizon",
        "point_forecast",
        "residual",
        "residual_basis",
        "time_safety_status",
    ):
        if field_name not in row:
            issues.append(_missing_field_issue(field_name, index=index))
    if "realized_value" not in row and "realized_observation" not in row:
        issues.append(_missing_field_issue("realized_value", index=index))
    missing_availability = [
        field_name
        for field_name in ("origin_available_at", "target_available_at")
        if field_name not in row
    ]
    if missing_availability:
        issues.append(
            ResidualHistoryValidationIssue(
                code="missing_origin_or_target_availability",
                message=(
                    "residual rows require origin_available_at and "
                    "target_available_at; missing "
                    f"{', '.join(missing_availability)}"
                ),
                field_path=f"residual_rows[{index}]",
                row_index=index,
                entity=_optional_str(row.get("entity")),
                origin_index=_optional_int(row.get("origin_index")),
                horizon=_optional_int(row.get("horizon")),
                details={"missing_fields": tuple(missing_availability)},
            )
        )
    if "split_role" not in row:
        issues.append(
            ResidualHistoryValidationIssue(
                code="missing_split_role_metadata",
                message="residual rows require split_role metadata",
                field_path=f"residual_rows[{index}].split_role",
                row_index=index,
                entity=_optional_str(row.get("entity")),
                origin_index=_optional_int(row.get("origin_index")),
                horizon=_optional_int(row.get("horizon")),
            )
        )
    if "replay_identity" not in row:
        issues.append(
            ResidualHistoryValidationIssue(
                code="missing_replay_identity",
                message="residual rows require a replay_identity",
                field_path=f"residual_rows[{index}].replay_identity",
                row_index=index,
                entity=_optional_str(row.get("entity")),
                origin_index=_optional_int(row.get("origin_index")),
                horizon=_optional_int(row.get("horizon")),
            )
        )


def _validate_residual_geometry(
    row: Mapping[str, Any],
    *,
    index: int,
    issues: list[ResidualHistoryValidationIssue],
) -> None:
    if not {"origin_index", "target_index", "horizon"} <= set(row):
        return
    origin_index = _optional_int(row.get("origin_index"))
    target_index = _optional_int(row.get("target_index"))
    horizon = _optional_int(row.get("horizon"))
    if origin_index is None or target_index is None or horizon is None:
        issues.append(
            ResidualHistoryValidationIssue(
                code="invalid_residual_geometry",
                message="origin_index, target_index, and horizon must be integers",
                field_path=f"residual_rows[{index}]",
                row_index=index,
            )
        )
        return
    if origin_index < 0 or target_index < 0 or horizon < 1:
        issues.append(
            ResidualHistoryValidationIssue(
                code="invalid_residual_geometry",
                message=(
                    "origin_index and target_index must be non-negative and "
                    "horizon must be positive"
                ),
                field_path=f"residual_rows[{index}]",
                row_index=index,
                entity=_optional_str(row.get("entity")),
                origin_index=origin_index,
                horizon=horizon,
            )
        )
        return
    if target_index != origin_index + horizon:
        issues.append(
            ResidualHistoryValidationIssue(
                code="invalid_residual_geometry",
                message="target_index must equal origin_index + horizon",
                field_path=f"residual_rows[{index}].target_index",
                row_index=index,
                entity=_optional_str(row.get("entity")),
                origin_index=origin_index,
                horizon=horizon,
                details={
                    "target_index": target_index,
                    "expected_target_index": origin_index + horizon,
                },
            )
        )


def _missing_field_issue(
    field_name: str,
    *,
    index: int,
) -> ResidualHistoryValidationIssue:
    return ResidualHistoryValidationIssue(
        code="missing_residual_required_field",
        message=f"residual rows require {field_name}",
        field_path=f"residual_rows[{index}].{field_name}",
        row_index=index,
    )


def _canonical_residual_row(record: ResidualRecordLike) -> dict[str, Any]:
    row = _record_payload(record)
    realized = row.get("realized_observation", row.get("realized_value"))
    canonical = dict(row)
    if realized is not None:
        canonical["realized_observation"] = float(realized)
        canonical["realized_value"] = float(realized)
    for field_name in ("point_forecast", "residual"):
        if field_name in canonical:
            canonical[field_name] = float(canonical[field_name])
    if "weight" in canonical and canonical["weight"] is not None:
        canonical["weight"] = float(canonical["weight"])
    return canonical


def _record_payload(record: ResidualRecordLike) -> dict[str, Any]:
    if isinstance(record, ForecastResidualRecord):
        return record.as_dict()
    return dict(record)


def _source_row_set_digest(rows: tuple[Mapping[str, Any], ...]) -> str:
    source_rows = [
        {
            "candidate_id": row["candidate_id"],
            "fit_window_id": row["fit_window_id"],
            "entity": row["entity"],
            "origin_index": row["origin_index"],
            "target_index": row["target_index"],
            "horizon": row["horizon"],
            "split_role": row["split_role"],
        }
        for row in rows
    ]
    return sha256_digest(sorted(source_rows, key=canonicalize_json))


def _require_finite_float(value: float, *, field_name: str) -> None:
    if not math.isfinite(float(value)):
        raise ContractValidationError(
            code="invalid_residual_record",
            message=f"{field_name} must be finite",
            field_path=field_name,
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "ForecastResidualRecord",
    "ResidualHistorySummary",
    "ResidualHistoryValidationIssue",
    "ResidualHistoryValidationResult",
    "residual_history_digest",
    "summarize_residual_history",
    "validate_residual_history",
]
