from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.modules.residual_history import (
    ForecastResidualRecord,
    summarize_residual_history,
    validate_residual_history,
)


@dataclass(frozen=True)
class StochasticPredictiveSupport:
    location: float
    scale: float
    distribution_family: str
    support_kind: str = "all_real"

    def as_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "scale": self.scale,
            "distribution_family": self.distribution_family,
            "support_kind": self.support_kind,
        }


@dataclass(frozen=True)
class FittedResidualStochasticModel:
    candidate_id: str
    observation_family: str
    residual_family: str
    residual_location: float
    residual_scale: float
    residual_parameter_summary: Mapping[str, float]
    point_path: Mapping[int, float]
    horizon_scale_law: str
    residual_count: int
    min_count_policy: Mapping[str, int]
    residual_source_kind: str
    horizon_coverage: tuple[int, ...]
    evidence_status: str
    production_evidence: bool
    heuristic_gaussian_support: bool
    replay_identity: str
    residual_history_ref: TypedRef | None = None
    residual_history_digest: str | None = None
    support_kind: str = "all_real"

    def support_path(self) -> dict[int, StochasticPredictiveSupport]:
        return {
            horizon: StochasticPredictiveSupport(
                location=_stable_float(point_value + self.residual_location),
                scale=_stable_float(
                    self.residual_scale
                    * _horizon_multiplier(
                        horizon=horizon,
                        law=self.horizon_scale_law,
                    )
                ),
                distribution_family=_distribution_family(self.observation_family),
                support_kind=_support_kind(self.observation_family),
            )
            for horizon, point_value in sorted(self.point_path.items())
        }

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "stochastic_model_manifest@1.0.0",
            "stochastic_model_id": self.replay_identity,
            "candidate_id": self.candidate_id,
            "residual_history_ref": (
                self.residual_history_ref.as_dict()
                if self.residual_history_ref is not None
                else None
            ),
            "observation_family": self.observation_family,
            "residual_family": self.residual_family,
            "support_kind": self.support_kind,
            "residual_location": self.residual_location,
            "residual_scale": self.residual_scale,
            "residual_parameter_summary": dict(self.residual_parameter_summary),
            "fitted_parameters": dict(self.residual_parameter_summary),
            "residual_count": self.residual_count,
            "min_count_policy": dict(self.min_count_policy),
            "residual_source_kind": self.residual_source_kind,
            "horizon_coverage": list(self.horizon_coverage),
            "residual_history_digest": self.residual_history_digest,
            "horizon_scale_law": self.horizon_scale_law,
            "evidence_status": self.evidence_status,
            "production_evidence": self.production_evidence,
            "heuristic_gaussian_support": self.heuristic_gaussian_support,
            "replay_identity": self.replay_identity,
        }


def fit_residual_stochastic_model(
    *,
    candidate_id: str,
    residuals: Sequence[float] | None = None,
    residual_history: (
        Sequence[ForecastResidualRecord | Mapping[str, Any]] | None
    ) = None,
    point_path: Mapping[int, float],
    family_id: str = "gaussian",
    support_kind: str = "all_real",
    horizon_scale_law: str = "sqrt_horizon",
    min_residual_count: int = 2,
    required_horizon_set: Sequence[int] | None = None,
    residual_history_ref: TypedRef | None = None,
    evidence_status: str | None = None,
    heuristic_gaussian_support: bool | None = None,
    student_t_degrees_of_freedom: float | None = None,
) -> FittedResidualStochasticModel:
    family = _normalize_family(family_id)
    resolved_support_kind = _normalize_support_kind(
        support_kind=support_kind,
        family=family,
    )
    (
        residual_tuple,
        residual_source_kind,
        residual_history_digest,
        horizon_coverage,
    ) = _resolve_residual_source(
        residuals=residuals,
        residual_history=residual_history,
    )
    resolved_evidence_status = (
        evidence_status
        if evidence_status is not None
        else (
            "production"
            if residual_source_kind == "validated_residual_history"
            and residual_history_ref is not None
            else "compatibility"
        )
    )
    if resolved_evidence_status not in {"production", "compatibility"}:
        raise ContractValidationError(
            code="invalid_stochastic_evidence_status",
            message="stochastic evidence status must be production or compatibility",
            field_path="evidence_status",
            details={"evidence_status": resolved_evidence_status},
        )
    if (
        resolved_evidence_status == "production"
        and residual_source_kind != "validated_residual_history"
    ):
        raise ContractValidationError(
            code="synthetic_residual_source_not_production",
            message=(
                "production stochastic evidence must consume a validated "
                "residual history"
            ),
            field_path="residual_history",
        )
    if resolved_evidence_status == "production" and residual_history_ref is None:
        raise ContractValidationError(
            code="missing_residual_history_evidence",
            message="production stochastic evidence requires residual history evidence",
            field_path="residual_history_ref",
        )
    if len(residual_tuple) < int(min_residual_count):
        raise ContractValidationError(
            code="insufficient_stochastic_training_support",
            message="stochastic process fitting needs residual support",
            field_path="residuals",
            details={"residual_count": len(residual_tuple)},
        )
    if any(not math.isfinite(value) for value in residual_tuple):
        raise ContractValidationError(
            code="nonfinite_stochastic_training_value",
            message="stochastic process fitting requires finite residuals",
            field_path="residuals",
        )
    _require_horizon_coverage(
        horizon_coverage=horizon_coverage,
        required_horizon_set=required_horizon_set,
    )
    if not point_path:
        raise ContractValidationError(
            code="missing_point_path_for_stochastic_support",
            message="stochastic support generation requires a point path",
            field_path="point_path",
        )
    normalized_point_path = {
        int(horizon): float(value) for horizon, value in point_path.items()
    }
    if any(
        horizon <= 0 or not math.isfinite(value)
        for horizon, value in normalized_point_path.items()
    ):
        raise ContractValidationError(
            code="invalid_point_path_for_stochastic_support",
            message="point path horizons and values must be finite",
            field_path="point_path",
        )
    resolved_heuristic_gaussian_support = (
        bool(heuristic_gaussian_support)
        if heuristic_gaussian_support is not None
        else (
            resolved_evidence_status == "compatibility"
            and residual_source_kind == "synthetic_residuals"
            and family == "gaussian"
        )
    )
    if resolved_evidence_status == "production" and resolved_heuristic_gaussian_support:
        raise ContractValidationError(
            code="heuristic_gaussian_support_not_production",
            message="heuristic Gaussian support is compatibility evidence only",
            field_path="heuristic_gaussian_support",
        )
    if horizon_scale_law not in {"constant", "sqrt_horizon", "linear_horizon"}:
        raise ContractValidationError(
            code="unsupported_horizon_scale_law",
            message="unsupported stochastic horizon scale law",
            field_path="horizon_scale_law",
            details={"horizon_scale_law": horizon_scale_law},
        )

    parameter_summary = _parameter_summary(
        family=family,
        residuals=residual_tuple,
        student_t_degrees_of_freedom=student_t_degrees_of_freedom,
    )
    residual_location = _stable_float(float(parameter_summary["location"]))
    residual_scale = _stable_float(float(parameter_summary["scale"]))
    identity_payload = {
        "candidate_id": candidate_id,
        "family": family,
        "support_kind": resolved_support_kind,
        "residual_history_ref": (
            residual_history_ref.as_dict() if residual_history_ref is not None else None
        ),
        "residual_history_digest": residual_history_digest,
        "residual_source_kind": residual_source_kind,
        "horizon_coverage": horizon_coverage,
        "evidence_status": resolved_evidence_status,
        "horizon_scale_law": horizon_scale_law,
        "point_path": normalized_point_path,
        "residual_parameter_summary": parameter_summary,
        "residual_count": len(residual_tuple),
    }
    return FittedResidualStochasticModel(
        candidate_id=str(candidate_id),
        observation_family=family,
        residual_family=family,
        residual_location=residual_location,
        residual_scale=residual_scale,
        residual_parameter_summary=parameter_summary,
        point_path=normalized_point_path,
        horizon_scale_law=horizon_scale_law,
        residual_count=len(residual_tuple),
        min_count_policy={"minimum_residual_count": int(min_residual_count)},
        residual_source_kind=residual_source_kind,
        horizon_coverage=horizon_coverage,
        evidence_status=resolved_evidence_status,
        production_evidence=resolved_evidence_status == "production",
        heuristic_gaussian_support=resolved_heuristic_gaussian_support,
        replay_identity=f"stochastic-model:{_digest(identity_payload)}",
        residual_history_ref=residual_history_ref,
        residual_history_digest=residual_history_digest,
        support_kind=resolved_support_kind,
    )


def _resolve_residual_source(
    *,
    residuals: Sequence[float] | None,
    residual_history: Sequence[ForecastResidualRecord | Mapping[str, Any]] | None,
) -> tuple[tuple[float, ...], str, str | None, tuple[int, ...]]:
    if residual_history is not None:
        validation = validate_residual_history(residual_history)
        if validation.status != "passed":
            raise ContractValidationError(
                code="invalid_residual_history_evidence",
                message="residual history evidence failed production validation",
                field_path="residual_history",
                details=validation.as_dict(),
            )
        summary = summarize_residual_history(residual_history)
        rows = tuple(
            row.as_dict() if isinstance(row, ForecastResidualRecord) else dict(row)
            for row in residual_history
        )
        return (
            tuple(float(row["residual"]) for row in rows),
            "validated_residual_history",
            summary.residual_history_digest,
            summary.horizon_set,
        )
    if residuals is None:
        raise ContractValidationError(
            code="missing_stochastic_training_support",
            message="stochastic process fitting requires residual support",
            field_path="residuals",
        )
    return (
        tuple(float(value) for value in residuals),
        "synthetic_residuals",
        None,
        (),
    )


def _require_horizon_coverage(
    *,
    horizon_coverage: tuple[int, ...],
    required_horizon_set: Sequence[int] | None,
) -> None:
    if required_horizon_set is None:
        return
    required = tuple(sorted({int(horizon) for horizon in required_horizon_set}))
    missing = tuple(horizon for horizon in required if horizon not in horizon_coverage)
    if missing:
        raise ContractValidationError(
            code="insufficient_residual_horizon_coverage",
            message="residual history does not cover the required horizon set",
            field_path="residual_history.horizon_set",
            details={
                "required_horizon_set": list(required),
                "observed_horizon_set": list(horizon_coverage),
                "missing_horizons": list(missing),
            },
        )


def _parameter_summary(
    *,
    family: str,
    residuals: tuple[float, ...],
    student_t_degrees_of_freedom: float | None,
) -> dict[str, float]:
    if family == "laplace":
        location = _stable_float(_median(residuals))
        scale = _stable_float(
            max(fmean(abs(value - location) for value in residuals), 1e-9)
        )
        return {"location": location, "scale": scale}

    location = _stable_float(fmean(residuals))
    centered = tuple(value - location for value in residuals)
    variance = fmean(value * value for value in centered)
    scale = _stable_float(max(math.sqrt(max(variance, 0.0)), 1e-9))
    if family == "student_t":
        df = (
            5.0
            if student_t_degrees_of_freedom is None
            else float(student_t_degrees_of_freedom)
        )
        if not math.isfinite(df) or df <= 2.0:
            raise ContractValidationError(
                code="invalid_student_t_degrees_of_freedom",
                message="Student-t residual fitting requires finite df > 2",
                field_path="student_t_degrees_of_freedom",
            )
        return {"location": location, "scale": scale, "df": _stable_float(df)}
    return {"location": location, "scale": scale}


def _normalize_family(family_id: str) -> str:
    family = str(family_id)
    if family in {"gaussian", "student_t", "laplace"}:
        return family
    raise ContractValidationError(
        code="unsupported_stochastic_process_family",
        message=(
            "residual process fitting currently supports continuous residual "
            "families"
        ),
        field_path="family_id",
        details={"family_id": family},
    )


def _distribution_family(family: str) -> str:
    return {
        "gaussian": "gaussian_location_scale",
        "student_t": "student_t_location_scale",
        "laplace": "laplace_location_scale",
    }[family]


def _support_kind(family: str) -> str:
    return "all_real"


def _normalize_support_kind(*, support_kind: str, family: str) -> str:
    del family
    normalized = str(support_kind)
    if normalized == "all_real":
        return normalized
    raise ContractValidationError(
        code="unsupported_residual_support_kind",
        message="residual process fitting currently supports all-real residuals only",
        field_path="support_kind",
        details={"support_kind": normalized},
    )


def _median(values: Sequence[float]) -> float:
    sorted_values = sorted(float(value) for value in values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def _horizon_multiplier(*, horizon: int, law: str) -> float:
    if law == "constant":
        return 1.0
    if law == "sqrt_horizon":
        return math.sqrt(float(horizon))
    if law == "linear_horizon":
        return float(horizon)
    raise AssertionError("unreachable")


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "FittedResidualStochasticModel",
    "StochasticPredictiveSupport",
    "fit_residual_stochastic_model",
]
