from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

from euclid.contracts.errors import ContractValidationError


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
    residual_location: float
    residual_scale: float
    residual_parameter_summary: Mapping[str, float]
    point_path: Mapping[int, float]
    horizon_scale_law: str
    production_evidence: bool
    heuristic_gaussian_support: bool
    replay_identity: str

    def support_path(self) -> dict[int, StochasticPredictiveSupport]:
        return {
            horizon: StochasticPredictiveSupport(
                location=_stable_float(point_value + self.residual_location),
                scale=_stable_float(
                    self.residual_scale * _horizon_multiplier(
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
            "candidate_id": self.candidate_id,
            "observation_family": self.observation_family,
            "residual_location": self.residual_location,
            "residual_scale": self.residual_scale,
            "residual_parameter_summary": dict(self.residual_parameter_summary),
            "horizon_scale_law": self.horizon_scale_law,
            "production_evidence": self.production_evidence,
            "heuristic_gaussian_support": self.heuristic_gaussian_support,
            "replay_identity": self.replay_identity,
        }


def fit_residual_stochastic_model(
    *,
    candidate_id: str,
    residuals: Sequence[float],
    point_path: Mapping[int, float],
    family_id: str = "gaussian",
    horizon_scale_law: str = "sqrt_horizon",
    min_residual_count: int = 2,
) -> FittedResidualStochasticModel:
    residual_tuple = tuple(float(value) for value in residuals)
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
    family = _normalize_family(family_id)
    if horizon_scale_law not in {"constant", "sqrt_horizon", "linear_horizon"}:
        raise ContractValidationError(
            code="unsupported_horizon_scale_law",
            message="unsupported stochastic horizon scale law",
            field_path="horizon_scale_law",
            details={"horizon_scale_law": horizon_scale_law},
        )

    residual_location = _stable_float(fmean(residual_tuple))
    centered = tuple(value - residual_location for value in residual_tuple)
    variance = fmean(value * value for value in centered)
    residual_scale = _stable_float(max(math.sqrt(max(variance, 0.0)), 1e-9))
    parameter_summary = _parameter_summary(
        family=family,
        location=residual_location,
        scale=residual_scale,
    )
    identity_payload = {
        "candidate_id": candidate_id,
        "family": family,
        "horizon_scale_law": horizon_scale_law,
        "point_path": normalized_point_path,
        "residual_parameter_summary": parameter_summary,
    }
    return FittedResidualStochasticModel(
        candidate_id=str(candidate_id),
        observation_family=family,
        residual_location=residual_location,
        residual_scale=residual_scale,
        residual_parameter_summary=parameter_summary,
        point_path=normalized_point_path,
        horizon_scale_law=horizon_scale_law,
        production_evidence=True,
        heuristic_gaussian_support=False,
        replay_identity=f"stochastic-model:{_digest(identity_payload)}",
    )


def _parameter_summary(*, family: str, location: float, scale: float) -> dict[str, float]:
    if family == "student_t":
        return {"location": location, "scale": scale, "df": 5.0}
    if family == "laplace":
        return {"location": location, "scale": scale / math.sqrt(2.0)}
    return {"location": location, "scale": scale}


def _normalize_family(family_id: str) -> str:
    family = str(family_id)
    if family in {"gaussian", "student_t", "laplace"}:
        return family
    raise ContractValidationError(
        code="unsupported_stochastic_process_family",
        message="residual process fitting currently supports continuous residual families",
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
