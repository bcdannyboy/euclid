from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from euclid.contracts.errors import ContractValidationError


@dataclass(frozen=True)
class EventDefinition:
    event_id: str
    variable: str
    operator: str
    threshold: float
    threshold_source: str
    units: str | None = None
    scope: str | None = None
    calibration_required: bool = True

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "EventDefinition":
        threshold_source = str(payload.get("threshold_source", ""))
        if threshold_source in {"", "origin_target", "implicit_origin_target"}:
            raise ContractValidationError(
                code="undeclared_event_threshold_source",
                message="event thresholds must come from an explicit manifest source",
                field_path="threshold_source",
                details={"threshold_source": threshold_source},
            )
        try:
            threshold = float(payload["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractValidationError(
                code="invalid_event_threshold",
                message="event threshold must be finite",
                field_path="threshold",
            ) from exc
        if not math.isfinite(threshold):
            raise ContractValidationError(
                code="invalid_event_threshold",
                message="event threshold must be finite",
                field_path="threshold",
            )
        operator = str(payload.get("operator", ""))
        if operator not in {
            "greater_than",
            "greater_than_or_equal",
            "less_than",
            "less_than_or_equal",
        }:
            raise ContractValidationError(
                code="unsupported_event_operator",
                message="unsupported event operator",
                field_path="operator",
                details={"operator": operator},
            )
        return cls(
            event_id=str(payload["event_id"]),
            variable=str(payload.get("variable", "target")),
            operator=operator,
            threshold=threshold,
            threshold_source=threshold_source,
            units=(str(payload["units"]) if payload.get("units") is not None else None),
            scope=(str(payload["scope"]) if payload.get("scope") is not None else None),
            calibration_required=bool(payload.get("calibration_required", True)),
        )

    def evaluate(self, value: float) -> bool:
        observed = float(value)
        if self.operator == "greater_than":
            return observed > self.threshold
        if self.operator == "greater_than_or_equal":
            return observed >= self.threshold
        if self.operator == "less_than":
            return observed < self.threshold
        if self.operator == "less_than_or_equal":
            return observed <= self.threshold
        raise AssertionError("unreachable")

    def probability(self, model: Any) -> float:
        if self.operator == "greater_than":
            return _clamp_probability(model.survival(self.threshold))
        if self.operator == "greater_than_or_equal":
            return _clamp_probability(model.survival(self.threshold))
        if self.operator == "less_than":
            return _clamp_probability(model.cdf(self.threshold))
        if self.operator == "less_than_or_equal":
            return _clamp_probability(model.cdf(self.threshold))
        raise AssertionError("unreachable")

    def as_manifest(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_name": "event_definition@1.0.0",
            "event_id": self.event_id,
            "variable": self.variable,
            "operator": self.operator,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "calibration_required": self.calibration_required,
        }
        if self.units is not None:
            body["units"] = self.units
        if self.scope is not None:
            body["scope"] = self.scope
        return body


def _clamp_probability(value: float) -> float:
    probability = float(value)
    if not math.isfinite(probability):
        raise ContractValidationError(
            code="nonfinite_event_probability",
            message="event probability must be finite",
            field_path="event_probability",
        )
    return max(0.0, min(1.0, probability))


__all__ = ["EventDefinition"]
