from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from euclid.contracts.errors import ContractValidationError


def _require_name(value: str, *, field_path: str = "name") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(
            code="invalid_fit_parameter",
            message=f"{field_path} must be a non-empty string",
            field_path=field_path,
        )
    return value.strip()


def _finite(value: float, *, field_path: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ContractValidationError(
            code="invalid_fit_parameter",
            message=f"{field_path} must be finite",
            field_path=field_path,
        )
    return normalized


@dataclass(frozen=True)
class ParameterBounds:
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        lower = None if self.lower is None else _finite(self.lower, field_path="bounds.lower")
        upper = None if self.upper is None else _finite(self.upper, field_path="bounds.upper")
        if lower is not None and upper is not None and lower > upper:
            raise ContractValidationError(
                code="invalid_fit_parameter_bounds",
                message="parameter lower bound must be <= upper bound",
                field_path="bounds.lower",
                details={"lower": lower, "upper": upper},
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def as_dict(self) -> dict[str, float | None]:
        return {"lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class ParameterPrior:
    kind: str
    location: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _require_name(self.kind, field_path="prior.kind"))
        object.__setattr__(self, "location", _finite(self.location, field_path="prior.location"))
        scale = _finite(self.scale, field_path="prior.scale")
        if scale <= 0.0:
            raise ContractValidationError(
                code="invalid_fit_parameter_prior",
                message="prior scale must be positive",
                field_path="prior.scale",
            )
        object.__setattr__(self, "scale", scale)

    def as_dict(self) -> dict[str, float | str]:
        return {"kind": self.kind, "location": self.location, "scale": self.scale}


@dataclass(frozen=True)
class ParameterPenalty:
    kind: str
    weight: float
    center: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _require_name(self.kind, field_path="penalty.kind"))
        weight = _finite(self.weight, field_path="penalty.weight")
        if weight < 0.0:
            raise ContractValidationError(
                code="invalid_fit_parameter_penalty",
                message="penalty weight must be non-negative",
                field_path="penalty.weight",
            )
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "center", _finite(self.center, field_path="penalty.center"))

    def as_dict(self) -> dict[str, float | str]:
        return {"kind": self.kind, "weight": self.weight, "center": self.center}


@dataclass(frozen=True)
class ParameterDeclaration:
    name: str
    initial_value: float
    bounds: ParameterBounds | None = None
    transform: str = "identity"
    fixed: bool = False
    shared: bool = True
    entity_local: bool = False
    regime_local: bool = False
    prior: ParameterPrior | None = None
    penalty: ParameterPenalty | None = None
    unit: str | None = None
    description_length_bits: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_name(self.name))
        initial_value = _finite(self.initial_value, field_path=f"parameter[{self.name}].initial_value")
        if self.transform not in {"identity", "positive_log"}:
            raise ContractValidationError(
                code="invalid_fit_parameter_transform",
                message=f"{self.transform!r} is not a supported parameter transform",
                field_path=f"parameter[{self.name}].transform",
            )
        if self.transform == "positive_log" and initial_value <= 0.0:
            raise ContractValidationError(
                code="invalid_fit_parameter_transform",
                message="positive_log parameters require positive initial values",
                field_path=f"parameter[{self.name}].initial_value",
            )
        object.__setattr__(self, "initial_value", initial_value)
        object.__setattr__(
            self,
            "description_length_bits",
            _finite(
                self.description_length_bits,
                field_path=f"parameter[{self.name}].description_length_bits",
            ),
        )

    def external_initial_value(self) -> float:
        if self.transform == "positive_log":
            return math.log(self.initial_value)
        return self.initial_value

    def external_bounds(self) -> tuple[float, float]:
        bounds = self.bounds or ParameterBounds()
        if self.transform == "positive_log":
            lower = -math.inf if bounds.lower is None or bounds.lower <= 0.0 else math.log(bounds.lower)
            upper = math.inf if bounds.upper is None else math.log(bounds.upper)
            return lower, upper
        return (
            -math.inf if bounds.lower is None else bounds.lower,
            math.inf if bounds.upper is None else bounds.upper,
        )

    def to_internal_value(self, external_value: float) -> float:
        if self.transform == "positive_log":
            return float(math.exp(external_value))
        return float(external_value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "initial_value": self.initial_value,
            "bounds": (self.bounds or ParameterBounds()).as_dict(),
            "transform": self.transform,
            "fixed": self.fixed,
            "shared": self.shared,
            "entity_local": self.entity_local,
            "regime_local": self.regime_local,
            "prior": None if self.prior is None else self.prior.as_dict(),
            "penalty": None if self.penalty is None else self.penalty.as_dict(),
            "unit": self.unit,
            "description_length_bits": self.description_length_bits,
        }


@dataclass(frozen=True)
class ParameterVector:
    declarations: tuple[ParameterDeclaration, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for declaration in self.declarations:
            if declaration.name in seen:
                duplicates.append(declaration.name)
            seen.add(declaration.name)
        if duplicates:
            raise ContractValidationError(
                code="duplicate_fit_parameter",
                message="fit parameter declarations must not contain duplicate names",
                field_path="declarations",
                details={"duplicate_names": sorted(set(duplicates))},
            )
        object.__setattr__(self, "declarations", tuple(self.declarations))

    @property
    def free_declarations(self) -> tuple[ParameterDeclaration, ...]:
        return tuple(declaration for declaration in self.declarations if not declaration.fixed)

    @property
    def free_names(self) -> tuple[str, ...]:
        return tuple(declaration.name for declaration in self.free_declarations)

    def initial_free_vector(self) -> tuple[float, ...]:
        return tuple(declaration.external_initial_value() for declaration in self.free_declarations)

    def free_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        lower: list[float] = []
        upper: list[float] = []
        for declaration in self.free_declarations:
            lo, hi = declaration.external_bounds()
            lower.append(lo)
            upper.append(hi)
        return tuple(lower), tuple(upper)

    def bind_free_values(self, free_values: tuple[float, ...] | list[float]) -> dict[str, float]:
        if len(free_values) != len(self.free_declarations):
            raise ContractValidationError(
                code="invalid_fit_parameter_vector",
                message="free parameter vector length does not match declarations",
                field_path="free_values",
                details={
                    "expected": len(self.free_declarations),
                    "actual": len(free_values),
                },
            )
        bound = {
            declaration.name: declaration.initial_value
            for declaration in self.declarations
            if declaration.fixed
        }
        for declaration, value in zip(self.free_declarations, free_values, strict=True):
            bound[declaration.name] = declaration.to_internal_value(float(value))
        return dict(sorted(bound.items()))

    def as_dict(self) -> list[dict[str, Any]]:
        return [declaration.as_dict() for declaration in self.declarations]


__all__ = [
    "ParameterBounds",
    "ParameterDeclaration",
    "ParameterPenalty",
    "ParameterPrior",
    "ParameterVector",
]
