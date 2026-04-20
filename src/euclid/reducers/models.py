from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Mapping, TypeAlias

from euclid.contracts.errors import ContractValidationError
from euclid.math.observation_models import PointObservationModel
from euclid.reducers.composition import (
    ALL_COMPOSITION_OPERATORS,
    AdditiveResidualComposition,
    PiecewiseComposition,
    PiecewisePartitionSegment,
    ReducerCompositionObject,
    RegimeConditionedBranch,
    RegimeConditionedComposition,
    RegimeGatingLaw,
    parse_reducer_composition,
)

ScalarValue: TypeAlias = str | bool | int | float | None
StateUpdateFn: TypeAlias = Callable[
    ["ReducerStateObject", "ReducerStateUpdateContext"], "ReducerStateObject"
]

ALL_REDUCER_FAMILIES: tuple[str, ...] = (
    "analytic",
    "recursive",
    "spectral",
    "algorithmic",
)


def _require_identifier(value: str, *, code: str, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(
            code=code,
            message=f"{field_path} must be a non-empty string",
            field_path=field_path,
        )
    return value


def _normalize_scalar(
    value: ScalarValue,
    *,
    code: str,
    field_path: str,
) -> ScalarValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(
                code=code,
                message=f"{field_path} must be finite",
                field_path=field_path,
            )
        return value
    raise ContractValidationError(
        code=code,
        message=f"{field_path} must be a scalar literal",
        field_path=field_path,
    )


@dataclass(frozen=True)
class ReducerFamilyId:
    family_id: str
    _allowed_families: ClassVar[tuple[str, ...]] = ALL_REDUCER_FAMILIES

    def __post_init__(self) -> None:
        _require_identifier(
            self.family_id,
            code="invalid_reducer_family",
            field_path="family_id",
        )
        if self.family_id not in self._allowed_families:
            raise ContractValidationError(
                code="invalid_reducer_family",
                message=f"{self.family_id!r} is not a legal reducer family",
                field_path="family_id",
                details={"allowed_families": self._allowed_families},
            )

    def as_dict(self) -> dict[str, str]:
        return {"family_id": self.family_id}


@dataclass(frozen=True)
class ReducerParameter:
    name: str
    value: ScalarValue

    def __post_init__(self) -> None:
        _require_identifier(
            self.name,
            code="invalid_reducer_parameter",
            field_path="name",
        )
        object.__setattr__(
            self,
            "value",
            _normalize_scalar(
                self.value,
                code="invalid_reducer_parameter",
                field_path=f"parameter[{self.name}]",
            ),
        )

    def as_dict(self) -> dict[str, ScalarValue]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True)
class ReducerParameterObject:
    parameters: tuple[ReducerParameter, ...] = ()

    def __post_init__(self) -> None:
        _require_unique_names(
            [parameter.name for parameter in self.parameters],
            code="duplicate_reducer_parameter",
            field_path="parameters",
        )

    def as_dict(self) -> dict[str, Any]:
        return {"parameters": [parameter.as_dict() for parameter in self.parameters]}


@dataclass(frozen=True)
class ReducerStateSlot:
    name: str
    value: ScalarValue

    def __post_init__(self) -> None:
        _require_identifier(
            self.name,
            code="invalid_reducer_state_slot",
            field_path="name",
        )
        object.__setattr__(
            self,
            "value",
            _normalize_scalar(
                self.value,
                code="invalid_reducer_state_slot",
                field_path=f"state_slot[{self.name}]",
            ),
        )

    def as_dict(self) -> dict[str, ScalarValue]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True)
class ReducerStateObject:
    slots: tuple[ReducerStateSlot, ...] = ()

    def __post_init__(self) -> None:
        _require_unique_names(
            [slot.name for slot in self.slots],
            code="duplicate_reducer_state_slot",
            field_path="slots",
        )

    def get(self, name: str) -> ScalarValue:
        for slot in self.slots:
            if slot.name == name:
                return slot.value
        raise KeyError(name)

    def as_dict(self) -> dict[str, Any]:
        return {"slots": [slot.as_dict() for slot in self.slots]}


@dataclass(frozen=True)
class BoundObservationModel:
    family: str
    forecast_type: str = "point"
    support_kind: str = "all_real"
    compatible_point_losses: tuple[str, ...] = ()

    @classmethod
    def from_runtime(cls, runtime: PointObservationModel) -> "BoundObservationModel":
        return cls(
            family=runtime.family,
            compatible_point_losses=runtime.compatible_point_losses,
        )

    def supports_point_loss(self, loss_name: str) -> bool:
        return loss_name in self.compatible_point_losses

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "forecast_type": self.forecast_type,
            "support_kind": self.support_kind,
            "compatible_point_losses": list(self.compatible_point_losses),
        }


@dataclass(frozen=True)
class ReducerAdmissibilityObject:
    family_membership: bool
    composition_closure: bool
    observation_model_compatibility: bool
    valid_state_semantics: bool
    codelength_comparability: bool
    diagnostic_codes: tuple[str, ...] = ()

    @property
    def is_admissible(self) -> bool:
        return all(
            (
                self.family_membership,
                self.composition_closure,
                self.observation_model_compatibility,
                self.valid_state_semantics,
                self.codelength_comparability,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_membership": self.family_membership,
            "composition_closure": self.composition_closure,
            "observation_model_compatibility": self.observation_model_compatibility,
            "valid_state_semantics": self.valid_state_semantics,
            "codelength_comparability": self.codelength_comparability,
            "diagnostic_codes": list(self.diagnostic_codes),
        }


@dataclass(frozen=True)
class ReducerStateUpdateContext:
    observation_index: int
    history: tuple[float, ...]
    side_information: Mapping[str, ScalarValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.observation_index < 0:
            raise ContractValidationError(
                code="invalid_reducer_state_update_context",
                message="observation_index must be non-negative",
                field_path="observation_index",
            )
        normalized_history = tuple(
            float(
                _normalize_scalar(
                    value,
                    code="invalid_reducer_state_update_context",
                    field_path=f"history[{index}]",
                )
            )
            for index, value in enumerate(self.history)
        )
        normalized_side_information = {
            key: _normalize_scalar(
                value,
                code="invalid_reducer_state_update_context",
                field_path=f"side_information.{key}",
            )
            for key, value in sorted(self.side_information.items())
        }
        object.__setattr__(self, "history", normalized_history)
        object.__setattr__(self, "side_information", normalized_side_information)

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_index": self.observation_index,
            "history": list(self.history),
            "side_information": dict(self.side_information),
        }


@dataclass(frozen=True)
class ReducerStateUpdateRule:
    update_rule_id: str
    implementation: StateUpdateFn = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_identifier(
            self.update_rule_id,
            code="invalid_reducer_state_update_rule",
            field_path="update_rule_id",
        )
        if not callable(self.implementation):
            raise ContractValidationError(
                code="invalid_reducer_state_update_rule",
                message="implementation must be callable",
                field_path="implementation",
            )

    def apply(
        self,
        state: ReducerStateObject,
        context: ReducerStateUpdateContext,
    ) -> ReducerStateObject:
        next_state = self.implementation(state, context)
        if not isinstance(next_state, ReducerStateObject):
            raise ContractValidationError(
                code="invalid_reducer_state_update_result",
                message="state update rules must return ReducerStateObject instances",
                field_path="implementation",
                details={"update_rule_id": self.update_rule_id},
            )
        return next_state

    def as_dict(self) -> dict[str, str]:
        return {"update_rule_id": self.update_rule_id}


@dataclass(frozen=True)
class ReducerStateSemantics:
    persistent_state: ReducerStateObject
    update_rule: ReducerStateUpdateRule

    def initialize_state(self) -> ReducerStateObject:
        return ReducerStateObject(slots=self.persistent_state.slots)

    def update_state(
        self,
        state: ReducerStateObject,
        context: ReducerStateUpdateContext,
    ) -> ReducerStateObject:
        return self.update_rule.apply(state, context)

    def as_dict(self) -> dict[str, Any]:
        return {
            "persistent_state": self.persistent_state.as_dict(),
            "update_rule": self.update_rule.as_dict(),
        }


@dataclass(frozen=True)
class ReducerObject:
    family: ReducerFamilyId
    composition_object: ReducerCompositionObject
    fitted_parameters: ReducerParameterObject
    state_semantics: ReducerStateSemantics
    observation_model: BoundObservationModel
    admissibility: ReducerAdmissibilityObject

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.as_dict(),
            "composition_object": self.composition_object.as_dict(),
            "fitted_parameters": self.fitted_parameters.as_dict(),
            "state_semantics": self.state_semantics.as_dict(),
            "observation_model": self.observation_model.as_dict(),
            "admissibility": self.admissibility.as_dict(),
        }


def _require_unique_names(
    names: list[str],
    *,
    code: str,
    field_path: str,
) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen:
            duplicates.append(name)
            continue
        seen.add(name)
    if duplicates:
        raise ContractValidationError(
            code=code,
            message="names must be unique",
            field_path=field_path,
            details={"duplicate_names": tuple(duplicates)},
        )


__all__ = [
    "ALL_COMPOSITION_OPERATORS",
    "ALL_REDUCER_FAMILIES",
    "AdditiveResidualComposition",
    "BoundObservationModel",
    "PiecewiseComposition",
    "PiecewisePartitionSegment",
    "ReducerAdmissibilityObject",
    "ReducerCompositionObject",
    "ReducerFamilyId",
    "ReducerObject",
    "ReducerParameter",
    "ReducerParameterObject",
    "ReducerStateObject",
    "ReducerStateSemantics",
    "ReducerStateSlot",
    "ReducerStateUpdateContext",
    "ReducerStateUpdateRule",
    "RegimeConditionedBranch",
    "RegimeConditionedComposition",
    "RegimeGatingLaw",
    "ScalarValue",
    "StateUpdateFn",
    "parse_reducer_composition",
]
