from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, ClassVar, Mapping, TypeAlias

from euclid.contracts.errors import ContractValidationError
from euclid.runtime.hashing import canonicalize_json, sha256_digest

ScalarValue: TypeAlias = str | bool | int | float | None

ALL_COMPOSITION_OPERATORS: tuple[str, ...] = (
    "piecewise",
    "additive_residual",
    "regime_conditioned",
    "shared_plus_local_decomposition",
)
PIECEWISE_BRANCH_ORDER = "ascending_split_literal"
ADDITIVE_RESIDUAL_CHILD_ORDER = "base_then_residual"
REGIME_SELECTION_MODES: tuple[str, ...] = ("hard_switch", "convex_weighting")


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


def _normalize_numeric_literal(
    value: int | float,
    *,
    code: str,
    field_path: str,
) -> float:
    normalized = _normalize_scalar(
        value,
        code=code,
        field_path=field_path,
    )
    if isinstance(normalized, bool) or not isinstance(normalized, (int, float)):
        raise ContractValidationError(
            code=code,
            message=f"{field_path} must be numeric",
            field_path=field_path,
        )
    return float(normalized)


def _decimal_string(value: int | float) -> str:
    normalized = Decimal(str(value)).normalize()
    rendered = format(normalized, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", ""} else rendered


def _canonical_literal(value: ScalarValue) -> dict[str, Any]:
    normalized = _normalize_scalar(
        value,
        code="unstable_composition_literal",
        field_path="literal",
    )
    if normalized is None:
        return {"type": "null", "value": None}
    if isinstance(normalized, bool):
        return {"type": "bool", "value": normalized}
    if isinstance(normalized, int):
        return {"type": "int", "value": _decimal_string(normalized)}
    if isinstance(normalized, float):
        return {"type": "float", "value": _decimal_string(normalized)}
    return {"type": "string", "value": normalized}


def _literal_sort_key(value: ScalarValue) -> str:
    return canonicalize_json(_canonical_literal(value))


def _require_unique_strings(
    values: tuple[str, ...],
    *,
    code: str,
    field_path: str,
) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen:
            duplicates.append(value)
            continue
        seen.add(value)
    if duplicates:
        raise ContractValidationError(
            code=code,
            message=f"{field_path} must not contain duplicates",
            field_path=field_path,
            details={"duplicates": tuple(duplicates)},
        )


def _require_mapping(
    value: object,
    *,
    code: str,
    field_path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(
            code=code,
            message=f"{field_path} must be an object",
            field_path=field_path,
        )
    return value


@dataclass(frozen=True)
class PiecewisePartitionSegment:
    start_literal: float
    end_literal: float
    reducer_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_literal",
            _normalize_numeric_literal(
                self.start_literal,
                code="invalid_piecewise_partition",
                field_path="start_literal",
            ),
        )
        object.__setattr__(
            self,
            "end_literal",
            _normalize_numeric_literal(
                self.end_literal,
                code="invalid_piecewise_partition",
                field_path="end_literal",
            ),
        )
        _require_identifier(
            self.reducer_id,
            code="invalid_piecewise_partition",
            field_path="reducer_id",
        )
        if self.start_literal >= self.end_literal:
            raise ContractValidationError(
                code="invalid_piecewise_partition",
                message="piecewise segments must have strictly increasing bounds",
                field_path="ordered_partition",
                details={
                    "start_literal": self.start_literal,
                    "end_literal": self.end_literal,
                },
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_literal": self.start_literal,
            "end_literal": self.end_literal,
            "reducer_id": self.reducer_id,
        }

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "start_literal": _decimal_string(self.start_literal),
            "end_literal": _decimal_string(self.end_literal),
            "reducer_id": self.reducer_id,
        }


@dataclass(frozen=True)
class PiecewiseComposition:
    ordered_partition: tuple[PiecewisePartitionSegment, ...]
    branch_order_law: str = PIECEWISE_BRANCH_ORDER

    def __post_init__(self) -> None:
        if not self.ordered_partition:
            raise ContractValidationError(
                code="invalid_piecewise_partition",
                message="piecewise compositions must declare at least one segment",
                field_path="ordered_partition",
            )
        if self.branch_order_law != PIECEWISE_BRANCH_ORDER:
            raise ContractValidationError(
                code="invalid_piecewise_partition",
                message=(
                    f"branch_order_law must equal {PIECEWISE_BRANCH_ORDER!r} "
                    "for retained piecewise composition"
                ),
                field_path="branch_order_law",
            )

    @property
    def child_reducer_ids(self) -> tuple[str, ...]:
        return tuple(segment.reducer_id for segment in self.ordered_partition)

    def normalize(self) -> "PiecewiseComposition":
        ordered = tuple(
            sorted(
                self.ordered_partition,
                key=lambda segment: (
                    segment.start_literal,
                    segment.end_literal,
                    segment.reducer_id,
                ),
            )
        )
        merged: list[PiecewisePartitionSegment] = []
        for segment in ordered:
            if merged and segment.start_literal < merged[-1].end_literal:
                raise ContractValidationError(
                    code="invalid_piecewise_partition",
                    message="piecewise segments must not overlap after sorting",
                    field_path="ordered_partition",
                    details={
                        "left": merged[-1].as_dict(),
                        "right": segment.as_dict(),
                    },
                )
            if (
                merged
                and segment.start_literal == merged[-1].end_literal
                and segment.reducer_id == merged[-1].reducer_id
            ):
                merged[-1] = PiecewisePartitionSegment(
                    start_literal=merged[-1].start_literal,
                    end_literal=segment.end_literal,
                    reducer_id=segment.reducer_id,
                )
                continue
            merged.append(segment)
        return PiecewiseComposition(
            ordered_partition=tuple(merged),
            branch_order_law=self.branch_order_law,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch_order_law": self.branch_order_law,
            "ordered_partition": [
                segment.as_dict() for segment in self.ordered_partition
            ],
        }

    def canonical_dict(self) -> dict[str, Any]:
        normalized = self.normalize()
        return {
            "branch_order_law": normalized.branch_order_law,
            "ordered_partition": [
                segment.canonical_dict() for segment in normalized.ordered_partition
            ],
        }


@dataclass(frozen=True)
class AdditiveResidualComposition:
    base_reducer: str
    residual_reducer: str
    shared_observation_model: str

    def __post_init__(self) -> None:
        _require_identifier(
            self.base_reducer,
            code="invalid_additive_residual_composition",
            field_path="base_reducer",
        )
        _require_identifier(
            self.residual_reducer,
            code="invalid_additive_residual_composition",
            field_path="residual_reducer",
        )
        _require_identifier(
            self.shared_observation_model,
            code="invalid_additive_residual_composition",
            field_path="shared_observation_model",
        )
        if self.base_reducer == self.residual_reducer:
            raise ContractValidationError(
                code="invalid_additive_residual_composition",
                message=(
                    "additive_residual compositions require distinct "
                    "base_reducer and residual_reducer ids"
                ),
                field_path="residual_reducer",
                details={
                    "base_reducer": self.base_reducer,
                    "residual_reducer": self.residual_reducer,
                },
            )

    @property
    def child_reducer_ids(self) -> tuple[str, ...]:
        return (self.base_reducer, self.residual_reducer)

    def normalize(self) -> "AdditiveResidualComposition":
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_reducer": self.base_reducer,
            "residual_reducer": self.residual_reducer,
            "shared_observation_model": self.shared_observation_model,
        }

    def canonical_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class RegimeGatingLaw:
    gating_law_id: str
    selection_mode: str = "hard_switch"

    def __post_init__(self) -> None:
        _require_identifier(
            self.gating_law_id,
            code="invalid_regime_conditioned_composition",
            field_path="gating_law.gating_law_id",
        )
        if self.selection_mode not in REGIME_SELECTION_MODES:
            raise ContractValidationError(
                code="invalid_regime_conditioned_composition",
                message=(
                    "gating_law.selection_mode must be one of "
                    f"{REGIME_SELECTION_MODES!r}"
                ),
                field_path="gating_law.selection_mode",
            )

    def as_dict(self) -> dict[str, str]:
        return {
            "gating_law_id": self.gating_law_id,
            "selection_mode": self.selection_mode,
        }


@dataclass(frozen=True)
class RegimeConditionedBranch:
    regime_value: ScalarValue
    reducer_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "regime_value",
            _normalize_scalar(
                self.regime_value,
                code="invalid_regime_conditioned_composition",
                field_path="branch_reducers.regime_value",
            ),
        )
        _require_identifier(
            self.reducer_id,
            code="invalid_regime_conditioned_composition",
            field_path="branch_reducers.reducer_id",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime_value": self.regime_value,
            "reducer_id": self.reducer_id,
        }

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "regime_value": _canonical_literal(self.regime_value),
            "reducer_id": self.reducer_id,
        }


@dataclass(frozen=True)
class RegimeConditionedComposition:
    gating_law: RegimeGatingLaw
    branch_reducers: tuple[RegimeConditionedBranch, ...]
    regime_information_contract: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.branch_reducers:
            raise ContractValidationError(
                code="invalid_regime_conditioned_composition",
                message=(
                    "regime_conditioned compositions must declare at least one "
                    "branch reducer"
                ),
                field_path="branch_reducers",
            )
        if not self.regime_information_contract:
            raise ContractValidationError(
                code="invalid_regime_conditioned_composition",
                message=(
                    "regime_conditioned compositions must declare a "
                    "regime_information_contract"
                ),
                field_path="regime_information_contract",
            )
        normalized_contract = tuple(
            _require_identifier(
                entry,
                code="invalid_regime_conditioned_composition",
                field_path=f"regime_information_contract[{index}]",
            )
            for index, entry in enumerate(self.regime_information_contract)
        )
        object.__setattr__(self, "regime_information_contract", normalized_contract)
        _require_unique_strings(
            normalized_contract,
            code="invalid_regime_conditioned_composition",
            field_path="regime_information_contract",
        )
        branch_keys = tuple(
            _literal_sort_key(branch.regime_value) for branch in self.branch_reducers
        )
        _require_unique_strings(
            branch_keys,
            code="invalid_regime_conditioned_composition",
            field_path="branch_reducers",
        )

    @property
    def child_reducer_ids(self) -> tuple[str, ...]:
        return tuple(branch.reducer_id for branch in self.branch_reducers)

    def normalize(self) -> "RegimeConditionedComposition":
        ordered_branches = tuple(
            sorted(
                self.branch_reducers,
                key=lambda branch: canonicalize_json(branch.canonical_dict()),
            )
        )
        ordered_contract = tuple(sorted(self.regime_information_contract))
        return RegimeConditionedComposition(
            gating_law=self.gating_law,
            branch_reducers=ordered_branches,
            regime_information_contract=ordered_contract,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "gating_law": self.gating_law.as_dict(),
            "branch_reducers": [branch.as_dict() for branch in self.branch_reducers],
            "regime_information_contract": list(self.regime_information_contract),
        }

    def canonical_dict(self) -> dict[str, Any]:
        normalized = self.normalize()
        return {
            "gating_law": normalized.gating_law.as_dict(),
            "branch_reducers": [
                branch.canonical_dict() for branch in normalized.branch_reducers
            ],
            "regime_information_contract": list(normalized.regime_information_contract),
        }


@dataclass(frozen=True)
class SharedPlusLocalComposition:
    entity_index_set: tuple[str, ...]
    shared_component_ref: str
    local_component_refs: tuple[str, ...]
    sharing_map: tuple[str, ...]
    unseen_entity_rule: str

    def __post_init__(self) -> None:
        if not self.entity_index_set:
            raise ContractValidationError(
                code="invalid_shared_plus_local_composition",
                message="shared_plus_local_decomposition requires entity_index_set",
                field_path="entity_index_set",
            )
        entity_index_set = tuple(
            _require_identifier(
                entity,
                code="invalid_shared_plus_local_composition",
                field_path=f"entity_index_set[{index}]",
            )
            for index, entity in enumerate(self.entity_index_set)
        )
        _require_unique_strings(
            entity_index_set,
            code="invalid_shared_plus_local_composition",
            field_path="entity_index_set",
        )
        object.__setattr__(self, "entity_index_set", entity_index_set)
        _require_identifier(
            self.shared_component_ref,
            code="invalid_shared_plus_local_composition",
            field_path="shared_component_ref",
        )
        local_component_refs = tuple(
            _require_identifier(
                reducer_id,
                code="invalid_shared_plus_local_composition",
                field_path=f"local_component_refs[{index}]",
            )
            for index, reducer_id in enumerate(self.local_component_refs)
        )
        if len(local_component_refs) != len(entity_index_set):
            raise ContractValidationError(
                code="invalid_shared_plus_local_composition",
                message=(
                    "local_component_refs must align one-for-one with entity_index_set"
                ),
                field_path="local_component_refs",
                details={
                    "entity_count": len(entity_index_set),
                    "local_component_count": len(local_component_refs),
                },
            )
        object.__setattr__(self, "local_component_refs", local_component_refs)
        sharing_map = tuple(
            _require_identifier(
                value,
                code="invalid_shared_plus_local_composition",
                field_path=f"sharing_map[{index}]",
            )
            for index, value in enumerate(self.sharing_map)
        )
        if not sharing_map:
            raise ContractValidationError(
                code="invalid_shared_plus_local_composition",
                message="sharing_map must declare at least one shared quantity",
                field_path="sharing_map",
            )
        _require_unique_strings(
            sharing_map,
            code="invalid_shared_plus_local_composition",
            field_path="sharing_map",
        )
        object.__setattr__(self, "sharing_map", sharing_map)
        _require_identifier(
            self.unseen_entity_rule,
            code="invalid_shared_plus_local_composition",
            field_path="unseen_entity_rule",
        )

    @property
    def child_reducer_ids(self) -> tuple[str, ...]:
        return (self.shared_component_ref, *self.local_component_refs)

    def normalize(self) -> "SharedPlusLocalComposition":
        bindings = tuple(
            sorted(
                zip(self.entity_index_set, self.local_component_refs, strict=True),
                key=lambda item: item[0],
            )
        )
        return SharedPlusLocalComposition(
            entity_index_set=tuple(entity for entity, _ in bindings),
            shared_component_ref=self.shared_component_ref,
            local_component_refs=tuple(component for _, component in bindings),
            sharing_map=tuple(sorted(self.sharing_map)),
            unseen_entity_rule=self.unseen_entity_rule,
        )

    def as_dict(self) -> dict[str, Any]:
        normalized = self.normalize()
        return {
            "entity_index_set": list(normalized.entity_index_set),
            "shared_component_ref": normalized.shared_component_ref,
            "local_component_refs": list(normalized.local_component_refs),
            "sharing_map": list(normalized.sharing_map),
            "unseen_entity_rule": normalized.unseen_entity_rule,
        }

    def canonical_dict(self) -> dict[str, Any]:
        return self.as_dict()


CompositionSpec: TypeAlias = (
    PiecewiseComposition
    | AdditiveResidualComposition
    | RegimeConditionedComposition
    | SharedPlusLocalComposition
)


@dataclass(frozen=True)
class ReducerCompositionObject:
    operator_id: str | None = None
    composition: CompositionSpec | None = None

    _allowed_operators: ClassVar[tuple[str, ...]] = ALL_COMPOSITION_OPERATORS

    def __post_init__(self) -> None:
        if self.operator_id is None:
            if self.composition is not None:
                raise ContractValidationError(
                    code="invalid_composition_payload",
                    message="composition payload requires an operator_id",
                    field_path="composition",
                )
            return
        _require_identifier(
            self.operator_id,
            code="invalid_composition_operator",
            field_path="operator_id",
        )
        if self.operator_id not in self._allowed_operators:
            raise ContractValidationError(
                code="invalid_composition_operator",
                message=f"{self.operator_id!r} is not a legal composition operator",
                field_path="operator_id",
                details={"allowed_operators": self._allowed_operators},
            )
        expected_type = {
            "piecewise": PiecewiseComposition,
            "additive_residual": AdditiveResidualComposition,
            "regime_conditioned": RegimeConditionedComposition,
            "shared_plus_local_decomposition": SharedPlusLocalComposition,
        }.get(self.operator_id)
        if expected_type is None:
            raise ContractValidationError(
                code="unsupported_composition_operator",
                message=(
                    f"{self.operator_id!r} is known but not supported by the "
                    "retained reducer runtime yet"
                ),
                field_path="operator_id",
            )
        if self.composition is None:
            raise ContractValidationError(
                code="invalid_composition_payload",
                message="composition operators require an explicit typed payload",
                field_path="composition",
            )
        if not isinstance(self.composition, expected_type):
            raise ContractValidationError(
                code="invalid_composition_payload",
                message=(
                    f"{self.operator_id!r} requires a "
                    f"{expected_type.__name__} payload"
                ),
                field_path="composition",
            )

    @property
    def child_reducer_ids(self) -> tuple[str, ...]:
        normalized = self.normalize()
        if normalized.composition is None:
            return ()
        return normalized.composition.child_reducer_ids

    def normalize(self) -> "ReducerCompositionObject":
        if self.composition is None:
            return self
        return ReducerCompositionObject(
            operator_id=self.operator_id,
            composition=self.composition.normalize(),
        )

    def as_dict(self) -> dict[str, Any]:
        normalized = self.normalize()
        if normalized.composition is None:
            return {"operator_id": None, "child_reducer_ids": []}
        payload = {
            "operator_id": normalized.operator_id,
            "child_reducer_ids": list(normalized.composition.child_reducer_ids),
        }
        payload.update(normalized.composition.as_dict())
        return payload

    def canonical_payload(self) -> dict[str, Any]:
        normalized = self.normalize()
        if normalized.composition is None:
            return {"operator_id": None, "child_reducer_ids": []}
        payload = {
            "operator_id": normalized.operator_id,
            "child_reducer_ids": list(normalized.composition.child_reducer_ids),
        }
        payload.update(normalized.composition.canonical_dict())
        return payload

    def canonical_bytes(self) -> bytes:
        return canonicalize_json(self.canonical_payload()).encode("utf-8")

    def canonical_hash(self) -> str:
        return sha256_digest(self.canonical_payload())

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "ReducerCompositionObject":
        if payload is None:
            return cls()
        mapping = _require_mapping(
            payload,
            code="invalid_composition_payload",
            field_path="composition",
        )
        operator_id = mapping.get("operator_id")
        if operator_id is None:
            return cls()
        if operator_id == "piecewise":
            ordered_partition_payload = mapping.get("ordered_partition")
            if not isinstance(ordered_partition_payload, list):
                raise ContractValidationError(
                    code="invalid_piecewise_partition",
                    message="piecewise compositions require ordered_partition",
                    field_path="ordered_partition",
                )
            ordered_partition = tuple(
                PiecewisePartitionSegment(
                    start_literal=_require_mapping(
                        segment_payload,
                        code="invalid_piecewise_partition",
                        field_path=f"ordered_partition[{index}]",
                    )["start_literal"],
                    end_literal=_require_mapping(
                        segment_payload,
                        code="invalid_piecewise_partition",
                        field_path=f"ordered_partition[{index}]",
                    )["end_literal"],
                    reducer_id=_require_mapping(
                        segment_payload,
                        code="invalid_piecewise_partition",
                        field_path=f"ordered_partition[{index}]",
                    )["reducer_id"],
                )
                for index, segment_payload in enumerate(ordered_partition_payload)
            )
            branch_order_law = mapping.get("branch_order_law", PIECEWISE_BRANCH_ORDER)
            return cls(
                operator_id=operator_id,
                composition=PiecewiseComposition(
                    ordered_partition=ordered_partition,
                    branch_order_law=branch_order_law,
                ),
            )
        if operator_id == "additive_residual":
            return cls(
                operator_id=operator_id,
                composition=AdditiveResidualComposition(
                    base_reducer=mapping.get("base_reducer"),
                    residual_reducer=mapping.get("residual_reducer"),
                    shared_observation_model=mapping.get("shared_observation_model"),
                ),
            )
        if operator_id == "regime_conditioned":
            gating_law_payload = _require_mapping(
                mapping.get("gating_law"),
                code="invalid_regime_conditioned_composition",
                field_path="gating_law",
            )
            branch_payloads = mapping.get("branch_reducers")
            if not isinstance(branch_payloads, list):
                raise ContractValidationError(
                    code="invalid_regime_conditioned_composition",
                    message="branch_reducers must be a list of declared branches",
                    field_path="branch_reducers",
                )
            regime_contract_payload = mapping.get("regime_information_contract")
            if not isinstance(regime_contract_payload, list):
                raise ContractValidationError(
                    code="invalid_regime_conditioned_composition",
                    message=(
                        "regime_information_contract must be a list of explicit "
                        "regime inputs"
                    ),
                    field_path="regime_information_contract",
                )
            return cls(
                operator_id=operator_id,
                composition=RegimeConditionedComposition(
                    gating_law=RegimeGatingLaw(
                        gating_law_id=gating_law_payload.get("gating_law_id"),
                        selection_mode=gating_law_payload.get(
                            "selection_mode",
                            "hard_switch",
                        ),
                    ),
                    branch_reducers=tuple(
                        RegimeConditionedBranch(
                            regime_value=_require_mapping(
                                branch_payload,
                                code="invalid_regime_conditioned_composition",
                                field_path=f"branch_reducers[{index}]",
                            )["regime_value"],
                            reducer_id=_require_mapping(
                                branch_payload,
                                code="invalid_regime_conditioned_composition",
                                field_path=f"branch_reducers[{index}]",
                            )["reducer_id"],
                        )
                        for index, branch_payload in enumerate(branch_payloads)
                    ),
                    regime_information_contract=tuple(regime_contract_payload),
                ),
            )
        if operator_id == "shared_plus_local_decomposition":
            entity_index_set = mapping.get("entity_index_set")
            local_component_refs = mapping.get("local_component_refs")
            sharing_map = mapping.get("sharing_map")
            if not isinstance(entity_index_set, list):
                raise ContractValidationError(
                    code="invalid_shared_plus_local_composition",
                    message="entity_index_set must be a list of declared entities",
                    field_path="entity_index_set",
                )
            if not isinstance(local_component_refs, list):
                raise ContractValidationError(
                    code="invalid_shared_plus_local_composition",
                    message=(
                        "local_component_refs must be a list aligned with "
                        "entity_index_set"
                    ),
                    field_path="local_component_refs",
                )
            if not isinstance(sharing_map, list):
                raise ContractValidationError(
                    code="invalid_shared_plus_local_composition",
                    message="sharing_map must be a list of shared quantities",
                    field_path="sharing_map",
                )
            return cls(
                operator_id=operator_id,
                composition=SharedPlusLocalComposition(
                    entity_index_set=tuple(entity_index_set),
                    shared_component_ref=mapping.get("shared_component_ref"),
                    local_component_refs=tuple(local_component_refs),
                    sharing_map=tuple(sharing_map),
                    unseen_entity_rule=mapping.get("unseen_entity_rule"),
                ),
            )
        return cls(operator_id=operator_id, composition=None)


def parse_reducer_composition(
    payload: Mapping[str, Any] | None,
) -> ReducerCompositionObject:
    return ReducerCompositionObject.from_dict(payload).normalize()


def component_field_name(name: str, component_id: str) -> str:
    return f"{name}__{component_id}"


def extract_component_mapping(
    mapping: Mapping[str, Any],
    component_id: str,
) -> dict[str, Any]:
    suffix = f"__{component_id}"
    resolved = {
        name: value for name, value in mapping.items() if "__" not in name
    }
    for name, value in mapping.items():
        if name.endswith(suffix):
            resolved[name[: -len(suffix)]] = value
    return resolved


def merge_component_mapping(
    destination: dict[str, Any],
    *,
    component_id: str,
    mapping: Mapping[str, Any],
) -> None:
    for name, value in sorted(mapping.items()):
        destination[component_field_name(name, component_id)] = value


def composition_runtime_signature(
    composition_object: ReducerCompositionObject,
) -> str | None:
    normalized = composition_object.normalize()
    if normalized.operator_id is None:
        return None
    return normalized.canonical_hash()


def select_piecewise_segment(
    composition_object: ReducerCompositionObject,
    *,
    row: Mapping[str, Any],
) -> tuple[PiecewisePartitionSegment, dict[str, Any]]:
    normalized = composition_object.normalize()
    if normalized.operator_id != "piecewise" or not isinstance(
        normalized.composition,
        PiecewiseComposition,
    ):
        raise ContractValidationError(
            code="invalid_piecewise_partition",
            message="piecewise runtime selection requires a piecewise composition",
            field_path="composition_graph.operator_id",
        )
    if "piecewise_partition_value" not in row:
        raise ContractValidationError(
            code="missing_piecewise_partition_value",
            message=(
                "piecewise runtime semantics require an explicit "
                "piecewise_partition_value on every evaluated row"
            ),
            field_path="row.piecewise_partition_value",
        )
    partition_value = _normalize_numeric_literal(
        row["piecewise_partition_value"],
        code="invalid_piecewise_partition_value",
        field_path="row.piecewise_partition_value",
    )
    ordered_partition = normalized.composition.ordered_partition
    last_index = len(ordered_partition) - 1
    for index, segment in enumerate(ordered_partition):
        within_open_interval = (
            segment.start_literal <= partition_value < segment.end_literal
        )
        within_terminal_boundary = (
            index == last_index
            and math.isclose(
                partition_value,
                segment.end_literal,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        if within_open_interval or within_terminal_boundary:
            return segment, {
                "operator_id": "piecewise",
                "signal_field": "piecewise_partition_value",
                "partition_value": partition_value,
                "selected_branch_id": segment.reducer_id,
                "selected_interval": segment.as_dict(),
            }
    raise ContractValidationError(
        code="piecewise_partition_out_of_support",
        message=(
            "piecewise_partition_value must fall inside one declared "
            "piecewise segment"
        ),
        field_path="row.piecewise_partition_value",
        details={
            "piecewise_partition_value": partition_value,
            "ordered_partition": [
                segment.as_dict() for segment in ordered_partition
            ],
        },
    )


def resolve_regime_weights(
    composition_object: ReducerCompositionObject,
    *,
    row: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    normalized = composition_object.normalize()
    if normalized.operator_id != "regime_conditioned" or not isinstance(
        normalized.composition,
        RegimeConditionedComposition,
    ):
        raise ContractValidationError(
            code="invalid_regime_conditioned_composition",
            message=(
                "regime runtime selection requires a regime_conditioned "
                "composition"
            ),
            field_path="composition_graph.operator_id",
        )
    composition = normalized.composition
    if composition.gating_law.selection_mode == "hard_switch":
        if len(composition.regime_information_contract) != 1:
            raise ContractValidationError(
                code="invalid_regime_conditioned_composition",
                message=(
                    "hard_switch regime-conditioned compositions require exactly "
                    "one regime_information_contract field"
                ),
                field_path="regime_information_contract",
                details={
                    "regime_information_contract": list(
                        composition.regime_information_contract
                    )
                },
            )
        field_name = composition.regime_information_contract[0]
        if field_name not in row:
            raise ContractValidationError(
                code="missing_regime_information",
                message=(
                    "regime-conditioned runtime semantics require the declared "
                    "regime information to be present on every evaluated row"
                ),
                field_path=f"row.{field_name}",
            )
        observed_value = _normalize_scalar(
            row[field_name],
            code="invalid_regime_information",
            field_path=f"row.{field_name}",
        )
        observed_key = _literal_sort_key(observed_value)
        for branch in composition.branch_reducers:
            if _literal_sort_key(branch.regime_value) == observed_key:
                return (
                    {branch.reducer_id: 1.0},
                    {
                        "operator_id": "regime_conditioned",
                        "selection_mode": "hard_switch",
                        "gating_law_id": composition.gating_law.gating_law_id,
                        "signal_fields": [field_name],
                        "observed_regime_value": observed_value,
                        "selected_branch_id": branch.reducer_id,
                    },
                )
        raise ContractValidationError(
            code="unknown_regime_assignment",
            message=(
                "regime-conditioned runtime semantics require the observed "
                "regime value to match one declared branch"
            ),
            field_path=f"row.{field_name}",
            details={
                "observed_regime_value": observed_value,
                "declared_regime_values": [
                    branch.regime_value for branch in composition.branch_reducers
                ],
            },
        )

    contract_fields = composition.regime_information_contract
    if len(contract_fields) != len(composition.branch_reducers):
        raise ContractValidationError(
            code="invalid_regime_conditioned_composition",
            message=(
                "convex_weighting regime-conditioned compositions require one "
                "regime_information_contract field per branch reducer"
            ),
            field_path="regime_information_contract",
            details={
                "contract_field_count": len(contract_fields),
                "branch_count": len(composition.branch_reducers),
            },
        )
    raw_weights: list[float] = []
    for field_name in contract_fields:
        if field_name not in row:
            raise ContractValidationError(
                code="missing_regime_information",
                message=(
                    "regime-conditioned runtime semantics require the declared "
                    "regime information to be present on every evaluated row"
                ),
                field_path=f"row.{field_name}",
            )
        weight = _normalize_numeric_literal(
            row[field_name],
            code="invalid_regime_information",
            field_path=f"row.{field_name}",
        )
        if weight < 0.0:
            raise ContractValidationError(
                code="invalid_regime_information",
                message="convex weighting fields must be >= 0",
                field_path=f"row.{field_name}",
            )
        raw_weights.append(weight)
    total_weight = sum(raw_weights)
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ContractValidationError(
            code="invalid_regime_information",
            message="convex weighting fields must sum to a positive finite value",
            field_path="regime_information_contract",
        )
    normalized_weights = tuple(weight / total_weight for weight in raw_weights)
    branch_weights = {
        branch.reducer_id: normalized_weights[index]
        for index, branch in enumerate(composition.branch_reducers)
        if normalized_weights[index] > 0.0
    }
    return (
        branch_weights,
        {
            "operator_id": "regime_conditioned",
            "selection_mode": "convex_weighting",
            "gating_law_id": composition.gating_law.gating_law_id,
            "signal_fields": list(contract_fields),
            "branch_weights": [
                {
                    "reducer_id": branch.reducer_id,
                    "regime_value": branch.regime_value,
                    "weight": normalized_weights[index],
                }
                for index, branch in enumerate(composition.branch_reducers)
            ],
        },
    )


__all__ = [
    "ALL_COMPOSITION_OPERATORS",
    "ADDITIVE_RESIDUAL_CHILD_ORDER",
    "PIECEWISE_BRANCH_ORDER",
    "AdditiveResidualComposition",
    "PiecewiseComposition",
    "PiecewisePartitionSegment",
    "ReducerCompositionObject",
    "RegimeConditionedBranch",
    "RegimeConditionedComposition",
    "RegimeGatingLaw",
    "SharedPlusLocalComposition",
    "component_field_name",
    "composition_runtime_signature",
    "extract_component_mapping",
    "merge_component_mapping",
    "parse_reducer_composition",
    "resolve_regime_weights",
    "select_piecewise_segment",
]
