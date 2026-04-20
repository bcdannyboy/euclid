from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.reducers.models import (
    BoundObservationModel,
    ReducerCompositionObject,
    ReducerFamilyId,
    ReducerParameterObject,
    ReducerStateObject,
    ScalarValue,
)
from euclid.runtime.hashing import normalize_json_value


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


def _require_unique_names(
    names: tuple[str, ...],
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
            message=f"{field_path} must not contain duplicates",
            field_path=field_path,
            details={"duplicate_names": tuple(duplicates)},
        )


@dataclass(frozen=True)
class CIRInputSignature:
    target_series: str
    side_information_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(
            self.target_series,
            code="invalid_cir_input_signature",
            field_path="target_series",
        )
        normalized = tuple(
            _require_identifier(
                field_name,
                code="invalid_cir_input_signature",
                field_path=f"side_information_fields[{index}]",
            )
            for index, field_name in enumerate(self.side_information_fields)
        )
        _require_unique_names(
            normalized,
            code="invalid_cir_input_signature",
            field_path="side_information_fields",
        )
        object.__setattr__(self, "side_information_fields", normalized)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_series": self.target_series,
            "side_information_fields": list(self.side_information_fields),
        }


@dataclass(frozen=True)
class CIRHistoryAccessContract:
    contract_id: str
    access_mode: str
    max_lag: int | None = None
    allowed_side_information: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(
            self.contract_id,
            code="invalid_history_access_contract",
            field_path="contract_id",
        )
        _require_identifier(
            self.access_mode,
            code="invalid_history_access_contract",
            field_path="access_mode",
        )
        if self.max_lag is not None and self.max_lag < 0:
            raise ContractValidationError(
                code="invalid_history_access_contract",
                message="max_lag must be non-negative when provided",
                field_path="max_lag",
            )
        normalized_side_information = tuple(
            _require_identifier(
                field_name,
                code="invalid_history_access_contract",
                field_path=f"allowed_side_information[{index}]",
            )
            for index, field_name in enumerate(self.allowed_side_information)
        )
        _require_unique_names(
            normalized_side_information,
            code="invalid_history_access_contract",
            field_path="allowed_side_information",
        )
        object.__setattr__(
            self,
            "allowed_side_information",
            normalized_side_information,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "access_mode": self.access_mode,
            "max_lag": self.max_lag,
            "allowed_side_information": list(self.allowed_side_information),
        }


@dataclass(frozen=True)
class CIRStateSignature:
    persistent_state: ReducerStateObject = field(default_factory=ReducerStateObject)

    def as_dict(self) -> dict[str, Any]:
        return {
            "persistent_state": self.persistent_state.as_dict(),
            "state_topology": [slot.name for slot in self.persistent_state.slots],
        }


@dataclass(frozen=True)
class CIRLiteral:
    name: str
    value: ScalarValue

    def __post_init__(self) -> None:
        _require_identifier(
            self.name,
            code="invalid_cir_literal",
            field_path="name",
        )
        object.__setattr__(
            self,
            "value",
            _normalize_scalar(
                self.value,
                code="invalid_cir_literal",
                field_path=f"literal[{self.name}]",
            ),
        )

    def as_dict(self) -> dict[str, ScalarValue]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True)
class CIRLiteralBlock:
    literals: tuple[CIRLiteral, ...] = ()

    def __post_init__(self) -> None:
        _require_unique_names(
            tuple(literal.name for literal in self.literals),
            code="invalid_cir_literal_block",
            field_path="literals",
        )

    def as_dict(self) -> dict[str, Any]:
        return {"literals": [literal.as_dict() for literal in self.literals]}


@dataclass(frozen=True)
class CIRForecastOperator:
    operator_id: str
    horizon: int
    forecast_object_type: str = "point"

    def __post_init__(self) -> None:
        _require_identifier(
            self.operator_id,
            code="invalid_cir_forecast_operator",
            field_path="operator_id",
        )
        _require_identifier(
            self.forecast_object_type,
            code="invalid_cir_forecast_operator",
            field_path="forecast_object_type",
        )
        if self.horizon <= 0:
            raise ContractValidationError(
                code="invalid_cir_forecast_operator",
                message="horizon must be positive",
                field_path="horizon",
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "horizon": self.horizon,
            "forecast_object_type": self.forecast_object_type,
        }


@dataclass(frozen=True)
class CIRModelCodeDecomposition:
    L_family_bits: float
    L_structure_bits: float
    L_literals_bits: float
    L_params_bits: float
    L_state_bits: float

    def __post_init__(self) -> None:
        for field_name in (
            "L_family_bits",
            "L_structure_bits",
            "L_literals_bits",
            "L_params_bits",
            "L_state_bits",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ContractValidationError(
                    code="invalid_cir_model_code_decomposition",
                    message=f"{field_name} must be a finite non-negative float",
                    field_path=field_name,
                )
            object.__setattr__(self, field_name, value)

    def as_dict(self) -> dict[str, float]:
        return {
            "L_family_bits": self.L_family_bits,
            "L_structure_bits": self.L_structure_bits,
            "L_literals_bits": self.L_literals_bits,
            "L_params_bits": self.L_params_bits,
            "L_state_bits": self.L_state_bits,
        }


@dataclass(frozen=True)
class CIRBackendOriginRecord:
    adapter_id: str
    adapter_class: str
    source_candidate_id: str
    search_class: str
    backend_family: str | None = None
    proposal_rank: int | None = None
    normalization_scope: str = "cir_structural_execution_model_code_only"
    comparability_scope: str = "candidate_fitting_and_scoring"
    backend_private_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "adapter_id",
            "adapter_class",
            "source_candidate_id",
            "search_class",
            "normalization_scope",
            "comparability_scope",
        ):
            _require_identifier(
                getattr(self, field_name),
                code="invalid_cir_backend_origin",
                field_path=field_name,
            )
        if self.backend_family is not None:
            _require_identifier(
                self.backend_family,
                code="invalid_cir_backend_origin",
                field_path="backend_family",
            )
        if self.proposal_rank is not None and self.proposal_rank < 0:
            raise ContractValidationError(
                code="invalid_cir_backend_origin",
                message="proposal_rank must be non-negative",
                field_path="proposal_rank",
            )
        normalized_private_fields = tuple(
            _require_identifier(
                field_name,
                code="invalid_cir_backend_origin",
                field_path=f"backend_private_fields[{index}]",
            )
            for index, field_name in enumerate(self.backend_private_fields)
        )
        _require_unique_names(
            normalized_private_fields,
            code="invalid_cir_backend_origin",
            field_path="backend_private_fields",
        )
        object.__setattr__(self, "backend_private_fields", normalized_private_fields)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "adapter_id": self.adapter_id,
            "adapter_class": self.adapter_class,
            "source_candidate_id": self.source_candidate_id,
            "search_class": self.search_class,
            "normalization_scope": self.normalization_scope,
            "comparability_scope": self.comparability_scope,
            "backend_private_fields": list(self.backend_private_fields),
        }
        if self.backend_family is not None:
            payload["backend_family"] = self.backend_family
        if self.proposal_rank is not None:
            payload["proposal_rank"] = self.proposal_rank
        return payload


@dataclass(frozen=True)
class CIRReplayHook:
    hook_name: str
    hook_ref: str

    def __post_init__(self) -> None:
        _require_identifier(
            self.hook_name,
            code="invalid_cir_replay_hook",
            field_path="hook_name",
        )
        _require_identifier(
            self.hook_ref,
            code="invalid_cir_replay_hook",
            field_path="hook_ref",
        )

    def as_dict(self) -> dict[str, str]:
        return {"hook_name": self.hook_name, "hook_ref": self.hook_ref}


@dataclass(frozen=True)
class CIRReplayHooks:
    hooks: tuple[CIRReplayHook, ...] = ()

    def __post_init__(self) -> None:
        _require_unique_names(
            tuple(hook.hook_name for hook in self.hooks),
            code="invalid_cir_replay_hooks",
            field_path="hooks",
        )

    def as_dict(self) -> dict[str, Any]:
        return {"hooks": [hook.as_dict() for hook in self.hooks]}


@dataclass(frozen=True)
class CIRCanonicalSerialization:
    canonical_bytes: bytes
    content_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_bytes, bytes):
            raise ContractValidationError(
                code="invalid_cir_canonical_serialization",
                message="canonical_bytes must be bytes",
                field_path="canonical_bytes",
            )
        if not isinstance(self.content_hash, str) or not self.content_hash.startswith(
            "sha256:"
        ):
            raise ContractValidationError(
                code="invalid_cir_canonical_serialization",
                message="content_hash must be a sha256-prefixed digest",
                field_path="content_hash",
            )

    def as_dict(self) -> dict[str, str]:
        return {
            "canonical_utf8": self.canonical_bytes.decode("utf-8"),
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class CIRStructuralLayer:
    cir_family_id: str
    cir_form_class: str
    input_signature: CIRInputSignature
    state_signature: CIRStateSignature
    literal_block: CIRLiteralBlock = field(default_factory=CIRLiteralBlock)
    parameter_block: ReducerParameterObject = field(
        default_factory=ReducerParameterObject
    )
    composition_graph: ReducerCompositionObject = field(
        default_factory=ReducerCompositionObject
    )

    def __post_init__(self) -> None:
        ReducerFamilyId(self.cir_family_id)
        _require_identifier(
            self.cir_form_class,
            code="invalid_cir_structural_layer",
            field_path="cir_form_class",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cir_family_id": self.cir_family_id,
            "cir_form_class": self.cir_form_class,
            "input_signature": self.input_signature.as_dict(),
            "state_signature": self.state_signature.as_dict(),
            "literal_block": self.literal_block.as_dict(),
            "parameter_block": self.parameter_block.as_dict(),
            "composition_graph": self.composition_graph.canonical_payload(),
        }


@dataclass(frozen=True)
class CIRExecutionLayer:
    history_access_contract: CIRHistoryAccessContract
    state_update_law_id: str
    forecast_operator: CIRForecastOperator
    observation_model_binding: BoundObservationModel

    def __post_init__(self) -> None:
        _require_identifier(
            self.state_update_law_id,
            code="invalid_cir_execution_layer",
            field_path="state_update_law_id",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "history_access_contract": self.history_access_contract.as_dict(),
            "state_update_law_id": self.state_update_law_id,
            "forecast_operator": self.forecast_operator.as_dict(),
            "observation_model_binding": self.observation_model_binding.as_dict(),
        }


@dataclass(frozen=True)
class CIREvidenceLayer:
    canonical_serialization: CIRCanonicalSerialization
    model_code_decomposition: CIRModelCodeDecomposition
    backend_origin_record: CIRBackendOriginRecord
    replay_hooks: CIRReplayHooks
    transient_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.transient_diagnostics, Mapping):
            raise ContractValidationError(
                code="invalid_cir_evidence_layer",
                message="transient_diagnostics must be a mapping",
                field_path="transient_diagnostics",
            )
        object.__setattr__(
            self,
            "transient_diagnostics",
            normalize_json_value(dict(self.transient_diagnostics)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical_serialization": self.canonical_serialization.as_dict(),
            "model_code_decomposition": self.model_code_decomposition.as_dict(),
            "backend_origin_record": self.backend_origin_record.as_dict(),
            "replay_hooks": self.replay_hooks.as_dict(),
            "transient_diagnostics": dict(self.transient_diagnostics),
        }


@dataclass(frozen=True)
class CandidateIntermediateRepresentation:
    structural_layer: CIRStructuralLayer
    execution_layer: CIRExecutionLayer
    evidence_layer: CIREvidenceLayer

    def canonical_bytes(self) -> bytes:
        return self.evidence_layer.canonical_serialization.canonical_bytes

    def canonical_hash(self) -> str:
        return self.evidence_layer.canonical_serialization.content_hash

    def as_dict(self) -> dict[str, Any]:
        return {
            "structural_layer": self.structural_layer.as_dict(),
            "execution_layer": self.execution_layer.as_dict(),
            "evidence_layer": self.evidence_layer.as_dict(),
        }


__all__ = [
    "CIRBackendOriginRecord",
    "CIRCanonicalSerialization",
    "CIREvidenceLayer",
    "CIRExecutionLayer",
    "CIRForecastOperator",
    "CIRHistoryAccessContract",
    "CIRInputSignature",
    "CIRLiteral",
    "CIRLiteralBlock",
    "CIRModelCodeDecomposition",
    "CIRReplayHook",
    "CIRReplayHooks",
    "CIRStateSignature",
    "CIRStructuralLayer",
    "CandidateIntermediateRepresentation",
]
