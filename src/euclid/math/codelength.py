from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.math.quantization import FixedStepMidTreadQuantizer
from euclid.runtime.hashing import sha256_digest

_DEFAULT_DATA_CODE_FAMILY = "residual_signed_integer_elias_delta_v1"


@dataclass(frozen=True)
class CodelengthPolicy:
    quantization_step: str
    literal_code_family: str = "mixed_literal_elias_delta_v1"
    parameter_code_family: str = "float_lattice_zigzag_elias_delta_v1"
    state_code_family: str = "float_lattice_zigzag_elias_delta_v1"
    data_code_family: str = _DEFAULT_DATA_CODE_FAMILY
    parameter_lattice_step: str | None = None
    state_lattice_step: str | None = None

    def __post_init__(self) -> None:
        if self.parameter_lattice_step is None:
            object.__setattr__(
                self,
                "parameter_lattice_step",
                self.quantization_step,
            )
        if self.state_lattice_step is None:
            object.__setattr__(self, "state_lattice_step", self.quantization_step)


@dataclass(frozen=True)
class PrequentialCodeResult:
    total_bits: float
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class StrictComparabilityResult:
    comparable: bool
    reason_code: str | None = None
    details: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CodelengthComparisonKey:
    quantization_mode: str
    quantization_step: str
    reference_policy_id: str
    data_code_family: str
    support_kind: str
    horizon_geometry: tuple[int, ...]
    coding_row_set_id: str
    residual_history_construction: str
    parameter_lattice_step: str
    state_lattice_step: str
    runtime_signature: str = "none"

    def with_update(self, **updates: Any) -> "CodelengthComparisonKey":
        return replace(self, **updates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "quantization_mode": self.quantization_mode,
            "quantization_step": self.quantization_step,
            "reference_policy_id": self.reference_policy_id,
            "data_code_family": self.data_code_family,
            "support_kind": self.support_kind,
            "horizon_geometry": list(self.horizon_geometry),
            "coding_row_set_id": self.coding_row_set_id,
            "residual_history_construction": self.residual_history_construction,
            "parameter_lattice_step": self.parameter_lattice_step,
            "state_lattice_step": self.state_lattice_step,
            "runtime_signature": self.runtime_signature,
        }


def natural_integer_code_length(value: int) -> int:
    if value < 0:
        raise ContractValidationError(
            code="invalid_codelength_value",
            message="natural integer code expects a non-negative integer",
            field_path="value",
        )
    level_1 = math.floor(math.log2(value + 1))
    level_2 = math.floor(math.log2(level_1 + 1))
    return level_1 + (2 * level_2) + 1


def zigzag_signed_index(value: int) -> int:
    return (2 * value) if value >= 0 else (-2 * value) - 1


def signed_integer_code_length(value: int) -> int:
    return natural_integer_code_length(zigzag_signed_index(value))


def float_lattice_index(
    value: float,
    *,
    quantizer: FixedStepMidTreadQuantizer,
) -> int:
    return quantizer.quantize_index(value)


def float_lattice_code_length(
    value: float,
    *,
    quantizer: FixedStepMidTreadQuantizer,
) -> int:
    return signed_integer_code_length(float_lattice_index(value, quantizer=quantizer))


def string_literal_code_length(value: str) -> int:
    encoded = value.encode("utf-8")
    return natural_integer_code_length(len(encoded)) + sum(
        natural_integer_code_length(byte) for byte in encoded
    )


def program_literal_code_length(value: str) -> int:
    token_count = len([token for token in value.replace("(", " ").replace(")", " ").split() if token])
    return natural_integer_code_length(token_count) + string_literal_code_length(value)


def literal_code_length(
    value: Any,
    *,
    quantizer: FixedStepMidTreadQuantizer,
    literal_kind: str = "scalar",
) -> int:
    if literal_kind == "program":
        return program_literal_code_length(str(value))
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return signed_integer_code_length(value)
    if isinstance(value, float):
        return float_lattice_code_length(value, quantizer=quantizer)
    if isinstance(value, str):
        return string_literal_code_length(value)
    if value is None:
        return 1
    raise ContractValidationError(
        code="unsupported_literal_code_value",
        message="literal codelength supports scalar literals only",
        field_path="literal",
    )


def parameter_code_length(
    value: float | int,
    *,
    quantizer: FixedStepMidTreadQuantizer,
) -> int:
    return float_lattice_code_length(float(value), quantizer=quantizer)


def state_code_length(
    value: float | int,
    *,
    quantizer: FixedStepMidTreadQuantizer,
) -> int:
    return float_lattice_code_length(float(value), quantizer=quantizer)


def data_code_length(
    residual_indices: Sequence[int],
    *,
    data_code_family: str = _DEFAULT_DATA_CODE_FAMILY,
) -> float:
    if data_code_family == _DEFAULT_DATA_CODE_FAMILY:
        return float(natural_integer_code_length(len(residual_indices))) + float(
            sum(signed_integer_code_length(index) for index in residual_indices)
        )
    if data_code_family == "prequential_laplace_residual_bin_v1":
        return float(natural_integer_code_length(len(residual_indices))) + (
            prequential_laplace_residual_bin_code(residual_indices).total_bits
        )
    raise ContractValidationError(
        code="unsupported_data_code_family",
        message="data code family is not supported",
        field_path="data_code_family",
        details={"data_code_family": data_code_family},
    )


def prequential_laplace_residual_bin_code(
    residual_indices: Sequence[int],
) -> PrequentialCodeResult:
    counts: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    total_bits = 0.0
    for prefix_count, residual_index in enumerate(residual_indices):
        alphabet_size_before = len(counts)
        symbol_count_before = counts.get(int(residual_index), 0)
        denominator = prefix_count + alphabet_size_before + (
            0 if symbol_count_before else 1
        )
        denominator = max(denominator, 1)
        numerator = symbol_count_before + 1
        probability = numerator / denominator
        bits = -math.log2(probability)
        total_bits += bits
        rows.append(
            {
                "row_index": prefix_count,
                "residual_index": int(residual_index),
                "prefix_count": prefix_count,
                "symbol_count_before": symbol_count_before,
                "alphabet_size_before": alphabet_size_before,
                "probability": round(probability, 12),
                "incremental_bits": round(bits, 12),
                "future_count_used": 0,
            }
        )
        counts[int(residual_index)] = symbol_count_before + 1
    return PrequentialCodeResult(
        total_bits=round(total_bits, 12),
        rows=tuple(rows),
    )


def data_code_diagnostics(
    residual_indices: Sequence[int],
    *,
    data_code_family: str = _DEFAULT_DATA_CODE_FAMILY,
) -> dict[str, Any]:
    if data_code_family == "prequential_laplace_residual_bin_v1":
        result = prequential_laplace_residual_bin_code(residual_indices)
        return {
            "code_family": data_code_family,
            "rows": [dict(row) for row in result.rows],
            "prequential_bits": result.total_bits,
        }
    return {"code_family": data_code_family}


def codelength_terms(
    *,
    family_id: str,
    structure_bits: float,
    literals: Mapping[str, Any],
    parameters: Mapping[str, float | int],
    state: Mapping[str, float | int],
    residual_indices: Sequence[int],
    quantizer: FixedStepMidTreadQuantizer,
    family_bank_size: int,
    data_code_family: str = _DEFAULT_DATA_CODE_FAMILY,
) -> dict[str, float]:
    del family_id
    family_bits = float(natural_integer_code_length(family_bank_size))
    literal_bits = float(
        sum(
            literal_code_length(
                value,
                quantizer=quantizer,
                literal_kind=("program" if "program" in name else "scalar"),
            )
            for name, value in sorted(literals.items())
        )
    )
    parameter_bits = float(
        sum(
            parameter_code_length(value, quantizer=quantizer)
            for _, value in sorted(parameters.items())
        )
    )
    state_bits = float(
        sum(
            state_code_length(value, quantizer=quantizer)
            for _, value in sorted(state.items())
        )
    )
    data_bits = float(
        data_code_length(residual_indices, data_code_family=data_code_family)
    )
    total = (
        family_bits
        + float(structure_bits)
        + literal_bits
        + parameter_bits
        + state_bits
        + data_bits
    )
    return {
        "L_family_bits": family_bits,
        "L_structure_bits": float(structure_bits),
        "L_literals_bits": literal_bits,
        "L_params_bits": parameter_bits,
        "L_state_bits": state_bits,
        "L_data_bits": data_bits,
        "L_total_bits": float(total),
    }


def description_components(
    *,
    family_id: str,
    parameters: Mapping[str, float | int],
    fitted_values: tuple[float, ...],
    actual_values: tuple[float, ...],
    reference_bits: float,
    quantizer: FixedStepMidTreadQuantizer,
    data_code_family: str = _DEFAULT_DATA_CODE_FAMILY,
    family_bank_size: int = 4,
) -> dict[str, float]:
    residual_indices = tuple(
        quantizer.quantize_index(actual - fitted)
        for actual, fitted in zip(actual_values, fitted_values, strict=True)
    )
    structure_bits = 1.0 if family_id == "seasonal_naive" else 0.0
    literal_parameters: dict[str, Any] = {}
    scalar_parameters: dict[str, float | int] = dict(parameters)
    if family_id == "seasonal_naive" and "season_length" in scalar_parameters:
        literal_parameters["season_length"] = int(scalar_parameters.pop("season_length"))
    terms = codelength_terms(
        family_id=family_id,
        structure_bits=structure_bits,
        literals=literal_parameters,
        parameters=scalar_parameters,
        state={},
        residual_indices=residual_indices,
        quantizer=quantizer,
        family_bank_size=family_bank_size,
        data_code_family=data_code_family,
    )
    return {**terms, "reference_bits": float(round(float(reference_bits), 12))}


def coding_row_set_id(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "entity": row.get("entity"),
                "event_time": row.get("event_time"),
                "available_at": row.get("available_at"),
                "target": row.get("target"),
            }
        )
    return f"sha256:{sha256_digest(normalized)}"


def strict_single_class_law_eligibility(
    keys: Sequence[CodelengthComparisonKey],
) -> StrictComparabilityResult:
    if len(keys) <= 1:
        return StrictComparabilityResult(comparable=True)
    first = keys[0]
    field_order = (
        "quantization_mode",
        "quantization_step",
        "reference_policy_id",
        "data_code_family",
        "support_kind",
        "horizon_geometry",
        "coding_row_set_id",
        "residual_history_construction",
        "parameter_lattice_step",
        "state_lattice_step",
        "runtime_signature",
    )
    for other in keys[1:]:
        for field_name in field_order:
            if getattr(first, field_name) != getattr(other, field_name):
                return StrictComparabilityResult(
                    comparable=False,
                    reason_code=f"{field_name}_mismatch",
                    details={
                        "field": field_name,
                        "expected": getattr(first, field_name),
                        "actual": getattr(other, field_name),
                    },
                )
    return StrictComparabilityResult(comparable=True)


def build_codelength_policy_manifest(
    catalog: ContractCatalog,
    *,
    quantizer: FixedStepMidTreadQuantizer,
    target_transform_ref: TypedRef,
    base_measure_policy_ref: TypedRef,
    reference_description_policy_ref: TypedRef,
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="codelength_policy_manifest@1.1.0",
        module_id="candidate_fitting",
        body={
            "policy_id": "prototype_codelength_policy_v1",
            "owner_prompt_id": "prompt.mdl-observation-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "target_transform_ref": target_transform_ref.as_dict(),
            "base_measure_policy_ref": base_measure_policy_ref.as_dict(),
            "quantization_mode": "fixed_step_mid_tread",
            "quantization_step": quantizer.step_string,
            "family_code_policy": "elias_delta_family_bank_size_v1",
            "literal_code_family": "mixed_literal_elias_delta_v1",
            "literal_lattice_step": quantizer.step_string,
            "parameter_code_family": "float_lattice_zigzag_elias_delta_v1",
            "parameter_lattice_step": quantizer.step_string,
            "state_code_family": "float_lattice_zigzag_elias_delta_v1",
            "state_lattice_step": quantizer.step_string,
            "data_code_family": _DEFAULT_DATA_CODE_FAMILY,
            "reference_description_policy_ref": (
                reference_description_policy_ref.as_dict()
            ),
            "cross_family_comparison_rule": "strict_single_class_v1",
            "compatibility_policy_label": "legacy_fixed_step_raw_reference_mdl",
        },
        catalog=catalog,
    )


__all__ = [
    "CodelengthComparisonKey",
    "CodelengthPolicy",
    "PrequentialCodeResult",
    "StrictComparabilityResult",
    "build_codelength_policy_manifest",
    "codelength_terms",
    "coding_row_set_id",
    "data_code_diagnostics",
    "data_code_length",
    "description_components",
    "float_lattice_code_length",
    "float_lattice_index",
    "literal_code_length",
    "natural_integer_code_length",
    "parameter_code_length",
    "prequential_laplace_residual_bin_code",
    "signed_integer_code_length",
    "state_code_length",
    "strict_single_class_law_eligibility",
    "string_literal_code_length",
    "zigzag_signed_index",
]
