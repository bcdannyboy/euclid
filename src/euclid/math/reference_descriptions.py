from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.manifests.base import ManifestEnvelope
from euclid.math.codelength import (
    data_code_length,
    natural_integer_code_length,
    zigzag_signed_index,
)
from euclid.math.quantization import FixedStepMidTreadQuantizer


def _natural_code_length(value: int) -> int:
    return natural_integer_code_length(value)


def _zigzag_encode(value: int) -> int:
    return zigzag_signed_index(value)


@dataclass(frozen=True)
class ReferenceDescriptionPolicy:
    reference_family_id: str = "raw_quantized_transformed_sequence"
    policy_id: str | None = None

    @property
    def resolved_policy_id(self) -> str:
        if self.policy_id is not None:
            return self.policy_id
        return f"{self.reference_family_id}_v1"


@dataclass(frozen=True)
class ReferenceDescription:
    quantized_sequence: tuple[int, ...]
    reference_bits: float
    reference_family_id: str = "raw_quantized_transformed_sequence"
    reference_policy_id: str = "raw_quantized_transformed_sequence_v1"
    family_selection_bits: float = 0.0
    data_bits: float = 0.0
    encoded_residual_indices: tuple[int, ...] = ()


def reference_family_bank() -> tuple[str, ...]:
    return (
        "raw_quantized_transformed_sequence",
        "naive_last_observation",
        "seasonal_naive",
        "differenced_local_linear",
    )


def build_reference_description(
    observed_values: Iterable[float],
    *,
    quantizer: FixedStepMidTreadQuantizer,
    policy: ReferenceDescriptionPolicy | None = None,
    seasonal_period: int | None = None,
) -> ReferenceDescription:
    values = tuple(float(value) for value in observed_values)
    quantized_sequence = quantizer.quantize_sequence(values)
    active_policy = policy or ReferenceDescriptionPolicy()
    family_id = active_policy.reference_family_id
    if family_id not in reference_family_bank():
        raise ContractValidationError(
            code="unsupported_reference_family",
            message="reference description family is not supported",
            field_path="reference_description_policy.reference_family_id",
            details={"reference_family_id": family_id},
        )
    encoded_residual_indices = _reference_residual_indices(
        family_id=family_id,
        values=values,
        quantizer=quantizer,
        seasonal_period=seasonal_period,
    )
    data_bits = data_code_length(encoded_residual_indices)
    family_selection_bits = float(natural_integer_code_length(len(reference_family_bank())))
    return ReferenceDescription(
        quantized_sequence=quantized_sequence,
        reference_bits=float(family_selection_bits + data_bits),
        reference_family_id=family_id,
        reference_policy_id=active_policy.resolved_policy_id,
        family_selection_bits=family_selection_bits,
        data_bits=float(data_bits),
        encoded_residual_indices=encoded_residual_indices,
    )


def _reference_residual_indices(
    *,
    family_id: str,
    values: tuple[float, ...],
    quantizer: FixedStepMidTreadQuantizer,
    seasonal_period: int | None,
) -> tuple[int, ...]:
    if family_id == "raw_quantized_transformed_sequence":
        return quantizer.quantize_sequence(values)
    if family_id == "naive_last_observation":
        residuals = []
        for index, value in enumerate(values):
            baseline = 0.0 if index == 0 else values[index - 1]
            residuals.append(quantizer.quantize_index(value - baseline))
        return tuple(residuals)
    if family_id == "seasonal_naive":
        period = int(seasonal_period or 1)
        if period <= 0:
            raise ContractValidationError(
                code="invalid_reference_family_config",
                message="seasonal_naive reference requires a positive seasonal_period",
                field_path="seasonal_period",
            )
        residuals = []
        for index, value in enumerate(values):
            baseline = 0.0 if index < period else values[index - period]
            residuals.append(quantizer.quantize_index(value - baseline))
        return tuple(residuals)
    if family_id == "differenced_local_linear":
        residuals = []
        for index, value in enumerate(values):
            if index == 0:
                baseline = 0.0
            elif index == 1:
                baseline = values[index - 1]
            else:
                baseline = values[index - 1] + (values[index - 1] - values[index - 2])
            residuals.append(quantizer.quantize_index(value - baseline))
        return tuple(residuals)
    raise AssertionError(f"unhandled reference family: {family_id}")


def build_reference_description_policy_manifest(
    catalog: ContractCatalog,
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="reference_description_policy_manifest@1.1.0",
        module_id="candidate_fitting",
        body={
            "policy_id": "raw_quantized_reference_description_policy_v1",
            "owner_prompt_id": "prompt.mdl-observation-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "reference_kind": "raw_quantized_transformed_sequence",
            "reference_family_bank": list(reference_family_bank()),
            "sequence_length_code_family": "elias_delta_nat_v1",
            "symbol_code_family": "zigzag_elias_delta_v1",
            "family_selection_code_family": "elias_delta_family_bank_size_v1",
            "formula_id": "raw_quantized_transformed_sequence_v1",
            "compatibility_policy_label": "legacy_raw_reference_description",
        },
        catalog=catalog,
    )


__all__ = [
    "ReferenceDescription",
    "ReferenceDescriptionPolicy",
    "build_reference_description",
    "build_reference_description_policy_manifest",
    "reference_family_bank",
]
