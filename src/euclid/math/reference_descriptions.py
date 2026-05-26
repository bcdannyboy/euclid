from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.manifests.base import ManifestEnvelope
from euclid.math.codelength import (
    data_code_diagnostics,
    data_code_length,
    natural_integer_code_length,
    zigzag_signed_index,
)
from euclid.math.quantization import FixedStepMidTreadQuantizer

_DEFAULT_REFERENCE_SCOPE = "raw_observation_reference"
_SUPPORTED_REFERENCE_SCOPES = frozenset(
    {
        "raw_observation_reference",
        "raw_reference_baseline",
        "same_family_reference",
        "same_family_candidate_reference",
        "global_family_code",
        "global_family_mixture_reference",
    }
)


def _natural_code_length(value: int) -> int:
    return natural_integer_code_length(value)


def _zigzag_encode(value: int) -> int:
    return zigzag_signed_index(value)


@dataclass(frozen=True)
class ReferenceDescriptionPolicy:
    reference_family_id: str = "raw_quantized_transformed_sequence"
    policy_id: str | None = None
    reference_scope: str = _DEFAULT_REFERENCE_SCOPE

    @property
    def resolved_policy_id(self) -> str:
        if self.policy_id is not None:
            return self.policy_id
        return f"{self.reference_family_id}_v1"

    def __post_init__(self) -> None:
        if self.reference_scope not in _SUPPORTED_REFERENCE_SCOPES:
            raise ContractValidationError(
                code="unsupported_reference_scope",
                message="reference description scope is not supported",
                field_path="reference_description_policy.reference_scope",
                details={"reference_scope": self.reference_scope},
            )


@dataclass(frozen=True)
class ReferenceDescription:
    quantized_sequence: tuple[int, ...]
    reference_bits: float
    reference_family_id: str = "raw_quantized_transformed_sequence"
    reference_policy_id: str = "raw_quantized_transformed_sequence_v1"
    reference_scope: str = _DEFAULT_REFERENCE_SCOPE
    data_code_family: str = "residual_signed_integer_elias_delta_v1"
    family_selection_bits: float = 0.0
    data_bits: float = 0.0
    encoded_residual_indices: tuple[int, ...] = ()
    data_code_diagnostics: dict[str, object] | None = None


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
    data_code_family: str = "residual_signed_integer_elias_delta_v1",
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
    data_bits = data_code_length(
        encoded_residual_indices,
        data_code_family=data_code_family,
    )
    diagnostics = data_code_diagnostics(
        encoded_residual_indices,
        data_code_family=data_code_family,
    )
    diagnostics = {
        **diagnostics,
        "candidate_data_code_family": data_code_family,
        "reference_data_code_family": data_code_family,
        "reference_candidate_coding_match": True,
    }
    family_selection_bits = float(natural_integer_code_length(len(reference_family_bank())))
    return ReferenceDescription(
        quantized_sequence=quantized_sequence,
        reference_bits=float(family_selection_bits + data_bits),
        reference_family_id=family_id,
        reference_policy_id=active_policy.resolved_policy_id,
        reference_scope=active_policy.reference_scope,
        data_code_family=data_code_family,
        family_selection_bits=family_selection_bits,
        data_bits=float(data_bits),
        encoded_residual_indices=encoded_residual_indices,
        data_code_diagnostics=diagnostics,
    )


def assert_reference_comparison_group_is_eligible(
    *,
    candidate_family_ids: Iterable[str],
    reference_scope: str,
    global_family_code_policy: str | None,
    observation_representation_id: str | None,
) -> None:
    if reference_scope not in _SUPPORTED_REFERENCE_SCOPES:
        raise ContractValidationError(
            code="unsupported_reference_scope",
            message="reference description scope is not supported",
            field_path="reference_scope",
            details={"reference_scope": reference_scope},
        )
    unique_family_ids = frozenset(str(family_id) for family_id in candidate_family_ids)
    if len(unique_family_ids) <= 1:
        return
    if reference_scope in {"same_family_reference", "same_family_candidate_reference"}:
        raise ContractValidationError(
            code="cross_family_reference_requires_global_family_code",
            message=(
                "cross-family reference comparisons require a global family code "
                "rather than a same-family reference scope"
            ),
            field_path="reference_scope",
            details={"candidate_family_ids": sorted(unique_family_ids)},
        )
    if reference_scope in {"global_family_code", "global_family_mixture_reference"}:
        if not global_family_code_policy:
            raise ContractValidationError(
                code="cross_family_reference_requires_global_family_code",
                message="cross-family reference comparisons require a global family code policy",
                field_path="global_family_code_policy",
                details={"candidate_family_ids": sorted(unique_family_ids)},
            )
        if not observation_representation_id:
            raise ContractValidationError(
                code="cross_family_reference_requires_common_observation_representation",
                message=(
                    "cross-family reference comparisons require a common observation "
                    "representation"
                ),
                field_path="observation_representation_id",
                details={"candidate_family_ids": sorted(unique_family_ids)},
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
            "reference_scope": _DEFAULT_REFERENCE_SCOPE,
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
    "assert_reference_comparison_group_is_eligible",
    "build_reference_description",
    "build_reference_description_policy_manifest",
    "reference_family_bank",
]
