from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from euclid.contracts.loader import ContractCatalog
from euclid.manifests.base import ManifestEnvelope
from euclid.math.quantization import FixedStepMidTreadQuantizer


def _natural_code_length(value: int) -> int:
    level_1 = math.floor(math.log2(value + 1))
    level_2 = math.floor(math.log2(level_1 + 1))
    return level_1 + (2 * level_2) + 1


def _zigzag_encode(value: int) -> int:
    return (2 * value) if value >= 0 else (-2 * value) - 1


@dataclass(frozen=True)
class ReferenceDescription:
    quantized_sequence: tuple[int, ...]
    reference_bits: int


def build_reference_description(
    observed_values: Iterable[float],
    *,
    quantizer: FixedStepMidTreadQuantizer,
) -> ReferenceDescription:
    quantized_sequence = quantizer.quantize_sequence(observed_values)
    reference_bits = _natural_code_length(len(quantized_sequence))
    reference_bits += sum(
        _natural_code_length(_zigzag_encode(value))
        for value in quantized_sequence
    )
    return ReferenceDescription(
        quantized_sequence=quantized_sequence,
        reference_bits=reference_bits,
    )


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
            "sequence_length_code_family": "elias_delta_nat_v1",
            "symbol_code_family": "zigzag_elias_delta_v1",
            "formula_id": "raw_quantized_transformed_sequence_v1",
        },
        catalog=catalog,
    )
