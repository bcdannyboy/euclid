from __future__ import annotations

from euclid.math.codelength import data_code_length
from euclid.math.quantization import FixedStepMidTreadQuantizer
from euclid.math.reference_descriptions import (
    ReferenceDescriptionPolicy,
    build_reference_description,
)


def test_same_family_reference_description_uses_declared_data_code_family() -> None:
    quantizer = FixedStepMidTreadQuantizer.from_string("0.5")
    policy = ReferenceDescriptionPolicy(
        reference_family_id="naive_last_observation",
        reference_scope="same_family_reference",
    )

    description = build_reference_description(
        (10.0, 11.0, 11.5),
        quantizer=quantizer,
        policy=policy,
        data_code_family="prequential_laplace_residual_bin_v1",
    )

    assert description.reference_scope == "same_family_reference"
    assert description.data_code_family == "prequential_laplace_residual_bin_v1"
    assert description.data_bits == data_code_length(
        description.encoded_residual_indices,
        data_code_family="prequential_laplace_residual_bin_v1",
    )
