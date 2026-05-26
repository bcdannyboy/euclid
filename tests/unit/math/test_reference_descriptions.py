from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.math.codelength import (
    CodelengthComparisonKey,
    strict_single_class_law_eligibility,
)
from euclid.math.quantization import FixedStepMidTreadQuantizer
from euclid.math.reference_descriptions import (
    ReferenceDescriptionPolicy,
    build_reference_description,
)


def test_reference_description_uses_declared_candidate_data_code_family() -> None:
    quantizer = FixedStepMidTreadQuantizer.from_string("0.5")

    description = build_reference_description(
        (1.0, 1.5, 1.0),
        quantizer=quantizer,
        policy=ReferenceDescriptionPolicy(
            reference_family_id="naive_last_observation",
            reference_scope="same_family_candidate_reference",
        ),
        data_code_family="prequential_escape_residual_bin_v1",
    )

    assert description.data_code_family == "prequential_escape_residual_bin_v1"
    assert description.reference_scope == "same_family_candidate_reference"
    assert description.data_code_diagnostics["code_family"] == (
        "prequential_escape_residual_bin_v1"
    )


def test_codelength_comparison_key_carries_reference_family_and_scope() -> None:
    key = _comparison_key(
        reference_family_id="naive_last_observation",
        reference_scope="same_family_candidate_reference",
    )

    assert key.as_dict()["reference_family_id"] == "naive_last_observation"
    assert key.as_dict()["reference_scope"] == "same_family_candidate_reference"


def test_same_family_reference_key_cannot_compare_against_raw_reference_key() -> None:
    same_family_key = _comparison_key(
        reference_family_id="naive_last_observation",
        reference_scope="same_family_candidate_reference",
    )
    raw_reference_key = _comparison_key(
        reference_family_id="raw_quantized_transformed_sequence",
        reference_scope="raw_reference_baseline",
    )

    result = strict_single_class_law_eligibility((same_family_key, raw_reference_key))

    assert result.comparable is False
    assert result.reason_code == "reference_scope_mismatch"
    assert result.details["field"] == "reference_scope"


def test_cross_family_reference_comparison_requires_global_family_code() -> None:
    from euclid.math.reference_descriptions import (
        assert_reference_comparison_group_is_eligible,
    )

    with pytest.raises(ContractValidationError) as exc_info:
        assert_reference_comparison_group_is_eligible(
            candidate_family_ids=("analytic", "recursive"),
            reference_scope="same_family_candidate_reference",
            global_family_code_policy=None,
            observation_representation_id="quantized_target_residuals_v1",
        )

    assert exc_info.value.code == "cross_family_reference_requires_global_family_code"


def test_cross_family_reference_comparison_requires_common_observation_representation() -> None:
    from euclid.math.reference_descriptions import (
        assert_reference_comparison_group_is_eligible,
    )

    with pytest.raises(ContractValidationError) as exc_info:
        assert_reference_comparison_group_is_eligible(
            candidate_family_ids=("analytic", "recursive"),
            reference_scope="global_family_code",
            global_family_code_policy="elias_delta_family_bank_size_v1",
            observation_representation_id=None,
        )

    assert exc_info.value.code == (
        "cross_family_reference_requires_common_observation_representation"
    )


def _comparison_key(
    *,
    reference_family_id: str,
    reference_scope: str,
) -> CodelengthComparisonKey:
    return CodelengthComparisonKey(
        quantization_mode="fixed_step_mid_tread",
        quantization_step="0.5",
        reference_policy_id=f"{reference_family_id}_v1",
        reference_family_id=reference_family_id,
        reference_scope=reference_scope,
        data_code_family="prequential_escape_residual_bin_v1",
        support_kind="all_real",
        horizon_geometry=(1,),
        coding_row_set_id="rows:test",
        residual_history_construction="prefix_residuals_v1",
        parameter_lattice_step="0.5",
        state_lattice_step="0.5",
    )
