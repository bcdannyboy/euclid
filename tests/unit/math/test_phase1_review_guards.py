from __future__ import annotations

import inspect
import math

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.math.codelength import (
    CodelengthComparisonKey,
    CodelengthPolicy,
    data_code_diagnostics,
    natural_integer_code_length,
    signed_integer_code_length,
    strict_single_class_law_eligibility,
)
from euclid.math.quantization import FixedStepMidTreadQuantizer
from euclid.math.reference_descriptions import (
    ReferenceDescriptionPolicy,
    build_reference_description,
)


def test_legacy_prequential_laplace_diagnostics_are_claim_tiered_as_proxy() -> None:
    diagnostics = data_code_diagnostics(
        (0, 1, 0),
        data_code_family="prequential_laplace_residual_bin_v1",
    )

    assert diagnostics["coding_claim_tier"] == "mdl_inspired_proxy_score"
    assert "coding_claim_tier_reason_code" in diagnostics
    assert diagnostics["coding_claim_tier_reason_code"] == (
        "legacy_laplace_residual_bin_lacks_explicit_escape_identity_code"
    )


def test_exact_escape_residual_code_emits_esc_identity_and_conserved_bits() -> None:
    try:
        from euclid.math.residual_coding import (
            ResidualAlphabetPolicy,
            prequential_escape_residual_bin_code_v1,
        )
    except ModuleNotFoundError:
        pytest.fail("missing_residual_coding_module_for_phase1_escape_code_contract")

    result = prequential_escape_residual_bin_code_v1(
        (0, 1, 1),
        alphabet_policy=ResidualAlphabetPolicy.fixed_finite(
            (0,),
            escape_policy="explicit_escape_then_symbol_identity",
        ),
    )

    events = [_event_dict(event) for event in result.events]
    assert all("event_type" in event for event in events)
    assert all("symbol" in event for event in events)
    row_one_events = [event for event in events if event["row_index"] == 1]
    row_two_events = [event for event in events if event["row_index"] == 2]
    event_bits = sum(event["incremental_bits"] for event in events)

    assert [event["event_type"] for event in row_one_events] == [
        "ESC",
        "symbol_identity",
    ]
    assert row_one_events[0]["symbol"] == "ESC"
    assert row_one_events[1]["symbol"] == 1
    assert row_one_events[1]["incremental_bits"] == signed_integer_code_length(1)
    assert [event["event_type"] for event in row_two_events] == ["symbol"]
    assert {event["future_count_used"] for event in events} == {0}
    assert math.isclose(
        result.total_bits,
        result.sequence_length_bits + event_bits,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert result.sequence_length_bits == natural_integer_code_length(3)


def test_fixed_residual_alphabet_rejects_unseen_symbol_without_escape_policy() -> None:
    try:
        from euclid.math.residual_coding import (
            ResidualAlphabetPolicy,
            prequential_escape_residual_bin_code_v1,
        )
    except ModuleNotFoundError:
        pytest.fail("missing_residual_coding_module_for_fixed_alphabet_rejection")

    with pytest.raises(ContractValidationError) as exc_info:
        prequential_escape_residual_bin_code_v1(
            (0, 2),
            alphabet_policy=ResidualAlphabetPolicy.fixed_finite(
                (0, 1),
                escape_policy="none",
            ),
        )

    assert exc_info.value.code == "residual_symbol_outside_fixed_alphabet"


def test_reference_description_uses_spec_scope_and_candidate_data_code_family() -> None:
    quantizer = FixedStepMidTreadQuantizer.from_string("0.5")
    _assert_signature_has_parameters(
        ReferenceDescriptionPolicy,
        ("reference_scope",),
        "reference_description_policy_missing_reference_scope",
    )
    _assert_signature_has_parameters(
        build_reference_description,
        ("data_code_family",),
        "build_reference_description_missing_data_code_family",
    )
    policy = ReferenceDescriptionPolicy(
        reference_family_id="naive_last_observation",
        reference_scope="same_family_reference",
    )

    description = build_reference_description(
        (10.0, 11.0, 11.0),
        quantizer=quantizer,
        policy=policy,
        data_code_family="prequential_laplace_residual_bin_v1",
    )

    assert description.reference_scope == "same_family_reference"
    assert description.data_code_family == "prequential_laplace_residual_bin_v1"
    assert description.data_code_diagnostics["coding_claim_tier"] == (
        "mdl_inspired_proxy_score"
    )


def test_codelength_key_separates_reference_scope_and_family() -> None:
    _assert_signature_has_parameters(
        CodelengthComparisonKey,
        ("reference_scope", "reference_family_id"),
        "codelength_comparison_key_missing_reference_scope_or_family",
    )
    same_family_key = _comparison_key(
        reference_scope="same_family_reference",
        reference_family_id="naive_last_observation",
    )
    raw_observation_key = _comparison_key(
        reference_scope="raw_observation_reference",
        reference_family_id="raw_quantized_transformed_sequence",
    )

    result = strict_single_class_law_eligibility(
        (same_family_key, raw_observation_key)
    )

    assert same_family_key.as_dict()["reference_scope"] == "same_family_reference"
    assert same_family_key.as_dict()["reference_family_id"] == "naive_last_observation"
    assert result.comparable is False
    assert result.reason_code == "reference_scope_mismatch"
    assert result.details["field"] == "reference_scope"


def test_cross_family_same_family_reference_requires_global_family_code() -> None:
    try:
        from euclid.math.reference_descriptions import (
            assert_reference_comparison_group_is_eligible,
        )
    except ImportError:
        pytest.fail("missing_reference_comparison_group_eligibility_guard")

    with pytest.raises(ContractValidationError) as exc_info:
        assert_reference_comparison_group_is_eligible(
            candidate_family_ids=("analytic", "recursive"),
            reference_scope="same_family_reference",
            global_family_code_policy=None,
            observation_representation_id="quantized_residual_symbols_v1",
        )

    assert exc_info.value.code == "cross_family_reference_requires_global_family_code"


def test_codelength_policy_records_quantization_lattice_fallback_reason_codes() -> None:
    policy = CodelengthPolicy(quantization_step="0.5")

    assert policy.parameter_lattice_step == "0.5"
    assert policy.state_lattice_step == "0.5"
    assert policy.lattice_fallback_reason_codes == {
        "parameter_lattice_step": "parameter_lattice_defaulted_to_quantization_step",
        "state_lattice_step": "state_lattice_defaulted_to_quantization_step",
    }


def test_refit_api_accepts_active_lattice_policy_for_replay_metadata() -> None:
    from euclid.fit.refit import fit_cir_candidate

    signature = inspect.signature(fit_cir_candidate)

    assert "lattice_policy" in signature.parameters


def _comparison_key(
    *,
    reference_scope: str,
    reference_family_id: str,
) -> CodelengthComparisonKey:
    return CodelengthComparisonKey(
        quantization_mode="fixed_step_mid_tread",
        quantization_step="0.5",
        reference_policy_id=f"{reference_family_id}_v1",
        reference_scope=reference_scope,
        reference_family_id=reference_family_id,
        data_code_family="prequential_laplace_residual_bin_v1",
        support_kind="all_real",
        horizon_geometry=(1,),
        coding_row_set_id="rows:phase1-review",
        residual_history_construction="prefix_residuals_v1",
        parameter_lattice_step="0.5",
        state_lattice_step="0.5",
    )


def _event_dict(event: object) -> dict[str, object]:
    if hasattr(event, "as_dict"):
        return dict(event.as_dict())
    if hasattr(event, "as_diagnostic_row"):
        return dict(event.as_diagnostic_row())
    pytest.fail("residual_code_event_missing_serialization_method")


def _assert_signature_has_parameters(
    target: object,
    parameter_names: tuple[str, ...],
    reason_code: str,
) -> None:
    signature = inspect.signature(target)
    missing = [
        parameter_name
        for parameter_name in parameter_names
        if parameter_name not in signature.parameters
    ]
    assert missing == [], reason_code
