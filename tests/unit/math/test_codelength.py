from __future__ import annotations

import math
from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.math.codelength import (
    CodelengthComparisonKey,
    build_codelength_policy_manifest,
    codelength_terms,
    data_code_length,
    data_code_diagnostics,
    float_lattice_code_length,
    float_lattice_index,
    literal_code_length,
    natural_integer_code_length,
    parameter_code_length,
    prequential_laplace_residual_bin_code,
    signed_integer_code_length,
    state_code_length,
    strict_single_class_law_eligibility,
    string_literal_code_length,
    zigzag_signed_index,
)
from euclid.math.quantization import FixedStepMidTreadQuantizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_natural_integer_and_zigzag_signed_code_laws_are_stable() -> None:
    assert natural_integer_code_length(0) == 1
    assert natural_integer_code_length(1) == 4
    assert natural_integer_code_length(3) == 5
    assert zigzag_signed_index(0) == 0
    assert zigzag_signed_index(-1) == 1
    assert zigzag_signed_index(1) == 2
    assert zigzag_signed_index(-2) == 3


def test_float_lattice_code_uses_quantized_signed_integer_index() -> None:
    quantizer = FixedStepMidTreadQuantizer.from_string("0.5")

    assert float_lattice_index(1.25, quantizer=quantizer) == 3
    assert float_lattice_code_length(1.25, quantizer=quantizer) == (
        natural_integer_code_length(zigzag_signed_index(3))
    )


def test_literal_parameter_and_state_code_terms_share_lattice_laws() -> None:
    quantizer = FixedStepMidTreadQuantizer.from_string("0.5")

    assert literal_code_length(2, quantizer=quantizer) == (
        natural_integer_code_length(zigzag_signed_index(2))
    )
    assert parameter_code_length(1.0, quantizer=quantizer) == (
        float_lattice_code_length(1.0, quantizer=quantizer)
    )
    assert state_code_length(-1.0, quantizer=quantizer) == (
        float_lattice_code_length(-1.0, quantizer=quantizer)
    )


def test_string_and_program_literals_have_content_sensitive_cost() -> None:
    quantizer = FixedStepMidTreadQuantizer.from_string("0.5")

    short = string_literal_code_length("x")
    long = string_literal_code_length("longer literal")
    program = literal_code_length(
        "(add (lag 1) (const 2))",
        quantizer=quantizer,
        literal_kind="program",
    )

    assert long > short
    assert program > short


def test_finer_precision_does_not_reduce_scalar_codelength() -> None:
    coarse = FixedStepMidTreadQuantizer.from_string("0.5")
    fine = FixedStepMidTreadQuantizer.from_string("0.25")

    assert parameter_code_length(1.25, quantizer=fine) >= parameter_code_length(
        1.25,
        quantizer=coarse,
    )


def test_model_codelength_terms_include_literals_parameters_state_and_data() -> None:
    quantizer = FixedStepMidTreadQuantizer.from_string("0.5")
    terms = codelength_terms(
        family_id="analytic",
        structure_bits=2.0,
        literals={"season_length": 4, "program": "(lag 1)"},
        parameters={"intercept": 1.0},
        state={"level": 2.0},
        residual_indices=(0, -1, 2),
        quantizer=quantizer,
        family_bank_size=4,
    )

    assert terms["L_family_bits"] == natural_integer_code_length(4)
    assert terms["L_structure_bits"] == 2.0
    assert terms["L_literals_bits"] > 0
    assert terms["L_params_bits"] == parameter_code_length(1.0, quantizer=quantizer)
    assert terms["L_state_bits"] == state_code_length(2.0, quantizer=quantizer)
    assert terms["L_data_bits"] == data_code_length((0, -1, 2))
    assert terms["L_total_bits"] == sum(
        terms[field]
        for field in (
            "L_family_bits",
            "L_structure_bits",
            "L_literals_bits",
            "L_params_bits",
            "L_state_bits",
            "L_data_bits",
        )
    )


def test_strict_single_class_law_eligibility_reports_each_key_difference() -> None:
    base = CodelengthComparisonKey(
        quantization_mode="fixed_step_mid_tread",
        quantization_step="0.5",
        reference_policy_id="raw_quantized_transformed_sequence_v1",
        data_code_family="residual_signed_integer_elias_delta_v1",
        support_kind="all_real",
        horizon_geometry=(1,),
        coding_row_set_id="rows:a",
        residual_history_construction="none",
        parameter_lattice_step="0.5",
        state_lattice_step="0.5",
    )

    for field_name, replacement in (
        ("quantization_step", "0.25"),
        ("reference_policy_id", "seasonal_naive_reference_v1"),
        ("data_code_family", "prequential_laplace_residual_bin_v1"),
        ("support_kind", "positive_real"),
        ("horizon_geometry", (1, 3)),
        ("coding_row_set_id", "rows:b"),
        ("residual_history_construction", "legal_panel_v1"),
        ("parameter_lattice_step", "0.25"),
        ("state_lattice_step", "0.25"),
        ("runtime_signature", "sha256:runtime"),
    ):
        candidate = base.with_update(**{field_name: replacement})
        result = strict_single_class_law_eligibility((base, candidate))
        assert result.comparable is False
        assert result.reason_code == f"{field_name}_mismatch"


def test_prequential_laplace_residual_bin_code_uses_prefix_only_evidence() -> None:
    result = prequential_laplace_residual_bin_code((0, 0, 1, 0))

    assert result.total_bits > 0
    assert [row["prefix_count"] for row in result.rows] == [0, 1, 2, 3]
    assert [row["future_count_used"] for row in result.rows] == [0, 0, 0, 0]
    assert result.rows[1]["symbol_count_before"] == 1
    assert result.rows[2]["symbol_count_before"] == 0


def test_escape_residual_code_charges_escape_before_new_symbol_identity() -> None:
    from euclid.math.residual_coding import (
        ResidualAlphabetPolicy,
        prequential_escape_residual_bin_code_v1,
    )

    result = prequential_escape_residual_bin_code_v1(
        (0, 1, 0),
        alphabet_policy=ResidualAlphabetPolicy(
            alphabet=(0,),
            escape_policy="explicit_escape_then_symbol_identity",
        ),
    )

    assert [event.event_type for event in result.events[:3]] == [
        "symbol",
        "ESC",
        "symbol_identity",
    ]
    assert result.events[1].symbol == "ESC"
    assert result.events[2].symbol == 1
    assert result.events[1].incremental_bits > 0
    assert result.events[2].incremental_bits == signed_integer_code_length(1)


def test_escape_residual_code_does_not_charge_escape_for_seen_symbol() -> None:
    from euclid.math.residual_coding import (
        ResidualAlphabetPolicy,
        prequential_escape_residual_bin_code_v1,
    )

    result = prequential_escape_residual_bin_code_v1(
        (0, 1, 1),
        alphabet_policy=ResidualAlphabetPolicy(
            alphabet=(0,),
            escape_policy="explicit_escape_then_symbol_identity",
        ),
    )

    row_two_events = [
        event for event in result.events if event.residual_index == 1 and event.row_index == 2
    ]
    assert [event.event_type for event in row_two_events] == ["symbol"]
    assert row_two_events[0].symbol == 1


def test_escape_residual_diagnostics_are_prefix_only_and_sum_to_data_bits() -> None:
    residuals = (0, 1, 1, -1)
    diagnostics = data_code_diagnostics(
        residuals,
        data_code_family="prequential_escape_residual_bin_v1",
        alphabet_policy={
            "alphabet": (0,),
            "escape_policy": "explicit_escape_then_symbol_identity",
        },
    )

    event_bits = sum(event["incremental_bits"] for event in diagnostics["events"])
    sequence_length_bits = diagnostics["sequence_length_bits"]

    assert {event["future_count_used"] for event in diagnostics["events"]} == {0}
    assert math.isclose(
        diagnostics["total_bits"],
        sequence_length_bits + event_bits,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_fixed_residual_alphabet_rejects_unseen_symbol_without_escape_policy() -> None:
    from euclid.math.residual_coding import (
        ResidualAlphabetPolicy,
        prequential_escape_residual_bin_code_v1,
    )

    with pytest.raises(ContractValidationError) as exc_info:
        prequential_escape_residual_bin_code_v1(
            (0, 2),
            alphabet_policy=ResidualAlphabetPolicy(
                alphabet=(0, 1),
                escape_policy="none",
            ),
        )

    assert exc_info.value.code == "residual_symbol_outside_fixed_alphabet"


def test_legacy_fixed_step_raw_reference_policy_declares_proxy_claim_tier() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    manifest = build_codelength_policy_manifest(
        catalog,
        quantizer=FixedStepMidTreadQuantizer.from_string("0.5"),
        target_transform_ref=TypedRef(
            "target_transform_manifest@1.0.0",
            "target-transform",
        ),
        base_measure_policy_ref=TypedRef(
            "base_measure_policy_manifest@1.0.0",
            "base-measure",
        ),
        reference_description_policy_ref=TypedRef(
            "reference_description_policy_manifest@1.1.0",
            "raw-reference",
        ),
    )

    assert manifest.body["compatibility_policy_label"] == (
        "legacy_fixed_step_raw_reference_mdl"
    )
    assert manifest.body["coding_claim_tier"] == "mdl_inspired_proxy_score"
    assert manifest.body["coding_claim_tier_reason_code"] == (
        "legacy_fixed_step_raw_reference_policy_is_proxy_score"
    )
