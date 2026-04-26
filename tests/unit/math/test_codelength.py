from __future__ import annotations

from euclid.math.codelength import (
    CodelengthComparisonKey,
    codelength_terms,
    data_code_length,
    float_lattice_code_length,
    float_lattice_index,
    literal_code_length,
    natural_integer_code_length,
    parameter_code_length,
    prequential_laplace_residual_bin_code,
    state_code_length,
    strict_single_class_law_eligibility,
    string_literal_code_length,
    zigzag_signed_index,
)
from euclid.math.quantization import FixedStepMidTreadQuantizer


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
