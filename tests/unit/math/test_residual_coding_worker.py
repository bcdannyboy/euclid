from __future__ import annotations

import importlib
import importlib.util
import math

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.math.codelength import data_code_diagnostics, data_code_length


def test_prequential_escape_code_records_explicit_escape_and_identity_charge() -> None:
    spec = importlib.util.find_spec("euclid.math.residual_coding")
    assert spec is not None
    residual_coding = importlib.import_module("euclid.math.residual_coding")

    policy = residual_coding.ResidualAlphabetPolicy.open_prequential()
    result = residual_coding.prequential_escape_residual_bin_code_v1(
        (0, 0, 2),
        alphabet_policy=policy,
    )

    event_kinds = [event.event_kind for event in result.events]
    assert event_kinds == [
        "ESC",
        "SYMBOL_IDENTITY",
        "SYMBOL",
        "ESC",
        "SYMBOL_IDENTITY",
    ]
    assert [event.residual_index for event in result.events if event.event_kind == "ESC"] == [
        0,
        2,
    ]
    assert all(event.future_count_used == 0 for event in result.events)

    event_bits = sum(event.event_bits for event in result.events)
    assert result.total_bits == pytest.approx(result.sequence_length_bits + event_bits)
    assert result.sequence_length_bits > 0


def test_prequential_escape_diagnostics_expose_policy_and_claim_tier() -> None:
    diagnostics = data_code_diagnostics(
        (0, 0, 2),
        data_code_family="prequential_escape_residual_bin_code_v1",
    )

    assert diagnostics["coding_claim_tier"] == "exact_prequential_symbol_code"
    assert diagnostics["sequence_length_bits"] > 0
    assert diagnostics["prequential_bits"] == pytest.approx(
        diagnostics["sequence_length_bits"]
        + sum(row["incremental_bits"] for row in diagnostics["rows"])
    )
    assert {row["future_count_used"] for row in diagnostics["rows"]} == {0}
    assert {row["alphabet_mode"] for row in diagnostics["rows"]} == {
        "open_prequential"
    }
    assert {row["escape_policy"] for row in diagnostics["rows"]} == {
        "explicit_escape_on_first_seen_v1"
    }
    assert {row["innovation_code_family"] for row in diagnostics["rows"]} == {
        "signed_integer_elias_delta_v1"
    }
    assert all("symbol_identity_bits" in row for row in diagnostics["rows"])
    assert data_code_length(
        (0, 0, 2),
        data_code_family="prequential_escape_residual_bin_code_v1",
    ) == pytest.approx(diagnostics["prequential_bits"])


def test_fixed_finite_alphabet_rejects_out_of_alphabet_without_escape() -> None:
    residual_coding = importlib.import_module("euclid.math.residual_coding")
    policy = residual_coding.ResidualAlphabetPolicy.fixed_finite(
        (-1, 0, 1),
        escape_policy=None,
    )

    with pytest.raises(ContractValidationError) as exc_info:
        residual_coding.prequential_escape_residual_bin_code_v1(
            (0, 2),
            alphabet_policy=policy,
        )

    assert exc_info.value.code == "residual_symbol_outside_fixed_alphabet"


def test_fixed_finite_alphabet_allows_out_of_alphabet_with_escape_policy() -> None:
    residual_coding = importlib.import_module("euclid.math.residual_coding")
    policy = residual_coding.ResidualAlphabetPolicy.fixed_finite(
        (-1, 0, 1),
        escape_policy="explicit_escape_on_first_seen_v1",
    )

    result = residual_coding.prequential_escape_residual_bin_code_v1(
        (0, 2),
        alphabet_policy=policy,
    )

    assert any(event.event_kind == "ESC" for event in result.events)
    assert math.isfinite(result.total_bits)


def test_legacy_laplace_diagnostics_are_marked_proxy() -> None:
    diagnostics = data_code_diagnostics(
        (0, 1),
        data_code_family="prequential_laplace_residual_bin_v1",
    )

    assert diagnostics["coding_claim_tier"] == "mdl_inspired_proxy_score"
