from __future__ import annotations

from types import SimpleNamespace

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.math.codelength import CodelengthComparisonKey, data_code_diagnostics
from euclid.math.lattice import LatticePolicy
from euclid.math.quantization import FixedStepMidTreadQuantizer
from euclid.math.reference_descriptions import (
    ReferenceDescriptionPolicy,
    build_reference_description,
)
from euclid.modules.claims import assert_claim_scope_publication
from euclid.reducers.composition import ReducerCompositionObject
from euclid.search.descriptive_coding import _comparison_status


def test_acceptance_gate_residual_unseen_symbol_behavior_is_auditable() -> None:
    diagnostics = data_code_diagnostics(
        (2, 2, -1),
        data_code_family="prequential_escape_residual_bin_v1",
    )

    row_zero_events = _events_for_row(diagnostics["events"], 0)
    row_one_events = _events_for_row(diagnostics["events"], 1)
    row_two_events = _events_for_row(diagnostics["events"], 2)

    assert [event["event_type"] for event in row_zero_events] == [
        "ESC",
        "symbol_identity",
    ]
    assert [event["event_type"] for event in row_one_events] == ["symbol"]
    assert [event["event_type"] for event in row_two_events] == [
        "ESC",
        "symbol_identity",
    ]
    assert {event["future_count_used"] for event in diagnostics["events"]} == {0}
    assert diagnostics["unseen_symbol_behavior"] == (
        "explicit_escape_then_symbol_identity"
    )


def test_acceptance_gate_reference_and_candidate_coding_match_is_auditable() -> None:
    description = build_reference_description(
        (1.0, 1.5, 1.0),
        quantizer=FixedStepMidTreadQuantizer.from_string("0.5"),
        policy=ReferenceDescriptionPolicy(
            reference_family_id="naive_last_observation",
            reference_scope="same_family_reference",
        ),
        data_code_family="prequential_escape_residual_bin_v1",
    )

    diagnostics = description.data_code_diagnostics or {}

    assert diagnostics["candidate_data_code_family"] == (
        "prequential_escape_residual_bin_v1"
    )
    assert diagnostics["reference_data_code_family"] == (
        "prequential_escape_residual_bin_v1"
    )
    assert diagnostics["reference_candidate_coding_match"] is True


def test_acceptance_gate_lattice_policy_is_active_artifact_not_loose_metadata() -> None:
    policy = LatticePolicy(
        parameter_lattice_step="0.25",
        state_lattice_step="0.125",
        parameter_lattice_reason="adapter_declared_parameter_precision",
        state_lattice_reason="state_encoder_declared_precision",
    )

    artifact = policy.as_artifact()

    assert artifact["artifact_kind"] == "lattice_policy"
    assert artifact["artifact_status"] == "active"
    assert artifact["policy_id"] == "active_lattice_policy_v1"


def test_acceptance_gate_mixed_keys_keep_comparable_group_active() -> None:
    left = _candidate("left")
    right = _candidate("right")
    incompatible = _candidate("incompatible")
    base_key = _comparison_key()

    status = _comparison_status(
        (left, right, incompatible),
        quantization_mode="fixed_step_mid_tread",
        quantizer=FixedStepMidTreadQuantizer.from_string("0.5"),
        reference_policy_id="raw_quantized_transformed_sequence_v1",
        reference_family_id="raw_quantized_transformed_sequence",
        reference_scope="raw_observation_reference",
        data_code_family="prequential_escape_residual_bin_v1",
        horizon_geometry=(1,),
        coding_row_set_id="rows:phase1-acceptance-review",
        residual_history_construction="prefix_residuals_v1",
        parameter_lattice_step="0.5",
        state_lattice_step="0.5",
        comparison_key_overrides={
            left.canonical_hash(): base_key,
            right.canonical_hash(): base_key,
            incompatible.canonical_hash(): base_key.with_update(
                parameter_lattice_step="0.25"
            ),
        },
    )

    assert status.comparable_hashes == frozenset(
        {left.canonical_hash(), right.canonical_hash()}
    )
    assert status.group_size_by_hash[left.canonical_hash()] == 2
    assert status.group_size_by_hash[right.canonical_hash()] == 2

    incompatible_details = status.diagnostic_details(incompatible.canonical_hash())
    assert incompatible_details["comparison_failure_reason_code"] == (
        "no_comparable_peer_in_batch"
    )
    assert incompatible_details["comparison_key"]["parameter_lattice_step"] == "0.25"


def test_acceptance_gate_mdl_language_is_claim_scoped_to_coding_tier() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        assert_claim_scope_publication(
            {
                "claim_type": "descriptive_structure",
                "claim_ceiling": "descriptive_structure",
                "claim_text": (
                    "The candidate is selected by a minimum-description-length "
                    "codelength comparison."
                ),
                "coding_claim_tier": "mdl_inspired_proxy_score",
                "invariance_support_status": "not_requested",
                "transport_support_status": "not_requested",
                "stochastic_support_status": "not_requested",
            }
        )

    assert exc_info.value.code == "claim_scope_overstatement"
    assert "mdl_language_requires_eligible_codelength_claim_tier" in (
        exc_info.value.details["reason_codes"]
    )


def _events_for_row(
    events: list[dict[str, object]],
    row_index: int,
) -> list[dict[str, object]]:
    return [event for event in events if event["row_index"] == row_index]


def _candidate(candidate_hash: str) -> object:
    return SimpleNamespace(
        canonical_hash=lambda: candidate_hash,
        structural_layer=SimpleNamespace(
            composition_graph=ReducerCompositionObject(),
        ),
        execution_layer=SimpleNamespace(
            observation_model_binding=SimpleNamespace(support_kind="all_real"),
        ),
    )


def _comparison_key() -> CodelengthComparisonKey:
    return CodelengthComparisonKey(
        quantization_mode="fixed_step_mid_tread",
        quantization_step="0.5",
        reference_policy_id="raw_quantized_transformed_sequence_v1",
        reference_scope="raw_observation_reference",
        reference_family_id="raw_quantized_transformed_sequence",
        data_code_family="prequential_escape_residual_bin_v1",
        support_kind="all_real",
        horizon_geometry=(1,),
        coding_row_set_id="rows:phase1-acceptance-review",
        residual_history_construction="prefix_residuals_v1",
        parameter_lattice_step="0.5",
        state_lattice_step="0.5",
    )
