from __future__ import annotations

import importlib.util
from pathlib import Path

from euclid.math.codelength import CodelengthComparisonKey
from euclid.search.descriptive_coding import evaluate_descriptive_candidates

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_descriptive_coding_test_helpers",
    Path(__file__).with_name("test_descriptive_coding.py"),
)
assert _HELPER_SPEC is not None
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
assert _HELPER_SPEC.loader is not None
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_analytic_intercept_candidate = _HELPER_MODULE._analytic_intercept_candidate
_feature_view = _HELPER_MODULE._feature_view


def test_mixed_comparison_keys_create_multiple_comparable_groups() -> None:
    feature_view = _feature_view((10.0, 10.0, 10.0, 10.0))
    group_a_left = _analytic_intercept_candidate(
        intercept=10.0,
        candidate_id="group_a_left",
    )
    group_a_right = _analytic_intercept_candidate(
        intercept=10.0,
        candidate_id="group_a_right",
    )
    group_b_left = _analytic_intercept_candidate(
        intercept=10.0,
        candidate_id="group_b_left",
    )
    group_b_right = _analytic_intercept_candidate(
        intercept=10.0,
        candidate_id="group_b_right",
    )
    base_key = _comparison_key()
    group_b_key = base_key.with_update(reference_family_id="naive_last_observation")

    result = evaluate_descriptive_candidates(
        (group_a_left, group_a_right, group_b_left, group_b_right),
        feature_view=feature_view,
        comparison_key_overrides={
            group_a_left.canonical_hash(): base_key,
            group_a_right.canonical_hash(): base_key,
            group_b_left.canonical_hash(): group_b_key,
            group_b_right.canonical_hash(): group_b_key,
        },
    )

    group_ids = {
        artifact.comparable_group_id for artifact in result.description_artifacts
    }
    assert len(group_ids) == 2
    assert {
        artifact.candidate_id: artifact.comparable_group_size
        for artifact in result.description_artifacts
    } == {
        "group_a_left": 2,
        "group_a_right": 2,
        "group_b_left": 2,
        "group_b_right": 2,
    }
    assert all(
        diagnostic.codelength_comparability
        for diagnostic in result.admissibility_diagnostics
    )


def test_singleton_mixed_key_candidate_gets_explicit_non_comparable_diagnostic() -> None:
    feature_view = _feature_view((10.0, 10.0, 10.0, 10.0))
    group_left = _analytic_intercept_candidate(
        intercept=10.0,
        candidate_id="group_left",
    )
    group_right = _analytic_intercept_candidate(
        intercept=10.0,
        candidate_id="group_right",
    )
    singleton = _analytic_intercept_candidate(
        intercept=10.0,
        candidate_id="singleton",
    )
    base_key = _comparison_key()

    result = evaluate_descriptive_candidates(
        (group_left, group_right, singleton),
        feature_view=feature_view,
        comparison_key_overrides={
            group_left.canonical_hash(): base_key,
            group_right.canonical_hash(): base_key,
            singleton.canonical_hash(): base_key.with_update(
                parameter_lattice_step="0.25"
            ),
        },
    )

    singleton_diagnostic = next(
        diagnostic
        for diagnostic in result.admissibility_diagnostics
        if diagnostic.candidate_id == "singleton"
    )
    assert singleton_diagnostic.codelength_comparability is False
    assert singleton_diagnostic.reason_codes == ("codelength_comparability_failed",)
    assert (
        singleton_diagnostic.details["comparison_failure_reason_code"]
        == "no_comparable_peer_in_batch"
    )
    assert singleton_diagnostic.details["comparable_group_size"] == 1


def test_best_candidate_is_selected_only_within_each_comparable_group() -> None:
    feature_view = _feature_view((10.0, 10.0, 10.0, 10.0))
    group_a_best = _candidate_with_param_bits("group_a_best", bits=1.0)
    group_a_loser = _candidate_with_param_bits("group_a_loser", bits=2.0)
    group_b_best = _candidate_with_param_bits("group_b_best", bits=20.0)
    group_b_loser = _candidate_with_param_bits("group_b_loser", bits=30.0)
    base_key = _comparison_key()
    group_b_key = base_key.with_update(reference_family_id="naive_last_observation")

    result = evaluate_descriptive_candidates(
        (group_a_best, group_a_loser, group_b_best, group_b_loser),
        feature_view=feature_view,
        comparison_key_overrides={
            group_a_best.canonical_hash(): base_key,
            group_a_loser.canonical_hash(): base_key,
            group_b_best.canonical_hash(): group_b_key,
            group_b_loser.canonical_hash(): group_b_key,
        },
    )

    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.selected_candidates
    ] == ["group_a_best", "group_b_best"]
    assert {
        artifact.candidate_id: artifact.comparable_group_rank
        for artifact in result.description_artifacts
    } == {
        "group_a_best": 1,
        "group_a_loser": 2,
        "group_b_best": 1,
        "group_b_loser": 2,
    }


def _candidate_with_param_bits(candidate_id: str, *, bits: float):
    candidate = _analytic_intercept_candidate(
        intercept=10.0,
        candidate_id=candidate_id,
    )
    object.__setattr__(
        candidate.evidence_layer.model_code_decomposition,
        "L_params_bits",
        bits,
    )
    return candidate


def _comparison_key() -> CodelengthComparisonKey:
    return CodelengthComparisonKey(
        quantization_mode="fixed_step_mid_tread",
        quantization_step="0.5",
        reference_policy_id="raw_quantized_transformed_sequence_v1",
        reference_family_id="raw_quantized_transformed_sequence",
        reference_scope="raw_observation_reference",
        data_code_family="residual_signed_integer_elias_delta_v1",
        support_kind="all_real",
        horizon_geometry=(1,),
        coding_row_set_id="rows:a",
        residual_history_construction="none",
        parameter_lattice_step="0.5",
        state_lattice_step="0.5",
    )
