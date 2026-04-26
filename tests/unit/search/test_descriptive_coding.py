from __future__ import annotations

from dataclasses import replace

from euclid.cir.models import (
    CIRBackendOriginRecord,
    CIRForecastOperator,
    CIRHistoryAccessContract,
    CIRInputSignature,
    CIRLiteralBlock,
    CIRModelCodeDecomposition,
    CIRReplayHook,
    CIRReplayHooks,
)
from euclid.cir.normalize import build_cir_candidate_from_reducer
from euclid.math.codelength import CodelengthComparisonKey
from euclid.math.observation_models import PointObservationModel
from euclid.math.reference_descriptions import (
    ReferenceDescriptionPolicy,
    build_reference_description,
)
from euclid.math.quantization import FixedStepMidTreadQuantizer
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.reducers.models import (
    BoundObservationModel,
    ReducerAdmissibilityObject,
    ReducerCompositionObject,
    ReducerFamilyId,
    ReducerObject,
    ReducerParameter,
    ReducerParameterObject,
    ReducerStateObject,
    ReducerStateSemantics,
    ReducerStateUpdateContext,
    ReducerStateUpdateRule,
)
from euclid.search.descriptive_coding import evaluate_descriptive_candidates


def test_evaluate_descriptive_candidates_emits_exact_codelength_artifact() -> None:
    feature_view = _feature_view((10.0, 10.0, 10.0, 10.0))
    candidate = _analytic_intercept_candidate(intercept=10.0)

    result = evaluate_descriptive_candidates(
        (candidate,),
        feature_view=feature_view,
    )

    assert len(result.description_artifacts) == 1
    artifact = result.description_artifacts[0]
    diagnostic = result.admissibility_diagnostics[0]
    assert artifact.L_family_bits == 2.0
    assert artifact.L_params_bits == 1.0
    assert artifact.L_data_bits == 8.0
    assert artifact.L_total_bits == 11.0
    assert artifact.reference_bits == 40.0
    assert artifact.description_gain_bits == 29.0
    assert diagnostic.is_admissible is True
    assert diagnostic.reason_codes == ()
    assert result.accepted_candidates == (candidate,)


def test_evaluate_descriptive_candidates_rejects_nonfinite_code_terms() -> None:
    feature_view = _feature_view((10.0, 10.0, 10.0, 10.0))
    candidate = _analytic_intercept_candidate(intercept=10.0)
    object.__setattr__(
        candidate.evidence_layer.model_code_decomposition,
        "L_params_bits",
        float("nan"),
    )

    result = evaluate_descriptive_candidates(
        (candidate,),
        feature_view=feature_view,
    )

    assert result.accepted_candidates == ()
    assert result.description_artifacts == ()
    assert result.admissibility_diagnostics[0].reason_codes == ("nonfinite_code_term",)


def test_evaluate_descriptive_candidates_rejects_invalid_support() -> None:
    feature_view = _feature_view((1.0, 0.0, 1.0, 2.0))
    candidate = _analytic_intercept_candidate(intercept=0.0)
    candidate = replace(
        candidate,
        execution_layer=replace(
            candidate.execution_layer,
            observation_model_binding=BoundObservationModel(
                family="gaussian_location_scale",
                forecast_type="point",
                support_kind="positive_real",
                compatible_point_losses=("absolute_error",),
            ),
        ),
    )

    result = evaluate_descriptive_candidates(
        (candidate,),
        feature_view=feature_view,
    )

    assert result.accepted_candidates == ()
    assert result.admissibility_diagnostics[0].reason_codes == ("support_invalid",)


def test_evaluate_descriptive_candidates_rejects_mixed_comparison_classes() -> None:
    feature_view = _feature_view((10.0, 10.0, 10.0, 10.0))
    left = _analytic_intercept_candidate(intercept=10.0)
    right = replace(
        _analytic_intercept_candidate(intercept=10.0, candidate_id="analytic_b"),
        execution_layer=replace(
            left.execution_layer,
            observation_model_binding=BoundObservationModel(
                family="poisson_count",
                forecast_type="point",
                support_kind="non_negative_real",
                compatible_point_losses=("absolute_error",),
            ),
        ),
    )

    result = evaluate_descriptive_candidates(
        (left, right),
        feature_view=feature_view,
    )

    assert result.accepted_candidates == ()
    assert result.description_artifacts == ()
    assert {
        diagnostic.candidate_id: diagnostic.reason_codes
        for diagnostic in result.admissibility_diagnostics
    } == {
        "analytic_intercept": ("codelength_comparability_failed",),
        "analytic_b": ("codelength_comparability_failed",),
    }


def test_reference_family_bank_supports_raw_naive_seasonal_and_local_linear() -> None:
    quantizer = FixedStepMidTreadQuantizer.from_string("0.5")
    values = (10.0, 11.0, 10.0, 11.0, 10.0, 11.0)

    raw = build_reference_description(
        values,
        quantizer=quantizer,
        policy=ReferenceDescriptionPolicy(
            reference_family_id="raw_quantized_transformed_sequence"
        ),
    )
    naive = build_reference_description(
        values,
        quantizer=quantizer,
        policy=ReferenceDescriptionPolicy(reference_family_id="naive_last_observation"),
    )
    seasonal = build_reference_description(
        values,
        quantizer=quantizer,
        policy=ReferenceDescriptionPolicy(reference_family_id="seasonal_naive"),
        seasonal_period=2,
    )
    local_linear = build_reference_description(
        values,
        quantizer=quantizer,
        policy=ReferenceDescriptionPolicy(
            reference_family_id="differenced_local_linear"
        ),
    )

    assert raw.reference_family_id == "raw_quantized_transformed_sequence"
    assert naive.reference_family_id == "naive_last_observation"
    assert seasonal.reference_family_id == "seasonal_naive"
    assert local_linear.reference_family_id == "differenced_local_linear"
    assert seasonal.reference_bits < raw.reference_bits
    assert all(item.family_selection_bits > 0 for item in (raw, naive, seasonal))


def test_reference_family_selection_cost_is_included_in_reference_bits() -> None:
    quantizer = FixedStepMidTreadQuantizer.from_string("0.5")
    description = build_reference_description(
        (1.0, 1.0, 1.0, 1.0),
        quantizer=quantizer,
        policy=ReferenceDescriptionPolicy(reference_family_id="naive_last_observation"),
    )

    assert description.reference_bits == (
        description.family_selection_bits + description.data_bits
    )
    assert description.family_selection_bits > 0


def test_descriptive_coding_strict_comparability_uses_comparison_key() -> None:
    feature_view = _feature_view((10.0, 10.0, 10.0, 10.0))
    left = _analytic_intercept_candidate(intercept=10.0)
    right = _analytic_intercept_candidate(
        intercept=10.0,
        candidate_id="analytic_intercept_other_quantizer",
    )
    base_key = CodelengthComparisonKey(
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

    result = evaluate_descriptive_candidates(
        (left, right),
        feature_view=feature_view,
        comparison_key_overrides={
            left.canonical_hash(): base_key,
            right.canonical_hash(): base_key.with_update(quantization_step="0.25"),
        },
    )

    assert result.accepted_candidates == ()
    assert {
        diagnostic.details["comparison_failure_reason_code"]
        for diagnostic in result.admissibility_diagnostics
    } == {"quantization_step_mismatch"}


def test_descriptive_coding_strict_comparability_rejects_runtime_signature_mismatch() -> None:
    feature_view = _feature_view((10.0, 10.0, 10.0, 10.0))
    left = _analytic_intercept_candidate(intercept=10.0)
    right = _analytic_intercept_candidate(
        intercept=10.0,
        candidate_id="analytic_intercept_other_runtime",
    )
    base_key = CodelengthComparisonKey(
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
        runtime_signature="none",
    )

    result = evaluate_descriptive_candidates(
        (left, right),
        feature_view=feature_view,
        comparison_key_overrides={
            left.canonical_hash(): base_key,
            right.canonical_hash(): base_key.with_update(
                runtime_signature="sha256:other_runtime"
            ),
        },
    )

    assert result.accepted_candidates == ()
    assert {
        diagnostic.details["comparison_failure_reason_code"]
        for diagnostic in result.admissibility_diagnostics
    } == {"runtime_signature_mismatch"}


def test_prequential_data_code_records_prefix_only_evidence() -> None:
    feature_view = _feature_view((10.0, 10.0, 10.5, 10.0))
    candidate = _analytic_intercept_candidate(intercept=10.0)

    result = evaluate_descriptive_candidates(
        (candidate,),
        feature_view=feature_view,
        data_code_family="prequential_laplace_residual_bin_v1",
    )

    artifact = result.description_artifacts[0]
    assert artifact.data_code_family == "prequential_laplace_residual_bin_v1"
    assert artifact.data_code_diagnostics["rows"]
    assert {
        row["future_count_used"] for row in artifact.data_code_diagnostics["rows"]
    } == {0}
    assert [row["prefix_count"] for row in artifact.data_code_diagnostics["rows"]] == [
        0,
        1,
        2,
    ]


def _feature_view(values: tuple[float, ...]):
    snapshot = FrozenDatasetSnapshot(
        series_id="demo-series",
        cutoff_available_at="2026-01-06T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-0{index + 1}T00:00:00Z",
                available_at=f"2026-01-0{index + 1}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:{index}",
            )
            for index, value in enumerate(values)
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    return materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )


def _analytic_intercept_candidate(
    *,
    intercept: float,
    candidate_id: str = "analytic_intercept",
):
    reducer = ReducerObject(
        family=ReducerFamilyId("analytic"),
        composition_object=ReducerCompositionObject(),
        fitted_parameters=ReducerParameterObject(
            parameters=(ReducerParameter(name="intercept", value=intercept),)
        ),
        state_semantics=ReducerStateSemantics(
            persistent_state=ReducerStateObject(),
            update_rule=ReducerStateUpdateRule(
                update_rule_id=f"{candidate_id}_identity_update",
                implementation=_identity_update,
            ),
        ),
        observation_model=BoundObservationModel.from_runtime(PointObservationModel()),
        admissibility=ReducerAdmissibilityObject(
            family_membership=True,
            composition_closure=True,
            observation_model_compatibility=True,
            valid_state_semantics=True,
            codelength_comparability=True,
        ),
    )
    return build_cir_candidate_from_reducer(
        reducer=reducer,
        cir_form_class="closed_form_expression",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("lag_1",),
        ),
        history_access_contract=CIRHistoryAccessContract(
            contract_id=f"{candidate_id}_history",
            access_mode="full_prefix",
            allowed_side_information=("lag_1",),
        ),
        literal_block=CIRLiteralBlock(),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=CIRModelCodeDecomposition(
            L_family_bits=2.0,
            L_structure_bits=0.0,
            L_literals_bits=0.0,
            L_params_bits=1.0,
            L_state_bits=0.0,
        ),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="analytic-search",
            adapter_class="bounded_grammar",
            source_candidate_id=candidate_id,
            search_class="exact_finite_enumeration",
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(
                CIRReplayHook(
                    hook_name="budget_record",
                    hook_ref="budget:phase04:exact",
                ),
                CIRReplayHook(
                    hook_name="search_seed",
                    hook_ref="seed:0",
                ),
            )
        ),
    )


def _identity_update(
    state: ReducerStateObject,
    context: ReducerStateUpdateContext,
) -> ReducerStateObject:
    return state
