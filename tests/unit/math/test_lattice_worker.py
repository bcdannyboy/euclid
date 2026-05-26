from __future__ import annotations

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
from euclid.cir.normalize import (
    build_cir_candidate_from_expression,
    build_cir_candidate_from_reducer,
)
from euclid.expr.ast import BinaryOp, Feature, Parameter
from euclid.fit.parameterization import ParameterDeclaration
from euclid.fit.refit import FitDataSplit, fit_cir_candidate
from euclid.math.codelength import CodelengthComparisonKey
from euclid.math.lattice import LatticePolicy
from euclid.math.observation_models import PointObservationModel
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


def test_lattice_policy_serializes_parameter_and_state_steps_with_reasons() -> None:
    policy = LatticePolicy(
        parameter_lattice_step="0.25",
        state_lattice_step="0.5",
        parameter_lattice_reason="adapter_declared_parameter_precision",
        state_lattice_reason="state_encoder_declared_precision",
    )

    assert policy.as_dict() == {
        "policy_id": "active_lattice_policy_v1",
        "parameter_lattice_step": "0.25",
        "state_lattice_step": "0.5",
        "parameter_lattice_reason": "adapter_declared_parameter_precision",
        "state_lattice_reason": "state_encoder_declared_precision",
    }


def test_descriptive_coding_records_active_lattice_policy_and_fallback_reason() -> None:
    candidate = _analytic_intercept_candidate(intercept=10.0)

    result = evaluate_descriptive_candidates(
        (candidate,),
        feature_view=_feature_view((10.0, 10.0, 10.0, 10.0)),
        parameter_lattice_step="0.25",
    )

    artifact = result.description_artifacts[0]
    assert artifact.lattice_policy["parameter_lattice_step"] == "0.25"
    assert artifact.lattice_policy["state_lattice_step"] == "0.5"
    assert artifact.lattice_policy["state_lattice_reason"] == (
        "defaults_to_residual_quantization_step"
    )
    assert artifact.model_code_decomposition["parameter_lattice_step"] == "0.25"
    assert artifact.model_code_decomposition["state_lattice_step"] == "0.5"


def test_fit_refit_replay_metadata_includes_active_lattice_policy() -> None:
    policy = LatticePolicy(
        parameter_lattice_step="0.25",
        state_lattice_step="0.125",
        parameter_lattice_reason="adapter_declared_parameter_precision",
        state_lattice_reason="state_encoder_declared_precision",
    )

    result = fit_cir_candidate(
        candidate=_linear_expression_candidate(),
        data=FitDataSplit(
            train_rows=(
                _row("2026-01-01T00:00:00Z", target=1.0, lag_1=0.0),
                _row("2026-01-02T00:00:00Z", target=3.0, lag_1=1.0),
                _row("2026-01-03T00:00:00Z", target=5.0, lag_1=2.0),
            )
        ),
        fit_window_id="outer_fold_0",
        parameter_declarations=(
            ParameterDeclaration("intercept", initial_value=0.0),
            ParameterDeclaration("slope", initial_value=0.0),
        ),
        lattice_policy=policy,
    )

    assert result.replay_metadata["lattice_policy"] == policy.as_dict()
    assert result.optimizer_diagnostics["lattice_policy"] == policy.as_dict()
    assert result.as_redacted_evidence()["replay_metadata"]["lattice_policy"] == (
        policy.as_dict()
    )


def test_descriptive_coding_rejects_different_parameter_lattices() -> None:
    left = _analytic_intercept_candidate(intercept=10.0)
    right = _analytic_intercept_candidate(
        intercept=10.0,
        candidate_id="analytic_intercept_fine_parameter_lattice",
    )
    base_key = CodelengthComparisonKey(
        quantization_mode="fixed_step_mid_tread",
        quantization_step="0.5",
        reference_policy_id="raw_quantized_transformed_sequence_v1",
        data_code_family="residual_signed_integer_elias_delta_v1",
        support_kind="all_real",
        horizon_geometry=(1,),
        coding_row_set_id="rows:lattice-worker",
        residual_history_construction="none",
        parameter_lattice_step="0.5",
        state_lattice_step="0.5",
    )

    result = evaluate_descriptive_candidates(
        (left, right),
        feature_view=_feature_view((10.0, 10.0, 10.0, 10.0)),
        comparison_key_overrides={
            left.canonical_hash(): base_key,
            right.canonical_hash(): base_key.with_update(parameter_lattice_step="0.25"),
        },
    )

    assert result.accepted_candidates == ()
    assert result.description_artifacts == ()
    assert {
        diagnostic.details["comparison_failure_reason_code"]
        for diagnostic in result.admissibility_diagnostics
    } == {"parameter_lattice_step_mismatch"}


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
            )
        ),
    )


def _linear_expression_candidate():
    expression = BinaryOp(
        "add",
        BinaryOp("mul", Parameter("slope"), Feature("lag_1")),
        Parameter("intercept"),
    )
    return build_cir_candidate_from_expression(
        expression=expression,
        cir_family_id="analytic",
        cir_form_class="expression_ir",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("lag_1",),
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=CIRModelCodeDecomposition(
            L_family_bits=1.0,
            L_structure_bits=1.0,
            L_literals_bits=0.0,
            L_params_bits=2.0,
            L_state_bits=0.0,
        ),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="test-expression",
            adapter_class="unit_test",
            source_candidate_id="linear-expression",
            search_class="unit_test",
            proposal_rank=0,
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(CIRReplayHook(hook_name="test", hook_ref="unit:test"),)
        ),
    )


def _identity_update(
    state: ReducerStateObject,
    context: ReducerStateUpdateContext,
) -> ReducerStateObject:
    del context
    return state


def _row(event_time: str, *, target: float, lag_1: float) -> dict[str, object]:
    return {
        "event_time": event_time,
        "available_at": event_time,
        "target": target,
        "lag_1": lag_1,
    }
