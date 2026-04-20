from __future__ import annotations

from euclid.cir.models import (
    CIRBackendOriginRecord,
    CIRForecastOperator,
    CIRHistoryAccessContract,
    CIRInputSignature,
    CIRLiteral,
    CIRLiteralBlock,
    CIRModelCodeDecomposition,
    CIRReplayHook,
    CIRReplayHooks,
)
from euclid.cir.normalize import build_cir_candidate_from_reducer
from euclid.math.observation_models import PointObservationModel
from euclid.reducers.models import (
    BoundObservationModel,
    ReducerAdmissibilityObject,
    ReducerFamilyId,
    ReducerObject,
    ReducerParameter,
    ReducerParameterObject,
    ReducerStateObject,
    ReducerStateSemantics,
    ReducerStateSlot,
    ReducerStateUpdateContext,
    ReducerStateUpdateRule,
    parse_reducer_composition,
)


def test_equivalent_cir_candidates_share_canonical_bytes_after_normalization() -> None:
    left = build_cir_candidate_from_reducer(
        reducer=_piecewise_reducer(
            parameter_names=("beta", "alpha"),
            state_slot_names=("step_count", "running_total"),
            ordered_partition=(
                {"start_literal": 2.0, "end_literal": 3.0, "reducer_id": "tail"},
                {"start_literal": 0.0, "end_literal": 1.0, "reducer_id": "head"},
                {"start_literal": 1.0, "end_literal": 2.0, "reducer_id": "tail"},
            ),
        ),
        cir_form_class="piecewise_closed_form",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("volume", "regime"),
        ),
        history_access_contract=CIRHistoryAccessContract(
            contract_id="full_prefix",
            access_mode="full_prefix",
            allowed_side_information=("volume", "regime"),
        ),
        literal_block=CIRLiteralBlock(
            literals=(
                CIRLiteral(name="upper_cut", value=3.0),
                CIRLiteral(name="lower_cut", value=1.0),
            )
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=_model_code_decomposition(),
        backend_origin_record=_backend_origin_record(),
        replay_hooks=CIRReplayHooks(
            hooks=(
                CIRReplayHook(hook_name="budget_record", hook_ref="budget:phase04"),
                CIRReplayHook(hook_name="search_seed", hook_ref="seed:17"),
            )
        ),
    )
    right = build_cir_candidate_from_reducer(
        reducer=_piecewise_reducer(
            parameter_names=("alpha", "beta"),
            state_slot_names=("running_total", "step_count"),
            ordered_partition=(
                {"start_literal": 0.0, "end_literal": 1.0, "reducer_id": "head"},
                {"start_literal": 1.0, "end_literal": 3.0, "reducer_id": "tail"},
            ),
        ),
        cir_form_class="piecewise_closed_form",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("regime", "volume"),
        ),
        history_access_contract=CIRHistoryAccessContract(
            contract_id="full_prefix",
            access_mode="full_prefix",
            allowed_side_information=("regime", "volume"),
        ),
        literal_block=CIRLiteralBlock(
            literals=(
                CIRLiteral(name="lower_cut", value=1.0),
                CIRLiteral(name="upper_cut", value=3.0),
            )
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=_model_code_decomposition(),
        backend_origin_record=_backend_origin_record(),
        replay_hooks=CIRReplayHooks(
            hooks=(
                CIRReplayHook(hook_name="search_seed", hook_ref="seed:17"),
                CIRReplayHook(hook_name="budget_record", hook_ref="budget:phase04"),
            )
        ),
    )

    assert left.evidence_layer.canonical_serialization.canonical_bytes == (
        right.evidence_layer.canonical_serialization.canonical_bytes
    )
    assert left.evidence_layer.canonical_serialization.content_hash == (
        right.evidence_layer.canonical_serialization.content_hash
    )


def test_transient_diagnostics_do_not_change_canonical_bytes() -> None:
    left = build_cir_candidate_from_reducer(
        reducer=_piecewise_reducer(
            parameter_names=("alpha",),
            state_slot_names=("running_total",),
            ordered_partition=(
                {"start_literal": 0.0, "end_literal": 1.0, "reducer_id": "head"},
            ),
        ),
        cir_form_class="piecewise_closed_form",
        input_signature=CIRInputSignature(target_series="target"),
        history_access_contract=CIRHistoryAccessContract(
            contract_id="full_prefix",
            access_mode="full_prefix",
        ),
        literal_block=CIRLiteralBlock(literals=(CIRLiteral(name="cut", value=1.0),)),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=_model_code_decomposition(),
        backend_origin_record=_backend_origin_record(),
        replay_hooks=CIRReplayHooks(
            hooks=(CIRReplayHook(hook_name="search_seed", hook_ref="seed:17"),)
        ),
        transient_diagnostics={"optimizer_trace": {"best_loss": 0.5}},
    )
    right = build_cir_candidate_from_reducer(
        reducer=_piecewise_reducer(
            parameter_names=("alpha",),
            state_slot_names=("running_total",),
            ordered_partition=(
                {"start_literal": 0.0, "end_literal": 1.0, "reducer_id": "head"},
            ),
        ),
        cir_form_class="piecewise_closed_form",
        input_signature=CIRInputSignature(target_series="target"),
        history_access_contract=CIRHistoryAccessContract(
            contract_id="full_prefix",
            access_mode="full_prefix",
        ),
        literal_block=CIRLiteralBlock(literals=(CIRLiteral(name="cut", value=1.0),)),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=_model_code_decomposition(),
        backend_origin_record=_backend_origin_record(),
        replay_hooks=CIRReplayHooks(
            hooks=(CIRReplayHook(hook_name="search_seed", hook_ref="seed:17"),)
        ),
        transient_diagnostics={"optimizer_trace": {"best_loss": 10.0}},
    )

    assert left.evidence_layer.canonical_serialization.canonical_bytes == (
        right.evidence_layer.canonical_serialization.canonical_bytes
    )
    assert (
        b"optimizer_trace"
        not in left.evidence_layer.canonical_serialization.canonical_bytes
    )
    assert (
        left.evidence_layer.transient_diagnostics["optimizer_trace"]["best_loss"] == 0.5
    )
    assert (
        right.evidence_layer.transient_diagnostics["optimizer_trace"]["best_loss"]
        == 10.0
    )


def test_provenance_fields_do_not_change_canonical_bytes() -> None:
    reducer = _piecewise_reducer(
        parameter_names=("alpha", "beta"),
        state_slot_names=("running_total", "step_count"),
        ordered_partition=(
            {"start_literal": 0.0, "end_literal": 1.0, "reducer_id": "head"},
            {"start_literal": 1.0, "end_literal": 3.0, "reducer_id": "tail"},
        ),
    )
    left = build_cir_candidate_from_reducer(
        reducer=reducer,
        cir_form_class="piecewise_closed_form",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("volume", "regime"),
        ),
        history_access_contract=CIRHistoryAccessContract(
            contract_id="full_prefix",
            access_mode="full_prefix",
            allowed_side_information=("volume", "regime"),
        ),
        literal_block=CIRLiteralBlock(
            literals=(
                CIRLiteral(name="upper_cut", value=3.0),
                CIRLiteral(name="lower_cut", value=1.0),
            )
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=_model_code_decomposition(),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="analytic-search",
            adapter_class="bounded_grammar",
            source_candidate_id="candidate-1",
            search_class="exact_finite_enumeration",
            proposal_rank=0,
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(
                CIRReplayHook(hook_name="budget_record", hook_ref="budget:phase04:a"),
                CIRReplayHook(hook_name="search_seed", hook_ref="seed:17"),
            )
        ),
    )
    right = build_cir_candidate_from_reducer(
        reducer=reducer,
        cir_form_class="piecewise_closed_form",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("regime", "volume"),
        ),
        history_access_contract=CIRHistoryAccessContract(
            contract_id="full_prefix",
            access_mode="full_prefix",
            allowed_side_information=("regime", "volume"),
        ),
        literal_block=CIRLiteralBlock(
            literals=(
                CIRLiteral(name="lower_cut", value=1.0),
                CIRLiteral(name="upper_cut", value=3.0),
            )
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=_model_code_decomposition(),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="recursive-search",
            adapter_class="bounded_grammar",
            source_candidate_id="candidate-2",
            search_class="bounded_heuristic",
            proposal_rank=7,
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(
                CIRReplayHook(hook_name="search_seed", hook_ref="seed:999"),
                CIRReplayHook(hook_name="budget_record", hook_ref="budget:phase04:b"),
            )
        ),
    )

    assert left.canonical_bytes() == right.canonical_bytes()
    assert left.canonical_hash() == right.canonical_hash()
    assert (
        left.evidence_layer.backend_origin_record.source_candidate_id
        == "candidate-1"
    )
    assert (
        right.evidence_layer.backend_origin_record.source_candidate_id
        == "candidate-2"
    )


def _piecewise_reducer(
    *,
    parameter_names: tuple[str, ...],
    state_slot_names: tuple[str, ...],
    ordered_partition: tuple[dict[str, float | str], ...],
) -> ReducerObject:
    parameter_values = {
        "alpha": 0.75,
        "beta": 0.2,
    }
    state_values = {
        "running_total": 0.0,
        "step_count": 1.0,
    }
    return ReducerObject(
        family=ReducerFamilyId("analytic"),
        composition_object=parse_reducer_composition(
            {
                "operator_id": "piecewise",
                "ordered_partition": list(ordered_partition),
            }
        ),
        fitted_parameters=ReducerParameterObject(
            parameters=tuple(
                ReducerParameter(name=name, value=parameter_values[name])
                for name in parameter_names
            )
        ),
        state_semantics=ReducerStateSemantics(
            persistent_state=ReducerStateObject(
                slots=tuple(
                    ReducerStateSlot(name=name, value=state_values[name])
                    for name in state_slot_names
                )
            ),
            update_rule=ReducerStateUpdateRule(
                update_rule_id="identity_update",
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


def _identity_update(
    state: ReducerStateObject,
    context: ReducerStateUpdateContext,
) -> ReducerStateObject:
    return state


def _model_code_decomposition() -> CIRModelCodeDecomposition:
    return CIRModelCodeDecomposition(
        L_family_bits=1.0,
        L_structure_bits=2.0,
        L_literals_bits=1.0,
        L_params_bits=3.0,
        L_state_bits=0.5,
    )


def _backend_origin_record() -> CIRBackendOriginRecord:
    return CIRBackendOriginRecord(
        adapter_id="analytic-search",
        adapter_class="bounded_grammar",
        source_candidate_id="candidate-42",
        search_class="exact_finite_enumeration",
    )
