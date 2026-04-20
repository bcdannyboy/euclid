from __future__ import annotations

import pytest

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
from euclid.contracts.errors import ContractValidationError
from euclid.math.observation_models import PointObservationModel
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
    ReducerStateSlot,
    ReducerStateUpdateContext,
    ReducerStateUpdateRule,
)


def test_build_cir_candidate_from_reducer_materializes_required_layers() -> None:
    reducer = _sample_reducer()

    candidate = build_cir_candidate_from_reducer(
        reducer=reducer,
        cir_form_class="state_recurrence",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("volume", "regime"),
        ),
        history_access_contract=CIRHistoryAccessContract(
            contract_id="bounded_prefix",
            access_mode="bounded_lag_window",
            max_lag=3,
            allowed_side_information=("regime",),
        ),
        literal_block=CIRLiteralBlock(
            literals=(
                CIRLiteral(name="season_length", value=7),
                CIRLiteral(name="smoothing_floor", value=0.1),
            )
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=CIRModelCodeDecomposition(
            L_family_bits=1.0,
            L_structure_bits=2.5,
            L_literals_bits=1.5,
            L_params_bits=3.0,
            L_state_bits=0.5,
        ),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="recursive-search",
            adapter_class="bounded_grammar",
            source_candidate_id="candidate-17",
            search_class="bounded_heuristic",
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(
                CIRReplayHook(
                    hook_name="budget_record",
                    hook_ref="budget:phase04:heuristic",
                ),
                CIRReplayHook(
                    hook_name="search_seed",
                    hook_ref="seed:17",
                ),
            )
        ),
    )

    assert candidate.structural_layer.cir_family_id == "recursive"
    assert candidate.structural_layer.cir_form_class == "state_recurrence"
    assert candidate.execution_layer.state_update_law_id == (
        "running_total_accumulator"
    )
    assert (
        candidate.execution_layer.observation_model_binding.family
        == "gaussian_location_scale"
    )
    assert (
        candidate.structural_layer.state_signature.persistent_state.get("running_total")
        == 0.0
    )
    assert candidate.evidence_layer.canonical_serialization.content_hash.startswith(
        "sha256:"
    )
    assert candidate.evidence_layer.model_code_decomposition.L_structure_bits == 2.5


def test_history_access_contract_rejects_negative_max_lag() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        CIRHistoryAccessContract(
            contract_id="bounded_prefix",
            access_mode="bounded_lag_window",
            max_lag=-1,
        )

    assert exc_info.value.code == "invalid_history_access_contract"


def _sample_reducer() -> ReducerObject:
    return ReducerObject(
        family=ReducerFamilyId("recursive"),
        composition_object=ReducerCompositionObject(),
        fitted_parameters=ReducerParameterObject(
            parameters=(
                ReducerParameter(name="beta", value=0.2),
                ReducerParameter(name="alpha", value=0.75),
            )
        ),
        state_semantics=ReducerStateSemantics(
            persistent_state=ReducerStateObject(
                slots=(
                    ReducerStateSlot(name="step_count", value=0),
                    ReducerStateSlot(name="running_total", value=0.0),
                )
            ),
            update_rule=ReducerStateUpdateRule(
                update_rule_id="running_total_accumulator",
                implementation=_running_total_update,
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


def _running_total_update(
    state: ReducerStateObject,
    context: ReducerStateUpdateContext,
) -> ReducerStateObject:
    running_total = float(state.get("running_total"))
    updated_total = running_total + context.history[-1]
    step_count = int(state.get("step_count")) + 1
    return ReducerStateObject(
        slots=(
            ReducerStateSlot(name="running_total", value=updated_total),
            ReducerStateSlot(name="step_count", value=step_count),
        )
    )
