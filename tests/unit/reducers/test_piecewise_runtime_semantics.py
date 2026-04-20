from __future__ import annotations

import pytest

from euclid.cir.models import (
    CIRBackendOriginRecord,
    CIRForecastOperator,
    CIRHistoryAccessContract,
    CIRInputSignature,
    CIRLiteralBlock,
    CIRModelCodeDecomposition,
    CIRReplayHooks,
)
from euclid.cir.normalize import build_cir_candidate_from_reducer
from euclid.contracts.errors import ContractValidationError
from euclid.math.observation_models import PointObservationModel
from euclid.modules.evaluation import _piecewise_forecast_path
from euclid.reducers.composition import select_piecewise_segment
from euclid.reducers.models import (
    BoundObservationModel,
    ReducerAdmissibilityObject,
    ReducerFamilyId,
    ReducerObject,
    ReducerParameter,
    ReducerParameterObject,
    ReducerStateObject,
    ReducerStateSemantics,
    ReducerStateUpdateRule,
    parse_reducer_composition,
)


def test_piecewise_branch_selection_is_runtime_real() -> None:
    candidate = _candidate(
        composition_payload={
            "operator_id": "piecewise",
            "ordered_partition": [
                {"start_literal": 0.0, "end_literal": 1.0, "reducer_id": "head"},
                {"start_literal": 1.0, "end_literal": 3.0, "reducer_id": "tail"},
            ],
        },
        parameter_values={
            "intercept__head": 10.0,
            "intercept__tail": 30.0,
        },
    )

    head_segment, head_evidence = select_piecewise_segment(
        candidate.structural_layer.composition_graph,
        row={"piecewise_partition_value": 0.5},
    )
    tail_segment, tail_evidence = select_piecewise_segment(
        candidate.structural_layer.composition_graph,
        row={"piecewise_partition_value": 3.0},
    )
    head_path = _piecewise_forecast_path(
        candidate=candidate,
        parameters={
            "intercept__head": 10.0,
            "intercept__tail": 30.0,
        },
        origin_row={"target": 12.0, "piecewise_partition_value": 0.5},
        max_horizon=1,
    )
    tail_path = _piecewise_forecast_path(
        candidate=candidate,
        parameters={
            "intercept__head": 10.0,
            "intercept__tail": 30.0,
        },
        origin_row={"target": 12.0, "piecewise_partition_value": 2.5},
        max_horizon=1,
    )

    assert head_segment.reducer_id == "head"
    assert head_evidence["selected_branch_id"] == "head"
    assert tail_segment.reducer_id == "tail"
    assert tail_evidence["selected_branch_id"] == "tail"
    assert head_path.predictions == {1: 10.0}
    assert tail_path.predictions == {1: 30.0}


def test_piecewise_requires_explicit_runtime_partition_values() -> None:
    candidate = _candidate(
        composition_payload={
            "operator_id": "piecewise",
            "ordered_partition": [
                {"start_literal": 0.0, "end_literal": 1.0, "reducer_id": "head"},
            ],
        },
    )

    with pytest.raises(ContractValidationError) as exc_info:
        select_piecewise_segment(
            candidate.structural_layer.composition_graph,
            row={"target": 1.0},
        )

    assert exc_info.value.code == "missing_piecewise_partition_value"


def _candidate(
    *,
    composition_payload: dict[str, object],
    parameter_values: dict[str, float] | None = None,
):
    reducer = ReducerObject(
        family=ReducerFamilyId("analytic"),
        composition_object=parse_reducer_composition(composition_payload),
        fitted_parameters=ReducerParameterObject(
            parameters=tuple(
                ReducerParameter(name=name, value=value)
                for name, value in sorted((parameter_values or {}).items())
            )
        ),
        state_semantics=ReducerStateSemantics(
            persistent_state=ReducerStateObject(),
            update_rule=ReducerStateUpdateRule(
                update_rule_id="analytic_identity_update",
                implementation=lambda state, context: state,
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
        input_signature=CIRInputSignature(target_series="target"),
        history_access_contract=CIRHistoryAccessContract(
            contract_id="piecewise_history",
            access_mode="full_prefix",
        ),
        literal_block=CIRLiteralBlock(),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=CIRModelCodeDecomposition(
            L_family_bits=1.0,
            L_structure_bits=1.0,
            L_literals_bits=0.0,
            L_params_bits=0.0,
            L_state_bits=0.0,
        ),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="piecewise-test",
            adapter_class="test",
            source_candidate_id="piecewise_candidate",
            search_class="exact_finite_enumeration",
            backend_family="analytic",
            proposal_rank=0,
        ),
        replay_hooks=CIRReplayHooks(),
    )
