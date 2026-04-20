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
from euclid.modules.evaluation import _additive_residual_forecast_path
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


def test_residual_component_changes_predictions() -> None:
    candidate = _candidate(
        composition_payload={
            "operator_id": "additive_residual",
            "base_reducer": "trend_component",
            "residual_reducer": "seasonal_component",
            "shared_observation_model": "point_identity",
        },
        parameter_values={
            "intercept__trend_component": 10.0,
            "intercept__seasonal_component": 0.0,
            "lag_coefficient__seasonal_component": 0.5,
        },
    )

    forecast_path = _additive_residual_forecast_path(
        candidate=candidate,
        parameters={
            "intercept__trend_component": 10.0,
            "intercept__seasonal_component": 0.0,
            "lag_coefficient__seasonal_component": 0.5,
        },
        origin_row={"target": 4.0, "residual_lag_1": 6.0},
        max_horizon=1,
    )

    horizon_trace = forecast_path.runtime_evidence["horizon_trace"][0]

    assert horizon_trace["base_prediction"] == 10.0
    assert horizon_trace["residual_prediction"] == 3.0
    assert forecast_path.predictions == {1: 13.0}


def test_additive_residual_rejects_identical_component_ids() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        parse_reducer_composition(
            {
                "operator_id": "additive_residual",
                "base_reducer": "shared_component",
                "residual_reducer": "shared_component",
                "shared_observation_model": "point_identity",
            }
        )

    assert exc_info.value.code == "invalid_additive_residual_composition"


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
            contract_id="additive_residual_history",
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
            adapter_id="additive-residual-test",
            adapter_class="test",
            source_candidate_id="additive_residual_candidate",
            search_class="exact_finite_enumeration",
            backend_family="analytic",
            proposal_rank=0,
        ),
        replay_hooks=CIRReplayHooks(),
    )
