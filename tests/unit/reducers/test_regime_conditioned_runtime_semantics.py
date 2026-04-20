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
from euclid.modules.evaluation import _regime_conditioned_forecast_path
from euclid.reducers.composition import resolve_regime_weights
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


def test_regime_assignment_affects_predictions() -> None:
    candidate = _candidate(
        composition_payload={
            "operator_id": "regime_conditioned",
            "gating_law": {
                "gating_law_id": "market_regime_gate",
                "selection_mode": "hard_switch",
            },
            "regime_information_contract": ["regime_flag"],
            "branch_reducers": [
                {"regime_value": "stable", "reducer_id": "stable_branch"},
                {"regime_value": "volatile", "reducer_id": "volatile_branch"},
            ],
        },
        parameter_values={
            "intercept__stable_branch": 10.0,
            "intercept__volatile_branch": 30.0,
        },
    )

    stable_path = _regime_conditioned_forecast_path(
        candidate=candidate,
        parameters={
            "intercept__stable_branch": 10.0,
            "intercept__volatile_branch": 30.0,
        },
        origin_row={"target": 12.0, "regime_flag": "stable"},
        max_horizon=1,
    )
    volatile_path = _regime_conditioned_forecast_path(
        candidate=candidate,
        parameters={
            "intercept__stable_branch": 10.0,
            "intercept__volatile_branch": 30.0,
        },
        origin_row={"target": 12.0, "regime_flag": "volatile"},
        max_horizon=1,
    )

    assert stable_path.predictions == {1: 10.0}
    assert volatile_path.predictions == {1: 30.0}
    assert stable_path.runtime_evidence["selected_branch_id"] == "stable_branch"
    assert volatile_path.runtime_evidence["selected_branch_id"] == "volatile_branch"


def test_regime_conditioned_convex_weights_are_normalized() -> None:
    candidate = _candidate(
        composition_payload={
            "operator_id": "regime_conditioned",
            "gating_law": {
                "gating_law_id": "market_regime_gate",
                "selection_mode": "convex_weighting",
            },
            "regime_information_contract": ["stable_weight", "volatile_weight"],
            "branch_reducers": [
                {"regime_value": "stable", "reducer_id": "stable_branch"},
                {"regime_value": "volatile", "reducer_id": "volatile_branch"},
            ],
        },
    )

    weights, evidence = resolve_regime_weights(
        candidate.structural_layer.composition_graph,
        row={"stable_weight": 2.0, "volatile_weight": 1.0},
    )

    assert weights == {
        "stable_branch": pytest.approx(2.0 / 3.0),
        "volatile_branch": pytest.approx(1.0 / 3.0),
    }
    assert evidence["selection_mode"] == "convex_weighting"


def test_regime_conditioned_requires_declared_runtime_information() -> None:
    candidate = _candidate(
        composition_payload={
            "operator_id": "regime_conditioned",
            "gating_law": {
                "gating_law_id": "market_regime_gate",
                "selection_mode": "hard_switch",
            },
            "regime_information_contract": ["regime_flag"],
            "branch_reducers": [
                {"regime_value": "stable", "reducer_id": "stable_branch"},
            ],
        },
    )

    with pytest.raises(ContractValidationError) as exc_info:
        resolve_regime_weights(
            candidate.structural_layer.composition_graph,
            row={"target": 1.0},
        )

    assert exc_info.value.code == "missing_regime_information"


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
            contract_id="regime_conditioned_history",
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
            adapter_id="regime-conditioned-test",
            adapter_class="test",
            source_candidate_id="regime_conditioned_candidate",
            search_class="exact_finite_enumeration",
            backend_family="analytic",
            proposal_rank=0,
        ),
        replay_hooks=CIRReplayHooks(),
    )
