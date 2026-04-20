from __future__ import annotations

import pytest

from euclid.cir.models import (
    CandidateIntermediateRepresentation,
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
from euclid.modules.shared_plus_local_decomposition import (
    fit_shared_plus_local_decomposition,
)
from euclid.reducers.models import (
    BoundObservationModel,
    ReducerAdmissibilityObject,
    ReducerFamilyId,
    ReducerObject,
    ReducerParameterObject,
    ReducerStateObject,
    ReducerStateSemantics,
    ReducerStateUpdateRule,
    parse_reducer_composition,
)


def test_general_optimizer_outperforms_baseline_when_structure_requires_it() -> None:
    summary = fit_shared_plus_local_decomposition(
        candidate=_shared_local_candidate(),
        training_rows=(
            {"entity": "entity-a", "target": 3.0, "lag_1": 1.0},
            {"entity": "entity-a", "target": 5.0, "lag_1": 2.0},
            {"entity": "entity-a", "target": 7.0, "lag_1": 3.0},
            {"entity": "entity-b", "target": 8.0, "lag_1": 1.0},
            {"entity": "entity-b", "target": 12.0, "lag_1": 2.0},
            {"entity": "entity-b", "target": 16.0, "lag_1": 3.0},
        ),
        random_seed="0",
    )

    assert summary.backend_id == "deterministic_shared_local_panel_optimizer_v1"
    assert summary.baseline_backend_id == "deterministic_shared_local_mean_offsets_v1"
    assert summary.final_loss == pytest.approx(0.0)
    assert summary.parameter_summary["shared_intercept"] == pytest.approx(1.0)
    assert summary.parameter_summary["shared_lag_coefficient"] == pytest.approx(2.0)
    assert summary.parameter_summary["local_adjustment__entity-a"] == pytest.approx(0.0)
    assert summary.parameter_summary["local_adjustment__entity-b"] == pytest.approx(3.0)
    assert summary.parameter_summary["local_lag_adjustment__entity-a"] == pytest.approx(
        0.0
    )
    assert summary.parameter_summary["local_lag_adjustment__entity-b"] == pytest.approx(
        2.0
    )


def test_selected_backend_is_persisted_in_diagnostics() -> None:
    summary = fit_shared_plus_local_decomposition(
        candidate=_shared_local_candidate(),
        training_rows=(
            {"entity": "entity-a", "target": 11.0, "lag_1": 10.0},
            {"entity": "entity-a", "target": 12.0, "lag_1": 11.0},
            {"entity": "entity-b", "target": 21.0, "lag_1": 20.0},
            {"entity": "entity-b", "target": 22.0, "lag_1": 21.0},
        ),
        random_seed="0",
    )

    assert summary.as_diagnostics()["baseline_backend_id"] == (
        "deterministic_shared_local_mean_offsets_v1"
    )
    assert summary.as_diagnostics()["selected_backend_id"] == (
        "deterministic_shared_local_panel_optimizer_v1"
    )


def test_panel_entity_rules_are_enforced() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        fit_shared_plus_local_decomposition(
            candidate=_shared_local_candidate(),
            training_rows=(
                {"entity": "entity-a", "target": 10.0, "lag_1": 9.0},
                {"entity": "entity-c", "target": 20.0, "lag_1": 19.0},
            ),
            random_seed="0",
        )

    assert exc_info.value.code == "entity_panel_mismatch"


def _shared_local_candidate() -> CandidateIntermediateRepresentation:
    reducer = ReducerObject(
        family=ReducerFamilyId("analytic"),
        composition_object=parse_reducer_composition(
            {
                "operator_id": "shared_plus_local_decomposition",
                "entity_index_set": ["entity-a", "entity-b"],
                "shared_component_ref": "shared_component",
                "local_component_refs": ["local_entity_a", "local_entity_b"],
                "sharing_map": ["intercept"],
                "unseen_entity_rule": "panel_entities_only",
            }
        ),
        fitted_parameters=ReducerParameterObject(),
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
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("lag_1",),
        ),
        history_access_contract=CIRHistoryAccessContract(
            contract_id="shared_local_panel_prefix",
            access_mode="full_prefix",
            max_lag=None,
            allowed_side_information=("lag_1",),
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
            adapter_id="shared_local_test",
            adapter_class="test",
            source_candidate_id="shared_local_candidate",
            search_class="exact_finite_enumeration",
            backend_family="analytic",
            proposal_rank=0,
        ),
        replay_hooks=CIRReplayHooks(),
        transient_diagnostics={},
    )
