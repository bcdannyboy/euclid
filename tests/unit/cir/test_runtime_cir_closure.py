from __future__ import annotations

import pytest

from euclid.cir.models import (
    CandidateIntermediateRepresentation,
    CIRBackendOriginRecord,
    CIRCanonicalSerialization,
    CIREvidenceLayer,
    CIRExecutionLayer,
    CIRForecastOperator,
    CIRHistoryAccessContract,
    CIRInputSignature,
    CIRLiteralBlock,
    CIRModelCodeDecomposition,
    CIRReplayHook,
    CIRReplayHooks,
    CIRStateSignature,
    CIRStructuralLayer,
)
from euclid.cir.normalize import (
    build_cir_candidate_from_reducer,
    require_full_cir_closure,
)
from euclid.contracts.errors import ContractValidationError
from euclid.math.observation_models import PointObservationModel
from euclid.reducers.models import (
    BoundObservationModel,
    ReducerAdmissibilityObject,
    ReducerCompositionObject,
    ReducerFamilyId,
    ReducerObject,
    ReducerParameterObject,
    ReducerStateObject,
    ReducerStateSemantics,
    ReducerStateUpdateRule,
)


def test_candidate_without_full_cir_layers_is_rejected() -> None:
    candidate = CandidateIntermediateRepresentation(
        structural_layer=CIRStructuralLayer(
            cir_family_id="analytic",
            cir_form_class="closed_form_expression",
            input_signature=CIRInputSignature(target_series="target"),
            state_signature=CIRStateSignature(
                persistent_state=ReducerStateObject()
            ),
        ),
        execution_layer=CIRExecutionLayer(
            history_access_contract=CIRHistoryAccessContract(
                contract_id="full_prefix",
                access_mode="full_prefix",
            ),
            state_update_law_id="analytic_identity_update",
            forecast_operator=CIRForecastOperator(
                operator_id="one_step_point_forecast",
                horizon=1,
            ),
            observation_model_binding=BoundObservationModel.from_runtime(
                PointObservationModel()
            ),
        ),
        evidence_layer=CIREvidenceLayer(
            canonical_serialization=CIRCanonicalSerialization(
                canonical_bytes=b"{}",
                content_hash="sha256:placeholder",
            ),
            model_code_decomposition=CIRModelCodeDecomposition(
                L_family_bits=1.0,
                L_structure_bits=1.0,
                L_literals_bits=0.0,
                L_params_bits=0.0,
                L_state_bits=0.0,
            ),
            backend_origin_record=CIRBackendOriginRecord(
                adapter_id="analytic-search",
                adapter_class="test",
                source_candidate_id="analytic_intercept",
                search_class="exact_finite_enumeration",
            ),
            replay_hooks=CIRReplayHooks(
                hooks=(CIRReplayHook(hook_name="search_seed", hook_ref="seed:0"),)
            ),
        ),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        require_full_cir_closure(candidate, consumer="test.runtime_cir_closure")

    assert exc_info.value.code == "cir_closure_required"


def test_search_class_and_object_type_survive_cir_normalization() -> None:
    reducer = ReducerObject(
        family=ReducerFamilyId("analytic"),
        composition_object=ReducerCompositionObject(),
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
    candidate = build_cir_candidate_from_reducer(
        reducer=reducer,
        cir_form_class="closed_form_expression",
        input_signature=CIRInputSignature(target_series="target"),
        history_access_contract=CIRHistoryAccessContract(
            contract_id="full_prefix",
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
            adapter_id="analytic-search",
            adapter_class="test",
            source_candidate_id="analytic_intercept",
            search_class="stochastic_heuristic",
            backend_family="analytic",
            proposal_rank=0,
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(CIRReplayHook(hook_name="search_seed", hook_ref="seed:17"),)
        ),
    )

    closed = require_full_cir_closure(candidate, consumer="test.runtime_cir_closure")

    assert closed.evidence_layer.backend_origin_record.search_class == (
        "stochastic_heuristic"
    )
    assert closed.execution_layer.observation_model_binding.forecast_type == "point"
