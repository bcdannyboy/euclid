from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

from euclid.algorithmic_dsl import (
    canonicalize_fraction,
    evaluate_algorithmic_program,
    initialize_algorithmic_state,
)
from euclid.cir.models import (
    CandidateIntermediateRepresentation,
    CIRBackendOriginRecord,
    CIRCanonicalSerialization,
    CIREvidenceLayer,
    CIRExecutionLayer,
    CIRForecastOperator,
    CIRHistoryAccessContract,
    CIRInputSignature,
    CIRLiteral,
    CIRLiteralBlock,
    CIRModelCodeDecomposition,
    CIRReplayHook,
    CIRReplayHooks,
    CIRStateSignature,
    CIRStructuralLayer,
)
from euclid.contracts.errors import ContractValidationError
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
from euclid.runtime.hashing import canonicalize_json, sha256_digest


def normalize_cir_candidate(
    candidate: CandidateIntermediateRepresentation,
) -> CandidateIntermediateRepresentation:
    structural_layer = _normalize_structural_layer(candidate.structural_layer)
    execution_layer = _normalize_execution_layer(candidate.execution_layer)
    replay_hooks = _normalize_replay_hooks(candidate.evidence_layer.replay_hooks)
    # Canonical identity is the normalized scientific candidate plus its
    # shared model-code accounting. Provenance and replay metadata stay on the
    # evidence layer for audit/replay, but they must not split duplicate
    # candidates into distinct canonical hashes.
    canonical_payload = {
        "structural_layer": structural_layer.as_dict(),
        "execution_layer": execution_layer.as_dict(),
        "evidence_layer": {
            "model_code_decomposition": (
                candidate.evidence_layer.model_code_decomposition.as_dict()
            ),
        },
    }
    canonical_bytes = canonicalize_json(canonical_payload).encode("utf-8")
    canonical_serialization = CIRCanonicalSerialization(
        canonical_bytes=canonical_bytes,
        content_hash=sha256_digest(canonical_payload),
    )
    evidence_layer = CIREvidenceLayer(
        canonical_serialization=canonical_serialization,
        model_code_decomposition=candidate.evidence_layer.model_code_decomposition,
        backend_origin_record=candidate.evidence_layer.backend_origin_record,
        replay_hooks=replay_hooks,
        transient_diagnostics=candidate.evidence_layer.transient_diagnostics,
    )
    return CandidateIntermediateRepresentation(
        structural_layer=structural_layer,
        execution_layer=execution_layer,
        evidence_layer=evidence_layer,
    )


def build_cir_candidate_from_reducer(
    *,
    reducer: ReducerObject,
    cir_form_class: str,
    input_signature: CIRInputSignature,
    history_access_contract: CIRHistoryAccessContract,
    literal_block: CIRLiteralBlock,
    forecast_operator: CIRForecastOperator,
    model_code_decomposition: CIRModelCodeDecomposition,
    backend_origin_record: CIRBackendOriginRecord,
    replay_hooks: CIRReplayHooks,
    transient_diagnostics: Mapping[str, object] | None = None,
) -> CandidateIntermediateRepresentation:
    candidate = CandidateIntermediateRepresentation(
        structural_layer=CIRStructuralLayer(
            cir_family_id=reducer.family.family_id,
            cir_form_class=cir_form_class,
            input_signature=input_signature,
            state_signature=CIRStateSignature(
                persistent_state=reducer.state_semantics.persistent_state,
            ),
            literal_block=literal_block,
            parameter_block=reducer.fitted_parameters,
            composition_graph=reducer.composition_object,
        ),
        execution_layer=CIRExecutionLayer(
            history_access_contract=history_access_contract,
            state_update_law_id=reducer.state_semantics.update_rule.update_rule_id,
            forecast_operator=forecast_operator,
            observation_model_binding=reducer.observation_model,
        ),
        evidence_layer=CIREvidenceLayer(
            canonical_serialization=CIRCanonicalSerialization(
                canonical_bytes=b"{}",
                content_hash="sha256:placeholder",
            ),
            model_code_decomposition=model_code_decomposition,
            backend_origin_record=backend_origin_record,
            replay_hooks=replay_hooks,
            transient_diagnostics=transient_diagnostics or {},
        ),
    )
    return normalize_cir_candidate(candidate)


def build_cir_candidate_from_algorithmic_program(
    *,
    program,
    cir_form_class: str,
    input_signature: CIRInputSignature,
    observation_model: BoundObservationModel,
    forecast_operator: CIRForecastOperator,
    model_code_decomposition: CIRModelCodeDecomposition | None,
    backend_origin_record: CIRBackendOriginRecord,
    replay_hooks: CIRReplayHooks,
    transient_diagnostics: Mapping[str, object] | None = None,
) -> CandidateIntermediateRepresentation:
    initial_state = initialize_algorithmic_state(program)
    reducer = ReducerObject(
        family=ReducerFamilyId("algorithmic"),
        composition_object=ReducerCompositionObject(),
        fitted_parameters=ReducerParameterObject(),
        state_semantics=ReducerStateSemantics(
            persistent_state=ReducerStateObject(
                slots=tuple(
                    ReducerStateSlot(
                        name=f"state_{index}",
                        value=canonicalize_fraction(value),
                    )
                    for index, value in enumerate(initial_state)
                )
            ),
            update_rule=_algorithmic_update_rule(program),
        ),
        observation_model=observation_model,
        admissibility=ReducerAdmissibilityObject(
            family_membership=True,
            composition_closure=True,
            observation_model_compatibility=True,
            valid_state_semantics=True,
            codelength_comparability=True,
        ),
    )
    diagnostics = {
        "program_node_count": program.node_count,
        "allowed_observation_lags": list(program.allowed_observation_lags),
        **dict(transient_diagnostics or {}),
    }
    return build_cir_candidate_from_reducer(
        reducer=reducer,
        cir_form_class=cir_form_class,
        input_signature=input_signature,
        history_access_contract=CIRHistoryAccessContract(
            contract_id=(
                f"algorithmic_history__slots_{program.state_slot_count}"
                "__lags_"
                + "_".join(str(lag) for lag in program.allowed_observation_lags)
            ),
            access_mode=(
                "causal_current_observation"
                if max(program.allowed_observation_lags) == 0
                else "bounded_lag_window"
            ),
            max_lag=max(program.allowed_observation_lags),
            allowed_side_information=(),
        ),
        literal_block=CIRLiteralBlock(
            literals=(
                CIRLiteral(
                    name="algorithmic_dsl_id",
                    value="canonical_algorithmic_reducer_fragment",
                ),
                CIRLiteral(
                    name="algorithmic_state_slot_count",
                    value=program.state_slot_count,
                ),
                CIRLiteral(name="algorithmic_program", value=program.canonical_source),
                CIRLiteral(name="program_node_count", value=program.node_count),
                CIRLiteral(
                    name="algorithmic_allowed_observation_lags",
                    value=",".join(
                        str(lag) for lag in program.allowed_observation_lags
                    ),
                ),
            )
        ),
        forecast_operator=forecast_operator,
        model_code_decomposition=model_code_decomposition
        or CIRModelCodeDecomposition(
            L_family_bits=2.0,
            L_structure_bits=float(program.node_count),
            L_literals_bits=2.0,
            L_params_bits=0.0,
            L_state_bits=float(program.state_slot_count),
        ),
        backend_origin_record=backend_origin_record,
        replay_hooks=replay_hooks,
        transient_diagnostics=diagnostics,
    )


def rebind_cir_backend_origin(
    candidate: CandidateIntermediateRepresentation,
    *,
    backend_origin_record: CIRBackendOriginRecord,
    transient_diagnostics: Mapping[str, object] | None = None,
) -> CandidateIntermediateRepresentation:
    diagnostics = dict(candidate.evidence_layer.transient_diagnostics)
    if transient_diagnostics is not None:
        diagnostics.update(dict(transient_diagnostics))
    rebound = CandidateIntermediateRepresentation(
        structural_layer=candidate.structural_layer,
        execution_layer=candidate.execution_layer,
        evidence_layer=CIREvidenceLayer(
            canonical_serialization=CIRCanonicalSerialization(
                canonical_bytes=b"{}",
                content_hash="sha256:placeholder",
            ),
            model_code_decomposition=candidate.evidence_layer.model_code_decomposition,
            backend_origin_record=backend_origin_record,
            replay_hooks=candidate.evidence_layer.replay_hooks,
            transient_diagnostics=diagnostics,
        ),
    )
    return normalize_cir_candidate(rebound)


def require_full_cir_closure(
    candidate: CandidateIntermediateRepresentation,
    *,
    consumer: str,
) -> CandidateIntermediateRepresentation:
    normalized = normalize_cir_candidate(candidate)
    if (
        candidate.structural_layer.as_dict() != normalized.structural_layer.as_dict()
        or candidate.execution_layer.as_dict() != normalized.execution_layer.as_dict()
        or candidate.canonical_bytes() != normalized.canonical_bytes()
        or candidate.canonical_hash() != normalized.canonical_hash()
        or candidate.evidence_layer.replay_hooks.as_dict()
        != normalized.evidence_layer.replay_hooks.as_dict()
    ):
        raise ContractValidationError(
            code="cir_closure_required",
            message=(
                f"{consumer} requires fully normalized CIR structural, execution, "
                "and evidence layers"
            ),
            field_path="candidate",
            details={
                "source_candidate_id": (
                    candidate.evidence_layer.backend_origin_record.source_candidate_id
                ),
                "consumer": consumer,
            },
        )
    return normalized


def _normalize_structural_layer(layer: CIRStructuralLayer) -> CIRStructuralLayer:
    return CIRStructuralLayer(
        cir_family_id=layer.cir_family_id,
        cir_form_class=layer.cir_form_class,
        input_signature=CIRInputSignature(
            target_series=layer.input_signature.target_series,
            side_information_fields=tuple(
                sorted(layer.input_signature.side_information_fields)
            ),
        ),
        state_signature=CIRStateSignature(
            persistent_state=_normalize_state_object(
                layer.state_signature.persistent_state
            ),
        ),
        literal_block=CIRLiteralBlock(
            literals=tuple(
                sorted(
                    layer.literal_block.literals,
                    key=lambda literal: literal.name,
                )
            ),
        ),
        parameter_block=_normalize_parameter_block(layer.parameter_block),
        composition_graph=layer.composition_graph.normalize(),
    )


def _normalize_execution_layer(layer: CIRExecutionLayer) -> CIRExecutionLayer:
    return CIRExecutionLayer(
        history_access_contract=CIRHistoryAccessContract(
            contract_id=layer.history_access_contract.contract_id,
            access_mode=layer.history_access_contract.access_mode,
            max_lag=layer.history_access_contract.max_lag,
            allowed_side_information=tuple(
                sorted(layer.history_access_contract.allowed_side_information)
            ),
        ),
        state_update_law_id=layer.state_update_law_id,
        forecast_operator=layer.forecast_operator,
        observation_model_binding=BoundObservationModel(
            family=layer.observation_model_binding.family,
            forecast_type=layer.observation_model_binding.forecast_type,
            support_kind=layer.observation_model_binding.support_kind,
            compatible_point_losses=tuple(
                sorted(layer.observation_model_binding.compatible_point_losses)
            ),
        ),
    )


def _normalize_parameter_block(
    parameter_block: ReducerParameterObject,
) -> ReducerParameterObject:
    return ReducerParameterObject(
        parameters=tuple(
            ReducerParameter(name=parameter.name, value=parameter.value)
            for parameter in sorted(
                parameter_block.parameters,
                key=lambda parameter: parameter.name,
            )
        )
    )


def _normalize_state_object(state_object: ReducerStateObject) -> ReducerStateObject:
    return ReducerStateObject(
        slots=tuple(
            ReducerStateSlot(name=slot.name, value=slot.value)
            for slot in sorted(state_object.slots, key=lambda slot: slot.name)
        )
    )


def _normalize_replay_hooks(replay_hooks: CIRReplayHooks) -> CIRReplayHooks:
    return CIRReplayHooks(
        hooks=tuple(
            CIRReplayHook(hook_name=hook.hook_name, hook_ref=hook.hook_ref)
            for hook in sorted(
                replay_hooks.hooks,
                key=lambda hook: (hook.hook_name, hook.hook_ref),
            )
        )
    )


def _algorithmic_update_rule(program) -> ReducerStateUpdateRule:
    def update(
        state: ReducerStateObject,
        context: ReducerStateUpdateContext,
    ) -> ReducerStateObject:
        max_lag = max(program.allowed_observation_lags)
        history = tuple(context.history)
        observation = tuple(
            Fraction(str(float(history[-1 - lag])))
            if len(history) > lag
            else Fraction(0, 1)
            for lag in range(max_lag + 1)
        )
        state_tuple = tuple(
            Fraction(str(state.get(f"state_{index}")))
            for index in range(program.state_slot_count)
        )
        step = evaluate_algorithmic_program(
            program,
            state=state_tuple,
            observation=observation,
        )
        return ReducerStateObject(
            slots=tuple(
                ReducerStateSlot(
                    name=f"state_{index}",
                    value=canonicalize_fraction(value),
                )
                for index, value in enumerate(step.next_state)
            )
        )

    return ReducerStateUpdateRule(
        update_rule_id=f"algorithmic_program::{program.canonical_source}",
        implementation=update,
    )


__all__ = [
    "build_cir_candidate_from_algorithmic_program",
    "build_cir_candidate_from_reducer",
    "normalize_cir_candidate",
    "require_full_cir_closure",
]
