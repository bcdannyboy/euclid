from __future__ import annotations

import pandas as pd

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
from euclid.contracts.refs import TypedRef
from euclid.math.observation_models import PointObservationModel
from euclid.modules.candidate_fitting import fit_candidate_window
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.ingestion import ingest_dataframe_dataset
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import freeze_dataset_snapshot
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
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


def test_fit_candidate_window_uses_panel_scoped_rows_for_shared_local_candidates() -> (
    None
):
    frame = pd.DataFrame(
        [
            {
                "entity": entity,
                "event_time": f"2026-01-{day:02d}T00:00:00Z",
                "availability_time": f"2026-01-{day:02d}T06:00:00Z",
                "target": value,
            }
            for entity, values in (
                ("entity-a", (10.0, 11.0, 12.0, 13.0, 14.0)),
                ("entity-b", (20.0, 21.0, 22.0, 23.0, 24.0)),
            )
            for day, value in enumerate(values, start=1)
        ]
    )
    dataset = ingest_dataframe_dataset(frame)
    snapshot = freeze_dataset_snapshot(
        dataset.observations,
        cutoff_available_at="2026-01-06T00:00:00Z",
    )
    audit = audit_snapshot_time_safety(snapshot)
    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=2,
        horizon=1,
    )
    canonicalization_policy = build_canonicalization_policy()
    search_plan = build_search_plan(
        evaluation_plan=evaluation_plan,
        canonicalization_policy_ref=TypedRef(
            "canonicalization_policy_manifest@1.0.0",
            canonicalization_policy.canonicalization_policy_id,
        ),
        codelength_policy_ref=TypedRef(
            "codelength_policy_manifest@1.1.0",
            "mdl_policy_default",
        ),
        reference_description_policy_ref=TypedRef(
            "reference_description_policy_manifest@1.1.0",
            "reference_description_default",
        ),
        observation_model_ref=TypedRef(
            "observation_model_manifest@1.1.0",
            "observation_model_default",
        ),
        candidate_family_ids=("analytic",),
        search_class="exact_finite_enumeration",
        proposal_limit=1,
    )

    fit = fit_candidate_window(
        candidate=_shared_local_candidate(),
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    assert "shared_plus_local_decomposition" in search_plan.composition_operators
    assert fit.backend_id == "legacy_non_claim_shared_local_panel_optimizer_v1"
    assert fit.training_row_count == 4
    assert fit.parameter_summary == {
        "shared_intercept": 1.0,
        "shared_lag_coefficient": 1.0,
        "local_adjustment__entity-a": 0.0,
        "local_adjustment__entity-b": 0.0,
        "local_lag_adjustment__entity-a": 0.0,
        "local_lag_adjustment__entity-b": 0.0,
    }
    assert fit.fitted_candidate.structural_layer.composition_graph.operator_id == (
        "shared_plus_local_decomposition"
    )
    assert fit.fitted_candidate.evidence_layer.transient_diagnostics[
        "shared_local"
    ] == {
        "entity_panel": ["entity-a", "entity-b"],
        "shared_component": {
            "component_id": "shared_component",
            "fit_rule": "panel_joint_least_squares",
            "row_count": 4,
            "parameter_summary": {
                "shared_intercept": 1.0,
                "shared_lag_coefficient": 1.0,
            },
        },
        "local_components": [
            {
                "component_id": "local_entity_a",
                "entity": "entity-a",
                "fit_rule": "panel_joint_local_effects",
                "row_count": 2,
                "parameter_summary": {
                    "local_adjustment__entity-a": 0.0,
                    "local_lag_adjustment__entity-a": 0.0,
                },
            },
            {
                "component_id": "local_entity_b",
                "entity": "entity-b",
                "fit_rule": "panel_joint_local_effects",
                "row_count": 2,
                "parameter_summary": {
                    "local_adjustment__entity-b": 0.0,
                    "local_lag_adjustment__entity-b": 0.0,
                },
            },
        ],
        "sharing_map": ["intercept"],
        "unseen_entity_rule": "panel_entities_only",
        "baseline_backend_id": "legacy_non_claim_shared_local_mean_offsets_v1",
        "selected_backend_id": "legacy_non_claim_shared_local_panel_optimizer_v1",
        "evidence_role": "legacy_non_claim_adapter",
        "claim_lane_ceiling": "descriptive_structure",
        "universal_law_evidence_allowed": False,
        "legacy_adapter_status": "legacy_non_claim_adapter",
    }


def _shared_local_candidate():
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
