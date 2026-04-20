from __future__ import annotations

from dataclasses import replace

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
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.math.observation_models import PointObservationModel
from euclid.modules.candidate_fitting import (
    build_candidate_fit_artifacts,
    fit_candidate_window,
)
from euclid.modules.evaluation import emit_point_prediction_artifact
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
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


def test_regime_conditioned_pipeline_is_behavioral_not_structural_only(
    contract_catalog,
) -> None:
    feature_view, evaluation_plan, search_plan = _search_context(
        candidate_id="regime_conditioned_candidate",
        regime_values=("stable", "stable", "volatile", "volatile", "volatile"),
    )
    score_policy = _score_policy_manifest(contract_catalog, evaluation_plan)
    candidate = _candidate(
        candidate_id="regime_conditioned_candidate",
        composition_payload={
            "operator_id": "regime_conditioned",
            "gating_law": {
                "gating_law_id": "market_regime_gate",
                "selection_mode": "hard_switch",
            },
            "regime_information_contract": ["regime_flag"],
            "branch_reducers": [
                {"regime_value": "stable", "reducer_id": "fast_path"},
                {"regime_value": "volatile", "reducer_id": "slow_path"},
            ],
        },
        side_information_fields=("lag_1", "regime_flag"),
    )

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )
    artifacts = build_candidate_fit_artifacts(
        catalog=contract_catalog,
        fit_result=fit,
        search_plan_ref=TypedRef(
            "search_plan_manifest@1.0.0",
            search_plan.search_plan_id,
        ),
        selection_floor_bits=0.0,
    )
    prediction_artifact = emit_point_prediction_artifact(
        catalog=contract_catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.development_segments[0],
        fit_result=fit,
        score_policy_manifest=score_policy,
        stage_id="outer_test",
    )

    scored_origin = prediction_artifact.body["composition_runtime_evidence"][
        "scored_origins"
    ][0]

    assert fit.parameter_summary["intercept__fast_path"] == pytest.approx(11.5)
    assert fit.parameter_summary["intercept__slow_path"] == pytest.approx(30.0)
    assert (
        artifacts.reducer_artifact.body["optimizer_diagnostics"]["composition_runtime"][
            "branch_trace"
        ]
        == [
            {
                "row_index": 0,
                "event_time": "2026-01-02T00:00:00Z",
                "selected_branch_id": "fast_path",
                "selection_mode": "hard_switch",
            },
            {
                "row_index": 1,
                "event_time": "2026-01-03T00:00:00Z",
                "selected_branch_id": "fast_path",
                "selection_mode": "hard_switch",
            },
            {
                "row_index": 2,
                "event_time": "2026-01-04T00:00:00Z",
                "selected_branch_id": "slow_path",
                "selection_mode": "hard_switch",
            },
        ]
    )
    assert scored_origin["selected_branch_id"] == "slow_path"
    assert scored_origin["horizon_trace"][0]["point_forecast"] == pytest.approx(30.0)
    assert prediction_artifact.body["rows"][0]["point_forecast"] == pytest.approx(30.0)


def test_regime_conditioned_replay_is_exact(contract_catalog) -> None:
    feature_view, evaluation_plan, search_plan = _search_context(
        candidate_id="regime_conditioned_candidate",
        regime_values=("stable", "stable", "volatile", "volatile", "volatile"),
    )
    score_policy = _score_policy_manifest(contract_catalog, evaluation_plan)
    candidate = _candidate(
        candidate_id="regime_conditioned_candidate",
        composition_payload={
            "operator_id": "regime_conditioned",
            "gating_law": {
                "gating_law_id": "market_regime_gate",
                "selection_mode": "hard_switch",
            },
            "regime_information_contract": ["regime_flag"],
            "branch_reducers": [
                {"regime_value": "stable", "reducer_id": "fast_path"},
                {"regime_value": "volatile", "reducer_id": "slow_path"},
            ],
        },
        side_information_fields=("lag_1", "regime_flag"),
    )

    first_fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )
    second_fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )
    first_prediction = emit_point_prediction_artifact(
        catalog=contract_catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.development_segments[0],
        fit_result=first_fit,
        score_policy_manifest=score_policy,
        stage_id="outer_test",
    )
    second_prediction = emit_point_prediction_artifact(
        catalog=contract_catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.development_segments[0],
        fit_result=second_fit,
        score_policy_manifest=score_policy,
        stage_id="outer_test",
    )

    assert first_fit.parameter_summary == second_fit.parameter_summary
    assert first_prediction.body["rows"] == second_prediction.body["rows"]
    assert first_prediction.body["composition_runtime_evidence"] == (
        second_prediction.body["composition_runtime_evidence"]
    )


def _search_context(
    *,
    candidate_id: str,
    regime_values: tuple[str, ...],
):
    snapshot = FrozenDatasetSnapshot(
        series_id="regime-conditioned-pipeline-series",
        cutoff_available_at="2026-01-06T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=(
            SnapshotRow(
                event_time="2026-01-01T00:00:00Z",
                available_at="2026-01-01T00:00:00Z",
                observed_value=10.0,
                revision_id=0,
                payload_hash="sha256:a",
            ),
            SnapshotRow(
                event_time="2026-01-02T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                observed_value=11.0,
                revision_id=0,
                payload_hash="sha256:b",
            ),
            SnapshotRow(
                event_time="2026-01-03T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                observed_value=12.0,
                revision_id=0,
                payload_hash="sha256:c",
            ),
            SnapshotRow(
                event_time="2026-01-04T00:00:00Z",
                available_at="2026-01-04T00:00:00Z",
                observed_value=30.0,
                revision_id=0,
                payload_hash="sha256:d",
            ),
            SnapshotRow(
                event_time="2026-01-05T00:00:00Z",
                available_at="2026-01-05T00:00:00Z",
                observed_value=31.0,
                revision_id=0,
                payload_hash="sha256:e",
            ),
            SnapshotRow(
                event_time="2026-01-06T00:00:00Z",
                available_at="2026-01-06T00:00:00Z",
                observed_value=32.0,
                revision_id=0,
                payload_hash="sha256:f",
            ),
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )
    assert len(regime_values) == len(feature_view.rows)
    feature_view = replace(
        feature_view,
        rows=tuple(
            {
                **row,
                "regime_flag": regime_values[index],
            }
            for index, row in enumerate(feature_view.rows)
        ),
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
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
        candidate_family_ids=(candidate_id,),
        search_class="exact_finite_enumeration",
        proposal_limit=1,
    )
    return feature_view, evaluation_plan, search_plan


def _candidate(
    *,
    candidate_id: str,
    composition_payload: dict[str, object],
    side_information_fields: tuple[str, ...],
):
    reducer = ReducerObject(
        family=ReducerFamilyId("analytic"),
        composition_object=parse_reducer_composition(composition_payload),
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
            side_information_fields=side_information_fields,
        ),
        history_access_contract=CIRHistoryAccessContract(
            contract_id=f"{candidate_id}_history_contract",
            access_mode="full_prefix",
            allowed_side_information=side_information_fields,
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
            adapter_id=f"{candidate_id}_adapter",
            adapter_class="test",
            source_candidate_id=candidate_id,
            search_class="exact_finite_enumeration",
            backend_family="analytic",
            proposal_rank=0,
        ),
        replay_hooks=CIRReplayHooks(),
        transient_diagnostics={},
    )


def _score_policy_manifest(contract_catalog, evaluation_plan):
    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": "regime_conditioned_pipeline_policy_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": "point",
            "point_loss_id": "absolute_error",
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [
                weight.as_dict() for weight in evaluation_plan.horizon_weights
            ],
            "entity_aggregation_mode": (
                "single_entity_only_no_cross_entity_aggregation"
            ),
            "secondary_diagnostic_ids": [],
            "forbidden_primary_metric_ids": [],
            "lower_is_better": True,
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=contract_catalog,
    )
