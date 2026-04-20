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
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import PredictionArtifactManifest, PredictionRow
from euclid.math.observation_models import PointObservationModel
from euclid.modules.candidate_fitting import fit_candidate_window
from euclid.modules.claims import resolve_claim_publication
from euclid.modules.evaluation import emit_point_prediction_artifact
from euclid.modules.evaluation_governance import build_baseline_registry
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.ingestion import ingest_dataframe_dataset
from euclid.modules.scoring import (
    evaluate_point_comparators,
    score_point_prediction_artifact,
)
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


def test_shared_local_publication_keeps_entity_panel_through_evaluation_scoring_and_claims(  # noqa: E501
    project_root,
) -> None:
    expected_composition_signature = (
        "sha256:e4435ba85116221364a79abb91c3050d1d848d60651880ed27501889c8e6f002"
    )
    catalog = load_contract_catalog(project_root)
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
    search_plan = build_search_plan(
        evaluation_plan=evaluation_plan,
        canonicalization_policy_ref=TypedRef(
            "canonicalization_policy_manifest@1.0.0",
            build_canonicalization_policy().canonicalization_policy_id,
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
    candidate = _shared_local_candidate()
    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.confirmatory_segment,
        search_plan=search_plan,
        stage_id="confirmatory_holdout",
    )
    score_policy = _score_policy_manifest(catalog)

    artifact = emit_point_prediction_artifact(
        catalog=catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.confirmatory_segment,
        fit_result=fit,
        score_policy_manifest=score_policy,
        stage_id="confirmatory_holdout",
    )
    score_result = score_point_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy,
        prediction_artifact_manifest=artifact,
    )
    comparator_result = evaluate_point_comparators(
        catalog=catalog,
        score_policy_manifest=score_policy,
        candidate_prediction_artifact=artifact,
        baseline_registry_manifest=build_baseline_registry(
            compatible_point_score_policy_ref=score_policy.ref
        ).to_manifest(catalog),
        comparator_prediction_artifacts={
            "constant_baseline": _baseline_artifact(
                catalog=catalog,
                score_policy=score_policy,
                entity_panel=tuple(artifact.body["entity_panel"]),
                entity_weights=tuple(
                    (item["entity"], item["weight"])
                    for item in artifact.body["entity_weights"]
                ),
                scored_origin_panel=tuple(artifact.body["scored_origin_panel"]),
                scored_origin_set_id=str(artifact.body["scored_origin_set_id"]),
            )
        },
    )
    claim = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": "passed",
            "descriptive_reason_codes": [],
            "predictive_status": "passed",
            "predictive_reason_codes": [],
            "forecast_object_type": "point",
            "entity_panel": artifact.body["entity_panel"],
        }
    )

    assert artifact.body["entity_panel"] == ["entity-a", "entity-b"]
    assert artifact.body["entity_weights"] == [
        {"entity": "entity-a", "weight": "0.5"},
        {"entity": "entity-b", "weight": "0.5"},
    ]
    assert artifact.body["rows"] == [
        {
            "entity": "entity-a",
            "origin_time": "2026-01-04T00:00:00Z",
            "available_at": "2026-01-05T06:00:00Z",
            "horizon": 1,
            "point_forecast": 14.0,
            "realized_observation": 14.0,
        },
        {
            "entity": "entity-b",
            "origin_time": "2026-01-04T00:00:00Z",
            "available_at": "2026-01-05T06:00:00Z",
            "horizon": 1,
            "point_forecast": 24.0,
            "realized_observation": 24.0,
        },
    ]
    assert artifact.body["comparison_key"] == {
        "composition_signature": expected_composition_signature,
        "forecast_object_type": "point",
        "horizon_set": [1],
        "score_law_id": "absolute_error",
        "scored_origin_set_id": artifact.body["scored_origin_set_id"],
        "entity_panel": ["entity-a", "entity-b"],
        "entity_weights": [
            {"entity": "entity-a", "weight": "0.5"},
            {"entity": "entity-b", "weight": "0.5"},
        ],
    }
    assert score_result.body["comparison_status"] == "comparable"
    assert score_result.body["aggregated_primary_score"] == 0.0
    assert comparator_result.comparison_universe.body["comparison_class_status"] == (
        "comparable"
    )
    assert (
        comparator_result.comparison_universe.body["candidate_beats_baseline"] is True
    )
    assert comparator_result.comparison_universe.body["candidate_comparison_key"] == {
        "composition_signature": expected_composition_signature,
        "forecast_object_type": "point",
        "score_policy_ref": score_policy.ref.as_dict(),
        "horizon_set": [1],
        "scored_origin_set_id": artifact.body["scored_origin_set_id"],
        "entity_panel": ["entity-a", "entity-b"],
        "entity_weights": [
            {"entity": "entity-a", "weight": "0.5"},
            {"entity": "entity-b", "weight": "0.5"},
        ],
    }
    assert claim.claim_type == "predictively_supported"
    assert claim.predictive_support_status == "confirmatory_supported"
    assert claim.allowed_interpretation_codes == (
        "historical_structure_summary",
        "point_forecast_within_declared_validation_scope",
        "cross_entity_panel_forecast_within_declared_validation_scope",
    )


def _score_policy_manifest(catalog) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": "shared_local_point_policy",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": "point",
            "point_loss_id": "absolute_error",
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [{"horizon": 1, "weight": "1.0"}],
            "entity_aggregation_mode": (
                "per_entity_primary_score_then_declared_entity_weights"
            ),
            "secondary_diagnostic_ids": [],
            "forbidden_primary_metric_ids": [],
            "lower_is_better": True,
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )


def _baseline_artifact(
    *,
    catalog,
    score_policy: ManifestEnvelope,
    entity_panel: tuple[str, ...],
    entity_weights: tuple[tuple[str, str], ...],
    scored_origin_panel: tuple[dict[str, object], ...],
    scored_origin_set_id: str,
) -> ManifestEnvelope:
    return PredictionArtifactManifest(
        prediction_artifact_id="constant_baseline_prediction",
        candidate_id="constant_baseline",
        stage_id="confirmatory_holdout",
        fit_window_id="fit_window",
        test_window_id="confirmatory_segment",
        model_freeze_status="global_finalist_frozen",
        refit_rule_applied="pre_holdout_development_refit",
        score_policy_ref=score_policy.ref,
        rows=(
            PredictionRow(
                entity="entity-a",
                origin_time="2026-01-04T00:00:00Z",
                available_at="2026-01-05T06:00:00Z",
                horizon=1,
                point_forecast=10.0,
                realized_observation=14.0,
            ),
            PredictionRow(
                entity="entity-b",
                origin_time="2026-01-04T00:00:00Z",
                available_at="2026-01-05T06:00:00Z",
                horizon=1,
                point_forecast=20.0,
                realized_observation=24.0,
            ),
        ),
        score_law_id="absolute_error",
        horizon_weights=({"horizon": 1, "weight": "1.0"},),
        entity_panel=entity_panel,
        entity_weights=tuple(
            {"entity": entity, "weight": weight} for entity, weight in entity_weights
        ),
        scored_origin_panel=scored_origin_panel,
        scored_origin_set_id=scored_origin_set_id,
        comparison_key={
            "forecast_object_type": "point",
            "horizon_set": [1],
            "score_law_id": "absolute_error",
            "scored_origin_set_id": scored_origin_set_id,
            "entity_panel": list(entity_panel),
            "entity_weights": [
                {"entity": entity, "weight": weight}
                for entity, weight in entity_weights
            ],
        },
        missing_scored_origins=(),
        timeguard_checks=(
            {
                "scored_origin_id": "constant_baseline_origin_0",
                "expected_available_at": "2026-01-05T06:00:00Z",
                "observed_available_at": "2026-01-05T06:00:00Z",
                "status": "passed",
            },
            {
                "scored_origin_id": "constant_baseline_origin_1",
                "expected_available_at": "2026-01-05T06:00:00Z",
                "observed_available_at": "2026-01-05T06:00:00Z",
                "status": "passed",
            },
        ),
    ).to_manifest(catalog)


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
