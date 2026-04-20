from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from euclid.cir.models import (
    CIRBackendOriginRecord,
    CIRCanonicalSerialization,
    CIREvidenceLayer,
    CIRExecutionLayer,
    CIRForecastOperator,
    CIRHistoryAccessContract,
    CIRInputSignature,
    CIRModelCodeDecomposition,
    CIRReplayHooks,
    CIRStateSignature,
    CIRStructuralLayer,
    CandidateIntermediateRepresentation,
)
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.modules.candidate_fitting import CandidateWindowFitResult, fit_candidate_window
from euclid.modules.evaluation import emit_point_prediction_artifact
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import EvaluationPlan, build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.math.observation_models import PointObservationModel
from euclid.reducers.models import BoundObservationModel, ReducerStateObject
from euclid.search.backends import run_descriptive_search_backends
from euclid.search_planning import build_canonicalization_policy, build_search_plan

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_emit_point_prediction_artifact_rolls_multi_horizon_without_leakage() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=2,
    )
    search_plan = _search_plan(evaluation_plan)
    score_policy = _score_policy_manifest(
        catalog=catalog,
        evaluation_plan=evaluation_plan,
        point_loss_id="absolute_error",
    )
    candidate = _candidate(feature_view, search_plan, "analytic_lag1_affine")
    fit_result = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )

    artifact = emit_point_prediction_artifact(
        catalog=catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.development_segments[0],
        fit_result=fit_result,
        score_policy_manifest=score_policy,
        stage_id="outer_test",
    )

    assert [
        (row["horizon"], row["point_forecast"]) for row in artifact.body["rows"]
    ] == [
        (1, 5.0),
        (2, 6.0),
    ]
    assert [row["realized_observation"] for row in artifact.body["rows"]] == [
        100.0,
        200.0,
    ]
    assert artifact.body["missing_scored_origins"] == []


def test_emit_point_prediction_artifact_preserves_comparison_metadata() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=2,
        horizon_weight_strings=("0.25", "0.75"),
    )
    search_plan = _search_plan(evaluation_plan)
    score_policy = _score_policy_manifest(
        catalog=catalog,
        evaluation_plan=evaluation_plan,
        point_loss_id="squared_error",
    )
    candidate = _candidate(feature_view, search_plan, "recursive_level_smoother")
    fit_result = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.confirmatory_segment,
        search_plan=search_plan,
        stage_id="confirmatory_holdout",
    )

    artifact = emit_point_prediction_artifact(
        catalog=catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.confirmatory_segment,
        fit_result=fit_result,
        score_policy_manifest=score_policy,
        stage_id="confirmatory_holdout",
    )

    assert artifact.body["stage_id"] == "confirmatory_holdout"
    assert artifact.body["model_freeze_status"] == "global_finalist_frozen"
    assert artifact.body["refit_rule_applied"] == "pre_holdout_development_refit"
    assert "outer_fold_id" not in artifact.body
    assert artifact.body["horizon_weights"] == [
        {"horizon": 1, "weight": "0.25"},
        {"horizon": 2, "weight": "0.75"},
    ]
    assert [
        origin["scored_origin_id"] for origin in artifact.body["scored_origin_panel"]
    ] == list(evaluation_plan.confirmatory_segment.scored_origin_ids)
    assert artifact.body["comparison_key"] == {
        "forecast_object_type": "point",
        "horizon_set": [1, 2],
        "score_law_id": "squared_error",
        "scored_origin_set_id": artifact.body["scored_origin_set_id"],
    }
    assert all(
        check["status"] == "passed" for check in artifact.body["timeguard_checks"]
    )


def test_emit_point_prediction_artifact_records_timeguard_omissions() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=2,
    )
    search_plan = _search_plan(evaluation_plan)
    score_policy = _score_policy_manifest(
        catalog=catalog,
        evaluation_plan=evaluation_plan,
        point_loss_id="absolute_error",
    )
    candidate = _candidate(feature_view, search_plan, "analytic_lag1_affine")
    fit_result = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )
    delayed_rows = list(feature_view.rows)
    delayed_row = dict(delayed_rows[4])
    delayed_row["available_at"] = "2030-01-01T00:00:00Z"
    delayed_rows[4] = delayed_row
    delayed_feature_view = replace(feature_view, rows=tuple(delayed_rows))

    artifact = emit_point_prediction_artifact(
        catalog=catalog,
        feature_view=delayed_feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.development_segments[0],
        fit_result=fit_result,
        score_policy_manifest=score_policy,
        stage_id="outer_test",
    )

    assert [
        (row["horizon"], row["point_forecast"]) for row in artifact.body["rows"]
    ] == [(1, 5.0)]
    assert artifact.body["missing_scored_origins"] == [
        {
            "scored_origin_id": "outer_fold_0_h2",
            "horizon": 2,
            "reason_code": "non_time_safe_prediction",
            "expected_available_at": "2026-01-06T00:00:00Z",
            "observed_available_at": "2030-01-01T00:00:00Z",
        }
    ]
    assert artifact.body["timeguard_checks"] == [
        {
            "scored_origin_id": "outer_fold_0_h1",
            "expected_available_at": "2026-01-05T00:00:00Z",
            "observed_available_at": "2026-01-05T00:00:00Z",
            "status": "passed",
        },
        {
            "scored_origin_id": "outer_fold_0_h2",
            "expected_available_at": "2026-01-06T00:00:00Z",
            "observed_available_at": "2030-01-01T00:00:00Z",
            "status": "failed",
        },
    ]


def test_emit_point_prediction_artifact_requires_closed_cir_candidate() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=2,
    )
    search_plan = _search_plan(evaluation_plan)
    score_policy = _score_policy_manifest(
        catalog=catalog,
        evaluation_plan=evaluation_plan,
        point_loss_id="absolute_error",
    )
    fit_result = fit_candidate_window(
        candidate=_candidate(feature_view, search_plan, "analytic_lag1_affine"),
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )
    unclosed_fit_result = CandidateWindowFitResult(
        candidate_id=fit_result.candidate_id,
        family_id=fit_result.family_id,
        candidate_hash=fit_result.candidate_hash,
        fit_window_id=fit_result.fit_window_id,
        stage_id=fit_result.stage_id,
        training_row_count=fit_result.training_row_count,
        backend_id=fit_result.backend_id,
        objective_id=fit_result.objective_id,
        parameter_summary=fit_result.parameter_summary,
        initial_state=fit_result.initial_state,
        final_state=fit_result.final_state,
        state_transitions=fit_result.state_transitions,
        optimizer_diagnostics=fit_result.optimizer_diagnostics,
        fitted_candidate=_unnormalized_candidate(),
    )

    with pytest.raises(ContractValidationError, match="fully normalized CIR"):
        emit_point_prediction_artifact(
            catalog=catalog,
            feature_view=feature_view,
            evaluation_plan=evaluation_plan,
            evaluation_segment=evaluation_plan.development_segments[0],
            fit_result=unclosed_fit_result,
            score_policy_manifest=score_policy,
            stage_id="outer_test",
        )


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="evaluation-series",
        cutoff_available_at="2026-01-08T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=(
            SnapshotRow(
                event_time="2026-01-01T00:00:00Z",
                available_at="2026-01-01T00:00:00Z",
                observed_value=1.0,
                revision_id=0,
                payload_hash="sha256:a",
            ),
            SnapshotRow(
                event_time="2026-01-02T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                observed_value=2.0,
                revision_id=0,
                payload_hash="sha256:b",
            ),
            SnapshotRow(
                event_time="2026-01-03T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                observed_value=3.0,
                revision_id=0,
                payload_hash="sha256:c",
            ),
            SnapshotRow(
                event_time="2026-01-04T00:00:00Z",
                available_at="2026-01-04T00:00:00Z",
                observed_value=4.0,
                revision_id=0,
                payload_hash="sha256:d",
            ),
            SnapshotRow(
                event_time="2026-01-05T00:00:00Z",
                available_at="2026-01-05T00:00:00Z",
                observed_value=100.0,
                revision_id=0,
                payload_hash="sha256:e",
            ),
            SnapshotRow(
                event_time="2026-01-06T00:00:00Z",
                available_at="2026-01-06T00:00:00Z",
                observed_value=200.0,
                revision_id=0,
                payload_hash="sha256:f",
            ),
            SnapshotRow(
                event_time="2026-01-07T00:00:00Z",
                available_at="2026-01-07T00:00:00Z",
                observed_value=300.0,
                revision_id=0,
                payload_hash="sha256:g",
            ),
            SnapshotRow(
                event_time="2026-01-08T00:00:00Z",
                available_at="2026-01-08T00:00:00Z",
                observed_value=400.0,
                revision_id=0,
                payload_hash="sha256:h",
            ),
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    return materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    ), audit


def _search_plan(evaluation_plan: EvaluationPlan):
    canonicalization_policy = build_canonicalization_policy()
    return build_search_plan(
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
        candidate_family_ids=(
            "analytic_intercept",
            "analytic_lag1_affine",
            "recursive_level_smoother",
            "recursive_running_mean",
        ),
        proposal_limit=4,
        search_class="exact_finite_enumeration",
        seasonal_period=4,
        minimum_description_gain_bits=-2.0,
    )


def _candidate(feature_view, search_plan, candidate_id: str):
    search_result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )
    for candidate in search_result.accepted_candidates:
        if (
            candidate.evidence_layer.backend_origin_record.source_candidate_id
            == candidate_id
        ):
            return candidate
    raise AssertionError(f"missing candidate {candidate_id}")


def _score_policy_manifest(
    *,
    catalog,
    evaluation_plan: EvaluationPlan,
    point_loss_id: str,
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": f"test_{point_loss_id}_policy_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": "point",
            "point_loss_id": point_loss_id,
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
        catalog=catalog,
    )


def _unnormalized_candidate() -> CandidateIntermediateRepresentation:
    return CandidateIntermediateRepresentation(
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
                L_params_bits=1.0,
                L_state_bits=0.0,
            ),
            backend_origin_record=CIRBackendOriginRecord(
                adapter_id="analytic-search",
                adapter_class="bounded_grammar",
                source_candidate_id="analytic_lag1_affine",
                search_class="exact_finite_enumeration",
                proposal_rank=0,
            ),
            replay_hooks=CIRReplayHooks(),
            transient_diagnostics={},
        ),
    )
