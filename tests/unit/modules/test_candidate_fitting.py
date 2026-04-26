from __future__ import annotations

import math
from pathlib import Path

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
    CIRReplayHooks,
    CIRStateSignature,
    CIRStructuralLayer,
)
from euclid.cir.normalize import build_cir_candidate_from_reducer
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.fit.multi_horizon import resolve_fit_strategy
from euclid.math.observation_models import PointObservationModel
from euclid.modules.candidate_fitting import (
    build_candidate_fit_artifacts,
    fit_candidate_window,
)
from euclid.modules.features import FeatureSpec, default_feature_spec, materialize_feature_view
from euclid.modules.residual_history import residual_history_digest
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
    ReducerParameter,
    ReducerParameterObject,
    ReducerStateObject,
    ReducerStateSemantics,
    ReducerStateUpdateRule,
    parse_reducer_composition,
)
from euclid.search.backends import (
    AlgorithmicSearchBackendAdapter,
    AnalyticSearchBackendAdapter,
    RecursiveSearchBackendAdapter,
    SpectralSearchBackendAdapter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_fit_candidate_window_uses_fold_local_training_rows_for_recursive_state() -> (
    None
):
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("recursive_level_smoother",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        RecursiveSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="recursive_level_smoother",
    )

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    alpha = float(fit.parameter_summary["alpha"])
    segment = evaluation_plan.development_segments[0]
    training_targets = tuple(
        float(row["target"])
        for row in feature_view.rows[
            segment.train_start_index : segment.train_end_index + 1
        ]
    )
    level = training_targets[0]
    for observed in training_targets:
        level = (alpha * observed) + ((1.0 - alpha) * level)

    assert fit.backend_id == "deterministic_alpha_grid_v1"
    assert fit.optimizer_diagnostics["seed_value"] == "0"
    assert fit.training_row_count == 3
    assert fit.state_transition_count == 3
    assert fit.state_transitions[-1].observation_index == 2
    assert fit.final_state["level"] == pytest.approx(level)
    assert fit.final_state["step_count"] == 3


def test_fit_candidate_window_is_deterministic_for_spectral_candidates() -> None:
    feature_view, audit = _feature_view((0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=4,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("spectral_harmonic_1",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        SpectralSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="spectral_harmonic_1",
    )

    first = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )
    second = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    assert first.backend_id == "deterministic_harmonic_least_squares_v1"
    assert first.parameter_summary == second.parameter_summary
    assert first.final_state == second.final_state
    assert first.state_transitions == second.state_transitions
    assert first.fitted_candidate.canonical_hash() == (
        second.fitted_candidate.canonical_hash()
    )


def test_build_candidate_fit_artifacts_links_candidate_state_and_fit_backend() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_lag1_affine",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_lag1_affine",
    )
    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    artifacts = build_candidate_fit_artifacts(
        catalog=catalog,
        fit_result=fit,
        search_plan_ref=TypedRef(
            "search_plan_manifest@1.0.0",
            search_plan.search_plan_id,
        ),
        selection_floor_bits=0.0,
        description_gain_bits=1.25,
    )

    assert artifacts.candidate_spec.body["fit_window_id"] == "outer_fold_0"
    assert artifacts.candidate_state.body["optimizer_backend_id"] == fit.backend_id
    assert artifacts.candidate_state.body["state_transition_count"] == 3
    assert artifacts.reducer_artifact.body["candidate_state_ref"] == (
        artifacts.candidate_state.ref.as_dict()
    )
    assert artifacts.reducer_artifact.body["fit_backend_id"] == fit.backend_id


def test_analytic_fitter_emits_legal_residual_history_records() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_lag1_affine",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_lag1_affine",
    )

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )
    artifacts = build_candidate_fit_artifacts(
        catalog=catalog,
        fit_result=fit,
        search_plan_ref=TypedRef(
            "search_plan_manifest@1.0.0",
            search_plan.search_plan_id,
        ),
        selection_floor_bits=0.0,
    )

    assert fit.residual_history_validation.status == "passed"
    assert fit.residual_history_summary.residual_count == 2
    assert fit.residual_history_summary.horizon_set == (1,)
    assert fit.residual_history_summary.split_roles == ("development",)
    assert fit.optimizer_diagnostics["residual_history"]["status"] == "passed"
    assert (
        fit.optimizer_diagnostics["residual_history"]["residual_history_digest"]
        == fit.residual_history_summary.residual_history_digest
    )
    assert [record.origin_index for record in fit.residual_history] == [0, 1]
    assert [record.target_index for record in fit.residual_history] == [1, 2]
    assert all(record.origin_available_at for record in fit.residual_history)
    assert all(record.target_available_at for record in fit.residual_history)
    assert all(record.split_role == "development" for record in fit.residual_history)
    assert all(
        record.residual
        == pytest.approx(record.realized_observation - record.point_forecast)
        for record in fit.residual_history
    )
    assert artifacts.residual_history.body["residual_rows"] == [
        record.as_dict() for record in fit.residual_history
    ]


def test_analytic_residual_history_records_weights_and_weighted_summary() -> None:
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_lag1_affine",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_lag1_affine",
    )

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    assert [record.weight for record in fit.residual_history] == [1.0, 1.0]
    assert fit.residual_history_summary.weighted_residual_mean == pytest.approx(
        fit.residual_history_summary.residual_mean
    )
    assert fit.residual_history_summary.weighted_residual_rmse == pytest.approx(
        fit.residual_history_summary.residual_rmse
    )
    assert fit.optimizer_diagnostics["residual_history"][
        "weighted_residual_rmse"
    ] == pytest.approx(fit.residual_history_summary.weighted_residual_rmse)


def test_recursive_residuals_use_per_origin_state_not_final_state() -> None:
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("recursive_level_smoother",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        RecursiveSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="recursive_level_smoother",
    )

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    first_record = fit.residual_history[0]
    first_target_transition = fit.state_transitions[first_record.target_index]
    assert first_record.point_forecast == pytest.approx(
        first_target_transition.state_before["level"]
    )
    assert first_record.point_forecast != pytest.approx(fit.final_state["level"])
    assert f":o{first_record.origin_index}:h1:" in first_record.replay_identity


def test_spectral_and_algorithmic_fitters_emit_residual_histories() -> None:
    spectral_view, spectral_audit = _feature_view(
        (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0)
    )
    spectral_plan = build_evaluation_plan(
        feature_view=spectral_view,
        audit=spectral_audit,
        min_train_size=4,
        horizon=1,
    )
    spectral_search_plan = _search_plan(
        feature_view=spectral_view,
        audit=spectral_audit,
        candidate_ids=("spectral_harmonic_1",),
        seasonal_period=4,
    )
    spectral_candidate = _realize_default_candidate(
        SpectralSearchBackendAdapter(),
        search_plan=spectral_search_plan,
        feature_view=spectral_view,
        candidate_id="spectral_harmonic_1",
    )

    spectral_fit = fit_candidate_window(
        candidate=spectral_candidate,
        feature_view=spectral_view,
        fit_window=spectral_plan.development_segments[0],
        search_plan=spectral_search_plan,
    )

    algorithmic_view, algorithmic_audit = _feature_view(
        (10.0, 12.0, 13.0, 15.0, 16.0, 18.0)
    )
    algorithmic_plan = build_evaluation_plan(
        feature_view=algorithmic_view,
        audit=algorithmic_audit,
        min_train_size=3,
        horizon=1,
    )
    algorithmic_search_plan = _search_plan(
        feature_view=algorithmic_view,
        audit=algorithmic_audit,
        candidate_ids=("algorithmic_last_observation",),
        seasonal_period=4,
    )
    algorithmic_candidate = _realize_default_candidate(
        AlgorithmicSearchBackendAdapter(),
        search_plan=algorithmic_search_plan,
        feature_view=algorithmic_view,
        candidate_id="algorithmic_last_observation",
    )

    algorithmic_fit = fit_candidate_window(
        candidate=algorithmic_candidate,
        feature_view=algorithmic_view,
        fit_window=algorithmic_plan.development_segments[0],
        search_plan=algorithmic_search_plan,
    )

    assert spectral_fit.residual_history_validation.status == "passed"
    assert spectral_fit.residual_history_summary.residual_count == 3
    assert "phase_index" in spectral_fit.residual_history[0].replay_identity
    assert algorithmic_fit.residual_history_validation.status == "passed"
    assert algorithmic_fit.residual_history_summary.residual_count == 2
    assert "state_0" in algorithmic_fit.residual_history[0].replay_identity


def test_additive_residual_capture_aligns_component_predictions_by_origin() -> None:
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("additive_residual_candidate",),
        seasonal_period=4,
    )
    candidate = _analytic_composition_candidate(
        candidate_id="additive_residual_candidate",
        composition_payload={
            "operator_id": "additive_residual",
            "base_reducer": "trend_component",
            "residual_reducer": "residual_component",
            "shared_observation_model": "point_identity",
        },
        side_information_fields=("lag_1",),
    )

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    alignment = fit.optimizer_diagnostics["residual_history"]["component_alignment"]
    assert len(alignment) == len(fit.residual_history)
    for aligned, record in zip(alignment, fit.residual_history, strict=True):
        assert aligned["origin_index"] == record.origin_index
        assert aligned["target_index"] == record.target_index
        assert aligned["horizon"] == record.horizon
        assert aligned["point_forecast"] == pytest.approx(
            aligned["base_prediction"] + aligned["residual_prediction"]
        )
        assert aligned["point_forecast"] == pytest.approx(record.point_forecast)


def test_shared_local_panel_residual_history_records_entity_fields() -> None:
    feature_view, audit = _panel_feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=2,
        horizon=1,
    )
    search_plan = _search_plan_with_min_train(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("shared_local_candidate",),
        min_train_size=2,
    )

    fit = fit_candidate_window(
        candidate=_shared_local_candidate(),
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    assert fit.residual_history_summary.entity_count == 2
    assert {record.entity for record in fit.residual_history} == {
        "entity-a",
        "entity-b",
    }
    assert all(
        record.entity in record.replay_identity for record in fit.residual_history
    )


def test_late_unavailable_outlier_does_not_affect_earlier_residual_history() -> None:
    baseline_view, baseline_audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    outlier_view, outlier_audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 999.0))

    baseline_fit = _fit_first_analytic_development_window(
        feature_view=baseline_view,
        audit=baseline_audit,
    )
    outlier_fit = _fit_first_analytic_development_window(
        feature_view=outlier_view,
        audit=outlier_audit,
    )

    assert [record.as_dict() for record in baseline_fit.residual_history] == [
        record.as_dict() for record in outlier_fit.residual_history
    ]
    assert baseline_fit.residual_history_summary.residual_history_digest == (
        outlier_fit.residual_history_summary.residual_history_digest
    )
    assert all(
        record.realized_observation != 999.0 for record in outlier_fit.residual_history
    )


def test_candidate_fit_residual_history_digest_is_deterministic() -> None:
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    first = _fit_first_analytic_development_window(
        feature_view=feature_view, audit=audit
    )
    second = _fit_first_analytic_development_window(
        feature_view=feature_view,
        audit=audit,
    )

    assert first.residual_history == second.residual_history
    assert first.residual_history_summary.residual_history_digest == (
        second.residual_history_summary.residual_history_digest
    )
    assert first.residual_history_summary.residual_history_digest == (
        residual_history_digest(first.residual_history)
    )


def test_analytic_affine_fitting_labels_current_one_step_geometry_as_legacy() -> None:
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_lag1_affine",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_lag1_affine",
    )

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    assert fit.objective_id == "least_squares_one_step_residual_v1"
    assert fit.optimizer_diagnostics["fit_geometry"] == "legacy_one_step"
    assert fit.optimizer_diagnostics["fit_strategy"]["strategy_id"] == (
        "legacy_one_step"
    )
    assert fit.optimizer_diagnostics["fit_strategy"]["horizon_set"] == [1]
    assert fit.optimizer_diagnostics["fit_strategy"]["horizon_weights"] == [
        {"horizon": 1, "weight": "1"}
    ]
    assert (
        fit.optimizer_diagnostics["fit_strategy_identity"]
        == fit.optimizer_diagnostics["fit_strategy"]["identity_hash"]
    )
    assert fit.optimizer_diagnostics["training_scored_origin_set_id"] == (
        fit.optimizer_diagnostics["residual_history"]["training_scored_origin_set_id"]
    )
    assert "fit_strategy_identity" in fit.residual_history[0].replay_identity
    assert fit.optimizer_diagnostics["fit_strategy_identity"] in (
        fit.residual_history[0].replay_identity
    )


def test_explicit_legacy_fit_strategy_preserves_legacy_one_step_behavior() -> None:
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_lag1_affine",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_lag1_affine",
    )

    default_fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )
    explicit_fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        fit_strategy=resolve_fit_strategy(horizon_set=(1,)),
    )

    assert explicit_fit.parameter_summary == default_fit.parameter_summary
    assert explicit_fit.final_state == default_fit.final_state
    assert explicit_fit.state_transitions == default_fit.state_transitions
    assert [record.as_dict() for record in explicit_fit.residual_history] == [
        record.as_dict() for record in default_fit.residual_history
    ]


def test_recursive_rollout_objective_differs_from_legacy_one_step_when_weights_matter(
) -> None:
    feature_view, audit = _feature_view(
        (1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0)
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=4,
        horizon=3,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_lag1_affine",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_lag1_affine",
    )

    legacy_fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        fit_strategy=resolve_fit_strategy(horizon_set=(1,)),
    )
    rollout_fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        fit_strategy=resolve_fit_strategy(
            strategy_id="recursive_rollout",
            horizon_set=(1, 3),
            horizon_weights=((1, "0.1"), (3, "0.9")),
        ),
    )

    rollout_objective = rollout_fit.optimizer_diagnostics[
        "rollout_primary_objective"
    ]
    assert rollout_fit.objective_id == "least_squares_recursive_rollout_v1"
    assert rollout_objective["horizon_set"] == [1, 3]
    assert rollout_objective["horizon_weights"] == [
        {"horizon": 1, "weight": "0.1"},
        {"horizon": 3, "weight": "0.9"},
    ]
    assert rollout_objective["aggregated_primary_score"] != pytest.approx(
        legacy_fit.optimizer_diagnostics["final_loss"] / legacy_fit.training_row_count
    )


def test_direct_analytic_strategy_emits_horizon_specific_parameters() -> None:
    feature_view, audit = _feature_view(
        (1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0)
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=4,
        horizon=3,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_lag1_affine",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_lag1_affine",
    )

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        fit_strategy=resolve_fit_strategy(
            strategy_id="direct",
            horizon_set=(1, 3),
            horizon_weights=((1, "0.5"), (3, "0.5")),
        ),
    )

    assert fit.objective_id == "least_squares_direct_analytic_v1"
    assert "horizon_1__intercept" in fit.parameter_summary
    assert "horizon_1__lag_coefficient" in fit.parameter_summary
    assert "horizon_3__intercept" in fit.parameter_summary
    assert "horizon_3__lag_coefficient" in fit.parameter_summary
    assert "horizon_2__intercept" not in fit.parameter_summary
    assert fit.residual_history_summary.horizon_set == (1, 3)


def test_joint_analytic_strategy_changes_parameters_when_horizon_weights_change(
) -> None:
    feature_view, audit = _feature_view(
        (1.0, 1.5, 2.8, 4.9, 8.1, 12.0, 17.5, 23.0, 31.0)
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=4,
        horizon=3,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_lag1_affine",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_lag1_affine",
    )

    h1_fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        fit_strategy=resolve_fit_strategy(
            strategy_id="joint",
            horizon_set=(1, 3),
            horizon_weights=((1, "0.9"), (3, "0.1")),
        ),
    )
    h3_fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        fit_strategy=resolve_fit_strategy(
            strategy_id="joint",
            horizon_set=(1, 3),
            horizon_weights=((1, "0.1"), (3, "0.9")),
        ),
    )

    assert h1_fit.objective_id == "least_squares_joint_rollout_v1"
    assert h3_fit.objective_id == "least_squares_joint_rollout_v1"
    assert h1_fit.parameter_summary != h3_fit.parameter_summary
    assert h1_fit.optimizer_diagnostics["rollout_primary_objective"][
        "horizon_weights"
    ] == [{"horizon": 1, "weight": "0.9"}, {"horizon": 3, "weight": "0.1"}]
    assert h3_fit.optimizer_diagnostics["rollout_primary_objective"][
        "horizon_weights"
    ] == [{"horizon": 1, "weight": "0.1"}, {"horizon": 3, "weight": "0.9"}]


def test_rectify_strategy_trains_corrections_only_on_legal_training_origin_panel(
) -> None:
    feature_view, audit = _feature_view(
        (1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0)
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=4,
        horizon=3,
    )
    fit_window = evaluation_plan.development_segments[0]
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_lag1_affine",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_lag1_affine",
    )

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=fit_window,
        search_plan=search_plan,
        fit_strategy=resolve_fit_strategy(
            strategy_id="rectify",
            horizon_set=(1, 3),
            horizon_weights=((1, "0.5"), (3, "0.5")),
        ),
    )

    rectify = fit.optimizer_diagnostics["rectify"]
    correction_rows = rectify["correction_training_rows"]
    assert fit.objective_id == "least_squares_rectify_analytic_v1"
    assert correction_rows
    assert all(row["target_index"] <= fit_window.train_end_index for row in correction_rows)
    assert {row["horizon"] for row in correction_rows} == {1, 3}
    assert rectify["correction_training_origin_set_id"] == (
        fit.optimizer_diagnostics["training_scored_origin_set_id"]
    )
    assert fit.residual_history_summary.horizon_set == (1, 3)


def test_incompatible_fit_strategy_fails_closed() -> None:
    feature_view, audit = _feature_view(
        (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0)
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=5,
        horizon=2,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("spectral_harmonic_1",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        SpectralSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="spectral_harmonic_1",
    )

    with pytest.raises(ContractValidationError) as exc_info:
        fit_candidate_window(
            candidate=candidate,
            feature_view=feature_view,
            fit_window=evaluation_plan.development_segments[0],
            search_plan=search_plan,
            fit_strategy=resolve_fit_strategy(
                strategy_id="direct",
                horizon_set=(1, 2),
                horizon_weights=((1, "0.5"), (2, "0.5")),
            ),
        )

    assert exc_info.value.code == "incompatible_fit_strategy"


def test_sparse_multilag_analytic_fit_records_identity_and_residual_history() -> None:
    feature_view, audit = _feature_view_with_spec(
            (
                2.0,
                3.0,
                1.9,
                1.04,
                0.744,
                0.7384,
                0.79424,
                0.828864,
                0.8384704,
                0.83730944,
        ),
        feature_spec=FeatureSpec(
            feature_spec_id="selected_lag_feature_spec_v1",
            features=(
                {"feature_id": "lag_1", "kind": "lag", "lag_steps": 1},
                {"feature_id": "lag_2", "kind": "lag", "lag_steps": 2},
            ),
        ),
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=5,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_selected_lag_1_2_affine",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_selected_lag_1_2_affine",
    )

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    assert fit.objective_id == "least_squares_selected_feature_affine_v1"
    assert fit.optimizer_diagnostics["analytic_feature_terms"] == ["lag_1", "lag_2"]
    assert fit.parameter_summary["intercept"] == pytest.approx(0.5, abs=1e-9)
    assert fit.parameter_summary["lag_1__coefficient"] == pytest.approx(0.6, abs=1e-9)
    assert fit.parameter_summary["lag_2__coefficient"] == pytest.approx(
        -0.2,
        abs=1e-9,
    )
    assert fit.residual_history_validation.status == "passed"
    assert fit.residual_history_summary.residual_count == 4
    assert all("lag_2" in record.replay_identity for record in fit.residual_history)
    fitted_parameters = {
        parameter.name
        for parameter in fit.fitted_candidate.structural_layer.parameter_block.parameters
    }
    assert fitted_parameters >= {
        "intercept",
        "lag_1__coefficient",
        "lag_2__coefficient",
    }


def test_selected_feature_analytic_rollout_strategies_fail_closed() -> None:
    feature_view, audit = _feature_view_with_spec(
        (2.0, 3.0, 1.9, 1.04, 0.744, 0.7384, 0.79424, 0.828864, 0.8384704),
        feature_spec=FeatureSpec(
            feature_spec_id="selected_lag_feature_spec_v1",
            features=(
                {"feature_id": "lag_1", "kind": "lag", "lag_steps": 1},
                {"feature_id": "lag_2", "kind": "lag", "lag_steps": 2},
            ),
        ),
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=4,
        horizon=2,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_selected_lag_1_2_affine",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_selected_lag_1_2_affine",
    )

    with pytest.raises(ContractValidationError) as exc_info:
        fit_candidate_window(
            candidate=candidate,
            feature_view=feature_view,
            fit_window=evaluation_plan.development_segments[0],
            search_plan=search_plan,
            fit_strategy=resolve_fit_strategy(
                strategy_id="joint",
                horizon_set=(1, 2),
                horizon_weights=((1, "0.5"), (2, "0.5")),
            ),
        )

    assert exc_info.value.code == "incompatible_fit_strategy"
    assert exc_info.value.details["reason_code"] == (
        "selected_feature_analytic_rollout_not_supported"
    )


def test_seasonal_lag_trend_analytic_fit_uses_declared_legal_features() -> None:
    feature_view, audit = _feature_view_with_spec(
        (10.0, 12.0, 14.0, 16.0, 16.5, 19.5, 22.5, 25.5, 31.0, 35.0, 39.0),
        feature_spec=FeatureSpec(
            feature_spec_id="seasonal_lag_trend_feature_spec_v1",
            features=(
                {"feature_id": "lag_1", "kind": "lag", "lag_steps": 1},
                {"feature_id": "seasonal_lag", "kind": "lag", "lag_steps": 4},
                {"feature_id": "trend_anchor", "kind": "rolling_mean", "window": 2},
            ),
        ),
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=5,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_seasonal_lag_trend",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_seasonal_lag_trend",
    )

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    assert fit.objective_id == "least_squares_selected_feature_affine_v1"
    assert fit.optimizer_diagnostics["analytic_feature_terms"] == [
        "seasonal_lag",
        "trend_anchor",
    ]
    assert {
        "seasonal_lag__coefficient",
        "trend_anchor__coefficient",
    } <= set(fit.parameter_summary)
    assert fit.residual_history_validation.status == "passed"
    assert all(
        "seasonal_lag" in record.replay_identity
        for record in fit.residual_history
    )


def test_multi_harmonic_spectral_fit_records_harmonic_group_identity() -> None:
    values = tuple(
        (2.0 * math.cos((2.0 * math.pi * (index - 1)) / 6.0))
        + (0.75 * math.sin((4.0 * math.pi * (index - 1)) / 6.0))
        for index in range(12)
    )
    feature_view, audit = _feature_view_with_spec(
        values,
        feature_spec=default_feature_spec(),
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=6,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("spectral_harmonic_group_1_2",),
        seasonal_period=6,
    )
    candidate = _realize_default_candidate(
        SpectralSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="spectral_harmonic_group_1_2",
    )

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    assert fit.objective_id == "least_squares_harmonic_group_basis_v1"
    assert fit.optimizer_diagnostics["spectral_harmonic_group"] == [1, 2]
    assert fit.parameter_summary["cosine_1_coefficient"] == pytest.approx(
        2.0,
        abs=1e-9,
    )
    assert fit.parameter_summary["sine_2_coefficient"] == pytest.approx(
        0.75,
        abs=1e-9,
    )
    assert fit.residual_history_validation.status == "passed"
    assert all("harmonic_group" in record.replay_identity for record in fit.residual_history)


def test_fit_candidate_window_requires_closed_cir_before_frontier_scoring() -> None:
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_intercept",),
        seasonal_period=4,
    )

    with pytest.raises(ContractValidationError, match="fully normalized CIR"):
        fit_candidate_window(
            candidate=_unnormalized_candidate(),
            feature_view=feature_view,
            fit_window=evaluation_plan.development_segments[0],
            search_plan=search_plan,
        )


def _search_plan(
    *,
    feature_view,
    audit,
    candidate_ids: tuple[str, ...],
    seasonal_period: int,
):
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3 if len(feature_view.rows) <= 6 else 4,
        horizon=1,
    )
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
        candidate_family_ids=candidate_ids,
        search_class="exact_finite_enumeration",
        proposal_limit=len(candidate_ids),
        seasonal_period=seasonal_period,
    )


def _search_plan_with_min_train(
    *,
    feature_view,
    audit,
    candidate_ids: tuple[str, ...],
    min_train_size: int,
):
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=min_train_size,
        horizon=1,
    )
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
        candidate_family_ids=candidate_ids,
        search_class="exact_finite_enumeration",
        proposal_limit=len(candidate_ids),
        seasonal_period=4,
    )


def _realize_default_candidate(
    adapter,
    *,
    search_plan,
    feature_view,
    candidate_id: str,
):
    observation_model = BoundObservationModel.from_runtime(PointObservationModel())
    for rank, proposal in enumerate(
        adapter.default_proposals(
            search_plan=search_plan,
            feature_view=feature_view,
        )
    ):
        if proposal.candidate_id != candidate_id:
            continue
        return adapter.realize_proposal(
            proposal=proposal,
            proposal_rank=rank,
            search_plan=search_plan,
            feature_view=feature_view,
            observation_model=observation_model,
        )
    raise AssertionError(f"missing candidate proposal: {candidate_id}")


def _feature_view(values: tuple[float, ...]):
    snapshot = FrozenDatasetSnapshot(
        series_id="candidate-fitting-series",
        cutoff_available_at=f"2026-01-0{len(values)}T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-{index + 1:02d}T00:00:00Z",
                available_at=f"2026-01-{index + 1:02d}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:{index}",
            )
            for index, value in enumerate(values)
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )
    return feature_view, audit


def _feature_view_with_spec(values: tuple[float, ...], *, feature_spec: FeatureSpec):
    snapshot = FrozenDatasetSnapshot(
        series_id="candidate-fitting-series",
        cutoff_available_at=f"2026-01-{len(values):02d}T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-{index + 1:02d}T00:00:00Z",
                available_at=f"2026-01-{index + 1:02d}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:custom-{index}",
            )
            for index, value in enumerate(values)
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=feature_spec,
    )
    return feature_view, audit


def _panel_feature_view():
    rows: list[SnapshotRow] = []
    for entity, offset in (("entity-a", 0.0), ("entity-b", 10.0)):
        for index, value in enumerate((10.0, 12.0, 13.0, 15.0, 16.0)):
            rows.append(
                SnapshotRow(
                    entity=entity,
                    event_time=f"2026-01-{index + 1:02d}T00:00:00Z",
                    available_at=f"2026-01-{index + 1:02d}T00:00:00Z",
                    observed_value=value + offset,
                    revision_id=0,
                    payload_hash=f"sha256:{entity}-{index}",
                )
            )
    snapshot = FrozenDatasetSnapshot(
        series_id="candidate-fitting-panel-series",
        cutoff_available_at="2026-01-05T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(rows),
    )
    audit = audit_snapshot_time_safety(snapshot)
    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )
    return feature_view, audit


def _fit_first_analytic_development_window(*, feature_view, audit):
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_lag1_affine",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_lag1_affine",
    )
    return fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )


def _analytic_composition_candidate(
    *,
    candidate_id: str,
    composition_payload: dict[str, object],
    side_information_fields: tuple[str, ...],
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


def _shared_local_candidate() -> CandidateIntermediateRepresentation:
    return _analytic_composition_candidate(
        candidate_id="shared_local_candidate",
        composition_payload={
            "operator_id": "shared_plus_local_decomposition",
            "entity_index_set": ["entity-a", "entity-b"],
            "shared_component_ref": "shared_component",
            "local_component_refs": ["local_entity_a", "local_entity_b"],
            "sharing_map": ["intercept"],
            "unseen_entity_rule": "panel_entities_only",
        },
        side_information_fields=("lag_1",),
    )


def _unnormalized_candidate() -> CandidateIntermediateRepresentation:
    return CandidateIntermediateRepresentation(
        structural_layer=CIRStructuralLayer(
            cir_family_id="analytic",
            cir_form_class="closed_form_expression",
            input_signature=CIRInputSignature(target_series="target"),
            state_signature=CIRStateSignature(persistent_state=ReducerStateObject()),
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
                source_candidate_id="analytic_intercept",
                search_class="exact_finite_enumeration",
                proposal_rank=0,
            ),
            replay_hooks=CIRReplayHooks(),
            transient_diagnostics={},
        ),
    )
