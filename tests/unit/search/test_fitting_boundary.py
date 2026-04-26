from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.fit.multi_horizon import (
    FitStrategySpec,
    evaluate_rollout_objective,
    resolve_fit_strategy,
)
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import PredictionArtifactManifest, PredictionRow
from euclid.math.observation_models import PointObservationModel
from euclid.modules.candidate_fitting import fit_candidate_window
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.scoring import score_point_prediction_artifact
from euclid.modules.search_planning import build_canonicalization_policy, build_search_plan
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.reducers.models import BoundObservationModel
from euclid.search.backends import AnalyticSearchBackendAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_analytic_search_proposals_do_not_fit_parameters_from_targets() -> None:
    adapter = AnalyticSearchBackendAdapter()
    first_feature_view, first_audit = _feature_view((10.0, 20.0, 40.0, 80.0, 160.0))
    second_feature_view, second_audit = _feature_view((1000.0, -5.0, 12.0, 99.0, -42.0))

    first = adapter.default_proposals(
        search_plan=_search_plan(first_feature_view, first_audit),
        feature_view=first_feature_view,
    )
    second = adapter.default_proposals(
        search_plan=_search_plan(second_feature_view, second_audit),
        feature_view=second_feature_view,
    )

    first_params = {proposal.candidate_id: proposal.parameter_values for proposal in first}
    second_params = {
        proposal.candidate_id: proposal.parameter_values for proposal in second
    }
    assert first_params == second_params
    assert first_params["analytic_intercept"] == {"intercept": 13.0}
    assert first_params["analytic_lag1_affine"] == {
        "intercept": 1.0,
        "lag_coefficient": 1.0,
    }


def test_fit_strategy_spec_defaults_to_legacy_one_step() -> None:
    strategy = FitStrategySpec()

    assert strategy.strategy_id == "legacy_one_step"
    assert strategy.horizon_set == (1,)
    assert strategy.horizon_weights == ({"horizon": 1, "weight": "1"},)
    assert strategy.point_loss_id == "squared_error"
    assert (
        strategy.entity_aggregation_mode
        == "single_entity_only_no_cross_entity_aggregation"
    )


def test_fit_strategy_identity_covers_strategy_geometry() -> None:
    baseline = resolve_fit_strategy()

    assert baseline.identity_components == {
        "entity_aggregation_mode": "single_entity_only_no_cross_entity_aggregation",
        "horizon_set": [1],
        "horizon_weights": [{"horizon": 1, "weight": "1"}],
        "point_loss_id": "squared_error",
        "strategy_id": "legacy_one_step",
    }
    assert resolve_fit_strategy(strategy_id="recursive_rollout").identity_hash != (
        baseline.identity_hash
    )
    assert resolve_fit_strategy(horizon_set=(1, 2)).identity_hash != (
        baseline.identity_hash
    )
    assert resolve_fit_strategy(point_loss_id="absolute_error").identity_hash != (
        baseline.identity_hash
    )
    assert resolve_fit_strategy(
        entity_aggregation_mode="per_entity_primary_score_then_declared_entity_weights"
    ).identity_hash != baseline.identity_hash


@pytest.mark.parametrize(
    "horizon_weights",
    (
        ((1, "1"),),
        ((1, "-0.25"), (2, "1.25")),
        ((1, "0.25"), (2, "0.25")),
        ((1, "0.5"), (1, "0.5")),
        ((1, "0.5"), (3, "0.5")),
        ((1, "NaN"), (2, "1")),
    ),
)
def test_fit_strategy_rejects_invalid_horizon_weights(horizon_weights) -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        resolve_fit_strategy(
            horizon_set=(1, 2),
            horizon_weights=horizon_weights,
        )

    assert exc_info.value.code == "invalid_horizon_weight_simplex"


def test_fit_strategy_allows_non_contiguous_horizons() -> None:
    strategy = resolve_fit_strategy(
        strategy_id="recursive_rollout",
        horizon_set=(1, 3),
        horizon_weights=((1, "0.2"), (3, "0.8")),
    )

    assert strategy.horizon_set == (1, 3)
    assert strategy.horizon_weights == (
        {"horizon": 1, "weight": "0.2"},
        {"horizon": 3, "weight": "0.8"},
    )
    assert strategy.identity_components["horizon_set"] == [1, 3]


def test_rollout_objective_matches_scoring_aggregation_on_same_panel() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view(
        (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=4,
        horizon=3,
    )
    search_plan = _search_plan(feature_view, audit)
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
    strategy = resolve_fit_strategy(
        horizon_set=(1, 3),
        horizon_weights=((1, "0.25"), (3, "0.75")),
        point_loss_id="squared_error",
    )

    objective = evaluate_rollout_objective(
        candidate=fit.fitted_candidate,
        fit_result=fit,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        fit_strategy=strategy,
    )
    score_policy = _score_policy_manifest(catalog=catalog, fit_strategy=strategy)
    score = score_point_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy,
        prediction_artifact_manifest=_prediction_artifact_from_objective(
            catalog=catalog,
            score_policy=score_policy,
            objective=objective,
        ),
    )

    assert objective.training_origin_panel.status == "passed"
    assert objective.training_origin_set_id
    assert score.body["comparison_status"] == "comparable"
    assert objective.aggregated_primary_score == pytest.approx(
        score.body["aggregated_primary_score"]
    )
    assert [
        row["horizon"] for row in objective.rows
    ] == [1, 3]
    assert objective.per_horizon == tuple(
        (
            item["horizon"],
            item["valid_origin_count"],
            item["mean_point_loss"],
        )
        for item in score.body["per_horizon"]
    )


def test_legacy_one_step_strategy_preserves_fit_result_surface() -> None:
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(feature_view, audit)
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
    explicit_legacy_fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        fit_strategy=resolve_fit_strategy(horizon_set=(1,)),
    )

    assert explicit_legacy_fit.parameter_summary == default_fit.parameter_summary
    assert explicit_legacy_fit.final_state == default_fit.final_state
    assert explicit_legacy_fit.optimizer_diagnostics["fit_geometry"] == (
        default_fit.optimizer_diagnostics["fit_geometry"]
    )
    assert [record.as_dict() for record in explicit_legacy_fit.residual_history] == [
        record.as_dict() for record in default_fit.residual_history
    ]


def _search_plan(feature_view, audit):
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
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
        candidate_family_ids=("analytic_intercept", "analytic_lag1_affine"),
        search_class="exact_finite_enumeration",
        proposal_limit=2,
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


def _score_policy_manifest(*, catalog, fit_strategy) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": "rollout_objective_crosscheck_policy",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": "point",
            "point_loss_id": fit_strategy.point_loss_id,
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": list(fit_strategy.horizon_weights),
            "entity_aggregation_mode": fit_strategy.entity_aggregation_mode,
            "secondary_diagnostic_ids": [],
            "forbidden_primary_metric_ids": [],
            "lower_is_better": True,
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )


def _prediction_artifact_from_objective(
    *,
    catalog,
    score_policy: ManifestEnvelope,
    objective,
) -> ManifestEnvelope:
    return PredictionArtifactManifest(
        prediction_artifact_id="rollout_objective_prediction",
        candidate_id="rollout_objective_candidate",
        stage_id="inner_search",
        fit_window_id=objective.fit_window_id,
        test_window_id=objective.fit_window_id,
        model_freeze_status="training_rollout_objective",
        refit_rule_applied="fit_window_training_origin_panel",
        score_policy_ref=score_policy.ref,
        rows=tuple(
            PredictionRow(
                entity=row.get("entity"),
                origin_time=row["origin_time"],
                available_at=row["available_at"],
                horizon=row["horizon"],
                point_forecast=row["point_forecast"],
                realized_observation=row["realized_observation"],
            )
            for row in objective.rows
        ),
        score_law_id=objective.point_loss_id,
        horizon_weights=tuple(objective.horizon_weights),
        scored_origin_panel=tuple(
            {
                "entity": row.get("entity"),
                "origin_time": row["origin_time"],
                "available_at": row["available_at"],
                "horizon": row["horizon"],
            }
            for row in objective.rows
        ),
        scored_origin_set_id=objective.training_origin_set_id,
        comparison_key={
            "forecast_object_type": "point",
            "horizon_set": list(objective.horizon_set),
            "score_law_id": objective.point_loss_id,
            "scored_origin_set_id": objective.training_origin_set_id,
        },
    ).to_manifest(catalog)


def _feature_view(values: tuple[float, ...]):
    snapshot = FrozenDatasetSnapshot(
        series_id="search-fitting-boundary-series",
        cutoff_available_at=f"2026-01-{len(values):02d}T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-{index + 1:02d}T00:00:00Z",
                available_at=f"2026-01-{index + 1:02d}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:search-fit-{index}",
            )
            for index, value in enumerate(values)
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    return (
        materialize_feature_view(
            snapshot=snapshot,
            audit=audit,
            feature_spec=default_feature_spec(),
        ),
        audit,
    )
