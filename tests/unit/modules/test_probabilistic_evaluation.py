from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.math.observation_models import PointObservationModel
from euclid.modules.candidate_fitting import (
    build_candidate_fit_artifacts,
    fit_candidate_window,
)
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.probabilistic_evaluation import (
    emit_probabilistic_prediction_artifact,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import EvaluationPlan, build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.reducers.models import BoundObservationModel
from euclid.search.backends import (
    AlgorithmicSearchBackendAdapter,
    AnalyticSearchBackendAdapter,
    RecursiveSearchBackendAdapter,
    SpectralSearchBackendAdapter,
    run_descriptive_search_backends,
)
from euclid.search_planning import build_canonicalization_policy, build_search_plan

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_SCORE_POLICY_SCHEMAS = {
    "distribution": "probabilistic_score_policy_manifest@1.0.0",
    "interval": "interval_score_policy_manifest@1.0.0",
    "quantile": "quantile_score_policy_manifest@1.0.0",
    "event_probability": "event_probability_score_policy_manifest@1.0.0",
}


@pytest.mark.parametrize(
    ("candidate_id", "forecast_object_type", "required_fields"),
    (
        (
            "analytic_lag1_affine",
            "distribution",
            ("distribution_family", "location", "scale", "support_kind"),
        ),
        (
            "recursive_level_smoother",
            "interval",
            ("nominal_coverage", "lower_bound", "upper_bound"),
        ),
        (
            "spectral_harmonic_1",
            "quantile",
            ("quantiles",),
        ),
        (
            "algorithmic_last_observation",
            "event_probability",
            ("event_definition", "event_probability", "realized_event"),
        ),
    ),
)
def test_emit_probabilistic_prediction_artifact_preserves_forecast_object_typing(
    candidate_id: str,
    forecast_object_type: str,
    required_fields: tuple[str, ...],
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view(candidate_id)
    min_train_size = 4 if candidate_id.startswith("spectral_") else 3
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=min_train_size,
        horizon=2,
    )
    search_plan = _search_plan(evaluation_plan)
    score_policy = _probabilistic_score_policy_manifest(
        catalog=catalog,
        evaluation_plan=evaluation_plan,
        forecast_object_type=forecast_object_type,
    )
    candidate = _candidate(feature_view, search_plan, candidate_id)
    fit_result = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )

    artifact = emit_probabilistic_prediction_artifact(
        catalog=catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.development_segments[0],
        fit_result=fit_result,
        score_policy_manifest=score_policy,
        stage_id="outer_test",
        forecast_object_type=forecast_object_type,
    )

    assert artifact.body["forecast_object_type"] == forecast_object_type
    assert artifact.body["comparison_key"] == {
        "forecast_object_type": forecast_object_type,
        "horizon_set": [1, 2],
        "score_law_id": artifact.body["score_law_id"],
        "scored_origin_set_id": artifact.body["scored_origin_set_id"],
    }
    assert (
        artifact.body["score_policy_ref"]["schema_name"]
        == _SCORE_POLICY_SCHEMAS[forecast_object_type]
    )
    assert artifact.body["missing_scored_origins"] == []
    assert all(
        check["status"] == "passed" for check in artifact.body["timeguard_checks"]
    )
    assert len(artifact.body["rows"]) == 2
    for row in artifact.body["rows"]:
        assert row["horizon"] in {1, 2}
        assert "realized_observation" in row
        assert "point_forecast" not in row
        for field_name in required_fields:
            assert field_name in row
        if forecast_object_type == "event_probability":
            assert (
                row["event_definition"]["threshold_source"]
                == "declared_literal"
            )
            assert row["event_definition"]["event_id"] != "target_ge_origin_target"


def test_emit_probabilistic_prediction_artifact_rejects_wrong_policy_type() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view("analytic_lag1_affine")
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=2,
    )
    search_plan = _search_plan(evaluation_plan)
    interval_policy = _probabilistic_score_policy_manifest(
        catalog=catalog,
        evaluation_plan=evaluation_plan,
        forecast_object_type="interval",
    )
    candidate = _candidate(feature_view, search_plan, "analytic_lag1_affine")
    fit_result = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )

    with pytest.raises(ContractValidationError) as excinfo:
        emit_probabilistic_prediction_artifact(
            catalog=catalog,
            feature_view=feature_view,
            evaluation_plan=evaluation_plan,
            evaluation_segment=evaluation_plan.development_segments[0],
            fit_result=fit_result,
            score_policy_manifest=interval_policy,
            stage_id="outer_test",
            forecast_object_type="distribution",
        )

    assert excinfo.value.code == "probabilistic_policy_forecast_object_type_mismatch"


def test_gaussian_proxy_support_is_labeled_compatibility_only() -> None:
    artifact = _probabilistic_artifact(
        candidate_id="analytic_lag1_affine",
        forecast_object_type="distribution",
    )

    assert artifact.body["stochastic_support_status"] == "compatibility"
    assert artifact.body["stochastic_support_reason_codes"] == [
        "heuristic_gaussian_support_not_production"
    ]
    assert artifact.body["residual_history_refs"] == []
    assert artifact.body["stochastic_model_refs"] == []
    assert {
        row["distribution_family"] for row in artifact.body["rows"]
    } == {"gaussian_location_scale"}


def test_production_probabilistic_artifacts_require_stochastic_model_refs() -> None:
    context = _probabilistic_context(
        candidate_id="analytic_lag1_affine",
        forecast_object_type="distribution",
    )

    with pytest.raises(ContractValidationError) as excinfo:
        emit_probabilistic_prediction_artifact(
            catalog=context["catalog"],
            feature_view=context["feature_view"],
            evaluation_plan=context["evaluation_plan"],
            evaluation_segment=context["evaluation_segment"],
            fit_result=context["fit_result"],
            score_policy_manifest=context["score_policy"],
            stage_id="outer_test",
            forecast_object_type="distribution",
            stochastic_evidence_mode="production",
        )

    assert excinfo.value.code == "missing_stochastic_model_evidence"

    artifact = _production_probabilistic_artifact(
        forecast_object_type="distribution",
    )

    assert artifact.body["stochastic_support_status"] == "production"
    assert artifact.body["stochastic_support_reason_codes"] == []
    assert artifact.body["residual_history_refs"] == [
        _residual_history_ref(context).as_dict()
    ]
    assert artifact.body["stochastic_model_refs"]
    assert {
        ref["schema_name"] for ref in artifact.body["stochastic_model_refs"]
    } == {"stochastic_model_manifest@1.0.0"}


def test_production_probabilistic_artifact_exposes_stochastic_model_manifests(
) -> None:
    context = _probabilistic_context(
        candidate_id="analytic_lag1_affine",
        forecast_object_type="distribution",
    )
    residual_history_ref = _residual_history_ref(context)
    supporting_manifests = []

    artifact = _emit_with_context(
        context=context,
        residual_history_ref=residual_history_ref,
        stochastic_evidence_mode="production",
        supporting_artifact_sink=supporting_manifests,
    )

    assert supporting_manifests
    assert [manifest.ref.as_dict() for manifest in supporting_manifests] == (
        artifact.body["stochastic_model_refs"]
    )
    assert {
        manifest.schema_name for manifest in supporting_manifests
    } == {"stochastic_model_manifest@1.0.0"}
    assert {
        manifest.body["residual_history_ref"]["object_id"]
        for manifest in supporting_manifests
    } == {residual_history_ref.object_id}


def test_production_support_scale_ignores_optimizer_loss_proxy() -> None:
    context = _probabilistic_context(
        candidate_id="analytic_lag1_affine",
        forecast_object_type="distribution",
    )
    residual_history_ref = _residual_history_ref(context)
    original = _emit_with_context(
        context=context,
        residual_history_ref=residual_history_ref,
        stochastic_evidence_mode="production",
    )
    inflated_loss_fit = replace(
        context["fit_result"],
        optimizer_diagnostics={
            **dict(context["fit_result"].optimizer_diagnostics),
            "final_loss": 1_000_000_000.0,
        },
    )
    inflated = _emit_with_context(
        context={**context, "fit_result": inflated_loss_fit},
        residual_history_ref=residual_history_ref,
        stochastic_evidence_mode="production",
    )

    assert [row["scale"] for row in inflated.body["rows"]] == [
        row["scale"] for row in original.body["rows"]
    ]


def test_distribution_rows_carry_fitted_distribution_parameters() -> None:
    artifact = _production_probabilistic_artifact(
        forecast_object_type="distribution",
        stochastic_family_id="student_t",
        student_t_degrees_of_freedom=7.0,
    )

    row = artifact.body["rows"][0]

    assert row["distribution_family"] == "student_t_location_scale"
    assert row["distribution_parameters"]["location"] == row["location"]
    assert row["distribution_parameters"]["scale"] == row["scale"]
    assert row["distribution_parameters"]["df"] == 7.0


def test_intervals_and_quantiles_use_declared_laplace_family() -> None:
    interval_artifact = _production_probabilistic_artifact(
        forecast_object_type="interval",
        stochastic_family_id="laplace",
    )
    quantile_artifact = _production_probabilistic_artifact(
        forecast_object_type="quantile",
        stochastic_family_id="laplace",
    )

    interval_row = interval_artifact.body["rows"][0]
    interval_parameters = interval_row["distribution_parameters"]
    expected_lower = interval_parameters["location"] + (
        interval_parameters["scale"] * math.log(0.2)
    )
    expected_upper = interval_parameters["location"] - (
        interval_parameters["scale"] * math.log(0.2)
    )

    assert interval_row["distribution_family"] == "laplace_location_scale"
    assert interval_row["lower_bound"] == pytest.approx(expected_lower)
    assert interval_row["upper_bound"] == pytest.approx(expected_upper)

    quantile_row = quantile_artifact.body["rows"][0]
    quantile_parameters = quantile_row["distribution_parameters"]
    quantiles = {item["level"]: item["value"] for item in quantile_row["quantiles"]}

    assert quantile_row["distribution_family"] == "laplace_location_scale"
    assert quantiles[0.1] == pytest.approx(
        quantile_parameters["location"]
        + (quantile_parameters["scale"] * math.log(0.2))
    )
    assert quantiles[0.5] == pytest.approx(quantile_parameters["location"])
    assert quantiles[0.9] == pytest.approx(
        quantile_parameters["location"]
        - (quantile_parameters["scale"] * math.log(0.2))
    )


def test_interval_and_quantile_rows_use_configured_levels() -> None:
    interval_artifact = _production_probabilistic_artifact(
        forecast_object_type="interval",
        interval_levels=(0.5, 0.9),
    )
    quantile_artifact = _production_probabilistic_artifact(
        forecast_object_type="quantile",
        quantile_levels=(0.2, 0.8),
    )

    interval_row = interval_artifact.body["rows"][0]
    quantile_row = quantile_artifact.body["rows"][0]

    assert [
        item["nominal_coverage"] for item in interval_row["intervals"]
    ] == [0.5, 0.9]
    assert [
        item["level"] for item in quantile_row["quantiles"]
    ] == [0.2, 0.8]
    assert interval_artifact.body["effective_probabilistic_config"][
        "interval_levels"
    ] == [0.5, 0.9]
    assert quantile_artifact.body["effective_probabilistic_config"][
        "quantile_levels"
    ] == [0.2, 0.8]


def test_event_probabilities_bind_declared_family() -> None:
    artifact = _production_probabilistic_artifact(
        forecast_object_type="event_probability",
        stochastic_family_id="laplace",
    )
    row = artifact.body["rows"][0]
    parameters = row["distribution_parameters"]
    threshold = float(row["event_definition"]["threshold"])
    location = parameters["location"]
    scale = parameters["scale"]
    expected = (
        0.5 * math.exp(-(threshold - location) / scale)
        if threshold >= location
        else 1.0 - (0.5 * math.exp((threshold - location) / scale))
    )

    assert row["distribution_family"] == "laplace_location_scale"
    assert row["event_probability"] == pytest.approx(expected)


def test_heuristic_gaussian_support_cannot_satisfy_production_evidence() -> None:
    context = _probabilistic_context(
        candidate_id="analytic_lag1_affine",
        forecast_object_type="distribution",
    )

    compatibility_artifact = _emit_with_context(context=context)

    assert compatibility_artifact.body["stochastic_support_status"] == "compatibility"
    assert compatibility_artifact.body["stochastic_model_refs"] == []

    with pytest.raises(ContractValidationError) as excinfo:
        _emit_with_context(
            context=context,
            stochastic_evidence_mode="production",
        )

    assert excinfo.value.code == "missing_stochastic_model_evidence"


def test_legacy_interval_and_quantile_defaults_are_stable() -> None:
    interval_artifact = _probabilistic_artifact(
        candidate_id="analytic_lag1_affine",
        forecast_object_type="interval",
    )
    quantile_artifact = _probabilistic_artifact(
        candidate_id="analytic_lag1_affine",
        forecast_object_type="quantile",
    )

    assert {
        row["nominal_coverage"] for row in interval_artifact.body["rows"]
    } == {0.8}
    assert [
        quantile["level"]
        for quantile in quantile_artifact.body["rows"][0]["quantiles"]
    ] == [0.1, 0.5, 0.9]


def _feature_view(candidate_id: str):
    observations = (
        (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0)
        if candidate_id.startswith("spectral_")
        else (1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0)
    )
    snapshot = FrozenDatasetSnapshot(
        series_id="probabilistic-evaluation-series",
        cutoff_available_at="2026-01-08T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-0{index + 1}T00:00:00Z",
                available_at=f"2026-01-0{index + 1}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:{chr(ord('a') + index)}",
            )
            for index, value in enumerate(observations)
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    return materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    ), audit


def _probabilistic_artifact(*, candidate_id: str, forecast_object_type: str):
    context = _probabilistic_context(
        candidate_id=candidate_id,
        forecast_object_type=forecast_object_type,
    )
    return _emit_with_context(context=context)


def _production_probabilistic_artifact(
    *,
    forecast_object_type: str,
    stochastic_family_id: str = "gaussian",
    student_t_degrees_of_freedom: float | None = None,
    interval_levels: tuple[float, ...] = (0.8,),
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
):
    context = _probabilistic_context(
        candidate_id="analytic_lag1_affine",
        forecast_object_type=forecast_object_type,
    )
    return _emit_with_context(
        context=context,
        residual_history_ref=_residual_history_ref(context),
        stochastic_evidence_mode="production",
        stochastic_family_id=stochastic_family_id,
        student_t_degrees_of_freedom=student_t_degrees_of_freedom,
        interval_levels=interval_levels,
        quantile_levels=quantile_levels,
    )


def _probabilistic_context(*, candidate_id: str, forecast_object_type: str):
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view(candidate_id)
    min_train_size = 4 if candidate_id.startswith("spectral_") else 3
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=min_train_size,
        horizon=2,
    )
    search_plan = _search_plan(evaluation_plan)
    score_policy = _probabilistic_score_policy_manifest(
        catalog=catalog,
        evaluation_plan=evaluation_plan,
        forecast_object_type=forecast_object_type,
    )
    candidate = _candidate(feature_view, search_plan, candidate_id)
    fit_result = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )
    return {
        "catalog": catalog,
        "feature_view": feature_view,
        "evaluation_plan": evaluation_plan,
        "evaluation_segment": evaluation_plan.development_segments[0],
        "search_plan": search_plan,
        "score_policy": score_policy,
        "fit_result": fit_result,
    }


def _residual_history_ref(context) -> TypedRef:
    fit_artifacts = build_candidate_fit_artifacts(
        catalog=context["catalog"],
        fit_result=context["fit_result"],
        search_plan_ref=TypedRef(
            "search_plan_manifest@1.0.0",
            context["search_plan"].search_plan_id,
        ),
        selection_floor_bits=0.0,
    )
    return fit_artifacts.residual_history.ref


def _emit_with_context(
    *,
    context,
    residual_history_ref: TypedRef | None = None,
    stochastic_evidence_mode: str = "compatibility",
    stochastic_family_id: str = "gaussian",
    student_t_degrees_of_freedom: float | None = None,
    interval_levels: tuple[float, ...] = (0.8,),
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    supporting_artifact_sink=None,
):
    kwargs = {}
    if supporting_artifact_sink is not None:
        kwargs["supporting_artifact_sink"] = supporting_artifact_sink
    return emit_probabilistic_prediction_artifact(
        catalog=context["catalog"],
        feature_view=context["feature_view"],
        evaluation_plan=context["evaluation_plan"],
        evaluation_segment=context["evaluation_segment"],
        fit_result=context["fit_result"],
        score_policy_manifest=context["score_policy"],
        stage_id="outer_test",
        forecast_object_type=context["score_policy"].body["forecast_object_type"],
        stochastic_evidence_mode=stochastic_evidence_mode,
        residual_history_ref=residual_history_ref,
        stochastic_family_id=stochastic_family_id,
        student_t_degrees_of_freedom=student_t_degrees_of_freedom,
        interval_levels=interval_levels,
        quantile_levels=quantile_levels,
        **kwargs,
    )


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
            "analytic_lag1_affine",
            "recursive_level_smoother",
            "spectral_harmonic_1",
            "algorithmic_last_observation",
        ),
        proposal_limit=4,
        search_class="exact_finite_enumeration",
        seasonal_period=4,
        minimum_description_gain_bits=-2.0,
    )


def _candidate(feature_view, search_plan, candidate_id: str):
    adapter = _adapter_for_candidate(candidate_id)
    if adapter is not None:
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


def _adapter_for_candidate(candidate_id: str):
    if candidate_id.startswith("analytic_"):
        return AnalyticSearchBackendAdapter()
    if candidate_id.startswith("recursive_"):
        return RecursiveSearchBackendAdapter()
    if candidate_id.startswith("spectral_"):
        return SpectralSearchBackendAdapter()
    if candidate_id.startswith("algorithmic_"):
        return AlgorithmicSearchBackendAdapter()
    return None


def _probabilistic_score_policy_manifest(
    *,
    catalog,
    evaluation_plan: EvaluationPlan,
    forecast_object_type: str,
) -> ManifestEnvelope:
    schema_name = _SCORE_POLICY_SCHEMAS[forecast_object_type]
    primary_score = {
        "distribution": "continuous_ranked_probability_score",
        "interval": "interval_score",
        "quantile": "pinball_loss",
        "event_probability": "brier_score",
    }[forecast_object_type]
    return ManifestEnvelope.build(
        schema_name=schema_name,
        module_id="scoring",
        body={
            "score_policy_id": f"test_{forecast_object_type}_policy_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": forecast_object_type,
            "primary_score": primary_score,
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [
                weight.as_dict() for weight in evaluation_plan.horizon_weights
            ],
            "entity_aggregation_mode": (
                "single_entity_only_no_cross_entity_aggregation"
            ),
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )
