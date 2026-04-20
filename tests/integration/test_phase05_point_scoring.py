from __future__ import annotations

from pathlib import Path
from statistics import fmean

from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import PredictionArtifactManifest, PredictionRow
from euclid.modules.candidate_fitting import fit_candidate_window
from euclid.modules.evaluation import emit_point_prediction_artifact
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.scoring import evaluate_point_comparators
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import EvaluationPlan, build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.search.backends import run_descriptive_search_backends
from euclid.search_planning import build_canonicalization_policy, build_search_plan

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_search_fit_handoff_scores_candidate_against_declared_constant_baseline() -> (
    None
):
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=4,
        horizon=1,
    )
    search_plan = _search_plan(evaluation_plan)
    score_policy = _score_policy_manifest(catalog, evaluation_plan)
    search_result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )
    candidate = next(
        candidate
        for candidate in search_result.accepted_candidates
        if candidate.evidence_layer.backend_origin_record.source_candidate_id
        == "analytic_lag1_affine"
    )
    fit_result = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.confirmatory_segment,
        search_plan=search_plan,
        stage_id="confirmatory_holdout",
    )
    candidate_artifact = emit_point_prediction_artifact(
        catalog=catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.confirmatory_segment,
        fit_result=fit_result,
        score_policy_manifest=score_policy,
        stage_id="confirmatory_holdout",
    )
    baseline_registry = _baseline_registry_manifest(catalog, score_policy)
    confirmatory_train = feature_view.rows[
        evaluation_plan.confirmatory_segment.train_start_index : (
            evaluation_plan.confirmatory_segment.train_end_index + 1
        )
    ]

    result = evaluate_point_comparators(
        catalog=catalog,
        score_policy_manifest=score_policy,
        candidate_prediction_artifact=candidate_artifact,
        baseline_registry_manifest=baseline_registry,
        comparator_prediction_artifacts={
            "constant_baseline": _constant_baseline_prediction_artifact(
                catalog=catalog,
                evaluation_plan=evaluation_plan,
                score_policy=score_policy,
                candidate_artifact=candidate_artifact,
                training_values=tuple(
                    float(row["target"]) for row in confirmatory_train
                ),
            )
        },
        practical_significance_margin=0.0,
    )

    comparator_statuses = [
        item.body["comparison_status"] for item in result.comparator_score_results
    ]

    assert result.candidate_score_result.body["comparison_status"] == "comparable"
    assert comparator_statuses == ["comparable"]
    assert result.comparison_universe.body["comparison_class_status"] == "comparable"
    assert result.comparison_universe.body["candidate_beats_baseline"] is True
    assert result.comparison_universe.body["paired_comparison_records"] == [
        {
            "comparator_id": "constant_baseline",
            "comparator_kind": "baseline",
            "comparison_status": "comparable",
            "failure_reason_code": None,
            "candidate_primary_score": result.candidate_score_result.body[
                "aggregated_primary_score"
            ],
            "comparator_primary_score": result.comparator_score_results[0].body[
                "aggregated_primary_score"
            ],
            "primary_score_delta": result.comparator_score_results[0].body[
                "aggregated_primary_score"
            ]
            - result.candidate_score_result.body["aggregated_primary_score"],
            "paired_origin_count": 1,
            "mean_loss_differential": result.comparator_score_results[0].body[
                "aggregated_primary_score"
            ]
            - result.candidate_score_result.body["aggregated_primary_score"],
            "per_horizon_mean_loss_differentials": [
                {
                    "horizon": 1,
                    "mean_loss_differential": result.comparator_score_results[0].body[
                        "aggregated_primary_score"
                    ]
                    - result.candidate_score_result.body["aggregated_primary_score"],
                }
            ],
            "practical_significance_margin": 0.0,
            "practical_significance_status": "candidate_better_than_margin",
            "score_result_ref": result.comparator_score_results[0].ref.as_dict(),
        }
    ]


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="phase05-point-score-series",
        cutoff_available_at="2026-01-08T00:00:00Z",
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
                observed_value=12.0,
                revision_id=0,
                payload_hash="sha256:b",
            ),
            SnapshotRow(
                event_time="2026-01-03T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                observed_value=14.0,
                revision_id=0,
                payload_hash="sha256:c",
            ),
            SnapshotRow(
                event_time="2026-01-04T00:00:00Z",
                available_at="2026-01-04T00:00:00Z",
                observed_value=16.0,
                revision_id=0,
                payload_hash="sha256:d",
            ),
            SnapshotRow(
                event_time="2026-01-05T00:00:00Z",
                available_at="2026-01-05T00:00:00Z",
                observed_value=18.0,
                revision_id=0,
                payload_hash="sha256:e",
            ),
            SnapshotRow(
                event_time="2026-01-06T00:00:00Z",
                available_at="2026-01-06T00:00:00Z",
                observed_value=20.0,
                revision_id=0,
                payload_hash="sha256:f",
            ),
            SnapshotRow(
                event_time="2026-01-07T00:00:00Z",
                available_at="2026-01-07T00:00:00Z",
                observed_value=22.0,
                revision_id=0,
                payload_hash="sha256:g",
            ),
            SnapshotRow(
                event_time="2026-01-08T00:00:00Z",
                available_at="2026-01-08T00:00:00Z",
                observed_value=24.0,
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
        ),
        proposal_limit=3,
        search_class="exact_finite_enumeration",
        seasonal_period=4,
    )


def _score_policy_manifest(
    catalog, evaluation_plan: EvaluationPlan
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": "phase05_point_score_policy_v1",
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
        catalog=catalog,
    )


def _baseline_registry_manifest(
    catalog, score_policy: ManifestEnvelope
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="baseline_registry_manifest@1.1.0",
        module_id="evaluation_governance",
        body={
            "baseline_registry_id": "phase05_point_score_baselines_v1",
            "owner_prompt_id": "prompt.predictive-validation-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "primary_baseline_id": "constant_baseline",
            "baseline_ids": ["constant_baseline"],
            "baseline_declarations": [
                {
                    "baseline_id": "constant_baseline",
                    "comparator_declaration_id": "constant_baseline_declaration",
                    "comparator_kind": "baseline",
                    "family_id": "constant",
                    "forecast_object_type": "point",
                    "freeze_rule": "frozen_before_confirmatory_access",
                }
            ],
            "compatible_point_score_policy_ref": score_policy.ref.as_dict(),
        },
        catalog=catalog,
    )


def _constant_baseline_prediction_artifact(
    *,
    catalog,
    evaluation_plan: EvaluationPlan,
    score_policy: ManifestEnvelope,
    candidate_artifact: ManifestEnvelope,
    training_values: tuple[float, ...],
) -> ManifestEnvelope:
    baseline_forecast = float(fmean(training_values))
    rows = tuple(
        PredictionRow(
            origin_time=row["origin_time"],
            available_at=row["available_at"],
            horizon=int(row["horizon"]),
            point_forecast=baseline_forecast,
            realized_observation=float(row["realized_observation"]),
        )
        for row in candidate_artifact.body["rows"]
    )
    return PredictionArtifactManifest(
        prediction_artifact_id="constant_baseline_confirmatory_prediction",
        candidate_id="constant_baseline",
        stage_id="confirmatory_holdout",
        fit_window_id=evaluation_plan.confirmatory_segment.segment_id,
        test_window_id=evaluation_plan.confirmatory_segment.segment_id,
        model_freeze_status="confirmatory_baseline_frozen",
        refit_rule_applied="pre_holdout_development_refit",
        score_policy_ref=score_policy.ref,
        rows=rows,
        score_law_id=str(score_policy.body["point_loss_id"]),
        horizon_weights=tuple(
            dict(weight) for weight in candidate_artifact.body["horizon_weights"]
        ),
        scored_origin_panel=tuple(
            dict(origin) for origin in candidate_artifact.body["scored_origin_panel"]
        ),
        scored_origin_set_id=str(candidate_artifact.body["scored_origin_set_id"]),
        comparison_key=dict(candidate_artifact.body["comparison_key"]),
        missing_scored_origins=(),
        timeguard_checks=tuple(
            dict(check) for check in candidate_artifact.body["timeguard_checks"]
        ),
    ).to_manifest(catalog)
