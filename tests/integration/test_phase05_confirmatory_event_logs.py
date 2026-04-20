from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.modules.candidate_fitting import fit_candidate_window
from euclid.modules.evaluation import emit_point_prediction_artifact
from euclid.modules.evaluation_governance import build_evaluation_event_log
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_freeze_event,
    build_frozen_shortlist,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import EvaluationPlan, build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.search.backends import run_descriptive_search_backends

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_phase05_confirmatory_event_log_proves_search_holdout_separation() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(evaluation_plan)
    search_plan_manifest = search_plan.to_manifest(catalog)
    score_policy = _score_policy_manifest(catalog, evaluation_plan)
    search_result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )
    candidate = search_result.accepted_candidates[0]
    fit_result = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.confirmatory_segment,
        search_plan=search_plan,
        stage_id="confirmatory_holdout",
    )
    prediction_artifact = emit_point_prediction_artifact(
        catalog=catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.confirmatory_segment,
        fit_result=fit_result,
        score_policy_manifest=score_policy,
        stage_id="confirmatory_holdout",
    )

    candidate_ref = TypedRef(
        schema_name="reducer_artifact_manifest@1.0.0",
        object_id=fit_result.candidate_id,
    )
    frozen_shortlist_manifest = build_frozen_shortlist(
        search_plan_ref=search_plan_manifest.ref,
        candidate_ref=candidate_ref,
    ).to_manifest(catalog)
    freeze_event_manifest = build_freeze_event(
        frozen_candidate_ref=candidate_ref,
        frozen_shortlist_ref=frozen_shortlist_manifest.ref,
        confirmatory_baseline_id="constant_baseline",
    ).to_manifest(catalog)

    event_log = build_evaluation_event_log(
        search_plan_ref=search_plan_manifest.ref,
        frozen_shortlist_ref=frozen_shortlist_manifest.ref,
        freeze_event_ref=freeze_event_manifest.ref,
        freeze_event_manifest=freeze_event_manifest,
        search_local_segment_ids=tuple(
            segment.segment_id for segment in evaluation_plan.development_segments
        ),
        holdout_access_count=1,
        confirmatory_segment_id=evaluation_plan.confirmatory_segment.segment_id,
        prediction_artifact_ref=prediction_artifact.ref,
    ).to_manifest(catalog)

    assert event_log.body["search_isolation_evidence"] == {
        "search_plan_ref": search_plan_manifest.ref.as_dict(),
        "frozen_shortlist_ref": frozen_shortlist_manifest.ref.as_dict(),
        "freeze_event_ref": freeze_event_manifest.ref.as_dict(),
        "search_local_segment_ids": [
            segment.segment_id for segment in evaluation_plan.development_segments
        ],
        "holdout_materialized_before_freeze": False,
        "post_freeze_candidate_mutation_count": 0,
        "confirmatory_holdout_influenced_search": False,
    }
    assert event_log.body["stage_bookkeeping"] == [
        {
            "stage_id": "inner_search",
            "stage_kind": "search_local_fitting",
            "segment_ids": [
                segment.segment_id for segment in evaluation_plan.development_segments
            ],
            "status": "completed",
            "uses_confirmatory_holdout": False,
        },
        {
            "stage_id": "global_pair_freeze_pre_holdout",
            "stage_kind": "shortlist_freeze",
            "segment_ids": ["global_pair_freeze_pre_holdout"],
            "status": "completed",
            "uses_confirmatory_holdout": False,
        },
        {
            "stage_id": "confirmatory_holdout",
            "stage_kind": "confirmatory_access",
            "segment_ids": [evaluation_plan.confirmatory_segment.segment_id],
            "status": "materialized",
            "access_count": 1,
            "uses_confirmatory_holdout": True,
        },
    ]
    holdout_event = next(
        event
        for event in event_log.body["events"]
        if event.get("event_type") == "holdout_materialized"
    )
    assert holdout_event["stage_id"] == "confirmatory_holdout"
    assert (
        holdout_event["segment_id"] == evaluation_plan.confirmatory_segment.segment_id
    )
    assert holdout_event["ref_kind"] == "freeze_event"


def test_phase05_confirmatory_event_log_requires_fresh_replication_after_mutation() -> (
    None
):
    catalog = load_contract_catalog(PROJECT_ROOT)
    freeze_event_manifest = build_freeze_event(
        frozen_candidate_ref=TypedRef(
            schema_name="reducer_artifact_manifest@1.0.0",
            object_id="candidate",
        ),
        frozen_shortlist_ref=TypedRef(
            schema_name="frozen_shortlist_manifest@1.0.0",
            object_id="shortlist",
        ),
        confirmatory_baseline_id="constant_baseline",
    ).to_manifest(catalog)

    with pytest.raises(ContractValidationError) as exc_info:
        build_evaluation_event_log(
            search_plan_ref=TypedRef(
                schema_name="search_plan_manifest@1.0.0",
                object_id="search",
            ),
            frozen_shortlist_ref=TypedRef(
                schema_name="frozen_shortlist_manifest@1.0.0",
                object_id="shortlist",
            ),
            freeze_event_ref=freeze_event_manifest.ref,
            freeze_event_manifest=freeze_event_manifest,
            holdout_access_count=1,
            post_holdout_mutation_count=1,
        )

    assert exc_info.value.code == "post_holdout_mutation_detected"


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="phase05-confirmatory-event-log-series",
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
                observed_value=12.0,
                revision_id=0,
                payload_hash="sha256:b",
            ),
            SnapshotRow(
                event_time="2026-01-03T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                observed_value=13.0,
                revision_id=0,
                payload_hash="sha256:c",
            ),
            SnapshotRow(
                event_time="2026-01-04T00:00:00Z",
                available_at="2026-01-04T00:00:00Z",
                observed_value=15.0,
                revision_id=0,
                payload_hash="sha256:d",
            ),
            SnapshotRow(
                event_time="2026-01-05T00:00:00Z",
                available_at="2026-01-05T00:00:00Z",
                observed_value=16.0,
                revision_id=0,
                payload_hash="sha256:e",
            ),
            SnapshotRow(
                event_time="2026-01-06T00:00:00Z",
                available_at="2026-01-06T00:00:00Z",
                observed_value=18.0,
                revision_id=0,
                payload_hash="sha256:f",
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


def _score_policy_manifest(catalog, evaluation_plan: EvaluationPlan):
    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": "phase05_confirmatory_event_log_policy_v1",
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
