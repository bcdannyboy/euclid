from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.modules.evaluation_governance import (
    build_baseline_registry,
    build_comparison_key,
    build_comparison_universe,
    build_evaluation_event_log,
    build_evaluation_governance,
    build_forecast_comparison_policy,
    build_predictive_gate_policy,
    resolve_confirmatory_promotion_allowed,
)
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="demo-series",
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


def test_build_evaluation_governance_freezes_matching_comparison_keys() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()
    plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    score_policy_ref = TypedRef(
        schema_name="point_score_policy_manifest@1.0.0",
        object_id="point_policy",
    )

    baseline_registry = build_baseline_registry(
        compatible_point_score_policy_ref=score_policy_ref,
    )
    comparison_policy = build_forecast_comparison_policy(
        primary_score_policy_ref=score_policy_ref,
        primary_baseline_id="constant_baseline",
    )
    comparison_key = build_comparison_key(
        evaluation_plan=plan,
        score_policy_ref=score_policy_ref,
    )
    comparison_universe = build_comparison_universe(
        selected_candidate_id="prototype_candidate",
        baseline_id="constant_baseline",
        candidate_primary_score=0.25,
        baseline_primary_score=0.50,
        candidate_comparison_key=comparison_key,
        baseline_comparison_key=comparison_key,
    )
    event_log = build_evaluation_event_log(
        search_plan_ref=TypedRef(
            schema_name="search_plan_manifest@1.0.0",
            object_id="search",
        ),
        frozen_shortlist_ref=TypedRef(
            schema_name="frozen_shortlist_manifest@1.0.0",
            object_id="shortlist",
        ),
        freeze_event_ref=TypedRef(
            schema_name="freeze_event_manifest@1.0.0",
            object_id="freeze",
        ),
        comparison_universe_ref=TypedRef(
            schema_name="comparison_universe_manifest@1.0.0",
            object_id="comparison",
        ),
    )
    governance = build_evaluation_governance(
        comparison_universe_ref=TypedRef(
            schema_name="comparison_universe_manifest@1.0.0",
            object_id="comparison",
        ),
        event_log_ref=TypedRef(
            schema_name="evaluation_event_log_manifest@1.0.0",
            object_id="event-log",
        ),
        freeze_event_ref=TypedRef(
            schema_name="freeze_event_manifest@1.0.0",
            object_id="freeze",
        ),
        frozen_shortlist_ref=TypedRef(
            schema_name="frozen_shortlist_manifest@1.0.0",
            object_id="shortlist",
        ),
        confirmatory_promotion_allowed=True,
    )
    predictive_gate_policy = build_predictive_gate_policy()

    assert baseline_registry.primary_baseline_id == "constant_baseline"
    assert comparison_policy.required_comparison_key_fields == (
        "forecast_object_type",
        "score_policy_ref",
        "horizon_set",
        "scored_origin_set_id",
    )
    assert comparison_universe.comparison_class_status == "comparable"
    assert comparison_universe.candidate_beats_baseline is True
    assert predictive_gate_policy.forecast_object_type == "point"

    baseline_manifest = baseline_registry.to_manifest(catalog)
    policy_manifest = comparison_policy.to_manifest(catalog)
    universe_manifest = comparison_universe.to_manifest(catalog)
    event_log_manifest = event_log.to_manifest(catalog)
    governance_manifest = governance.to_manifest(catalog)
    gate_manifest = predictive_gate_policy.to_manifest(catalog)

    assert baseline_manifest.body["baseline_declarations"][0]["baseline_id"] == (
        "constant_baseline"
    )
    assert policy_manifest.body["required_comparison_key_fields"] == [
        "forecast_object_type",
        "score_policy_ref",
        "horizon_set",
        "scored_origin_set_id",
    ]
    assert universe_manifest.body["comparison_class_status"] == "comparable"
    assert (
        universe_manifest.body["candidate_comparison_key"]["scored_origin_set_id"]
        == plan.scored_origin_set_id
    )
    assert event_log_manifest.body["events"][0]["ref_kind"] == "search_plan"
    assert governance_manifest.body["comparison_regime_id"] == (
        "unconditional_model_pair"
    )
    assert gate_manifest.body["allowed_forecast_object_types"] == ["point"]


def test_build_comparison_universe_rejects_mismatched_scored_origin_sets() -> None:
    feature_view, audit = _feature_view()
    plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    score_policy_ref = TypedRef(
        schema_name="point_score_policy_manifest@1.0.0",
        object_id="point_policy",
    )
    candidate_key = build_comparison_key(
        evaluation_plan=plan,
        score_policy_ref=score_policy_ref,
    )
    baseline_key = replace(
        candidate_key,
        scored_origin_set_id="sha256:mismatched-panel",
    )

    with pytest.raises(ContractValidationError) as exc_info:
        build_comparison_universe(
            selected_candidate_id="prototype_candidate",
            baseline_id="constant_baseline",
            candidate_primary_score=0.25,
            baseline_primary_score=0.50,
            candidate_comparison_key=candidate_key,
            baseline_comparison_key=baseline_key,
        )

    assert exc_info.value.code == "comparison_key_mismatch"


def test_build_comparison_universe_rejects_mismatched_forecast_object_types() -> None:
    feature_view, audit = _feature_view()
    plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    candidate_key = build_comparison_key(
        evaluation_plan=plan,
        score_policy_ref=TypedRef(
            schema_name="probabilistic_score_policy_manifest@1.0.0",
            object_id="distribution_policy",
        ),
        forecast_object_type="distribution",
    )
    baseline_key = replace(candidate_key, forecast_object_type="interval")

    with pytest.raises(ContractValidationError) as exc_info:
        build_comparison_universe(
            selected_candidate_id="prototype_candidate",
            baseline_id="probabilistic_baseline",
            candidate_primary_score=0.25,
            baseline_primary_score=0.30,
            candidate_comparison_key=candidate_key,
            baseline_comparison_key=baseline_key,
        )

    assert exc_info.value.code == "comparison_key_mismatch"


def test_resolve_confirmatory_promotion_blocks_failed_calibration() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    predictive_gate_policy = build_predictive_gate_policy(
        allowed_forecast_object_types=(
            "distribution",
            "interval",
            "quantile",
            "event_probability",
        )
    ).to_manifest(catalog)
    calibration_result = ManifestEnvelope.build(
        schema_name="calibration_result_manifest@1.0.0",
        module_id="scoring",
        body={
            "calibration_result_id": "probabilistic_calibration_failure_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "calibration_contract_ref": {
                "schema_name": "calibration_contract_manifest@1.0.0",
                "object_id": "probabilistic_contract_v1",
            },
            "prediction_artifact_ref": {
                "schema_name": "prediction_artifact_manifest@1.1.0",
                "object_id": "probabilistic_prediction_v1",
            },
            "forecast_object_type": "distribution",
            "status": "failed",
            "failure_reason_code": "calibration_failed",
            "pass": False,
            "gate_effect": "required_for_probabilistic_publication",
            "diagnostics": [],
        },
        catalog=catalog,
    )

    assert (
        resolve_confirmatory_promotion_allowed(
            candidate_beats_baseline=True,
            predictive_gate_policy_manifest=predictive_gate_policy,
            calibration_result_manifest=calibration_result,
        )
        is False
    )


def test_resolve_confirmatory_promotion_rejects_boolean_only_evidence() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    predictive_gate_policy = build_predictive_gate_policy(
        allowed_forecast_object_types=("point",)
    ).to_manifest(catalog)

    assert (
        resolve_confirmatory_promotion_allowed(
            candidate_beats_baseline=True,
            predictive_gate_policy_manifest=predictive_gate_policy,
        )
        is False
    )


def test_event_log_records_search_isolation_and_holdout_access() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    freeze_event_manifest = _freeze_event_manifest(catalog)

    event_log = build_evaluation_event_log(
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
        search_local_segment_ids=("outer_fold_0", "outer_fold_1"),
        confirmatory_segment_id="confirmatory_holdout",
        holdout_access_count=1,
        prediction_artifact_ref=TypedRef(
            schema_name="prediction_artifact_manifest@1.1.0",
            object_id="prediction",
        ),
        run_result_ref=TypedRef(
            schema_name="run_result_manifest@1.1.0",
            object_id="run-result",
        ),
    )

    manifest = event_log.to_manifest(catalog)

    assert manifest.body["search_isolation_evidence"] == {
        "search_plan_ref": {
            "schema_name": "search_plan_manifest@1.0.0",
            "object_id": "search",
        },
        "frozen_shortlist_ref": {
            "schema_name": "frozen_shortlist_manifest@1.0.0",
            "object_id": "shortlist",
        },
        "freeze_event_ref": freeze_event_manifest.ref.as_dict(),
        "search_local_segment_ids": ["outer_fold_0", "outer_fold_1"],
        "holdout_materialized_before_freeze": False,
        "post_freeze_candidate_mutation_count": 0,
        "confirmatory_holdout_influenced_search": False,
    }
    assert manifest.body["stage_bookkeeping"] == [
        {
            "stage_id": "inner_search",
            "stage_kind": "search_local_fitting",
            "segment_ids": ["outer_fold_0", "outer_fold_1"],
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
            "segment_ids": ["confirmatory_holdout"],
            "status": "materialized",
            "access_count": 1,
            "uses_confirmatory_holdout": True,
        },
    ]
    assert manifest.body["confirmatory_access_count"] == 1
    assert manifest.body["post_holdout_mutation_count"] == 0
    assert manifest.body["confirmatory_segment_status"] == "accessed_once"
    assert [
        (event["event_id"], event.get("event_type"), event["ref_kind"])
        for event in manifest.body["events"]
    ] == [
        ("search_plan_frozen", None, "search_plan"),
        ("shortlist_frozen", None, "frozen_shortlist"),
        ("candidate_frozen", None, "freeze_event"),
        ("confirmatory_holdout_materialized", "holdout_materialized", "freeze_event"),
        ("confirmatory_prediction_emitted", None, "prediction_artifact"),
        ("run_result_published", None, "run_result"),
    ]


def test_event_log_rejects_post_holdout_mutation_without_replication() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    freeze_event_manifest = _freeze_event_manifest(catalog)

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


def test_event_log_records_replication_after_holdout_mutation() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    freeze_event_manifest = _freeze_event_manifest(catalog)

    event_log = build_evaluation_event_log(
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
        post_holdout_mutation_count=2,
        replication_segment_ids=("replication_segment_0",),
    )

    manifest = event_log.to_manifest(catalog)

    assert manifest.body["confirmatory_segment_status"] == "replicated_after_mutation"
    assert manifest.body["stage_bookkeeping"][-1] == {
        "stage_id": "fresh_replication",
        "stage_kind": "replication_segment",
        "segment_ids": ["replication_segment_0"],
        "status": "completed",
        "uses_confirmatory_holdout": True,
    }
    assert [
        event["event_type"]
        for event in manifest.body["events"]
        if "event_type" in event
    ] == [
        "holdout_materialized",
        "post_holdout_mutation",
        "fresh_replication_started",
        "fresh_replication_completed",
    ]


def _freeze_event_manifest(catalog):
    return ManifestEnvelope.build(
        schema_name="freeze_event_manifest@1.0.0",
        module_id="search_planning",
        body={
            "freeze_event_id": "freeze",
            "owner_id": "module.search-planning-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "freeze_stage": "global_pair_freeze_pre_holdout",
            "frozen_candidate_ref": {
                "schema_name": "reducer_artifact_manifest@1.0.0",
                "object_id": "candidate",
            },
            "frozen_shortlist_ref": {
                "schema_name": "frozen_shortlist_manifest@1.0.0",
                "object_id": "shortlist",
            },
            "confirmatory_baseline_id": "constant_baseline",
            "baseline_selection_rule": "ex_ante_fixed",
            "holdout_materialized_before_freeze": False,
            "post_freeze_candidate_mutation_count": 0,
        },
        catalog=catalog,
    )
