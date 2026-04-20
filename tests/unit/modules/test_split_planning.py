from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
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


def test_build_evaluation_plan_creates_nested_walk_forward_geometry() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()

    plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )

    assert plan.outer_fold_count == 2
    assert plan.folds[-1]["role"] == "confirmatory_holdout"
    manifest = plan.to_manifest(
        catalog,
        time_safety_audit_ref=audit.to_manifest(catalog).ref,
    )
    assert manifest.schema_name == "evaluation_plan_manifest@1.1.0"
    assert manifest.body["split_strategy"] == "nested_walk_forward_only"


def test_build_evaluation_plan_freezes_segments_horizons_and_origin_panel() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()

    plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )

    assert plan.horizon_set == (1,)
    assert [weight.as_dict() for weight in plan.horizon_weights] == [
        {"horizon": 1, "weight": "1"}
    ]
    assert len(plan.development_segments) == 1
    assert plan.development_segments[0].role == "development"
    assert plan.development_segments[0].outer_fold_id == "outer_fold_0"
    assert plan.confirmatory_segment.role == "confirmatory_holdout"
    assert plan.confirmatory_segment.outer_fold_id == "outer_fold_1"
    assert plan.scored_origin_set_id.startswith("sha256:")
    assert [origin.horizon for origin in plan.scored_origin_panel] == [1, 1]
    assert [origin.role for origin in plan.scored_origin_panel] == [
        "development",
        "confirmatory_holdout",
    ]

    manifest = plan.to_manifest(
        catalog,
        time_safety_audit_ref=audit.to_manifest(catalog).ref,
    )

    assert manifest.body["horizon_weights"] == [{"horizon": 1, "weight": "1"}]
    assert manifest.body["development_segments"][0]["role"] == "development"
    assert manifest.body["confirmatory_segment"]["role"] == "confirmatory_holdout"
    assert manifest.body["comparison_key"]["forecast_object_type"] == "point"
    assert manifest.body["comparison_key"]["horizon_set"] == [1]


def test_build_evaluation_plan_rejects_invalid_horizon_weight_simplex() -> None:
    feature_view, audit = _feature_view()

    with pytest.raises(ContractValidationError) as exc_info:
        build_evaluation_plan(
            feature_view=feature_view,
            audit=audit,
            min_train_size=2,
            horizon=2,
            horizon_weight_strings=("0.2", "0.2"),
        )

    assert exc_info.value.code == "invalid_horizon_weight_simplex"
