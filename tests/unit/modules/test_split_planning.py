from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import (
    EvaluationSegment,
    build_evaluation_plan,
    build_legal_training_origin_panel,
)
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
    return (
        materialize_feature_view(
            snapshot=snapshot,
            audit=audit,
            feature_spec=default_feature_spec(),
        ),
        audit,
    )


def _multi_entity_feature_view():
    def row(
        *,
        entity: str,
        day: int,
        value: float,
        payload_hash: str,
    ) -> SnapshotRow:
        return SnapshotRow(
            entity=entity,
            event_time=f"2026-01-{day:02d}T00:00:00Z",
            available_at=f"2026-01-{day:02d}T00:00:00Z",
            observed_value=value,
            revision_id=0,
            payload_hash=payload_hash,
        )

    snapshot = FrozenDatasetSnapshot(
        series_id="panel-series",
        cutoff_available_at="2026-01-06T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=(
            row(entity="A", day=1, value=10.0, payload_hash="sha256:a1"),
            row(entity="A", day=2, value=11.0, payload_hash="sha256:a2"),
            row(entity="A", day=3, value=12.0, payload_hash="sha256:a3"),
            row(entity="A", day=4, value=13.0, payload_hash="sha256:a4"),
            row(entity="B", day=1, value=20.0, payload_hash="sha256:b1"),
            row(entity="B", day=2, value=21.0, payload_hash="sha256:b2"),
            row(entity="B", day=3, value=22.0, payload_hash="sha256:b3"),
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    return materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )


def _training_segment(
    *,
    train_end_index: int,
    horizon_set: tuple[int, ...],
    entity_panel: tuple[str, ...] = (),
) -> EvaluationSegment:
    test_end_day = train_end_index + max(horizon_set) + 2
    return EvaluationSegment(
        segment_id="outer_fold_0",
        outer_fold_id="outer_fold_0",
        role="development",
        train_start_index=0,
        train_end_index=train_end_index,
        test_start_index=train_end_index + 1,
        test_end_index=train_end_index + max(horizon_set),
        train_row_count=train_end_index + 1,
        test_row_count=max(horizon_set),
        origin_index=train_end_index,
        origin_time=f"2026-01-{train_end_index + 2:02d}T00:00:00Z",
        train_end_event_time=f"2026-01-{train_end_index + 2:02d}T00:00:00Z",
        test_end_event_time=f"2026-01-{test_end_day:02d}T00:00:00Z",
        horizon_set=horizon_set,
        scored_origin_ids=(),
        entity_panel=entity_panel,
    )


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


def test_training_origin_panel_includes_complete_targets_inside_training_slice() -> (
    None
):
    feature_view, audit = _feature_view()
    plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )

    panel = build_legal_training_origin_panel(
        feature_view=feature_view,
        evaluation_segment=plan.development_segments[0],
    )

    assert panel.status == "passed"
    assert panel.split_role == "development"
    assert panel.horizon_set == (1,)
    assert panel.diagnostics == ()
    assert [record.origin_index for record in panel.records] == [0, 1]
    assert [record.target_index for record in panel.records] == [1, 2]
    assert all(
        record.target_index <= plan.development_segments[0].train_end_index
        for record in panel.records
    )
    assert all(record.origin_available_at for record in panel.records)
    assert all(record.target_available_at for record in panel.records)


def test_training_origin_panel_supports_non_contiguous_horizon_sets() -> None:
    feature_view, _ = _feature_view()
    segment = _training_segment(train_end_index=4, horizon_set=(1, 3))

    panel = build_legal_training_origin_panel(
        feature_view=feature_view,
        evaluation_segment=segment,
        horizon_set=(1, 3),
    )

    assert panel.status == "passed"
    assert panel.horizon_set == (1, 3)
    assert [(record.origin_index, record.horizon) for record in panel.records] == [
        (0, 1),
        (0, 3),
        (1, 1),
        (1, 3),
    ]
    assert {record.horizon for record in panel.records} == {1, 3}
    assert all(
        record.target_index == record.origin_index + record.horizon
        for record in panel.records
    )
    assert all(
        record.target_index <= segment.train_end_index for record in panel.records
    )


def test_training_origin_panel_excludes_targets_outside_training_slice() -> None:
    feature_view, _ = _feature_view()
    segment = _training_segment(train_end_index=2, horizon_set=(1, 3))

    panel = build_legal_training_origin_panel(
        feature_view=feature_view,
        evaluation_segment=segment,
        horizon_set=(1, 3),
    )

    assert panel.status == "failed"
    assert panel.records == ()
    assert [diagnostic.code for diagnostic in panel.diagnostics] == [
        "missing_horizon_target",
        "target_outside_training_slice",
    ]
    assert panel.diagnostics[1].origin_index == 0
    assert panel.diagnostics[1].horizon == 3


def test_training_origin_panel_reports_missing_entity_targets() -> None:
    feature_view = _multi_entity_feature_view()
    segment = _training_segment(
        train_end_index=2,
        horizon_set=(1,),
        entity_panel=("A", "B"),
    )

    panel = build_legal_training_origin_panel(
        feature_view=feature_view,
        evaluation_segment=segment,
        horizon_set=(1,),
    )

    assert panel.status == "failed"
    assert panel.records == ()
    assert any(
        diagnostic.code == "missing_entity_target"
        and diagnostic.entity == "B"
        and diagnostic.target_index == 2
        for diagnostic in panel.diagnostics
    )
