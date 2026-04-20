from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.modules.features import (
    FeatureSpec,
    default_feature_spec,
    materialize_feature_view,
)
from euclid.modules.ingestion import ingest_csv_dataset
from euclid.modules.snapshotting import (
    FrozenDatasetSnapshot,
    SnapshotRow,
    freeze_dataset_snapshot,
)
from euclid.modules.timeguard import audit_snapshot_time_safety


def _snapshot() -> FrozenDatasetSnapshot:
    return FrozenDatasetSnapshot(
        series_id="demo-series",
        cutoff_available_at="2026-01-05T00:00:00Z",
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
        ),
    )


def test_materialize_feature_view_generates_past_only_lag_features() -> None:
    snapshot = _snapshot()
    audit = audit_snapshot_time_safety(snapshot)

    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )

    assert feature_view.feature_names == ("lag_1",)
    assert feature_view.rows[0]["target"] == 12.0
    assert feature_view.rows[0]["lag_1"] == 10.0
    assert feature_view.rows[-1]["target"] == 15.0
    assert feature_view.rows[-1]["lag_1"] == 13.0


def test_materialize_feature_view_rejects_centered_features() -> None:
    snapshot = _snapshot()
    audit = audit_snapshot_time_safety(snapshot)
    spec = FeatureSpec(
        feature_spec_id="invalid-centered",
        features=(
            {
                "feature_id": "centered_mean_3",
                "kind": "centered_rolling_mean",
                "window": 3,
            },
        ),
    )

    with pytest.raises(ContractValidationError) as excinfo:
        materialize_feature_view(snapshot=snapshot, audit=audit, feature_spec=spec)

    assert excinfo.value.code == "illegal_feature_spec"


def test_materialize_feature_view_uses_only_legal_history_by_availability() -> None:
    snapshot = FrozenDatasetSnapshot(
        series_id="demo-series",
        cutoff_available_at="2026-01-06T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=(
            SnapshotRow(
                event_time="2026-01-01T00:00:00Z",
                available_at="2026-01-04T00:00:00Z",
                observed_value=10.0,
                revision_id=0,
                payload_hash="sha256:late-first",
            ),
            SnapshotRow(
                event_time="2026-01-02T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                observed_value=20.0,
                revision_id=0,
                payload_hash="sha256:second",
            ),
            SnapshotRow(
                event_time="2026-01-03T00:00:00Z",
                available_at="2026-01-05T00:00:00Z",
                observed_value=30.0,
                revision_id=0,
                payload_hash="sha256:third",
            ),
            SnapshotRow(
                event_time="2026-01-04T00:00:00Z",
                available_at="2026-01-06T00:00:00Z",
                observed_value=40.0,
                revision_id=0,
                payload_hash="sha256:fourth",
            ),
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)

    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )

    assert [row["event_time"] for row in feature_view.rows] == [
        "2026-01-03T00:00:00Z",
        "2026-01-04T00:00:00Z",
    ]
    assert feature_view.rows[0]["lag_1"] == 20.0
    assert feature_view.rows[1]["lag_1"] == 30.0
    assert feature_view.materialization_report.excluded_coordinate_count == 1
    assert feature_view.materialization_report.coordinate_audits[0].reason_codes == (
        "insufficient_legal_history",
    )
    assert feature_view.materialization_report.canary_failures[0].reason_code == (
        "late_source_availability"
    )


def test_materialize_feature_view_rejects_cached_or_derived_sources() -> None:
    snapshot = _snapshot()
    audit = audit_snapshot_time_safety(snapshot)
    spec = FeatureSpec(
        feature_spec_id="invalid-derived",
        features=(
            {
                "feature_id": "lag_1_cached",
                "kind": "lag",
                "lag_steps": 1,
                "cached_transform": "lag-cache-v1",
            },
        ),
    )

    with pytest.raises(ContractValidationError) as excinfo:
        materialize_feature_view(snapshot=snapshot, audit=audit, feature_spec=spec)

    assert excinfo.value.code == "illegal_feature_spec"


def test_materialize_feature_view_records_only_relevant_late_source_canaries() -> None:
    snapshot = FrozenDatasetSnapshot(
        series_id="demo-series",
        cutoff_available_at="2026-01-05T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=(
            SnapshotRow(
                event_time="2026-01-01T00:00:00Z",
                available_at="2026-01-05T00:00:00Z",
                observed_value=10.0,
                revision_id=0,
                payload_hash="sha256:first",
            ),
            SnapshotRow(
                event_time="2026-01-02T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                observed_value=20.0,
                revision_id=0,
                payload_hash="sha256:second",
            ),
            SnapshotRow(
                event_time="2026-01-03T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                observed_value=30.0,
                revision_id=0,
                payload_hash="sha256:third",
            ),
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)

    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )

    assert [
        failure.coordinate_index
        for failure in feature_view.materialization_report.canary_failures
    ] == [1]


def test_materialize_feature_view_fixture_blocks_leakage_trap_by_availability(
    phase03_runtime_fixture_dir: Path,
) -> None:
    dataset = ingest_csv_dataset(
        phase03_runtime_fixture_dir / "leakage-trap-series.csv"
    )
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

    assert [row["event_time"] for row in feature_view.rows] == [
        "2026-01-03T00:00:00Z",
        "2026-01-04T00:00:00Z",
    ]
    assert feature_view.rows[0]["lag_1"] == 20.0
    assert feature_view.rows[1]["lag_1"] == 30.0
    assert feature_view.materialization_report.excluded_coordinate_count == 1
    assert feature_view.materialization_report.coordinate_audits[0].reason_codes == (
        "insufficient_legal_history",
    )
    assert feature_view.materialization_report.canary_failures[0].reason_code == (
        "late_source_availability"
    )
