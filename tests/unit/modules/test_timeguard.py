from __future__ import annotations

from pathlib import Path

from euclid.modules.ingestion import ingest_csv_dataset
from euclid.modules.snapshotting import (
    FrozenDatasetSnapshot,
    SnapshotRow,
    freeze_dataset_snapshot,
)
from euclid.modules.timeguard import audit_snapshot_time_safety


def test_audit_snapshot_time_safety_marks_legal_snapshot_as_passed() -> None:
    snapshot = FrozenDatasetSnapshot(
        series_id="demo-series",
        cutoff_available_at="2026-01-03T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=(
            SnapshotRow(
                event_time="2026-01-01T00:00:00Z",
                available_at="2026-01-01T06:00:00Z",
                observed_value=10.0,
                revision_id=0,
                payload_hash="sha256:good-1",
            ),
            SnapshotRow(
                event_time="2026-01-02T00:00:00Z",
                available_at="2026-01-02T06:00:00Z",
                observed_value=12.0,
                revision_id=0,
                payload_hash="sha256:good-2",
            ),
        ),
    )

    audit = audit_snapshot_time_safety(snapshot)

    assert audit.status == "passed"
    assert audit.block_reasons == ()


def test_audit_snapshot_time_safety_blocks_future_availability() -> None:
    snapshot = FrozenDatasetSnapshot(
        series_id="demo-series",
        cutoff_available_at="2026-01-02T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=(
            SnapshotRow(
                event_time="2026-01-01T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                observed_value=10.0,
                revision_id=0,
                payload_hash="sha256:bad",
            ),
        ),
    )

    audit = audit_snapshot_time_safety(snapshot)

    assert audit.status == "blocked"
    assert "future_availability" in audit.block_reasons
    assert audit.coordinate_audits[0].status == "blocked"
    assert audit.coordinate_audits[0].reason_codes == ("future_availability",)
    assert audit.canary_failures[0].reason_code == "future_availability"
    assert audit.causal_availability_window["coordinate_count"] == 1
    assert (
        audit.causal_availability_window["max_available_at"]
        == "2026-01-03T00:00:00Z"
    )


def test_audit_snapshot_time_safety_tracks_each_predictive_coordinate() -> None:
    snapshot = FrozenDatasetSnapshot(
        series_id="demo-series",
        cutoff_available_at="2026-01-03T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=(
            SnapshotRow(
                event_time="2026-01-01T00:00:00Z",
                available_at="2026-01-01T06:00:00Z",
                observed_value=10.0,
                revision_id=0,
                payload_hash="sha256:good-1",
            ),
            SnapshotRow(
                event_time="2026-01-04T00:00:00Z",
                available_at="2026-01-02T06:00:00Z",
                observed_value=12.0,
                revision_id=0,
                payload_hash="sha256:bad-event-time",
            ),
        ),
    )

    audit = audit_snapshot_time_safety(snapshot)

    assert audit.checked_row_count == 2
    assert audit.coordinate_audits[0].status == "passed"
    assert audit.coordinate_audits[1].status == "blocked"
    assert audit.coordinate_audits[1].reason_codes == ("future_event_time",)
    assert audit.canary_failures[0].coordinate_index == 1
    assert audit.canary_failures[0].reason_code == "future_event_time"


def test_audit_snapshot_time_safety_fixture_handles_cutoff_boundary_rows(
    phase03_runtime_fixture_dir: Path,
) -> None:
    dataset = ingest_csv_dataset(
        phase03_runtime_fixture_dir / "availability-edge-series.csv"
    )
    snapshot = freeze_dataset_snapshot(
        dataset.observations,
        cutoff_available_at="2026-01-02T00:00:00Z",
    )

    audit = audit_snapshot_time_safety(snapshot)

    assert [row.event_time for row in snapshot.rows] == [
        "2026-01-01T00:00:00Z",
        "2026-01-03T00:00:00Z",
    ]
    assert [row.available_at for row in snapshot.rows] == [
        "2026-01-02T00:00:00Z",
        "2026-01-02T00:00:00Z",
    ]
    assert audit.status == "blocked"
    assert audit.block_reasons == ("future_event_time",)
    assert audit.coordinate_audits[0].status == "passed"
    assert audit.coordinate_audits[1].reason_codes == ("future_event_time",)
    assert audit.causal_availability_window["coordinate_count"] == 2
    assert audit.causal_availability_window["max_available_at"] == (
        "2026-01-02T00:00:00Z"
    )
