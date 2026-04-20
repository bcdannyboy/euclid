from __future__ import annotations

from pathlib import Path

from euclid.contracts.loader import load_contract_catalog
from euclid.modules.ingestion import ObservationRecord, ingest_csv_dataset
from euclid.modules.snapshotting import freeze_dataset_snapshot
from euclid.runtime.hashing import sha256_digest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_freeze_dataset_snapshot_uses_latest_visible_revision_per_event_time() -> None:
    observations = (
        ObservationRecord(
            entity="demo-series",
            event_time="2026-01-01T00:00:00Z",
            availability_time="2026-01-01T00:30:00Z",
            target=10.0,
            side_information={"revision_id": 0},
            payload_hash="sha256:first",
            source_id="demo-source",
        ),
        ObservationRecord(
            entity="demo-series",
            event_time="2026-01-01T00:00:00Z",
            availability_time="2026-01-03T00:30:00Z",
            target=11.0,
            side_information={"revision_id": 1},
            payload_hash="sha256:second",
            source_id="demo-source",
        ),
        ObservationRecord(
            entity="demo-series",
            event_time="2026-01-02T00:00:00Z",
            availability_time="2026-01-02T00:30:00Z",
            target=12.0,
            side_information={"revision_id": 0},
            payload_hash="sha256:third",
            source_id="demo-source",
        ),
    )

    early_snapshot = freeze_dataset_snapshot(
        observations,
        cutoff_available_at="2026-01-02T12:00:00Z",
    )
    late_snapshot = freeze_dataset_snapshot(
        observations,
        cutoff_available_at="2026-01-04T00:00:00Z",
    )

    assert [row.observed_value for row in early_snapshot.rows] == [10.0, 12.0]
    assert [row.observed_value for row in late_snapshot.rows] == [11.0, 12.0]


def test_freeze_dataset_snapshot_materializes_raw_and_coded_views_with_provenance(
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    observations = (
        ObservationRecord(
            entity="demo-series",
            event_time="2026-01-01T00:00:00Z",
            availability_time="2026-01-01T00:30:00Z",
            target=10.0,
            side_information={"revision_id": 0},
            payload_hash="sha256:first",
            source_id="source-a",
        ),
        ObservationRecord(
            entity="demo-series",
            event_time="2026-01-01T00:00:00Z",
            availability_time="2026-01-03T00:30:00Z",
            target=11.0,
            side_information={"revision_id": 1},
            payload_hash="sha256:second",
            source_id="source-a",
        ),
        ObservationRecord(
            entity="demo-series",
            event_time="2026-01-02T00:00:00Z",
            availability_time="2026-01-02T00:30:00Z",
            target=None,
            side_information={"revision_id": 0},
            payload_hash="sha256:third",
            source_id="source-a",
        ),
        ObservationRecord(
            entity="demo-series",
            event_time="2026-01-03T00:00:00Z",
            availability_time="2026-01-02T12:00:00Z",
            target=12.0,
            side_information={"revision_id": 0},
            payload_hash="sha256:fourth",
            source_id="source-b",
        ),
    )

    snapshot = freeze_dataset_snapshot(
        observations,
        cutoff_available_at="2026-01-04T00:00:00Z",
    )
    manifest = snapshot.to_manifest(catalog)

    assert [row.target for row in snapshot.raw_rows] == [11.0, None, 12.0]
    assert [row.observed_value for row in snapshot.rows] == [11.0, 12.0]
    assert snapshot.sampling_metadata.visible_observation_count == 4
    assert snapshot.sampling_metadata.raw_row_count == 3
    assert snapshot.sampling_metadata.coded_row_count == 2
    assert snapshot.sampling_metadata.excluded_missing_target_count == 1
    assert [record.source_id for record in snapshot.source_provenance] == [
        "source-a",
        "source-b",
    ]
    assert [record.raw_row_count for record in snapshot.source_provenance] == [2, 1]
    assert [record.coded_row_count for record in snapshot.source_provenance] == [1, 1]
    assert manifest.body["raw_observation_hash"] == sha256_digest(
        manifest.body["raw_observations"]
    )
    assert manifest.body["coded_target_hash"] == sha256_digest(
        manifest.body["coded_targets"]
    )
    assert manifest.body["lineage_payload_hash"] == sha256_digest(
        manifest.body["lineage_payload_hashes"]
    )
    assert manifest.body["sampling_metadata"] == {
        "coded_row_count": 2,
        "excluded_missing_target_count": 1,
        "order_relation": "entity_then_event_time_then_availability_time",
        "raw_row_count": 3,
        "visible_observation_count": 4,
    }
    assert manifest.body["admitted_side_information_fields"] == ["revision_id"]


def test_freeze_dataset_snapshot_is_deterministic_under_input_permutation(
    phase03_runtime_fixture_dir: Path,
) -> None:
    dataset = ingest_csv_dataset(
        phase03_runtime_fixture_dir / "missing-target-series.csv"
    )

    forward = freeze_dataset_snapshot(
        dataset.observations,
        cutoff_available_at="2026-01-03T12:00:00Z",
    )
    reversed_snapshot = freeze_dataset_snapshot(
        tuple(reversed(dataset.observations)),
        cutoff_available_at="2026-01-03T12:00:00Z",
    )

    assert reversed_snapshot == forward


def test_freeze_dataset_snapshot_fixture_excludes_latest_missing_revision_from_coding(
    phase03_runtime_fixture_dir: Path,
) -> None:
    dataset = ingest_csv_dataset(
        phase03_runtime_fixture_dir / "missing-target-series.csv"
    )

    snapshot = freeze_dataset_snapshot(
        dataset.observations,
        cutoff_available_at="2026-01-03T12:00:00Z",
    )

    assert [row.target for row in snapshot.raw_rows] == [None, 12.0, 13.0]
    assert [row.observed_value for row in snapshot.rows] == [12.0, 13.0]
    assert snapshot.sampling_metadata.visible_observation_count == 4
    assert snapshot.sampling_metadata.raw_row_count == 3
    assert snapshot.sampling_metadata.coded_row_count == 2
    assert snapshot.sampling_metadata.excluded_missing_target_count == 1
