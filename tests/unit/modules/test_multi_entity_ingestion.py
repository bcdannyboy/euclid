from __future__ import annotations

import pandas as pd

from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.ingestion import ingest_dataframe_dataset
from euclid.modules.snapshotting import freeze_dataset_snapshot
from euclid.modules.timeguard import audit_snapshot_time_safety


def test_multi_entity_pipeline_preserves_panel_snapshot_and_feature_identity() -> None:
    frame = pd.DataFrame(
        [
            {
                "entity": "entity-b",
                "event_time": "2026-01-02T00:00:00Z",
                "availability_time": "2026-01-02T02:00:00Z",
                "target": 20.0,
                "revision_id": 0,
            },
            {
                "entity": "entity-a",
                "event_time": "2026-01-01T00:00:00Z",
                "availability_time": "2026-01-01T01:00:00Z",
                "target": 10.0,
                "revision_id": 0,
            },
            {
                "entity": "entity-b",
                "event_time": "2026-01-01T00:00:00Z",
                "availability_time": "2026-01-01T02:00:00Z",
                "target": 18.0,
                "revision_id": 0,
            },
            {
                "entity": "entity-a",
                "event_time": "2026-01-02T00:00:00Z",
                "availability_time": "2026-01-02T01:00:00Z",
                "target": 12.0,
                "revision_id": 0,
            },
        ]
    )

    dataset = ingest_dataframe_dataset(frame)

    assert dataset.entity_panel == ("entity-a", "entity-b")
    assert [
        (observation.entity, observation.event_time)
        for observation in dataset.observations
    ] == [
        ("entity-a", "2026-01-01T00:00:00Z"),
        ("entity-a", "2026-01-02T00:00:00Z"),
        ("entity-b", "2026-01-01T00:00:00Z"),
        ("entity-b", "2026-01-02T00:00:00Z"),
    ]

    snapshot = freeze_dataset_snapshot(
        dataset.observations,
        cutoff_available_at="2026-01-03T00:00:00Z",
    )
    audit = audit_snapshot_time_safety(snapshot)
    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )

    assert snapshot.entity_panel == ("entity-a", "entity-b")
    assert [row.entity for row in snapshot.rows] == [
        "entity-a",
        "entity-a",
        "entity-b",
        "entity-b",
    ]
    assert [coordinate.entity for coordinate in audit.coordinate_audits] == [
        "entity-a",
        "entity-a",
        "entity-b",
        "entity-b",
    ]
    assert feature_view.entity_panel == ("entity-a", "entity-b")
    assert feature_view.rows == (
        {
            "entity": "entity-a",
            "entity_row_index": 1,
            "event_time": "2026-01-02T00:00:00Z",
            "available_at": "2026-01-02T01:00:00Z",
            "target": 12.0,
            "lag_1": 10.0,
        },
        {
            "entity": "entity-b",
            "entity_row_index": 1,
            "event_time": "2026-01-02T00:00:00Z",
            "available_at": "2026-01-02T02:00:00Z",
            "target": 20.0,
            "lag_1": 18.0,
        },
    )
