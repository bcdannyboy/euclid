from __future__ import annotations

import pandas as pd

from euclid.contracts.loader import load_contract_catalog
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.ingestion import ingest_dataframe_dataset
from euclid.modules.snapshotting import freeze_dataset_snapshot
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety


def test_build_evaluation_plan_binds_declared_entity_panel_without_collapsing_entities(
    project_root,
) -> None:
    catalog = load_contract_catalog(project_root)
    frame = pd.DataFrame(
        [
            {
                "entity": entity,
                "event_time": f"2026-01-{day:02d}T00:00:00Z",
                "availability_time": f"2026-01-{day:02d}T06:00:00Z",
                "target": value,
            }
            for entity, values in (
                ("entity-a", (10.0, 11.0, 12.0, 13.0, 14.0)),
                ("entity-b", (20.0, 21.0, 22.0, 23.0, 24.0)),
            )
            for day, value in enumerate(values, start=1)
        ]
    )

    dataset = ingest_dataframe_dataset(frame)
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

    plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=2,
        horizon=1,
    )
    manifest = plan.to_manifest(
        catalog,
        time_safety_audit_ref=audit.to_manifest(catalog).ref,
    )

    assert plan.entity_panel == ("entity-a", "entity-b")
    assert len(plan.development_segments) == 1
    assert plan.confirmatory_segment.role == "confirmatory_holdout"
    assert {origin.entity for origin in plan.scored_origin_panel} == {
        "entity-a",
        "entity-b",
    }
    assert plan.comparison_key == {
        "forecast_object_type": "point",
        "horizon_set": [1],
        "entity_panel": ["entity-a", "entity-b"],
        "scored_origin_set_id": plan.scored_origin_set_id,
    }
    assert manifest.body["entity_panel"] == ["entity-a", "entity-b"]
    assert manifest.body["development_segments"][0]["entity_panel"] == [
        "entity-a",
        "entity-b",
    ]
    assert manifest.body["confirmatory_segment"]["entity_panel"] == [
        "entity-a",
        "entity-b",
    ]
    assert manifest.body["comparison_key"]["entity_panel"] == [
        "entity-a",
        "entity-b",
    ]
    assert manifest.body["scored_origin_panel"] == [
        {
            "entity": "entity-a",
            "available_at": "2026-01-04T06:00:00Z",
            "horizon": 1,
            "origin_index": 1,
            "origin_time": "2026-01-03T00:00:00Z",
            "outer_fold_id": "outer_fold_0",
            "role": "development",
            "scored_origin_id": "outer_fold_0__entity-a__h1",
            "segment_id": "outer_fold_0",
            "target_event_time": "2026-01-04T00:00:00Z",
            "target_index": 2,
        },
        {
            "entity": "entity-b",
            "available_at": "2026-01-04T06:00:00Z",
            "horizon": 1,
            "origin_index": 1,
            "origin_time": "2026-01-03T00:00:00Z",
            "outer_fold_id": "outer_fold_0",
            "role": "development",
            "scored_origin_id": "outer_fold_0__entity-b__h1",
            "segment_id": "outer_fold_0",
            "target_event_time": "2026-01-04T00:00:00Z",
            "target_index": 2,
        },
        {
            "entity": "entity-a",
            "available_at": "2026-01-05T06:00:00Z",
            "horizon": 1,
            "origin_index": 2,
            "origin_time": "2026-01-04T00:00:00Z",
            "outer_fold_id": "outer_fold_1",
            "role": "confirmatory_holdout",
            "scored_origin_id": "outer_fold_1__entity-a__h1",
            "segment_id": "confirmatory_holdout",
            "target_event_time": "2026-01-05T00:00:00Z",
            "target_index": 3,
        },
        {
            "entity": "entity-b",
            "available_at": "2026-01-05T06:00:00Z",
            "horizon": 1,
            "origin_index": 2,
            "origin_time": "2026-01-04T00:00:00Z",
            "outer_fold_id": "outer_fold_1",
            "role": "confirmatory_holdout",
            "scored_origin_id": "outer_fold_1__entity-b__h1",
            "segment_id": "confirmatory_holdout",
            "target_event_time": "2026-01-05T00:00:00Z",
            "target_index": 3,
        },
    ]
