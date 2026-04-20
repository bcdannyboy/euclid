from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.modules.ingestion import (
    ingest_csv_dataset,
    ingest_csv_observations,
    ingest_dataframe_dataset,
    ingest_parquet_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_ingest_csv_dataset_preserves_canonical_fields_and_side_information(
    tmp_path,
) -> None:
    csv_path = tmp_path / "series.csv"
    csv_path.write_text(
        "\n".join(
            [
                "entity,event_time,availability_time,target,revision_id,temperature_c",
                "demo-entity,2026-01-01T00:00:00Z,2026-01-01T06:00:00Z,10.0,0,17.5",
                "demo-entity,2026-01-02T00:00:00Z,2026-01-02T09:00:00Z,12.5,1,18.0",
            ]
        ),
        encoding="utf-8",
    )
    catalog = load_contract_catalog(PROJECT_ROOT)

    dataset = ingest_csv_dataset(csv_path)
    observations = dataset.observations

    assert dataset.entity == "demo-entity"
    assert [observation.event_time for observation in observations] == [
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
    ]
    assert [observation.availability_time for observation in observations] == [
        "2026-01-01T06:00:00Z",
        "2026-01-02T09:00:00Z",
    ]
    assert [observation.target for observation in observations] == [10.0, 12.5]
    assert observations[0].side_information == {"revision_id": 0, "temperature_c": 17.5}
    assert observations[0].payload_hash.startswith("sha256:")
    assert (
        observations[0].to_manifest(catalog).schema_name
        == "observation_record@1.0.0"
    )
    assert observations[0].to_manifest(catalog).body["entity"] == "demo-entity"
    assert dataset.coded_targets == (10.0, 12.5)


def test_ingest_csv_dataset_supports_legacy_aliases_and_excludes_missing_targets(
    tmp_path,
) -> None:
    csv_path = tmp_path / "series.csv"
    csv_path.write_text(
        "\n".join(
            [
                "source_id,series_id,event_time,available_at,observed_value,revision_id",
                "demo-source,demo-series,2026-01-02T00:00:00Z,2026-01-02T06:00:00Z,12.0,1",
                "demo-source,demo-series,2026-01-01T00:00:00Z,2026-01-01T07:00:00Z,,0",
                "demo-source,demo-series,2026-01-01T00:00:00Z,2026-01-01T06:00:00Z,10.0,0",
            ]
        ),
        encoding="utf-8",
    )

    dataset = ingest_csv_dataset(csv_path)

    assert [observation.event_time for observation in dataset.observations] == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
    ]
    assert [observation.availability_time for observation in dataset.observations] == [
        "2026-01-01T06:00:00Z",
        "2026-01-01T07:00:00Z",
        "2026-01-02T06:00:00Z",
    ]
    assert [observation.target for observation in dataset.observations] == [
        10.0,
        None,
        12.0,
    ]
    assert dataset.coded_targets == (10.0, 12.0)
    assert [
        observation.observed_value
        for observation in ingest_csv_observations(csv_path)
    ] == [10.0, 12.0]


def test_ingest_dataframe_dataset_admits_multi_entity_panels() -> None:
    frame = pd.DataFrame(
        [
            {
                "entity": "entity-a",
                "event_time": "2026-01-02T00:00:00Z",
                "availability_time": "2026-01-02T02:00:00Z",
                "target": 12.0,
            },
            {
                "entity": "entity-b",
                "event_time": "2026-01-01T00:00:00Z",
                "availability_time": "2026-01-01T02:00:00Z",
                "target": 10.0,
            },
        ]
    )

    dataset = ingest_dataframe_dataset(frame)

    assert dataset.entity_panel == ("entity-a", "entity-b")


def test_ingest_dataframe_dataset_rejects_missing_entity_field() -> None:
    frame = pd.DataFrame(
        [
            {
                "event_time": "2026-01-01T00:00:00Z",
                "availability_time": "2026-01-01T02:00:00Z",
                "target": 10.0,
            }
        ]
    )

    with pytest.raises(ContractValidationError) as excinfo:
        ingest_dataframe_dataset(frame)

    assert excinfo.value.code == "missing_required_field"
    assert excinfo.value.field_path == "<dataframe>[0].entity"


def test_ingest_dataframe_dataset_rejects_nonfinite_present_targets() -> None:
    frame = pd.DataFrame(
        [
            {
                "entity": "entity-a",
                "event_time": "2026-01-01T00:00:00Z",
                "availability_time": "2026-01-01T00:30:00Z",
                "target": 10.0,
            },
            {
                "entity": "entity-a",
                "event_time": "2026-01-02T00:00:00Z",
                "availability_time": "2026-01-02T00:30:00Z",
                "target": float("inf"),
            },
        ]
    )

    with pytest.raises(ContractValidationError) as excinfo:
        ingest_dataframe_dataset(frame)

    assert excinfo.value.code == "nonfinite_numeric_value"


def test_ingest_parquet_dataset_reads_parquet_inputs(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "entity": "entity-a",
                "event_time": "2026-01-02T00:00:00Z",
                "availability_time": "2026-01-02T06:00:00Z",
                "target": 12.0,
                "revision_id": 1,
            },
            {
                "entity": "entity-a",
                "event_time": "2026-01-01T00:00:00Z",
                "availability_time": "2026-01-01T06:00:00Z",
                "target": 10.0,
                "revision_id": 0,
            },
        ]
    )
    parquet_path = tmp_path / "series.parquet"
    frame.to_parquet(parquet_path)

    dataset = ingest_parquet_dataset(parquet_path)

    assert dataset.entity == "entity-a"
    assert dataset.coded_targets == (10.0, 12.0)
    assert dataset.observations[0].side_information["revision_id"] == 0


def test_ingest_csv_dataset_fixture_keeps_missing_targets_out_of_coding(
    phase03_runtime_fixture_dir: Path,
) -> None:
    dataset = ingest_csv_dataset(
        phase03_runtime_fixture_dir / "missing-target-series.csv"
    )

    assert [observation.event_time for observation in dataset.observations] == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
        "2026-01-03T00:00:00Z",
    ]
    assert [observation.availability_time for observation in dataset.observations] == [
        "2026-01-01T06:00:00Z",
        "2026-01-02T06:00:00Z",
        "2026-01-02T06:00:00Z",
        "2026-01-03T06:00:00Z",
    ]
    assert [observation.target for observation in dataset.observations] == [
        10.0,
        None,
        12.0,
        13.0,
    ]
    assert dataset.coded_targets == (10.0, 12.0, 13.0)
