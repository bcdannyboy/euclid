from __future__ import annotations

from pathlib import Path

from euclid.artifacts import FilesystemArtifactStore
from euclid.contracts.loader import load_contract_catalog
from euclid.control_plane import SQLiteMetadataStore
from euclid.manifest_registry import ManifestRegistry
from euclid.prototype.workflow import run_prototype_reducer_workflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_abstention_publication_stays_free_of_candidate_refs(tmp_path: Path) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
    )

    result = run_prototype_reducer_workflow(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
        minimum_description_gain_bits=10_000.0,
    )

    run_result_body = result.run_result.manifest.body
    assert run_result_body["result_mode"] == "abstention_only_publication"
    assert "primary_reducer_artifact_ref" not in run_result_body
    assert "primary_scorecard_ref" not in run_result_body
    assert "primary_claim_card_ref" not in run_result_body
    assert result.publication_record.manifest.body["comparator_exposure_status"] == (
        "not_applicable_abstention_only"
    )
