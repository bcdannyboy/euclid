from __future__ import annotations

from pathlib import Path

from euclid.contracts.loader import load_contract_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_load_contract_catalog_exposes_known_schema_enum_and_ref_metadata() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    schema = catalog.get_schema("reproducibility_bundle_manifest@1.0.0")

    assert schema.owning_module == "replay"
    assert "candidate_publication" in catalog.enum_values("replay_bundle_mode")
    assert catalog.allowed_ref_schema_names(
        "reproducibility_bundle_manifest@1.0.0",
        "run_result_ref",
    ) == {"run_result_manifest@1.1.0"}


def test_load_contract_catalog_exposes_schema_family_and_contract_metadata() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    schema = catalog.get_schema("publication_record_manifest@1.1.0")
    enum_definition = catalog.get_enum("claim_lanes")
    contract = catalog.get_contract_document("publication_lifecycle")

    assert schema.family == "publication_record_manifest"
    assert schema.version == "1.1.0"
    assert catalog.schema_names_for_family("publication_record_manifest") == (
        "publication_record_manifest@1.1.0",
    )
    assert enum_definition.source_kind == "core_vocabulary"
    assert "predictively_supported" in enum_definition.allowed_values
    assert contract.kind == "publication_lifecycle"
    assert "publication_completed" in contract.state_ids
