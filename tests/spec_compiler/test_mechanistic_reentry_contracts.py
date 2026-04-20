from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
MODELING_PIPELINE_PATH = REPO_ROOT / "docs/reference/modeling-pipeline.md"
SYSTEM_PATH = REPO_ROOT / "docs/reference/system.md"
CONTRACTS_MANIFESTS_PATH = REPO_ROOT / "docs/reference/contracts-manifests.md"
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"
SCHEMA_REGISTRY_PATH = REPO_ROOT / "schemas/contracts/schema-registry.yaml"
REFERENCE_TYPES_PATH = REPO_ROOT / "schemas/contracts/reference-types.yaml"

REENTRY_CONTRACTS = {
    "domain-specific-mechanism-mapping.yaml": (
        "domain_specific_mechanism_mapping_manifest@1.0.0"
    ),
    "units-check.yaml": "units_check_manifest@1.0.0",
    "invariance-test.yaml": "invariance_test_manifest@1.0.0",
    "external-evidence-manifest.yaml": "external_evidence_manifest@1.0.0",
    "evidence-independence-protocol.yaml": (
        "evidence_independence_protocol_manifest@1.0.0"
    ),
}
EXPECTED_OWNERS = {
    "domain_specific_mechanism_mapping_manifest@1.0.0": (
        "mechanistic_evidence_owner",
        "mechanistic_evidence",
    ),
    "units_check_manifest@1.0.0": (
        "mechanistic_evidence_owner",
        "mechanistic_evidence",
    ),
    "invariance_test_manifest@1.0.0": (
        "mechanistic_evidence_owner",
        "mechanistic_evidence",
    ),
    "external_evidence_manifest@1.0.0": (
        "external_evidence_ingestion_owner",
        "external_evidence_ingestion",
    ),
    "evidence_independence_protocol_manifest@1.0.0": (
        "mechanistic_evidence_owner",
        "mechanistic_evidence",
    ),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing required file: {path.relative_to(REPO_ROOT)}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _profile(payload: dict[str, Any], schema_name: str) -> dict[str, Any]:
    for profile in payload["reference_profiles"]:
        if profile["schema_name"] == schema_name:
            return profile
    raise AssertionError(f"missing ref profile for {schema_name}")


def _field(profile: dict[str, Any], path: str) -> dict[str, Any]:
    for field in profile["fields"]:
        if field["path"] == path:
            return field
    raise AssertionError(f"missing field {path!r} for {profile['schema_name']}")


def test_mechanistic_reentry_contracts_are_versioned_registered_and_documented() -> None:
    schema_registry = _load_yaml(SCHEMA_REGISTRY_PATH)
    registry_entries = {
        entry["schema_name"]: entry for entry in schema_registry["schemas"]
    }

    for filename, schema_name in REENTRY_CONTRACTS.items():
        path = REPO_ROOT / "schemas/contracts" / filename
        payload = _load_yaml(path)

        assert payload["version"] == 1
        assert "kind" in payload
        assert payload["owners"], f"{filename} must declare an owner"
        assert schema_name in registry_entries, f"{schema_name} must be registered"

        registry_entry = registry_entries[schema_name]
        expected_owner_ref, expected_module = EXPECTED_OWNERS[schema_name]
        assert registry_entry["owner_ref"] == expected_owner_ref
        assert registry_entry["owning_module"] == expected_module
        assert registry_entry["canonical_source_path"] == "docs/reference/contracts-manifests.md"


def test_mechanistic_lane_policy_profile_requires_all_reentry_contract_refs() -> None:
    reference_types = _load_yaml(REFERENCE_TYPES_PATH)
    profile = _profile(reference_types, "mechanistic_lane_policy_manifest@1.0.0")

    expected_fields = {
        "mechanism_mapping_ref": "domain_specific_mechanism_mapping_manifest@1.0.0",
        "units_check_ref": "units_check_manifest@1.0.0",
        "invariance_test_ref": "invariance_test_manifest@1.0.0",
        "external_evidence_ref": "external_evidence_manifest@1.0.0",
        "evidence_independence_ref": (
            "evidence_independence_protocol_manifest@1.0.0"
        ),
    }

    for path, schema_name in expected_fields.items():
        field = _field(profile, path)
        assert field["required"] is True
        assert field["cardinality"] == "exactly_one"
        assert field["allowed_schema_names"] == [schema_name]


def test_reference_docs_treat_mechanistic_evidence_as_live_publication_input() -> None:
    readme_body = README_PATH.read_text(encoding="utf-8")
    modeling_body = MODELING_PIPELINE_PATH.read_text(encoding="utf-8")
    system_body = SYSTEM_PATH.read_text(encoding="utf-8")
    contracts_body = CONTRACTS_MANIFESTS_PATH.read_text(encoding="utf-8")
    source_map_text = SOURCE_MAP_PATH.read_text(encoding="utf-8")

    assert "calibration, robustness, and mechanistic evidence artifacts" in readme_body
    assert "mechanistic inputs" in modeling_body
    assert "`modules/claims.py` maps scorecards into claim cards or abstentions and caps interpretation scope." in modeling_body
    assert "`modules/replay.py` builds reproducibility bundles and verifies replay." in modeling_body
    assert "mechanistic evidence" in system_body
    assert "scorecards, claims, and abstentions" in contracts_body
    assert "run results and publication records" in contracts_body

    for deleted_path in (
        "docs/module-specs/mechanistic-evidence.md",
        "docs/math/mechanistic-interpretation.md",
    ):
        assert deleted_path not in source_map_text
