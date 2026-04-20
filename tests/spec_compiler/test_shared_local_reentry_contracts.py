from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
SEARCH_CORE_PATH = REPO_ROOT / "docs/reference/search-core.md"
MODELING_PIPELINE_PATH = REPO_ROOT / "docs/reference/modeling-pipeline.md"
CONTRACTS_MANIFESTS_PATH = REPO_ROOT / "docs/reference/contracts-manifests.md"
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"
SCHEMA_REGISTRY_PATH = REPO_ROOT / "schemas/contracts/schema-registry.yaml"
REFERENCE_TYPES_PATH = REPO_ROOT / "schemas/contracts/reference-types.yaml"

REENTRY_CONTRACTS = {
    "typed-pooling.yaml": "typed_pooling_manifest@1.0.0",
    "conditional-shared-local-codelength.yaml": (
        "conditional_shared_local_codelength_contract@1.0.0"
    ),
    "exchangeability.yaml": "exchangeability_manifest@1.0.0",
    "unseen-entity-prediction-policy.yaml": (
        "unseen_entity_prediction_policy_manifest@1.0.0"
    ),
    "multi-entity-predictive-evaluation.yaml": (
        "multi_entity_predictive_evaluation_manifest@1.0.0"
    ),
    "shared-local-freeze-refit-protocol.yaml": (
        "shared_local_freeze_refit_protocol_manifest@1.0.0"
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


def test_shared_local_reentry_contracts_are_versioned_registered_and_documented() -> None:
    schema_registry = _load_yaml(SCHEMA_REGISTRY_PATH)
    search_body = SEARCH_CORE_PATH.read_text(encoding="utf-8")
    modeling_body = MODELING_PIPELINE_PATH.read_text(encoding="utf-8")
    contracts_body = CONTRACTS_MANIFESTS_PATH.read_text(encoding="utf-8")

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
        assert registry_entry["owner_ref"] == "shared_plus_local_decomposition_owner"
        assert registry_entry["owning_module"] == "shared_plus_local_decomposition"
        assert registry_entry["canonical_source_path"] == "docs/reference/contracts-manifests.md"

    assert "shared_plus_local_decomposition" in search_body
    assert "shared-plus-local fitting and unseen-entity constraints" in modeling_body
    assert "schemas/contracts" in contracts_body


def test_shared_local_decomposition_policy_profile_requires_all_reentry_contract_refs(
) -> None:
    reference_types = _load_yaml(REFERENCE_TYPES_PATH)
    profile = _profile(
        reference_types,
        "shared_plus_local_decomposition_policy_manifest@1.0.0",
    )

    expected_fields = {
        "typed_pooling_ref": "typed_pooling_manifest@1.0.0",
        "conditional_shared_local_codelength_ref": (
            "conditional_shared_local_codelength_contract@1.0.0"
        ),
        "exchangeability_ref": "exchangeability_manifest@1.0.0",
        "unseen_entity_prediction_policy_ref": (
            "unseen_entity_prediction_policy_manifest@1.0.0"
        ),
        "multi_entity_predictive_evaluation_ref": (
            "multi_entity_predictive_evaluation_manifest@1.0.0"
        ),
        "shared_local_freeze_refit_protocol_ref": (
            "shared_local_freeze_refit_protocol_manifest@1.0.0"
        ),
    }

    for path, schema_name in expected_fields.items():
        field = _field(profile, path)
        assert field["required"] is True
        assert field["cardinality"] == "exactly_one"
        assert field["allowed_schema_names"] == [schema_name]


def test_reference_docs_treat_shared_local_as_a_live_runtime_surface() -> None:
    search_body = SEARCH_CORE_PATH.read_text(encoding="utf-8")
    modeling_body = MODELING_PIPELINE_PATH.read_text(encoding="utf-8")
    readme_body = README_PATH.read_text(encoding="utf-8")
    source_map_text = SOURCE_MAP_PATH.read_text(encoding="utf-8")
    source_map = _load_yaml(SOURCE_MAP_PATH)

    assert "Composition families include:" in search_body
    assert "shared_plus_local_decomposition" in search_body
    assert "`modules/shared_plus_local_decomposition.py` handles panel-specific shared-plus-local fitting and unseen-entity constraints." in modeling_body
    assert "docs/reference/modeling-pipeline.md" in readme_body
    assert "docs/reference/search-core.md" in readme_body

    modules_targets = next(
        entry["canonical_targets"]
        for entry in source_map["entries"]
        if entry["source"] == "src/euclid/modules"
    )
    search_targets = next(
        entry["canonical_targets"]
        for entry in source_map["entries"]
        if entry["source"] == "src/euclid/search"
    )
    assert modules_targets == ["docs/reference/modeling-pipeline.md"]
    assert search_targets == ["docs/reference/search-core.md"]

    for deleted_path in (
        "docs/module-specs/hierarchical-modeling.md",
        "docs/architecture/data-contracts.md",
    ):
        assert deleted_path not in source_map_text
