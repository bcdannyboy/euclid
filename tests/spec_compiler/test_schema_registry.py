from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
SYSTEM_PATH = REPO_ROOT / "docs/reference/system.md"
CONTRACTS_MANIFESTS_PATH = REPO_ROOT / "docs/reference/contracts-manifests.md"
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"
MODULE_REGISTRY_PATH = REPO_ROOT / "schemas/contracts/module-registry.yaml"
SCHEMA_REGISTRY_PATH = REPO_ROOT / "schemas/contracts/schema-registry.yaml"
REFERENCE_TYPES_PATH = REPO_ROOT / "schemas/contracts/reference-types.yaml"
ENUM_REGISTRY_PATH = REPO_ROOT / "schemas/contracts/enum-registry.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _field(profile: dict, path: str) -> dict:
    for field in profile["fields"]:
        if field["path"] == path:
            return field
    raise AssertionError(f"missing ref field {path!r} for {profile['schema_name']}")


def test_schema_registry_closes_documented_manifest_inventory_with_unique_owners() -> None:
    payload = _load_yaml(SCHEMA_REGISTRY_PATH)
    module_registry = _load_yaml(MODULE_REGISTRY_PATH)
    valid_owner_ids = {owner["id"] for owner in module_registry["owners"]}

    assert payload["version"] == 1
    assert payload["kind"] == "schema_registry"

    schemas = payload["schemas"]
    assert isinstance(schemas, list) and schemas, "schema registry must declare schemas"

    schema_names = [entry["schema_name"] for entry in schemas]
    assert len(schema_names) == len(set(schema_names)), "schema names must be unique"
    assert {
        "scope_ledger_manifest@1.0.0",
        "search_plan_manifest@1.0.0",
        "prediction_artifact_manifest@1.1.0",
        "mechanistic_evidence_dossier_manifest@1.0.0",
        "publication_record_manifest@1.1.0",
    } <= set(schema_names)

    for entry in schemas:
        assert entry["owner_ref"] in valid_owner_ids, (
            f"unknown owner_ref for {entry['schema_name']}"
        )
        assert entry["owning_module"], (
            f"{entry['schema_name']} must declare an owning module"
        )
        assert entry["canonical_source_path"] == "docs/reference/contracts-manifests.md"


def test_reference_types_capture_required_optional_and_discriminated_refs() -> None:
    payload = _load_yaml(REFERENCE_TYPES_PATH)
    profiles = {
        profile["schema_name"]: profile for profile in payload["reference_profiles"]
    }

    assert payload["version"] == 1
    assert payload["kind"] == "reference_type_registry"
    assert payload["typed_ref_shape"]["required_fields"] == ["schema_name", "object_id"]
    assert "free_string_ref" in payload["typed_ref_shape"]["forbidden_placeholders"]

    algorithmic_dsl = profiles["algorithmic_dsl_manifest@1.1.0"]
    observational_equivalence = _field(
        algorithmic_dsl,
        "observational_equivalence_suite_ref",
    )
    assert observational_equivalence["required"] is False
    assert observational_equivalence["allowed_schema_names"] == [
        "observational_equivalence_suite_manifest@1.0.0"
    ]

    search_plan = profiles["search_plan_manifest@1.0.0"]
    assert _field(search_plan, "canonicalization_policy_ref")["required"] is True
    assert _field(search_plan, "observation_model_ref")["allowed_schema_names"] == [
        "observation_model_manifest@1.1.0"
    ]

    evaluation_event_log = profiles["evaluation_event_log_manifest@1.0.0"]
    related_object_ref = _field(evaluation_event_log, "events[].related_object_ref")
    assert related_object_ref["required"] is True
    assert related_object_ref["discriminator_field"] == "ref_kind"
    assert set(related_object_ref["allowed_schema_names_by_discriminator"]) == {
        "search_plan",
        "comparison_universe",
        "freeze_event",
        "frozen_shortlist",
        "prediction_artifact",
        "run_result",
    }

    validation_scope = profiles["validation_scope_manifest@1.0.0"]
    deferred_scope_refs = _field(validation_scope, "deferred_scope_refs[]")
    assert deferred_scope_refs["required"] is False
    assert deferred_scope_refs["cardinality"] == "zero_or_more"
    assert set(deferred_scope_refs["allowed_schema_names"]) == {
        "mechanistic_lane_policy_manifest@1.0.0",
        "shared_plus_local_decomposition_policy_manifest@1.0.0",
        "algorithmic_search_policy_manifest@1.0.0",
        "algorithmic_frontier_policy_manifest@1.0.0",
    }

    reproducibility_bundle = profiles["reproducibility_bundle_manifest@1.0.0"]
    required_manifest_refs = _field(reproducibility_bundle, "required_manifest_refs[]")
    assert required_manifest_refs["required"] is True
    assert required_manifest_refs["cardinality"] == "one_or_more"
    assert set(required_manifest_refs["allowed_schema_names"]) == {
        "prediction_artifact_manifest@1.1.0",
        "validation_scope_manifest@1.0.0",
        "point_score_result_manifest@1.0.0",
        "probabilistic_score_result_manifest@1.0.0",
        "calibration_result_manifest@1.0.0",
        "mechanistic_evidence_dossier_manifest@1.0.0",
        "external_evidence_manifest@1.0.0",
        "reducer_artifact_manifest@1.0.0",
        "scorecard_manifest@1.1.0",
        "claim_card_manifest@1.1.0",
        "abstention_manifest@1.1.0",
    }

    prediction_artifact = profiles["prediction_artifact_manifest@1.1.0"]
    score_policy_ref = _field(prediction_artifact, "score_policy_ref")
    assert score_policy_ref["required"] is True
    assert score_policy_ref["discriminator_field"] == "forecast_object_type"
    assert score_policy_ref["allowed_schema_names_by_discriminator"] == {
        "point": ["point_score_policy_manifest@1.0.0"],
        "distribution": ["probabilistic_score_policy_manifest@1.0.0"],
        "interval": ["interval_score_policy_manifest@1.0.0"],
        "quantile": ["quantile_score_policy_manifest@1.0.0"],
        "event_probability": ["event_probability_score_policy_manifest@1.0.0"],
    }

    run_result = profiles["run_result_manifest@1.1.0"]
    validation_scope_ref = _field(run_result, "primary_validation_scope_ref")
    assert validation_scope_ref["required"] is False
    assert validation_scope_ref["allowed_schema_names"] == [
        "validation_scope_manifest@1.0.0"
    ]


def test_enum_registry_tracks_core_vocabularies_and_typed_ref_closed_sets() -> None:
    payload = _load_yaml(ENUM_REGISTRY_PATH)
    enums = {entry["enum_name"]: entry for entry in payload["enums"]}

    assert payload["version"] == 1
    assert payload["kind"] == "enum_registry"

    claim_lanes = enums["claim_lanes"]
    claim_lane_values = {
        entry["id"]
        for entry in _load_yaml(REPO_ROOT / "schemas/core/claim-lanes.yaml")["entries"]
    }
    assert set(claim_lanes["allowed_values"]) == claim_lane_values

    forecast_object_types = enums["forecast_object_types"]
    forecast_object_type_values = {
        entry["id"]
        for entry in _load_yaml(REPO_ROOT / "schemas/core/forecast-object-types.yaml")[
            "entries"
        ]
    }
    assert set(forecast_object_types["allowed_values"]) == forecast_object_type_values

    assert set(enums["evaluation_event_related_object_kind"]["allowed_values"]) == {
        "search_plan",
        "comparison_universe",
        "freeze_event",
        "frozen_shortlist",
        "prediction_artifact",
        "run_result",
    }
    assert set(
        enums["validation_scope_deferred_scope_schema_names"]["allowed_values"]
    ) == {
        "mechanistic_lane_policy_manifest@1.0.0",
        "shared_plus_local_decomposition_policy_manifest@1.0.0",
        "algorithmic_search_policy_manifest@1.0.0",
        "algorithmic_frontier_policy_manifest@1.0.0",
    }
    assert set(enums["leakage_stage_evidence_kind"]["allowed_values"]) == {
        "feature_spec",
        "time_safety_audit",
        "evaluation_plan",
    }


def test_reference_docs_point_to_machine_readable_schema_and_ref_registries() -> None:
    contracts_doc = CONTRACTS_MANIFESTS_PATH.read_text(encoding="utf-8")
    system_doc = SYSTEM_PATH.read_text(encoding="utf-8")
    readme_doc = README_PATH.read_text(encoding="utf-8")
    source_map = _load_yaml(SOURCE_MAP_PATH)

    for required_path in (
        "schemas/contracts/schema-registry.yaml",
        "schemas/contracts/reference-types.yaml",
        "schemas/contracts/enum-registry.yaml",
    ):
        assert required_path in contracts_doc
        assert required_path in readme_doc or "docs/reference/contracts-manifests.md" in readme_doc

    assert "contracts-manifests.md" in system_doc
    contracts_targets = {
        tuple(entry["canonical_targets"])
        for entry in source_map["entries"]
        if entry["source"] in {"src/euclid/contracts", "src/euclid/manifests"}
    }
    assert contracts_targets == {("docs/reference/contracts-manifests.md",)}
