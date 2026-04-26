from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_REGISTRY_PATH = REPO_ROOT / "schemas/contracts/schema-registry.yaml"
REFERENCE_TYPES_PATH = REPO_ROOT / "schemas/contracts/reference-types.yaml"
RESIDUAL_HISTORY_PATH = REPO_ROOT / "schemas/contracts/residual-history.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_residual_history_manifest_is_declared_in_schema_and_ref_registries() -> None:
    schema_registry = _load_yaml(SCHEMA_REGISTRY_PATH)
    reference_types = _load_yaml(REFERENCE_TYPES_PATH)

    schema_entry = next(
        (
            entry
            for entry in schema_registry["schemas"]
            if entry["schema_name"] == "residual_history_manifest@1.0.0"
        ),
        None,
    )
    assert schema_entry == {
        "schema_name": "residual_history_manifest@1.0.0",
        "owner_ref": "candidate_fitting_owner",
        "owning_module": "candidate_fitting",
        "canonical_source_path": "docs/reference/contracts-manifests.md",
    }

    profile = next(
        (
            profile
            for profile in reference_types["reference_profiles"]
            if profile["schema_name"] == "residual_history_manifest@1.0.0"
        ),
        None,
    )
    assert profile is not None
    assert profile["owner_ref"] == "candidate_fitting_owner"


def test_residual_history_contract_requires_replayable_row_geometry_and_digest() -> None:
    payload = _load_yaml(RESIDUAL_HISTORY_PATH)

    assert payload["schema_name"] == "residual_history_manifest@1.0.0"
    assert payload["kind"] == "contract_schema"
    assert set(payload["required_fields"]) >= {
        "candidate_id",
        "fit_window_id",
        "residual_rows",
        "residual_row_count",
        "residual_history_digest",
        "residual_basis",
        "construction_policy",
        "replay_identity",
    }
    assert set(payload["residual_row_schema"]["required_fields"]) >= {
        "candidate_id",
        "fit_window_id",
        "origin_index",
        "origin_time",
        "origin_available_at",
        "target_index",
        "target_event_time",
        "target_available_at",
        "entity",
        "realized_value",
        "point_forecast",
        "residual",
        "split_role",
        "replay_identity",
    }
    assert set(payload["digest"]["required_fields"]) >= {
        "residual_history_digest",
        "residual_row_count",
        "residual_basis",
        "construction_policy",
        "source_row_set_digest",
    }


def test_residual_history_contract_rejects_missing_split_role_metadata() -> None:
    payload = _load_yaml(RESIDUAL_HISTORY_PATH)

    rejection = payload["validation_rules"]["missing_split_role_metadata"]

    assert rejection["required_field"] == "residual_rows[].split_role"
    assert rejection["effect"] == "reject"
    assert rejection["reason_code"] == "missing_split_role_metadata"


def test_residual_history_contract_rejects_missing_origin_and_target_availability(
) -> None:
    payload = _load_yaml(RESIDUAL_HISTORY_PATH)

    rejection = payload["validation_rules"]["missing_time_availability_metadata"]

    assert set(rejection["required_fields"]) == {
        "residual_rows[].origin_available_at",
        "residual_rows[].target_available_at",
    }
    assert rejection["effect"] == "reject"
    assert rejection["reason_code"] == "missing_origin_or_target_availability"
