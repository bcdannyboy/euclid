from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_TYPES_PATH = REPO_ROOT / "schemas/contracts/reference-types.yaml"
STOCHASTIC_MODEL_PATH = REPO_ROOT / "schemas/contracts/stochastic-law.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _field(profile: dict[str, Any], path: str) -> dict[str, Any]:
    for field in profile["fields"]:
        if field["path"] == path:
            return field
    raise AssertionError(f"missing reference field {path!r}")


def test_stochastic_model_contract_requires_residual_backed_production_evidence() -> None:
    payload = _load_yaml(STOCHASTIC_MODEL_PATH)

    assert payload["schema_name"] == "stochastic_model_manifest@1.0.0"
    assert set(payload["required_fields"]) >= {
        "candidate_id",
        "residual_history_ref",
        "observation_family",
        "residual_family",
        "support_kind",
        "horizon_scale_law",
        "fitted_parameters",
        "residual_count",
        "min_count_policy",
        "evidence_status",
        "production_evidence",
        "heuristic_gaussian_support",
        "replay_identity",
    }
    assert payload["production_evidence"]["requires_residual_history_ref"] is True
    assert payload["production_evidence"]["requires_residual_count_at_least"] == (
        "min_count_policy.minimum_residual_count"
    )
    assert payload["production_evidence"]["forbidden_when"] == {
        "heuristic_gaussian_support": True
    }


def test_stochastic_model_refs_residual_history_in_reference_registry() -> None:
    reference_types = _load_yaml(REFERENCE_TYPES_PATH)
    profiles = {
        profile["schema_name"]: profile
        for profile in reference_types["reference_profiles"]
    }

    stochastic_model = profiles["stochastic_model_manifest@1.0.0"]
    residual_history_ref = _field(stochastic_model, "residual_history_ref")

    assert residual_history_ref["required"] is False
    assert residual_history_ref["required_when"] == "evidence_status == production"
    assert residual_history_ref["allowed_schema_names"] == [
        "residual_history_manifest@1.0.0"
    ]


def test_heuristic_gaussian_support_is_compatibility_evidence_only() -> None:
    payload = _load_yaml(STOCHASTIC_MODEL_PATH)

    compatibility = payload["compatibility_evidence"]["heuristic_gaussian_support"]

    assert compatibility["allowed_evidence_status"] == "compatibility"
    assert compatibility["production_evidence"] is False
    assert compatibility["reason_code"] == "heuristic_gaussian_support_not_production"
    assert "heuristic_gaussian_scale_inflation" in payload[
        "forbidden_production_evidence"
    ]


def test_reproducibility_bundle_allows_residual_and_stochastic_required_refs() -> None:
    reference_types = _load_yaml(REFERENCE_TYPES_PATH)
    profile = next(
        profile
        for profile in reference_types["reference_profiles"]
        if profile["schema_name"] == "reproducibility_bundle_manifest@1.0.0"
    )
    required_refs = _field(profile, "required_manifest_refs[]")

    assert {
        "residual_history_manifest@1.0.0",
        "stochastic_model_manifest@1.0.0",
    } <= set(required_refs["allowed_schema_names"])
