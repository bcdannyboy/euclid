from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
MODELING_PIPELINE_PATH = REPO_ROOT / "docs/reference/modeling-pipeline.md"
SYSTEM_PATH = REPO_ROOT / "docs/reference/system.md"
CONTRACTS_MANIFESTS_PATH = REPO_ROOT / "docs/reference/contracts-manifests.md"
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"
CLAIM_LANES_CONTRACT_PATH = REPO_ROOT / "schemas/contracts/claim-lanes.yaml"
FORECAST_OBJECT_TYPES_CONTRACT_PATH = REPO_ROOT / "schemas/contracts/forecast-object-types.yaml"
ABSTENTION_TYPES_CONTRACT_PATH = REPO_ROOT / "schemas/contracts/abstention-types.yaml"

EXPECTED_LANES = {
    "descriptive_structure",
    "predictive_within_declared_scope",
    "mechanistically_compatible_law",
}
EXPECTED_FORECAST_OBJECT_TYPES = {
    "point",
    "distribution",
    "interval",
    "quantile",
    "event_probability",
}
EXPECTED_ABSTENTION_TYPES = {
    "no_admissible_reducer",
    "codelength_comparability_failed",
    "robustness_failed",
}


def _parse_front_matter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    assert match, f"{path.relative_to(REPO_ROOT).as_posix()} must start with YAML front matter"
    return yaml.safe_load(match.group(1)), match.group(2)


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing required file: {path.relative_to(REPO_ROOT).as_posix()}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _canonical_targets(payload: dict[str, Any], source: str) -> set[str]:
    for entry in payload["entries"]:
        if entry["source"] == source:
            return set(entry["canonical_targets"])
    raise AssertionError(f"missing source-map entry for {source}")


def test_claim_output_reference_docs_describe_publication_axes_and_artifacts() -> None:
    modeling_front_matter, modeling_body = _parse_front_matter(MODELING_PIPELINE_PATH)
    system_front_matter, system_body = _parse_front_matter(SYSTEM_PATH)
    contracts_front_matter, contracts_body = _parse_front_matter(CONTRACTS_MANIFESTS_PATH)
    readme_body = README_PATH.read_text(encoding="utf-8")

    assert modeling_front_matter["title"] == "Modeling Pipeline"
    assert {
        "system.md",
        "search-core.md",
        "contracts-manifests.md",
    } <= set(modeling_front_matter["related"])
    for required_string in {
        "`modules/claims.py` maps scorecards into claim cards or abstentions and caps interpretation scope.",
        "Point and non-point outputs are separate publication lanes.",
        "Non-point outputs require forecast-object-specific evaluation and, for predictive promotion, successful calibration.",
        "distribution",
        "interval",
        "quantile",
        "event_probability",
    }:
        assert required_string in modeling_body

    assert system_front_matter["title"] == "System"
    for required_string in {
        "scorecards, claim or abstention resolution, replay bundles, run results, publication records",
        "Resolve a scorecard into a claim card or abstention.",
        "Build and verify a reproducibility bundle.",
    }:
        assert required_string in system_body

    assert contracts_front_matter["title"] == "Contracts And Manifests"
    for required_string in {
        "scorecards, claims, and abstentions",
        "reproducibility bundles",
        "run results and publication records",
        "typed refs in the body conform to the contract catalog",
    }:
        assert required_string in contracts_body

    for required_string in {
        "abstentions",
        "scorecards, claim cards, and abstentions",
        "Publication is gated.",
    }:
        assert required_string in readme_body


def test_claim_output_contracts_exist_and_cover_lane_forecast_and_abstention_sets() -> None:
    claim_lane_contracts = _load_yaml(CLAIM_LANES_CONTRACT_PATH)
    forecast_object_contracts = _load_yaml(FORECAST_OBJECT_TYPES_CONTRACT_PATH)
    abstention_contracts = _load_yaml(ABSTENTION_TYPES_CONTRACT_PATH)

    assert claim_lane_contracts["kind"] == "claim_lane_semantics"
    assert forecast_object_contracts["kind"] == "forecast_object_type_semantics"
    assert abstention_contracts["kind"] == "abstention_type_semantics"

    assert {entry["claim_lane"] for entry in claim_lane_contracts["contracts"]} == EXPECTED_LANES
    assert {entry["forecast_object_type"] for entry in forecast_object_contracts["contracts"]} == (
        EXPECTED_FORECAST_OBJECT_TYPES
    )
    assert {entry["abstention_type"] for entry in abstention_contracts["contracts"]} == EXPECTED_ABSTENTION_TYPES


def test_claim_lane_contract_closes_meaning_evidence_publication_and_abstention_behavior() -> None:
    payload = _load_yaml(CLAIM_LANES_CONTRACT_PATH)

    assert payload["meaning_axis"] == "claim_lane"
    assert payload["global_rules"]["predictive_nonpromotion_after_descriptive_success"] == (
        "downgrade_to_descriptive_structure_without_abstention"
    )
    assert payload["global_rules"]["probabilistic_meaning_axis"] == "forecast_object_type"
    assert payload["global_rules"]["mechanistic_meaning_axis"] == "claim_lane_with_external_evidence"

    entries = {entry["claim_lane"]: entry for entry in payload["contracts"]}

    descriptive = entries["descriptive_structure"]
    assert "descriptive_compression" in descriptive["required_evidence_classes"]
    assert "historical_structure_summary" in descriptive["allowed_to_say"]
    assert "prediction_artifact" in descriptive["may_publish"]
    assert "probabilistic_forecast_claim" in descriptive["forbidden_without_upgrade"]
    assert descriptive["abstention_interaction"] == "typed_abstention_only_when_descriptive_publication_fails"

    predictive = entries["predictive_within_declared_scope"]
    assert "predictive_generalization" in predictive["required_evidence_classes"]
    assert "point_forecast_within_declared_validation_scope" in predictive["allowed_to_say"]
    assert "claim_card" in predictive["may_publish"]
    assert predictive["abstention_interaction"] == "downgrade_to_descriptive_structure_when_predictive_support_fails"

    mechanistic = entries["mechanistically_compatible_law"]
    assert "boundary_specific_external_evidence" in mechanistic["required_evidence_classes"]
    assert mechanistic["requires_lower_lane_support"] == "predictive_within_declared_scope"
    assert "mechanism_claim" in mechanistic["allowed_to_say"]
    assert mechanistic["publication_status"] == "requires_bound_external_evidence_contracts"


def test_forecast_object_contract_closes_point_and_probabilistic_output_meaning() -> None:
    payload = _load_yaml(FORECAST_OBJECT_TYPES_CONTRACT_PATH)

    assert payload["meaning_axis"] == "forecast_object_type"
    assert payload["global_rules"]["claim_lane_axis"] == "claim_lane"
    assert payload["global_rules"]["probabilistic_objects_require_type_matched_calibration"] is True

    entries = {entry["forecast_object_type"]: entry for entry in payload["contracts"]}

    point = entries["point"]
    assert point["minimum_claim_lane"] == "descriptive_structure"
    assert point["semantic_family"] == "point_functional"
    assert point["calibration_mode"] == "not_applicable_for_point_only_publication"
    assert "location_parameter" in point["meaning"]

    for probabilistic_id in {"distribution", "interval", "quantile", "event_probability"}:
        entry = entries[probabilistic_id]
        assert entry["minimum_claim_lane"] == "predictive_within_declared_scope"
        assert entry["semantic_family"] == "probabilistic"
        assert entry["calibration_mode"] == "required"
        assert entry["requires_contracts"], f"{probabilistic_id} must declare required contracts"


def test_abstention_type_contract_closes_blocking_behavior_without_lane_confusion() -> None:
    payload = _load_yaml(ABSTENTION_TYPES_CONTRACT_PATH)

    assert payload["global_rules"]["result_mode_for_abstention"] == "abstention_only_publication"
    assert payload["global_rules"]["predictive_nonpromotion_is_not_an_abstention"] is True

    entries = {entry["abstention_type"]: entry for entry in payload["contracts"]}

    for abstention_type, entry in entries.items():
        assert set(entry["blocked_claim_lanes"]) == EXPECTED_LANES
        assert entry["result_mode"] == "abstention_only_publication"
        assert entry["reason_codes"], f"{abstention_type} must declare reason codes"

    assert "descriptive_gate_failed" in entries["no_admissible_reducer"]["reason_codes"]
    assert "codelength_comparability_failed" in entries["codelength_comparability_failed"]["reason_codes"]
    assert "robustness_failed" in entries["robustness_failed"]["reason_codes"]


def test_claim_output_reference_spine_routes_live_claim_surfaces() -> None:
    source_map = _load_yaml(SOURCE_MAP_PATH)
    readme_body = README_PATH.read_text(encoding="utf-8")
    system_front_matter, _ = _parse_front_matter(SYSTEM_PATH)

    assert "docs/reference/modeling-pipeline.md" in readme_body
    assert "docs/reference/contracts-manifests.md" in readme_body
    assert {"modeling-pipeline.md", "contracts-manifests.md"} <= set(
        system_front_matter["related"]
    )
    assert _canonical_targets(source_map, "src/euclid/modules") == {
        "docs/reference/modeling-pipeline.md"
    }
    assert _canonical_targets(source_map, "src/euclid/contracts") == {
        "docs/reference/contracts-manifests.md"
    }
    assert _canonical_targets(source_map, "src/euclid/manifests") == {
        "docs/reference/contracts-manifests.md"
    }
