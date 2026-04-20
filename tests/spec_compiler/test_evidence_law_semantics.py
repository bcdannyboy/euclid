from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
SEARCH_CORE_PATH = REPO_ROOT / "docs/reference/search-core.md"
MODELING_PIPELINE_PATH = REPO_ROOT / "docs/reference/modeling-pipeline.md"
CONTRACTS_MANIFESTS_PATH = REPO_ROOT / "docs/reference/contracts-manifests.md"
SYSTEM_PATH = REPO_ROOT / "docs/reference/system.md"
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"

CONTRACT_PATHS = {
    "scoring": REPO_ROOT / "schemas/contracts/scoring.yaml",
    "calibration": REPO_ROOT / "schemas/contracts/calibration.yaml",
    "evaluation_governance": REPO_ROOT / "schemas/contracts/evaluation-governance.yaml",
    "robustness": REPO_ROOT / "schemas/contracts/robustness.yaml",
    "mechanistic_evidence": REPO_ROOT / "schemas/contracts/mechanistic-evidence.yaml",
    "shared_plus_local_evaluation": REPO_ROOT / "schemas/contracts/shared-plus-local-evaluation.yaml",
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


def test_reference_evidence_docs_cover_scoring_governance_robustness_and_mechanistic_flow() -> None:
    search_front_matter, search_body = _parse_front_matter(SEARCH_CORE_PATH)
    modeling_front_matter, modeling_body = _parse_front_matter(MODELING_PIPELINE_PATH)
    contracts_front_matter, contracts_body = _parse_front_matter(CONTRACTS_MANIFESTS_PATH)
    system_front_matter, system_body = _parse_front_matter(SYSTEM_PATH)
    readme_body = README_PATH.read_text(encoding="utf-8")

    assert search_front_matter["title"] == "Search Core"
    for required_string in {
        "shared_plus_local_decomposition",
        "law-eligible",
        "description_gain = reference_bits - total_code_bits",
    }:
        assert required_string in search_body

    assert modeling_front_matter["title"] == "Modeling Pipeline"
    for required_string in {
        "### 7. Evaluation and scoring",
        "### 8. Calibration, decision rules, and gates",
        "### 9. Claims, replay, and publication",
        "`modules/calibration.py` applies object-type-specific calibration policies and results.",
        "predictive gate policy",
        "mechanistic inputs",
        "`modules/claims.py` maps scorecards into claim cards or abstentions and caps interpretation scope.",
        "successful calibration",
    }:
        assert required_string in modeling_body

    assert contracts_front_matter["title"] == "Contracts And Manifests"
    for required_string in {
        "formal specification layer",
        "scorecards, claims, and abstentions",
        "reproducibility bundles",
        "run results and publication records",
    }:
        assert required_string in contracts_body

    assert system_front_matter["title"] == "System"
    for required_string in {
        "replayable experiments",
        "mechanistic evidence",
        "replay bundles",
        "publication records",
    }:
        assert required_string in system_body

    for required_string in {
        "calibration, robustness, and mechanistic evidence artifacts",
        "Publication is gated.",
    }:
        assert required_string in readme_body


def test_contract_artifacts_close_scoring_calibration_governance_robustness_and_mechanistic_sets() -> None:
    scoring = _load_yaml(CONTRACT_PATHS["scoring"])
    calibration = _load_yaml(CONTRACT_PATHS["calibration"])
    governance = _load_yaml(CONTRACT_PATHS["evaluation_governance"])
    robustness = _load_yaml(CONTRACT_PATHS["robustness"])
    mechanistic = _load_yaml(CONTRACT_PATHS["mechanistic_evidence"])
    shared_local = _load_yaml(CONTRACT_PATHS["shared_plus_local_evaluation"])

    assert scoring["kind"] == "scoring_policies"
    assert calibration["kind"] == "calibration_policies"
    assert governance["kind"] == "predictive_evidence_policies"
    assert robustness["kind"] == "robustness_policies"
    assert mechanistic["kind"] == "mechanistic_evidence_requirements"
    assert shared_local["kind"] == "shared_plus_local_evaluation_rules"

    assert {entry["forecast_object_type"] for entry in scoring["contracts"]} == {
        "point",
        "distribution",
        "interval",
        "quantile",
        "event_probability",
    }
    assert {entry["forecast_object_type"] for entry in calibration["contracts"]} == {
        "point",
        "distribution",
        "interval",
        "quantile",
        "event_probability",
    }
    assert {entry["comparison_regime_id"] for entry in governance["comparison_regimes"]} == {
        "unconditional_model_pair",
        "conditional_model_pair",
    }
    assert {entry["adjustment_id"] for entry in governance["many_model_adjustments"]} == {
        "none",
        "hansen_spa",
        "fresh_replication_only",
    }
    assert {entry["family_id"] for entry in robustness["null_families"]} == {"iid_permutation_null"}
    assert {entry["family_id"] for entry in robustness["perturbation_families"]} == {
        "recent_history_truncation",
        "quantization_coarsening",
    }
    assert {entry["canary_type"] for entry in robustness["leakage_canary_types"]} == {
        "future_target_level_feature",
        "late_available_target_copy",
        "holdout_membership_feature",
        "post_cutoff_revision_level_feature",
    }
    assert {entry["requirement_id"] for entry in mechanistic["contracts"]} == {
        "mechanism_mapping",
        "units_consistency",
        "invariance_or_intervention_support",
        "external_evidence_bundle",
        "evidence_independence_attestation",
    }
    assert {entry["rule_id"] for entry in shared_local["contracts"]} == {
        "entity_panel_comparability",
        "shared_component_fit_scope",
        "local_component_fit_scope",
        "cross_entity_aggregation",
        "unseen_entity_publication",
    }


def test_contract_contents_lock_core_policy_choices() -> None:
    scoring = _load_yaml(CONTRACT_PATHS["scoring"])
    calibration = _load_yaml(CONTRACT_PATHS["calibration"])
    governance = _load_yaml(CONTRACT_PATHS["evaluation_governance"])
    robustness = _load_yaml(CONTRACT_PATHS["robustness"])
    mechanistic = _load_yaml(CONTRACT_PATHS["mechanistic_evidence"])
    shared_local = _load_yaml(CONTRACT_PATHS["shared_plus_local_evaluation"])

    scoring_entries = {entry["forecast_object_type"]: entry for entry in scoring["contracts"]}
    assert scoring_entries["point"]["allowed_primary_scores"] == ["squared_error", "absolute_error"]
    assert scoring_entries["distribution"]["allowed_primary_scores"] == [
        "log_score",
        "continuous_ranked_probability_score",
    ]
    assert scoring_entries["interval"]["allowed_primary_scores"] == ["interval_score"]
    assert scoring_entries["quantile"]["allowed_primary_scores"] == ["pinball_loss"]
    assert scoring_entries["event_probability"]["allowed_primary_scores"] == ["brier_score", "log_score"]

    calibration_entries = {entry["forecast_object_type"]: entry for entry in calibration["contracts"]}
    assert calibration_entries["point"]["calibration_mode"] == "not_applicable"
    assert calibration_entries["distribution"]["calibration_mode"] == "required"
    assert calibration_entries["distribution"]["required_diagnostics"] == [
        "pit_or_randomized_pit_uniformity"
    ]
    assert calibration_entries["interval"]["required_diagnostics"] == ["nominal_coverage"]
    assert calibration_entries["quantile"]["required_diagnostics"] == ["quantile_hit_balance"]
    assert calibration_entries["event_probability"]["required_diagnostics"] == [
        "reliability_curve_or_binned_frequency"
    ]

    gate_entries = {entry["policy_id"]: entry for entry in governance["predictive_gate_policies"]}
    assert set(gate_entries) == {"point_predictive_gate", "probabilistic_predictive_gate"}
    assert gate_entries["point_predictive_gate"]["allowed_forecast_object_types"] == ["point"]
    assert gate_entries["probabilistic_predictive_gate"]["allowed_forecast_object_types"] == [
        "distribution",
        "interval",
        "quantile",
        "event_probability",
    ]
    assert gate_entries["probabilistic_predictive_gate"]["requires_calibration_pass"] is True

    assert robustness["global_rules"]["leakage_canaries_are_structural"] is True
    assert mechanistic["global_rules"]["requires_lower_lane_support"] == "predictively_supported"
    assert mechanistic["global_rules"]["missing_requirement_effect"] == (
        "downgrade_to_predictively_supported"
    )
    assert shared_local["global_rules"]["entity_index_set_required"] is True
    assert shared_local["global_rules"]["unseen_entity_publication_requires_separate_contract"] is True


def test_reference_spine_routes_evidence_surfaces_to_live_docs() -> None:
    source_map = _load_yaml(SOURCE_MAP_PATH)
    readme_body = README_PATH.read_text(encoding="utf-8")

    assert "docs/reference/modeling-pipeline.md" in readme_body
    assert "docs/reference/search-core.md" in readme_body
    assert "docs/reference/contracts-manifests.md" in readme_body
    assert _canonical_targets(source_map, "src/euclid/modules") == {
        "docs/reference/modeling-pipeline.md"
    }
    assert _canonical_targets(source_map, "src/euclid/search") == {
        "docs/reference/search-core.md"
    }
    assert _canonical_targets(source_map, "src/euclid/contracts") == {
        "docs/reference/contracts-manifests.md"
    }
    assert _canonical_targets(source_map, "src/euclid/manifests") == {
        "docs/reference/contracts-manifests.md"
    }
