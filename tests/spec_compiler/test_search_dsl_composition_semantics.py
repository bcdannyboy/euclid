from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
REFERENCE_INDEX_PATH = REPO_ROOT / "docs/reference/README.md"
SEARCH_CORE_PATH = REPO_ROOT / "docs/reference/search-core.md"
MODELING_PIPELINE_PATH = REPO_ROOT / "docs/reference/modeling-pipeline.md"
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"
ALGORITHMIC_DSL_CONTRACT_PATH = REPO_ROOT / "schemas/contracts/algorithmic-dsl.yaml"
SEARCH_CLASSES_CONTRACT_PATH = REPO_ROOT / "schemas/contracts/search-classes.yaml"
SEARCH_FRONTIER_CONTRACT_PATH = REPO_ROOT / "schemas/contracts/search-frontier.yaml"
REWRITE_GUARANTEES_CONTRACT_PATH = REPO_ROOT / "schemas/contracts/rewrite-guarantees.yaml"
COMPOSITION_SEMANTICS_CONTRACT_PATH = REPO_ROOT / "schemas/contracts/composition-semantics.yaml"

EXPECTED_SEARCH_CLASSES = {
    "exact_finite_enumeration",
    "bounded_heuristic",
    "equality_saturation_heuristic",
    "stochastic_heuristic",
}
EXPECTED_COMPOSITION_OPERATORS = {
    "piecewise",
    "additive_residual",
    "regime_conditioned",
    "shared_plus_local_decomposition",
}
DELETED_DOC_PATHS = {
    "docs/canonical/math/search-and-derivation.md",
    "docs/module-specs/algorithmic-dsl.md",
    "docs/module-specs/dsl-semantics.md",
    "docs/module-specs/search-planning.md",
    "docs/module-specs/hierarchical-modeling.md",
}
FORBIDDEN_LEGACY_PHRASES = {
    "deferred exploratory only",
    "deferred_non_binding",
    "boundary only",
    "not part of v1",
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


def test_search_core_doc_exists_with_live_sections_and_semantics() -> None:
    front_matter, body = _parse_front_matter(SEARCH_CORE_PATH)

    assert front_matter["title"] == "Search Core"
    assert {
        "system.md",
        "modeling-pipeline.md",
        "contracts-manifests.md",
    } <= set(front_matter["related"])

    for section in {
        "## Search classes",
        "## Candidate Intermediate Representation",
        "## Reducer families and compositions",
        "## Description gain and comparison classes",
        "## Algorithmic DSL",
    }:
        assert section in body

    for required_string in {
        "exact_finite_enumeration",
        "bounded_heuristic",
        "equality_saturation_heuristic",
        "stochastic_heuristic",
        "algorithmic",
        "piecewise",
        "additive_residual",
        "regime_conditioned",
        "shared_plus_local_decomposition",
        "bounded proposal generation",
        "candidate normalization",
        "replay-aware backend disclosures",
        "Stable aliases such as `algorithmic_last_observation`",
    }:
        assert required_string in body

    lowered = body.lower()
    for phrase in FORBIDDEN_LEGACY_PHRASES:
        assert phrase not in lowered, f"search reference doc must replace legacy phrase: {phrase}"


def test_search_and_composition_contracts_exist_with_expected_shapes() -> None:
    algorithmic_dsl = _load_yaml(ALGORITHMIC_DSL_CONTRACT_PATH)
    search_classes = _load_yaml(SEARCH_CLASSES_CONTRACT_PATH)
    search_frontier = _load_yaml(SEARCH_FRONTIER_CONTRACT_PATH)
    rewrite_guarantees = _load_yaml(REWRITE_GUARANTEES_CONTRACT_PATH)
    composition_semantics = _load_yaml(COMPOSITION_SEMANTICS_CONTRACT_PATH)

    assert algorithmic_dsl["kind"] == "algorithmic_dsl_semantics"
    assert search_classes["kind"] == "search_class_semantics"
    assert search_frontier["kind"] == "search_frontier_semantics"
    assert rewrite_guarantees["kind"] == "rewrite_guarantee_semantics"
    assert composition_semantics["kind"] == "composition_operator_semantics"

    assert {entry["search_class"] for entry in search_classes["contracts"]} == EXPECTED_SEARCH_CLASSES
    assert {
        entry["composition_operator"] for entry in composition_semantics["contracts"]
    } == EXPECTED_COMPOSITION_OPERATORS


def test_search_class_contract_closes_disclosure_frontier_and_exactness_ceiling_rules() -> None:
    payload = _load_yaml(SEARCH_CLASSES_CONTRACT_PATH)

    assert payload["global_rules"]["heuristic_classes_require_disclosure"] is True
    assert payload["global_rules"]["frontier_contract"] == "schemas/contracts/search-frontier.yaml"
    assert payload["global_rules"]["exactness_claim_scope"] == "declared_canonical_program_space_only"

    entries = {entry["search_class"]: entry for entry in payload["contracts"]}

    exact = entries["exact_finite_enumeration"]
    assert exact["coverage_statement"] == "complete_over_declared_canonical_program_space"
    assert exact["exactness_ceiling"] == "exact_over_declared_fragment_only"
    assert exact["requires_finite_canonical_space_proof"] is True

    for search_class in {
        "bounded_heuristic",
        "equality_saturation_heuristic",
        "stochastic_heuristic",
    }:
        entry = entries[search_class]
        assert entry["coverage_statement"] == "incomplete_search_disclosed"
        assert entry["exactness_ceiling"] == "no_global_exactness_claim"
        assert entry["requires_disclosure"], f"{search_class} must declare required disclosure fields"


def test_rewrite_guarantee_contract_separates_canonicalization_from_semantic_identity() -> None:
    payload = _load_yaml(REWRITE_GUARANTEES_CONTRACT_PATH)

    assert payload["global_rules"]["canonicalization_identity_scope"] == "normalization_pipeline_output_only"
    assert payload["global_rules"]["observational_equivalence_identity_claim"] == (
        "heuristic_duplicate_screen_only"
    )
    assert payload["global_rules"]["equality_saturation_not_global_identity"] is True

    guarantees = {entry["guarantee_id"]: entry for entry in payload["contracts"]}
    assert guarantees["canonical_serialization"]["guarantee_class"] == "deterministic_normal_form"
    assert guarantees["observational_equivalence"]["guarantee_class"] == "finite_suite_duplicate_screen"
    assert guarantees["observational_equivalence"]["identity_claim"] == "heuristic_duplicate_screen_only"


def test_algorithmic_dsl_and_composition_contracts_close_hidden_rule_gaps() -> None:
    algorithmic_dsl = _load_yaml(ALGORITHMIC_DSL_CONTRACT_PATH)
    composition_semantics = _load_yaml(COMPOSITION_SEMANTICS_CONTRACT_PATH)

    assert algorithmic_dsl["global_rules"]["value_domain"] == "exact_rational"
    assert algorithmic_dsl["global_rules"]["loop_support"] == "forbidden"
    assert algorithmic_dsl["global_rules"]["recursion_support"] == "forbidden"
    assert algorithmic_dsl["global_rules"]["hidden_randomness"] == "forbidden"
    assert algorithmic_dsl["global_rules"]["observational_equivalence_contract"] == (
        "schemas/contracts/rewrite-guarantees.yaml"
    )

    composition_entries = {entry["composition_operator"]: entry for entry in composition_semantics["contracts"]}

    piecewise = composition_entries["piecewise"]
    assert piecewise["combiner_mode"] == "segment_partition"
    assert "ordered_partition" in piecewise["required_objects"]

    additive = composition_entries["additive_residual"]
    assert additive["combiner_mode"] == "base_plus_residual"
    assert "base_reducer" in additive["required_objects"]
    assert "residual_reducer" in additive["required_objects"]

    regime = composition_entries["regime_conditioned"]
    assert regime["combiner_mode"] == "gated_branch_selection_or_weighting"
    assert regime["requires_explicit_gating_law"] is True
    assert regime["hidden_rule_forbidden"] == "implicit_regime_switching"

    shared_local = composition_entries["shared_plus_local_decomposition"]
    assert shared_local["combiner_mode"] == "shared_and_entity_local"
    assert shared_local["requires_entity_index_set"] is True
    assert shared_local["requires_unseen_entity_rule"] is True
    assert shared_local["singleton_entity_behavior"] == "degenerate_one_entity_case"


def test_reference_spine_points_search_surfaces_at_live_reference_docs() -> None:
    source_map = _load_yaml(SOURCE_MAP_PATH)
    source_map_text = SOURCE_MAP_PATH.read_text(encoding="utf-8")
    readme_body = README_PATH.read_text(encoding="utf-8")
    reference_index_body = REFERENCE_INDEX_PATH.read_text(encoding="utf-8")
    modeling_front_matter, _ = _parse_front_matter(MODELING_PIPELINE_PATH)

    assert _canonical_targets(source_map, "README.md") >= {
        "docs/reference/search-core.md",
        "docs/reference/modeling-pipeline.md",
    }
    for source in (
        "src/euclid/search",
        "src/euclid/cir",
        "src/euclid/reducers",
        "src/euclid/adapters",
        "src/euclid/math",
    ):
        assert _canonical_targets(source_map, source) == {"docs/reference/search-core.md"}
    assert _canonical_targets(source_map, "src/euclid/modules") == {
        "docs/reference/modeling-pipeline.md"
    }

    assert "docs/reference/search-core.md" in readme_body
    assert "search-core.md" in reference_index_body
    assert "search-core.md" in modeling_front_matter["related"]

    for deleted_path in DELETED_DOC_PATHS:
        assert deleted_path not in source_map_text
