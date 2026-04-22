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
SYSTEM_PATH = REPO_ROOT / "docs/reference/system.md"
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"

DELETED_DOC_FRAGMENTS = {
    "docs/canonical/math/",
    "docs/module-specs/",
    "docs/architecture/",
}


def _parse_front_matter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    assert match, f"{path.relative_to(REPO_ROOT).as_posix()} must start with YAML front matter"
    return yaml.safe_load(match.group(1)), match.group(2)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _canonical_targets(payload: dict[str, Any], source: str) -> set[str]:
    entries = payload["entries"]
    for entry in entries:
        if entry["source"] == source:
            return set(entry["canonical_targets"])
    raise AssertionError(f"missing source-map entry for {source}")


def test_reference_math_core_docs_exist_with_live_front_matter_and_sections() -> None:
    search_front_matter, search_body = _parse_front_matter(SEARCH_CORE_PATH)
    modeling_front_matter, modeling_body = _parse_front_matter(MODELING_PIPELINE_PATH)

    assert search_front_matter["title"] == "Search Core"
    assert {
        "system.md",
        "modeling-pipeline.md",
        "contracts-manifests.md",
    } <= set(search_front_matter["related"])

    for section in {
        "## Search classes",
        "## Candidate Intermediate Representation",
        "## Reducer families and compositions",
        "## Description gain and comparison classes",
        "## Algorithmic DSL",
    }:
        assert section in search_body

    for required_string in {
        "exact_finite_enumeration",
        "bounded_heuristic",
        "equality_saturation_heuristic",
        "stochastic_heuristic",
        "analytic",
        "recursive",
        "spectral",
        "algorithmic",
        "piecewise",
        "additive_residual",
        "regime_conditioned",
        "shared_plus_local_decomposition",
        "description_gain = reference_bits - total_code_bits",
        "candidate normalization",
    }:
        assert required_string in search_body

    assert modeling_front_matter["title"] == "Modeling Pipeline"
    assert {
        "system.md",
        "search-core.md",
        "contracts-manifests.md",
    } <= set(modeling_front_matter["related"])

    for section in {
        "## Stage map",
        "### 1. Ingestion",
        "### 2. Snapshotting and timeguard",
        "### 5. Search planning",
        "### 6. Candidate fitting",
        "### 7. Evaluation and scoring",
        "### 8. Calibration, decision rules, and gates",
        "### 9. Claims, replay, and publication",
    }:
        assert section in modeling_body

    for required_string in {
        "`src/euclid/modules/shared_plus_local_decomposition.py` handles panel-specific shared-plus-local fitting and unseen-entity constraints.",
        "Cross-object comparisons are invalid by design.",
        "Publication requires replay-verifiable bundles.",
        "distribution",
        "interval",
        "quantile",
        "event_probability",
    }:
        assert required_string in modeling_body


def test_reference_math_core_spine_routes_search_and_modeling_surfaces() -> None:
    source_map = _load_yaml(SOURCE_MAP_PATH)

    assert source_map["reference_workspace"]["docs_root"] == "docs/reference"
    assert {
        "docs/reference/search-core.md",
        "docs/reference/modeling-pipeline.md",
        "docs/reference/system.md",
    } <= _canonical_targets(source_map, "README.md")

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


def test_reference_math_core_docs_do_not_route_through_deleted_doc_sets() -> None:
    source_map_text = SOURCE_MAP_PATH.read_text(encoding="utf-8")
    readme_text = README_PATH.read_text(encoding="utf-8")
    reference_index_text = REFERENCE_INDEX_PATH.read_text(encoding="utf-8")

    for deleted_fragment in DELETED_DOC_FRAGMENTS:
        assert deleted_fragment not in source_map_text
        assert deleted_fragment not in readme_text
        assert deleted_fragment not in reference_index_text

    assert "docs/reference/search-core.md" in readme_text
    assert "docs/reference/modeling-pipeline.md" in readme_text
    assert "docs/reference/system.md" in readme_text
    assert "search-core.md" in reference_index_text
    assert "modeling-pipeline.md" in reference_index_text
