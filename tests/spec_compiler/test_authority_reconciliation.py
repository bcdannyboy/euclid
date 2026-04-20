from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RECONCILIATION_PATH = REPO_ROOT / "docs/implementation/authority-reconciliation.yaml"
AUTHORITY_SNAPSHOT_PATH = REPO_ROOT / "docs/implementation/authority-snapshot.yaml"
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"
CORE_VOCABULARY_AUTHORITY_REFS = {
    "reducer-families.yaml": {
        "README.md",
        "docs/reference/search-core.md",
    },
    "composition-operators.yaml": {
        "README.md",
        "docs/reference/search-core.md",
    },
    "claim-lanes.yaml": {
        "README.md",
        "docs/reference/search-core.md",
        "docs/reference/modeling-pipeline.md",
    },
    "forecast-object-types.yaml": {
        "README.md",
        "docs/reference/search-core.md",
        "docs/reference/modeling-pipeline.md",
    },
    "evidence-classes.yaml": {
        "README.md",
        "docs/reference/contracts-manifests.md",
        "docs/reference/modeling-pipeline.md",
    },
    "abstention-types.yaml": {
        "README.md",
        "docs/reference/search-core.md",
        "docs/reference/modeling-pipeline.md",
    },
    "scope-axes.yaml": {
        "README.md",
        "docs/reference/search-core.md",
        "docs/reference/modeling-pipeline.md",
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), (
        "missing required file: " f"{path.relative_to(REPO_ROOT).as_posix()}"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _core_schema_paths(filename: str) -> tuple[Path, Path]:
    return (
        REPO_ROOT / "schemas/core" / filename,
        REPO_ROOT / "src/euclid/_assets/schemas/core" / filename,
    )


def test_every_normative_authority_surface_is_mapped_or_explicitly_flagged_for_matrix_expansion(
) -> None:
    payload = _load_yaml(RECONCILIATION_PATH)
    allowed_statuses = set(payload["allowed_statuses"])
    entries = payload["entries"]

    assert payload["source_authority_refs"]
    assert len({entry["concept_id"] for entry in entries}) == len(entries)

    for entry in entries:
        assert entry["status"] in allowed_statuses
        assert entry["authority_refs"]
        if entry["status"] == "mapped":
            assert entry["mapped_row_ids"]
            continue
        if entry["status"] == "requires_matrix_expansion":
            assert entry["required_new_row_ids"]
            continue
        assert entry["status"] == "excluded"
        assert entry["exclusion_reason"] in set(payload["allowed_exclusion_reasons"])


def test_live_authority_docs_route_through_readme_and_reference_workspace() -> None:
    snapshot = _load_yaml(AUTHORITY_SNAPSHOT_PATH)
    source_map = _load_yaml(SOURCE_MAP_PATH)
    scope_docs = {entry["path"] for entry in snapshot["scope_authority_docs"]}

    assert {
        "README.md",
        "docs/reference/system.md",
        "docs/reference/contracts-manifests.md",
        "docs/reference/benchmarks-readiness.md",
    } <= scope_docs
    assert "EUCLID.md" not in scope_docs
    assert not any(path.startswith("docs/canonical/") for path in scope_docs)

    readme_entry = next(
        entry for entry in source_map["entries"] if entry["source"] == "README.md"
    )
    assert {
        "docs/reference/README.md",
        "docs/reference/system.md",
        "docs/reference/modeling-pipeline.md",
        "docs/reference/search-core.md",
        "docs/reference/contracts-manifests.md",
        "docs/reference/benchmarks-readiness.md",
        "docs/reference/testing-truthfulness.md",
    } <= set(readme_entry["canonical_targets"])


def test_required_matrix_expansions_are_concrete_and_named() -> None:
    payload = _load_yaml(RECONCILIATION_PATH)
    expansion_rows = {
        row_id
        for entry in payload["entries"]
        if entry["status"] == "requires_matrix_expansion"
        for row_id in entry["required_new_row_ids"]
    }

    assert {
        "evaluation_surface:score_law_comparability",
        "evaluation_surface:horizon_weight_comparability",
        "evaluation_surface:scored_origin_set_comparability",
        "evaluation_surface:descriptive_floor_abstention",
        "evaluation_surface:finite_code_term_validity",
        "search_surface:exact_enumeration_traversal_count",
        "search_surface:stochastic_seed_restart_disclosure",
        "search_surface:frontier_dominance_semantics",
        "contract_surface:backend_adapter_contract",
        "contract_surface:typed_reference_registry",
        "contract_surface:schema_enum_registry",
    } <= expansion_rows

    for row_id in expansion_rows:
        assert ":" in row_id
        assert row_id == row_id.strip()
        lowered = row_id.lower()
        assert "todo" not in lowered
        assert "tbd" not in lowered
        assert "<" not in row_id
        assert ">" not in row_id


def test_compiler_loaded_vocabularies_cite_live_authority_sources() -> None:
    readme_entry = next(
        entry
        for entry in _load_yaml(SOURCE_MAP_PATH)["entries"]
        if entry["source"] == "README.md"
    )
    canonical_targets = set(readme_entry["canonical_targets"])

    for filename, expected_refs in CORE_VOCABULARY_AUTHORITY_REFS.items():
        source_path, packaged_path = _core_schema_paths(filename)
        source_refs = set(_load_yaml(source_path)["source_doc_refs"])
        packaged_refs = set(_load_yaml(packaged_path)["source_doc_refs"])

        assert source_refs == expected_refs, source_path.relative_to(REPO_ROOT).as_posix()
        assert packaged_refs == expected_refs, (
            packaged_path.relative_to(REPO_ROOT).as_posix()
        )
        assert source_refs == packaged_refs
        assert "README.md" in source_refs
        assert {ref for ref in source_refs if ref.startswith("docs/reference/")} <= (
            canonical_targets
        )
        assert "EUCLID.md" not in source_refs
        assert not any(ref.startswith("docs/canonical/") for ref in source_refs)
        assert not any(ref.startswith("docs/semantic/") for ref in source_refs)
        assert not any(ref.startswith("docs/module-specs/") for ref in source_refs)
