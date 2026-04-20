from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_REGISTRY_PATH = REPO_ROOT / "schemas/contracts/module-registry.yaml"
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"
LIVE_REFERENCE_DOCS = {
    "docs/reference/system.md",
    "docs/reference/modeling-pipeline.md",
    "docs/reference/search-core.md",
    "docs/reference/contracts-manifests.md",
}
ALLOWED_CANONICAL_REFS = LIVE_REFERENCE_DOCS | {"docs/reference/runtime-cli.md"}
REQUIRED_MODULES = {
    "manifest_registry",
    "external_evidence_ingestion",
    "ingestion",
    "snapshotting",
    "timeguard",
    "features",
    "split_planning",
    "search_planning",
    "evaluation_governance",
    "algorithmic_dsl",
    "shared_plus_local_decomposition",
    "candidate_fitting",
    "evaluation",
    "probabilistic_evaluation",
    "scoring",
    "robustness",
    "mechanistic_evidence",
    "gate_lifecycle",
    "claims",
    "replay",
    "catalog_publishing",
}
DOC_TERM_EXPECTATIONS = {
    "docs/reference/system.md": {
        "subsystem map",
        "execution planes",
        "src/euclid/modules",
        "src/euclid/contracts",
    },
    "docs/reference/modeling-pipeline.md": {
        "modules/ingestion.py",
        "modules/search_planning.py",
        "modules/probabilistic_evaluation.py",
        "modules/catalog_publishing.py",
    },
    "docs/reference/search-core.md": {
        "candidate intermediate representation",
        "algorithmic dsl",
        "shared_plus_local_decomposition",
        "description gain",
    },
    "docs/reference/contracts-manifests.md": {
        "module-registry.yaml",
        "formal spec assets",
        "docs/implementation/*.yaml",
        "ManifestEnvelope.build",
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_module_registry_defines_full_runtime_inventory() -> None:
    assert MODULE_REGISTRY_PATH.is_file(), "missing schemas/contracts/module-registry.yaml"

    payload = _load_yaml(MODULE_REGISTRY_PATH)
    assert payload["version"] == 1
    assert payload["kind"] == "module_registry"

    owners = payload["owners"]
    modules = payload["modules"]
    assert isinstance(owners, list) and owners, "module registry must declare owners"
    assert isinstance(modules, list) and modules, "module registry must declare modules"

    owner_ids = {owner["id"] for owner in owners}
    module_ids = [module["module"] for module in modules]
    canonical_refs_seen: set[str] = set()

    assert len(module_ids) == len(set(module_ids)), "module ids must be unique"
    assert REQUIRED_MODULES <= set(module_ids), "module registry is missing required full-runtime modules"

    for module in modules:
        assert module["owner_ref"] in owner_ids, f"unknown owner_ref for {module['module']}"
        assert module["inputs"], f"{module['module']} must declare inputs"
        assert module["outputs"], f"{module['module']} must declare outputs"
        assert module["allowed_dependencies"] is not None
        assert module["forbidden_dependencies"] is not None
        assert module["deterministic_obligations"], f"{module['module']} must declare deterministic obligations"
        canonical_refs = module["canonical_refs"]
        assert canonical_refs, f"{module['module']} must route to at least one live reference doc"
        assert set(canonical_refs) <= ALLOWED_CANONICAL_REFS, (
            f"{module['module']} must only cite live docs/reference targets"
        )
        for canonical_ref in canonical_refs:
            assert (REPO_ROOT / canonical_ref).is_file(), (
                f"{module['module']} cites missing canonical ref {canonical_ref}"
            )
        canonical_refs_seen.update(canonical_refs)

    assert LIVE_REFERENCE_DOCS <= canonical_refs_seen, (
        "module registry must cover the live reference workspace across system, "
        "modeling, search, and contracts docs"
    )


def test_module_registry_dependency_edges_are_closed_and_acyclic() -> None:
    payload = _load_yaml(MODULE_REGISTRY_PATH)
    modules = payload["modules"]
    module_ids = {module["module"] for module in modules}
    indegree = {module_id: 0 for module_id in module_ids}
    adjacency = {module_id: set() for module_id in module_ids}

    for module in modules:
        module_id = module["module"]
        allowed = set(module["allowed_dependencies"])
        forbidden = set(module["forbidden_dependencies"])

        assert module_id not in allowed, f"{module_id} may not depend on itself"
        assert module_id not in forbidden, f"{module_id} may not forbid itself"
        assert allowed.isdisjoint(forbidden), f"{module_id} has overlapping allowed/forbidden dependencies"
        assert allowed <= module_ids, f"{module_id} declares unknown allowed dependencies"
        assert forbidden <= module_ids, f"{module_id} declares unknown forbidden dependencies"

        for dependency in allowed:
            adjacency[dependency].add(module_id)
            indegree[module_id] += 1

    queue = deque(sorted(module_id for module_id, degree in indegree.items() if degree == 0))
    visited: list[str] = []

    while queue:
        module_id = queue.popleft()
        visited.append(module_id)
        for downstream in sorted(adjacency[module_id]):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                queue.append(downstream)

    assert len(visited) == len(module_ids), "module dependency graph must be acyclic"


def test_module_registry_routes_into_live_reference_workspace() -> None:
    payload = _load_yaml(MODULE_REGISTRY_PATH)
    registry_by_module = {module["module"]: set(module["canonical_refs"]) for module in payload["modules"]}

    source_map = _load_yaml(SOURCE_MAP_PATH)
    entries = {entry["source"]: entry for entry in source_map["entries"]}

    assert entries["src/euclid/modules"]["canonical_targets"] == ["docs/reference/modeling-pipeline.md"]
    assert entries["src/euclid/search"]["canonical_targets"] == ["docs/reference/search-core.md"]
    assert entries["src/euclid/contracts"]["canonical_targets"] == ["docs/reference/contracts-manifests.md"]
    assert entries["src/euclid/manifests"]["canonical_targets"] == ["docs/reference/contracts-manifests.md"]
    assert "docs/reference/system.md" in entries["README.md"]["canonical_targets"]

    assert "docs/reference/contracts-manifests.md" in registry_by_module["manifest_registry"]
    assert "docs/reference/search-core.md" in registry_by_module["search_planning"]
    assert "docs/reference/search-core.md" in registry_by_module["algorithmic_dsl"]
    assert "docs/reference/modeling-pipeline.md" in registry_by_module["probabilistic_evaluation"]
    assert "docs/reference/contracts-manifests.md" in registry_by_module["catalog_publishing"]

    for relative_path, required_terms in DOC_TERM_EXPECTATIONS.items():
        body = (REPO_ROOT / relative_path).read_text(encoding="utf-8").lower()
        for term in required_terms:
            assert term.lower() in body, f"{relative_path} must mention {term}"
