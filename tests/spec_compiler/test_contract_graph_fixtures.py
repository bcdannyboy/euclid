from __future__ import annotations

import json
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "fixtures/canonical/contract-graph-golden"
MISSING_LIFECYCLE_COVERAGE_FIXTURE_ROOT = REPO_ROOT / "fixtures/canonical/contract-graph-missing-lifecycle-coverage"
MODULE_REGISTRY_PATH = Path("schemas/contracts/module-registry.yaml")
SCHEMA_REGISTRY_PATH = Path("schemas/contracts/schema-registry.yaml")
REFERENCE_TYPES_PATH = Path("schemas/contracts/reference-types.yaml")
ENUM_REGISTRY_PATH = Path("schemas/contracts/enum-registry.yaml")
RUN_LIFECYCLE_PATH = Path("schemas/contracts/run-lifecycle.yaml")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _copy_fixture(root: Path, tmp_path: Path) -> Path:
    destination = tmp_path / root.name
    shutil.copytree(root, destination)
    return destination


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_contract_graph_fixture_builds_expected_system_graph(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import build_pack

    source_root = _copy_fixture(FIXTURE_ROOT, tmp_path)
    result = build_pack(source_root=source_root, output_root=tmp_path / "build")
    payload = json.loads(result.contract_graph_json_path.read_text())

    assert payload["summary"] == {
        "module_count": 3,
        "module_dependency_edge_count": 2,
        "schema_count": 3,
        "typed_ref_edge_count": 2,
        "enum_count": 1,
        "markdown_enum_source_count": 0,
        "lifecycle_count": 1,
        "lifecycle_state_count": 3,
        "lifecycle_transition_count": 2,
    }
    assert [module["module"] for module in payload["modules"]] == [
        "claims",
        "manifest_registry",
        "search_planning",
    ]
    assert [schema["schema_name"] for schema in payload["schemas"]] == [
        "claim_card_manifest@1.1.0",
        "run_manifest@1.0.0",
        "search_plan_manifest@1.0.0",
    ]
    assert payload["lifecycles"] == [
        {
            "lifecycle": "run_lifecycle",
            "kind": "run_lifecycle",
            "path": "schemas/contracts/run-lifecycle.yaml",
            "state_ids": [
                "claims_resolved",
                "run_declared",
                "search_contract_frozen",
            ],
            "transition_ids": [
                "freeze_search_contract",
                "resolve_claims",
            ],
        }
    ]

    node_ids = {node["id"] for node in payload["nodes"]}
    assert {
        "module:manifest_registry",
        "module:search_planning",
        "module:claims",
        "schema:run_manifest@1.0.0",
        "schema:search_plan_manifest@1.0.0",
        "schema:claim_card_manifest@1.1.0",
        "enum:run_lifecycle_states",
        "lifecycle:run_lifecycle",
        "state:run_lifecycle:run_declared",
        "state:run_lifecycle:search_contract_frozen",
        "state:run_lifecycle:claims_resolved",
    } <= node_ids

    edge_triples = {(edge["source"], edge["target"], edge["kind"]) for edge in payload["edges"]}
    assert {
        ("module:claims", "module:search_planning", "depends_on"),
        ("module:search_planning", "module:manifest_registry", "depends_on"),
        ("schema:claim_card_manifest@1.1.0", "schema:search_plan_manifest@1.0.0", "typed_ref"),
        ("schema:search_plan_manifest@1.0.0", "schema:run_manifest@1.0.0", "typed_ref"),
        ("module:claims", "module:search_planning", "schema_ref"),
        ("module:search_planning", "module:manifest_registry", "schema_ref"),
    } <= edge_triples

    markdown = result.contract_graph_markdown_path.read_text()
    assert "# Euclid Contract Graph" in markdown
    assert "## Module Dependency DAG" in markdown
    assert "## Typed References" in markdown
    assert "## Lifecycle Coverage" in markdown


def test_contract_graph_fixture_duplicate_schema_owners_fail_build(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_fixture(FIXTURE_ROOT, tmp_path)
    schema_registry_path = source_root / SCHEMA_REGISTRY_PATH
    payload = _load_yaml(schema_registry_path)
    payload["schemas"].append(
        {
            "schema_name": "run_manifest@1.0.0",
            "owner_ref": "claims_owner",
            "owning_module": "claims",
            "canonical_source_path": "docs/canonical/system-map.md",
        }
    )
    _write_yaml(schema_registry_path, payload)

    with pytest.raises(SpecCompilerError) as exc_info:
        build_pack(source_root=source_root, output_root=tmp_path / "build")

    assert "duplicate schema owner" in str(exc_info.value)


def test_contract_graph_fixture_detects_cyclic_module_dependencies(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_fixture(FIXTURE_ROOT, tmp_path)
    module_registry_path = source_root / MODULE_REGISTRY_PATH
    payload = _load_yaml(module_registry_path)
    for module in payload["modules"]:
        if module["module"] == "manifest_registry":
            module["allowed_dependencies"].append("search_planning")
            break
    _write_yaml(module_registry_path, payload)

    with pytest.raises(SpecCompilerError, match="module dependency graph must be acyclic"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_contract_graph_fixture_keeps_schema_names_unique_and_dependency_graph_acyclic() -> None:
    schema_registry = _load_yaml(FIXTURE_ROOT / SCHEMA_REGISTRY_PATH)
    module_registry = _load_yaml(FIXTURE_ROOT / MODULE_REGISTRY_PATH)

    schema_names = [entry["schema_name"] for entry in schema_registry["schemas"]]
    assert len(schema_names) == len(set(schema_names))

    modules = module_registry["modules"]
    module_ids = {entry["module"] for entry in modules}
    indegree = {module_id: 0 for module_id in module_ids}
    adjacency = {module_id: set() for module_id in module_ids}

    for module in modules:
        for dependency in module["allowed_dependencies"]:
            adjacency[dependency].add(module["module"])
            indegree[module["module"]] += 1

    queue = deque(sorted(module_id for module_id, degree in indegree.items() if degree == 0))
    visited: list[str] = []
    while queue:
        module_id = queue.popleft()
        visited.append(module_id)
        for downstream in sorted(adjacency[module_id]):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                queue.append(downstream)

    assert len(visited) == len(module_ids)


def test_contract_graph_fixture_detects_unreachable_lifecycle_states(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_fixture(FIXTURE_ROOT, tmp_path)
    lifecycle_path = source_root / RUN_LIFECYCLE_PATH
    enum_registry_path = source_root / ENUM_REGISTRY_PATH

    lifecycle_payload = _load_yaml(lifecycle_path)
    lifecycle_payload["states"].append(
        {
            "state_id": "replay_verified",
            "owner_ref": "run_lifecycle_council",
            "responsible_modules": ["claims"],
        }
    )
    _write_yaml(lifecycle_path, lifecycle_payload)

    enum_payload = _load_yaml(enum_registry_path)
    enum_payload["enums"][0]["allowed_values"].append("replay_verified")
    _write_yaml(enum_registry_path, enum_payload)

    with pytest.raises(SpecCompilerError, match="unreachable lifecycle state"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_contract_graph_fixture_detects_forbidden_cross_scope_refs(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_fixture(FIXTURE_ROOT, tmp_path)
    reference_types_path = source_root / REFERENCE_TYPES_PATH
    payload = _load_yaml(reference_types_path)

    for profile in payload["reference_profiles"]:
        if profile["schema_name"] == "search_plan_manifest@1.0.0":
            profile["fields"][0]["allowed_schema_names"] = ["claim_card_manifest@1.1.0"]
            break
    _write_yaml(reference_types_path, payload)

    with pytest.raises(SpecCompilerError, match="forbidden cross-scope reference"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_contract_graph_missing_lifecycle_coverage_fixture_fails_build(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_fixture(MISSING_LIFECYCLE_COVERAGE_FIXTURE_ROOT, tmp_path)

    with pytest.raises(SpecCompilerError, match="missing lifecycle node coverage"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")
