from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _copy_repo(tmp_path: Path) -> Path:
    destination = tmp_path / "repo-copy"
    shutil.copytree(
        REPO_ROOT,
        destination,
        ignore=shutil.ignore_patterns("build", "__pycache__", ".pytest_cache", ".DS_Store"),
    )
    return destination


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_build_pack_emits_contract_graph_artifacts_for_live_repo(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import build_pack

    result = build_pack(source_root=REPO_ROOT, output_root=tmp_path / "build")

    assert result.contract_graph_json_path == tmp_path / "build" / "euclid-contract-graph.json"
    assert result.contract_graph_markdown_path == tmp_path / "build" / "euclid-contract-graph.md"
    assert result.contract_graph_json_path.is_file()
    assert result.contract_graph_markdown_path.is_file()

    payload = json.loads(result.contract_graph_json_path.read_text())
    assert payload["summary"]["module_count"] >= 20
    assert payload["summary"]["schema_count"] >= 50
    assert payload["summary"]["enum_count"] >= 10
    assert payload["summary"]["lifecycle_count"] == 3

    node_ids = {node["id"] for node in payload["nodes"]}
    assert "module:manifest_registry" in node_ids
    assert "schema:reproducibility_bundle_manifest@1.0.0" in node_ids
    assert "enum:claim_lanes" in node_ids
    assert "lifecycle:run_lifecycle" in node_ids
    assert "state:run_lifecycle:run_declared" in node_ids

    typed_ref_edges = {
        (edge["source"], edge["target"])
        for edge in payload["edges"]
        if edge["kind"] == "typed_ref"
    }
    assert (
        "schema:reproducibility_bundle_manifest@1.0.0",
        "schema:run_result_manifest@1.1.0",
    ) in typed_ref_edges

    markdown = result.contract_graph_markdown_path.read_text()
    assert "# Euclid Contract Graph" in markdown
    assert "## Module Dependency DAG" in markdown
    assert "## Schema Ownership" in markdown
    assert "## Lifecycle Coverage" in markdown


def test_build_pack_detects_duplicate_schema_owners(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_repo(tmp_path)
    schema_registry_path = source_root / "schemas/contracts/schema-registry.yaml"
    payload = _load_yaml(schema_registry_path)
    payload["schemas"].append(
        {
            "schema_name": "run_manifest@1.0.0",
            "owner_ref": "claims_owner",
            "owning_module": "claims",
            "canonical_source_path": "docs/reference/contracts-manifests.md",
        }
    )
    _write_yaml(schema_registry_path, payload)

    with pytest.raises(SpecCompilerError, match="duplicate schema owner"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_build_pack_detects_invalid_dependency_edges(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_repo(tmp_path)
    module_registry_path = source_root / "schemas/contracts/module-registry.yaml"
    payload = _load_yaml(module_registry_path)
    for module in payload["modules"]:
        if module["module"] == "features":
            module["allowed_dependencies"].append("missing_runtime_module")
            break
    _write_yaml(module_registry_path, payload)

    with pytest.raises(SpecCompilerError, match="invalid dependency edge"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_build_pack_detects_missing_lifecycle_nodes(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_repo(tmp_path)
    lifecycle_path = source_root / "schemas/contracts/run-lifecycle.yaml"
    payload = _load_yaml(lifecycle_path)
    payload["global_rules"]["initial_state"] = "missing_run_state"
    _write_yaml(lifecycle_path, payload)

    with pytest.raises(SpecCompilerError, match="missing lifecycle node"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_build_pack_detects_illegal_cross_module_references(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_repo(tmp_path)
    reference_types_path = source_root / "schemas/contracts/reference-types.yaml"
    payload = _load_yaml(reference_types_path)
    for profile in payload["reference_profiles"]:
        if profile["schema_name"] == "claim_card_manifest@1.1.0":
            for field in profile["fields"]:
                if field["path"] == "candidate_ref":
                    field["allowed_schema_names"] = ["missing_schema_manifest@1.0.0"]
                    break
            break
    _write_yaml(reference_types_path, payload)

    with pytest.raises(SpecCompilerError, match="illegal cross-module reference"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_build_pack_detects_markdown_enums_missing_from_registry(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_repo(tmp_path)
    source_doc_path = source_root / "docs/reference/runtime-cli.md"
    source_doc_path.write_text(
        source_doc_path.read_text()
        + '\n\n```json\n{"bundle_mode": "candidate_publication|abstention_only_publication|shadow_publication"}\n```\n'
    )

    with pytest.raises(SpecCompilerError, match="markdown enum values missing from registry"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")
