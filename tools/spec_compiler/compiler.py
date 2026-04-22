from __future__ import annotations

import argparse
import json
import os
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SYSTEM_SCHEMA_RELATIVE_PATH = Path("schemas/core/euclid-system.yaml")
SOURCE_MAP_RELATIVE_PATH = Path("schemas/core/source-map.yaml")
MODULE_REGISTRY_RELATIVE_PATH = Path("schemas/contracts/module-registry.yaml")
SCHEMA_REGISTRY_RELATIVE_PATH = Path("schemas/contracts/schema-registry.yaml")
REFERENCE_TYPES_RELATIVE_PATH = Path("schemas/contracts/reference-types.yaml")
ENUM_REGISTRY_RELATIVE_PATH = Path("schemas/contracts/enum-registry.yaml")
LIFECYCLE_RELATIVE_PATHS = {
    "run_lifecycle": Path("schemas/contracts/run-lifecycle.yaml"),
    "candidate_state_machine": Path("schemas/contracts/candidate-state-machine.yaml"),
    "publication_lifecycle": Path("schemas/contracts/publication-lifecycle.yaml"),
}
READINESS_CONTRACT_RELATIVE_PATH = Path("schemas/readiness/euclid-readiness.yaml")
LIVE_ENTRY_RELATIVE_PATHS = (
    Path("README.md"),
    Path("docs/README.md"),
    Path("docs/system.md"),
)
MATH_FIXTURE_RELATIVE_DIRECTORY = Path("fixtures/canonical/math")
FIXTURE_COVERAGE_RELATIVE_PATH = Path("fixtures/canonical/fixture-coverage.yaml")
FIXTURE_WALKTHROUGH_RELATIVE_DIRECTORY = Path("docs/reference/examples")
DOC_FRONT_MATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
DOUBLE_QUOTE_PATTERN = re.compile(r'"([^"]+)"')
IDENTIFIER_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_@\.\-\[\]]+$")
INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)
FENCED_CODE_BLOCK_PATTERN = re.compile(r"```[^\n]*\n(.*?)\n```", re.DOTALL)
SEMANTIC_SENTINEL_PATTERN = re.compile(
    r"(are only|is only|only valid)",
    re.IGNORECASE,
)
LINK_SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")
READINESS_MANIFEST_SCHEMA = "readiness_judgment_manifest@1.0.0"
READINESS_OBJECT_FIELD_PATTERNS = (
    re.compile(r'["\']?(schema_name|schema)["\']?\s*[:=]\s*["\']readiness_judgment_manifest@1\.0\.0["\']'),
    re.compile(r'["\']?judgment_id["\']?\s*[:=]'),
    re.compile(r'["\']?final_verdict["\']?\s*[:=]'),
)
ENUM_FIELD_TO_VOCABULARY = {
    "reducer_family": "reducer_families",
    "reducer_families": "reducer_families",
    "composition_operator": "composition_operators",
    "composition_operators": "composition_operators",
    "claim_lane": "claim_lanes",
    "claim_lanes": "claim_lanes",
    "lower_claim_lane": "claim_lanes",
    "required_for_claim_lane": "claim_lanes",
    "forecast_object_type": "forecast_object_types",
    "forecast_object_types": "forecast_object_types",
    "allowed_forecast_object_types": "forecast_object_types",
    "evidence_class": "evidence_classes",
    "evidence_classes": "evidence_classes",
    "abstention_type": "abstention_types",
    "abstention_types": "abstention_types",
    "failure_abstention_type": "abstention_types",
    "scope_axis": "scope_axes",
    "scope_axes": "scope_axes",
}
OWNER_REFERENCE_FIELDS = {"owner", "owner_ref"}
STRUCTURED_FILE_SUFFIXES = {".yaml", ".yml", ".json"}
READINESS_EVIDENCE_KINDS = {"doc", "schema", "fixture"}
MARKDOWN_ENUM_VALUE_PATTERNS = {
    "evaluation_event_related_object_kind": re.compile(
        r'"related_object_ref":\s*\{.*?"ref_kind": "([^"]+)"',
        re.DOTALL,
    ),
    "evaluation_event_type": re.compile(r'"event_type": "([^"]+)"'),
    "leakage_stage_evidence_kind": re.compile(
        r'"stage_evidence_ref":\s*\{.*?"ref_kind": "([^"]+)"',
        re.DOTALL,
    ),
    "leakage_downstream_evidence_kind": re.compile(
        r'"downstream_evidence_refs":\s*\[.*?"ref_kind": "([^"]+)"',
        re.DOTALL,
    ),
    "replay_bundle_mode": re.compile(r'"bundle_mode": "([^"]+)"'),
    "replay_artifact_role": re.compile(r'"artifact_role": "([^"]+)"'),
    "replay_seed_scope": re.compile(r'"seed_scope": "([^"]+)"'),
}
ENTRYPOINT_REQUIRED_FRAGMENTS = {
    "README.md": (
        "docs/system.md",
        "docs/runtime-cli.md",
        "docs/modeling-pipeline.md",
        "docs/search-core.md",
        "docs/contracts-manifests.md",
        "docs/benchmarks-readiness.md",
        "docs/workbench.md",
        "docs/testing-truthfulness.md",
        "schemas/core/euclid-system.yaml",
        "schemas/core/source-map.yaml",
    ),
    "docs/README.md": (
        "../README.md",
        "system.md",
        "runtime-cli.md",
        "modeling-pipeline.md",
        "search-core.md",
        "contracts-manifests.md",
        "benchmarks-readiness.md",
        "workbench.md",
        "testing-truthfulness.md",
    ),
    "docs/system.md": (
        "runtime-cli.md",
        "modeling-pipeline.md",
        "search-core.md",
        "contracts-manifests.md",
        "benchmarks-readiness.md",
        "workbench.md",
    ),
}
DISALLOWED_LIVE_AUTHORITY_PHRASES = {
    "README.md": (),
    "docs/README.md": (),
    "docs/system.md": (),
}


class SpecCompilerError(RuntimeError):
    """Raised when the canonical pack fails validation."""


@dataclass(frozen=True)
class BuildResult:
    source_root: Path
    output_root: Path
    markdown_path: Path
    json_path: Path
    contract_graph_markdown_path: Path
    contract_graph_json_path: Path
    fixture_closure_markdown_path: Path | None = None
    fixture_closure_json_path: Path | None = None
    readiness_pack_markdown_path: Path | None = None
    readiness_pack_json_path: Path | None = None


def build_pack(source_root: Path | str, output_root: Path | str | None = None) -> BuildResult:
    root = Path(source_root).resolve()
    build_root = Path(output_root).resolve() if output_root is not None else root / "build"
    build_root.mkdir(parents=True, exist_ok=True)

    system_schema = _load_required_file(root, SYSTEM_SCHEMA_RELATIVE_PATH)
    required_refs_checked = 0

    canonical_docs: list[dict[str, Any]] = []
    for relative_path in system_schema.get("canonical_doc_refs", []):
        required_refs_checked += 1
        canonical_docs.append(_load_document_summary(root, relative_path))

    vocabularies, allowed_values = _load_vocabularies(root, system_schema.get("vocabulary_refs", {}))
    required_refs_checked += len(vocabularies)

    contracts, owner_ids, contract_lookup = _load_contract_artifacts(root, allowed_values)
    math_fixtures = _load_math_fixtures(root, allowed_values, contract_lookup)

    payload = {
        "project_name": system_schema["project_name"],
        "scope_statement": system_schema["scope_statement"],
        "canonical_docs": canonical_docs,
        "vocabularies": vocabularies,
        "runtime_modules": system_schema.get("major_runtime_modules", {}),
        "artifact_classes": system_schema.get("artifact_classes", {}),
        "contracts": contracts,
        "math_fixtures": math_fixtures,
        "owner_ids": owner_ids,
        "validation_summary": {
            "required_refs_checked": required_refs_checked,
            "closed_vocabularies_loaded": len(vocabularies),
            "contract_artifacts_loaded": len(contracts),
            "math_fixtures_loaded": len(math_fixtures),
            "owners_declared": len(owner_ids),
        },
    }

    markdown_path = build_root / "euclid-canonical-pack.md"
    json_path = build_root / "euclid-canonical-pack.json"
    markdown_path.write_text(_render_markdown(payload))
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    contract_graph_payload = _build_contract_graph(root)
    contract_graph_markdown_path = build_root / "euclid-contract-graph.md"
    contract_graph_json_path = build_root / "euclid-contract-graph.json"
    contract_graph_markdown_path.write_text(_render_contract_graph_markdown(contract_graph_payload))
    contract_graph_json_path.write_text(json.dumps(contract_graph_payload, indent=2) + "\n")
    fixture_closure_payload = _build_fixture_closure(root)
    fixture_closure_markdown_path: Path | None = None
    fixture_closure_json_path: Path | None = None
    if fixture_closure_payload is not None:
        fixture_closure_markdown_path = build_root / "euclid-fixture-closure.md"
        fixture_closure_json_path = build_root / "euclid-fixture-closure.json"
        fixture_closure_markdown_path.write_text(_render_fixture_closure_markdown(fixture_closure_payload))
        fixture_closure_json_path.write_text(json.dumps(fixture_closure_payload, indent=2) + "\n")
    readiness_pack_payload = _build_readiness_pack(
        root,
        system_schema,
        contract_graph_payload,
        fixture_closure_payload,
    )
    readiness_pack_markdown_path: Path | None = None
    readiness_pack_json_path: Path | None = None
    if readiness_pack_payload is not None:
        readiness_pack_markdown_path = build_root / "euclid-readiness-pack.md"
        readiness_pack_json_path = build_root / "euclid-readiness-pack.json"
        readiness_pack_markdown_path.write_text(_render_readiness_pack_markdown(readiness_pack_payload))
        readiness_pack_json_path.write_text(json.dumps(readiness_pack_payload, indent=2) + "\n")

    return BuildResult(
        source_root=root,
        output_root=build_root,
        markdown_path=markdown_path,
        json_path=json_path,
        contract_graph_markdown_path=contract_graph_markdown_path,
        contract_graph_json_path=contract_graph_json_path,
        fixture_closure_markdown_path=fixture_closure_markdown_path,
        fixture_closure_json_path=fixture_closure_json_path,
        readiness_pack_markdown_path=readiness_pack_markdown_path,
        readiness_pack_json_path=readiness_pack_json_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Euclid canonical documentation pack.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Validate and build the canonical pack.")
    build_parser.add_argument(
        "--source-root",
        default=".",
        help="Repository or fixture root containing docs/, schemas/, and optional build/ output.",
    )
    build_parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for generated build artifacts. Defaults to <source-root>/build.",
    )

    args = parser.parse_args(argv)

    if args.command != "build":
        parser.error(f"unsupported command: {args.command}")

    result = build_pack(source_root=args.source_root, output_root=args.output_root)
    print(f"Wrote {result.markdown_path}")
    print(f"Wrote {result.json_path}")
    print(f"Wrote {result.contract_graph_markdown_path}")
    print(f"Wrote {result.contract_graph_json_path}")
    if result.fixture_closure_markdown_path is not None:
        print(f"Wrote {result.fixture_closure_markdown_path}")
    if result.fixture_closure_json_path is not None:
        print(f"Wrote {result.fixture_closure_json_path}")
    if result.readiness_pack_markdown_path is not None:
        print(f"Wrote {result.readiness_pack_markdown_path}")
    if result.readiness_pack_json_path is not None:
        print(f"Wrote {result.readiness_pack_json_path}")
    return 0


def _build_contract_graph(root: Path) -> dict[str, Any]:
    module_payload = _load_optional_file(root, MODULE_REGISTRY_RELATIVE_PATH)
    schema_payload = _load_optional_file(root, SCHEMA_REGISTRY_RELATIVE_PATH)
    reference_types_payload = _load_optional_file(root, REFERENCE_TYPES_RELATIVE_PATH)
    enum_payload = _load_optional_file(root, ENUM_REGISTRY_RELATIVE_PATH)

    node_map: dict[str, dict[str, Any]] = {}
    edge_map: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def add_node(node_id: str, kind: str, label: str, **metadata: Any) -> None:
        node = {"id": node_id, "kind": kind, "label": label}
        for key, value in metadata.items():
            if value is not None:
                node[key] = value
        node_map[node_id] = node

    def add_edge(source: str, target: str, kind: str, **metadata: Any) -> None:
        clean_metadata = {key: value for key, value in metadata.items() if value is not None}
        metadata_key = json.dumps(clean_metadata, sort_keys=True)
        edge_map[(kind, source, target, metadata_key)] = {
            "source": source,
            "target": target,
            "kind": kind,
            **clean_metadata,
        }

    module_index, module_dependency_edges = _validate_module_registry(module_payload)
    schema_index = _validate_schema_registry(schema_payload, module_index)
    enum_index = _validate_enum_registry(root, enum_payload)
    lifecycle_index = _validate_lifecycle_contracts(root, module_index, enum_index)
    reference_edges = _validate_reference_profiles(reference_types_payload, schema_index, module_index)
    _validate_markdown_enum_coverage(root, enum_index)

    for module_id, module in sorted(module_index.items()):
        add_node(
            f"module:{module_id}",
            "module",
            module_id,
            plane=module["plane"],
            owner_ref=module["owner_ref"],
            path=module["path"],
        )

    for source_module, target_module in module_dependency_edges:
        add_edge(
            f"module:{source_module}",
            f"module:{target_module}",
            "depends_on",
            path=MODULE_REGISTRY_RELATIVE_PATH.as_posix(),
        )

    for schema_name, schema in sorted(schema_index.items()):
        add_node(
            f"schema:{schema_name}",
            "schema",
            schema_name,
            owning_module=schema["owning_module"],
            owner_ref=schema["owner_ref"],
            canonical_source_path=schema["canonical_source_path"],
        )
        if schema["owning_module"] in module_index:
            add_edge(
                f"module:{schema['owning_module']}",
                f"schema:{schema_name}",
                "owns_schema",
                path=SCHEMA_REGISTRY_RELATIVE_PATH.as_posix(),
            )

    for edge in reference_edges:
        add_edge(
            f"schema:{edge['source_schema']}",
            f"schema:{edge['target_schema']}",
            "typed_ref",
            field_path=edge["field_path"],
            source_module=edge["source_module"],
            target_module=edge["target_module"],
            discriminator=edge.get("discriminator"),
            path=REFERENCE_TYPES_RELATIVE_PATH.as_posix(),
        )
        if edge["source_module"] in module_index and edge["target_module"] in module_index:
            add_edge(
                f"module:{edge['source_module']}",
                f"module:{edge['target_module']}",
                "schema_ref",
                field_path=edge["field_path"],
                path=REFERENCE_TYPES_RELATIVE_PATH.as_posix(),
            )

    for enum_name, enum_entry in sorted(enum_index.items()):
        source_path = enum_entry["canonical_source_path"]
        add_node(
            f"enum:{enum_name}",
            "enum",
            enum_name,
            path=source_path,
            owner_ref=enum_entry["owner_ref"],
        )
        for value in enum_entry["allowed_values"]:
            add_node(
                f"enum_value:{enum_name}:{value}",
                "enum_value",
                value,
                enum_name=enum_name,
            )
            add_edge(
                f"enum:{enum_name}",
                f"enum_value:{enum_name}:{value}",
                "enum_value",
                path=source_path,
            )

    for lifecycle_name, lifecycle in sorted(lifecycle_index.items()):
        add_node(
            f"lifecycle:{lifecycle_name}",
            "lifecycle",
            lifecycle_name,
            kind_name=lifecycle["kind"],
            path=lifecycle["path"],
        )
        for state_id in lifecycle["state_ids"]:
            add_node(
                f"state:{lifecycle_name}:{state_id}",
                "lifecycle_state",
                state_id,
                lifecycle=lifecycle_name,
            )
            add_edge(
                f"lifecycle:{lifecycle_name}",
                f"state:{lifecycle_name}:{state_id}",
                "has_state",
                path=lifecycle["path"],
            )
        for transition in lifecycle["transitions"]:
            transition_id = transition["transition_id"]
            add_node(
                f"transition:{lifecycle_name}:{transition_id}",
                "lifecycle_transition",
                transition_id,
                lifecycle=lifecycle_name,
            )
            add_edge(
                f"lifecycle:{lifecycle_name}",
                f"transition:{lifecycle_name}:{transition_id}",
                "has_transition",
                path=lifecycle["path"],
            )
            add_edge(
                f"transition:{lifecycle_name}:{transition_id}",
                f"state:{lifecycle_name}:{transition['from']}",
                "transition_from",
                path=lifecycle["path"],
            )
            add_edge(
                f"transition:{lifecycle_name}:{transition_id}",
                f"state:{lifecycle_name}:{transition['to']}",
                "transition_to",
                path=lifecycle["path"],
            )

    nodes = sorted(node_map.values(), key=lambda item: (item["kind"], item["id"]))
    edges = sorted(
        edge_map.values(),
        key=lambda item: (
            item["kind"],
            item["source"],
            item["target"],
            item.get("field_path", ""),
            item.get("discriminator", ""),
        ),
    )

    summary = {
        "module_count": len(module_index),
        "module_dependency_edge_count": len(module_dependency_edges),
        "schema_count": len(schema_index),
        "typed_ref_edge_count": len(reference_edges),
        "enum_count": len(enum_index),
        "markdown_enum_source_count": sum(
            1 for enum_entry in enum_index.values() if Path(enum_entry["canonical_source_path"]).suffix == ".md"
        ),
        "lifecycle_count": len(lifecycle_index),
        "lifecycle_state_count": sum(len(lifecycle["state_ids"]) for lifecycle in lifecycle_index.values()),
        "lifecycle_transition_count": sum(len(lifecycle["transitions"]) for lifecycle in lifecycle_index.values()),
    }

    return {
        "source_root": root.as_posix(),
        "summary": summary,
        "modules": [
            {
                "module": module_id,
                "plane": module["plane"],
                "owner_ref": module["owner_ref"],
                "allowed_dependencies": module["allowed_dependencies"],
                "forbidden_dependencies": module["forbidden_dependencies"],
            }
            for module_id, module in sorted(module_index.items())
        ],
        "schemas": [
            {
                "schema_name": schema_name,
                "owner_ref": schema["owner_ref"],
                "owning_module": schema["owning_module"],
                "canonical_source_path": schema["canonical_source_path"],
            }
            for schema_name, schema in sorted(schema_index.items())
        ],
        "enums": [
            {
                "enum_name": enum_name,
                "canonical_source_path": enum_entry["canonical_source_path"],
                "allowed_values": enum_entry["allowed_values"],
            }
            for enum_name, enum_entry in sorted(enum_index.items())
        ],
        "lifecycles": [
            {
                "lifecycle": lifecycle_name,
                "kind": lifecycle["kind"],
                "path": lifecycle["path"],
                "state_ids": lifecycle["state_ids"],
                "transition_ids": [transition["transition_id"] for transition in lifecycle["transitions"]],
            }
            for lifecycle_name, lifecycle in sorted(lifecycle_index.items())
        ],
        "nodes": nodes,
        "edges": edges,
    }


def _validate_module_registry(payload: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]]]:
    if payload is None:
        return {}, []

    if payload.get("kind") != "module_registry":
        raise SpecCompilerError(f"{MODULE_REGISTRY_RELATIVE_PATH.as_posix()} must declare kind: module_registry")

    owner_ids = {owner["id"] for owner in payload.get("owners", [])}
    module_index: dict[str, dict[str, Any]] = {}

    for entry in payload.get("modules", []):
        module_id = entry.get("module")
        if not module_id:
            raise SpecCompilerError(f"{MODULE_REGISTRY_RELATIVE_PATH.as_posix()} module entries must include module")
        if module_id in module_index:
            raise SpecCompilerError(f"invalid dependency edge: duplicate module '{module_id}'")
        owner_ref = entry.get("owner_ref")
        if owner_ref not in owner_ids:
            raise SpecCompilerError(
                f"{MODULE_REGISTRY_RELATIVE_PATH.as_posix()} unknown owner_ref '{owner_ref}' for module '{module_id}'"
            )
        module_index[module_id] = {
            "owner_ref": owner_ref,
            "plane": entry.get("plane"),
            "allowed_dependencies": sorted(set(entry.get("allowed_dependencies", []))),
            "forbidden_dependencies": sorted(set(entry.get("forbidden_dependencies", []))),
            "path": MODULE_REGISTRY_RELATIVE_PATH.as_posix(),
        }

    dependency_edges: list[tuple[str, str]] = []
    indegree = {module_id: 0 for module_id in module_index}
    adjacency = {module_id: set() for module_id in module_index}

    for module_id, module in module_index.items():
        allowed = set(module["allowed_dependencies"])
        forbidden = set(module["forbidden_dependencies"])
        invalid_targets = sorted((allowed | forbidden) - set(module_index))
        if invalid_targets:
            raise SpecCompilerError(
                f"invalid dependency edge for '{module_id}': unknown modules {', '.join(invalid_targets)}"
            )
        if module_id in allowed or module_id in forbidden:
            raise SpecCompilerError(f"invalid dependency edge for '{module_id}': self references are forbidden")
        overlap = sorted(allowed & forbidden)
        if overlap:
            raise SpecCompilerError(
                f"invalid dependency edge for '{module_id}': overlapping allow/forbid targets {', '.join(overlap)}"
            )
        for dependency in sorted(allowed):
            dependency_edges.append((module_id, dependency))
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

    if len(visited) != len(module_index):
        raise SpecCompilerError("invalid dependency edge: module dependency graph must be acyclic")

    return module_index, sorted(dependency_edges)


def _validate_schema_registry(
    payload: dict[str, Any] | None,
    module_index: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if payload is None:
        return {}

    if payload.get("kind") != "schema_registry":
        raise SpecCompilerError(f"{SCHEMA_REGISTRY_RELATIVE_PATH.as_posix()} must declare kind: schema_registry")

    schema_index: dict[str, dict[str, Any]] = {}
    known_owner_ids = {module["owner_ref"] for module in module_index.values()}
    for entry in payload.get("schemas", []):
        schema_name = entry.get("schema_name")
        if not schema_name:
            raise SpecCompilerError(f"{SCHEMA_REGISTRY_RELATIVE_PATH.as_posix()} schema entries must include schema_name")
        if schema_name in schema_index:
            raise SpecCompilerError(f"duplicate schema owner for '{schema_name}'")
        owner_ref = entry.get("owner_ref")
        owning_module = entry.get("owning_module")
        if module_index:
            if owner_ref not in known_owner_ids:
                raise SpecCompilerError(
                    f"{SCHEMA_REGISTRY_RELATIVE_PATH.as_posix()} unknown owner_ref '{owner_ref}' for '{schema_name}'"
                )
            if owning_module not in module_index:
                raise SpecCompilerError(
                    f"{SCHEMA_REGISTRY_RELATIVE_PATH.as_posix()} unknown owning_module '{owning_module}' for "
                    f"'{schema_name}'"
                )
            expected_owner = module_index[owning_module]["owner_ref"]
            if expected_owner != owner_ref:
                raise SpecCompilerError(
                    f"duplicate schema owner for '{schema_name}': owner_ref '{owner_ref}' does not match module "
                    f"'{owning_module}'"
                )
        schema_index[schema_name] = {
            "owner_ref": owner_ref,
            "owning_module": owning_module,
            "canonical_source_path": entry.get("canonical_source_path"),
        }

    return schema_index


def _validate_enum_registry(root: Path, payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if payload is None:
        return {}

    if payload.get("kind") != "enum_registry":
        raise SpecCompilerError(f"{ENUM_REGISTRY_RELATIVE_PATH.as_posix()} must declare kind: enum_registry")

    owner_ids = {owner["id"] for owner in payload.get("owners", [])}
    enum_index: dict[str, dict[str, Any]] = {}
    for entry in payload.get("enums", []):
        enum_name = entry.get("enum_name")
        if not enum_name:
            raise SpecCompilerError(f"{ENUM_REGISTRY_RELATIVE_PATH.as_posix()} enum entries must include enum_name")
        if enum_name in enum_index:
            raise SpecCompilerError(f"{ENUM_REGISTRY_RELATIVE_PATH.as_posix()} duplicate enum '{enum_name}'")
        owner_ref = entry.get("owner_ref")
        if owner_ref not in owner_ids:
            raise SpecCompilerError(
                f"{ENUM_REGISTRY_RELATIVE_PATH.as_posix()} unknown owner_ref '{owner_ref}' for '{enum_name}'"
            )
        canonical_source_path = entry.get("canonical_source_path")
        if not canonical_source_path or not (root / canonical_source_path).is_file():
            raise SpecCompilerError(f"missing required reference: {canonical_source_path}")
        allowed_values = entry.get("allowed_values", [])
        if not allowed_values:
            raise SpecCompilerError(f"{ENUM_REGISTRY_RELATIVE_PATH.as_posix()} enum '{enum_name}' has no values")
        if len(allowed_values) != len(set(allowed_values)):
            raise SpecCompilerError(f"{ENUM_REGISTRY_RELATIVE_PATH.as_posix()} enum '{enum_name}' has duplicate values")
        enum_index[enum_name] = {
            "owner_ref": owner_ref,
            "canonical_source_path": canonical_source_path,
            "allowed_values": sorted(allowed_values),
        }

    return enum_index


def _validate_lifecycle_contracts(
    root: Path,
    module_index: dict[str, dict[str, Any]],
    enum_index: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lifecycle_index: dict[str, dict[str, Any]] = {}
    lifecycle_enum_names = {
        "run_lifecycle": "run_lifecycle_states",
        "candidate_state_machine": "candidate_lifecycle_states",
        "publication_lifecycle": "publication_lifecycle_states",
    }

    for lifecycle_name, relative_path in LIFECYCLE_RELATIVE_PATHS.items():
        payload = _load_optional_file(root, relative_path)
        if payload is None:
            continue

        owner_ids = {owner["id"] for owner in payload.get("owners", [])}
        states = payload.get("states", [])
        transitions = payload.get("transitions", [])
        state_ids: list[str] = []
        seen_state_ids: set[str] = set()
        for state in states:
            state_id = state.get("state_id")
            if not state_id:
                raise SpecCompilerError(f"{relative_path.as_posix()} states must include state_id")
            if state_id in seen_state_ids:
                raise SpecCompilerError(f"missing lifecycle node uniqueness in {relative_path.as_posix()}: {state_id}")
            seen_state_ids.add(state_id)
            state_ids.append(state_id)
            owner_ref = state.get("owner_ref")
            if owner_ref not in owner_ids:
                raise SpecCompilerError(f"missing lifecycle node owner for {state_id} in {relative_path.as_posix()}")
            for module_id in state.get("responsible_modules", []):
                if module_index and module_id not in module_index:
                    raise SpecCompilerError(
                        f"missing lifecycle node module '{module_id}' for state '{state_id}' in {relative_path.as_posix()}"
                    )

        transition_ids: set[str] = set()
        for transition in transitions:
            transition_id = transition.get("transition_id")
            if not transition_id:
                raise SpecCompilerError(f"{relative_path.as_posix()} transitions must include transition_id")
            if transition_id in transition_ids:
                raise SpecCompilerError(
                    f"missing lifecycle node uniqueness in {relative_path.as_posix()}: {transition_id}"
                )
            transition_ids.add(transition_id)
            owner_ref = transition.get("owner_ref")
            if owner_ref not in owner_ids:
                raise SpecCompilerError(
                    f"missing lifecycle node owner for {transition_id} in {relative_path.as_posix()}"
                )
            if transition.get("from") not in seen_state_ids or transition.get("to") not in seen_state_ids:
                raise SpecCompilerError(
                    f"missing lifecycle node for transition '{transition_id}' in {relative_path.as_posix()}"
                )
            for module_id in transition.get("activating_modules", []):
                if module_index and module_id not in module_index:
                    raise SpecCompilerError(
                        f"missing lifecycle node module '{module_id}' for transition '{transition_id}' in "
                        f"{relative_path.as_posix()}"
                    )

        global_rules = payload.get("global_rules", {})
        for key in ("initial_state",):
            state_id = global_rules.get(key)
            if state_id and state_id not in seen_state_ids:
                raise SpecCompilerError(
                    f"missing lifecycle node '{state_id}' referenced by {key} in {relative_path.as_posix()}"
                )
        for key in ("terminal_states", "freeze_boundaries"):
            for state_id in global_rules.get(key, []):
                if state_id not in seen_state_ids:
                    raise SpecCompilerError(
                        f"missing lifecycle node '{state_id}' referenced by {key} in {relative_path.as_posix()}"
                    )

        enum_name = lifecycle_enum_names[lifecycle_name]
        if enum_name in enum_index:
            enum_state_ids = set(enum_index[enum_name]["allowed_values"])
            lifecycle_state_ids = set(seen_state_ids)
            if enum_state_ids != lifecycle_state_ids:
                missing = sorted((enum_state_ids ^ lifecycle_state_ids))
                raise SpecCompilerError(
                    f"missing lifecycle node coverage for {relative_path.as_posix()}: {', '.join(missing)}"
                )

        initial_state = global_rules.get("initial_state")
        if initial_state:
            adjacency = {state_id: set() for state_id in seen_state_ids}
            for transition in transitions:
                adjacency[transition["from"]].add(transition["to"])

            reachable: set[str] = set()
            queue = deque([initial_state])
            while queue:
                state_id = queue.popleft()
                if state_id in reachable:
                    continue
                reachable.add(state_id)
                for downstream in sorted(adjacency[state_id]):
                    if downstream not in reachable:
                        queue.append(downstream)

            unreachable = sorted(set(seen_state_ids) - reachable)
            if unreachable:
                raise SpecCompilerError(
                    f"unreachable lifecycle state in {relative_path.as_posix()}: {', '.join(unreachable)}"
                )

        lifecycle_index[lifecycle_name] = {
            "kind": payload.get("kind"),
            "path": relative_path.as_posix(),
            "state_ids": sorted(seen_state_ids),
            "transitions": sorted(transitions, key=lambda entry: entry["transition_id"]),
        }

    return lifecycle_index


def _validate_reference_profiles(
    payload: dict[str, Any] | None,
    schema_index: dict[str, dict[str, Any]],
    module_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if payload is None:
        return []

    reference_edges: list[dict[str, Any]] = []
    for profile in payload.get("reference_profiles", []):
        source_schema = profile.get("schema_name")
        if source_schema not in schema_index:
            raise SpecCompilerError(f"illegal cross-module reference from unknown schema '{source_schema}'")
        source_schema_entry = schema_index[source_schema]
        if profile.get("owner_ref") != source_schema_entry["owner_ref"]:
            raise SpecCompilerError(
                f"illegal cross-module reference owner mismatch for '{source_schema}'"
            )
        for field in profile.get("fields", []):
            forbidden_modules = set(field.get("forbidden_modules", []))
            if module_index:
                invalid_forbidden_modules = sorted(forbidden_modules - set(module_index))
                if invalid_forbidden_modules:
                    raise SpecCompilerError(
                        f"illegal cross-module reference from '{source_schema}' to unknown forbidden module "
                        f"{', '.join(invalid_forbidden_modules)}"
                    )
            for discriminator, target_schema in _iter_allowed_schema_targets(field):
                if target_schema not in schema_index:
                    raise SpecCompilerError(
                        f"illegal cross-module reference from '{source_schema}' to unknown schema '{target_schema}'"
                    )
                target_entry = schema_index[target_schema]
                source_module = source_schema_entry["owning_module"]
                target_module = target_entry["owning_module"]
                if module_index and target_module not in module_index:
                    raise SpecCompilerError(
                        f"illegal cross-module reference from '{source_schema}' to unknown module "
                        f"'{target_module}'"
                    )
                if source_module != target_module and target_module in forbidden_modules:
                    raise SpecCompilerError(
                        f"forbidden cross-scope reference from '{source_schema}' to '{target_schema}' "
                        f"via '{source_module}' -> '{target_module}'"
                    )
                reference_edges.append(
                    {
                        "source_schema": source_schema,
                        "target_schema": target_schema,
                        "field_path": field.get("path"),
                        "discriminator": discriminator,
                        "source_module": source_module,
                        "target_module": target_module,
                    }
                )

    return sorted(
        reference_edges,
        key=lambda entry: (
            entry["source_schema"],
            entry["field_path"],
            entry["discriminator"] or "",
            entry["target_schema"],
        ),
    )


def _validate_markdown_enum_coverage(root: Path, enum_index: dict[str, dict[str, Any]]) -> None:
    for enum_name, enum_entry in enum_index.items():
        source_path = Path(enum_entry["canonical_source_path"])
        if source_path.suffix != ".md":
            continue
        if enum_name not in MARKDOWN_ENUM_VALUE_PATTERNS and enum_name != "validation_scope_deferred_scope_schema_names":
            continue
        observed_values = _extract_markdown_enum_values(
            (root / source_path).read_text(),
            enum_name,
        )
        missing_values = sorted(observed_values - set(enum_entry["allowed_values"]))
        if missing_values:
            raise SpecCompilerError(
                f"markdown enum values missing from registry for {enum_name} in {source_path.as_posix()}: "
                f"{', '.join(missing_values)}"
            )


def _iter_allowed_schema_targets(field: dict[str, Any]) -> list[tuple[str | None, str]]:
    targets: list[tuple[str | None, str]] = []
    for schema_name in field.get("allowed_schema_names", []):
        targets.append((None, schema_name))
    for discriminator, schema_names in sorted(field.get("allowed_schema_names_by_discriminator", {}).items()):
        for schema_name in schema_names:
            targets.append((discriminator, schema_name))
    return targets


def _extract_markdown_enum_values(text: str, enum_name: str) -> set[str]:
    if enum_name == "validation_scope_deferred_scope_schema_names":
        match = re.search(r'"deferred_scope_refs":\s*\[(.*?)\n\s*\]', text, re.DOTALL)
        if not match:
            return set()
        values = {
            token
            for token in DOUBLE_QUOTE_PATTERN.findall(match.group(1))
            if IDENTIFIER_TOKEN_PATTERN.match(token) and "_manifest@" in token
        }
        return values

    pattern = MARKDOWN_ENUM_VALUE_PATTERNS[enum_name]
    values: set[str] = set()
    for raw_token in pattern.findall(text):
        values.update(_split_candidate_tokens(raw_token, allow_schema_names=False))
    return values


def _split_candidate_tokens(raw_token: str, *, allow_schema_names: bool) -> list[str]:
    parts = [part.strip() for part in re.split(r"[|,]", raw_token)]
    return [
        part
        for part in parts
        if part
        and IDENTIFIER_TOKEN_PATTERN.match(part)
        and ("." not in part or "@" in part)
        and "/" not in part
        and (allow_schema_names or "_manifest@" not in part)
        and part not in {"string", "timestamp", "integer", "decimal_string", "sha256:hex", "null", "true", "false"}
    ]


def _render_contract_graph_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Euclid Contract Graph",
        "",
        "## Summary",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Module Dependency DAG"])
    if payload["modules"]:
        for module in payload["modules"]:
            dependencies = ", ".join(module["allowed_dependencies"]) if module["allowed_dependencies"] else "None"
            lines.append(f"- `{module['module']}` -> {dependencies}")
    else:
        lines.append("- None")

    lines.extend(["", "## Schema Ownership"])
    if payload["schemas"]:
        for schema in payload["schemas"]:
            lines.append(
                f"- `{schema['schema_name']}` -> `{schema['owning_module']}` "
                f"(`{schema['canonical_source_path']}`)"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Typed References"])
    typed_ref_edges = [edge for edge in payload["edges"] if edge["kind"] == "typed_ref"]
    if typed_ref_edges:
        for edge in typed_ref_edges:
            discriminator = f" [{edge['discriminator']}]" if edge.get("discriminator") else ""
            lines.append(
                f"- `{edge['source'].removeprefix('schema:')}` -> `{edge['target'].removeprefix('schema:')}` "
                f"via `{edge['field_path']}`{discriminator}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Enum Coverage"])
    if payload["enums"]:
        for enum_entry in payload["enums"]:
            values = ", ".join(enum_entry["allowed_values"])
            lines.append(
                f"- `{enum_entry['enum_name']}` (`{enum_entry['canonical_source_path']}`) -> {values}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Lifecycle Coverage"])
    if payload["lifecycles"]:
        for lifecycle in payload["lifecycles"]:
            lines.append(
                f"- `{lifecycle['lifecycle']}` (`{lifecycle['path']}`) | "
                f"states={len(lifecycle['state_ids'])} | transitions={len(lifecycle['transition_ids'])}"
            )
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def _load_vocabularies(
    root: Path,
    vocabulary_refs: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    vocabularies: dict[str, dict[str, Any]] = {}
    allowed_values: dict[str, set[str]] = {}

    for vocabulary_name in sorted(vocabulary_refs):
        relative_path = vocabulary_refs[vocabulary_name]
        payload = _load_required_file(root, Path(relative_path))
        if payload.get("closed") is not True:
            raise SpecCompilerError(f"{relative_path} must declare a closed vocabulary")

        entries = payload.get("entries", [])
        entry_ids = [entry["id"] for entry in entries]
        vocabularies[vocabulary_name] = {
            "path": relative_path,
            "title": payload["title"],
            "entry_ids": entry_ids,
        }
        allowed_values[vocabulary_name] = set(entry_ids)

    return vocabularies, allowed_values


def _load_contract_artifacts(
    root: Path,
    allowed_values: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    owner_registry: dict[str, str] = {}
    contract_summaries: list[dict[str, Any]] = []
    contract_lookup: dict[str, Any] = {
        "comparison_regimes": {},
        "many_model_adjustments": {},
        "predictive_gate_policies": {},
        "scoring_by_forecast_object_type": {},
        "calibration_by_forecast_object_type": {},
        "mechanistic_requirements": {},
        "mechanistic_global_rules": {},
        "shared_plus_local_rules": {},
        "composition_operator_rules": {},
    }

    for relative_path in _iter_contract_paths(root):
        payload = _load_structured_file(root / relative_path)
        _collect_owner_declarations(relative_path, payload, owner_registry)
        _validate_enum_values(relative_path, payload, allowed_values)
        _validate_readiness_contract(root, relative_path, payload)
        _update_contract_lookup(relative_path, payload, contract_lookup)

        contract_modules = []
        for contract in payload.get("contracts", []):
            if isinstance(contract, dict) and "module" in contract:
                contract_modules.append(contract["module"])

        contract_summaries.append(
            {
                "path": relative_path.as_posix(),
                "kind": payload.get("kind", "artifact"),
                "owners": [owner["id"] for owner in payload.get("owners", [])],
                "contract_modules": contract_modules,
            }
        )

    for relative_path in _iter_contract_paths(root):
        payload = _load_structured_file(root / relative_path)
        _validate_owner_references(relative_path, payload, set(owner_registry))

    return contract_summaries, sorted(owner_registry), contract_lookup


def _validate_readiness_contract(root: Path, relative_path: Path, payload: dict[str, Any]) -> None:
    if payload.get("kind") != "euclid_readiness_contract":
        return

    contract_id = payload.get("contract_id")
    if not isinstance(contract_id, str) or not contract_id:
        raise SpecCompilerError(f"{relative_path.as_posix()} must declare contract_id")

    if payload.get("readiness_judgment_schema") != "readiness_judgment_manifest@1.0.0":
        raise SpecCompilerError(
            f"{relative_path.as_posix()} readiness_judgment_schema must equal readiness_judgment_manifest@1.0.0"
        )

    if payload.get("area_status_values") != ["passed", "failed"]:
        raise SpecCompilerError(f"{relative_path.as_posix()} area_status_values must equal ['passed', 'failed']")

    if payload.get("final_verdict_values") != ["ready", "not_ready"]:
        raise SpecCompilerError(f"{relative_path.as_posix()} final_verdict_values must equal ['ready', 'not_ready']")

    source_root_requirements = payload.get("source_root_requirements", [])
    if not isinstance(source_root_requirements, list):
        raise SpecCompilerError(f"{relative_path.as_posix()} source_root_requirements must be a list when present")
    for index, requirement in enumerate(source_root_requirements):
        if not isinstance(requirement, str) or not requirement:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} source_root_requirements[{index}] must be a non-empty string"
            )
    if source_root_requirements and not all((root / requirement).exists() for requirement in source_root_requirements):
        return

    pass_condition_catalog = payload.get("pass_condition_catalog")
    if not isinstance(pass_condition_catalog, list) or not pass_condition_catalog:
        raise SpecCompilerError(f"{relative_path.as_posix()} must declare pass_condition_catalog")
    pass_condition_ids: set[str] = set()
    for index, entry in enumerate(pass_condition_catalog):
        if not isinstance(entry, dict):
            raise SpecCompilerError(f"{relative_path.as_posix()} pass_condition_catalog[{index}] must be an object")
        condition_id = entry.get("id")
        description = entry.get("description")
        if not isinstance(condition_id, str) or not condition_id:
            raise SpecCompilerError(f"{relative_path.as_posix()} pass_condition_catalog[{index}].id must be a string")
        if condition_id in pass_condition_ids:
            raise SpecCompilerError(f"{relative_path.as_posix()} duplicate pass_condition id '{condition_id}'")
        if not isinstance(description, str) or not description:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} pass_condition_catalog[{index}].description must be a string"
            )
        pass_condition_ids.add(condition_id)

    blocking_reason_catalog = payload.get("blocking_reason_catalog")
    if not isinstance(blocking_reason_catalog, list) or not blocking_reason_catalog:
        raise SpecCompilerError(f"{relative_path.as_posix()} must declare blocking_reason_catalog")
    blocking_reason_codes: set[str] = set()
    for index, entry in enumerate(blocking_reason_catalog):
        if not isinstance(entry, dict):
            raise SpecCompilerError(f"{relative_path.as_posix()} blocking_reason_catalog[{index}] must be an object")
        reason_code = entry.get("code")
        description = entry.get("description")
        if not isinstance(reason_code, str) or not reason_code:
            raise SpecCompilerError(f"{relative_path.as_posix()} blocking_reason_catalog[{index}].code must be a string")
        if reason_code in blocking_reason_codes:
            raise SpecCompilerError(f"{relative_path.as_posix()} duplicate blocking reason code '{reason_code}'")
        if not isinstance(description, str) or not description:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} blocking_reason_catalog[{index}].description must be a string"
            )
        blocking_reason_codes.add(reason_code)

    closure_areas = payload.get("closure_areas")
    if not isinstance(closure_areas, list) or not closure_areas:
        raise SpecCompilerError(f"{relative_path.as_posix()} must declare closure_areas")

    seen_area_ids: set[str] = set()
    for index, area in enumerate(closure_areas):
        if not isinstance(area, dict):
            raise SpecCompilerError(f"{relative_path.as_posix()} closure_areas[{index}] must be an object")

        area_id = area.get("area_id")
        title = area.get("title")
        if not isinstance(area_id, str) or not area_id:
            raise SpecCompilerError(f"{relative_path.as_posix()} closure_areas[{index}].area_id must be a string")
        if area_id in seen_area_ids:
            raise SpecCompilerError(f"{relative_path.as_posix()} duplicate closure area '{area_id}'")
        seen_area_ids.add(area_id)
        if not isinstance(title, str) or not title:
            raise SpecCompilerError(f"{relative_path.as_posix()} closure_areas[{index}].title must be a string")

        required_evidence = area.get("required_evidence")
        if not isinstance(required_evidence, list) or not required_evidence:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} closure_areas[{index}].required_evidence must be a non-empty list"
            )
        for evidence_index, evidence_ref in enumerate(required_evidence):
            if not isinstance(evidence_ref, dict):
                raise SpecCompilerError(
                    f"{relative_path.as_posix()} closure_areas[{index}].required_evidence[{evidence_index}] "
                    "must be an object"
                )
            evidence_kind = evidence_ref.get("kind")
            evidence_path = evidence_ref.get("path")
            if evidence_kind not in READINESS_EVIDENCE_KINDS:
                raise SpecCompilerError(
                    f"{relative_path.as_posix()} invalid readiness evidence kind '{evidence_kind}' for area '{area_id}'"
                )
            if not isinstance(evidence_path, str) or not evidence_path:
                raise SpecCompilerError(
                    f"{relative_path.as_posix()} closure_areas[{index}].required_evidence[{evidence_index}].path "
                    "must be a string"
                )
            if not (root / evidence_path).exists():
                raise SpecCompilerError(f"missing readiness evidence path: {evidence_path}")

        area_pass_condition_ids = area.get("pass_condition_ids")
        if not isinstance(area_pass_condition_ids, list) or not area_pass_condition_ids:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} closure_areas[{index}].pass_condition_ids must be a non-empty list"
            )
        missing_pass_conditions = sorted(set(area_pass_condition_ids) - pass_condition_ids)
        if missing_pass_conditions:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} unknown pass_condition_ids for area '{area_id}': "
                f"{', '.join(missing_pass_conditions)}"
            )

        area_blocking_reason_codes = area.get("blocking_reason_codes")
        if not isinstance(area_blocking_reason_codes, list) or not area_blocking_reason_codes:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} closure_areas[{index}].blocking_reason_codes must be a non-empty list"
            )
        missing_reason_codes = sorted(set(area_blocking_reason_codes) - blocking_reason_codes)
        if missing_reason_codes:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} unknown blocking_reason_codes for area '{area_id}': "
                f"{', '.join(missing_reason_codes)}"
            )


def _update_contract_lookup(relative_path: Path, payload: dict[str, Any], contract_lookup: dict[str, Any]) -> None:
    kind = payload.get("kind")

    if kind == "predictive_evidence_policies":
        contract_lookup["comparison_regimes"] = {
            entry["comparison_regime_id"]: entry for entry in payload.get("comparison_regimes", [])
        }
        contract_lookup["many_model_adjustments"] = {
            entry["adjustment_id"]: entry for entry in payload.get("many_model_adjustments", [])
        }
        contract_lookup["predictive_gate_policies"] = {
            entry["policy_id"]: entry for entry in payload.get("predictive_gate_policies", [])
        }
    elif kind == "scoring_policies":
        contract_lookup["scoring_by_forecast_object_type"] = {
            entry["forecast_object_type"]: entry for entry in payload.get("contracts", [])
        }
    elif kind == "calibration_policies":
        contract_lookup["calibration_by_forecast_object_type"] = {
            entry["forecast_object_type"]: entry for entry in payload.get("contracts", [])
        }
    elif kind == "mechanistic_evidence_requirements":
        contract_lookup["mechanistic_requirements"] = {
            entry["requirement_id"]: entry for entry in payload.get("contracts", [])
        }
        contract_lookup["mechanistic_global_rules"] = payload.get("global_rules", {})
    elif kind == "shared_plus_local_evaluation_rules":
        contract_lookup["shared_plus_local_rules"] = {
            entry["rule_id"]: entry for entry in payload.get("contracts", [])
        }
    elif kind == "composition_operator_semantics":
        contract_lookup["composition_operator_rules"] = {
            entry["composition_operator"]: entry for entry in payload.get("contracts", [])
        }


def _load_math_fixtures(
    root: Path,
    allowed_values: dict[str, set[str]],
    contract_lookup: dict[str, Any],
) -> list[dict[str, Any]]:
    directory = root / MATH_FIXTURE_RELATIVE_DIRECTORY
    if not directory.is_dir():
        return []

    summaries: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix not in STRUCTURED_FILE_SUFFIXES:
            continue

        relative_path = path.relative_to(root)
        payload = _load_structured_file(path)
        if payload.get("kind") != "canonical_math_fixture":
            raise SpecCompilerError(f"{relative_path.as_posix()} must declare kind: canonical_math_fixture")

        _validate_enum_values(relative_path, payload, allowed_values)
        _validate_math_fixture(relative_path, payload, contract_lookup)
        summaries.append(
            {
                "path": relative_path.as_posix(),
                "fixture_id": payload["fixture_id"],
                "claim_lane": payload["claim_lane"],
                "forecast_object_type": payload.get("forecast_object_type"),
                "composition_operator": payload["reducer_object"]["composition_operator"],
            }
        )

    return summaries


def _validate_math_fixture(
    relative_path: Path,
    payload: dict[str, Any],
    contract_lookup: dict[str, Any],
) -> None:
    required_objects = [
        "admitted_data_object",
        "target_transform_object",
        "quantization_object",
        "observation_model_object",
        "reference_description_object",
        "codelength_policy_object",
        "reducer_object",
        "descriptive_admissibility_object",
    ]
    for key in required_objects:
        if not isinstance(payload.get(key), dict):
            raise SpecCompilerError(f"{relative_path.as_posix()} must declare {key} as an object")

    object_registry = _collect_math_object_registry(relative_path, payload)
    _validate_math_object_references(relative_path, payload, set(object_registry))
    _validate_math_object_identity(relative_path, payload)
    _validate_predictive_fixture_requirements(relative_path, payload, contract_lookup)
    _validate_mechanistic_fixture_requirements(relative_path, payload, contract_lookup)
    _validate_shared_plus_local_fixture_requirements(relative_path, payload, contract_lookup)


def _collect_math_object_registry(relative_path: Path, payload: Any) -> dict[str, tuple[str, ...]]:
    registry: dict[str, tuple[str, ...]] = {}

    def walk(node: Any, trail: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            object_id = node.get("object_id")
            if isinstance(object_id, str):
                if object_id in registry:
                    original_trail = ".".join(registry[object_id])
                    raise SpecCompilerError(
                        f"duplicate math object '{object_id}' in {relative_path.as_posix()} at "
                        f"{'.'.join(trail)}; already declared at {original_trail}"
                    )
                registry[object_id] = trail
            for key, value in node.items():
                walk(value, (*trail, key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, (*trail, str(index)))

    walk(payload, tuple())
    return registry


def _validate_math_object_references(relative_path: Path, payload: Any, object_ids: set[str]) -> None:
    def walk(node: Any, trail: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith("_ref") and key not in OWNER_REFERENCE_FIELDS and isinstance(value, str):
                    if value not in object_ids:
                        dotted_trail = ".".join((*trail, key))
                        raise SpecCompilerError(
                            f"unknown math object '{value}' for {dotted_trail} in {relative_path.as_posix()}"
                        )
                if key.endswith("_refs") and isinstance(value, list):
                    for index, ref in enumerate(value):
                        if isinstance(ref, str) and ref not in object_ids:
                            dotted_trail = ".".join((*trail, key, str(index)))
                            raise SpecCompilerError(
                                f"unknown math object '{ref}' for {dotted_trail} in {relative_path.as_posix()}"
                            )
                walk(value, (*trail, key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, (*trail, str(index)))

    walk(payload, tuple())


def _validate_math_object_identity(relative_path: Path, payload: dict[str, Any]) -> None:
    observation_model = payload["observation_model_object"]
    reducer_object = payload["reducer_object"]
    descriptive_admissibility = payload["descriptive_admissibility_object"]

    if observation_model.get("forecast_object_type") != payload.get(
        "forecast_object_type", observation_model.get("forecast_object_type")
    ):
        raise SpecCompilerError(
            f"{relative_path.as_posix()} forecast_object_type must match observation_model_object.forecast_object_type"
        )

    if descriptive_admissibility.get("reducer_object_ref") != reducer_object.get("object_id"):
        raise SpecCompilerError(
            f"{relative_path.as_posix()} descriptive_admissibility_object.reducer_object_ref must point to reducer_object"
        )


def _validate_predictive_fixture_requirements(
    relative_path: Path,
    payload: dict[str, Any],
    contract_lookup: dict[str, Any],
) -> None:
    claim_lane = payload["claim_lane"]
    predictive_claim_lanes = {
        "predictively_supported",
        "predictive_within_declared_scope",
        "mechanistically_compatible_hypothesis",
        "mechanistically_compatible_law",
    }
    if claim_lane not in predictive_claim_lanes:
        return

    required_fields = [
        "forecast_object_type",
        "primary_score",
        "comparison_regime_id",
        "adjustment_id",
        "predictive_gate_policy_id",
        "time_safety_status",
        "confirmatory_status",
    ]
    for field in required_fields:
        if field not in payload:
            raise SpecCompilerError(f"{relative_path.as_posix()} missing required predictive field: {field}")

    forecast_object_type = payload["forecast_object_type"]
    scoring_entry = contract_lookup["scoring_by_forecast_object_type"].get(forecast_object_type)
    if scoring_entry is None:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} forecast_object_type '{forecast_object_type}' has no scoring contract"
        )
    if payload["primary_score"] not in scoring_entry["allowed_primary_scores"]:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} primary_score '{payload['primary_score']}' is not allowed for "
            f"forecast_object_type '{forecast_object_type}'"
        )

    comparison_regime = contract_lookup["comparison_regimes"].get(payload["comparison_regime_id"])
    if comparison_regime is None:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} unknown comparison_regime_id '{payload['comparison_regime_id']}'"
        )
    if forecast_object_type not in comparison_regime["allowed_forecast_object_types"]:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} comparison_regime_id '{payload['comparison_regime_id']}' does not admit "
            f"forecast_object_type '{forecast_object_type}'"
        )

    if payload["adjustment_id"] not in contract_lookup["many_model_adjustments"]:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} unknown adjustment_id '{payload['adjustment_id']}'"
        )

    predictive_gate = contract_lookup["predictive_gate_policies"].get(payload["predictive_gate_policy_id"])
    if predictive_gate is None:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} unknown predictive_gate_policy_id '{payload['predictive_gate_policy_id']}'"
        )
    if forecast_object_type not in predictive_gate["allowed_forecast_object_types"]:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} predictive_gate_policy_id '{payload['predictive_gate_policy_id']}' does "
            f"not admit forecast_object_type '{forecast_object_type}'"
        )
    if payload["time_safety_status"] not in predictive_gate["required_time_safety_statuses"]:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} time_safety_status '{payload['time_safety_status']}' does not satisfy "
            f"predictive_gate_policy_id '{payload['predictive_gate_policy_id']}'"
        )
    if payload["confirmatory_status"] not in predictive_gate["required_confirmatory_statuses"]:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} confirmatory_status '{payload['confirmatory_status']}' does not satisfy "
            f"predictive_gate_policy_id '{payload['predictive_gate_policy_id']}'"
        )

    calibration_entry = contract_lookup["calibration_by_forecast_object_type"].get(forecast_object_type)
    if calibration_entry is None:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} forecast_object_type '{forecast_object_type}' has no calibration contract"
        )

    calibration_diagnostics = payload.get("calibration_diagnostics", [])
    if calibration_entry["calibration_mode"] == "required":
        if not isinstance(calibration_diagnostics, list):
            raise SpecCompilerError(
                f"{relative_path.as_posix()} calibration_diagnostics must be a list for probabilistic fixtures"
            )
        missing = [
            diagnostic
            for diagnostic in calibration_entry["required_diagnostics"]
            if diagnostic not in calibration_diagnostics
        ]
        if missing:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} missing required calibration diagnostics: {', '.join(missing)}"
            )
    elif calibration_diagnostics:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} point fixtures may not declare calibration_diagnostics"
        )


def _validate_mechanistic_fixture_requirements(
    relative_path: Path,
    payload: dict[str, Any],
    contract_lookup: dict[str, Any],
) -> None:
    if payload["claim_lane"] not in {"mechanistically_compatible_hypothesis", "mechanistically_compatible_law"}:
        return

    expected_lower_lane = contract_lookup["mechanistic_global_rules"].get("requires_lower_lane_support")
    if payload.get("lower_claim_lane") != expected_lower_lane:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} lower_claim_lane must equal {expected_lower_lane!r}"
        )

    requirement_ids = payload.get("requirement_ids")
    if not isinstance(requirement_ids, list) or not requirement_ids:
        raise SpecCompilerError(f"{relative_path.as_posix()} mechanistic fixtures must declare requirement_ids")

    known_requirement_ids = set(contract_lookup["mechanistic_requirements"])
    missing = sorted(known_requirement_ids - set(requirement_ids))
    unknown = sorted(set(requirement_ids) - known_requirement_ids)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise SpecCompilerError(
            f"{relative_path.as_posix()} invalid mechanistic requirement_ids: {'; '.join(details)}"
        )


def _validate_shared_plus_local_fixture_requirements(
    relative_path: Path,
    payload: dict[str, Any],
    contract_lookup: dict[str, Any],
) -> None:
    reducer_object = payload["reducer_object"]
    if reducer_object["composition_operator"] != "shared_plus_local_decomposition":
        return

    required_fields = ["entity_index_set", "shared_component_ref", "local_component_refs", "unseen_entity_rule"]
    for field in required_fields:
        value = reducer_object.get(field)
        if not value:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} shared_plus_local_decomposition fixtures must declare {field}"
            )

    composition_rule = contract_lookup["composition_operator_rules"].get("shared_plus_local_decomposition")
    if composition_rule is None:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} missing composition semantics for shared_plus_local_decomposition"
        )

    rule_ids = payload.get("shared_plus_local_rule_ids")
    if not isinstance(rule_ids, list) or not rule_ids:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} shared_plus_local_decomposition fixtures must declare shared_plus_local_rule_ids"
        )

    known_rule_ids = set(contract_lookup["shared_plus_local_rules"])
    missing_rule_ids = sorted(known_rule_ids - set(rule_ids))
    unknown_rule_ids = sorted(set(rule_ids) - known_rule_ids)
    if missing_rule_ids or unknown_rule_ids:
        details = []
        if missing_rule_ids:
            details.append(f"missing {', '.join(missing_rule_ids)}")
        if unknown_rule_ids:
            details.append(f"unknown {', '.join(unknown_rule_ids)}")
        raise SpecCompilerError(
            f"{relative_path.as_posix()} invalid shared_plus_local_rule_ids: {'; '.join(details)}"
        )


def _collect_owner_declarations(relative_path: Path, payload: Any, owner_registry: dict[str, str]) -> None:
    owners = payload.get("owners", []) if isinstance(payload, dict) else []
    for owner in owners:
        owner_id = owner.get("id")
        if not owner_id:
            raise SpecCompilerError(f"{relative_path.as_posix()} owner entries must include an id")
        if owner_id in owner_registry:
            original_path = owner_registry[owner_id]
            raise SpecCompilerError(
                f"duplicate owner '{owner_id}' declared in {relative_path.as_posix()} and {original_path}"
            )
        owner_registry[owner_id] = relative_path.as_posix()


def _validate_enum_values(relative_path: Path, payload: Any, allowed_values: dict[str, set[str]]) -> None:
    def walk(node: Any, trail: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ENUM_FIELD_TO_VOCABULARY:
                    vocabulary_name = ENUM_FIELD_TO_VOCABULARY[key]
                    allowed = allowed_values[vocabulary_name]
                    for item in _coerce_to_list(value):
                        if item not in allowed:
                            dotted_trail = ".".join((*trail, key))
                            raise SpecCompilerError(
                                f"unknown enum '{item}' for {dotted_trail} in {relative_path.as_posix()}"
                            )
                walk(value, (*trail, key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, (*trail, str(index)))

    walk(payload, tuple())


def _validate_owner_references(relative_path: Path, payload: Any, owner_ids: set[str]) -> None:
    def walk(node: Any, trail: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in OWNER_REFERENCE_FIELDS and isinstance(value, str) and value not in owner_ids:
                    dotted_trail = ".".join((*trail, key))
                    raise SpecCompilerError(
                        f"unknown owner '{value}' for {dotted_trail} in {relative_path.as_posix()}"
                    )
                walk(value, (*trail, key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, (*trail, str(index)))

    walk(payload, tuple())


def _build_fixture_closure(root: Path) -> dict[str, Any] | None:
    coverage_path = root / FIXTURE_COVERAGE_RELATIVE_PATH
    if not coverage_path.is_file():
        return None

    coverage_plan = _load_structured_file(coverage_path)
    if coverage_plan.get("kind") != "canonical_fixture_coverage_matrix":
        raise SpecCompilerError(
            f"{FIXTURE_COVERAGE_RELATIVE_PATH.as_posix()} must declare kind: canonical_fixture_coverage_matrix"
        )

    scenarios = coverage_plan.get("scenarios", [])
    if not isinstance(scenarios, list) or not scenarios:
        raise SpecCompilerError(f"{FIXTURE_COVERAGE_RELATIVE_PATH.as_posix()} must declare scenarios")

    walkthrough_entries, walkthrough_lookup = _load_fixture_walkthroughs(root, scenarios)
    lifecycle_order = _load_publication_lifecycle_order(root)

    scenario_results: list[dict[str, Any]] = []
    global_object_ids: dict[str, str] = {}
    total_artifacts = 0
    total_typed_refs = 0

    for scenario in scenarios:
        scenario_result, bundle_object_ids, typed_ref_count = _validate_fixture_scenario(
            root,
            scenario,
            walkthrough_lookup[scenario["scenario_id"]],
            lifecycle_order,
        )
        bundle_path = scenario_result["bundle_path"]
        for object_id in bundle_object_ids:
            if object_id in global_object_ids:
                raise SpecCompilerError(
                    f"duplicate fixture object_id '{object_id}' declared in {bundle_path} and "
                    f"{global_object_ids[object_id]}"
                )
            global_object_ids[object_id] = bundle_path

        total_artifacts += scenario_result["artifact_count"]
        total_typed_refs += typed_ref_count
        scenario_results.append(scenario_result)

    return {
        "coverage_plan": {
            "path": FIXTURE_COVERAGE_RELATIVE_PATH.as_posix(),
            "coverage_plan_id": coverage_plan.get("coverage_plan_id"),
            "scenario_count": len(scenarios),
            "global_invariants": coverage_plan.get("global_invariants", []),
        },
        "walkthroughs": walkthrough_entries,
        "scenarios": scenario_results,
        "summary": {
            "all_scenarios_closed": True,
            "scenarios_checked": len(scenario_results),
            "bundles_loaded": len(scenario_results),
            "walkthroughs_checked": len(walkthrough_entries),
            "artifacts_checked": total_artifacts,
            "typed_refs_checked": total_typed_refs,
            "unique_object_ids": len(global_object_ids),
        },
    }


def _load_fixture_walkthroughs(
    root: Path,
    scenarios: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    directory = root / FIXTURE_WALKTHROUGH_RELATIVE_DIRECTORY
    if not directory.is_dir():
        raise SpecCompilerError(f"missing required reference: {FIXTURE_WALKTHROUGH_RELATIVE_DIRECTORY.as_posix()}")

    scenario_map = {scenario["scenario_id"]: scenario for scenario in scenarios}
    walkthrough_entries: list[dict[str, Any]] = []
    walkthrough_lookup: dict[str, str] = {}

    for path in sorted(directory.glob("*.md")):
        payload, body = _parse_front_matter_document(path)
        relative_path = path.relative_to(root).as_posix()
        scenario_ids = payload.get("scenario_ids", [])
        fixture_bundles = payload.get("fixture_bundles", [])
        body_scenario_ids = {
            match.group("scenario_id")
            for match in re.finditer(
                r"(?m)^\s*-\s+(?P<scenario_id>[A-Za-z0-9_.-]+):",
                body,
            )
        }
        unknown_body_scenario_ids = sorted(body_scenario_ids - set(scenario_map))
        if unknown_body_scenario_ids:
            raise SpecCompilerError(
                "walkthrough references unknown scenario "
                f"'{unknown_body_scenario_ids[0]}' in {relative_path}"
            )

        if not scenario_ids and not fixture_bundles:
            fixture_bundles = _infer_fixture_bundle_links(root, path, body)
            scenario_ids = sorted(
                scenario_id
                for scenario_id, scenario in scenario_map.items()
                if scenario["planned_fixture_bundle"] in fixture_bundles
            )

        if not isinstance(scenario_ids, list):
            raise SpecCompilerError(f"{relative_path} scenario_ids must be a list")
        if not isinstance(fixture_bundles, list):
            raise SpecCompilerError(f"{relative_path} fixture_bundles must be a list")

        if scenario_ids or fixture_bundles:
            expected_bundles: set[str] = set()
            for scenario_id in scenario_ids:
                if scenario_id not in scenario_map:
                    raise SpecCompilerError(
                        f"walkthrough references unknown scenario '{scenario_id}' in {relative_path}"
                    )
                if scenario_id in walkthrough_lookup:
                    raise SpecCompilerError(
                        f"duplicate walkthrough coverage for '{scenario_id}' in {relative_path}"
                    )
                expected_bundles.add(scenario_map[scenario_id]["planned_fixture_bundle"])
                walkthrough_lookup[scenario_id] = relative_path

            if set(fixture_bundles) != expected_bundles:
                raise SpecCompilerError(f"fixture walkthrough bundle mismatch in {relative_path}")

            for bundle_path in fixture_bundles:
                bundle_file = root / bundle_path
                if not bundle_file.is_file():
                    raise SpecCompilerError(f"missing referenced fixture bundle: {bundle_path}")
                relative_link = os.path.relpath(bundle_file, path.parent).replace(os.sep, "/")
                if relative_link not in body:
                    raise SpecCompilerError(
                        f"walkthrough missing direct fixture link for {bundle_path} in {relative_path}"
                    )

        walkthrough_entries.append(
            {
                "path": relative_path,
                "title": payload.get("title"),
                "scenario_count": len(scenario_ids),
                "scenario_ids": scenario_ids,
                "fixture_bundles": fixture_bundles,
            }
        )

    missing = sorted(set(scenario_map) - set(walkthrough_lookup))
    if missing:
        raise SpecCompilerError(f"fixture walkthrough coverage hole: {', '.join(missing)}")

    return walkthrough_entries, walkthrough_lookup


def _parse_front_matter_document(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text()
    match = DOC_FRONT_MATTER_PATTERN.match(text)
    relative_path = path.as_posix()
    if match is None:
        return {}, text

    front_matter = yaml.safe_load(match.group(1))
    if not isinstance(front_matter, dict):
        raise SpecCompilerError(f"{relative_path} front matter must deserialize to an object")
    return front_matter, match.group(2)


def _load_publication_lifecycle_order(root: Path) -> dict[str, int]:
    payload = _load_required_file(root, LIFECYCLE_RELATIVE_PATHS["publication_lifecycle"])
    return {
        state["state_id"]: index
        for index, state in enumerate(payload.get("states", []))
        if isinstance(state, dict) and isinstance(state.get("state_id"), str)
    }


def _validate_fixture_scenario(
    root: Path,
    scenario: dict[str, Any],
    walkthrough_path: str,
    lifecycle_order: dict[str, int],
) -> tuple[dict[str, Any], list[str], int]:
    relative_path = Path(scenario["planned_fixture_bundle"])
    bundle = _load_required_file(root, relative_path)

    if bundle.get("version") != 1:
        raise SpecCompilerError(f"{relative_path.as_posix()} must declare version: 1")
    if bundle.get("kind") != "canonical_publication_fixture_bundle":
        raise SpecCompilerError(f"{relative_path.as_posix()} must declare kind: canonical_publication_fixture_bundle")
    if bundle.get("scenario_id") != scenario["scenario_id"]:
        raise SpecCompilerError(f"{relative_path.as_posix()} scenario_id must match the coverage plan")
    if bundle.get("scenario_class") != scenario["scenario_class"]:
        raise SpecCompilerError(f"{relative_path.as_posix()} scenario_class must match the coverage plan")

    for field in ("required_modules", "required_schema_families", "required_contract_families", "required_evidence_classes"):
        bundle_values = bundle.get(field, [])
        if not isinstance(bundle_values, list):
            raise SpecCompilerError(f"{relative_path.as_posix()} {field} must be a list")
        missing = sorted(set(scenario.get(field, [])) - set(bundle_values))
        if missing:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} must include declared coverage for {field}: {', '.join(missing)}"
            )

    expected_outcome = bundle.get("expected_outcome")
    if not isinstance(expected_outcome, dict):
        raise SpecCompilerError(f"{relative_path.as_posix()} expected_outcome must be an object")

    expected_pairs = {
        "validator_disposition": scenario["validator_disposition"],
        "publication_mode": scenario["expected_publication_mode"],
        "terminal_lifecycle_state": scenario["expected_terminal_lifecycle_state"],
        "claim_lane": scenario["expected_claim_lane"],
        "forecast_object_type": scenario["expected_forecast_object_type"],
        "abstention_type": scenario["expected_abstention_type"],
        "failure_effect": scenario["expected_failure_effect"],
    }
    for key, expected_value in expected_pairs.items():
        observed_value = expected_outcome.get(key, "none" if key == "failure_effect" else None)
        if observed_value != expected_value:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} expected_outcome.{key} must match the coverage plan"
            )

    expected_reason_codes = scenario.get("expected_reason_codes")
    if expected_reason_codes is not None and sorted(expected_outcome.get("reason_codes", [])) != sorted(expected_reason_codes):
        raise SpecCompilerError(f"{relative_path.as_posix()} expected_outcome.reason_codes must match the coverage plan")

    artifact_registry, bundle_object_ids, artifact_schema_names, artifact_module_ids = _build_fixture_artifact_registry(
        relative_path,
        bundle,
    )

    declared_schema_families = set(bundle["required_schema_families"])
    if artifact_schema_names != declared_schema_families:
        undeclared = sorted(artifact_schema_names - declared_schema_families)
        unused = sorted(declared_schema_families - artifact_schema_names)
        details: list[str] = []
        if undeclared:
            details.append(f"undeclared schemas {', '.join(undeclared)}")
        if unused:
            details.append(f"unused declarations {', '.join(unused)}")
        raise SpecCompilerError(
            f"illegal fixture schema usage in {relative_path.as_posix()}: {'; '.join(details)}"
        )

    undeclared_modules = sorted(artifact_module_ids - set(bundle["required_modules"]))
    if undeclared_modules:
        raise SpecCompilerError(
            f"illegal fixture module usage in {relative_path.as_posix()}: {', '.join(undeclared_modules)}"
        )

    if scenario["scenario_class"] == "invalid_fixture":
        typed_ref_count = _validate_invalid_fixture_bundle(relative_path, bundle, artifact_registry)
        lifecycle_states: list[str] = []
    else:
        typed_ref_count = _validate_fixture_refs(relative_path, bundle, artifact_registry)
        lifecycle_states = _validate_fixture_lifecycle_trace(
            relative_path,
            bundle,
            expected_outcome,
            lifecycle_order,
            artifact_registry,
        )
        _validate_fixture_terminal_artifacts(relative_path, bundle, artifact_registry, expected_outcome)

    return (
        {
            "scenario_id": scenario["scenario_id"],
            "scenario_class": scenario["scenario_class"],
            "bundle_id": bundle.get("bundle_id"),
            "bundle_path": relative_path.as_posix(),
            "walkthrough_path": walkthrough_path,
            "status": "closed",
            "artifact_count": len(bundle.get("artifacts", [])),
            "typed_refs_checked": typed_ref_count,
            "terminal_lifecycle_state": expected_outcome["terminal_lifecycle_state"],
            "publication_mode": expected_outcome["publication_mode"],
            "validator_disposition": expected_outcome["validator_disposition"],
            "lifecycle_trace": lifecycle_states,
        },
        bundle_object_ids,
        typed_ref_count,
    )


def _build_fixture_artifact_registry(
    relative_path: Path,
    bundle: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str], set[str], set[str]]:
    artifacts = bundle.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        raise SpecCompilerError(f"{relative_path.as_posix()} artifacts must be a non-empty list")

    registry: dict[tuple[str, str], dict[str, Any]] = {}
    object_ids: list[str] = []
    seen_object_ids: set[str] = set()
    schema_names: set[str] = set()
    module_ids: set[str] = set()

    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise SpecCompilerError(f"{relative_path.as_posix()} artifacts[{index}] must be an object")
        schema_name = artifact.get("schema_name")
        object_id = artifact.get("object_id")
        module_id = artifact.get("module_id")
        if not isinstance(schema_name, str) or not schema_name:
            raise SpecCompilerError(f"{relative_path.as_posix()} artifacts[{index}].schema_name must be a string")
        if not isinstance(object_id, str) or not object_id:
            raise SpecCompilerError(f"{relative_path.as_posix()} artifacts[{index}].object_id must be a string")
        if not isinstance(module_id, str) or not module_id:
            raise SpecCompilerError(f"{relative_path.as_posix()} artifacts[{index}].module_id must be a string")
        if object_id in seen_object_ids:
            raise SpecCompilerError(f"duplicate fixture object_id '{object_id}' in {relative_path.as_posix()}")
        key = (schema_name, object_id)
        if key in registry:
            raise SpecCompilerError(
                f"duplicate fixture artifact identity '{schema_name}'/'{object_id}' in {relative_path.as_posix()}"
            )

        registry[key] = artifact
        seen_object_ids.add(object_id)
        object_ids.append(object_id)
        schema_names.add(schema_name)
        module_ids.add(module_id)

    return registry, object_ids, schema_names, module_ids


def _validate_fixture_refs(
    relative_path: Path,
    bundle: dict[str, Any],
    registry: dict[tuple[str, str], dict[str, Any]],
) -> int:
    typed_ref_count = 0

    def walk(node: Any, trail: tuple[str, ...]) -> None:
        nonlocal typed_ref_count

        if isinstance(node, dict):
            has_schema_name = "schema_name" in node
            has_object_id = "object_id" in node
            if has_schema_name or has_object_id:
                dotted_trail = ".".join(trail) or "<root>"
                if not (has_schema_name and has_object_id):
                    raise SpecCompilerError(
                        f"invalid typed ref shape in {relative_path.as_posix()} at {dotted_trail}"
                    )
                schema_name = node.get("schema_name")
                object_id = node.get("object_id")
                if not isinstance(schema_name, str) or not isinstance(object_id, str):
                    raise SpecCompilerError(
                        f"invalid typed ref shape in {relative_path.as_posix()} at {dotted_trail}"
                    )
                if (schema_name, object_id) not in registry:
                    raise SpecCompilerError(
                        f"unmaterialized fixture ref ({schema_name}, {object_id}) in {relative_path.as_posix()} "
                        f"at {dotted_trail}"
                    )
                typed_ref_count += 1
            for key, value in node.items():
                walk(value, (*trail, key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, (*trail, str(index)))

    walk(bundle, tuple())
    return typed_ref_count


def _validate_invalid_fixture_bundle(
    relative_path: Path,
    bundle: dict[str, Any],
    registry: dict[tuple[str, str], dict[str, Any]],
) -> int:
    if bundle.get("lifecycle_trace"):
        raise SpecCompilerError(f"{relative_path.as_posix()} invalid fixtures may not declare lifecycle_trace")

    validator_errors = bundle.get("validator_errors")
    if not isinstance(validator_errors, list) or not validator_errors:
        raise SpecCompilerError(f"{relative_path.as_posix()} invalid fixtures must declare validator_errors")

    for index, error in enumerate(validator_errors):
        if not isinstance(error, dict):
            raise SpecCompilerError(f"{relative_path.as_posix()} validator_errors[{index}] must be an object")
        if not isinstance(error.get("error_code"), str) or not error["error_code"]:
            raise SpecCompilerError(f"{relative_path.as_posix()} validator_errors[{index}].error_code must be a string")
        if not isinstance(error.get("message"), str) or not error["message"]:
            raise SpecCompilerError(f"{relative_path.as_posix()} validator_errors[{index}].message must be a string")

        artifact_ref = error.get("artifact_ref")
        if not isinstance(artifact_ref, dict):
            raise SpecCompilerError(f"{relative_path.as_posix()} validator_errors[{index}].artifact_ref must be an object")
        schema_name = artifact_ref.get("schema_name")
        object_id = artifact_ref.get("object_id")
        if not isinstance(schema_name, str) or not isinstance(object_id, str):
            raise SpecCompilerError(
                f"{relative_path.as_posix()} validator_errors[{index}].artifact_ref must be a typed ref"
            )
        if (schema_name, object_id) not in registry:
            raise SpecCompilerError(
                f"unmaterialized fixture ref ({schema_name}, {object_id}) in {relative_path.as_posix()} "
                f"at validator_errors.{index}.artifact_ref"
            )

    return len(validator_errors)


def _validate_fixture_lifecycle_trace(
    relative_path: Path,
    bundle: dict[str, Any],
    expected_outcome: dict[str, Any],
    lifecycle_order: dict[str, int],
    registry: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    lifecycle_trace = bundle.get("lifecycle_trace")
    if not isinstance(lifecycle_trace, list) or not lifecycle_trace:
        raise SpecCompilerError(f"{relative_path.as_posix()} lifecycle_trace must be a non-empty list")

    states: list[str] = []
    terminal_states = {"publication_completed", "publication_blocked"}

    for index, entry in enumerate(lifecycle_trace):
        if not isinstance(entry, dict):
            raise SpecCompilerError(f"{relative_path.as_posix()} lifecycle_trace[{index}] must be an object")
        state = entry.get("state")
        if not isinstance(state, str):
            raise SpecCompilerError(f"{relative_path.as_posix()} lifecycle_trace[{index}].state must be a string")
        if state not in lifecycle_order:
            raise SpecCompilerError(f"broken lifecycle sequence in {relative_path.as_posix()}: unknown state '{state}'")
        if index == 0 and state != "publication_requested":
            raise SpecCompilerError(
                f"broken lifecycle sequence in {relative_path.as_posix()}: traces must start at publication_requested"
            )
        if states:
            previous_state = states[-1]
            if previous_state in terminal_states:
                raise SpecCompilerError(
                    f"broken lifecycle sequence in {relative_path.as_posix()}: no state may follow {previous_state}"
                )
            if lifecycle_order[state] < lifecycle_order[previous_state]:
                raise SpecCompilerError(
                    f"broken lifecycle sequence in {relative_path.as_posix()}: '{state}' may not follow "
                    f"'{previous_state}'"
                )
        states.append(state)

        produced_refs = entry.get("produced_artifact_refs", [])
        if not isinstance(produced_refs, list):
            raise SpecCompilerError(
                f"{relative_path.as_posix()} lifecycle_trace[{index}].produced_artifact_refs must be a list"
            )
        for ref_index, ref in enumerate(produced_refs):
            if not isinstance(ref, dict):
                raise SpecCompilerError(
                    f"{relative_path.as_posix()} lifecycle_trace[{index}].produced_artifact_refs[{ref_index}] "
                    "must be a typed ref"
                )
            schema_name = ref.get("schema_name")
            object_id = ref.get("object_id")
            if not isinstance(schema_name, str) or not isinstance(object_id, str):
                raise SpecCompilerError(
                    f"{relative_path.as_posix()} lifecycle_trace[{index}].produced_artifact_refs[{ref_index}] "
                    "must be a typed ref"
                )
            if (schema_name, object_id) not in registry:
                raise SpecCompilerError(
                    f"unmaterialized fixture ref ({schema_name}, {object_id}) in {relative_path.as_posix()} "
                    f"at lifecycle_trace.{index}.produced_artifact_refs.{ref_index}"
                )

    if states[-1] != expected_outcome["terminal_lifecycle_state"]:
        raise SpecCompilerError(
            f"broken lifecycle sequence in {relative_path.as_posix()}: terminal state must be "
            f"{expected_outcome['terminal_lifecycle_state']}"
        )

    publication_mode = expected_outcome["publication_mode"]
    if publication_mode == "candidate_publication" and "candidate_publication_selected" not in states:
        raise SpecCompilerError(
            f"broken lifecycle sequence in {relative_path.as_posix()}: candidate publication must select "
            "candidate_publication_selected"
        )
    if publication_mode == "abstention_only_publication" and "abstention_only_publication_selected" not in states:
        raise SpecCompilerError(
            f"broken lifecycle sequence in {relative_path.as_posix()}: abstention publication must select "
            "abstention_only_publication_selected"
        )
    if "replay_verified" in states and "replay_bundle_assembled" not in states:
        raise SpecCompilerError(
            f"broken lifecycle sequence in {relative_path.as_posix()}: replay_verified requires replay_bundle_assembled"
        )
    if "publication_record_written" in states and "replay_verified" not in states:
        raise SpecCompilerError(
            f"broken lifecycle sequence in {relative_path.as_posix()}: publication_record_written requires replay_verified"
        )
    if "catalog_projected" in states and "publication_record_written" not in states:
        raise SpecCompilerError(
            f"broken lifecycle sequence in {relative_path.as_posix()}: catalog_projected requires publication_record_written"
        )
    if states[-1] == "publication_completed" and "publication_record_written" not in states:
        raise SpecCompilerError(
            f"broken lifecycle sequence in {relative_path.as_posix()}: publication_completed requires publication_record_written"
        )

    return states


def _validate_fixture_terminal_artifacts(
    relative_path: Path,
    bundle: dict[str, Any],
    registry: dict[tuple[str, str], dict[str, Any]],
    expected_outcome: dict[str, Any],
) -> None:
    artifacts = list(registry.values())
    run_results = [artifact for artifact in artifacts if artifact["schema_name"] == "run_result_manifest@1.1.0"]
    publication_records = [
        artifact for artifact in artifacts if artifact["schema_name"] == "publication_record_manifest@1.1.0"
    ]
    claim_cards = [artifact for artifact in artifacts if artifact["schema_name"] == "claim_card_manifest@1.1.0"]
    abstentions = [artifact for artifact in artifacts if artifact["schema_name"] == "abstention_manifest@1.1.0"]

    if run_results:
        if run_results[0]["body"].get("result_mode") != expected_outcome["publication_mode"]:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} run_result_manifest result_mode must match expected_outcome.publication_mode"
            )

    if expected_outcome["terminal_lifecycle_state"] == "publication_completed" and not publication_records:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} publication_completed scenarios must materialize publication_record_manifest"
        )

    if publication_records:
        publication_record = publication_records[0]
        if publication_record["body"].get("publication_mode") != expected_outcome["publication_mode"]:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} publication_record_manifest publication_mode must match expected_outcome"
            )
        run_result_ref = publication_record["body"].get("run_result_ref")
        if isinstance(run_result_ref, dict):
            key = (run_result_ref.get("schema_name"), run_result_ref.get("object_id"))
            if key not in registry:
                raise SpecCompilerError(
                    f"unmaterialized fixture ref {key} in {relative_path.as_posix()} at publication_record.run_result_ref"
                )

    expected_abstention_type = expected_outcome["abstention_type"]
    expected_claim_lane = expected_outcome["claim_lane"]
    if expected_abstention_type is not None:
        if len(abstentions) != 1 or claim_cards:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} abstention-only scenarios must emit one abstention and no claim card"
            )
        if abstentions[0]["body"].get("abstention_type") != expected_abstention_type:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} abstention_manifest abstention_type must match expected_outcome"
            )
    elif expected_claim_lane is not None:
        if len(claim_cards) != 1:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} claim-bearing scenarios must emit exactly one claim card"
            )
        claim_card = claim_cards[0]
        if claim_card["body"].get("claim_lane") != expected_claim_lane:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} claim_card_manifest claim_lane must match expected_outcome"
            )
        if claim_card["body"].get("forecast_object_type") != expected_outcome["forecast_object_type"]:
            raise SpecCompilerError(
                f"{relative_path.as_posix()} claim_card_manifest forecast_object_type must match expected_outcome"
            )
    elif claim_cards or abstentions:
        raise SpecCompilerError(
            f"{relative_path.as_posix()} blocked scenarios may not emit claim cards or abstention manifests"
        )


def _load_document_summary(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        raise SpecCompilerError(f"missing required reference: {relative_path}")

    front_matter, body = _parse_front_matter_document(path)
    related = front_matter.get("related", [])
    if related and not isinstance(related, list):
        raise SpecCompilerError(f"{relative_path} related front matter must be a list")
    if not related:
        related = _collect_local_markdown_links(root, Path(relative_path), body)

    title = front_matter.get("title")
    if not isinstance(title, str) or not title:
        heading_match = HEADING_PATTERN.search(body)
        title = heading_match.group(1).strip() if heading_match is not None else Path(relative_path).stem

    return {
        "path": relative_path,
        "title": title,
        "related": related,
    }


def _iter_contract_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for relative_directory in (Path("schemas/contracts"), Path("schemas/readiness")):
        directory = root / relative_directory
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in STRUCTURED_FILE_SUFFIXES:
                paths.append(path.relative_to(root))
    return paths


def _load_optional_file(root: Path, relative_path: Path) -> dict[str, Any] | None:
    path = root / relative_path
    if not path.is_file():
        return None
    return _load_structured_file(path)


def _load_required_file(root: Path, relative_path: Path) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        raise SpecCompilerError(f"missing required reference: {relative_path.as_posix()}")
    return _load_structured_file(path)


def _load_structured_file(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def _coerce_to_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Euclid Canonical Pack",
        "",
        "## Project",
        f"- Name: {payload['project_name']}",
        f"- Scope: {payload['scope_statement']}",
        "",
        "## Canonical Docs",
    ]

    for doc in payload["canonical_docs"]:
        related = ", ".join(doc["related"]) if doc["related"] else "None"
        lines.append(f"- {doc['title']} (`{doc['path']}`) | Related: {related}")

    lines.extend(["", "## Vocabularies"])
    for name, summary in payload["vocabularies"].items():
        entry_ids = ", ".join(summary["entry_ids"])
        lines.append(f"- {name}: `{summary['path']}` -> {entry_ids}")

    lines.extend(["", "## Runtime Modules"])
    for family, modules in payload["runtime_modules"].items():
        lines.append(f"- {family}: {', '.join(modules)}")

    lines.extend(["", "## Artifact Classes"])
    for family, artifacts in payload["artifact_classes"].items():
        lines.append(f"- {family}: {', '.join(artifacts)}")

    lines.extend(["", "## Contract Artifacts"])
    if payload["contracts"]:
        for contract in payload["contracts"]:
            owners = ", ".join(contract["owners"]) if contract["owners"] else "None"
            modules = ", ".join(contract["contract_modules"]) if contract["contract_modules"] else "None"
            lines.append(
                f"- `{contract['path']}` ({contract['kind']}) | Owners: {owners} | Modules: {modules}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Math Fixtures"])
    if payload["math_fixtures"]:
        for fixture in payload["math_fixtures"]:
            forecast_object_type = fixture["forecast_object_type"] or "None"
            lines.append(
                f"- `{fixture['path']}` | claim_lane={fixture['claim_lane']} | "
                f"forecast_object_type={forecast_object_type} | "
                f"composition_operator={fixture['composition_operator']}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Validation Summary"])
    for key, value in payload["validation_summary"].items():
        lines.append(f"- {key}: {value}")

    return "\n".join(lines) + "\n"


def _render_fixture_closure_markdown(payload: dict[str, Any]) -> str:
    coverage_plan = payload["coverage_plan"]
    summary = payload["summary"]

    lines = [
        "# Euclid Fixture Closure",
        "",
        "## Coverage Plan",
        f"- Path: `{coverage_plan['path']}`",
        f"- Coverage Plan ID: `{coverage_plan['coverage_plan_id']}`",
        f"- Scenarios: {coverage_plan['scenario_count']}",
        "",
        "## Summary",
        f"- all_scenarios_closed: {summary['all_scenarios_closed']}",
        f"- scenarios_checked: {summary['scenarios_checked']}",
        f"- bundles_loaded: {summary['bundles_loaded']}",
        f"- walkthroughs_checked: {summary['walkthroughs_checked']}",
        f"- artifacts_checked: {summary['artifacts_checked']}",
        f"- typed_refs_checked: {summary['typed_refs_checked']}",
        f"- unique_object_ids: {summary['unique_object_ids']}",
        "",
        "## Scenario Report",
    ]

    for scenario in payload["scenarios"]:
        lifecycle_trace = " -> ".join(scenario["lifecycle_trace"]) if scenario["lifecycle_trace"] else "validator_rejected"
        lines.append(
            f"- `{scenario['scenario_id']}` | class={scenario['scenario_class']} | status={scenario['status']} | "
            f"terminal={scenario['terminal_lifecycle_state']} | bundle=`{scenario['bundle_path']}` | "
            f"walkthrough=`{scenario['walkthrough_path']}` | artifacts={scenario['artifact_count']} | "
            f"typed_refs={scenario['typed_refs_checked']} | trace={lifecycle_trace}"
        )

    lines.extend(["", "## Walkthroughs"])
    for walkthrough in payload["walkthroughs"]:
        lines.append(
            f"- `{walkthrough['path']}` | scenarios={walkthrough['scenario_count']} | "
            f"bundles={len(walkthrough['fixture_bundles'])}"
        )

    return "\n".join(lines) + "\n"


def _build_readiness_pack(
    root: Path,
    system_schema: dict[str, Any],
    contract_graph_payload: dict[str, Any],
    fixture_closure_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not (root / READINESS_CONTRACT_RELATIVE_PATH).is_file():
        return None

    readiness_contract = _load_required_file(root, READINESS_CONTRACT_RELATIVE_PATH)
    if readiness_contract.get("kind") != "euclid_readiness_contract":
        return None

    live_doc_paths = _collect_live_doc_paths(root)
    doc_title_index = _collect_doc_title_index(root, live_doc_paths)
    known_machine_tokens = _collect_known_machine_tokens(
        root,
        system_schema,
        contract_graph_payload,
        readiness_contract,
    )
    findings = _audit_live_docs(root, live_doc_paths, doc_title_index, known_machine_tokens)

    closure_areas: list[dict[str, Any]] = []
    blocking_reasons: list[dict[str, Any]] = []
    blocker_keys: set[str] = set()
    failed_areas = 0

    for area in readiness_contract.get("closure_areas", []):
        area_id = area["area_id"]
        evidence_entries: list[dict[str, Any]] = []
        evidence_paths: set[str] = set()

        for evidence_ref in area.get("required_evidence", []):
            evidence_path = evidence_ref["path"]
            exists = (root / evidence_path).exists()
            evidence_entry = {
                "kind": evidence_ref["kind"],
                "path": evidence_path,
                "exists": exists,
            }
            evidence_entries.append(evidence_entry)
            evidence_paths.add(evidence_path)
            if not exists:
                code = _default_blocking_code_for_area(area_id, evidence_ref["kind"], area["blocking_reason_codes"])
                if code is not None:
                    blocker = {
                        "code": code,
                        "area_id": area_id,
                        "path": evidence_path,
                        "message": f"missing required readiness evidence: {evidence_path}",
                    }
                    blocker_key = json.dumps(blocker, sort_keys=True)
                    if blocker_key not in blocker_keys:
                        blocker_keys.add(blocker_key)
                        blocking_reasons.append(blocker)

        area_findings = [finding for finding in findings if finding["path"] in evidence_paths]
        area_blocking_reasons: list[dict[str, Any]] = []
        for finding in area_findings:
            code = _map_finding_to_blocking_code(area_id, finding, area["blocking_reason_codes"])
            if code is None:
                continue
            blocker = {
                "code": code,
                "area_id": area_id,
                "path": finding["path"],
                "message": finding["message"],
                "finding_kind": finding["kind"],
            }
            blocker_key = json.dumps(blocker, sort_keys=True)
            if blocker_key not in blocker_keys:
                blocker_keys.add(blocker_key)
                blocking_reasons.append(blocker)
            area_blocking_reasons.append(blocker)

        pass_condition_results = []
        for condition_id in area.get("pass_condition_ids", []):
            if condition_id == "evidence_paths_exist":
                passed = all(entry["exists"] for entry in evidence_entries)
            elif condition_id == "fixture_inventory_and_walkthroughs_present":
                passed = not area_findings and fixture_closure_payload is not None
            else:
                passed = not area_findings
            pass_condition_results.append({"id": condition_id, "passed": passed})

        status = "passed" if all(result["passed"] for result in pass_condition_results) and not area_blocking_reasons else "failed"
        if status == "failed":
            failed_areas += 1

        closure_areas.append(
            {
                "area_id": area_id,
                "title": area["title"],
                "status": status,
                "required_evidence": evidence_entries,
                "pass_condition_results": pass_condition_results,
                "blocking_reasons": area_blocking_reasons,
                "audit_findings": area_findings,
            }
        )

    final_verdict = "ready" if not blocking_reasons and failed_areas == 0 else "not_ready"
    payload = {
        "source_root": root.as_posix(),
        "contract_path": READINESS_CONTRACT_RELATIVE_PATH.as_posix(),
        "contract_id": readiness_contract["contract_id"],
        "pass_rule": readiness_contract.get("pass_rule"),
        "final_verdict": final_verdict,
        "closure_areas": closure_areas,
        "blocking_reasons": blocking_reasons,
        "audits": _summarize_audit_findings(findings),
        "summary": {
            "closure_areas_checked": len(closure_areas),
            "passed_area_count": len(closure_areas) - failed_areas,
            "failed_area_count": failed_areas,
            "blocking_reason_count": len(blocking_reasons),
            "audit_finding_count": len(findings),
            "live_doc_count": len(live_doc_paths),
            "fixture_closure_checked": fixture_closure_payload is not None,
        },
    }

    if final_verdict != "ready":
        raise SpecCompilerError(_readiness_failure_message(payload))

    return payload


def _collect_live_doc_paths(root: Path) -> list[Path]:
    live_doc_paths = {relative_path for relative_path in LIVE_ENTRY_RELATIVE_PATHS if (root / relative_path).is_file()}
    docs_root = root / "docs"
    if docs_root.is_dir():
        for path in docs_root.rglob("*.md"):
            live_doc_paths.add(path.relative_to(root))
    return sorted(live_doc_paths)


def _infer_fixture_bundle_links(root: Path, doc_path: Path, body: str) -> list[str]:
    fixture_bundles: list[str] = []
    for match in MARKDOWN_LINK_PATTERN.finditer(body):
        raw_target = _normalize_markdown_link_target(match.group(1))
        if not raw_target or LINK_SCHEME_PATTERN.match(raw_target):
            continue
        normalized_target = raw_target.split("#", 1)[0]
        if not normalized_target.endswith((".yaml", ".yml")):
            continue
        resolved_target = (doc_path.parent / normalized_target).resolve()
        try:
            relative_target = resolved_target.relative_to(root).as_posix()
        except ValueError:
            continue
        if relative_target.startswith("fixtures/"):
            fixture_bundles.append(relative_target)
    return sorted(dict.fromkeys(fixture_bundles))


def _collect_local_markdown_links(root: Path, relative_path: Path, body: str) -> list[str]:
    related: list[str] = []
    for match in MARKDOWN_LINK_PATTERN.finditer(body):
        raw_target = _normalize_markdown_link_target(match.group(1))
        if not raw_target or raw_target.startswith("#") or LINK_SCHEME_PATTERN.match(raw_target):
            continue
        normalized_target = raw_target.split("#", 1)[0]
        resolved_target = (root / relative_path).parent.joinpath(normalized_target).resolve()
        try:
            relative_target = resolved_target.relative_to(root).as_posix()
        except ValueError:
            continue
        if resolved_target.exists():
            related.append(relative_target)
    return sorted(dict.fromkeys(related))


def _collect_doc_title_index(root: Path, relative_paths: list[Path]) -> dict[str, str]:
    title_index: dict[str, str] = {}
    for relative_path in relative_paths:
        text = (root / relative_path).read_text()
        title = None
        if text.startswith("---\n"):
            match = DOC_FRONT_MATTER_PATTERN.match(text)
            if match is not None:
                front_matter = yaml.safe_load(match.group(1))
                if isinstance(front_matter, dict):
                    candidate = front_matter.get("title")
                    if isinstance(candidate, str) and candidate:
                        title = candidate
        if title is None:
            heading_match = HEADING_PATTERN.search(text)
            if heading_match is not None:
                title = heading_match.group(1).strip()
        if title is None:
            title = relative_path.stem.replace("-", " ").title()
        title_index[_normalize_doc_title(title)] = relative_path.as_posix()
    return title_index


def _normalize_doc_title(title: str) -> str:
    return " ".join(title.replace("-", " ").split()).casefold()


def _collect_known_machine_tokens(
    root: Path,
    system_schema: dict[str, Any],
    contract_graph_payload: dict[str, Any],
    readiness_contract: dict[str, Any],
) -> set[str]:
    tokens: set[str] = set()
    for modules in system_schema.get("major_runtime_modules", {}).values():
        tokens.update(str(module) for module in modules)
    for artifacts in system_schema.get("artifact_classes", {}).values():
        tokens.update(str(artifact) for artifact in artifacts)

    for vocabulary_path in system_schema.get("vocabulary_refs", {}).values():
        payload = _load_required_file(root, Path(vocabulary_path))
        tokens.update(str(entry["id"]) for entry in payload.get("entries", []) if isinstance(entry, dict) and "id" in entry)

    for schema in contract_graph_payload.get("schemas", []):
        tokens.add(schema["schema_name"])
    for enum_entry in contract_graph_payload.get("enums", []):
        tokens.add(enum_entry["enum_name"])
        tokens.update(enum_entry.get("allowed_values", []))
    for lifecycle in contract_graph_payload.get("lifecycles", []):
        tokens.update(lifecycle.get("state_ids", []))
        tokens.update(lifecycle.get("transition_ids", []))

    tokens.add(readiness_contract["contract_id"])
    tokens.add(readiness_contract["readiness_judgment_schema"])
    pass_rule = readiness_contract.get("pass_rule")
    if isinstance(pass_rule, str) and pass_rule:
        tokens.add(pass_rule)
    tokens.update(readiness_contract.get("area_status_values", []))
    tokens.update(readiness_contract.get("final_verdict_values", []))
    for entry in readiness_contract.get("pass_condition_catalog", []):
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            tokens.add(entry["id"])
    for entry in readiness_contract.get("blocking_reason_catalog", []):
        if isinstance(entry, dict) and isinstance(entry.get("code"), str):
            tokens.add(entry["code"])
    for area in readiness_contract.get("closure_areas", []):
        if isinstance(area, dict) and isinstance(area.get("area_id"), str):
            tokens.add(area["area_id"])

    for relative_path in _iter_contract_paths(root):
        _collect_machine_tokens_from_payload(_load_structured_file(root / relative_path), tokens)

    fixture_root = root / "fixtures"
    if fixture_root.is_dir():
        for path in fixture_root.rglob("*"):
            if path.is_file() and path.suffix in STRUCTURED_FILE_SUFFIXES:
                _collect_machine_tokens_from_payload(_load_structured_file(path), tokens)

    return tokens


def _collect_machine_tokens_from_payload(payload: Any, tokens: set[str]) -> None:
    if isinstance(payload, dict):
        for value in payload.values():
            _collect_machine_tokens_from_payload(value, tokens)
        return
    if isinstance(payload, list):
        for item in payload:
            _collect_machine_tokens_from_payload(item, tokens)
        return
    if isinstance(payload, str) and _looks_like_machine_token(payload):
        tokens.add(payload)


def _looks_like_machine_token(value: str) -> bool:
    if not value or " " in value or "/" in value or value.endswith((".md", ".yaml", ".yml", ".json")):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_@.\-]+", value))


def _audit_live_docs(
    root: Path,
    relative_paths: list[Path],
    doc_title_index: dict[str, str],
    known_machine_tokens: set[str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()

    def record(kind: str, path: Path, message: str, *, line: int | None = None, token: str | None = None) -> None:
        finding = {
            "kind": kind,
            "path": path.as_posix(),
            "message": message,
        }
        if line is not None:
            finding["line"] = line
        if token is not None:
            finding["token"] = token
        finding_key = json.dumps(finding, sort_keys=True)
        if finding_key not in seen:
            seen.add(finding_key)
            findings.append(finding)

    generated_outputs = {
        "euclid-canonical-pack.md",
        "euclid-canonical-pack.json",
        "euclid-contract-graph.md",
        "euclid-contract-graph.json",
        "euclid-fixture-closure.md",
        "euclid-fixture-closure.json",
        "euclid-readiness-pack.md",
        "euclid-readiness-pack.json",
    }

    for relative_path in relative_paths:
        text = (root / relative_path).read_text()
        relative_posix = relative_path.as_posix()
        normalized_text = text.casefold()

        for fragment in ENTRYPOINT_REQUIRED_FRAGMENTS.get(relative_posix, ()):
            if fragment not in text:
                record(
                    "missing_canonical_entry_fragment",
                    relative_path,
                    f"missing canonical entry fragment '{fragment}' in {relative_posix}",
                )

        for phrase in DISALLOWED_LIVE_AUTHORITY_PHRASES.get(relative_posix, ()):
            if phrase.casefold() in normalized_text:
                record(
                    "stale_authority_phrase",
                    relative_path,
                    f"stale live-authority phrase '{phrase}' in {relative_posix}",
                )

        for match in WIKI_LINK_PATTERN.finditer(text):
            target = _normalize_wiki_link_target(match.group(1))
            if target and _normalize_doc_title(target) not in doc_title_index:
                record(
                    "unresolved_live_doc_reference",
                    relative_path,
                    f"unresolved live doc reference '{target}' in {relative_posix}",
                    line=_line_number_for_offset(text, match.start()),
                    token=target,
                )

        for match in MARKDOWN_LINK_PATTERN.finditer(text):
            raw_target = match.group(1).strip()
            resolved_target = _normalize_markdown_link_target(raw_target)
            if not resolved_target or resolved_target.startswith("#") or LINK_SCHEME_PATTERN.match(resolved_target):
                continue
            if Path(resolved_target).name in generated_outputs and "build/" in resolved_target:
                continue
            error = _resolve_local_markdown_target(root, relative_path, resolved_target)
            if error is not None:
                record(
                    "unresolved_live_doc_reference",
                    relative_path,
                    error,
                    line=_line_number_for_offset(text, match.start()),
                    token=resolved_target,
                )

        for line_number, token in _iter_semantic_doc_tokens(text):
            if token not in known_machine_tokens:
                record(
                    "prose_only_semantic_token",
                    relative_path,
                    f"prose-only semantic token '{token}' in {relative_posix}",
                    line=line_number,
                    token=token,
                )

        for line_number in _iter_synthetic_readiness_object_lines(text):
            record(
                "synthetic_readiness_object",
                relative_path,
                f"synthetic readiness object in {relative_posix}; live docs must route through build/euclid-readiness-pack.* instead of embedding manifest instances",
                line=line_number,
                token=READINESS_MANIFEST_SCHEMA,
            )

    return findings


def _normalize_wiki_link_target(raw_target: str) -> str:
    target = raw_target.split("|", 1)[0].split("#", 1)[0].strip()
    return " ".join(target.split())


def _normalize_markdown_link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    return target.split(" ", 1)[0]


def _resolve_local_markdown_target(root: Path, relative_path: Path, link_target: str) -> str | None:
    normalized_target = link_target.split("#", 1)[0]
    if not normalized_target:
        return None
    if normalized_target.startswith("/"):
        absolute_target = Path(normalized_target)
        try:
            absolute_target.relative_to(root)
        except ValueError:
            return f"unresolved live doc reference '{link_target}' in {relative_path.as_posix()}: outside source root"
        if not absolute_target.exists():
            return f"unresolved live doc reference '{link_target}' in {relative_path.as_posix()}"
        return None
    resolved_target = (root / relative_path).parent.joinpath(normalized_target).resolve()
    if not resolved_target.exists():
        return f"unresolved live doc reference '{link_target}' in {relative_path.as_posix()}"
    return None


def _iter_semantic_doc_tokens(text: str) -> list[tuple[int, str]]:
    tokens: list[tuple[int, str]] = []
    in_code_block = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not SEMANTIC_SENTINEL_PATTERN.search(line):
            continue
        for token in INLINE_CODE_PATTERN.findall(line):
            if _looks_like_machine_token(token):
                tokens.append((line_number, token))
    return tokens


def _iter_synthetic_readiness_object_lines(text: str) -> list[int]:
    lines: list[int] = []
    for match in FENCED_CODE_BLOCK_PATTERN.finditer(text):
        block = match.group(1)
        if READINESS_MANIFEST_SCHEMA not in block:
            continue
        if not all(pattern.search(block) for pattern in READINESS_OBJECT_FIELD_PATTERNS):
            continue
        lines.append(_line_number_for_offset(text, match.start(1)))
    return lines


def _default_blocking_code_for_area(area_id: str, evidence_kind: str, allowed_codes: list[str]) -> str | None:
    preferred_codes = {
        "canonical_entrypoints": {
            "doc": "missing_canonical_entrypoint",
            "schema": "missing_source_map_target",
        },
        "canonical_math": {
            "doc": "missing_math_surface",
            "schema": "missing_closed_vocabulary",
        },
        "runtime_architecture": {
            "doc": "missing_architecture_surface",
            "schema": "missing_lifecycle_contract",
        },
        "runtime_contracts": {
            "schema": "missing_runtime_contract",
        },
        "extension_surfaces": {
            "doc": "missing_extension_contract",
            "schema": "missing_extension_contract",
            "fixture": "missing_fixture_bundle",
        },
        "fixtures_and_examples": {
            "doc": "missing_fixture_walkthrough",
            "fixture": "missing_fixture_bundle",
        },
    }
    candidate = preferred_codes.get(area_id, {}).get(evidence_kind)
    if candidate in allowed_codes:
        return candidate
    for code in preferred_codes.get(area_id, {}).values():
        if code in allowed_codes:
            return code
    return None


def _map_finding_to_blocking_code(area_id: str, finding: dict[str, Any], allowed_codes: list[str]) -> str | None:
    if finding["kind"] == "stale_authority_phrase" and "historical_readiness_authority_exposed" in allowed_codes:
        return "historical_readiness_authority_exposed"
    if finding["kind"] == "missing_canonical_entry_fragment" and "missing_canonical_entrypoint" in allowed_codes:
        return "missing_canonical_entrypoint"
    if finding["kind"] in {"unresolved_live_doc_reference", "prose_only_semantic_token", "synthetic_readiness_object"}:
        return _default_blocking_code_for_area(area_id, "doc", allowed_codes)
    return None


def _summarize_audit_findings(findings: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding["kind"]] = counts.get(finding["kind"], 0) + 1
    return {
        "categories": [
            {"kind": kind, "count": count}
            for kind, count in sorted(counts.items())
        ],
        "findings": findings,
    }


def _readiness_failure_message(payload: dict[str, Any]) -> str:
    findings = payload["audits"]["findings"]
    if findings:
        return findings[0]["message"]
    if payload["blocking_reasons"]:
        return payload["blocking_reasons"][0]["message"]
    return "readiness pack failed"


def _line_number_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _render_readiness_pack_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Euclid Readiness Pack",
        "",
        "## Summary",
        f"- Contract: `{payload['contract_id']}`",
        f"- Pass rule: `{payload.get('pass_rule')}`",
        f"- Final verdict: `{payload['final_verdict']}`",
    ]

    for key, value in payload["summary"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Closure Areas"])
    for area in payload["closure_areas"]:
        lines.append(f"- `{area['area_id']}` | status={area['status']} | title={area['title']}")
        for condition in area["pass_condition_results"]:
            lines.append(f"  - pass_condition `{condition['id']}`: {condition['passed']}")
        for blocker in area["blocking_reasons"]:
            lines.append(f"  - blocker `{blocker['code']}`: {blocker['message']}")

    lines.extend(["", "## Audit Findings"])
    findings = payload["audits"]["findings"]
    if findings:
        for finding in findings:
            detail = f" | line={finding['line']}" if "line" in finding else ""
            lines.append(f"- `{finding['kind']}` | `{finding['path']}`{detail} | {finding['message']}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"
