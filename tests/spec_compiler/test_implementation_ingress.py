from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_REGISTRY_PATH = REPO_ROOT / "schemas/contracts/module-registry.yaml"
IMPLEMENTATION_SEQUENCE_PATH = REPO_ROOT / "schemas/contracts/implementation-sequence.yaml"
MODELING_PIPELINE_DOC_PATH = REPO_ROOT / "docs/reference/modeling-pipeline.md"
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"

EXPECTED_LAYER_ORDER = [
    ("registration_and_intake", ["manifest_registry", "external_evidence_ingestion", "ingestion"]),
    (
        "freeze_and_planning",
        ["snapshotting", "timeguard", "features", "split_planning", "search_planning", "evaluation_governance"],
    ),
    (
        "candidate_realization",
        ["algorithmic_dsl", "shared_plus_local_decomposition", "candidate_fitting"],
    ),
    (
        "evidence_production",
        ["evaluation", "probabilistic_evaluation", "scoring", "robustness", "mechanistic_evidence"],
    ),
    ("meaning_replay_and_publication", ["gate_lifecycle", "claims", "replay", "catalog_publishing"]),
]
LIVE_SEQUENCE_DOC_REFS = {
    "docs/reference/system.md",
    "docs/reference/modeling-pipeline.md",
    "docs/reference/search-core.md",
    "docs/reference/contracts-manifests.md",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _parse_front_matter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    assert match, f"{path.relative_to(REPO_ROOT).as_posix()} must start with YAML front matter"
    return yaml.safe_load(match.group(1)), match.group(2)


def test_implementation_sequence_closes_module_order_and_evidence_paths() -> None:
    assert IMPLEMENTATION_SEQUENCE_PATH.is_file(), (
        "missing schemas/contracts/implementation-sequence.yaml"
    )

    module_registry = _load_yaml(MODULE_REGISTRY_PATH)
    payload = _load_yaml(IMPLEMENTATION_SEQUENCE_PATH)

    assert payload["version"] == 1
    assert payload["kind"] == "implementation_sequence"
    assert payload["sequence_id"] == "euclid_python_implementation_sequence@1.0.0"
    assert payload["source_module_registry"] == "schemas/contracts/module-registry.yaml"
    assert payload["ordering_rule"] == "dependency_topology_then_layer_order"

    registry_modules = {entry["module"]: entry for entry in module_registry["modules"]}
    sequence_entries = payload["module_sequence"]
    sequence_modules = [entry["module"] for entry in sequence_entries]

    assert set(sequence_modules) == set(registry_modules), "implementation sequence must cover every runtime module"
    assert len(sequence_modules) == len(set(sequence_modules)), "implementation sequence may not repeat modules"
    assert [entry["sequence_index"] for entry in sequence_entries] == list(range(1, len(sequence_entries) + 1))

    module_positions = {entry["module"]: entry["sequence_index"] for entry in sequence_entries}

    for entry in sequence_entries:
        module_id = entry["module"]
        registry_entry = registry_modules[module_id]
        assert entry["depends_on"] == registry_entry["allowed_dependencies"]
        assert set(entry["canonical_doc_refs"]) <= LIVE_SEQUENCE_DOC_REFS

        evidence = entry["evidence"]
        for evidence_key in ("math_refs", "contract_refs", "schema_names", "fixture_refs", "test_refs"):
            assert evidence_key in evidence, f"{module_id} must declare {evidence_key}"
            assert isinstance(evidence[evidence_key], list), f"{module_id} {evidence_key} must be a list"

        for relative_path in (
            entry["canonical_doc_refs"]
            + evidence["math_refs"]
            + evidence["contract_refs"]
            + evidence["fixture_refs"]
            + evidence["test_refs"]
        ):
            assert (REPO_ROOT / relative_path).exists(), f"{module_id} references missing path: {relative_path}"

        for schema_name in evidence["schema_names"]:
            assert "@" in schema_name, f"{module_id} schema_names must use canonical schema ids"

        assert evidence["contract_refs"], f"{module_id} must declare contract refs"
        assert evidence["fixture_refs"], f"{module_id} must declare fixture refs"
        assert evidence["test_refs"], f"{module_id} must declare test refs"

        for dependency in entry["depends_on"]:
            assert module_positions[dependency] < entry["sequence_index"], (
                f"{module_id} must come after dependency {dependency}"
            )


def test_implementation_sequence_layers_match_phase_order() -> None:
    payload = _load_yaml(IMPLEMENTATION_SEQUENCE_PATH)
    layers = payload["implementation_layers"]

    assert [layer["layer_id"] for layer in layers] == [layer_id for layer_id, _ in EXPECTED_LAYER_ORDER]
    assert len(layers) == len(EXPECTED_LAYER_ORDER)

    sequence_modules = {entry["module"]: entry for entry in payload["module_sequence"]}
    for layer, (expected_layer_id, expected_modules) in zip(layers, EXPECTED_LAYER_ORDER):
        assert layer["layer_id"] == expected_layer_id
        assert layer["modules"] == expected_modules
        for module_id in expected_modules:
            assert sequence_modules[module_id]["layer_id"] == expected_layer_id


def test_implementation_ingress_doc_is_structured_and_tracks_sequence_artifact() -> None:
    assert MODELING_PIPELINE_DOC_PATH.is_file(), "missing docs/reference/modeling-pipeline.md"

    front_matter, body = _parse_front_matter(MODELING_PIPELINE_DOC_PATH)
    assert front_matter["title"] == "Modeling Pipeline"
    assert front_matter["related"] == [
        "system.md",
        "search-core.md",
        "contracts-manifests.md",
        "workbench.md",
        "testing-truthfulness.md",
        "../../schemas/contracts/module-registry.yaml",
    ]

    normalized_body = " ".join(body.split()).lower()
    assert "src/euclid/modules" in body
    assert "legible mathematical structure" in normalized_body

    for source_anchor in (
        "`modules/ingestion.py`",
        "`modules/snapshotting.py`",
        "`modules/timeguard.py`",
        "`modules/features.py`",
        "`modules/split_planning.py`",
        "`modules/search_planning.py`",
        "`modules/candidate_fitting.py`",
        "`modules/shared_plus_local_decomposition.py`",
        "`modules/evaluation.py`",
        "`modules/probabilistic_evaluation.py`",
        "`modules/scoring.py`",
        "`modules/calibration.py`",
        "`modules/evaluation_governance.py`",
        "`modules/gate_lifecycle.py`",
        "`modules/claims.py`",
        "`modules/replay.py`",
        "`modules/catalog_publishing.py`",
    ):
        assert source_anchor in body, (
            f"docs/reference/modeling-pipeline.md must describe {source_anchor}"
        )


def test_source_map_and_live_entry_docs_point_to_implementation_ingress_surfaces() -> None:
    source_map = _load_yaml(SOURCE_MAP_PATH)
    entries = {entry["source"]: entry["canonical_targets"] for entry in source_map["entries"]}

    assert entries["README.md"] == [
        "docs/reference/README.md",
        "docs/reference/system.md",
        "docs/reference/runtime-cli.md",
        "docs/reference/modeling-pipeline.md",
        "docs/reference/search-core.md",
        "docs/reference/contracts-manifests.md",
        "docs/reference/benchmarks-readiness.md",
        "docs/reference/workbench.md",
        "docs/reference/testing-truthfulness.md",
    ]
    assert entries["src/euclid/modules"] == ["docs/reference/modeling-pipeline.md"]

    for source in (
        "src/euclid/search",
        "src/euclid/cir",
        "src/euclid/reducers",
        "src/euclid/adapters",
        "src/euclid/math",
    ):
        assert entries[source] == ["docs/reference/search-core.md"]

    for source in ("src/euclid/contracts", "src/euclid/manifests"):
        assert entries[source] == ["docs/reference/contracts-manifests.md"]

    assert entries["src/euclid/release.py"] == [
        "docs/reference/runtime-cli.md",
        "docs/reference/benchmarks-readiness.md",
        "docs/reference/contracts-manifests.md",
    ]
