from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_SCHEMA_PATH = REPO_ROOT / "schemas/core/euclid-system.yaml"
REFERENCE_DOCS = {
    "README.md": {
        "required_terms": {"descriptive equation", "unified equation", "current_release", "full_vision"},
    },
    "docs/reference/system.md": {
        "required_terms": {"subsystem map", "runtime", "workbench"},
    },
    "docs/reference/search-core.md": {
        "required_terms": {
            "exact_finite_enumeration",
            "bounded_heuristic",
            "equality_saturation_heuristic",
            "stochastic_heuristic",
            "candidate intermediate representation",
        },
    },
    "docs/reference/contracts-manifests.md": {
        "required_terms": {"typed refs", "ManifestEnvelope", "runtime_models.py"},
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _flatten_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        flattened: set[str] = set()
        for item in value:
            flattened |= _flatten_strings(item)
        return flattened
    if isinstance(value, dict):
        flattened: set[str] = set()
        for item in value.values():
            flattened |= _flatten_strings(item)
        return flattened
    return set()


def test_reference_docs_exist_and_cover_core_runtime_concepts() -> None:
    for relative_path, expectations in REFERENCE_DOCS.items():
        path = REPO_ROOT / relative_path
        assert path.is_file(), f"missing reference doc: {relative_path}"
        body = path.read_text(encoding="utf-8").lower()
        for term in expectations["required_terms"]:
            assert term.lower() in body, f"{relative_path} must mention {term}"


def test_system_schema_declares_reference_docs_and_vocabularies() -> None:
    schema = _load_yaml(SYSTEM_SCHEMA_PATH)

    assert schema["project_name"] == "Euclid"
    assert schema["scope_statement"]
    assert "README.md" in set(schema["canonical_doc_refs"])
    assert "docs/reference/system.md" in set(schema["canonical_doc_refs"])
    assert "docs/reference/workbench.md" in set(schema["canonical_doc_refs"])

    runtime_modules = _flatten_strings(schema["major_runtime_modules"])
    assert {
        "cli",
        "operator_runtime",
        "search",
        "contracts",
        "benchmarks",
        "workbench",
    } <= runtime_modules

    artifact_classes = _flatten_strings(schema["artifact_classes"])
    assert {
        "dataset_snapshot",
        "feature_view",
        "prediction_artifact",
        "claim_card",
        "readiness_judgment",
    } <= artifact_classes
