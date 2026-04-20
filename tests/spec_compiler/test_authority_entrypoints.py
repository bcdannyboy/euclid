from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"

LIVE_ENTRY_EXPECTATIONS = {
    "README.md": [
        "docs/reference/system.md",
        "docs/reference/runtime-cli.md",
        "docs/reference/modeling-pipeline.md",
        "docs/reference/search-core.md",
        "docs/reference/contracts-manifests.md",
        "docs/reference/benchmarks-readiness.md",
        "docs/reference/workbench.md",
        "docs/reference/testing-truthfulness.md",
    ],
    "docs/reference/README.md": [
        "../../README.md",
        "system.md",
        "runtime-cli.md",
        "modeling-pipeline.md",
        "search-core.md",
        "contracts-manifests.md",
        "benchmarks-readiness.md",
        "workbench.md",
        "testing-truthfulness.md",
    ],
    "docs/reference/system.md": [
        "src/euclid/cli",
        "src/euclid/modules",
        "src/euclid/search",
        "src/euclid/contracts",
        "src/euclid/workbench",
    ],
}


def _load_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _load_source_map() -> dict:
    return yaml.safe_load(SOURCE_MAP_PATH.read_text(encoding="utf-8"))


def test_live_entry_docs_point_to_reference_workspace() -> None:
    for relative_path, expected_fragments in LIVE_ENTRY_EXPECTATIONS.items():
        text = _load_text(relative_path)
        for fragment in expected_fragments:
            assert fragment in text, f"{relative_path} must point readers at {fragment}"


def test_source_map_tracks_current_reference_entrypoints() -> None:
    entries = {entry["source"]: entry for entry in _load_source_map()["entries"]}
    assert entries["README.md"]["status"] == "entrypoint"
    assert "docs/reference/system.md" in entries["README.md"]["canonical_targets"]
    assert entries["src/euclid/cli/__init__.py"]["status"] == "certified_surface"
    assert entries["src/euclid/workbench"]["status"] == "ui_surface"
