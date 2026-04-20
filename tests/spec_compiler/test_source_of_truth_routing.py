from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

START_HERE_DOCS = {
    "README.md": {
        "fragments": (
            "docs/reference/system.md",
            "docs/reference/runtime-cli.md",
            "docs/reference/modeling-pipeline.md",
            "docs/reference/search-core.md",
            "docs/reference/contracts-manifests.md",
            "docs/reference/benchmarks-readiness.md",
            "docs/reference/workbench.md",
            "docs/reference/testing-truthfulness.md",
            "schemas/core/euclid-system.yaml",
            "schemas/core/source-map.yaml",
        ),
    },
    "docs/reference/README.md": {
        "fragments": (
            "../../README.md",
            "system.md",
            "runtime-cli.md",
            "modeling-pipeline.md",
            "search-core.md",
            "contracts-manifests.md",
            "benchmarks-readiness.md",
            "workbench.md",
            "testing-truthfulness.md",
        ),
    },
    "docs/reference/system.md": {
        "fragments": (
            "src/euclid/cli",
            "src/euclid/modules",
            "src/euclid/search",
            "src/euclid/contracts",
            "src/euclid/workbench",
        ),
    },
}


def _load_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _normalized(text: str) -> str:
    return " ".join(text.replace(">", " ").split())


def test_start_here_docs_route_to_live_reference_set() -> None:
    for relative_path, expectations in START_HERE_DOCS.items():
        text = _normalized(_load_text(relative_path))
        for fragment in expectations["fragments"]:
            assert " ".join(fragment.split()) in text, (
                f"{relative_path} must route readers to {fragment}"
            )
