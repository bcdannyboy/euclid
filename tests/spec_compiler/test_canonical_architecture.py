from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_DOCS = {
    "docs/reference/system.md": {
        "required_terms": {
            "runtime and control plane",
            "modeling pipeline",
            "search and model semantics",
            "formal specification and release evidence",
            "local analysis workbench",
        },
    },
    "docs/reference/runtime-cli.md": {
        "required_terms": {
            "euclid run --config examples/current_release_run.yaml",
            "operatorrequest",
            "active-runs/<run_id>/control-plane/execution-state.sqlite3",
            "replay is part of the product, not an afterthought",
        },
    },
    "docs/reference/modeling-pipeline.md": {
        "required_terms": {
            "snapshotting and timeguard",
            "feature materialization",
            "evaluation geometry",
            "claims, replay, and publication",
        },
    },
    "docs/reference/search-core.md": {
        "required_terms": {
            "candidate intermediate representation",
            "description gain",
            "algorithmic dsl",
            "reducer families and compositions",
        },
    },
}


def test_reference_architecture_docs_cover_runtime_shape() -> None:
    for relative_path, expectations in ARCHITECTURE_DOCS.items():
        body = (REPO_ROOT / relative_path).read_text(encoding="utf-8").lower()
        for term in expectations["required_terms"]:
            assert term.lower() in body, f"{relative_path} must mention {term}"
