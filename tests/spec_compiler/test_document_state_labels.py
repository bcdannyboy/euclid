from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_REFERENCE_DOCS = (
    "docs/reference/README.md",
    "docs/reference/system.md",
    "docs/reference/runtime-cli.md",
    "docs/reference/modeling-pipeline.md",
    "docs/reference/search-core.md",
    "docs/reference/contracts-manifests.md",
    "docs/reference/benchmarks-readiness.md",
    "docs/reference/workbench.md",
    "docs/reference/testing-truthfulness.md",
)
ENTRYPOINT_DOCS = (
    "README.md",
    "docs/reference/README.md",
    "docs/reference/system.md",
)
BANNED_AMBIGUITY_LANGUAGE = (
    "if needed",
    "where possible",
    "as appropriate",
    "good enough",
    "substantially complete",
    "close enough",
)
LEGACY_DOC_PATHS = (
    "docs/overview.md",
    "docs/canonical/README.md",
    "docs/canonical/full-vision-boundaries.md",
    "docs/canonical/implementation-ingress.md",
    "docs/plans/2026-04-14-euclid-full-program-completion.md",
    "docs/plans/2026-04-14-euclid-100-percent-closure-plan.md",
)


def _text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _normalized(relative_path: str) -> str:
    return " ".join(_text(relative_path).replace(">", " ").split()).lower()


def _load_front_matter(relative_path: str) -> dict[str, Any]:
    text = _text(relative_path)
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"{relative_path} must start with YAML front matter"
    return yaml.safe_load(match.group(1))


def _load_yaml(relative_path: str) -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def test_active_docs_have_state_labels() -> None:
    readme_front_matter = _load_front_matter("README.md")
    assert readme_front_matter["related"] == [
        "docs/reference/README.md",
        "docs/reference/system.md",
        "schemas/core/euclid-system.yaml",
        "schemas/core/source-map.yaml",
    ]

    reference_front_matter = _load_front_matter("docs/reference/README.md")
    assert reference_front_matter["related"] == [
        "../../README.md",
        "system.md",
        "runtime-cli.md",
        "modeling-pipeline.md",
        "search-core.md",
        "contracts-manifests.md",
        "benchmarks-readiness.md",
        "workbench.md",
        "testing-truthfulness.md",
    ]

    source_map = _load_yaml("schemas/core/source-map.yaml")
    assert source_map["reference_workspace"]["docs_root"] == "docs/reference"
    readme_entry = next(
        entry for entry in source_map["entries"] if entry["source"] == "README.md"
    )
    assert readme_entry["canonical_targets"] == list(LIVE_REFERENCE_DOCS)


def test_scope_terms_are_used_consistently() -> None:
    for relative_path in ("README.md", "docs/reference/benchmarks-readiness.md"):
        text = _text(relative_path)
        for scope_term in ("`current_release`", "`full_vision`", "`shipped_releasable`"):
            assert scope_term in text, f"{relative_path} must use the canonical {scope_term} term"
        normalized = " ".join(text.replace(">", " ").split()).lower()
        assert " full-program " not in normalized, (
            f"{relative_path} must not use full-program as an active canonical scope term"
        )


def test_ambiguous_implemented_release_path_language_is_rejected() -> None:
    for relative_path in ("README.md", *LIVE_REFERENCE_DOCS):
        normalized = _normalized(relative_path)
        assert "implemented release path" not in normalized, (
            f"{relative_path} must name the live authority surface explicitly"
        )
        for legacy_path in LEGACY_DOC_PATHS:
            assert legacy_path.lower() not in normalized, (
                f"{relative_path} must not route readers to retired doc path {legacy_path}"
            )


def test_banned_ambiguity_language_is_absent_from_active_scope_docs() -> None:
    for relative_path in ENTRYPOINT_DOCS:
        normalized = _normalized(relative_path)
        for phrase in BANNED_AMBIGUITY_LANGUAGE:
            assert phrase not in normalized, (
                f"{relative_path} still contains banned ambiguity language: {phrase}"
            )
