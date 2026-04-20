from __future__ import annotations

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
ACTIVE_ROUTING_DOCS = (
    "README.md",
    "docs/reference/README.md",
    "docs/reference/system.md",
    "docs/reference/contracts-manifests.md",
    "docs/reference/benchmarks-readiness.md",
    "schemas/core/source-map.yaml",
    "schemas/core/euclid-system.yaml",
)
SUPERSEDED_PLANS = (
    "docs/plans/2026-04-14-euclid-full-program-completion.md",
    "docs/plans/2026-04-14-euclid-100-percent-closure-plan.md",
    "docs/plans/2026-04-15-euclid-honest-100-task-ledger.md",
)
LEGACY_DOC_PATHS = (
    "docs/overview.md",
    "docs/canonical/README.md",
    "docs/canonical/full-vision-boundaries.md",
    "docs/canonical/implementation-ingress.md",
)


def _load_yaml(relative_path: str) -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _normalized(relative_path: str) -> str:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    return " ".join(text.replace(">", " ").split()).lower()


def test_all_active_docs_point_to_definitive_master_plan() -> None:
    source_map = _load_yaml("schemas/core/source-map.yaml")
    readme_entry = next(
        entry for entry in source_map["entries"] if entry["source"] == "README.md"
    )
    assert readme_entry["canonical_targets"] == list(LIVE_REFERENCE_DOCS)

    system = _load_yaml("schemas/core/euclid-system.yaml")
    assert system["canonical_doc_refs"] == ["README.md", *LIVE_REFERENCE_DOCS]

    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for relative_path in LIVE_REFERENCE_DOCS:
        assert relative_path in readme_text, (
            f"README.md must route readers to live reference doc {relative_path}"
        )


def test_superseded_plans_are_labeled_historical() -> None:
    for relative_path in ACTIVE_ROUTING_DOCS:
        normalized = _normalized(relative_path)
        for superseded_plan in SUPERSEDED_PLANS:
            assert superseded_plan.lower() not in normalized, (
                f"{relative_path} must not route active readers to old plan {superseded_plan}"
            )


def test_no_live_doc_describes_superseded_plan_as_active() -> None:
    stale_claims = (
        "closed the full-program release",
        "records the completion sequence that closed the full-program release",
        "completed full-program scope",
    )

    for relative_path in ACTIVE_ROUTING_DOCS:
        normalized = _normalized(relative_path)
        for legacy_path in LEGACY_DOC_PATHS:
            assert legacy_path.lower() not in normalized, (
                f"{relative_path} must not route readers to retired doc path {legacy_path}"
            )
        for stale_claim in stale_claims:
            assert stale_claim not in normalized, (
                f"{relative_path} must not describe retired plan topology as active"
            )
