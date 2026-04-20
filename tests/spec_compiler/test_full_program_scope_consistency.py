from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DOC_EXPECTATIONS = {
    "README.md": {
        "required": (
            "`current_release`",
            "`full_vision`",
            "`shipped_releasable`",
            "docs/reference/system.md",
            "docs/reference/runtime-cli.md",
            "docs/reference/modeling-pipeline.md",
            "docs/reference/search-core.md",
            "docs/reference/contracts-manifests.md",
            "docs/reference/benchmarks-readiness.md",
            "docs/reference/workbench.md",
            "docs/reference/testing-truthfulness.md",
        ),
        "disallowed": (
            "docs/overview.md",
            "docs/canonical/full-vision-boundaries.md",
            "docs/canonical/implementation-ingress.md",
        ),
    },
    "docs/reference/README.md": {
        "required": (
            "`system.md`",
            "`runtime-cli.md`",
            "`modeling-pipeline.md`",
            "`search-core.md`",
            "`contracts-manifests.md`",
            "`benchmarks-readiness.md`",
            "`workbench.md`",
            "`testing-truthfulness.md`",
        ),
        "disallowed": (
            "docs/overview.md",
            "docs/canonical/README.md",
        ),
    },
    "docs/reference/system.md": {
        "required": (
            "Certified runtime and control plane",
            "Modeling pipeline",
            "Search and model semantics",
            "Formal specification and release evidence",
            "Local analysis workbench",
        ),
        "disallowed": (
            "docs/overview.md",
            "docs/canonical/system-map.md",
        ),
    },
    "docs/reference/runtime-cli.md": {
        "required": (
            "The main way to run Euclid is through `euclid` and `python -m euclid`.",
            "## Compatibility-only commands",
            "`euclid release certify-clean-install`",
            "`euclid workbench serve`",
        ),
        "disallowed": (
            "full-program release workflow",
            "certified operator target",
        ),
    },
    "docs/reference/modeling-pipeline.md": {
        "required": (
            "src/euclid/modules",
            "`modules/search_planning.py`",
            "`modules/replay.py`",
            "`point`",
            "`distribution`",
            "`interval`",
            "`quantile`",
            "`event_probability`",
        ),
        "disallowed": (
            "docs/canonical/implementation-ingress.md",
            "docs/module-specs/",
        ),
    },
    "docs/reference/search-core.md": {
        "required": (
            "`exact_finite_enumeration`",
            "`bounded_heuristic`",
            "`equality_saturation_heuristic`",
            "`stochastic_heuristic`",
            "`shared_plus_local_decomposition`",
            "`src/euclid/modules/algorithmic_dsl.py`",
        ),
        "disallowed": (
            "docs/module-specs/algorithmic-dsl.md",
            "docs/module-specs/hierarchical-modeling.md",
        ),
    },
    "docs/reference/contracts-manifests.md": {
        "required": (
            "`schemas/core`",
            "`schemas/contracts`",
            "`schemas/readiness`",
            "`docs/implementation/*.yaml`",
        ),
        "disallowed": (
            "docs/canonical/README.md",
            "docs/implementation/documentation-cleanup-register.md",
        ),
    },
    "docs/reference/benchmarks-readiness.md": {
        "required": (
            "`current_release`",
            "`full_vision`",
            "`shipped_releasable`",
            "`docs/implementation/*.yaml`",
            "`tools/spec_compiler/compiler.py`",
        ),
        "disallowed": (
            "docs/runbook/benchmarking.md",
            "full-program python surface",
        ),
    },
    "docs/reference/workbench.md": {
        "required": (
            "GET /api/config",
            "POST /api/analyze",
            "`descriptive_fit`",
            "`predictive_law`",
            "busy, failure, no-winner, and explainer-fallback states",
        ),
        "disallowed": (
            "docs/canonical/full-vision-boundaries.md",
            "docs/module-specs/",
        ),
    },
    "docs/reference/testing-truthfulness.md": {
        "required": (
            "`tests/spec_compiler/*`",
            "`tests/unit/modules/*`",
            "`tests/integration/test_operator_run_pipeline.py`",
        ),
        "disallowed": (
            "docs/overview.md",
            "docs/canonical/README.md",
        ),
    },
}


def _normalize(text: str) -> str:
    return " ".join(text.replace(">", " ").split()).lower()


def test_scope_docs_remove_deferred_only_language() -> None:
    for relative_path, expectations in DOC_EXPECTATIONS.items():
        text = _normalize((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        for fragment in expectations["required"]:
            assert _normalize(fragment) in text, (
                f"{relative_path} must mention {fragment}"
            )
        for fragment in expectations["disallowed"]:
            assert _normalize(fragment) not in text, (
                f"{relative_path} still contains stale documentation topology: {fragment}"
            )
