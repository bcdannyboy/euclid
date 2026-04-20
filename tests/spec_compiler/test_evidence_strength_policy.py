from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_VISION_MATRIX_PATH = REPO_ROOT / "schemas/readiness/full-vision-matrix.yaml"
EVIDENCE_STRENGTH_POLICY_PATH = (
    REPO_ROOT / "schemas/readiness/evidence-strength-policy.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing required file: {path.relative_to(REPO_ROOT).as_posix()}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_non_closing_sources_cannot_close_runtime_rows() -> None:
    matrix_payload = _load_yaml(FULL_VISION_MATRIX_PATH)
    payload = _load_yaml(EVIDENCE_STRENGTH_POLICY_PATH)

    assert payload["version"] == 1
    assert payload["kind"] == "evidence_strength_policy"

    strength_classes = payload["strength_classes"]
    assert {entry["id"] for entry in strength_classes} == {
        "semantic_runtime",
        "benchmark_semantic",
        "replay",
        "packaging_install",
        "golden_regression",
        "governance_spec",
        "docs_only",
    }

    ranks = [entry["rank"] for entry in strength_classes]
    assert len(ranks) == len(set(ranks)), "strength class ranks must be unique"

    non_closing_sources = set(payload["non_closing_evidence_sources"])
    assert {
        "docs_only",
        "spec_only",
        "wording_only",
        "notebook_smoke",
    } <= non_closing_sources

    for row in matrix_payload["rows"]:
        minimum_classes = set(row["minimum_closing_evidence_classes"])
        assert not (minimum_classes & non_closing_sources), (
            f"{row['row_id']} uses a banned non-closing source as closing proof"
        )


def test_benchmark_semantic_requires_runtime_surface() -> None:
    matrix_payload = _load_yaml(FULL_VISION_MATRIX_PATH)
    payload = _load_yaml(EVIDENCE_STRENGTH_POLICY_PATH)
    benchmark_policy = payload["benchmark_semantic_policy"]

    assert benchmark_policy["semantic_requires_runtime_surface"] is True
    assert set(benchmark_policy["structural_only_capability_types"]) == {
        "evidence_lane"
    }

    for row in matrix_payload["rows"]:
        if "benchmark_semantic" not in set(row["minimum_closing_evidence_classes"]):
            continue
        if row["runtime_surface"]:
            continue
        assert row["capability_type"] in set(
            benchmark_policy["structural_only_capability_types"]
        )


def test_packaging_install_is_required_for_public_installed_surfaces() -> None:
    matrix_payload = _load_yaml(FULL_VISION_MATRIX_PATH)
    payload = _load_yaml(EVIDENCE_STRENGTH_POLICY_PATH)
    packaging_policy = payload["packaging_install_policy"]
    protected_types = set(packaging_policy["required_capability_types"])

    assert packaging_policy["required_for_public_installed_surfaces"] is True
    assert {"clean_install_surface", "operator_runtime_surface"} <= protected_types

    for row in matrix_payload["rows"]:
        if row["capability_type"] not in protected_types:
            continue
        assert "packaging_install" in set(row["minimum_closing_evidence_classes"]), (
            f"{row['row_id']} must require packaging-install proof"
        )
