from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_VISION_MATRIX_PATH = REPO_ROOT / "schemas/readiness/full-vision-matrix.yaml"
REDUCER_FAMILIES_PATH = REPO_ROOT / "schemas/core/reducer-families.yaml"
COMPOSITION_OPERATORS_PATH = REPO_ROOT / "schemas/core/composition-operators.yaml"
FORECAST_OBJECT_TYPES_PATH = REPO_ROOT / "schemas/core/forecast-object-types.yaml"
EVIDENCE_CLASSES_PATH = REPO_ROOT / "schemas/core/evidence-classes.yaml"
SEARCH_CLASSES_PATH = REPO_ROOT / "schemas/contracts/search-classes.yaml"
RUN_LIFECYCLE_PATH = REPO_ROOT / "schemas/contracts/run-lifecycle.yaml"
CURRENT_RELEASE_SUITE_PATH = REPO_ROOT / "benchmarks/suites/current-release.yaml"
FULL_VISION_SUITE_PATH = REPO_ROOT / "benchmarks/suites/full-vision.yaml"
EUCLID_SYSTEM_PATH = REPO_ROOT / "schemas/core/euclid-system.yaml"

RUN_SUPPORT_OBJECT_IDS = {
    "target_transform_object",
    "quantization_object",
    "observation_model_object",
    "reference_description_object",
    "codelength_policy_object",
}
ADMISSIBILITY_RULE_IDS = {
    "family_membership",
    "composition_closure",
    "observation_model_compatibility",
    "valid_state_semantics",
    "codelength_comparability",
}
OPERATOR_RUNTIME_SURFACE_IDS = {
    "operator_run",
    "operator_replay",
}
CLEAN_INSTALL_SURFACE_IDS = {
    "release_status",
    "operator_run",
    "operator_replay",
    "benchmark_execution",
    "determinism_same_seed",
    "performance_runtime_smoke",
    "packaged_notebook_smoke",
}
REPLAY_SURFACE_IDS = {"operator_native_replay"}


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), (
        "missing required file: " f"{path.relative_to(REPO_ROOT).as_posix()}"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _entry_ids(path: Path) -> set[str]:
    return {entry["id"] for entry in _load_yaml(path)["entries"]}


def _search_class_ids() -> set[str]:
    payload = _load_yaml(SEARCH_CLASSES_PATH)
    return {entry["search_class"] for entry in payload["contracts"]}


def _lifecycle_artifact_ids() -> set[str]:
    payload = _load_yaml(RUN_LIFECYCLE_PATH)
    artifact_ids: set[str] = set()
    for state in payload["states"]:
        artifact_ids.update(state.get("required_artifacts", ()))
    return artifact_ids


def _benchmark_track_ids() -> set[str]:
    payload = _load_yaml(FULL_VISION_SUITE_PATH)
    return set(payload["required_tracks"])


def _benchmark_surface_ids() -> set[str]:
    payload = _load_yaml(FULL_VISION_SUITE_PATH)
    return {
        requirement["surface_id"]
        for requirement in payload["surface_requirements"]
    }


def _rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["row_id"]: row for row in rows}


def test_matrix_contains_all_euclid_capabilities() -> None:
    payload = _load_yaml(FULL_VISION_MATRIX_PATH)

    assert payload["version"] == 1
    assert payload["kind"] == "full_vision_matrix"
    assert set(payload["scope_authority_refs"]) == {"docs/reference/system.md"}

    rows = payload["rows"]
    assert isinstance(rows, list) and rows, "full vision matrix must declare rows"

    seen_row_ids: set[str] = set()
    by_type: dict[str, set[str]] = {}
    valid_types = {
        "reducer_family",
        "composition_operator",
        "forecast_object_type",
        "run_support_object",
        "admissibility_rule",
        "search_class",
        "evidence_lane",
        "lifecycle_artifact",
        "benchmark_track",
        "benchmark_surface",
        "operator_runtime_surface",
        "clean_install_surface",
        "replay_surface",
    }

    for row in rows:
        row_id = row["row_id"]
        capability_type = row["capability_type"]
        capability_id = row["capability_id"]

        assert row_id not in seen_row_ids, f"duplicate row_id: {row_id}"
        seen_row_ids.add(row_id)

        assert capability_type in valid_types
        assert row_id == f"{capability_type}:{capability_id}"
        assert isinstance(row["runtime_surface"], bool)
        assert row["governing_doc_refs"], f"{row_id} must trace back to authority docs"
        assert row["minimum_closing_evidence_classes"], (
            f"{row_id} must declare minimum closing evidence classes"
        )

        by_type.setdefault(capability_type, set()).add(capability_id)

    assert by_type["reducer_family"] == _entry_ids(REDUCER_FAMILIES_PATH)
    assert by_type["composition_operator"] == _entry_ids(COMPOSITION_OPERATORS_PATH)
    assert by_type["forecast_object_type"] == _entry_ids(FORECAST_OBJECT_TYPES_PATH)
    assert by_type["run_support_object"] == RUN_SUPPORT_OBJECT_IDS
    assert by_type["admissibility_rule"] == ADMISSIBILITY_RULE_IDS
    assert by_type["search_class"] == _search_class_ids()
    assert by_type["evidence_lane"] == _entry_ids(EVIDENCE_CLASSES_PATH)
    assert by_type["lifecycle_artifact"] == _lifecycle_artifact_ids()
    assert by_type["benchmark_track"] == _benchmark_track_ids()
    assert by_type["benchmark_surface"] == _benchmark_surface_ids()
    assert by_type["operator_runtime_surface"] == OPERATOR_RUNTIME_SURFACE_IDS
    assert by_type["clean_install_surface"] == CLEAN_INSTALL_SURFACE_IDS
    assert by_type["replay_surface"] == REPLAY_SURFACE_IDS

    vocabulary = payload["benchmark_governed_outcome_vocabulary"]
    assert set(vocabulary["abstention_modes"]) >= {
        "structural_miss",
        "calibrated_or_abstain",
        "honest_abstention",
    }
    assert set(vocabulary["insufficiency_reason_codes"]) >= {
        "external_evidence_insufficient",
        "mechanistic_evidence_insufficient",
        "robustness_floor_failed",
    }


def test_matrix_contains_run_support_object_rows() -> None:
    payload = _load_yaml(FULL_VISION_MATRIX_PATH)
    rows = _rows_by_id(payload["rows"])

    for object_id in RUN_SUPPORT_OBJECT_IDS:
        row = rows[f"run_support_object:{object_id}"]
        assert row["runtime_surface"] is True
        assert "docs/reference/search-core.md" in row["governing_doc_refs"]
        assert set(row["minimum_closing_evidence_classes"]) >= {
            "semantic_runtime",
            "replay",
        }


def test_matrix_contains_admissibility_rule_rows() -> None:
    payload = _load_yaml(FULL_VISION_MATRIX_PATH)
    rows = _rows_by_id(payload["rows"])

    for rule_id in ADMISSIBILITY_RULE_IDS:
        row = rows[f"admissibility_rule:{rule_id}"]
        assert row["runtime_surface"] is True
        assert "docs/reference/system.md" in row["governing_doc_refs"]
        assert "docs/reference/search-core.md" in row["governing_doc_refs"]
        assert set(row["minimum_closing_evidence_classes"]) >= {
            "semantic_runtime",
            "replay",
        }


def test_every_runtime_row_requires_runtime_closing_evidence() -> None:
    payload = _load_yaml(FULL_VISION_MATRIX_PATH)
    runtime_evidence_classes = {
        "semantic_runtime",
        "benchmark_semantic",
        "replay",
        "packaging_install",
    }

    for row in payload["rows"]:
        minimum_classes = set(row["minimum_closing_evidence_classes"])
        assert minimum_classes, f"{row['row_id']} must declare closing evidence"
        if row["runtime_surface"]:
            assert minimum_classes & runtime_evidence_classes, (
                f"{row['row_id']} must require runtime-facing proof"
            )


def test_system_map_and_matrix_use_same_capability_vocabulary() -> None:
    payload = _load_yaml(FULL_VISION_MATRIX_PATH)
    rows = _rows_by_id(payload["rows"])
    euclid_system = _load_yaml(EUCLID_SYSTEM_PATH)
    canonical_doc_refs = set(euclid_system["canonical_doc_refs"])

    for row in payload["rows"]:
        assert set(row["governing_doc_refs"]) <= canonical_doc_refs
        assert "docs/reference/system.md" in row["governing_doc_refs"]

    for object_id in RUN_SUPPORT_OBJECT_IDS:
        assert "docs/reference/search-core.md" in rows[
            f"run_support_object:{object_id}"
        ]["governing_doc_refs"]

    for rule_id in ADMISSIBILITY_RULE_IDS:
        assert "docs/reference/search-core.md" in rows[
            f"admissibility_rule:{rule_id}"
        ]["governing_doc_refs"]

    for object_id in _entry_ids(FORECAST_OBJECT_TYPES_PATH):
        assert "docs/reference/modeling-pipeline.md" in rows[
            f"forecast_object_type:{object_id}"
        ]["governing_doc_refs"]

    for track_id in _benchmark_track_ids():
        assert "docs/reference/benchmarks-readiness.md" in rows[
            f"benchmark_track:{track_id}"
        ]["governing_doc_refs"]

    for surface_id in _benchmark_surface_ids():
        assert "docs/reference/benchmarks-readiness.md" in rows[
            f"benchmark_surface:{surface_id}"
        ]["governing_doc_refs"]

    assert "docs/reference/modeling-pipeline.md" in rows[
        "replay_surface:operator_native_replay"
    ]["governing_doc_refs"]


def test_matrix_scope_is_not_derived_from_current_release_suite() -> None:
    matrix_payload = _load_yaml(FULL_VISION_MATRIX_PATH)
    current_release = _load_yaml(CURRENT_RELEASE_SUITE_PATH)
    full_vision = _load_yaml(FULL_VISION_SUITE_PATH)

    matrix_benchmark_surfaces = {
        row["capability_id"]
        for row in matrix_payload["rows"]
        if row["capability_type"] == "benchmark_surface"
    }
    current_release_surfaces = {
        requirement["surface_id"]
        for requirement in current_release["surface_requirements"]
    }
    full_vision_surfaces = {
        requirement["surface_id"]
        for requirement in full_vision["surface_requirements"]
    }

    assert matrix_benchmark_surfaces == full_vision_surfaces
    assert current_release_surfaces < matrix_benchmark_surfaces
    assert {
        "probabilistic_forecast_surface",
        "search_class_honesty",
        "composition_operator_semantics",
    } <= matrix_benchmark_surfaces - current_release_surfaces
