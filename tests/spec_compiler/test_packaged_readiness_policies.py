from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_VISION_MATRIX_PATH = REPO_ROOT / "schemas/readiness/full-vision-matrix.yaml"
CURRENT_RELEASE_POLICY_PATH = REPO_ROOT / "schemas/readiness/current-release-v1.yaml"
FULL_VISION_POLICY_PATH = REPO_ROOT / "schemas/readiness/full-vision-v1.yaml"
SHIPPED_RELEASABLE_POLICY_PATH = (
    REPO_ROOT / "schemas/readiness/shipped-releasable-v1.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), (
        "missing required file: " f"{path.relative_to(REPO_ROOT).as_posix()}"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_current_release_scope_is_proper_subset_of_full_vision() -> None:
    current_release = _load_yaml(CURRENT_RELEASE_POLICY_PATH)
    full_vision = _load_yaml(FULL_VISION_POLICY_PATH)

    current_rows = set(current_release["required_row_ids"])
    full_rows = set(full_vision["required_row_ids"])

    assert current_rows < full_rows
    assert "operator_runtime_surface:operator_run" in current_rows
    assert "clean_install_surface:release_status" in current_rows
    assert "evidence_lane:robustness" in current_rows
    assert "benchmark_surface:portfolio_orchestration" not in current_rows
    assert "forecast_object_type:distribution" not in current_rows
    assert "run_support_object:target_transform_object" not in current_rows
    assert "admissibility_rule:composition_closure" not in current_rows


def test_full_vision_policy_matches_matrix_rows() -> None:
    matrix_payload = _load_yaml(FULL_VISION_MATRIX_PATH)
    full_vision = _load_yaml(FULL_VISION_POLICY_PATH)

    assert set(full_vision["required_row_ids"]) == {
        row["row_id"] for row in matrix_payload["rows"]
    }
    assert "lifecycle_artifact:external_evidence_bundle" in set(
        full_vision["required_row_ids"]
    )
    assert {
        "benchmark_surface:external_evidence_ingestion",
        "benchmark_surface:robustness_lane",
    } <= set(full_vision["required_row_ids"])
    assert "clean_install_surface:benchmark_execution" in set(
        full_vision["required_row_ids"]
    )


def test_packaged_policies_do_not_default_to_same_row_set() -> None:
    current_release = _load_yaml(CURRENT_RELEASE_POLICY_PATH)
    full_vision = _load_yaml(FULL_VISION_POLICY_PATH)
    shipped_releasable = _load_yaml(SHIPPED_RELEASABLE_POLICY_PATH)

    current_rows = set(current_release["required_row_ids"])
    full_rows = set(full_vision["required_row_ids"])
    shipped_rows = set(shipped_releasable["required_row_ids"])

    assert current_release["policy_id"] == "current_release_v1"
    assert full_vision["policy_id"] == "full_vision_v1"
    assert shipped_releasable["policy_id"] == "shipped_releasable_v1"
    assert current_release["matrix_path"] == "schemas/readiness/full-vision-matrix.yaml"
    assert full_vision["matrix_path"] == "schemas/readiness/full-vision-matrix.yaml"
    assert (
        shipped_releasable["matrix_path"] == "schemas/readiness/full-vision-matrix.yaml"
    )
    assert current_rows != full_rows
    assert shipped_rows == current_rows
    assert shipped_releasable["scope_authority_refs"] != current_release[
        "scope_authority_refs"
    ]
