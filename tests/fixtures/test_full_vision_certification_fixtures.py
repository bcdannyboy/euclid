from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SPEC_PATH = REPO_ROOT / "src/euclid/_assets/docs/implementation/certification-fixture-spec.yaml"
FIXTURE_ROOT = REPO_ROOT / "fixtures/runtime/full_vision_certification"
FULL_VISION_SUITE_PATH = REPO_ROOT / "benchmarks/suites/full-vision.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_certification_fixture_sets_exist_for_all_major_surfaces() -> None:
    fixture_spec = _load_yaml(FIXTURE_SPEC_PATH)

    for family in fixture_spec["families"]:
        family_dir = FIXTURE_ROOT / family["family_id"]
        manifest_path = family_dir / "fixture-set.yaml"
        assert family_dir.is_dir()
        assert manifest_path.is_file()

        payload = _load_yaml(manifest_path)
        assert payload["fixture_family_id"] == family["family_id"]
        assert payload["fixture_spec_id"] == fixture_spec["fixture_spec_id"]
        assert len(payload["dataset_refs"]) >= family["minimum_dataset_count"]
        assert payload["series_count"] >= family["minimum_series_count"]
        assert payload["entity_count"] >= family["minimum_entity_count"]
        assert (
            len(payload["evidence_bundle_refs"])
            >= family["minimum_evidence_bundle_count"]
        )
        assert set(family["required_diversity_axes"]) <= set(payload["diversity_axes"])


def test_certification_tasks_reference_certification_fixtures() -> None:
    suite = _load_yaml(FULL_VISION_SUITE_PATH)

    for relative_path in suite["task_manifest_paths"]:
        task_manifest = _load_yaml(REPO_ROOT / relative_path)
        assert task_manifest["fixture_spec_id"] == "euclid-certification-fixtures-v1"
        assert task_manifest["dataset_ref"].startswith(
            "fixtures/runtime/full_vision_certification/"
        )

