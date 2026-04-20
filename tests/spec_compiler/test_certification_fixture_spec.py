from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SPEC_PATH = REPO_ROOT / "docs/implementation/certification-fixture-spec.yaml"
FULL_VISION_SUITE_PATH = REPO_ROOT / "benchmarks/suites/full-vision.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), (
        "missing required file: " f"{path.relative_to(REPO_ROOT).as_posix()}"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _full_vision_task_manifests() -> list[dict[str, Any]]:
    suite = _load_yaml(FULL_VISION_SUITE_PATH)
    return [
        _load_yaml(REPO_ROOT / relative_path)
        for relative_path in suite["task_manifest_paths"]
    ]


def test_fixture_spec_defines_minimum_thresholds_for_all_certification_families(
) -> None:
    payload = _load_yaml(FIXTURE_SPEC_PATH)
    families = payload["families"]

    assert {family["family_id"] for family in families} == {
        "single_entity_predictive",
        "panel_shared_local",
        "algorithmic_rediscovery",
        "mechanistic",
        "external_evidence",
        "robustness",
        "portfolio_comparison",
    }
    for family in families:
        assert family["minimum_dataset_count"] >= 1
        assert family["minimum_series_count"] >= 1
        assert family["minimum_entity_count"] >= 1
        assert family["minimum_evidence_bundle_count"] >= 1
        assert family["required_diversity_axes"]
        assert family["benchmark_task_ids"]
        assert family["suite_ids"]


def test_every_task_using_certification_fixtures_points_to_fixture_spec_rows() -> None:
    payload = _load_yaml(FIXTURE_SPEC_PATH)
    suite = _load_yaml(FULL_VISION_SUITE_PATH)
    known_family_ids = {family["family_id"] for family in payload["families"]}

    assert suite["fixture_spec_id"] == payload["fixture_spec_id"]
    for task_manifest in _full_vision_task_manifests():
        assert task_manifest["fixture_spec_id"] == payload["fixture_spec_id"]
        assert task_manifest["fixture_family_id"] in known_family_ids


def test_fixture_spec_benchmark_tasks_are_live_suite_tasks() -> None:
    payload = _load_yaml(FIXTURE_SPEC_PATH)
    live_task_ids = {
        task_manifest["task_id"] for task_manifest in _full_vision_task_manifests()
    }

    for family in payload["families"]:
        assert set(family["benchmark_task_ids"]) <= live_task_ids


def test_medium_size_language_is_banned_without_fixture_spec_reference() -> None:
    suite = _load_yaml(FULL_VISION_SUITE_PATH)

    assert "medium-size" not in suite["description"]
    for task_manifest in _full_vision_task_manifests():
        assert "medium_fixture" not in set(task_manifest["regime_tags"])
        assert "medium_fixture" not in set(task_manifest["adversarial_tags"])


def test_unbound_certification_scale_tags_are_rejected() -> None:
    payload = _load_yaml(FIXTURE_SPEC_PATH)
    banned_tokens = set(payload["banned_unbound_scale_tokens"])

    for task_manifest in _full_vision_task_manifests():
        assert not (set(task_manifest["regime_tags"]) & banned_tokens)
        assert not (set(task_manifest["adversarial_tags"]) & banned_tokens)
