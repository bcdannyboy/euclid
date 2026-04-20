from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_COVERAGE_PATH = REPO_ROOT / "fixtures/canonical/fixture-coverage.yaml"
WALKTHROUGH_DIR = REPO_ROOT / "docs/reference/examples"
FRONT_MATTER_PATTERN = re.compile(r"\A---\n(?P<front_matter>.*?)\n---\n(?P<body>.*)\Z", re.DOTALL)


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing required file: {path.relative_to(REPO_ROOT).as_posix()}"
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict), f"{path.relative_to(REPO_ROOT).as_posix()} must deserialize to an object"
    return payload


def _walkthrough_paths() -> list[Path]:
    return sorted(path for path in WALKTHROUGH_DIR.glob("*.md"))


def _parse_walkthrough(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text()
    match = FRONT_MATTER_PATTERN.match(text)
    assert match is not None, f"{path.relative_to(REPO_ROOT).as_posix()} must start with YAML front matter"
    payload = yaml.safe_load(match.group("front_matter"))
    assert isinstance(payload, dict), f"{path.relative_to(REPO_ROOT).as_posix()} front matter must be a mapping"
    return payload, match.group("body")


def test_fixture_walkthroughs_exist_with_structured_front_matter() -> None:
    walkthrough_paths = _walkthrough_paths()

    assert walkthrough_paths, "expected reference fixture walkthrough markdown files"

    for path in walkthrough_paths:
        payload, body = _parse_walkthrough(path)

        assert isinstance(payload["title"], str) and payload["title"]
        assert isinstance(payload.get("scenario_ids"), list), (
            f"{path.relative_to(REPO_ROOT).as_posix()} scenario_ids must be a list"
        )
        assert isinstance(payload.get("fixture_bundles"), list), (
            f"{path.relative_to(REPO_ROOT).as_posix()} fixture_bundles must be a list"
        )
        assert body.strip(), f"{path.relative_to(REPO_ROOT).as_posix()} should include markdown body content"


def test_fixture_walkthroughs_cover_every_scenario_in_the_coverage_plan() -> None:
    coverage_plan = _load_yaml(FIXTURE_COVERAGE_PATH)
    scenario_map = {
        scenario["scenario_id"]: scenario["planned_fixture_bundle"]
        for scenario in coverage_plan["scenarios"]
    }

    covered_scenarios: dict[str, Path] = {}

    for path in _walkthrough_paths():
        payload, body = _parse_walkthrough(path)
        scenario_ids = payload.get("scenario_ids", [])
        fixture_bundles = payload.get("fixture_bundles", [])

        if not scenario_ids and not fixture_bundles:
            continue

        assert isinstance(scenario_ids, list), f"{path.relative_to(REPO_ROOT).as_posix()} scenario_ids must be a list"
        assert isinstance(fixture_bundles, list), (
            f"{path.relative_to(REPO_ROOT).as_posix()} fixture_bundles must be a list"
        )

        expected_bundles = {scenario_map[scenario_id] for scenario_id in scenario_ids}
        assert set(fixture_bundles) == expected_bundles, (
            f"{path.relative_to(REPO_ROOT).as_posix()} must declare the planned bundle paths for its scenarios"
        )

        for scenario_id in scenario_ids:
            assert scenario_id not in covered_scenarios, f"duplicate walkthrough coverage for {scenario_id}"
            covered_scenarios[scenario_id] = path

        for bundle_path in fixture_bundles:
            bundle_file = REPO_ROOT / bundle_path
            assert bundle_file.is_file(), f"missing referenced fixture bundle: {bundle_path}"
            relative_link = os.path.relpath(bundle_file, path.parent).replace(os.sep, "/")
            assert relative_link in body, (
                f"{path.relative_to(REPO_ROOT).as_posix()} must link directly to {bundle_path}"
            )

    assert set(covered_scenarios) == set(scenario_map), "walkthrough scenarios must match the fixture coverage plan"
