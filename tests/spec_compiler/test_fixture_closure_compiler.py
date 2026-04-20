from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_COVERAGE_PATH = REPO_ROOT / "fixtures/canonical/fixture-coverage.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _copy_source_root(tmp_path: Path) -> Path:
    destination = tmp_path / "fixture-root"
    destination.mkdir()
    for relative in ("docs", "schemas", "fixtures"):
        shutil.copytree(REPO_ROOT / relative, destination / relative)
    shutil.copy2(REPO_ROOT / "README.md", destination / "README.md")
    return destination


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict)
    return payload


def _coverage_scenarios(*scenario_classes: str) -> list[dict[str, Any]]:
    coverage_plan = _load_yaml(FIXTURE_COVERAGE_PATH)
    if not scenario_classes:
        return coverage_plan["scenarios"]

    allowed_classes = set(scenario_classes)
    return [
        scenario
        for scenario in coverage_plan["scenarios"]
        if scenario["scenario_class"] in allowed_classes
    ]


def _coverage_scenario(scenario_id: str) -> dict[str, Any]:
    return next(scenario for scenario in _coverage_scenarios() if scenario["scenario_id"] == scenario_id)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _artifact(bundle: dict[str, Any], schema_name: str) -> dict[str, Any]:
    return next(entry for entry in bundle["artifacts"] if entry["schema_name"] == schema_name)


def _build_fixture_closure_payload(tmp_path: Path) -> dict[str, Any]:
    from tools.spec_compiler.compiler import build_pack

    result = build_pack(source_root=REPO_ROOT, output_root=tmp_path / "build")
    assert result.fixture_closure_json_path is not None
    return json.loads(result.fixture_closure_json_path.read_text())


def _scenario_results_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["scenario_id"]: entry for entry in payload["scenarios"]}


def test_live_repo_build_emits_fixture_closure_artifacts(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import build_pack

    output_root = tmp_path / "build"

    result = build_pack(source_root=REPO_ROOT, output_root=output_root)

    assert result.fixture_closure_markdown_path == output_root / "euclid-fixture-closure.md"
    assert result.fixture_closure_json_path == output_root / "euclid-fixture-closure.json"
    assert result.fixture_closure_markdown_path.is_file()
    assert result.fixture_closure_json_path.is_file()

    payload = json.loads(result.fixture_closure_json_path.read_text())

    assert payload["coverage_plan"]["path"] == "fixtures/canonical/fixture-coverage.yaml"
    assert payload["coverage_plan"]["scenario_count"] == len(_coverage_scenarios())
    assert payload["summary"]["all_scenarios_closed"] is True
    assert payload["summary"]["scenarios_checked"] == len(_coverage_scenarios())
    assert payload["summary"]["bundles_loaded"] == len(_coverage_scenarios())
    assert payload["summary"]["walkthroughs_checked"] >= 5
    assert all(entry["status"] == "closed" for entry in payload["scenarios"])


def test_fixture_closure_report_closes_positive_scenarios_from_coverage_plan(tmp_path: Path) -> None:
    payload = _build_fixture_closure_payload(tmp_path)
    scenario_results = _scenario_results_by_id(payload)

    for scenario in _coverage_scenarios("positive_publication"):
        result = scenario_results[scenario["scenario_id"]]

        assert result["bundle_path"] == scenario["planned_fixture_bundle"]
        assert result["scenario_class"] == "positive_publication"
        assert result["status"] == "closed"
        assert result["publication_mode"] == scenario["expected_publication_mode"]
        assert result["terminal_lifecycle_state"] == scenario["expected_terminal_lifecycle_state"]
        assert result["validator_disposition"] == scenario["validator_disposition"]
        assert result["artifact_count"] > 0
        assert result["typed_refs_checked"] > 0
        assert result["walkthrough_path"].startswith("docs/reference/examples/")
        assert result["lifecycle_trace"][-1] == "publication_completed"


def test_fixture_closure_report_tracks_abstention_and_blocked_terminations(tmp_path: Path) -> None:
    payload = _build_fixture_closure_payload(tmp_path)
    scenario_results = _scenario_results_by_id(payload)

    abstention = next(
        scenario
        for scenario in _coverage_scenarios("negative_publication")
        if scenario["expected_publication_mode"] == "abstention_only_publication"
    )
    abstention_result = scenario_results[abstention["scenario_id"]]

    assert abstention_result["publication_mode"] == "abstention_only_publication"
    assert abstention_result["terminal_lifecycle_state"] == "publication_completed"
    assert "abstention_only_publication_selected" in abstention_result["lifecycle_trace"]
    assert abstention_result["lifecycle_trace"][-1] == "publication_completed"

    for scenario in _coverage_scenarios("blocked_publication"):
        result = scenario_results[scenario["scenario_id"]]

        assert result["bundle_path"] == scenario["planned_fixture_bundle"]
        assert result["status"] == "closed"
        assert result["publication_mode"] == "publication_blocked"
        assert result["terminal_lifecycle_state"] == "publication_blocked"
        assert result["validator_disposition"] == scenario["validator_disposition"]
        assert result["typed_refs_checked"] > 0
        assert result["lifecycle_trace"][-1] == "publication_blocked"


def test_fixture_closure_report_rejects_invalid_fixture_scenarios_from_coverage_plan(tmp_path: Path) -> None:
    payload = _build_fixture_closure_payload(tmp_path)
    scenario_results = _scenario_results_by_id(payload)

    for scenario in _coverage_scenarios("invalid_fixture"):
        result = scenario_results[scenario["scenario_id"]]

        assert result["bundle_path"] == scenario["planned_fixture_bundle"]
        assert result["scenario_class"] == "invalid_fixture"
        assert result["status"] == "closed"
        assert result["publication_mode"] == "none"
        assert result["terminal_lifecycle_state"] == "not_applicable_fixture_rejected"
        assert result["validator_disposition"] == "must_reject"
        assert result["typed_refs_checked"] > 0
        assert result["lifecycle_trace"] == []


def test_fixture_closure_rejects_unmaterialized_typed_refs(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_source_root(tmp_path)
    bundle_path = source_root / _coverage_scenario("descriptive-publication")["planned_fixture_bundle"]
    bundle = _load_yaml(bundle_path)
    _artifact(bundle, "claim_card_manifest@1.1.0")["body"]["scorecard_ref"]["object_id"] = (
        "missing_descriptive_scorecard"
    )
    _write_yaml(bundle_path, bundle)

    with pytest.raises(SpecCompilerError, match="unmaterialized fixture ref"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_fixture_closure_rejects_duplicate_object_ids(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_source_root(tmp_path)
    bundle_path = source_root / "fixtures/canonical/publication/descriptive-publication.yaml"
    bundle = _load_yaml(bundle_path)
    bundle["artifacts"].append(bundle["artifacts"][0])
    _write_yaml(bundle_path, bundle)

    with pytest.raises(SpecCompilerError, match="duplicate fixture object_id"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_fixture_closure_rejects_illegal_schema_usage(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_source_root(tmp_path)
    bundle_path = source_root / "fixtures/canonical/publication/descriptive-publication.yaml"
    bundle = _load_yaml(bundle_path)
    bundle["artifacts"].append(
        {
            "schema_name": "unsupported_schema_manifest@9.9.9",
            "object_id": "descriptive_publication_illegal_schema_usage",
            "module_id": "catalog_publishing",
            "body": {"note": "undeclared schema should fail closure validation"},
        }
    )
    _write_yaml(bundle_path, bundle)

    with pytest.raises(SpecCompilerError, match="illegal fixture schema usage"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_fixture_closure_rejects_broken_lifecycle_sequences(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_source_root(tmp_path)
    bundle_path = source_root / "fixtures/canonical/publication/descriptive-publication.yaml"
    bundle = _load_yaml(bundle_path)
    bundle["lifecycle_trace"] = [
        bundle["lifecycle_trace"][0],
        bundle["lifecycle_trace"][5],
        bundle["lifecycle_trace"][1],
    ]
    _write_yaml(bundle_path, bundle)

    with pytest.raises(SpecCompilerError, match="broken lifecycle sequence"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_fixture_closure_fails_closed_on_walkthrough_coverage_holes(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_source_root(tmp_path)
    walkthrough_path = source_root / "docs/reference/examples/blocked-and-invalid-fixtures.md"
    walkthrough_text = walkthrough_path.read_text()
    walkthrough_path.write_text(walkthrough_text.replace("  - illegal-ref-shape", "  - nonexistent-scenario", 1))

    with pytest.raises(SpecCompilerError, match="walkthrough references unknown scenario"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")
