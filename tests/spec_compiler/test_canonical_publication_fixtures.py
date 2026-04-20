from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_COVERAGE_PATH = REPO_ROOT / "fixtures/canonical/fixture-coverage.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing required file: {path.relative_to(REPO_ROOT).as_posix()}"
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict), f"{path.relative_to(REPO_ROOT).as_posix()} must deserialize to an object"
    return payload


def _positive_scenarios() -> list[dict[str, Any]]:
    coverage_plan = _load_yaml(FIXTURE_COVERAGE_PATH)
    return [
        scenario
        for scenario in coverage_plan["scenarios"]
        if scenario["scenario_class"] == "positive_publication"
    ]


def _walk_typed_refs(node: Any) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            schema_name = value.get("schema_name")
            object_id = value.get("object_id")
            if isinstance(schema_name, str) and isinstance(object_id, str):
                refs.append((schema_name, object_id))
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(node)
    return refs


def _artifact_registry(bundle: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    registry: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact in bundle["artifacts"]:
        key = (artifact["schema_name"], artifact["object_id"])
        assert key not in registry, f"duplicate artifact identity in {bundle['bundle_id']}: {key}"
        registry[key] = artifact
    return registry


def test_positive_publication_bundles_exist_for_every_covered_scenario() -> None:
    for scenario in _positive_scenarios():
        bundle_path = REPO_ROOT / scenario["planned_fixture_bundle"]
        bundle = _load_yaml(bundle_path)

        assert bundle["version"] == 1
        assert bundle["kind"] == "canonical_publication_fixture_bundle"
        assert bundle["scenario_id"] == scenario["scenario_id"]
        assert bundle["scenario_class"] == scenario["scenario_class"]


def test_positive_publication_bundles_match_coverage_plan_and_materialize_refs() -> None:
    for scenario in _positive_scenarios():
        bundle_path = REPO_ROOT / scenario["planned_fixture_bundle"]
        bundle = _load_yaml(bundle_path)
        registry = _artifact_registry(bundle)

        assert set(scenario["required_modules"]) <= set(bundle["required_modules"])
        assert set(scenario["required_schema_families"]) <= set(bundle["required_schema_families"])
        assert set(scenario["required_contract_families"]) <= set(bundle["required_contract_families"])
        assert set(scenario["required_evidence_classes"]) <= set(bundle["required_evidence_classes"])

        bundle_schema_names = {artifact["schema_name"] for artifact in bundle["artifacts"]}
        assert set(scenario["required_schema_families"]) <= bundle_schema_names

        for ref in _walk_typed_refs(bundle):
            assert ref in registry, (
                f"{bundle_path.relative_to(REPO_ROOT).as_posix()} contains unmaterialized ref {ref}"
            )


def test_positive_publication_bundles_encode_expected_terminal_outputs() -> None:
    for scenario in _positive_scenarios():
        bundle_path = REPO_ROOT / scenario["planned_fixture_bundle"]
        bundle = _load_yaml(bundle_path)
        registry = _artifact_registry(bundle)

        expected = bundle["expected_outcome"]
        assert expected["validator_disposition"] == scenario["validator_disposition"]
        assert expected["publication_mode"] == scenario["expected_publication_mode"]
        assert expected["terminal_lifecycle_state"] == scenario["expected_terminal_lifecycle_state"]
        assert expected["claim_lane"] == scenario["expected_claim_lane"]
        assert expected["forecast_object_type"] == scenario["expected_forecast_object_type"]
        assert expected["abstention_type"] == scenario["expected_abstention_type"]

        lifecycle_trace = bundle["lifecycle_trace"]
        assert lifecycle_trace[-1]["state"] == scenario["expected_terminal_lifecycle_state"]

        run_result = next(
            artifact for artifact in bundle["artifacts"] if artifact["schema_name"] == "run_result_manifest@1.1.0"
        )
        publication_record = next(
            artifact
            for artifact in bundle["artifacts"]
            if artifact["schema_name"] == "publication_record_manifest@1.1.0"
        )
        claim_card = next(
            artifact for artifact in bundle["artifacts"] if artifact["schema_name"] == "claim_card_manifest@1.1.0"
        )

        assert run_result["body"]["result_mode"] == scenario["expected_publication_mode"]
        assert publication_record["body"]["publication_mode"] == scenario["expected_publication_mode"]
        assert publication_record["body"]["run_result_ref"] == {
            "schema_name": run_result["schema_name"],
            "object_id": run_result["object_id"],
        }
        assert claim_card["body"]["claim_lane"] == scenario["expected_claim_lane"]
        assert claim_card["body"]["forecast_object_type"] == scenario["expected_forecast_object_type"]

        reproducibility_bundle = registry[
            ("reproducibility_bundle_manifest@1.0.0", publication_record["body"]["reproducibility_bundle_ref"]["object_id"])
        ]
        assert reproducibility_bundle["body"]["replay_verification_status"] == "verified"
