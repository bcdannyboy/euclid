from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_COVERAGE_PATH = REPO_ROOT / "fixtures/canonical/fixture-coverage.yaml"
ACCEPTED_FAILURE_SCENARIO_CLASSES = {"negative_publication", "blocked_publication"}


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing required file: {path.relative_to(REPO_ROOT).as_posix()}"
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict), f"{path.relative_to(REPO_ROOT).as_posix()} must deserialize to an object"
    return payload


def _scenarios(*scenario_classes: str) -> list[dict[str, Any]]:
    coverage_plan = _load_yaml(FIXTURE_COVERAGE_PATH)
    return [
        scenario
        for scenario in coverage_plan["scenarios"]
        if scenario["scenario_class"] in set(scenario_classes)
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
    for artifact in bundle.get("artifacts", []):
        key = (artifact["schema_name"], artifact["object_id"])
        assert key not in registry, f"duplicate artifact identity in {bundle['bundle_id']}: {key}"
        registry[key] = artifact
    return registry


def _collect_reason_code_lists(node: Any) -> list[list[str]]:
    reason_lists: list[list[str]] = []
    reason_keys = {
        "reason_codes",
        "failure_reason_codes",
        "predictive_reason_codes",
        "descriptive_reason_codes",
        "fresh_replication_reason_codes",
        "blocking_reason_codes",
    }

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in reason_keys and isinstance(nested, list) and all(isinstance(item, str) for item in nested):
                    reason_lists.append(nested)
                walk(nested)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(node)
    return reason_lists


def test_failure_and_blocked_fixture_bundles_exist_for_every_covered_scenario() -> None:
    for scenario in _scenarios(*ACCEPTED_FAILURE_SCENARIO_CLASSES):
        bundle_path = REPO_ROOT / scenario["planned_fixture_bundle"]
        bundle = _load_yaml(bundle_path)

        assert bundle["version"] == 1
        assert bundle["kind"] == "canonical_publication_fixture_bundle"
        assert bundle["scenario_id"] == scenario["scenario_id"]
        assert bundle["scenario_class"] == scenario["scenario_class"]


def test_failure_and_blocked_bundles_match_coverage_plan_and_materialize_refs() -> None:
    for scenario in _scenarios(*ACCEPTED_FAILURE_SCENARIO_CLASSES):
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


def test_failure_and_blocked_bundles_encode_typed_terminal_outcomes() -> None:
    for scenario in _scenarios(*ACCEPTED_FAILURE_SCENARIO_CLASSES):
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
        assert expected["failure_effect"] == scenario["expected_failure_effect"]

        lifecycle_trace = bundle["lifecycle_trace"]
        assert lifecycle_trace[-1]["state"] == scenario["expected_terminal_lifecycle_state"]

        run_results = [artifact for artifact in bundle["artifacts"] if artifact["schema_name"] == "run_result_manifest@1.1.0"]
        publication_records = [
            artifact for artifact in bundle["artifacts"] if artifact["schema_name"] == "publication_record_manifest@1.1.0"
        ]
        abstentions = [artifact for artifact in bundle["artifacts"] if artifact["schema_name"] == "abstention_manifest@1.1.0"]
        claim_cards = [artifact for artifact in bundle["artifacts"] if artifact["schema_name"] == "claim_card_manifest@1.1.0"]

        if run_results:
            assert run_results[0]["body"]["result_mode"] == scenario["expected_publication_mode"]

        if publication_records:
            publication_record = publication_records[0]
            assert publication_record["body"]["publication_mode"] == scenario["expected_publication_mode"]
            reproducibility_bundle = registry[
                (
                    "reproducibility_bundle_manifest@1.0.0",
                    publication_record["body"]["reproducibility_bundle_ref"]["object_id"],
                )
            ]
            assert reproducibility_bundle["body"]["replay_verification_status"] == "verified"

        if scenario["expected_abstention_type"] is not None:
            assert len(abstentions) == 1
            assert not claim_cards
            assert abstentions[0]["body"]["abstention_type"] == scenario["expected_abstention_type"]
        elif scenario["expected_claim_lane"] is not None:
            assert len(claim_cards) == 1
            assert claim_cards[0]["body"]["claim_lane"] == scenario["expected_claim_lane"]
            assert claim_cards[0]["body"]["forecast_object_type"] == scenario["expected_forecast_object_type"]

        if scenario.get("expected_reason_codes"):
            assert set(expected["reason_codes"]) == set(scenario["expected_reason_codes"])
            observed_reason_codes = {
                code
                for codes in _collect_reason_code_lists(bundle)
                for code in codes
            }
            assert set(scenario["expected_reason_codes"]) <= observed_reason_codes


def test_invalid_fixture_bundles_exist_with_machine_readable_validator_errors() -> None:
    for scenario in _scenarios("invalid_fixture"):
        bundle_path = REPO_ROOT / scenario["planned_fixture_bundle"]
        bundle = _load_yaml(bundle_path)

        assert bundle["version"] == 1
        assert bundle["kind"] == "canonical_publication_fixture_bundle"
        assert bundle["scenario_id"] == scenario["scenario_id"]
        assert bundle["scenario_class"] == scenario["scenario_class"]
        assert set(scenario["required_modules"]) <= set(bundle["required_modules"])
        assert set(scenario["required_schema_families"]) <= set(bundle["required_schema_families"])
        assert set(scenario["required_contract_families"]) <= set(bundle["required_contract_families"])

        expected = bundle["expected_outcome"]
        assert expected["validator_disposition"] == "must_reject"
        assert expected["publication_mode"] == "none"
        assert expected["terminal_lifecycle_state"] == "not_applicable_fixture_rejected"
        assert expected["failure_effect"] == scenario["expected_failure_effect"]

        validator_errors = bundle.get("validator_errors")
        assert isinstance(validator_errors, list) and validator_errors, (
            f"{bundle_path.relative_to(REPO_ROOT).as_posix()} must declare explicit validator_errors"
        )
        for error in validator_errors:
            assert isinstance(error.get("error_code"), str) and error["error_code"]
            assert isinstance(error.get("message"), str) and error["message"]
            artifact_ref = error.get("artifact_ref")
            assert isinstance(artifact_ref, dict), "validator error must point at a typed artifact ref"
            assert isinstance(artifact_ref.get("schema_name"), str) and artifact_ref["schema_name"]
            assert isinstance(artifact_ref.get("object_id"), str) and artifact_ref["object_id"]
