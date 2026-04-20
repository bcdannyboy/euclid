from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPLETION_REPORT_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas/readiness/completion-report.schema.yaml"
)


def _scope_evidence_bundles_fixture() -> list[dict[str, object]]:
    return [
        {
            "evidence_bundle_id": "bundle-current",
            "scope_id": "current_release",
            "authority_snapshot_id": "euclid-authority-2026-04-15-b",
            "command_contract_id": "euclid-certification-commands-v1",
            "closure_map_id": "euclid-closure-map-2026-04-15-v1",
            "traceability_id": "euclid-traceability-2026-04-15-v1",
            "fixture_spec_id": "euclid-certification-fixtures-v1",
            "producer_command_id": "current_release_suite",
            "generated_at_utc": "2026-04-16T00:00:00Z",
            "input_manifest_digests": [],
            "source_tree_digest_or_wheel_digest": "repo_checkout_digest:test",
            "dirty_state_or_build_toolchain": "repo_checkout_dirty_state:clean",
        },
        {
            "evidence_bundle_id": "bundle-full",
            "scope_id": "full_vision",
            "authority_snapshot_id": "euclid-authority-2026-04-15-b",
            "command_contract_id": "euclid-certification-commands-v1",
            "closure_map_id": "euclid-closure-map-2026-04-15-v1",
            "traceability_id": "euclid-traceability-2026-04-15-v1",
            "fixture_spec_id": "euclid-certification-fixtures-v1",
            "producer_command_id": "full_vision_suite",
            "generated_at_utc": "2026-04-16T00:00:00Z",
            "input_manifest_digests": [],
            "source_tree_digest_or_wheel_digest": "repo_checkout_digest:test",
            "dirty_state_or_build_toolchain": "repo_checkout_dirty_state:clean",
        },
    ]


def _load_yaml(path: Path) -> dict[str, Any]:
    assert (
        path.is_file()
    ), f"missing required file: {path.relative_to(PROJECT_ROOT).as_posix()}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _validate_completion_report_payload(
    *,
    schema: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    required_fields = set(schema["required_fields"])
    policy_verdict_required_fields = set(schema["policy_verdict_required_fields"])
    scope_evidence_bundle_required_fields = set(
        schema["scope_evidence_bundle_required_fields"]
    )
    capability_row_required_fields = set(schema["capability_row_required_fields"])
    assert required_fields <= payload.keys()

    for verdict in payload["policy_verdicts"]:
        assert policy_verdict_required_fields <= set(verdict)
        assert isinstance(verdict["reason_codes"], list)
        assert isinstance(verdict["evidence_refs"], list)

    for bundle in payload["scope_evidence_bundles"]:
        assert scope_evidence_bundle_required_fields <= set(bundle)

    completion_field_ids = {entry["id"] for entry in schema["completion_value_fields"]}
    completion_values = payload["completion_values"]
    assert completion_field_ids == set(completion_values)
    for value in completion_values.values():
        assert isinstance(value, (int, float))
        assert 0.0 <= float(value) <= 1.0

    clean_install = payload["clean_install_certification"]
    assert isinstance(clean_install["surface_completion"], (int, float))
    assert 0.0 <= float(clean_install["surface_completion"]) <= 1.0
    surface_status_values = set(schema["clean_install_surface_status_values"])
    for surface in clean_install["surfaces"]:
        assert surface["status"] in surface_status_values
        assert isinstance(surface["reason_codes"], list)
        assert isinstance(surface["evidence_refs"], list)

    capability_status_values = set(schema["capability_status_values"])
    for row in payload["capability_rows"]:
        assert capability_row_required_fields <= set(row)
        assert row["status"] in capability_status_values
        assert isinstance(row["reason_codes"], list)
        assert isinstance(row["evidence_refs"], list)
        assert isinstance(row["evidence_bundle_ids"], list)

    proof_status_values = set(schema["proof_status_values"])
    for blocker in payload["unresolved_blockers"]:
        assert blocker["proof_status"] in proof_status_values
        assert isinstance(blocker["reason_codes"], list)
        assert isinstance(blocker["evidence_refs"], list)

    confidence = payload["confidence"]
    assert isinstance(confidence["score"], (int, float))
    assert 0.0 <= float(confidence["score"]) <= 1.0
    assert isinstance(confidence["reason_codes"], list)


def test_completion_report_schema_declares_split_completion_fields_and_status_values(  # noqa: E501
) -> (
    None
):
    schema = _load_yaml(COMPLETION_REPORT_SCHEMA_PATH)

    assert schema["version"] == 1
    assert schema["kind"] == "completion_report_schema"
    assert {entry["id"] for entry in schema["completion_value_fields"]} == {
        "full_vision_completion",
        "current_gate_completion",
        "shipped_releasable_completion",
    }
    assert {
        "report_id",
        "authority_snapshot_id",
        "command_contract_id",
        "closure_map_id",
        "traceability_id",
        "fixture_spec_id",
        "generated_at",
        "policy_verdicts",
        "scope_evidence_bundles",
        "completion_values",
        "clean_install_certification",
        "capability_rows",
        "residual_risk_codes",
        "unresolved_blockers",
        "confidence",
    } <= set(schema["required_fields"])
    assert set(schema["clean_install_surface_status_values"]) == {
        "passed",
        "missing",
        "failed",
    }
    assert set(schema["capability_status_values"]) == {
        "complete",
        "partial",
        "blocked",
    }
    assert set(schema["proof_status_values"]) == {
        "missing_proof",
        "failed_proof",
    }


def test_completion_report_contract_accepts_independent_completion_scores() -> None:
    schema = _load_yaml(COMPLETION_REPORT_SCHEMA_PATH)
    payload = {
        "report_id": "phase-01-full-vision-baseline",
        "authority_snapshot_id": "euclid-authority-2026-04-15-b",
        "command_contract_id": "euclid-certification-commands-v1",
        "closure_map_id": "euclid-closure-map-2026-04-15-v1",
        "traceability_id": "euclid-traceability-2026-04-15-v1",
        "fixture_spec_id": "euclid-certification-fixtures-v1",
        "generated_at": "2026-04-14T00:00:00Z",
        "policy_verdicts": [
            {
                "policy_id": "current_release_v1",
                "verdict": "ready",
                "reason_codes": [],
                "evidence_refs": ["readiness:current_release_v1"],
            },
            {
                "policy_id": "full_vision_v1",
                "verdict": "blocked",
                "reason_codes": ["forecast_object_type.distribution_missing_runtime"],
                "evidence_refs": ["matrix:forecast_object_type:distribution"],
            },
            {
                "policy_id": "shipped_releasable_v1",
                "verdict": "review_required",
                "reason_codes": ["clean_install.release_status_missing"],
                "evidence_refs": ["policy:shipped_releasable_v1"],
            },
        ],
        "scope_evidence_bundles": _scope_evidence_bundles_fixture(),
        "completion_values": {
            "full_vision_completion": 0.42,
            "current_gate_completion": 0.81,
            "shipped_releasable_completion": 0.67,
        },
        "clean_install_certification": {
            "surface_completion": 0.857143,
            "surfaces": [
                {
                    "surface_id": "release_status",
                    "status": "passed",
                    "reason_codes": [],
                    "evidence_refs": ["artifact:clean-install-certification.json"],
                },
                {
                    "surface_id": "performance_runtime_smoke",
                    "status": "missing",
                    "reason_codes": ["clean_install.performance_runtime_smoke_missing"],
                    "evidence_refs": [],
                },
            ],
        },
        "capability_rows": [
            {
                "row_id": "forecast_object_type:point",
                "status": "complete",
                "reason_codes": [],
                "evidence_refs": ["run_result:point"],
                "required_evidence_classes": ["benchmark_semantic", "replay"],
                "available_evidence_classes": ["benchmark_semantic", "replay"],
                "non_closing_evidence_classes": [],
                "evidence_bundle_ids": ["bundle-current"],
            },
            {
                "row_id": "forecast_object_type:distribution",
                "status": "partial",
                "reason_codes": ["semantic_runtime_missing"],
                "evidence_refs": ["notebook:probabilistic-demo"],
                "required_evidence_classes": ["benchmark_semantic", "replay"],
                "available_evidence_classes": [],
                "non_closing_evidence_classes": ["notebook_smoke"],
                "evidence_bundle_ids": ["bundle-full"],
            },
        ],
        "residual_risk_codes": [
            "probabilistic.operator_runtime_missing",
            "install.clean_wheel_unproven",
        ],
        "unresolved_blockers": [
            {
                "capability_row_id": "forecast_object_type:distribution",
                "proof_status": "missing_proof",
                "reason_codes": ["semantic_runtime_missing"],
                "evidence_refs": [],
            }
        ],
        "confidence": {
            "score": 0.61,
            "reason_codes": ["current_release_scope_is_narrower_than_full_vision"],
        },
    }

    _validate_completion_report_payload(schema=schema, payload=payload)

    assert (
        payload["completion_values"]["full_vision_completion"]
        < payload["completion_values"]["current_gate_completion"]
    )
    assert (
        payload["completion_values"]["shipped_releasable_completion"]
        < payload["completion_values"]["current_gate_completion"]
    )


def test_completion_report_contract_requires_missing_and_failed_proof_blockers() -> (
    None
):
    schema = _load_yaml(COMPLETION_REPORT_SCHEMA_PATH)
    payload = {
        "report_id": "phase-01-proof-statuses",
        "authority_snapshot_id": "euclid-authority-2026-04-15-b",
        "command_contract_id": "euclid-certification-commands-v1",
        "closure_map_id": "euclid-closure-map-2026-04-15-v1",
        "traceability_id": "euclid-traceability-2026-04-15-v1",
        "fixture_spec_id": "euclid-certification-fixtures-v1",
        "generated_at": "2026-04-14T00:00:00Z",
        "policy_verdicts": [],
        "scope_evidence_bundles": [],
        "completion_values": {
            "full_vision_completion": 0.4,
            "current_gate_completion": 0.75,
            "shipped_releasable_completion": 0.65,
        },
        "clean_install_certification": {
            "surface_completion": 0.428571,
            "surfaces": [
                {
                    "surface_id": "release_status",
                    "status": "failed",
                    "reason_codes": ["clean_install.release_status_failed"],
                    "evidence_refs": ["artifact:clean-install-log"],
                }
            ],
        },
        "capability_rows": [
            {
                "row_id": "lifecycle_artifact:run_result",
                "status": "blocked",
                "reason_codes": ["clean_install_failed"],
                "evidence_refs": ["artifact:clean-install-log"],
                "required_evidence_classes": ["semantic_runtime"],
                "available_evidence_classes": [],
                "non_closing_evidence_classes": [],
                "evidence_bundle_ids": [],
            }
        ],
        "residual_risk_codes": [
            "runtime.clean_install_failed",
            "runtime.probabilistic_support_missing",
        ],
        "unresolved_blockers": [
            {
                "capability_row_id": "forecast_object_type:distribution",
                "proof_status": "missing_proof",
                "reason_codes": ["semantic_runtime_missing"],
                "evidence_refs": [],
            },
            {
                "capability_row_id": "lifecycle_artifact:run_result",
                "proof_status": "failed_proof",
                "reason_codes": ["clean_install_failed"],
                "evidence_refs": ["artifact:clean-install-log"],
            },
        ],
        "confidence": {
            "score": 0.5,
            "reason_codes": ["multiple_runtime_surfaces_unproven"],
        },
    }

    _validate_completion_report_payload(schema=schema, payload=payload)

    assert {item["proof_status"] for item in payload["unresolved_blockers"]} == {
        "missing_proof",
        "failed_proof",
    }
