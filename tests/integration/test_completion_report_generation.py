from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from euclid.cli import app

RUNNER = CliRunner()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _full_vision_matrix_row_ids(project_root: Path) -> set[str]:
    matrix_payload = _load_yaml(
        project_root / "schemas" / "readiness" / "full-vision-matrix.yaml"
    )
    return {str(row["row_id"]) for row in matrix_payload["rows"]}


def _run_release_status_and_load_completion_report(
    project_root: Path,
) -> dict[str, Any]:
    report_path = project_root / "build" / "reports" / "completion-report.json"
    if report_path.exists():
        report_path.unlink()

    result = RUNNER.invoke(
        app,
        ["release", "status", "--project-root", str(project_root)],
    )

    assert result.exit_code == 0, result.stdout
    assert (
        report_path.is_file()
    ), "release status must materialize build/reports/completion-report.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


def _clean_install_surface_ids() -> set[str]:
    return {
        "release_status",
        "operator_run",
        "operator_replay",
        "benchmark_execution",
        "determinism_same_seed",
        "performance_runtime_smoke",
        "packaged_notebook_smoke",
    }


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
        {
            "evidence_bundle_id": "bundle-shipped",
            "scope_id": "shipped_releasable",
            "authority_snapshot_id": "euclid-authority-2026-04-15-b",
            "command_contract_id": "euclid-certification-commands-v1",
            "closure_map_id": "euclid-closure-map-2026-04-15-v1",
            "traceability_id": "euclid-traceability-2026-04-15-v1",
            "fixture_spec_id": "euclid-certification-fixtures-v1",
            "producer_command_id": "clean_install_certification",
            "generated_at_utc": "2026-04-16T00:00:00Z",
            "input_manifest_digests": [],
            "source_tree_digest_or_wheel_digest": "wheel_digest:test",
            "dirty_state_or_build_toolchain": "python3.11+build",
        },
    ]


def test_release_status_materializes_completion_report_with_split_scores_and_full_matrix_rows(  # noqa: E501
    project_root: Path,
) -> None:
    payload = _run_release_status_and_load_completion_report(project_root)

    assert set(payload["completion_values"]) == {
        "full_vision_completion",
        "current_gate_completion",
        "shipped_releasable_completion",
    }
    assert {entry["policy_id"] for entry in payload["policy_verdicts"]} >= {
        "current_release_v1",
        "full_vision_v1",
        "shipped_releasable_v1",
    }
    assert payload["authority_snapshot_id"] == "euclid-authority-2026-04-15-b"
    assert payload["command_contract_id"] == "euclid-certification-commands-v1"
    assert payload["closure_map_id"] == "euclid-closure-map-2026-04-15-v1"
    assert payload["traceability_id"] == "euclid-traceability-2026-04-15-v1"
    assert payload["fixture_spec_id"] == "euclid-certification-fixtures-v1"
    assert {
        bundle["scope_id"] for bundle in payload["scope_evidence_bundles"]
    } >= {"current_release", "full_vision"}

    capability_rows = payload["capability_rows"]
    assert {
        str(row["row_id"]) for row in capability_rows
    } == _full_vision_matrix_row_ids(project_root)
    assert {str(row["status"]) for row in capability_rows} <= {
        "complete",
        "partial",
        "blocked",
    }
    assert all(isinstance(row["reason_codes"], list) for row in capability_rows)
    assert all(
        row["evidence_refs"] for row in capability_rows
    ), "every capability row must carry at least one evidence reference"
    assert all("evidence_bundle_ids" in row for row in capability_rows)


def test_completion_report_makes_incomplete_rows_and_blockers_explicit(
    project_root: Path,
) -> None:
    payload = _run_release_status_and_load_completion_report(project_root)

    incomplete_rows = [
        row for row in payload["capability_rows"] if row["status"] != "complete"
    ]
    blocker_rows = {
        str(blocker["capability_row_id"]) for blocker in payload["unresolved_blockers"]
    }

    if incomplete_rows:
        assert payload[
            "unresolved_blockers"
        ], "incomplete capability rows must surface unresolved blockers"

    assert blocker_rows <= {str(row["row_id"]) for row in incomplete_rows}
    for row in incomplete_rows:
        assert row[
            "reason_codes"
        ], f"incomplete row {row['row_id']} must explain why it is not complete"
    for blocker in payload["unresolved_blockers"]:
        assert blocker["proof_status"] in {"missing_proof", "failed_proof"}
        assert blocker[
            "reason_codes"
        ], f"blocker for {blocker['capability_row_id']} must emit explicit reason codes"
        assert isinstance(blocker["evidence_refs"], list)


@pytest.mark.timeout(600)
def test_release_certification_flow_captures_clean_install_surfaces_and_packaging_evidence(  # noqa: E501
    project_root: Path,
) -> None:
    clean_install_report_path = (
        project_root / "build" / "reports" / "clean-install-certification.json"
    )
    if clean_install_report_path.exists():
        clean_install_report_path.unlink()

    certify_result = RUNNER.invoke(
        app,
        ["release", "certify-clean-install", "--project-root", str(project_root)],
    )

    assert certify_result.exit_code == 0, certify_result.stdout
    assert clean_install_report_path.is_file(), (
        "clean-install certification must materialize "
        "build/reports/clean-install-certification.json"
    )
    clean_install_payload = json.loads(
        clean_install_report_path.read_text(encoding="utf-8")
    )
    assert {
        str(surface["surface_id"]) for surface in clean_install_payload["surfaces"]
    } == _clean_install_surface_ids()
    assert all(
        str(surface["status"]) == "passed"
        for surface in clean_install_payload["surfaces"]
    )

    payload = _run_release_status_and_load_completion_report(project_root)

    assert set(payload["clean_install_certification"]) == {
        "surface_completion",
        "surfaces",
    }
    assert payload["clean_install_certification"]["surface_completion"] == 1.0
    assert {
        str(surface["surface_id"])
        for surface in payload["clean_install_certification"]["surfaces"]
    } == _clean_install_surface_ids()
    assert all(
        str(surface["status"]) == "passed"
        for surface in payload["clean_install_certification"]["surfaces"]
    )

    readiness_and_closure_row = next(
        row
        for row in payload["capability_rows"]
        if row["row_id"] == "evidence_lane:readiness_and_closure"
    )
    assert (
        "packaging_install" in readiness_and_closure_row["available_evidence_classes"]
    )
    assert any(
        "clean-install-certification.json" in ref
        for ref in readiness_and_closure_row["evidence_refs"]
    )


def test_verify_completion_requires_ready_state_when_full_closure_policy_is_active(
    project_root: Path,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "completion-report.json"
    report_path.write_text(
        json.dumps(
            {
                "report_id": "phase-01-full-closure",
                "generated_at": "2026-04-15T00:00:00Z",
                "policy_verdicts": [
                    {
                        "policy_id": "current_release_v1",
                        "verdict": "ready",
                        "reason_codes": [],
                        "evidence_refs": [],
                    },
                    {
                        "policy_id": "full_vision_v1",
                        "verdict": "review_required",
                        "reason_codes": [],
                        "evidence_refs": [],
                    },
                    {
                        "policy_id": "shipped_releasable_v1",
                        "verdict": "ready",
                        "reason_codes": [],
                        "evidence_refs": [],
                    },
                ],
                "authority_snapshot_id": "euclid-authority-2026-04-15-b",
                "command_contract_id": "euclid-certification-commands-v1",
                "closure_map_id": "euclid-closure-map-2026-04-15-v1",
                "traceability_id": "euclid-traceability-2026-04-15-v1",
                "fixture_spec_id": "euclid-certification-fixtures-v1",
                "scope_evidence_bundles": _scope_evidence_bundles_fixture(),
                "completion_values": {
                    "full_vision_completion": 1.0,
                    "current_gate_completion": 1.0,
                    "shipped_releasable_completion": 1.0,
                },
                "clean_install_certification": {
                    "surface_completion": 1.0,
                    "surfaces": [
                        {
                            "surface_id": surface_id,
                            "status": "passed",
                            "reason_codes": [],
                            "evidence_refs": ["artifact:clean-install-certification.json"],
                        }
                        for surface_id in _clean_install_surface_ids()
                    ],
                },
                "capability_rows": [
                    {
                        "row_id": "evidence_lane:readiness_and_closure",
                        "status": "complete",
                        "reason_codes": [],
                        "evidence_refs": ["artifact:clean-install-certification.json"],
                        "required_evidence_classes": ["packaging_install"],
                        "available_evidence_classes": ["packaging_install"],
                        "non_closing_evidence_classes": [],
                        "evidence_bundle_ids": ["bundle-shipped"],
                    }
                ],
                "residual_risk_codes": [],
                "unresolved_blockers": [],
                "confidence": {"score": 1.0, "reason_codes": []},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    policy_path = tmp_path / "completion-regression-policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "kind": "completion_regression_policy",
                "policy_state": "full_closure",
                "minimum_completion_values": {
                    "full_vision_completion": 1.0,
                    "current_gate_completion": 1.0,
                    "shipped_releasable_completion": 1.0,
                },
                "minimum_policy_verdicts": {
                    "current_release_v1": "ready",
                    "full_vision_v1": "ready",
                    "shipped_releasable_v1": "ready",
                },
                "required_clean_install_surface_ids": sorted(
                    _clean_install_surface_ids()
                ),
                "required_row_evidence_classes": [
                    {
                        "row_id": "evidence_lane:readiness_and_closure",
                        "required_evidence_classes": ["packaging_install"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = RUNNER.invoke(
        app,
        [
            "release",
            "verify-completion",
            "--project-root",
            str(project_root),
            "--report-path",
            str(report_path),
            "--policy-path",
            str(policy_path),
        ],
    )

    assert result.exit_code == 1
    assert "full_vision_v1 fell below ready" in result.stdout
