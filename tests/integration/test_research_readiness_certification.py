from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.research_readiness import (
    restore_seeded_inputs,
    seed_research_ready_inputs,
    write_json,
)
from typer.testing import CliRunner

from euclid.cli import app

RUNNER = CliRunner()


def test_research_readiness_certification_materializes_fail_closed_report(
    project_root: Path,
) -> None:
    seeded_inputs = seed_research_ready_inputs(project_root)
    report_path = project_root / "build" / "reports" / "research-readiness.json"
    if report_path.exists():
        report_path.unlink()

    try:
        result = RUNNER.invoke(
            app,
            [
                "release",
                "certify-research-readiness",
                "--project-root",
                str(project_root),
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert report_path.is_file()

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["authority_snapshot_id"] == "euclid-authority-2026-04-15-b"
        assert payload["command_contract_id"] == "euclid-certification-commands-v1"
        assert payload["status"] == "ready"
        assert payload["policy_verdicts"]["current_release_v1"] == "ready"
        assert payload["policy_verdicts"]["full_vision_v1"] == "ready"
        assert payload["policy_verdicts"]["shipped_releasable_v1"] == "ready"
        assert payload["required_surface_ids"] == [
            "retained_core_release",
            "probabilistic_forecast_surface",
            "algorithmic_backend",
            "search_class_honesty",
            "composition_operator_semantics",
            "shared_plus_local_decomposition",
            "mechanistic_lane",
            "external_evidence_ingestion",
            "robustness_lane",
            "portfolio_orchestration",
        ]
    finally:
        restore_seeded_inputs(seeded_inputs)


def test_research_readiness_rejects_failed_full_vision_surface_status(
    project_root: Path,
) -> None:
    seeded_inputs = seed_research_ready_inputs(project_root)
    full_suite_path = (
        project_root / "build" / "reports" / "full_vision_suite_evidence.json"
    )
    try:
        full_suite_payload = json.loads(full_suite_path.read_text(encoding="utf-8"))
        for row in full_suite_payload["surface_statuses"]:
            if row["surface_id"] == "mechanistic_lane":
                row["benchmark_status"] = "failed"
                row["replay_status"] = "passed"
                break
        write_json(full_suite_path, full_suite_payload)

        result = RUNNER.invoke(
            app,
            [
                "release",
                "certify-research-readiness",
                "--project-root",
                str(project_root),
            ],
        )

        assert result.exit_code == 1, result.stdout
        report_path = project_root / "build" / "reports" / "research-readiness.json"
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["status"] == "blocked"
        assert "full_vision_surface_status_failed" in payload["reason_codes"]
        assert payload["full_vision_surface_status_failures"] == [
            {
                "surface_id": "mechanistic_lane",
                "benchmark_status": "failed",
                "replay_status": "passed",
            }
        ]
    finally:
        restore_seeded_inputs(seeded_inputs)


def test_research_readiness_requires_declared_search_class_surface(
    project_root: Path,
) -> None:
    seeded_inputs = seed_research_ready_inputs(project_root)
    full_suite_path = (
        project_root / "build" / "reports" / "full_vision_suite_evidence.json"
    )
    try:
        full_suite_payload = json.loads(full_suite_path.read_text(encoding="utf-8"))
        full_suite_payload["surface_statuses"] = [
            row
            for row in full_suite_payload["surface_statuses"]
            if row["surface_id"] != "search_class_honesty"
        ]
        write_json(full_suite_path, full_suite_payload)

        result = RUNNER.invoke(
            app,
            [
                "release",
                "certify-research-readiness",
                "--project-root",
                str(project_root),
            ],
        )

        assert result.exit_code == 1, result.stdout
        report_path = project_root / "build" / "reports" / "research-readiness.json"
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["status"] == "blocked"
        assert "full_vision_surface_coverage_incomplete" in payload["reason_codes"]
    finally:
        restore_seeded_inputs(seeded_inputs)
