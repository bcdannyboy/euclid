from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from euclid.cli import app

RUNNER = CliRunner()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _seed_research_ready_inputs(project_root: Path) -> list[tuple[Path, str | None]]:
    build_reports = project_root / "build" / "reports"
    original_payloads: list[tuple[Path, str | None]] = []

    def seed(relative_path: str, payload: dict[str, object]) -> None:
        path = build_reports / relative_path
        original_payloads.append(
            (path, path.read_text(encoding="utf-8") if path.exists() else None)
        )
        _write_json(path, payload)

    seed(
        "completion-report.json",
        {
            "report_id": "euclid_completion_report_v1",
            "authority_snapshot_id": "euclid-authority-2026-04-15-b",
            "command_contract_id": "euclid-certification-commands-v1",
            "closure_map_id": "euclid-closure-map-2026-04-15-v1",
            "traceability_id": "euclid-traceability-2026-04-15-v1",
            "fixture_spec_id": "euclid-certification-fixtures-v1",
            "generated_at": "2026-04-16T00:00:00Z",
            "policy_verdicts": [
                {
                    "policy_id": "current_release_v1",
                    "verdict": "ready",
                    "reason_codes": [],
                    "evidence_refs": ["policy:current_release_v1"],
                },
                {
                    "policy_id": "full_vision_v1",
                    "verdict": "ready",
                    "reason_codes": [],
                    "evidence_refs": ["policy:full_vision_v1"],
                },
                {
                    "policy_id": "shipped_releasable_v1",
                    "verdict": "ready",
                    "reason_codes": [],
                    "evidence_refs": ["policy:shipped_releasable_v1"],
                },
            ],
            "scope_evidence_bundles": [],
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
                        "evidence_refs": [
                            "artifact:clean-install-certification.json"
                        ],
                    }
                    for surface_id in (
                        "release_status",
                        "operator_run",
                        "operator_replay",
                        "benchmark_execution",
                        "determinism_same_seed",
                        "performance_runtime_smoke",
                        "packaged_notebook_smoke",
                    )
                ],
            },
            "capability_rows": [
                {
                    "row_id": "benchmark_surface:portfolio_orchestration",
                    "status": "complete",
                    "reason_codes": [],
                    "evidence_refs": ["artifact:full_vision_suite_evidence.json"],
                    "required_evidence_classes": ["benchmark_semantic", "replay"],
                    "available_evidence_classes": ["benchmark_semantic", "replay"],
                    "non_closing_evidence_classes": [],
                    "evidence_bundle_ids": [],
                }
            ],
            "residual_risk_codes": [],
            "unresolved_blockers": [],
            "confidence": {"score": 1.0, "reason_codes": []},
        },
    )
    seed(
        "repo_test_matrix.json",
        {
            "report_id": "repo_test_matrix_v1",
            "authority_snapshot_id": "euclid-authority-2026-04-15-b",
            "command_contract_id": "euclid-certification-commands-v1",
            "command": "python3.11 -m pytest -q",
            "generated_at_utc": "2026-04-16T00:00:00Z",
            "passed": True,
            "exit_code": 0,
            "summary_line": "100 passed",
            "counts": {
                "passed": 100,
                "failed": 0,
                "skipped": 0,
                "xfailed": 0,
                "xpassed": 0,
            },
            "stdout": "",
            "stderr": "",
        },
    )
    seed(
        "current_release_suite_evidence.json",
        {
            "report_id": "current_release_suite_evidence_v1",
            "scope_id": "current_release",
            "surface_statuses": [
                {
                    "surface_id": "retained_core_release",
                    "benchmark_status": "passed",
                    "replay_status": "passed",
                }
            ],
        },
    )
    seed(
        "full_vision_suite_evidence.json",
        {
            "report_id": "full_vision_suite_evidence_v1",
            "scope_id": "full_vision",
            "surface_statuses": [
                {
                    "surface_id": surface_id,
                    "benchmark_status": "passed",
                    "replay_status": "passed",
                }
                for surface_id in (
                    "retained_core_release",
                    "probabilistic_forecast_surface",
                    "algorithmic_backend",
                    "composition_operator_semantics",
                    "shared_plus_local_decomposition",
                    "mechanistic_lane",
                    "external_evidence_ingestion",
                    "robustness_lane",
                    "portfolio_orchestration",
                )
            ],
        },
    )
    seed(
        "full_vision_operator_run_evidence.json",
        {
            "report_id": "operator_run_evidence_v1",
            "command_id": "full_vision_operator_run",
            "scope_id": "full_vision",
            "run_id": "full-vision-run",
        },
    )
    seed(
        "full_vision_operator_replay_evidence.json",
        {
            "report_id": "operator_replay_evidence_v1",
            "command_id": "full_vision_operator_replay",
            "scope_id": "full_vision",
            "run_id": "full-vision-run",
            "replay_verification_status": "verified",
        },
    )
    seed(
        "clean-install-certification.json",
        {
            "report_id": "euclid_clean_install_certification_v1",
            "surface_completion": 1.0,
            "surfaces": [
                {
                    "surface_id": surface_id,
                    "status": "passed",
                    "reason_codes": [],
                    "evidence_refs": ["artifact:clean-install-certification.json"],
                }
                for surface_id in (
                    "release_status",
                    "operator_run",
                    "operator_replay",
                    "benchmark_execution",
                    "determinism_same_seed",
                    "performance_runtime_smoke",
                    "packaged_notebook_smoke",
                )
            ],
        },
    )
    return original_payloads


def _restore_seeded_inputs(original_payloads: list[tuple[Path, str | None]]) -> None:
    for path, original_payload in reversed(original_payloads):
        if original_payload is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(original_payload, encoding="utf-8")


def test_research_readiness_certification_materializes_fail_closed_report(
    project_root: Path,
) -> None:
    seeded_inputs = _seed_research_ready_inputs(project_root)
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
            "composition_operator_semantics",
            "shared_plus_local_decomposition",
            "mechanistic_lane",
            "external_evidence_ingestion",
            "robustness_lane",
            "portfolio_orchestration",
        ]
    finally:
        _restore_seeded_inputs(seeded_inputs)
