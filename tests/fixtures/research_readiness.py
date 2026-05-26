from __future__ import annotations

import hashlib
import json
from pathlib import Path

import euclid.release as release


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def runtime_sha256(path: Path) -> str:
    return "runtime_sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_sha256_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "runtime_sha256:" + hashlib.sha256(encoded).hexdigest()


def seed_research_ready_inputs(project_root: Path) -> list[tuple[Path, str | None]]:
    build_reports = project_root / "build" / "reports"
    original_payloads: list[tuple[Path, str | None]] = []
    source_digest = release._release_source_digest_ref(project_root)
    shipped_bundle = release._bundle_by_scope_id()["shipped_releasable"]

    def seed(relative_path: str, payload: dict[str, object]) -> Path:
        path = build_reports / relative_path
        original_payloads.append(
            (path, path.read_text(encoding="utf-8") if path.exists() else None)
        )
        write_json(path, payload)
        return path

    def build_artifact(relative_path: str, payload: dict[str, object]) -> Path:
        path = project_root / "build" / relative_path
        original_payloads.append(
            (path, path.read_text(encoding="utf-8") if path.exists() else None)
        )
        write_json(path, payload)
        return path

    def build_text_artifact(relative_path: str, text: str) -> Path:
        path = project_root / "build" / relative_path
        original_payloads.append(
            (path, path.read_text(encoding="utf-8") if path.exists() else None)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    current_suite_summary = build_artifact(
        "research-ready/current-release-summary.json",
        {"suite_id": "current_release", "status": "passed"},
    )
    full_suite_summary = build_artifact(
        "research-ready/full-vision-summary.json",
        {"suite_id": "full_vision", "status": "passed"},
    )
    full_run_summary = build_artifact(
        "research-ready/full-vision-run-summary.json",
        {
            "request_id": "full-vision-run",
            "run_id": "full-vision-run",
            "status": "completed",
        },
    )
    clean_install_output_root = (
        project_root / "build" / "research-ready" / "clean-install"
    )
    full_run_output_root = project_root / "build" / "research-ready" / "full-run"
    clean_install_output_root.mkdir(parents=True, exist_ok=True)
    full_run_output_root.mkdir(parents=True, exist_ok=True)
    clean_surface_ids = (
        "release_status",
        "operator_run",
        "operator_replay",
        "benchmark_execution",
        "determinism_same_seed",
        "performance_runtime_smoke",
        "packaged_notebook_smoke",
    )
    clean_wheelhouse = clean_install_output_root / "dist"
    clean_wheelhouse.mkdir(parents=True, exist_ok=True)
    clean_wheel = build_text_artifact(
        "research-ready/clean-install/dist/euclid-1.0.0-py3-none-any.whl",
        "research-ready clean-install wheel\n",
    )
    clean_wheel_digest = hashlib.sha256(clean_wheel.read_bytes()).hexdigest()
    clean_surface_artifacts = {
        surface_id: build_text_artifact(
            f"research-ready/clean-install/logs/{surface_id}.stdout.log",
            f"{surface_id} passed\n",
        )
        for surface_id in clean_surface_ids
    }

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
                    for surface_id in clean_surface_ids
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
            "producer_command_id": "repo_test_matrix",
            "source_tree_digest_or_wheel_digest": source_digest,
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
            "summary_counts_parsed": True,
            "stdout": "",
            "stderr": "",
        },
    )
    seed(
        "current_release_suite_evidence.json",
        {
            "report_id": "current_release_suite_evidence_v1",
            "scope_id": "current_release",
            "producer_command_id": "current_release_suite",
            "source_tree_digest_or_wheel_digest": source_digest,
            "summary_path": str(current_suite_summary),
            "summary_sha256": runtime_sha256(current_suite_summary),
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
            "producer_command_id": "full_vision_suite",
            "source_tree_digest_or_wheel_digest": source_digest,
            "summary_path": str(full_suite_summary),
            "summary_sha256": runtime_sha256(full_suite_summary),
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
                    "search_class_honesty",
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
    run_result_ref = {
        "schema_name": "run_result_manifest@1.1.0",
        "object_id": "run-result-1",
    }
    bundle_ref = {
        "schema_name": "reproducibility_bundle_manifest@1.1.0",
        "object_id": "bundle-1",
    }
    operator_run_report_path = seed(
        "full_vision_operator_run_evidence.json",
        {
            "report_id": "operator_run_evidence_v1",
            "command_id": "full_vision_operator_run",
            "scope_id": "full_vision",
            "run_id": "full-vision-run",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(full_run_summary),
            "run_summary_sha256": runtime_sha256(full_run_summary),
            "output_root": str(full_run_output_root),
            "run_result_ref": run_result_ref,
            "bundle_ref": bundle_ref,
            "run_id_binding": {
                "request_id": "full-vision-run",
                "run_result_object_id": run_result_ref["object_id"],
                "run_summary_request_id": "full-vision-run",
            },
        },
    )
    operator_run_report_sha256 = runtime_sha256(operator_run_report_path)
    seed(
        "full_vision_operator_replay_evidence.json",
        {
            "report_id": "operator_replay_evidence_v1",
            "command_id": "full_vision_operator_replay",
            "scope_id": "full_vision",
            "run_id": "full-vision-run",
            "replay_verification_status": "verified",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(full_run_summary),
            "run_summary_sha256": runtime_sha256(full_run_summary),
            "output_root": str(full_run_output_root),
            "operator_run_evidence_report_path": str(operator_run_report_path),
            "operator_run_evidence_report_sha256": operator_run_report_sha256,
            "operator_run_evidence_report_binding": {
                "path": str(operator_run_report_path),
                "sha256": operator_run_report_sha256,
                "run_id": "full-vision-run",
            },
            "run_result_ref": run_result_ref,
            "bundle_ref": bundle_ref,
            "run_id_binding": {
                "requested_run_id": "full-vision-run",
                "run_result_object_id": run_result_ref["object_id"],
                "run_summary_request_id": "full-vision-run",
            },
            "replay_result_sha256": runtime_sha256_payload(
                {
                    "run_id": "full-vision-run",
                    "run_summary_sha256": runtime_sha256(full_run_summary),
                    "run_result_ref": run_result_ref,
                    "bundle_ref": bundle_ref,
                    "replay_verification_status": "verified",
                }
            ),
        },
    )
    seed(
        "clean-install-certification.json",
        {
            "report_id": "euclid_clean_install_certification_v1",
            "evidence_bundle_id": shipped_bundle["evidence_bundle_id"],
            "scope_id": shipped_bundle["scope_id"],
            "authority_snapshot_id": shipped_bundle["authority_snapshot_id"],
            "command_contract_id": shipped_bundle["command_contract_id"],
            "closure_map_id": shipped_bundle["closure_map_id"],
            "traceability_id": shipped_bundle["traceability_id"],
            "fixture_spec_id": shipped_bundle["fixture_spec_id"],
            "producer_command_id": shipped_bundle["producer_command_id"],
            "canonical_report_path": str(
                project_root / "build" / "reports" / "clean-install-certification.json"
            ),
            "source_tree_digest_at_build": source_digest,
            "source_tree_digest_or_wheel_digest": f"wheel_digest:{clean_wheel_digest}",
            "input_manifest_digests": [
                {
                    str(clean_wheelhouse): (
                        "runtime_directory_digest:"
                        f"{release._directory_digest(clean_wheelhouse)}"
                    )
                }
            ],
            "wheel_path": str(clean_wheel),
            "wheel_digest": clean_wheel_digest,
            "output_root": str(clean_install_output_root),
            "runtime_dependency_wheelhouse": str(clean_wheelhouse),
            "runtime_dependency_wheel_count": 0,
            "surface_completion": 1.0,
            "surfaces": [
                {
                    "surface_id": surface_id,
                    "status": "passed",
                    "reason_codes": [],
                    "evidence_refs": [
                        f"artifact:{clean_surface_artifacts[surface_id]}"
                    ],
                }
                for surface_id in clean_surface_ids
            ],
        },
    )
    return original_payloads


def restore_seeded_inputs(original_payloads: list[tuple[Path, str | None]]) -> None:
    for path, original_payload in reversed(original_payloads):
        if original_payload is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(original_payload, encoding="utf-8")
