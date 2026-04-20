from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import euclid
import euclid.release as release
from euclid.cli import app
from euclid.readiness import ReadinessGateResult

RUNNER = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def test_release_api_surfaces_split_policy_judgments_from_release_evidence(
    project_root: Path,
    tmp_path: Path,
) -> None:
    status = euclid.get_release_status(
        project_root=project_root,
        benchmark_root=tmp_path / "benchmarks",
        notebook_output_root=tmp_path / "notebook-smoke",
        supplemental_gate_results=(
            ReadinessGateResult(
                gate_id="determinism.same_seed",
                status="passed",
                required=True,
                summary="Repeated same-seed runs emitted stable hashes.",
                evidence={"check": "determinism"},
            ),
            ReadinessGateResult(
                gate_id="performance.runtime_smoke",
                status="passed",
                required=True,
                summary="Runtime smoke stayed within the declared threshold.",
                evidence={"check": "performance"},
            ),
        ),
    )

    assert status.project_root == project_root
    assert status.current_version == euclid.__version__
    assert status.target_version == "1.0.0"
    assert status.target_ready is False
    assert set(status.policy_judgments) == {
        "current_release_v1",
        "full_vision_v1",
        "shipped_releasable_v1",
    }
    assert status.readiness_judgment == status.shipped_releasable_judgment
    assert status.target_ready == (
        status.shipped_releasable_judgment.final_verdict == "ready"
    )
    assert status.catalog_scope == status.shipped_releasable_judgment.catalog_scope
    if status.target_ready:
        assert status.shipped_releasable_judgment.final_verdict == "ready"
        assert status.catalog_scope == "public"
        assert status.blocked_reason == ""
    else:
        assert status.shipped_releasable_judgment.final_verdict == "blocked"
        assert status.catalog_scope == "internal"
        assert status.blocked_reason != ""


def test_release_api_validates_contract_catalog(project_root: Path) -> None:
    result = euclid.validate_release_contracts(project_root=project_root)

    assert result.project_root == project_root
    assert result.schema_count > 0
    assert result.module_count > 0
    assert result.enum_count > 0
    assert result.contract_document_count > 0


def test_release_api_runs_benchmark_smoke_for_each_track(
    tmp_path: Path,
    project_root: Path,
) -> None:
    result = euclid.run_release_benchmark_smoke(
        project_root=project_root,
        benchmark_root=tmp_path / "benchmarks",
    )

    assert [case.track_id for case in result.cases] == [
        "rediscovery",
        "predictive_generalization",
        "adversarial_honesty",
    ]
    for case in result.cases:
        assert case.report_path.is_file()
        assert case.telemetry_path.is_file()


def test_release_api_executes_notebook_smoke_and_writes_summary(
    tmp_path: Path,
    project_root: Path,
) -> None:
    result = euclid.execute_release_notebook_smoke(
        project_root=project_root,
        output_root=tmp_path / "notebook-smoke",
    )

    assert result.summary_path.is_file()
    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["probabilistic_case_ids"] == [
        "distribution",
        "event_probability",
        "interval",
        "quantile",
    ]
    assert payload["catalog_entries"] >= 1
    assert payload["publication_mode"] in {
        "candidate_publication",
        "abstention_only_publication",
    }


def test_clean_install_certification_resolves_relative_wheel_paths_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StopAfterInstall(RuntimeError):
        pass

    captured: dict[str, Path] = {}

    def _fake_run_command_with_logs(
        *,
        args: list[str],
        cwd: Path,
        env: dict[str, str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["-m", "build"]:
            outdir = Path(args[args.index("--outdir") + 1])
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "euclid-1.0.0-py3-none-any.whl").write_text(
                "placeholder wheel",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="",
                stderr="",
            )
        if args[1:3] == ["-m", "venv"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="",
                stderr="",
            )
        if args[1:4] == ["-m", "pip", "install"]:
            captured["cwd"] = cwd
            captured["find_links"] = Path(args[args.index("--find-links") + 1])
            captured["wheel_path"] = Path(args[-1])
            raise _StopAfterInstall("captured install invocation")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(release, "_run_command_with_logs", _fake_run_command_with_logs)
    monkeypatch.setattr(
        release,
        "_build_runtime_dependency_wheelhouse",
        lambda *, checkout_root, dist_dir, log_root: (),
    )

    with pytest.raises(_StopAfterInstall):
        release.run_clean_install_certification(
            project_root=PROJECT_ROOT,
            output_root=Path("relative-clean-install"),
            wheel_dir=Path("relative-wheels"),
        )

    assert captured["cwd"].name == "outside-repo"
    assert captured["find_links"].is_absolute()
    assert captured["wheel_path"].is_absolute()


def test_release_status_command_prints_current_and_target_versions() -> None:
    result = RUNNER.invoke(app, ["release", "status"])

    assert result.exit_code == 0
    assert "Euclid release status" in result.stdout
    assert "Current version: 1.0.0" in result.stdout
    assert "Release target: 1.0.0" in result.stdout
    assert "Target ready:" in result.stdout
    assert "Current release verdict:" in result.stdout
    assert "Full vision verdict:" in result.stdout
    assert "Shipped or releasable verdict:" in result.stdout


def test_release_validate_contracts_command_prints_catalog_summary() -> None:
    result = RUNNER.invoke(
        app,
        ["release", "validate-contracts", "--project-root", str(PROJECT_ROOT)],
    )

    assert result.exit_code == 0
    assert "Euclid contract catalog validation" in result.stdout
    assert "Schema count:" in result.stdout
    assert "Contract document count:" in result.stdout


def test_release_status_writes_row_complete_report(
    project_root: Path,
    tmp_path: Path,
) -> None:
    workspace_root = release._resolve_workspace_root(project_root)
    report_path = workspace_root / "build" / "reports" / "completion-report.json"
    if report_path.exists():
        report_path.unlink()

    status = euclid.get_release_status(
        project_root=project_root,
        benchmark_root=tmp_path / "benchmarks",
        notebook_output_root=tmp_path / "notebook-smoke",
    )

    assert status.project_root == project_root
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    matrix = yaml.safe_load(
        (project_root / "schemas/readiness/full-vision-matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert payload["authority_snapshot_id"] == "euclid-authority-2026-04-15-b"
    assert payload["command_contract_id"] == "euclid-certification-commands-v1"
    assert {row["row_id"] for row in matrix["rows"]} == {
        row["row_id"] for row in payload["capability_rows"]
    }
    assert {
        row["row_id"]
        for row in payload["capability_rows"]
        if row["row_id"].startswith("run_support_object:")
    }


def test_regression_policy_rejects_permanent_blocked_baseline(tmp_path: Path) -> None:
    report_path = tmp_path / "completion-report.json"
    report_path.write_text(
        json.dumps(
            {
                "report_id": "demo",
                "generated_at": "2026-04-15T00:00:00Z",
                "policy_verdicts": [
                    {
                        "policy_id": "current_release_v1",
                        "verdict": "blocked",
                        "reason_codes": [],
                        "evidence_refs": [],
                    },
                    {
                        "policy_id": "full_vision_v1",
                        "verdict": "blocked",
                        "reason_codes": [],
                        "evidence_refs": [],
                    },
                    {
                        "policy_id": "shipped_releasable_v1",
                        "verdict": "blocked",
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
                    "full_vision_completion": 0.7,
                    "current_gate_completion": 0.7,
                    "shipped_releasable_completion": 0.7,
                },
                "clean_install_certification": {
                    "surface_completion": 1.0,
                    "surfaces": [
                        {
                            "surface_id": surface_id,
                            "status": "passed",
                            "reason_codes": [],
                            "evidence_refs": ["artifact:clean-install.json"],
                        }
                        for surface_id in release._CLEAN_INSTALL_REQUIRED_SURFACE_IDS
                    ],
                },
                "capability_rows": [
                    {
                        "row_id": "evidence_lane:readiness_and_closure",
                        "status": "complete",
                        "reason_codes": [],
                        "evidence_refs": ["artifact:clean-install.json"],
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
                    "full_vision_completion": 0.5,
                    "current_gate_completion": 0.5,
                    "shipped_releasable_completion": 0.5,
                },
                "minimum_policy_verdicts": {
                    "current_release_v1": "ready",
                    "full_vision_v1": "ready",
                    "shipped_releasable_v1": "ready",
                },
                "required_clean_install_surface_ids": list(
                    release._CLEAN_INSTALL_REQUIRED_SURFACE_IDS
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

    result = release.verify_completion_report(
        project_root=PROJECT_ROOT,
        report_path=report_path,
        policy_path=policy_path,
    )

    assert result.passed is False
    assert any("fell below ready" in message for message in result.failure_messages)


def test_regression_policy_tracks_three_scores_independently(tmp_path: Path) -> None:
    report_path = tmp_path / "completion-report.json"
    report_path.write_text(
        json.dumps(
            {
                "report_id": "demo",
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
                    "full_vision_completion": 0.49,
                    "current_gate_completion": 0.85,
                    "shipped_releasable_completion": 0.8,
                },
                "clean_install_certification": {
                    "surface_completion": 1.0,
                    "surfaces": [
                        {
                            "surface_id": surface_id,
                            "status": "passed",
                            "reason_codes": [],
                            "evidence_refs": ["artifact:clean-install.json"],
                        }
                        for surface_id in release._CLEAN_INSTALL_REQUIRED_SURFACE_IDS
                    ],
                },
                "capability_rows": [
                    {
                        "row_id": "evidence_lane:readiness_and_closure",
                        "status": "complete",
                        "reason_codes": [],
                        "evidence_refs": ["artifact:clean-install.json"],
                        "required_evidence_classes": ["packaging_install"],
                        "available_evidence_classes": ["packaging_install"],
                        "non_closing_evidence_classes": [],
                        "evidence_bundle_ids": ["bundle-shipped"],
                    }
                ],
                "residual_risk_codes": [],
                "unresolved_blockers": [],
                "confidence": {"score": 0.75, "reason_codes": []},
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
                "policy_state": "near_closure",
                "minimum_completion_values": {
                    "full_vision_completion": 0.5,
                    "current_gate_completion": 0.8,
                    "shipped_releasable_completion": 0.75,
                },
                "minimum_policy_verdicts": {
                    "current_release_v1": "ready",
                    "full_vision_v1": "review_required",
                    "shipped_releasable_v1": "ready",
                },
                "required_clean_install_surface_ids": list(
                    release._CLEAN_INSTALL_REQUIRED_SURFACE_IDS
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

    result = release.verify_completion_report(
        project_root=PROJECT_ROOT,
        report_path=report_path,
        policy_path=policy_path,
    )

    assert result.passed is False
    assert any(
        "full_vision_completion regressed below 0.5" in message
        for message in result.failure_messages
    )
