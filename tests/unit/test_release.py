from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version
from typer.testing import CliRunner

import euclid
import euclid.release as release
from euclid.cli import app
from euclid.readiness import ReadinessGateResult

RUNNER = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _minimal_release_workspace(tmp_path: Path) -> Path:
    workspace_root = tmp_path / "repo"
    (workspace_root / "src" / "euclid").mkdir(parents=True)
    (workspace_root / "pyproject.toml").write_text(
        "[build-system]\nrequires = []\nbuild-backend = 'setuptools.build_meta'\n",
        encoding="utf-8",
    )
    (workspace_root / "src" / "euclid" / "__init__.py").write_text(
        "__version__ = '1.0.0'\n",
        encoding="utf-8",
    )
    return workspace_root


def test_release_source_digest_ignores_build_egg_info_metadata(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    before = release._release_source_digest_ref(workspace_root)

    egg_info = workspace_root / "src" / "euclid.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("generated metadata\n", encoding="utf-8")
    (egg_info / "SOURCES.txt").write_text("generated sources\n", encoding="utf-8")

    assert release._release_source_digest_ref(workspace_root) == before
    assert not any(
        "egg-info" in path.name
        for path in release._release_source_digest_files(workspace_root)
    )


def test_release_source_digest_tracks_ci_workflow_and_dependency_lock(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    workflow_path = workspace_root / ".github" / "workflows" / "ci.yml"
    workflow_path.parent.mkdir(parents=True)
    workflow_path.write_text("jobs: {}\n", encoding="utf-8")
    uv_lock_path = workspace_root / "uv.lock"
    uv_lock_path.write_text("version = 1\n", encoding="utf-8")
    package_manifest_path = workspace_root / "package.json"
    package_manifest_path.write_text('{"scripts": {"test": "vitest"}}\n', encoding="utf-8")
    package_lock_path = workspace_root / "package-lock.json"
    package_lock_path.write_text('{"lockfileVersion": 3}\n', encoding="utf-8")

    digest_before = release._release_source_digest_ref(workspace_root)
    workflow_path.write_text("jobs:\n  release: {}\n", encoding="utf-8")
    digest_after_workflow_change = release._release_source_digest_ref(workspace_root)
    uv_lock_path.write_text("version = 1\nrevision = 2\n", encoding="utf-8")
    digest_after_uv_lock_change = release._release_source_digest_ref(workspace_root)
    package_manifest_path.write_text(
        '{"scripts": {"test": "vitest --run"}}\n',
        encoding="utf-8",
    )
    digest_after_package_manifest_change = release._release_source_digest_ref(
        workspace_root
    )
    package_lock_path.write_text(
        '{"lockfileVersion": 3, "packages": {}}\n',
        encoding="utf-8",
    )
    digest_after_package_lock_change = release._release_source_digest_ref(
        workspace_root
    )

    assert digest_after_workflow_change != digest_before
    assert digest_after_uv_lock_change != digest_after_workflow_change
    assert digest_after_package_manifest_change != digest_after_uv_lock_change
    assert digest_after_package_lock_change != digest_after_package_manifest_change
    digest_files = {
        path.relative_to(workspace_root).as_posix()
        for path in release._release_source_digest_files(workspace_root)
    }
    assert ".github/workflows/ci.yml" in digest_files
    assert "uv.lock" in digest_files
    assert "package.json" in digest_files
    assert "package-lock.json" in digest_files


def _valid_clean_install_report(workspace_root: Path) -> dict[str, object]:
    output_root = workspace_root / "build" / "clean-install"
    wheelhouse = output_root / "dist"
    log_root = output_root / "logs"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    wheel_path = wheelhouse / "euclid-1.0.0-py3-none-any.whl"
    wheel_path.write_text("wheel payload\n", encoding="utf-8")
    wheel_digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    canonical_report_path = release._clean_install_report_path(workspace_root)
    canonical_report_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_report_path.write_text("{}\n", encoding="utf-8")
    shipped_bundle = release._bundle_by_scope_id()["shipped_releasable"]
    for surface_id in release._CLEAN_INSTALL_REQUIRED_SURFACE_IDS:
        (log_root / f"{surface_id}.stdout.log").write_text(
            f"{surface_id} passed\n",
            encoding="utf-8",
        )

    return {
        "report_id": "euclid_clean_install_certification_v1",
        "evidence_bundle_id": shipped_bundle["evidence_bundle_id"],
        "scope_id": shipped_bundle["scope_id"],
        "authority_snapshot_id": shipped_bundle["authority_snapshot_id"],
        "command_contract_id": shipped_bundle["command_contract_id"],
        "closure_map_id": shipped_bundle["closure_map_id"],
        "traceability_id": shipped_bundle["traceability_id"],
        "fixture_spec_id": shipped_bundle["fixture_spec_id"],
        "producer_command_id": shipped_bundle["producer_command_id"],
        "canonical_report_path": str(canonical_report_path),
        "source_tree_digest_at_build": release._release_source_digest_ref(
            workspace_root
        ),
        "source_tree_digest_or_wheel_digest": f"wheel_digest:{wheel_digest}",
        "wheel_path": str(wheel_path),
        "wheel_digest": wheel_digest,
        "output_root": str(output_root),
        "runtime_dependency_wheelhouse": str(wheelhouse),
        "runtime_dependency_wheel_count": 0,
        "input_manifest_digests": [
            {
                str(wheelhouse): (
                    f"runtime_directory_digest:{release._directory_digest(wheelhouse)}"
                )
            }
        ],
        "surfaces": [
            {
                "surface_id": surface_id,
                "status": "passed",
                "reason_codes": [],
                "evidence_refs": [
                    f"artifact:{log_root / f'{surface_id}.stdout.log'}"
                ],
            }
            for surface_id in release._CLEAN_INSTALL_REQUIRED_SURFACE_IDS
        ],
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
    assert isinstance(status.target_ready, bool)
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


def test_release_status_can_reuse_clean_install_notebook_smoke_summary(
    tmp_path: Path,
    project_root: Path,
) -> None:
    summary_path = tmp_path / "notebook" / "notebook-smoke-summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "probabilistic_case_ids": [
                    "distribution",
                    "event_probability",
                    "interval",
                    "quantile",
                ],
                "catalog_entries": 2,
                "publication_mode": "candidate_publication",
            }
        ),
        encoding="utf-8",
    )
    clean_install_report = {
        "surfaces": [
            {
                "surface_id": "packaged_notebook_smoke",
                "status": "passed",
                "evidence_refs": [f"artifact:{summary_path}"],
            }
        ]
    }

    result = release._notebook_smoke_from_clean_install_report(
        project_root=project_root,
        clean_install_report=clean_install_report,
    )

    assert result is not None
    assert result.summary_path == summary_path
    assert result.probabilistic_case_ids == (
        "distribution",
        "event_probability",
        "interval",
        "quantile",
    )
    assert result.catalog_entries == 2


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
    workspace_root = _minimal_release_workspace(tmp_path)

    with pytest.raises(_StopAfterInstall):
        release.run_clean_install_certification(
            project_root=workspace_root,
            output_root=Path("repo/build/relative-clean-install"),
            wheel_dir=Path("repo/build/relative-clean-install/relative-wheels"),
        )

    assert captured["cwd"].name == "outside-repo"
    assert captured["find_links"].is_absolute()
    assert captured["wheel_path"].is_absolute()


def test_clean_install_certification_rejects_ambiguous_project_wheels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                "current wheel\n",
                encoding="utf-8",
            )
            (outdir / "euclid-9.9.9-py3-none-any.whl").write_text(
                "stale wheel\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=args, returncode=0)
        raise AssertionError(f"unexpected command: {args}")

    workspace_root = _minimal_release_workspace(tmp_path)
    monkeypatch.setattr(release, "_run_command_with_logs", _fake_run_command_with_logs)
    monkeypatch.setattr(
        release,
        "_build_runtime_dependency_wheelhouse",
        lambda *, checkout_root, dist_dir, log_root: pytest.fail(
            "ambiguous project wheels should fail before dependency wheelhouse build"
        ),
    )

    with pytest.raises(ValueError, match="exactly one project wheel"):
        release.run_clean_install_certification(project_root=workspace_root)


def test_clean_install_release_status_surface_passes_when_status_is_truthfully_blocked(
    tmp_path: Path,
) -> None:
    result = subprocess.CompletedProcess(
        args=("python", "-m", "euclid", "release", "status"),
        returncode=0,
        stdout="\n".join(
            [
                "Euclid release status",
                "Target ready: no",
                "Current release verdict: blocked (current_release_v1)",
                "Current release reason codes: current_release_suite_evidence_missing",
                "Full vision verdict: blocked (full_vision_v1)",
                "Full vision reason codes: full_vision_suite_evidence_missing",
                "Shipped or releasable verdict: blocked (shipped_releasable_v1)",
                "Shipped or releasable reason codes: clean_install_certification_incomplete",
            ]
        ),
        stderr="",
    )

    surface = release._build_clean_install_surface_result(
        surface_id="release_status",
        result=result,
        stdout_path=tmp_path / "release-status.stdout.log",
        stderr_path=tmp_path / "release-status.stderr.log",
    )

    assert surface.status == "passed"
    assert surface.reason_codes == ()


@pytest.mark.parametrize(
    "stdout_lines",
    [
        [
            "Euclid release status",
            "Target ready: no",
            "Current release verdict: blocked (current_release_v1)",
            "Current release reason codes: current_release_suite_evidence_missing",
            "Full vision reason codes: full_vision_suite_evidence_missing",
            "Shipped or releasable verdict: blocked (shipped_releasable_v1)",
            "Shipped or releasable reason codes: clean_install_certification_incomplete",
        ],
        [
            "Euclid release status",
            "Target ready: no",
            "Current release verdict: blocked (current_release_v1)",
            "Current release reason codes: current_release_suite_evidence_missing",
            "Full vision verdict: blocked",
            "Full vision reason codes: full_vision_suite_evidence_missing",
            "Shipped or releasable verdict: blocked (shipped_releasable_v1)",
            "Shipped or releasable reason codes: clean_install_certification_incomplete",
        ],
    ],
)
def test_clean_install_release_status_surface_fails_when_policy_verdict_lines_are_missing_or_malformed(
    stdout_lines: list[str],
    tmp_path: Path,
) -> None:
    result = subprocess.CompletedProcess(
        args=("python", "-m", "euclid", "release", "status"),
        returncode=0,
        stdout="\n".join(stdout_lines),
        stderr="",
    )

    surface = release._build_clean_install_surface_result(
        surface_id="release_status",
        result=result,
        stdout_path=tmp_path / "release-status.stdout.log",
        stderr_path=tmp_path / "release-status.stderr.log",
    )

    assert surface.status == "failed"
    assert surface.reason_codes == (
        "clean_install.release_status_policy_verdict_malformed",
    )


def test_pytest_summary_parser_extracts_passed_and_skipped_counts() -> None:
    counts, summary_line, parsed = release._parse_pytest_summary_counts(
        "...\n========= 1260 passed, 2 skipped in 3.21s =========\n"
    )

    assert parsed is True
    assert summary_line == "1260 passed, 2 skipped in 3.21s"
    assert counts["passed"] == 1260
    assert counts["skipped"] == 2


def test_repo_test_matrix_fails_closed_when_summary_is_unparseable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_subprocess_run(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=kwargs.get("args", ()),
            returncode=0,
            stdout="tests completed successfully but summary was truncated\n",
            stderr="",
        )

    report_path = tmp_path / "repo-test-matrix.json"
    monkeypatch.setattr(release.subprocess, "run", _fake_subprocess_run)

    result = release.run_repo_test_matrix(
        project_root=PROJECT_ROOT,
        report_path=report_path,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.passed is False
    assert payload["passed"] is False
    assert payload["summary_counts_parsed"] is False
    assert payload["summary_parse_status"] == "missing_or_unparsed_summary"


def test_repo_matrix_report_reader_rejects_tampered_pass_boolean() -> None:
    report = {
        "passed": True,
        "summary_counts_parsed": True,
        "counts": {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
        },
    }

    assert release._repo_test_matrix_report_passed(report) is False


def test_release_evidence_freshness_rejects_stale_source_digest(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "repo"
    (workspace_root / "src" / "euclid").mkdir(parents=True)
    (workspace_root / "pyproject.toml").write_text(
        "[build-system]\nrequires = []\nbuild-backend = 'setuptools.build_meta'\n",
        encoding="utf-8",
    )
    (workspace_root / "src" / "euclid" / "__init__.py").write_text(
        "__version__ = '1.0.0'\n",
        encoding="utf-8",
    )
    clean_output_root = workspace_root / "build" / "clean-install"
    clean_output_root.mkdir(parents=True)

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        repo_test_matrix_report={
            "producer_command_id": "repo_test_matrix",
            "source_tree_digest_or_wheel_digest": "repo_checkout_digest:stale",
        },
        clean_install_report={
            "producer_command_id": "clean_install_certification",
            "source_tree_digest_at_build": "repo_checkout_digest:stale",
            "source_tree_digest_or_wheel_digest": "wheel_digest:test",
            "output_root": str(clean_output_root),
            "surfaces": [],
        },
    )

    assert "repo_test_matrix_source_digest_mismatch" in failures
    assert "clean_install_source_digest_mismatch" in failures


def test_release_evidence_freshness_rejects_clean_install_missing_wheel_fields(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    clean_output_root = workspace_root / "build" / "clean-install"
    clean_output_root.mkdir(parents=True)

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        clean_install_report={
            "report_id": "euclid_clean_install_certification_v1",
            "producer_command_id": "clean_install_certification",
            "scope_id": "shipped_releasable",
            "source_tree_digest_at_build": release._release_source_digest_ref(
                workspace_root
            ),
            "output_root": str(clean_output_root),
            "surfaces": [],
        },
    )

    assert "clean_install_wheel_path_missing" in failures
    assert "clean_install_wheel_digest_missing" in failures
    assert "clean_install_wheel_digest_ref_missing" in failures
    assert "clean_install_runtime_dependency_wheelhouse_missing" in failures
    assert "clean_install_input_manifest_digests_missing" in failures


def test_release_evidence_freshness_rejects_clean_install_wheel_digest_mismatch(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    clean_output_root = workspace_root / "build" / "clean-install"
    wheelhouse = clean_output_root / "dist"
    wheelhouse.mkdir(parents=True)
    wheel_path = wheelhouse / "euclid-1.0.0-py3-none-any.whl"
    wheel_path.write_text("wheel payload\n", encoding="utf-8")

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        clean_install_report={
            "report_id": "euclid_clean_install_certification_v1",
            "producer_command_id": "clean_install_certification",
            "scope_id": "shipped_releasable",
            "source_tree_digest_at_build": release._release_source_digest_ref(
                workspace_root
            ),
            "source_tree_digest_or_wheel_digest": "wheel_digest:stale",
            "wheel_path": str(wheel_path),
            "wheel_digest": "stale",
            "runtime_dependency_wheelhouse": str(wheelhouse),
            "input_manifest_digests": [
                {str(wheelhouse): f"runtime_directory_digest:{release._directory_digest(wheelhouse)}"}
            ],
            "output_root": str(clean_output_root),
            "surfaces": [],
        },
    )

    assert "clean_install_wheel_digest_mismatch" in failures
    assert "clean_install_wheel_digest_ref_mismatch" in failures


def test_release_evidence_freshness_rejects_clean_install_surface_artifact_refs(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    clean_output_root = workspace_root / "build" / "clean-install"
    wheelhouse = clean_output_root / "dist"
    wheelhouse.mkdir(parents=True)
    wheel_path = wheelhouse / "euclid-1.0.0-py3-none-any.whl"
    wheel_path.write_text("wheel payload\n", encoding="utf-8")
    external_artifact = tmp_path / "external.log"
    external_artifact.write_text("outside build\n", encoding="utf-8")
    wheel_digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        clean_install_report={
            "report_id": "euclid_clean_install_certification_v1",
            "producer_command_id": "clean_install_certification",
            "scope_id": "shipped_releasable",
            "source_tree_digest_at_build": release._release_source_digest_ref(
                workspace_root
            ),
            "source_tree_digest_or_wheel_digest": f"wheel_digest:{wheel_digest}",
            "wheel_path": str(wheel_path),
            "wheel_digest": wheel_digest,
            "runtime_dependency_wheelhouse": str(wheelhouse),
            "input_manifest_digests": [
                {str(wheelhouse): f"runtime_directory_digest:{release._directory_digest(wheelhouse)}"}
            ],
            "output_root": str(clean_output_root),
            "surfaces": [
                {
                    "surface_id": "release_status",
                    "status": "passed",
                    "reason_codes": [],
                    "evidence_refs": [
                        f"artifact:{clean_output_root / 'missing.log'}",
                        f"artifact:{external_artifact}",
                    ],
                }
            ],
        },
    )

    assert "clean_install_surface_release_status_artifact_missing" in failures
    assert (
        "clean_install_surface_release_status_artifact_outside_workspace_build"
        in failures
    )


def test_release_evidence_freshness_rejects_clean_install_paths_outside_output_root(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    report = _valid_clean_install_report(workspace_root)
    output_root = Path(str(report["output_root"]))
    borrowed_root = workspace_root / "build" / "borrowed-clean-install"
    borrowed_wheelhouse = borrowed_root / "dist"
    borrowed_wheelhouse.mkdir(parents=True)
    borrowed_wheel = borrowed_wheelhouse / "euclid-1.0.0-py3-none-any.whl"
    borrowed_wheel.write_text("borrowed wheel\n", encoding="utf-8")
    borrowed_artifact = borrowed_root / "logs" / "release_status.stdout.log"
    borrowed_artifact.parent.mkdir(parents=True)
    borrowed_artifact.write_text("borrowed artifact\n", encoding="utf-8")
    borrowed_digest = hashlib.sha256(borrowed_wheel.read_bytes()).hexdigest()
    report["wheel_path"] = str(borrowed_wheel)
    report["wheel_digest"] = borrowed_digest
    report["source_tree_digest_or_wheel_digest"] = f"wheel_digest:{borrowed_digest}"
    report["runtime_dependency_wheelhouse"] = str(borrowed_wheelhouse)
    report["input_manifest_digests"] = [
        {
            str(borrowed_wheelhouse): (
                "runtime_directory_digest:"
                f"{release._directory_digest(borrowed_wheelhouse)}"
            )
        }
    ]
    report["surfaces"][0]["evidence_refs"] = [f"artifact:{borrowed_artifact}"]

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        clean_install_report=report,
    )

    assert output_root != borrowed_root
    assert "clean_install_wheel_path_outside_output_root" in failures
    assert (
        "clean_install_runtime_dependency_wheelhouse_outside_output_root"
        in failures
    )
    assert (
        "clean_install_surface_release_status_artifact_outside_output_root"
        in failures
    )


def test_release_evidence_freshness_rejects_clean_install_required_surface_gaps(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    report = _valid_clean_install_report(workspace_root)
    report["surfaces"] = [
        surface
        for surface in report["surfaces"]
        if surface["surface_id"] != "release_status"
    ]
    report["surfaces"][0]["status"] = "failed"
    report["surfaces"][0]["reason_codes"] = ["clean_install.operator_run_failed"]

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        clean_install_report=report,
    )

    assert "clean_install_surface_release_status_missing" in failures
    assert "clean_install_surface_operator_run_not_passed" in failures
    assert "clean_install_surface_operator_run_reason_codes_present" in failures


def test_release_evidence_freshness_rejects_clean_install_bundle_metadata_mismatch(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    report = _valid_clean_install_report(workspace_root)
    report["evidence_bundle_id"] = "wrong-bundle"
    report["authority_snapshot_id"] = "wrong-authority"
    report["command_contract_id"] = "wrong-command-contract"
    report["canonical_report_path"] = str(
        workspace_root / "build" / "reports" / "wrong-clean-install.json"
    )

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        clean_install_report=report,
    )

    assert "clean_install_evidence_bundle_id_mismatch" in failures
    assert "clean_install_authority_snapshot_id_mismatch" in failures
    assert "clean_install_command_contract_id_mismatch" in failures
    assert "clean_install_canonical_report_path_mismatch" in failures


def test_release_evidence_freshness_rejects_clean_install_wheel_count_mismatch(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    report = _valid_clean_install_report(workspace_root)
    report["runtime_dependency_wheel_count"] = 99

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        clean_install_report=report,
    )

    assert "clean_install_runtime_dependency_wheel_count_mismatch" in failures


def test_release_evidence_freshness_rejects_clean_install_multiple_project_wheels(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    report = _valid_clean_install_report(workspace_root)
    wheelhouse = Path(str(report["runtime_dependency_wheelhouse"]))
    extra_project_wheel = wheelhouse / "euclid-9.9.9-py3-none-any.whl"
    extra_project_wheel.write_text("stale project wheel\n", encoding="utf-8")
    report["runtime_dependency_wheel_count"] = 1
    report["input_manifest_digests"] = [
        {
            str(wheelhouse): (
                f"runtime_directory_digest:{release._directory_digest(wheelhouse)}"
            )
        }
    ]

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        clean_install_report=report,
    )

    assert "clean_install_project_wheel_count_mismatch" in failures


@pytest.mark.parametrize("relative_output_root", ("build", "build/reports"))
def test_release_evidence_freshness_rejects_clean_install_reserved_output_root(
    tmp_path: Path,
    relative_output_root: str,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    report = _valid_clean_install_report(workspace_root)
    reserved_output_root = workspace_root / relative_output_root
    reserved_output_root.mkdir(parents=True, exist_ok=True)
    report["output_root"] = str(reserved_output_root)

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        clean_install_report=report,
    )

    assert "clean_install_output_root_not_dedicated" in failures


def test_clean_install_certification_accepts_command_contract_output_root(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    contract_output_root = workspace_root / "build" / "certification" / "clean_install"

    assert (
        release._clean_install_work_root_dedication_failure(
            work_root=contract_output_root,
            workspace_root=workspace_root,
        )
        is None
    )


def test_clean_install_certification_rejects_output_root_outside_build_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    external_output_root = tmp_path / "outside-clean-install"
    external_output_root.mkdir()
    sentinel = external_output_root / "sentinel.txt"
    sentinel.write_text("must not be deleted\n", encoding="utf-8")
    monkeypatch.setattr(
        release,
        "_declared_build_toolchain",
        lambda _root: ("setuptools.build_meta", ()),
    )
    monkeypatch.setattr(release, "_declared_runtime_requirements", lambda _root: ())
    monkeypatch.setattr(
        release,
        "_run_command_with_logs",
        lambda **_kwargs: pytest.fail("clean install command should not run"),
    )

    with pytest.raises(ValueError, match="output root"):
        release.run_clean_install_certification(
            project_root=workspace_root,
            output_root=external_output_root,
        )

    assert sentinel.read_text(encoding="utf-8") == "must not be deleted\n"


@pytest.mark.parametrize("relative_output_root", ("build", "build/reports"))
def test_clean_install_certification_rejects_reserved_output_root_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_output_root: str,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    reserved_output_root = workspace_root / relative_output_root
    reserved_output_root.mkdir(parents=True, exist_ok=True)
    sentinel = reserved_output_root / "sentinel.txt"
    sentinel.write_text("must not be deleted\n", encoding="utf-8")
    monkeypatch.setattr(
        release,
        "_run_command_with_logs",
        lambda **_kwargs: pytest.fail("clean install command should not run"),
    )

    with pytest.raises(ValueError, match="dedicated clean-install output root"):
        release.run_clean_install_certification(
            project_root=workspace_root,
            output_root=reserved_output_root,
        )

    assert sentinel.read_text(encoding="utf-8") == "must not be deleted\n"


def test_release_evidence_freshness_rejects_external_suite_summary_path(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "repo"
    (workspace_root / "src" / "euclid").mkdir(parents=True)
    (workspace_root / "pyproject.toml").write_text(
        "[build-system]\nrequires = []\nbuild-backend = 'setuptools.build_meta'\n",
        encoding="utf-8",
    )
    (workspace_root / "src" / "euclid" / "__init__.py").write_text(
        "__version__ = '1.0.0'\n",
        encoding="utf-8",
    )
    external_summary = tmp_path / "pytest-tmp" / "summary.json"
    external_summary.parent.mkdir(parents=True)
    external_summary.write_text("{}", encoding="utf-8")

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        current_suite_report={
            "producer_command_id": "current_release_suite",
            "scope_id": "current_release",
            "source_tree_digest_or_wheel_digest": release._release_source_digest_ref(
                workspace_root
            ),
            "summary_path": str(external_summary),
        },
    )

    assert "current_release_suite_summary_path_outside_workspace_build" in failures


def test_release_evidence_freshness_rejects_suite_summary_digest_mismatch(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "repo"
    (workspace_root / "src" / "euclid").mkdir(parents=True)
    (workspace_root / "pyproject.toml").write_text(
        "[build-system]\nrequires = []\nbuild-backend = 'setuptools.build_meta'\n",
        encoding="utf-8",
    )
    (workspace_root / "src" / "euclid" / "__init__.py").write_text(
        "__version__ = '1.0.0'\n",
        encoding="utf-8",
    )
    summary_path = workspace_root / "build" / "suite" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text('{"status": "passed"}\n', encoding="utf-8")

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        current_suite_report={
            "producer_command_id": "current_release_suite",
            "scope_id": "current_release",
            "source_tree_digest_or_wheel_digest": release._release_source_digest_ref(
                workspace_root
            ),
            "summary_path": str(summary_path),
            "summary_sha256": "runtime_sha256:stale",
        },
    )

    assert "current_release_suite_summary_sha256_mismatch" in failures


def test_release_evidence_freshness_rejects_suite_summary_directory(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "repo"
    (workspace_root / "src" / "euclid").mkdir(parents=True)
    (workspace_root / "pyproject.toml").write_text(
        "[build-system]\nrequires = []\nbuild-backend = 'setuptools.build_meta'\n",
        encoding="utf-8",
    )
    (workspace_root / "src" / "euclid" / "__init__.py").write_text(
        "__version__ = '1.0.0'\n",
        encoding="utf-8",
    )
    summary_dir = workspace_root / "build" / "suite-directory"
    summary_dir.mkdir(parents=True)

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        current_suite_report={
            "producer_command_id": "current_release_suite",
            "scope_id": "current_release",
            "source_tree_digest_or_wheel_digest": release._release_source_digest_ref(
                workspace_root
            ),
            "summary_path": str(summary_dir),
            "summary_sha256": "runtime_sha256:any",
        },
    )

    assert "current_release_suite_summary_path_not_file" in failures


def test_release_evidence_freshness_rejects_operator_replay_run_id_mismatch(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    run_summary_path = workspace_root / "build" / "operator" / "run-summary.json"
    run_summary_path.parent.mkdir(parents=True)
    run_summary_path.write_text('{"run_id": "full-vision-run"}\n', encoding="utf-8")
    output_root = workspace_root / "build" / "operator"
    run_report_path = (
        workspace_root / "build" / "reports" / "full_vision_operator_run_evidence.json"
    )
    run_report_path.parent.mkdir(parents=True)
    run_report_path.write_text(
        '{"report_id": "operator_run_evidence_v1"}\n',
        encoding="utf-8",
    )
    source_digest = release._release_source_digest_ref(workspace_root)
    run_summary_sha256 = "runtime_sha256:" + hashlib.sha256(
        run_summary_path.read_bytes()
    ).hexdigest()
    run_report_sha256 = "runtime_sha256:" + hashlib.sha256(
        run_report_path.read_bytes()
    ).hexdigest()

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        operator_run_report={
            "command_id": "full_vision_operator_run",
            "run_id": "full-vision-run",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(run_summary_path),
            "run_summary_sha256": run_summary_sha256,
            "output_root": str(output_root),
            "run_result_ref": {"schema_name": "run_result", "object_id": "run-1"},
            "bundle_ref": {"schema_name": "bundle", "object_id": "bundle-1"},
        },
        operator_replay_report={
            "command_id": "full_vision_operator_replay",
            "run_id": "different-run",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(run_summary_path),
            "run_summary_sha256": run_summary_sha256,
            "output_root": str(output_root),
            "operator_run_evidence_report_path": str(run_report_path),
            "operator_run_evidence_report_sha256": run_report_sha256,
            "run_result_ref": {"schema_name": "run_result", "object_id": "run-1"},
            "bundle_ref": {"schema_name": "bundle", "object_id": "bundle-1"},
        },
    )

    assert "full_vision_operator_replay_run_id_mismatch" in failures


def test_release_evidence_freshness_rejects_operator_run_summary_identity_mismatch(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    run_summary_path = workspace_root / "build" / "operator" / "run-summary.json"
    run_summary_path.parent.mkdir(parents=True)
    run_summary_path.write_text(
        json.dumps({"request_id": "wrong-run"}) + "\n",
        encoding="utf-8",
    )
    output_root = workspace_root / "build" / "operator"
    run_report_path = (
        workspace_root / "build" / "reports" / "full_vision_operator_run_evidence.json"
    )
    run_report_path.parent.mkdir(parents=True)
    run_report_path.write_text(
        '{"report_id": "operator_run_evidence_v1"}\n',
        encoding="utf-8",
    )
    source_digest = release._release_source_digest_ref(workspace_root)
    run_summary_sha256 = "runtime_sha256:" + hashlib.sha256(
        run_summary_path.read_bytes()
    ).hexdigest()
    run_report_sha256 = "runtime_sha256:" + hashlib.sha256(
        run_report_path.read_bytes()
    ).hexdigest()
    run_result_ref = {"schema_name": "run_result", "object_id": "run-1"}
    bundle_ref = {"schema_name": "bundle", "object_id": "bundle-1"}

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        operator_run_report={
            "report_id": "operator_run_evidence_v1",
            "command_id": "full_vision_operator_run",
            "scope_id": "full_vision",
            "run_id": "full-vision-run",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(run_summary_path),
            "run_summary_sha256": run_summary_sha256,
            "run_id_binding": {
                "request_id": "full-vision-run",
                "run_result_object_id": "run-1",
                "run_summary_request_id": "wrong-run",
            },
            "output_root": str(output_root),
            "run_result_ref": run_result_ref,
            "bundle_ref": bundle_ref,
        },
        operator_replay_report={
            "report_id": "operator_replay_evidence_v1",
            "command_id": "full_vision_operator_replay",
            "scope_id": "full_vision",
            "run_id": "full-vision-run",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(run_summary_path),
            "run_summary_sha256": run_summary_sha256,
            "run_id_binding": {
                "requested_run_id": "full-vision-run",
                "run_result_object_id": "run-1",
                "run_summary_request_id": "wrong-run",
            },
            "output_root": str(output_root),
            "operator_run_evidence_report_path": str(run_report_path),
            "operator_run_evidence_report_sha256": run_report_sha256,
            "replay_verification_status": "verified",
            "replay_result_sha256": "runtime_sha256:any",
            "run_result_ref": run_result_ref,
            "bundle_ref": bundle_ref,
        },
    )

    assert "full_vision_operator_run_run_summary_request_id_mismatch" in failures
    assert "full_vision_operator_replay_run_summary_request_id_mismatch" in failures


def test_release_evidence_freshness_rejects_operator_replay_without_run_report_digest(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    run_summary_path = workspace_root / "build" / "operator" / "run-summary.json"
    run_summary_path.parent.mkdir(parents=True)
    run_summary_path.write_text('{"run_id": "full-vision-run"}\n', encoding="utf-8")
    output_root = workspace_root / "build" / "operator"
    source_digest = release._release_source_digest_ref(workspace_root)
    run_summary_sha256 = "runtime_sha256:" + hashlib.sha256(
        run_summary_path.read_bytes()
    ).hexdigest()

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        operator_run_report={
            "command_id": "full_vision_operator_run",
            "run_id": "full-vision-run",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(run_summary_path),
            "run_summary_sha256": run_summary_sha256,
            "output_root": str(output_root),
        },
        operator_replay_report={
            "command_id": "full_vision_operator_replay",
            "run_id": "full-vision-run",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(run_summary_path),
            "run_summary_sha256": run_summary_sha256,
            "output_root": str(output_root),
        },
    )

    assert (
        "full_vision_operator_replay_operator_run_evidence_report_path_missing"
        in failures
    )
    assert (
        "full_vision_operator_replay_operator_run_evidence_report_sha256_missing"
        in failures
    )


def test_release_evidence_freshness_rejects_operator_replay_not_verified(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    run_summary_path = workspace_root / "build" / "operator" / "run-summary.json"
    run_summary_path.parent.mkdir(parents=True)
    run_summary_path.write_text('{"run_id": "full-vision-run"}\n', encoding="utf-8")
    output_root = workspace_root / "build" / "operator"
    run_report_path = (
        workspace_root / "build" / "reports" / "full_vision_operator_run_evidence.json"
    )
    run_report_path.parent.mkdir(parents=True)
    run_report_path.write_text(
        '{"report_id": "operator_run_evidence_v1"}\n',
        encoding="utf-8",
    )
    source_digest = release._release_source_digest_ref(workspace_root)
    run_summary_sha256 = "runtime_sha256:" + hashlib.sha256(
        run_summary_path.read_bytes()
    ).hexdigest()
    run_report_sha256 = "runtime_sha256:" + hashlib.sha256(
        run_report_path.read_bytes()
    ).hexdigest()
    run_result_ref = {"schema_name": "run_result", "object_id": "run-1"}
    bundle_ref = {"schema_name": "bundle", "object_id": "bundle-1"}

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        operator_run_report={
            "command_id": "full_vision_operator_run",
            "run_id": "full-vision-run",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(run_summary_path),
            "run_summary_sha256": run_summary_sha256,
            "output_root": str(output_root),
            "run_result_ref": run_result_ref,
            "bundle_ref": bundle_ref,
        },
        operator_replay_report={
            "command_id": "full_vision_operator_replay",
            "run_id": "full-vision-run",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(run_summary_path),
            "run_summary_sha256": run_summary_sha256,
            "output_root": str(output_root),
            "operator_run_evidence_report_path": str(run_report_path),
            "operator_run_evidence_report_sha256": run_report_sha256,
            "replay_verification_status": "failed",
            "run_result_ref": run_result_ref,
            "bundle_ref": bundle_ref,
        },
    )

    assert "full_vision_operator_replay_not_verified" in failures


def test_release_evidence_freshness_rejects_operator_scope_and_report_mismatch(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    run_summary_path = workspace_root / "build" / "operator" / "run-summary.json"
    run_summary_path.parent.mkdir(parents=True)
    run_summary_path.write_text('{"run_id": "full-vision-run"}\n', encoding="utf-8")
    output_root = workspace_root / "build" / "operator"
    run_report_path = (
        workspace_root / "build" / "reports" / "full_vision_operator_run_evidence.json"
    )
    run_report_path.parent.mkdir(parents=True)
    run_report_path.write_text(
        '{"report_id": "operator_run_evidence_v1"}\n',
        encoding="utf-8",
    )
    source_digest = release._release_source_digest_ref(workspace_root)
    run_summary_sha256 = "runtime_sha256:" + hashlib.sha256(
        run_summary_path.read_bytes()
    ).hexdigest()
    run_report_sha256 = "runtime_sha256:" + hashlib.sha256(
        run_report_path.read_bytes()
    ).hexdigest()
    run_result_ref = {"schema_name": "run_result", "object_id": "run-1"}
    bundle_ref = {"schema_name": "bundle", "object_id": "bundle-1"}

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        operator_run_report={
            "report_id": "wrong_operator_run_report",
            "command_id": "full_vision_operator_run",
            "scope_id": "current_release",
            "run_id": "full-vision-run",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(run_summary_path),
            "run_summary_sha256": run_summary_sha256,
            "output_root": str(output_root),
            "run_result_ref": run_result_ref,
            "bundle_ref": bundle_ref,
        },
        operator_replay_report={
            "report_id": "wrong_operator_replay_report",
            "command_id": "full_vision_operator_replay",
            "scope_id": "current_release",
            "run_id": "full-vision-run",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(run_summary_path),
            "run_summary_sha256": run_summary_sha256,
            "output_root": str(output_root),
            "operator_run_evidence_report_path": str(run_report_path),
            "operator_run_evidence_report_sha256": run_report_sha256,
            "replay_verification_status": "verified",
            "run_result_ref": run_result_ref,
            "bundle_ref": bundle_ref,
        },
    )

    assert "full_vision_operator_run_report_id_mismatch" in failures
    assert "full_vision_operator_run_scope_mismatch" in failures
    assert "full_vision_operator_replay_report_id_mismatch" in failures
    assert "full_vision_operator_replay_scope_mismatch" in failures


def test_release_evidence_freshness_rejects_operator_output_root_file(
    tmp_path: Path,
) -> None:
    workspace_root = _minimal_release_workspace(tmp_path)
    run_summary_path = workspace_root / "build" / "operator" / "run-summary.json"
    run_summary_path.parent.mkdir(parents=True)
    run_summary_path.write_text('{"run_id": "full-vision-run"}\n', encoding="utf-8")
    output_root = workspace_root / "build" / "operator-output-file"
    output_root.write_text("not a directory\n", encoding="utf-8")
    run_report_path = (
        workspace_root / "build" / "reports" / "full_vision_operator_run_evidence.json"
    )
    run_report_path.parent.mkdir(parents=True)
    run_report_path.write_text(
        '{"report_id": "operator_run_evidence_v1"}\n',
        encoding="utf-8",
    )
    source_digest = release._release_source_digest_ref(workspace_root)
    run_summary_sha256 = "runtime_sha256:" + hashlib.sha256(
        run_summary_path.read_bytes()
    ).hexdigest()
    run_report_sha256 = "runtime_sha256:" + hashlib.sha256(
        run_report_path.read_bytes()
    ).hexdigest()
    run_result_ref = {"schema_name": "run_result", "object_id": "run-1"}
    bundle_ref = {"schema_name": "bundle", "object_id": "bundle-1"}

    failures = release._release_evidence_freshness_failures(
        workspace_root=workspace_root,
        operator_run_report={
            "report_id": "operator_run_evidence_v1",
            "command_id": "full_vision_operator_run",
            "scope_id": "full_vision",
            "run_id": "full-vision-run",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(run_summary_path),
            "run_summary_sha256": run_summary_sha256,
            "output_root": str(output_root),
            "run_result_ref": run_result_ref,
            "bundle_ref": bundle_ref,
        },
        operator_replay_report={
            "report_id": "operator_replay_evidence_v1",
            "command_id": "full_vision_operator_replay",
            "scope_id": "full_vision",
            "run_id": "full-vision-run",
            "source_tree_digest_or_wheel_digest": source_digest,
            "run_summary_path": str(run_summary_path),
            "run_summary_sha256": run_summary_sha256,
            "output_root": str(output_root),
            "operator_run_evidence_report_path": str(run_report_path),
            "operator_run_evidence_report_sha256": run_report_sha256,
            "replay_verification_status": "verified",
            "run_result_ref": run_result_ref,
            "bundle_ref": bundle_ref,
        },
    )

    assert "full_vision_operator_run_output_root_not_directory" in failures
    assert "full_vision_operator_replay_output_root_not_directory" in failures


def test_release_evidence_freshness_gate_blocks_release_readiness() -> None:
    suite_judgment = release.judge_readiness(
        judgment_id="suite",
        gate_results=(
            ReadinessGateResult(
                gate_id="release.evidence_freshness",
                status="failed",
                required=True,
                summary="stale evidence",
                evidence={"reason_codes": ["repo_test_matrix_source_digest_mismatch"]},
            ),
        ),
    )

    current_judgment = release._judge_current_release_gate_readiness(
        suite_readiness_judgment=suite_judgment
    )
    full_judgment = release._judge_full_vision_gate_readiness(
        suite_readiness_judgment=suite_judgment
    )

    assert (
        "release.evidence_freshness_repo_test_matrix_source_digest_mismatch"
        in current_judgment.reason_codes
    )
    assert (
        "release.evidence_freshness_repo_test_matrix_source_digest_mismatch"
        in full_judgment.reason_codes
    )


def test_completion_values_are_capped_when_policy_verdict_is_blocked() -> None:
    rows_by_id = {
        "row:a": release._CompletionLedgerRow(
            row_id="row:a",
            status="complete",
            required_closing_classes=("benchmark_semantic",),
            available_evidence_classes=("benchmark_semantic",),
            non_closing_evidence_classes=(),
            reason_codes=(),
            evidence_refs=(),
            evidence_bundle_ids=(),
            proof_status=None,
        )
    }
    policy = {"required_row_ids": ["row:a"]}
    blocked = release.judge_readiness(
        judgment_id="blocked_policy",
        gate_results=(
            ReadinessGateResult(
                gate_id="release.evidence_freshness",
                status="failed",
                required=True,
                evidence={"reason_codes": ["repo_test_matrix_source_digest_mismatch"]},
            ),
        ),
    )

    values = release._build_completion_values(
        rows_by_id=rows_by_id,
        current_policy=policy,
        full_policy=policy,
        shipped_policy=policy,
        clean_install_report={
            "surfaces": [
                {"surface_id": surface_id, "status": "passed"}
                for surface_id in release._CLEAN_INSTALL_REQUIRED_SURFACE_IDS
            ]
        },
        policy_judgments={
            "current_release_v1": blocked,
            "full_vision_v1": blocked,
            "shipped_releasable_v1": blocked,
        },
    )

    assert values["current_gate_completion"] < 1.0
    assert values["full_vision_completion"] < 1.0
    assert values["shipped_releasable_completion"] < 1.0


def test_policy_blockers_are_unresolved_completion_blockers() -> None:
    blocked = release.judge_readiness(
        judgment_id="current_release_v1",
        gate_results=(
            ReadinessGateResult(
                gate_id="release.evidence_freshness",
                status="failed",
                required=True,
                evidence={"reason_codes": ["repo_test_matrix_source_digest_mismatch"]},
            ),
        ),
    )

    blockers = release._policy_blocker_payloads(
        {"current_release_v1": blocked}
    )

    assert blockers == [
        {
            "capability_row_id": "policy:current_release_v1",
            "proof_status": "policy_blocked",
            "reason_codes": [
                "release.evidence_freshness_repo_test_matrix_source_digest_mismatch"
            ],
            "evidence_refs": ["gate:release.evidence_freshness"],
        }
    ]


def test_write_suite_evidence_bundle_skips_external_benchmark_root(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "repo"
    summary_path = tmp_path / "pytest-root" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text("{}", encoding="utf-8")
    source_path = workspace_root / "benchmarks" / "suites" / "current-release.yaml"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("suite_id: current_release\n", encoding="utf-8")
    task_path = workspace_root / "benchmarks" / "tasks" / "task.yaml"
    task_path.parent.mkdir(parents=True)
    task_path.write_text("task_id: demo\n", encoding="utf-8")
    suite_result = type(
        "SuiteResult",
        (),
        {
            "suite_manifest": type(
                "SuiteManifest",
                (),
                {
                    "suite_id": "current_release",
                    "source_path": source_path,
                    "task_manifest_paths": (task_path,),
                },
            )(),
            "summary_path": summary_path,
            "surface_statuses": (),
        },
    )()

    path = release.write_suite_evidence_bundle(
        suite_result=suite_result,
        workspace_root=workspace_root,
    )

    assert path is None
    assert not (
        workspace_root / "build" / "reports" / "current_release_suite_evidence.json"
    ).exists()


def test_write_suite_evidence_bundle_records_summary_sha256(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "repo"
    summary_path = workspace_root / "build" / "benchmark" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text('{"status": "passed"}\n', encoding="utf-8")
    source_path = workspace_root / "benchmarks" / "suites" / "current-release.yaml"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("suite_id: current_release\n", encoding="utf-8")
    task_path = workspace_root / "benchmarks" / "tasks" / "task.yaml"
    task_path.parent.mkdir(parents=True)
    task_path.write_text("task_id: demo\n", encoding="utf-8")
    suite_result = type(
        "SuiteResult",
        (),
        {
            "suite_manifest": type(
                "SuiteManifest",
                (),
                {
                    "suite_id": "current_release",
                    "source_path": source_path,
                    "task_manifest_paths": (task_path,),
                },
            )(),
            "summary_path": summary_path,
            "surface_statuses": (),
        },
    )()

    path = release.write_suite_evidence_bundle(
        suite_result=suite_result,
        workspace_root=workspace_root,
    )

    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary_path"] == str(summary_path)
    assert payload["summary_sha256"] == (
        "runtime_sha256:" + hashlib.sha256(summary_path.read_bytes()).hexdigest()
    )


def test_verify_completion_report_with_temp_report_does_not_overwrite_canonical(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "repo"
    canonical_verify_path = (
        project_root / "build" / "reports" / "verify-completion.json"
    )
    canonical_verify_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_verify_path.write_text(
        json.dumps({"report_id": "canonical-marker"}) + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "completion-report.json"
    report_path.write_text(
        json.dumps(
            {
                "report_id": "demo",
                "generated_at": "2026-04-15T00:00:00Z",
                "policy_verdicts": [],
                "authority_snapshot_id": "euclid-authority-2026-04-15-b",
                "command_contract_id": "euclid-certification-commands-v1",
                "closure_map_id": "euclid-closure-map-2026-04-15-v1",
                "traceability_id": "euclid-traceability-2026-04-15-v1",
                "fixture_spec_id": "euclid-certification-fixtures-v1",
                "scope_evidence_bundles": [],
                "completion_values": {
                    "full_vision_completion": 0.0,
                    "current_gate_completion": 0.0,
                    "shipped_releasable_completion": 0.0,
                },
                "clean_install_certification": {
                    "surface_completion": 0.0,
                    "surfaces": [],
                },
                "capability_rows": [],
                "enhancement_phase_gates": [],
                "residual_risk_codes": [],
                "unresolved_blockers": [],
                "confidence": {"score": 0.0, "reason_codes": []},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "policy_state": "test",
                "minimum_completion_values": {},
                "minimum_policy_verdicts": {},
                "required_clean_install_surface_ids": [],
                "required_row_evidence_classes": [],
            }
        ),
        encoding="utf-8",
    )

    release.verify_completion_report(
        project_root=project_root,
        report_path=report_path,
        policy_path=policy_path,
    )

    assert json.loads(canonical_verify_path.read_text(encoding="utf-8")) == {
        "report_id": "canonical-marker"
    }


def test_runtime_dependency_closure_preserves_selected_extras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                "dependencies = ['parent>=1']",
            ]
        ),
        encoding="utf-8",
    )

    class _FakeDistribution:
        def __init__(
            self,
            name: str,
            requires: tuple[str, ...] = (),
        ) -> None:
            self.metadata = {"Name": name}
            self.requires = requires

    distributions = {
        "parent": _FakeDistribution(
            "parent",
            ("child[format-nongpl]>=1",),
        ),
        "child": _FakeDistribution(
            "child",
            (
                "base-dep>=1",
                'format-dep>=1; extra == "format-nongpl"',
                'ignored-extra>=1; extra == "other"',
            ),
        ),
        "base-dep": _FakeDistribution("base-dep"),
        "format-dep": _FakeDistribution("format-dep"),
        "ignored-extra": _FakeDistribution("ignored-extra"),
    }

    def _fake_distribution(distribution_name: str) -> _FakeDistribution:
        return distributions[distribution_name.lower().replace("_", "-")]

    monkeypatch.setattr(release.importlib_metadata, "distribution", _fake_distribution)

    names = set(release._runtime_dependency_distribution_names(tmp_path))

    assert {"parent", "child", "base-dep", "format-dep"} <= names
    assert "ignored-extra" not in names


def test_declared_runtime_requirements_match_certified_wheelhouse_versions() -> None:
    mismatches = []
    declared_requirements = release._declared_runtime_requirements(  # noqa: SLF001
        PROJECT_ROOT
    )
    for raw_requirement in declared_requirements:
        requirement = Requirement(raw_requirement)
        if not release._requirement_marker_applies(  # noqa: SLF001
            requirement,
            extra_context="",
        ):
            continue
        distribution = release.importlib_metadata.distribution(requirement.name)
        installed_version = Version(distribution.version)
        if installed_version not in requirement.specifier:
            mismatches.append(
                (
                    requirement.name,
                    str(requirement.specifier),
                    str(installed_version),
                )
            )

    assert not mismatches, (
        "declared runtime requirements must be satisfiable from the certified "
        f"local wheelhouse; mismatches: {mismatches}"
    )


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


def test_release_status_command_prints_enhancement_phase_gate_summary() -> None:
    result = RUNNER.invoke(app, ["release", "status"])

    assert result.exit_code == 0
    assert "Enhancement phase gates:" in result.stdout
    assert "- P00: complete" in result.stdout
    assert "- P01: complete" in result.stdout
    assert "- P02: complete" in result.stdout
    assert "- P03: complete" in result.stdout
    assert "- P04: complete" in result.stdout
    assert "- P14: complete" in result.stdout
    assert "- P15: complete" in result.stdout
    assert "- P16: complete" in result.stdout


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


def test_verify_completion_fails_closed_on_unresolved_blockers_in_transition_policy(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "completion-report.json"
    report_path.write_text(
        json.dumps(
            {
                "report_id": "demo",
                "generated_at": "2026-04-21T00:00:00Z",
                "policy_verdicts": [
                    {
                        "policy_id": "current_release_v1",
                        "verdict": "blocked",
                        "reason_codes": ["phase_blocked"],
                        "evidence_refs": [],
                    },
                    {
                        "policy_id": "full_vision_v1",
                        "verdict": "blocked",
                        "reason_codes": ["phase_blocked"],
                        "evidence_refs": [],
                    },
                    {
                        "policy_id": "shipped_releasable_v1",
                        "verdict": "blocked",
                        "reason_codes": ["phase_blocked"],
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
                    "full_vision_completion": 0.9,
                    "current_gate_completion": 0.9,
                    "shipped_releasable_completion": 0.9,
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
                        "status": "partial",
                        "reason_codes": ["missing_phase_gate_manifest"],
                        "evidence_refs": ["gate:tests/gates/P02.yaml"],
                        "required_evidence_classes": ["packaging_install"],
                        "available_evidence_classes": ["packaging_install"],
                        "non_closing_evidence_classes": [],
                        "evidence_bundle_ids": ["bundle-shipped"],
                    }
                ],
                "residual_risk_codes": ["missing_phase_gate_manifest"],
                "unresolved_blockers": [
                    {
                        "capability_row_id": "evidence_lane:readiness_and_closure",
                        "proof_status": "missing_proof",
                        "reason_codes": ["missing_phase_gate_manifest"],
                        "evidence_refs": ["gate:tests/gates/P02.yaml"],
                    }
                ],
                "confidence": {"score": 0.9, "reason_codes": ["missing_proof_present"]},
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
                "policy_state": "transition",
                "policy_states": {
                    "transition": {
                        "minimum_completion_values": {
                            "full_vision_completion": 0.25,
                            "current_gate_completion": 0.2,
                            "shipped_releasable_completion": 0.3,
                        },
                        "minimum_policy_verdicts": {
                            "current_release_v1": "blocked",
                            "full_vision_v1": "blocked",
                            "shipped_releasable_v1": "blocked",
                        },
                    }
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
    assert any("unresolved blockers" in message for message in result.failure_messages)
