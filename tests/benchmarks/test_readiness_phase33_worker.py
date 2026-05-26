from __future__ import annotations

import json
from pathlib import Path

from euclid.benchmarks import (
    BenchmarkSuiteManifest,
    BenchmarkSuiteSurfaceRequirement,
    BenchmarkSubmitterResult,
    ProfiledBenchmarkSuiteResult,
    ProfiledBenchmarkTaskResult,
    load_benchmark_task_manifest,
    write_benchmark_task_report_artifacts,
)
from euclid.benchmarks import runtime as benchmark_runtime
from euclid.benchmarks.submitters import (
    ALGORITHMIC_SEARCH_SUBMITTER_ID,
    ANALYTIC_BACKEND_SUBMITTER_ID,
    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
    RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
)
from euclid.readiness import judge_benchmark_suite_readiness
from euclid.search.portfolio import PortfolioCandidateLedgerEntry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINT_TASK = PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"


def test_unverified_reproducibility_bundle_blocks_replay_readiness(
    tmp_path: Path,
) -> None:
    task_result = _profiled_task_result(tmp_path=tmp_path)
    _write_reproducibility_bundle_status(task_result, "failed")

    suite_result = _single_task_suite_result(
        tmp_path=tmp_path,
        task_result=task_result,
        surface_id="phase33_unverified_replay",
    )
    judgment = judge_benchmark_suite_readiness(
        judgment_id="phase33_unverified_replay_readiness",
        suite_result=suite_result,
    )

    assert (
        benchmark_runtime._task_replay_verification_status(task_result)
        == "failed"
    )
    assert judgment.final_verdict == "blocked"
    assert (
        "surface.phase33_unverified_replay_replay_unverified"
        in judgment.reason_codes
    )


def test_missing_replay_artifact_has_distinct_readiness_reason_code(
    tmp_path: Path,
) -> None:
    task_result = _profiled_task_result(tmp_path=tmp_path)
    first_replay_path = next(iter(task_result.report_paths.replay_ref_paths.values()))
    first_replay_path.unlink()

    suite_result = _single_task_suite_result(
        tmp_path=tmp_path,
        task_result=task_result,
        surface_id="phase33_missing_replay",
    )
    judgment = judge_benchmark_suite_readiness(
        judgment_id="phase33_missing_replay_readiness",
        suite_result=suite_result,
    )

    assert (
        benchmark_runtime._task_replay_verification_status(task_result)
        == "missing"
    )
    assert judgment.final_verdict == "blocked"
    assert (
        "surface.phase33_missing_replay_replay_artifact_missing"
        in judgment.reason_codes
    )
    assert (
        "surface.phase33_missing_replay_replay_unverified"
        not in judgment.reason_codes
    )


def test_present_replay_artifacts_without_status_are_unverified(
    tmp_path: Path,
) -> None:
    task_result = _profiled_task_result(tmp_path=tmp_path)
    _strip_one_replay_ref_verification_status(task_result)

    suite_result = _single_task_suite_result(
        tmp_path=tmp_path,
        task_result=task_result,
        surface_id="phase33_legacy_replay",
    )
    judgment = judge_benchmark_suite_readiness(
        judgment_id="phase33_legacy_replay_readiness",
        suite_result=suite_result,
    )

    assert (
        benchmark_runtime._task_replay_verification_status(task_result)
        == "unverified"
    )
    assert judgment.final_verdict == "blocked"
    assert (
        "surface.phase33_legacy_replay_replay_unverified"
        in judgment.reason_codes
    )
    assert (
        "surface.phase33_legacy_replay_replay_artifact_missing"
        not in judgment.reason_codes
    )


def test_replay_ref_candidate_mismatch_blocks_readiness(tmp_path: Path) -> None:
    task_result = _profiled_task_result(tmp_path=tmp_path)
    _tamper_selected_replay_candidate_hash(task_result)

    suite_result = _single_task_suite_result(
        tmp_path=tmp_path,
        task_result=task_result,
        surface_id="phase33_tampered_replay",
    )
    judgment = judge_benchmark_suite_readiness(
        judgment_id="phase33_tampered_replay_readiness",
        suite_result=suite_result,
    )

    assert benchmark_runtime._task_replay_verification_status(task_result) == "failed"
    assert judgment.final_verdict == "blocked"
    assert (
        "surface.phase33_tampered_replay_replay_unverified"
        in judgment.reason_codes
    )


def test_replay_ref_contract_mismatch_blocks_readiness(tmp_path: Path) -> None:
    task_result = _profiled_task_result(tmp_path=tmp_path)
    _tamper_selected_replay_contract_hooks(task_result)

    suite_result = _single_task_suite_result(
        tmp_path=tmp_path,
        task_result=task_result,
        surface_id="phase33_replay_contract_mismatch",
    )
    judgment = judge_benchmark_suite_readiness(
        judgment_id="phase33_replay_contract_mismatch_readiness",
        suite_result=suite_result,
    )

    assert benchmark_runtime._task_replay_verification_status(task_result) == "failed"
    assert judgment.final_verdict == "blocked"
    assert (
        "surface.phase33_replay_contract_mismatch_replay_unverified"
        in judgment.reason_codes
    )


def test_replay_ref_selected_candidate_must_appear_in_accepted_ledger(
    tmp_path: Path,
) -> None:
    task_result = _profiled_task_result(tmp_path=tmp_path)
    _tamper_selected_candidate_ledger(task_result)

    suite_result = _single_task_suite_result(
        tmp_path=tmp_path,
        task_result=task_result,
        surface_id="phase33_replay_ledger_mismatch",
    )
    judgment = judge_benchmark_suite_readiness(
        judgment_id="phase33_replay_ledger_mismatch_readiness",
        suite_result=suite_result,
    )

    assert benchmark_runtime._task_replay_verification_status(task_result) == "failed"
    assert judgment.final_verdict == "blocked"
    assert (
        "surface.phase33_replay_ledger_mismatch_replay_unverified"
        in judgment.reason_codes
    )


def test_submitter_replay_ref_digest_mismatch_blocks_readiness(
    tmp_path: Path,
) -> None:
    task_result = _profiled_task_result(tmp_path=tmp_path)
    _tamper_replay_ref_without_updating_submitter_digest(task_result)

    suite_result = _single_task_suite_result(
        tmp_path=tmp_path,
        task_result=task_result,
        surface_id="phase33_replay_digest_mismatch",
    )
    judgment = judge_benchmark_suite_readiness(
        judgment_id="phase33_replay_digest_mismatch_readiness",
        suite_result=suite_result,
    )

    assert benchmark_runtime._task_replay_verification_status(task_result) == "failed"
    assert judgment.final_verdict == "blocked"
    assert (
        "surface.phase33_replay_digest_mismatch_replay_unverified"
        in judgment.reason_codes
    )


def _single_task_suite_result(
    *,
    tmp_path: Path,
    task_result,
    surface_id: str,
) -> ProfiledBenchmarkSuiteResult:
    requirement = BenchmarkSuiteSurfaceRequirement(
        surface_id=surface_id,
        task_ids=(task_result.task_manifest.task_id,),
        replay_required=True,
    )
    suite_manifest = BenchmarkSuiteManifest(
        suite_id=surface_id,
        description="Phase 3.3 replay readiness worker fixture.",
        task_manifest_paths=(task_result.task_manifest.source_path,),
        required_tracks=(task_result.task_manifest.track_id,),
        surface_requirements=(requirement,),
        authority_snapshot_id=None,
        fixture_spec_id=None,
        source_path=tmp_path / f"{surface_id}.yaml",
    )
    surface_status = benchmark_runtime._surface_status(
        requirement=requirement,
        task_results=(task_result,),
    )
    summary_path = benchmark_runtime._write_suite_summary(
        suite_manifest=suite_manifest,
        benchmark_root=tmp_path / f"{surface_id}-summary",
        task_results=(task_result,),
        surface_statuses=(surface_status,),
    )
    return ProfiledBenchmarkSuiteResult(
        suite_manifest=suite_manifest,
        task_results=(task_result,),
        surface_statuses=(surface_status,),
        summary_path=summary_path,
    )


def _profiled_task_result(*, tmp_path: Path) -> ProfiledBenchmarkTaskResult:
    task_manifest = load_benchmark_task_manifest(POINT_TASK)
    submitter_results = _submitter_results(task_manifest)
    report_paths = write_benchmark_task_report_artifacts(
        benchmark_root=tmp_path / "benchmarks",
        task_manifest=task_manifest,
        submitter_results=submitter_results,
        task_status="completed",
        track_summary=benchmark_runtime._task_semantic_summary(task_manifest),
    )
    telemetry_path = (
        report_paths.task_result_path.parent / "performance-telemetry.json"
    )
    telemetry_path.write_text('{"profile_kind":"benchmark_task"}\n', encoding="utf-8")
    return ProfiledBenchmarkTaskResult(
        task_manifest=task_manifest,
        submitter_results=submitter_results,
        report_paths=report_paths,
        telemetry=None,
        telemetry_path=telemetry_path,
    )


def _submitter_results(task_manifest) -> tuple[BenchmarkSubmitterResult, ...]:
    protocol_contract = {
        "task_id": task_manifest.task_id,
        "track_id": task_manifest.track_id,
        "replay_policy": dict(task_manifest.frozen_protocol.replay_policy),
    }
    selected_metrics = {
        "practical_significance_margin": 0.02,
        "mean_absolute_error": 0.01,
        "total_code_bits": 10.0,
        "description_gain_bits": 2.0,
        "structure_code_bits": 4.0,
        "canonical_byte_length": 32,
    }
    common_budget = {
        "declared_candidate_limit": 1,
        "declared_wall_clock_seconds": 1,
        "attempted_candidate_count": 1,
        "accepted_candidate_count": 1,
        "rejected_candidate_count": 0,
        "omitted_candidate_count": 0,
    }
    analytic = BenchmarkSubmitterResult(
        submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        submitter_class="decomposition",
        task_id=task_manifest.task_id,
        track_id=task_manifest.track_id,
        status="selected",
        protocol_contract=protocol_contract,
        budget_consumption=common_budget,
        candidate_ledger=(
            PortfolioCandidateLedgerEntry(
                candidate_id="phase33_candidate",
                primitive_family="analytic",
                ledger_status="accepted",
                canonical_rank=0,
                attempted_rank=0,
                candidate_hash="sha256:phase33",
            ),
        ),
        selected_candidate_id="phase33_candidate",
        selected_candidate_hash="sha256:phase33",
        selected_candidate_metrics=selected_metrics,
        replay_contract={
            "search_plan_id": f"{task_manifest.task_id}__analytic_backend__search_plan",
            "candidate_id": "phase33_candidate",
            "candidate_hash": "sha256:phase33",
            "replay_hooks": [
                {
                    "hook_name": "phase33_fixture_hook",
                    "hook_ref": "phase33_fixture_replay_hook",
                }
            ],
            "replay_verification_status": "verified",
        },
    )
    recursive = _abstained_submitter(
        task_manifest=task_manifest,
        submitter_id=RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
        submitter_class="sparse_library",
        protocol_contract=protocol_contract,
        budget_consumption=common_budget,
    )
    algorithmic = _abstained_submitter(
        task_manifest=task_manifest,
        submitter_id=ALGORITHMIC_SEARCH_SUBMITTER_ID,
        submitter_class="bounded_grammar",
        protocol_contract=protocol_contract,
        budget_consumption=common_budget,
    )
    portfolio = BenchmarkSubmitterResult(
        submitter_id=PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
        submitter_class="portfolio",
        task_id=task_manifest.task_id,
        track_id=task_manifest.track_id,
        status="selected",
        protocol_contract=protocol_contract,
        budget_consumption=common_budget,
        selected_candidate_id="phase33_candidate",
        selected_candidate_hash="sha256:phase33",
        selected_candidate_metrics=selected_metrics,
        replay_contract={
            "selection_rule": "phase33_fixture",
            "selected_submitter_id": ANALYTIC_BACKEND_SUBMITTER_ID,
            "selected_candidate_id": "phase33_candidate",
            "selected_candidate_hash": "sha256:phase33",
            "replay_verification_status": "verified",
        },
        child_results=(analytic, recursive, algorithmic),
        compared_finalists=(
            {
                "submitter_id": ANALYTIC_BACKEND_SUBMITTER_ID,
                "candidate_id": "phase33_candidate",
                "total_code_bits": 10.0,
            },
        ),
    )
    return (analytic, recursive, algorithmic, portfolio)


def _abstained_submitter(
    *,
    task_manifest,
    submitter_id: str,
    submitter_class: str,
    protocol_contract,
    budget_consumption,
) -> BenchmarkSubmitterResult:
    return BenchmarkSubmitterResult(
        submitter_id=submitter_id,
        submitter_class=submitter_class,
        task_id=task_manifest.task_id,
        track_id=task_manifest.track_id,
        status="abstained",
        protocol_contract=protocol_contract,
        budget_consumption=budget_consumption,
        replay_contract={
            "search_plan_id": f"{task_manifest.task_id}__{submitter_id}__search_plan",
            "replay_verification_status": "verified",
        },
        abstention_reason="phase33_fixture_nonwinner",
    )


def _write_reproducibility_bundle_status(task_result, status: str) -> None:
    payload = json.loads(
        task_result.report_paths.task_result_path.read_text(encoding="utf-8")
    )
    payload["reproducibility_bundle_manifest"] = {
        "body": {
            "bundle_id": f"{task_result.task_manifest.task_id}_bundle",
            "replay_verification_status": status,
        },
        "schema_name": "reproducibility_bundle_manifest@1.0.0",
    }
    task_result.report_paths.task_result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _strip_one_replay_ref_verification_status(task_result) -> None:
    replay_path = next(iter(task_result.report_paths.replay_ref_paths.values()))
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    payload.pop("replay_verification_status", None)
    replay_contract = payload.get("replay_contract")
    if isinstance(replay_contract, dict):
        replay_contract.pop("replay_verification_status", None)
    replay_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    task_payload = json.loads(
        task_result.report_paths.task_result_path.read_text(encoding="utf-8")
    )
    track_summary = task_payload.get("track_summary")
    if isinstance(track_summary, dict):
        track_summary.pop("replay_verification_status", None)
    task_result.report_paths.task_result_path.write_text(
        json.dumps(task_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tamper_selected_replay_candidate_hash(task_result) -> None:
    replay_path = task_result.report_paths.replay_ref_paths[
        ANALYTIC_BACKEND_SUBMITTER_ID
    ]
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    replay_contract = payload["replay_contract"]
    replay_contract["candidate_hash"] = "sha256:tampered"
    replay_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tamper_selected_replay_contract_hooks(task_result) -> None:
    replay_path = task_result.report_paths.replay_ref_paths[
        ANALYTIC_BACKEND_SUBMITTER_ID
    ]
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    payload["replay_contract"]["replay_hooks"] = [
        {"hook_name": "tampered_hook", "hook_ref": "artifact:tampered"}
    ]
    replay_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tamper_selected_candidate_ledger(task_result) -> None:
    submitter_path = task_result.report_paths.submitter_result_paths[
        ANALYTIC_BACKEND_SUBMITTER_ID
    ]
    payload = json.loads(submitter_path.read_text(encoding="utf-8"))
    payload["candidate_ledger"][0]["candidate_hash"] = "sha256:ledger-tampered"
    submitter_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tamper_replay_ref_without_updating_submitter_digest(task_result) -> None:
    replay_path = task_result.report_paths.replay_ref_paths[
        ANALYTIC_BACKEND_SUBMITTER_ID
    ]
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    payload["track_id"] = "tampered_track"
    replay_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
