from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import euclid
from euclid.benchmarks import load_benchmark_suite_manifest, load_benchmark_task_manifest
from euclid.benchmarks import runtime as benchmark_runtime
from euclid.benchmarks.manifests import BenchmarkSuiteSurfaceRequirement
from euclid.benchmarks.reporting import evaluate_benchmark_semantic_assertions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINT_TASK = PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
LEAKAGE_TASK = PROJECT_ROOT / "benchmarks/tasks/adversarial_honesty/leakage-trap-demo.yaml"
CURRENT_RELEASE_SUITE = PROJECT_ROOT / "benchmarks/suites/current-release.yaml"
FULL_VISION_SUITE = PROJECT_ROOT / "benchmarks/suites/full-vision.yaml"


def test_p13_current_and_full_vision_task_results_emit_semantic_assertions(
    tmp_path: Path,
) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=POINT_TASK,
        benchmark_root=tmp_path / "semantic-assertions",
        resume=False,
    )

    payload = json.loads(result.report_paths.task_result_path.read_text(encoding="utf-8"))
    assertions = payload["semantic_assertions"]

    assert assertions["overall_status"] == "passed"
    assert assertions["claim_scope"]["status"] == "passed"
    assert assertions["claim_scope"]["counts_as_claim_evidence"] is False
    assert assertions["metric_thresholds"]["status"] == "passed"
    assert assertions["engine_requirements"]["status"] == "passed"
    assert assertions["semantic_readiness_row_ids"]["status"] == "passed"
    assert "benchmark_task_semantics:planted_analytic_demo" in assertions[
        "semantic_readiness_row_ids"
    ]["observed"]


def test_p13_adversarial_benchmark_asserts_no_false_publication(
    tmp_path: Path,
) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=LEAKAGE_TASK,
        benchmark_root=tmp_path / "false-publication",
        resume=False,
    )

    payload = json.loads(result.report_paths.task_result_path.read_text(encoding="utf-8"))
    false_claim = payload["semantic_assertions"]["false_claim_expectations"]

    assert false_claim["status"] == "passed"
    assert false_claim["expected_safe_outcome"] == "abstain"
    assert false_claim["local_winner_submitter_id"] is None
    assert false_claim["local_winner_candidate_id"] is None


def test_p13_surface_fails_closed_on_overfit_benchmark_only_success(
    tmp_path: Path,
) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=POINT_TASK,
        benchmark_root=tmp_path / "overfit-success",
        resume=False,
    )
    payload = json.loads(result.report_paths.task_result_path.read_text(encoding="utf-8"))
    payload["semantic_assertions"]["overall_status"] = "failed"
    payload["semantic_assertions"]["metric_thresholds"]["status"] = "failed"
    result.report_paths.task_result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status = benchmark_runtime._surface_status(
        requirement=BenchmarkSuiteSurfaceRequirement(
            surface_id="retained_core_release",
            task_ids=("planted_analytic_demo",),
            replay_required=True,
        ),
        task_results=(result,),
    )

    assert status.benchmark_status == "failed"
    assert status.evidence["semantic_assertion_status"] == "failed"
    assert status.evidence["task_semantic_assertion_status"] == {
        "planted_analytic_demo": "failed"
    }
    assert status.evidence["claim_evidence_status"] == "not_claim_evidence"


def test_p13_replay_refs_are_deterministic_for_same_manifest_and_seed(
    tmp_path: Path,
) -> None:
    first = euclid.profile_benchmark_task(
        manifest_path=POINT_TASK,
        benchmark_root=tmp_path / "first",
        resume=False,
    )
    second = euclid.profile_benchmark_task(
        manifest_path=POINT_TASK,
        benchmark_root=tmp_path / "second",
        resume=False,
    )

    first_replays = {
        submitter_id: json.loads(path.read_text(encoding="utf-8"))
        for submitter_id, path in first.report_paths.replay_ref_paths.items()
    }
    second_replays = {
        submitter_id: json.loads(path.read_text(encoding="utf-8"))
        for submitter_id, path in second.report_paths.replay_ref_paths.items()
    }

    assert first_replays == second_replays


def test_p13_suite_manifests_have_explicit_benchmark_universe_semantics() -> None:
    for suite_path in (CURRENT_RELEASE_SUITE, FULL_VISION_SUITE):
        suite = load_benchmark_suite_manifest(suite_path)
        for task_path in suite.task_manifest_paths:
            task = load_benchmark_task_manifest(task_path)
            assertions = evaluate_benchmark_semantic_assertions(
                task_manifest=task,
                submitter_results=(),
                local_winner_submitter_id=None,
                local_winner_candidate_id=None,
            )

            assert task.metric_thresholds
            assert task.expected_claim_ceiling
            assert task.engine_requirements
            assert task.semantic_readiness_row_ids
            assert assertions["claim_scope"]["counts_as_claim_evidence"] is False
            assert assertions["engine_requirements"]["expected"]
            assert assertions["semantic_readiness_row_ids"]["expected"]


def test_p13_surface_status_detects_missing_semantic_assertions(
    tmp_path: Path,
) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=POINT_TASK,
        benchmark_root=tmp_path / "missing-semantic-assertions",
        resume=False,
    )
    payload = json.loads(result.report_paths.task_result_path.read_text(encoding="utf-8"))
    payload.pop("semantic_assertions")
    result.report_paths.task_result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status = benchmark_runtime._surface_status(
        requirement=BenchmarkSuiteSurfaceRequirement(
            surface_id="retained_core_release",
            task_ids=("planted_analytic_demo",),
            replay_required=True,
        ),
        task_results=(replace(result),),
    )

    assert status.benchmark_status == "failed"
    assert status.evidence["semantic_assertion_status"] == "missing"
