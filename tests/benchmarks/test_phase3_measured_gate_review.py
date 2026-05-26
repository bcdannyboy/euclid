from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from euclid.benchmarks import runtime as benchmark_runtime
from euclid.benchmarks.manifests import (
    BenchmarkSuiteManifest,
    BenchmarkSuiteSurfaceRequirement,
)
from euclid.benchmarks.reporting import (
    evaluate_benchmark_semantic_assertions,
    write_benchmark_task_report_artifacts,
)
from euclid.benchmarks.runtime import (
    ProfiledBenchmarkSuiteResult,
    ProfiledBenchmarkTaskResult,
)
from euclid.benchmarks.submitters import (
    ANALYTIC_BACKEND_SUBMITTER_ID,
    BenchmarkSubmitterResult,
)
from euclid.benchmarks import load_benchmark_task_manifest
from euclid.readiness import judge_benchmark_suite_readiness

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINT_TASK = PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"


def test_unmeasured_required_threshold_cannot_pass_with_semantic_row_only() -> None:
    manifest = _phase3_manifest(
        metric_thresholds={
            "phase3_recovery_floor": {
                "metric_id": "planted_law_recovery_rate",
                "comparator": ">=",
                "threshold": 1.0,
            }
        }
    )

    assert manifest.semantic_readiness_row_ids

    assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=(
            _selected_submitter_result(
                manifest,
                metrics={"total_code_bits": 12.0},
                candidate_id="semantic-only-candidate",
            ),
        ),
        local_winner_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        local_winner_candidate_id="semantic-only-candidate",
    )

    assert assertions["metric_thresholds"]["status"] == "failed"
    assert assertions["overall_status"] == "failed"
    missing_row = assertions["metric_thresholds"]["assertions"][0]
    assert {
        "threshold_id": "phase3_recovery_floor",
        "metric_id": "planted_law_recovery_rate",
        "source_submitter_id": ANALYTIC_BACKEND_SUBMITTER_ID,
        "source_candidate_id": "semantic-only-candidate",
        "status": "failed",
        "reason_code": "missing_observed_metric",
    }.items() <= missing_row.items()


def test_missing_measurement_rows_identify_metric_threshold_and_submitter() -> None:
    manifest = _phase3_manifest(
        metric_thresholds={
            "phase3_measured_accuracy_floor": {
                "metric_id": "measured_accuracy",
                "comparator": ">=",
                "threshold": 0.5,
            },
            "phase3_recovery_floor": {
                "metric_id": "planted_law_recovery_rate",
                "comparator": ">=",
                "threshold": 1.0,
            },
        }
    )

    assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=(
            _selected_submitter_result(
                manifest,
                metrics={"measured_accuracy": 0.75},
                candidate_id="partial-measurement-candidate",
            ),
        ),
        local_winner_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        local_winner_candidate_id="partial-measurement-candidate",
    )
    rows_by_threshold = {
        row["threshold_id"]: row
        for row in assertions["metric_thresholds"]["assertions"]
    }

    missing_row = rows_by_threshold["phase3_recovery_floor"]
    assert {
        "threshold_id": "phase3_recovery_floor",
        "metric_id": "planted_law_recovery_rate",
        "source_submitter_id": ANALYTIC_BACKEND_SUBMITTER_ID,
        "source_candidate_id": "partial-measurement-candidate",
        "status": "failed",
        "reason_code": "missing_observed_metric",
    }.items() <= missing_row.items()


def test_runtime_does_not_synthesize_score_law_metric_from_threshold() -> None:
    manifest = _phase3_manifest(
        metric_thresholds={
            "phase3_score_law_floor": {
                "metric_id": "mean_absolute_error",
                "comparator": "<=",
                "threshold": 0.15,
            }
        }
    )
    submitter_result = _selected_submitter_result(
        manifest,
        metrics={"total_code_bits": 12.0},
        candidate_id="score-law-without-measurement",
    )

    augmented_results = benchmark_runtime._submitter_results_with_threshold_metrics(
        task_manifest=manifest,
        submitter_results=(submitter_result,),
    )

    assert "mean_absolute_error" not in (
        augmented_results[0].selected_candidate_metrics or {}
    )
    assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=augmented_results,
        local_winner_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        local_winner_candidate_id="score-law-without-measurement",
    )
    row = assertions["metric_thresholds"]["assertions"][0]
    assert assertions["metric_thresholds"]["status"] == "failed"
    assert {
        "threshold_id": "phase3_score_law_floor",
        "metric_id": "mean_absolute_error",
        "source_submitter_id": ANALYTIC_BACKEND_SUBMITTER_ID,
        "source_candidate_id": "score-law-without-measurement",
        "observed_value": None,
        "status": "failed",
        "reason_code": "missing_observed_metric",
    }.items() <= row.items()


def test_runtime_does_not_use_description_gain_as_practical_margin() -> None:
    manifest = _phase3_manifest(
        metric_thresholds={
            "phase3_margin_floor": {
                "metric_id": "practical_significance_margin",
                "comparator": ">=",
                "threshold": 0.01,
            }
        }
    )
    submitter_result = _selected_submitter_result(
        manifest,
        metrics={"description_gain_bits": 123.0},
        candidate_id="mdl-only-candidate",
    )

    augmented_results = benchmark_runtime._submitter_results_with_threshold_metrics(
        task_manifest=manifest,
        submitter_results=(submitter_result,),
    )

    assert "practical_significance_margin" not in (
        augmented_results[0].selected_candidate_metrics or {}
    )
    assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=augmented_results,
        local_winner_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        local_winner_candidate_id="mdl-only-candidate",
    )
    row = assertions["metric_thresholds"]["assertions"][0]
    assert assertions["metric_thresholds"]["status"] == "failed"
    assert {
        "threshold_id": "phase3_margin_floor",
        "metric_id": "practical_significance_margin",
        "observed_value": None,
        "status": "failed",
        "reason_code": "missing_observed_metric",
    }.items() <= row.items()


def test_declared_safe_abstention_requires_submitter_evidence() -> None:
    manifest = replace(
        _phase3_manifest(
            metric_thresholds={
                "phase3_margin_floor": {
                    "metric_id": "practical_significance_margin",
                    "comparator": ">=",
                    "threshold": 0.01,
                }
            }
        ),
        expected_safe_outcome="abstain",
    )

    assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=(),
        local_winner_submitter_id=None,
        local_winner_candidate_id=None,
    )

    row = assertions["metric_thresholds"]["assertions"][0]
    assert assertions["metric_thresholds"]["status"] == "failed"
    assert assertions["overall_status"] == "failed"
    assert {
        "threshold_id": "phase3_margin_floor",
        "metric_id": "practical_significance_margin",
        "observed_value": None,
        "status": "failed",
        "reason_code": "safe_abstention_evidence_missing",
    }.items() <= row.items()

    assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=(
            BenchmarkSubmitterResult(
                submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
                submitter_class="decomposition",
                task_id=manifest.task_id,
                track_id=manifest.track_id,
                status="abstained",
                protocol_contract={},
                budget_consumption={},
                abstention_reason="verified_no_publishable_candidate",
                safe_abstention_evidence={
                    "status": "verified",
                    "evidence_type": "falsification_gate",
                    "reason_code": "verified_no_publishable_candidate",
                    "support": [{"candidate_id": "phase3_nonwinner"}],
                },
            ),
        ),
        local_winner_submitter_id=None,
        local_winner_candidate_id=None,
    )

    row = assertions["metric_thresholds"]["assertions"][0]
    assert assertions["metric_thresholds"]["status"] == "passed"
    assert assertions["overall_status"] == "passed"
    assert {
        "threshold_id": "phase3_margin_floor",
        "metric_id": "practical_significance_margin",
        "observed_value": None,
        "status": "passed",
        "reason_code": "not_applicable_safe_abstention",
    }.items() <= row.items()


def test_replay_file_presence_without_verified_replay_blocks_readiness(
    tmp_path: Path,
) -> None:
    manifest = _phase3_manifest(
        metric_thresholds={
            "phase3_measured_accuracy_floor": {
                "metric_id": "measured_accuracy",
                "comparator": ">=",
                "threshold": 0.5,
            }
        }
    )
    paths = write_benchmark_task_report_artifacts(
        benchmark_root=tmp_path / "benchmarks",
        task_manifest=manifest,
        submitter_results=(
            _selected_submitter_result(
                manifest,
                metrics={"measured_accuracy": 0.75},
                replay_contract={
                    "candidate_id": "unverified-replay-candidate",
                    "replay_verification_status": "failed",
                    "failure_reason_codes": ["artifact_hash_mismatch"],
                },
                candidate_id="unverified-replay-candidate",
            ),
        ),
        task_status="completed",
    )
    task_payload = json.loads(paths.task_result_path.read_text(encoding="utf-8"))
    assert task_payload["semantic_assertions"]["overall_status"] == "passed"
    assert all(path.is_file() for path in paths.replay_ref_paths.values())

    task_result = ProfiledBenchmarkTaskResult(
        task_manifest=manifest,
        submitter_results=(),
        report_paths=paths,
        telemetry=None,
        telemetry_path=tmp_path / "telemetry.json",
    )
    surface_requirement = BenchmarkSuiteSurfaceRequirement(
        surface_id="phase3_measured_surface",
        task_ids=(manifest.task_id,),
        replay_required=True,
    )
    surface_status = benchmark_runtime._surface_status(
        requirement=surface_requirement,
        task_results=(task_result,),
    )
    suite_manifest = BenchmarkSuiteManifest(
        suite_id="phase3_measured_suite",
        description="Phase 3 measured gate review guard suite.",
        task_manifest_paths=(manifest.source_path,),
        required_tracks=(manifest.track_id,),
        surface_requirements=(surface_requirement,),
        authority_snapshot_id="phase3-review",
        fixture_spec_id="phase3-review",
        source_path=tmp_path / "phase3-suite.yaml",
    )
    summary_path = benchmark_runtime._write_suite_summary(
        suite_manifest=suite_manifest,
        benchmark_root=tmp_path / "benchmarks",
        task_results=(task_result,),
        surface_statuses=(surface_status,),
    )
    suite_result = ProfiledBenchmarkSuiteResult(
        suite_manifest=suite_manifest,
        task_results=(task_result,),
        surface_statuses=(surface_status,),
        summary_path=summary_path,
    )

    judgment = judge_benchmark_suite_readiness(
        judgment_id="phase3_replay_presence_is_not_readiness",
        suite_result=suite_result,
    )

    assert judgment.final_verdict == "blocked"
    surface_gate = next(
        gate
        for gate in judgment.gate_results
        if gate.gate_id == "surface.phase3_measured_surface"
    )
    assert surface_gate.status == "failed"
    assert surface_gate.evidence["replay_verification"] == "failed"
    assert surface_gate.evidence["replay_reason_codes"] == {
        manifest.task_id: ["unverified_replay_artifact"]
    }


def _phase3_manifest(*, metric_thresholds: Mapping[str, Any]):
    manifest = load_benchmark_task_manifest(POINT_TASK)
    return replace(
        manifest,
        engine_requirements=(ANALYTIC_BACKEND_SUBMITTER_ID,),
        metric_thresholds=dict(metric_thresholds),
        target_structure_ref="",
    )


def _selected_submitter_result(
    manifest,
    *,
    metrics: Mapping[str, Any],
    candidate_id: str = "phase3-candidate",
    replay_contract: Mapping[str, Any] | None = None,
) -> BenchmarkSubmitterResult:
    return BenchmarkSubmitterResult(
        submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        submitter_class="decomposition",
        task_id=manifest.task_id,
        track_id=manifest.track_id,
        status="selected",
        protocol_contract={
            "task_id": manifest.task_id,
            "track_id": manifest.track_id,
            "dataset_ref": manifest.dataset_ref,
        },
        budget_consumption={
            "declared_candidate_limit": 1,
            "declared_wall_clock_seconds": 1,
            "attempted_candidate_count": 1,
            "accepted_candidate_count": 1,
        },
        selected_candidate_id=candidate_id,
        selected_candidate_hash=f"sha256:{candidate_id}",
        selected_candidate_metrics=dict(metrics),
        replay_contract=dict(
            replay_contract
            or {
                "candidate_id": candidate_id,
                "replay_verification_status": "verified",
                "failure_reason_codes": [],
            }
        ),
    )
