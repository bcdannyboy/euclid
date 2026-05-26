from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from euclid.benchmarks import load_benchmark_task_manifest
from euclid.benchmarks.reporting import evaluate_benchmark_semantic_assertions
from euclid.benchmarks.submitters import (
    ANALYTIC_BACKEND_SUBMITTER_ID,
    BenchmarkSubmitterResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_missing_required_observed_metric_fails_closed_with_source_submitter() -> None:
    manifest = load_benchmark_task_manifest(
        PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
    )
    submitter_results = _submitter_results_for_manifest(
        manifest,
        winning_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        selected_candidate_metrics={},
    )

    assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=submitter_results,
        local_winner_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        local_winner_candidate_id="candidate_without_measurements",
    )

    threshold_assertions = assertions["metric_thresholds"]["assertions"]
    practical_margin_row = _row_for_threshold(
        threshold_assertions,
        "practical_significance_margin",
    )
    assert assertions["metric_thresholds"]["status"] == "failed"
    assert practical_margin_row["threshold_id"] == "practical_significance_margin"
    assert practical_margin_row["metric_id"] == "practical_significance_margin"
    assert practical_margin_row["source_submitter_id"] == ANALYTIC_BACKEND_SUBMITTER_ID
    assert practical_margin_row["observed_value"] is None
    assert practical_margin_row["status"] == "failed"
    assert practical_margin_row["reason"] == "missing_observed_metric"


def test_safe_abstention_missing_metrics_pass_when_expected_and_no_winner() -> None:
    manifest = load_benchmark_task_manifest(
        PROJECT_ROOT / "benchmarks/tasks/adversarial_honesty/leakage-trap-demo.yaml"
    )
    submitter_results = _submitter_results_for_manifest(manifest)

    assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=submitter_results,
        local_winner_submitter_id=None,
        local_winner_candidate_id=None,
    )

    threshold_assertions = assertions["metric_thresholds"]["assertions"]
    practical_margin_row = _row_for_threshold(
        threshold_assertions,
        "practical_significance_margin",
    )
    assert assertions["metric_thresholds"]["status"] == "passed"
    assert practical_margin_row["source_submitter_id"] is None
    assert practical_margin_row["observed_value"] is None
    assert practical_margin_row["status"] == "passed"
    assert practical_margin_row["reason"] == "not_applicable_safe_abstention"


def test_safe_abstention_missing_metrics_fail_when_winner_exists() -> None:
    manifest = load_benchmark_task_manifest(
        PROJECT_ROOT / "benchmarks/tasks/adversarial_honesty/leakage-trap-demo.yaml"
    )
    submitter_results = _submitter_results_for_manifest(
        manifest,
        winning_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        selected_candidate_metrics={},
    )

    assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=submitter_results,
        local_winner_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        local_winner_candidate_id="unsafe_candidate",
    )

    practical_margin_row = _row_for_threshold(
        assertions["metric_thresholds"]["assertions"],
        "practical_significance_margin",
    )
    assert assertions["metric_thresholds"]["status"] == "failed"
    assert practical_margin_row["source_submitter_id"] == ANALYTIC_BACKEND_SUBMITTER_ID
    assert practical_margin_row["status"] == "failed"
    assert practical_margin_row["reason"] == "missing_observed_metric"


def test_safe_abstention_missing_metrics_fail_when_winner_submitter_exists() -> None:
    manifest = load_benchmark_task_manifest(
        PROJECT_ROOT / "benchmarks/tasks/adversarial_honesty/leakage-trap-demo.yaml"
    )
    submitter_results = _submitter_results_for_manifest(manifest)

    assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=submitter_results,
        local_winner_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        local_winner_candidate_id=None,
    )

    practical_margin_row = _row_for_threshold(
        assertions["metric_thresholds"]["assertions"],
        "practical_significance_margin",
    )
    assert assertions["metric_thresholds"]["status"] == "failed"
    assert practical_margin_row["source_submitter_id"] is None
    assert practical_margin_row["status"] == "failed"
    assert practical_margin_row["reason"] == "missing_observed_metric"


def test_missing_metric_can_pass_only_when_measurement_not_required() -> None:
    manifest = load_benchmark_task_manifest(
        PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
    )
    manifest = replace(
        manifest,
        metric_thresholds={
            "optional_diagnostic_metric": {
                "metric_id": "diagnostic_only_metric",
                "comparator": ">=",
                "threshold": 0.5,
                "measurement_required": False,
            }
        },
    )
    submitter_results = _submitter_results_for_manifest(
        manifest,
        winning_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        selected_candidate_metrics={},
    )

    assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=submitter_results,
        local_winner_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        local_winner_candidate_id="candidate_without_optional_diagnostic",
    )

    optional_row = _row_for_threshold(
        assertions["metric_thresholds"]["assertions"],
        "optional_diagnostic_metric",
    )
    assert assertions["metric_thresholds"]["status"] == "passed"
    assert optional_row["source_submitter_id"] == ANALYTIC_BACKEND_SUBMITTER_ID
    assert optional_row["measurement_required"] is False
    assert optional_row["observed_value"] is None
    assert optional_row["status"] == "passed"
    assert optional_row["reason"] == "declared_not_measured_by_current_harness"


def _submitter_results_for_manifest(
    manifest: Any,
    *,
    winning_submitter_id: str | None = None,
    selected_candidate_metrics: Mapping[str, Any] | None = None,
) -> tuple[BenchmarkSubmitterResult, ...]:
    return tuple(
        _submitter_result(
            manifest,
            submitter_id=submitter_id,
            selected=submitter_id == winning_submitter_id,
            selected_candidate_metrics=(
                selected_candidate_metrics
                if submitter_id == winning_submitter_id
                else None
            ),
        )
        for submitter_id in manifest.submitter_ids
    )


def _submitter_result(
    manifest: Any,
    *,
    submitter_id: str,
    selected: bool,
    selected_candidate_metrics: Mapping[str, Any] | None,
) -> BenchmarkSubmitterResult:
    selected_candidate_id = f"{submitter_id}_candidate" if selected else None
    return BenchmarkSubmitterResult(
        submitter_id=submitter_id,
        submitter_class="phase31_worker_fixture",
        task_id=manifest.task_id,
        track_id=manifest.track_id,
        status="selected" if selected else "abstained",
        protocol_contract={"task_id": manifest.task_id, "submitter_id": submitter_id},
        budget_consumption={"attempted_candidate_count": 1 if selected else 0},
        selected_candidate_id=selected_candidate_id,
        selected_candidate_hash=(
            f"sha256:{selected_candidate_id}" if selected_candidate_id else None
        ),
        selected_candidate_metrics=selected_candidate_metrics,
        replay_contract={"candidate_id": selected_candidate_id},
        abstention_reason=None if selected else "no_admissible_candidate",
        safe_abstention_evidence=(
            {}
            if selected
            else {
                "status": "verified",
                "reason_code": "no_admissible_candidate",
                "evidence_type": "falsification_gate",
                "support": [
                    {
                        "task_id": manifest.task_id,
                        "submitter_id": submitter_id,
                    }
                ],
            }
        ),
    )


def _row_for_threshold(
    rows: list[dict[str, Any]],
    threshold_id: str,
) -> dict[str, Any]:
    return next(row for row in rows if row["threshold_id"] == threshold_id)
