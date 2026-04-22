from __future__ import annotations

import json
from pathlib import Path

import pytest

from euclid.benchmarks import profile_benchmark_task
from euclid.benchmarks.submitters import (
    ALGORITHMIC_SEARCH_SUBMITTER_ID,
    ANALYTIC_BACKEND_SUBMITTER_ID,
    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
    RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    (
        "relative_path",
        "expected_track",
        "expected_portfolio_status",
        "expect_local_winner",
    ),
    (
        (
            "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml",
            "rediscovery",
            "selected",
            True,
        ),
        (
            "benchmarks/tasks/predictive_generalization/seasonal-trend-demo.yaml",
            "predictive_generalization",
            "selected",
            True,
        ),
        (
            "benchmarks/tasks/adversarial_honesty/leakage-trap-demo.yaml",
            "adversarial_honesty",
            "abstained",
            False,
        ),
    ),
)
def test_phase08_benchmark_smokes_cover_each_track_with_budget_and_replay_guards(
    tmp_path: Path,
    relative_path: str,
    expected_track: str,
    expected_portfolio_status: str,
    expect_local_winner: bool,
) -> None:
    result = profile_benchmark_task(
        manifest_path=PROJECT_ROOT / relative_path,
        benchmark_root=tmp_path / "benchmarks",
        parallel_workers=2,
        resume=False,
    )

    submitter_by_id = {
        submitter_result.submitter_id: submitter_result
        for submitter_result in result.submitter_results
    }
    assert result.task_manifest.track_id == expected_track
    assert (
        submitter_by_id[PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID].status
        == expected_portfolio_status
    )

    task_result = json.loads(
        result.report_paths.task_result_path.read_text(encoding="utf-8")
    )
    assert task_result["track_id"] == expected_track
    if expect_local_winner:
        assert task_result["local_winner_submitter_id"] in {
            ANALYTIC_BACKEND_SUBMITTER_ID,
            RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
            ALGORITHMIC_SEARCH_SUBMITTER_ID,
        }
        assert task_result["local_winner_candidate_id"]
    else:
        assert "local_winner_submitter_id" not in task_result
        assert "local_winner_candidate_id" not in task_result
    assert task_result["semantic_assertions"]["overall_status"] == "passed"
    assert task_result["semantic_assertions"]["claim_scope"][
        "counts_as_claim_evidence"
    ] is False
    assert task_result["portfolio_selection_record_ref"]["artifact_type"] == (
        "portfolio_selection_record"
    )

    telemetry = json.loads(result.telemetry_path.read_text(encoding="utf-8"))
    assert telemetry["profile_kind"] == "benchmark_task"
    assert telemetry["wall_time_seconds"] <= float(
        result.task_manifest.frozen_protocol.budget_policy["wall_clock_seconds"]
    )
    budget_records = {
        record["submitter_id"]: record for record in telemetry["budget_records"]
    }
    assert set(budget_records) == {
        ANALYTIC_BACKEND_SUBMITTER_ID,
        RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
        ALGORITHMIC_SEARCH_SUBMITTER_ID,
    }
    for record in budget_records.values():
        assert record["attempted_candidate_count"] <= record["declared_candidate_limit"]
        assert record["declared_wall_clock_seconds"] <= int(
            result.task_manifest.frozen_protocol.budget_policy["wall_clock_seconds"]
        )

    report_text = result.report_paths.report_path.read_text(encoding="utf-8")
    assert "No single vanity score is reported here." in report_text

    for submitter_id, replay_path in result.report_paths.replay_ref_paths.items():
        replay_ref = json.loads(replay_path.read_text(encoding="utf-8"))
        replay_contract = replay_ref["replay_contract"]

        assert replay_ref["task_id"] == result.task_manifest.task_id
        assert replay_ref["track_id"] == expected_track
        if submitter_id == PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID:
            if expected_portfolio_status == "selected":
                assert replay_contract["selected_submitter_id"] in {
                    ANALYTIC_BACKEND_SUBMITTER_ID,
                    RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
                    ALGORITHMIC_SEARCH_SUBMITTER_ID,
                }
                assert replay_contract["selected_candidate_id"] == submitter_by_id[
                    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
                ].selected_candidate_id
            else:
                assert replay_contract["selected_submitter_id"] is None
                assert replay_contract["selected_candidate_id"] is None
            continue

        assert replay_contract["search_plan_id"].endswith(
            f"__{submitter_id}__search_plan"
        )
        if submitter_by_id[submitter_id].status == "selected":
            assert replay_contract["candidate_id"] == submitter_by_id[
                submitter_id
            ].selected_candidate_id
        else:
            assert "candidate_id" not in replay_contract


def test_phase08_predictive_portfolio_matches_best_single_family(
    tmp_path: Path,
) -> None:
    result = profile_benchmark_task(
        manifest_path=(
            PROJECT_ROOT
            / "benchmarks/tasks/predictive_generalization/seasonal-trend-demo.yaml"
        ),
        benchmark_root=tmp_path / "benchmarks",
        parallel_workers=2,
        resume=False,
    )

    submitter_by_id = {
        submitter_result.submitter_id: submitter_result
        for submitter_result in result.submitter_results
    }
    child_results = tuple(
        submitter_by_id[submitter_id]
        for submitter_id in (
            ANALYTIC_BACKEND_SUBMITTER_ID,
            RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
            ALGORITHMIC_SEARCH_SUBMITTER_ID,
        )
        if submitter_by_id[submitter_id].status == "selected"
    )
    best_single = min(child_results, key=_selected_metric_sort_key)
    portfolio = submitter_by_id[PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID]

    assert portfolio.status == "selected"
    assert portfolio.selected_candidate_id == best_single.selected_candidate_id
    assert (
        portfolio.replay_contract["selected_submitter_id"]
        == best_single.submitter_id
    )
    assert portfolio.compared_finalists[0]["submitter_id"] == best_single.submitter_id


def _selected_metric_sort_key(result) -> tuple[float, float, float, int, str]:
    metrics = result.selected_candidate_metrics or {}
    return (
        float(metrics.get("total_code_bits", float("inf"))),
        -float(metrics.get("description_gain_bits", float("-inf"))),
        float(metrics.get("structure_code_bits", float("inf"))),
        int(metrics.get("canonical_byte_length", 0)),
        result.submitter_id,
    )
