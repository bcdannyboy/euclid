from __future__ import annotations

from euclid.performance import (
    CandidateThroughputBudget,
    evaluate_candidate_throughput_budget,
)


def test_candidate_throughput_budget_reports_measured_rate_and_threshold() -> None:
    result = evaluate_candidate_throughput_budget(
        candidate_count=120,
        elapsed_seconds=2.0,
        budget=CandidateThroughputBudget(
            budget_id="candidate_throughput_smoke",
            min_candidates_per_second=50.0,
        ),
    )

    assert result.passed
    assert result.observed_candidates_per_second == 60.0
    assert result.as_dict()["thresholds"] == {
        "min_candidates_per_second": 50.0,
    }


def test_candidate_throughput_budget_fails_closed_on_regression() -> None:
    result = evaluate_candidate_throughput_budget(
        candidate_count=10,
        elapsed_seconds=2.0,
        budget=CandidateThroughputBudget(
            budget_id="candidate_throughput_regression",
            min_candidates_per_second=8.0,
        ),
    )

    assert not result.passed
    assert result.failure_reasons == (
        "candidates_per_second=5.000000 below 8.000000",
    )
