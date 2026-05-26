from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from euclid.benchmarks import profile_benchmark_task
from euclid.benchmarks.runtime import _task_replay_verification_status
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
        "expected_semantic_status",
    ),
    (
        (
            "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml",
            "rediscovery",
            "selected",
            True,
            "passed",
        ),
        (
            "benchmarks/tasks/predictive_generalization/seasonal-trend-demo.yaml",
            "predictive_generalization",
            "selected",
            True,
            "passed",
        ),
        (
            "benchmarks/tasks/adversarial_honesty/leakage-trap-demo.yaml",
            "adversarial_honesty",
            "abstained",
            False,
            "passed",
        ),
    ),
)
def test_phase08_benchmark_smokes_cover_each_track_with_budget_and_replay_guards(
    tmp_path: Path,
    relative_path: str,
    expected_track: str,
    expected_portfolio_status: str,
    expect_local_winner: bool,
    expected_semantic_status: str,
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
    semantic_assertions = task_result["semantic_assertions"]
    assert semantic_assertions["overall_status"] == expected_semantic_status
    if relative_path.endswith("planted-analytic-demo.yaml"):
        predictive_floor = {
            row["threshold_id"]: row
            for row in semantic_assertions["metric_thresholds"]["assertions"]
        }["predictive_adequacy_floor"]
        assert predictive_floor["metric_id"] == "mean_absolute_error"
        assert predictive_floor["observed_value"] <= predictive_floor["threshold"]
        assert predictive_floor["reason_code"] == "observed"
        assert predictive_floor["status"] == "passed"
        assert semantic_assertions["rediscovery_target"]["status"] == "passed"
        assert semantic_assertions["rediscovery_target"]["reason_code"] == (
            "selected_candidate_structurally_equivalent"
        )
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


def test_phase08_portfolio_medium_prefers_threshold_passing_candidate(
    tmp_path: Path,
) -> None:
    result = profile_benchmark_task(
        manifest_path=(
            PROJECT_ROOT
            / "benchmarks/tasks/predictive_generalization/"
            "portfolio-selection-medium.yaml"
        ),
        benchmark_root=tmp_path / "portfolio-medium-threshold-gate",
        parallel_workers=2,
        resume=False,
    )

    submitter_by_id = {
        submitter_result.submitter_id: submitter_result
        for submitter_result in result.submitter_results
    }
    portfolio = submitter_by_id[PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID]
    task_result = json.loads(
        result.report_paths.task_result_path.read_text(encoding="utf-8")
    )
    threshold_rows = {
        row["threshold_id"]: row
        for row in task_result["semantic_assertions"]["metric_thresholds"][
            "assertions"
        ]
    }

    assert portfolio.status == "selected"
    assert portfolio.selected_candidate_id == "analytic_lag1_affine"
    assert (
        portfolio.replay_contract["selected_submitter_id"]
        == ANALYTIC_BACKEND_SUBMITTER_ID
    )
    assert task_result["local_winner_submitter_id"] == ANALYTIC_BACKEND_SUBMITTER_ID
    assert task_result["local_winner_candidate_id"] == "analytic_lag1_affine"
    assert task_result["semantic_assertions"]["overall_status"] == "passed"
    assert threshold_rows["practical_significance_margin"]["status"] == "passed"


@pytest.mark.parametrize(
    ("relative_path", "expected_candidate_id", "expected_threshold_reason"),
    (
        (
            "benchmarks/tasks/mechanistic/mechanistic-lane-medium-positive.yaml",
            "analytic_lag1_affine",
            "observed",
        ),
        (
            "benchmarks/tasks/mechanistic/mechanistic-lane-medium-negative.yaml",
            None,
            "not_applicable_safe_abstention",
        ),
        (
            "benchmarks/tasks/mechanistic/mechanistic-lane-medium-insufficient.yaml",
            None,
            "not_applicable_safe_abstention",
        ),
    ),
)
def test_phase08_mechanistic_medium_tasks_have_truthful_claim_semantics(
    tmp_path: Path,
    relative_path: str,
    expected_candidate_id: str | None,
    expected_threshold_reason: str,
) -> None:
    result = profile_benchmark_task(
        manifest_path=PROJECT_ROOT / relative_path,
        benchmark_root=tmp_path / "mechanistic-medium-semantics",
        resume=False,
    )

    task_result = json.loads(
        result.report_paths.task_result_path.read_text(encoding="utf-8")
    )
    threshold_rows = {
        row["threshold_id"]: row
        for row in task_result["semantic_assertions"]["metric_thresholds"][
            "assertions"
        ]
    }
    practical_margin = threshold_rows["practical_significance_margin"]

    assert task_result["semantic_assertions"]["overall_status"] == "passed"
    assert task_result.get("local_winner_candidate_id") == expected_candidate_id
    assert practical_margin["status"] == "passed"
    assert practical_margin["reason_code"] == expected_threshold_reason


@pytest.mark.parametrize(
    ("relative_path", "expected_candidate_id", "expected_threshold_reason"),
    (
        (
            "benchmarks/tasks/robustness/robustness-medium-positive.yaml",
            "analytic_lag1_affine",
            "observed",
        ),
        (
            "benchmarks/tasks/robustness/"
            "robustness-medium-sensitivity-abstention.yaml",
            None,
            "not_applicable_safe_abstention",
        ),
    ),
)
def test_phase08_robustness_medium_tasks_have_truthful_claim_semantics(
    tmp_path: Path,
    relative_path: str,
    expected_candidate_id: str | None,
    expected_threshold_reason: str,
) -> None:
    result = profile_benchmark_task(
        manifest_path=PROJECT_ROOT / relative_path,
        benchmark_root=tmp_path / "robustness-medium-semantics",
        resume=False,
    )
    task_result = json.loads(
        result.report_paths.task_result_path.read_text(encoding="utf-8")
    )
    threshold_rows = {
        row["threshold_id"]: row
        for row in task_result["semantic_assertions"]["metric_thresholds"][
            "assertions"
        ]
    }
    practical_margin = threshold_rows["practical_significance_margin"]

    assert task_result["semantic_assertions"]["overall_status"] == "passed"
    assert task_result.get("local_winner_candidate_id") == expected_candidate_id
    assert practical_margin["status"] == "passed"
    assert practical_margin["reason_code"] == expected_threshold_reason


def test_phase08_algorithmic_task_uses_measured_evaluation_metric(
    tmp_path: Path,
) -> None:
    result = profile_benchmark_task(
        manifest_path=(
            PROJECT_ROOT
            / "benchmarks/tasks/algorithmic_rediscovery/"
            "causal-last-observation-medium.yaml"
        ),
        benchmark_root=tmp_path / "algorithmic-measured-benchmark",
        resume=False,
    )

    task_result = json.loads(
        result.report_paths.task_result_path.read_text(encoding="utf-8")
    )
    rows = {
        row["threshold_id"]: row
        for row in task_result["semantic_assertions"]["metric_thresholds"][
            "assertions"
        ]
    }

    predictive_floor = rows["predictive_adequacy_floor"]
    assert task_result["semantic_assertions"]["overall_status"] == "passed"
    assert predictive_floor["metric_id"] == "mean_absolute_error"
    assert predictive_floor["observed_value"] <= 2.0
    assert predictive_floor["observed_value"] != 21
    assert predictive_floor["reason_code"] == "observed"
    assert task_result["semantic_assertions"]["rediscovery_target"]["status"] == (
        "passed"
    )
    assert (
        task_result["local_winner_candidate_id"]
        == "algorithmic_running_half_average"
    )


def test_phase08_composition_task_attempts_declared_operator_candidate_first(
    tmp_path: Path,
) -> None:
    result = profile_benchmark_task(
        manifest_path=(
            PROJECT_ROOT
            / "benchmarks/tasks/predictive_generalization/"
            "composition-piecewise-medium.yaml"
        ),
        benchmark_root=tmp_path / "composition-proposal-order",
        resume=False,
    )
    submitter = result.submitter_results[0]

    assert submitter.candidate_ledger[0].candidate_id == "analytic_piecewise_surface"
    assert submitter.candidate_ledger[0].attempted_rank == 0
    assert submitter.status == "selected"


@pytest.mark.parametrize(
    ("relative_path", "expected_candidate_id"),
    (
        (
            "benchmarks/tasks/predictive_generalization/"
            "composition-additive-residual-medium.yaml",
            "analytic_additive_residual_surface",
        ),
        (
            "benchmarks/tasks/predictive_generalization/"
            "composition-regime-conditioned-medium.yaml",
            "analytic_regime_conditioned_surface",
        ),
    ),
)
def test_phase08_composition_medium_tasks_emit_replayed_margin_evidence(
    tmp_path: Path,
    relative_path: str,
    expected_candidate_id: str,
) -> None:
    result = profile_benchmark_task(
        manifest_path=PROJECT_ROOT / relative_path,
        benchmark_root=tmp_path / "composition-medium-margin",
        resume=False,
    )
    task_result = json.loads(
        result.report_paths.task_result_path.read_text(encoding="utf-8")
    )
    threshold_rows = {
        row["threshold_id"]: row
        for row in task_result["semantic_assertions"]["metric_thresholds"][
            "assertions"
        ]
    }

    assert task_result["semantic_assertions"]["overall_status"] == "passed"
    assert task_result["local_winner_candidate_id"] == expected_candidate_id
    assert threshold_rows["practical_significance_margin"]["status"] == "passed"
    composition_semantics = task_result["semantic_assertions"][
        "composition_operator_semantics"
    ]
    assert composition_semantics["status"] == "passed"
    if "additive-residual" in relative_path:
        additive_row = composition_semantics["assertions"][0]
        assert additive_row["reason_code"] == (
            "composition_operator_behavioral_margin_verified"
        )
        assert additive_row["graph_has_distinct_components"] is True


def test_phase08_replay_status_reads_verification_status_from_replay_files(
    tmp_path: Path,
) -> None:
    missing_task_result_path = tmp_path / "missing-task-result.json"
    missing_result = SimpleNamespace(
        report_paths=SimpleNamespace(
            task_result_path=missing_task_result_path,
            replay_ref_paths={"analytic_backend": tmp_path / "missing-replay.json"},
        )
    )
    replay_path = tmp_path / "unverified-replay.json"
    task_result_path = tmp_path / "unverified-task-result.json"
    task_result_path.write_text(
        json.dumps({"track_summary": {"replay_verification_status": "unverified"}}),
        encoding="utf-8",
    )
    replay_path.write_text(
        json.dumps(
            {
                "artifact_type": "benchmark_replay_ref",
                "replay_verification_status": "unverified",
            }
        ),
        encoding="utf-8",
    )
    unverified_result = SimpleNamespace(
        report_paths=SimpleNamespace(
            task_result_path=task_result_path,
            replay_ref_paths={"analytic_backend": replay_path},
        )
    )

    assert _task_replay_verification_status(missing_result) == "missing"
    assert _task_replay_verification_status(unverified_result) == "unverified"


def _selected_metric_sort_key(result) -> tuple[float, float, float, int, str]:
    metrics = result.selected_candidate_metrics or {}
    return (
        float(metrics.get("total_code_bits", float("inf"))),
        -float(metrics.get("description_gain_bits", float("-inf"))),
        float(metrics.get("structure_code_bits", float("inf"))),
        int(metrics.get("canonical_byte_length", 0)),
        result.submitter_id,
    )
