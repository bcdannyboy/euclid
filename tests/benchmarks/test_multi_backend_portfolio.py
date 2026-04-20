from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import euclid
from euclid.benchmarks.submitters import (
    ALGORITHMIC_SEARCH_SUBMITTER_ID,
    ANALYTIC_BACKEND_SUBMITTER_ID,
    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
    RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
)
from euclid.contracts.errors import ContractValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PORTFOLIO_MANIFEST = (
    PROJECT_ROOT
    / "benchmarks/tasks/predictive_generalization/portfolio-selection-medium.yaml"
)
POINT_TASK = PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"


def test_benchmark_portfolio_records_ranked_finalists_in_replay_contract(
    tmp_path: Path,
) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=PORTFOLIO_MANIFEST,
        benchmark_root=tmp_path / "multi-backend-portfolio",
        resume=False,
    )

    portfolio = next(
        item
        for item in result.submitter_results
        if item.submitter_id == PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
    )
    selection_record = json.loads(
        result.report_paths.portfolio_selection_record_path.read_text(encoding="utf-8")
    )

    assert portfolio.status == "selected"
    assert portfolio.selected_candidate_id == "analytic_lag1_affine"
    assert (
        portfolio.replay_contract["selection_scope"]
        == "benchmark_multi_backend_portfolio"
    )
    assert portfolio.replay_contract["selection_rule"].startswith("min_total_code_bits")
    assert len(portfolio.replay_contract["compared_finalists"]) == 3
    assert selection_record["selected_submitter_id"] == ANALYTIC_BACKEND_SUBMITTER_ID
    assert selection_record["selected_candidate_id"] == "analytic_lag1_affine"
    assert len(selection_record["compared_finalists"]) == 3
    assert selection_record["selection_explanation"]["winner"] == {
        "submitter_id": ANALYTIC_BACKEND_SUBMITTER_ID,
        "candidate_id": "analytic_lag1_affine",
        "total_code_bits": 144.0,
    }
    assert selection_record["selection_explanation"]["runner_up"] == {
        "submitter_id": RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
        "candidate_id": "recursive_level_smoother",
        "total_code_bits": 146.0,
    }
    assert selection_record["selection_explanation"]["decisive_axis"] == (
        "total_code_bits"
    )
    assert [step["step"] for step in selection_record["decision_trace"]] == [
        "collect_submitter_finalists",
        "rank_submitter_finalists",
        "select_portfolio_winner",
    ]


def test_submitter_artifacts_persist_true_backend_identity(tmp_path: Path) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=PORTFOLIO_MANIFEST,
        benchmark_root=tmp_path / "portfolio-backend-identity",
        resume=False,
    )

    submitter_artifact = json.loads(
        result.report_paths.submitter_result_paths[
            result.submitter_results[0].submitter_id
        ].read_text(encoding="utf-8")
    )

    assert submitter_artifact["submitter_id"] in {
        ANALYTIC_BACKEND_SUBMITTER_ID,
        RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
        ALGORITHMIC_SEARCH_SUBMITTER_ID,
    }
    assert submitter_artifact["protocol_contract"]["search_class"] in {
        "bounded_heuristic",
        "stochastic_heuristic",
    }
    assert submitter_artifact["backend_participation"]


def test_portfolio_requires_multiple_real_backend_finalists(tmp_path: Path) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=PORTFOLIO_MANIFEST,
        benchmark_root=tmp_path / "portfolio-multiple-finalists",
        resume=False,
    )

    portfolio = next(
        item
        for item in result.submitter_results
        if item.submitter_id == PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
    )
    finalist_submitter_ids = {
        finalist["submitter_id"] for finalist in portfolio.compared_finalists
    }
    finalist_backend_families = {
        finalist["backend_family"] for finalist in portfolio.compared_finalists
    }

    assert len(finalist_submitter_ids) >= 3
    assert len(finalist_backend_families) >= 3


def test_single_retained_core_task_cannot_close_portfolio_surface(
    tmp_path: Path,
) -> None:
    suite_manifest = tmp_path / "benchmarks" / "suites" / "dishonest-portfolio.yaml"
    suite_manifest.parent.mkdir(parents=True, exist_ok=True)
    suite_manifest.write_text(
        yaml.safe_dump(
            {
                "suite_id": "dishonest_portfolio_surface",
                "description": (
                    "Invalid suite that tries to close portfolio with one "
                    "retained-core task."
                ),
                "task_manifest_paths": [str(POINT_TASK.resolve())],
                "required_tracks": ["rediscovery"],
                "surface_requirements": [
                    {
                        "surface_id": "portfolio_orchestration",
                        "task_ids": ["planted_analytic_demo"],
                        "replay_required": True,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractValidationError):
        euclid.profile_benchmark_suite(
            manifest_path=suite_manifest,
            benchmark_root=tmp_path / "dishonest-portfolio-output",
            project_root=PROJECT_ROOT,
            resume=False,
        )
