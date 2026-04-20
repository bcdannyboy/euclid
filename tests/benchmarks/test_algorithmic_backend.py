from __future__ import annotations

from pathlib import Path

import euclid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "benchmarks/tasks/algorithmic_rediscovery/causal-last-observation-demo.yaml"
)


def test_algorithmic_backend_profiles_full_enumerator_and_records_replay(
    tmp_path: Path,
) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=MANIFEST_PATH,
        benchmark_root=tmp_path / "benchmark-output",
        resume=False,
    )

    assert result.task_manifest.task_family == "algorithmic_symbolic_regression"
    assert tuple(run.submitter_id for run in result.submitter_results) == (
        "algorithmic_search_backend",
    )

    submitter = result.submitter_results[0]
    assert submitter.status == "selected"
    assert submitter.selected_candidate_id is not None
    assert submitter.budget_consumption["canonical_program_count"] > 2
    assert (
        submitter.budget_consumption["attempted_candidate_count"]
        == submitter.budget_consumption["canonical_program_count"]
    )
    assert submitter.replay_contract["candidate_id"] == submitter.selected_candidate_id
    replay_hook_names = {
        hook["hook_name"] for hook in submitter.replay_contract["replay_hooks"]
    }
    assert replay_hook_names >= {"algorithmic_dsl", "search_scope", "search_seed"}
    assert result.report_paths.task_result_path.exists()
    assert result.report_paths.submitter_result_paths[
        "algorithmic_search_backend"
    ].exists()
    assert result.report_paths.replay_ref_paths[
        "algorithmic_search_backend"
    ].exists()
