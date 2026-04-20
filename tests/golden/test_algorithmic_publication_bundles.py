from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import euclid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "fixtures/runtime/phase06"
MANIFEST_PATH = (
    PROJECT_ROOT
    / "benchmarks/tasks/algorithmic_rediscovery/causal-last-observation-demo.yaml"
)


def _algorithmic_benchmark_snapshot(output_root: Path) -> dict[str, Any]:
    result = euclid.profile_benchmark_task(
        manifest_path=MANIFEST_PATH,
        benchmark_root=output_root,
        resume=False,
    )
    submitter = result.submitter_results[0]
    task_result = json.loads(
        result.report_paths.task_result_path.read_text(encoding="utf-8")
    )
    submitter_result = json.loads(
        result.report_paths.submitter_result_paths["algorithmic_search_backend"].read_text(
            encoding="utf-8"
        )
    )
    replay_ref = json.loads(
        result.report_paths.replay_ref_paths["algorithmic_search_backend"].read_text(
            encoding="utf-8"
        )
    )
    return {
        "task_id": result.task_manifest.task_id,
        "track_id": result.task_manifest.track_id,
        "telemetry_subject_id": result.telemetry.subject_id,
        "submitter": {
            "submitter_id": submitter.submitter_id,
            "status": submitter.status,
            "selected_candidate_id": submitter.selected_candidate_id,
            "canonical_program_count": submitter.budget_consumption[
                "canonical_program_count"
            ],
            "attempted_candidate_count": submitter.budget_consumption[
                "attempted_candidate_count"
            ],
            "accepted_candidate_count": submitter.budget_consumption[
                "accepted_candidate_count"
            ],
            "replay_hook_names": [
                hook["hook_name"]
                for hook in submitter.replay_contract.get("replay_hooks", [])
            ],
        },
        "task_result": {
            "status": task_result["status"],
            "local_winner_submitter_id": task_result.get("local_winner_submitter_id"),
            "local_winner_candidate_id": task_result.get("local_winner_candidate_id"),
            "submitter_result_refs": task_result["submitter_result_refs"],
            "replay_ref_refs": task_result["replay_ref_refs"],
        },
        "submitter_result": {
            "status": submitter_result["status"],
            "selected_candidate_id": submitter_result["selected_candidate_id"],
            "budget_consumption": submitter_result["budget_consumption"],
        },
        "replay_ref": {
            "submitter_id": replay_ref["submitter_id"],
            "candidate_id": replay_ref["replay_contract"].get("candidate_id"),
            "candidate_hash": replay_ref["replay_contract"].get("candidate_hash"),
        },
    }


def _expected_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def test_algorithmic_benchmark_publication_matches_golden_fixture(
    tmp_path: Path,
) -> None:
    assert _algorithmic_benchmark_snapshot(
        tmp_path / "algorithmic-benchmark-publication"
    ) == _expected_fixture("algorithmic-benchmark-publication-golden.json")
