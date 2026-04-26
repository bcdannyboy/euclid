from __future__ import annotations

from pathlib import Path

import euclid
from euclid.control_plane import SQLiteExecutionStateStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_MANIFEST = PROJECT_ROOT / "fixtures/runtime/prototype-demo.yaml"


def test_demo_workflow_rerun_and_replay_are_deterministic(tmp_path: Path) -> None:
    first_output_root = tmp_path / "demo-run-1"
    second_output_root = tmp_path / "demo-run-2"

    first_run = euclid.run_demo(
        manifest_path=SAMPLE_MANIFEST,
        output_root=first_output_root,
    )
    replay = euclid.replay_demo(output_root=first_output_root)
    second_run = euclid.run_demo(
        manifest_path=SAMPLE_MANIFEST,
        output_root=second_output_root,
    )

    assert first_run.request.dataset_csv == second_run.request.dataset_csv
    assert first_run.summary.selected_family == "seasonal_naive"
    assert replay.summary.replay_verification_status == "verified"
    assert replay.summary.bundle_ref == first_run.summary.bundle_ref
    assert replay.summary.run_result_ref == first_run.summary.run_result_ref
    assert replay.summary.selected_candidate_ref == (
        first_run.summary.selected_candidate_ref
    )
    assert second_run.summary.bundle_ref == first_run.summary.bundle_ref
    assert second_run.summary.run_result_ref == first_run.summary.run_result_ref
    assert second_run.summary.selected_candidate_ref == (
        first_run.summary.selected_candidate_ref
    )
    assert second_run.summary.confirmatory_primary_score == (
        first_run.summary.confirmatory_primary_score
    )
    assert first_run.paths.active_run_root.is_dir()
    assert first_run.paths.sealed_run_root.is_dir()
    assert first_run.paths.control_plane_store_path.is_file()

    control_plane_store = SQLiteExecutionStateStore(
        first_run.paths.control_plane_store_path
    )
    snapshot = control_plane_store.load_run_snapshot(first_run.request.request_id)

    assert {event.state_id for event in snapshot.stage_events} >= {
        "run_declared",
        "search_contract_frozen",
        "candidate_set_frozen",
        "replay_verified",
    }
    assert snapshot.freeze_markers[0].manifest_ref == (
        first_run.workflow_result.freeze_event.manifest.ref
    )
    assert snapshot.budget_counters[0].consumed == len(
        first_run.workflow_result.candidate_summaries
    )
    assert snapshot.seed_records[0].seed_scope == "search"
    assert snapshot.step_states[0].status == "completed"
