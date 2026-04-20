from __future__ import annotations

from pathlib import Path

from euclid.contracts.refs import TypedRef
from euclid.control_plane import SQLiteExecutionStateStore


def test_execution_state_store_round_trips_run_bookkeeping(tmp_path: Path) -> None:
    store = SQLiteExecutionStateStore(tmp_path / "control-plane.sqlite3")

    store.append_stage_event(
        run_id="demo_run",
        state_id="run_declared",
        stage="run_binding",
        module_id="manifest_registry",
        status="completed",
        details={"entrypoint_id": "demo.run"},
    )
    store.append_stage_event(
        run_id="demo_run",
        state_id="candidate_set_frozen",
        stage="candidate_freeze",
        module_id="search_planning",
        status="completed",
        details={"freeze_stage": "global_pair_freeze_pre_holdout"},
    )
    store.upsert_freeze_marker(
        run_id="demo_run",
        marker_id="candidate_set_frozen",
        state_id="candidate_set_frozen",
        manifest_ref=TypedRef(
            "freeze_event_manifest@1.0.0",
            "demo_freeze_event",
        ),
        details={"freeze_boundary": "candidate_freeze"},
    )
    store.set_budget_counter(
        run_id="demo_run",
        counter_name="candidate_family_evaluations",
        consumed=4,
        limit=4,
        unit="candidates",
    )
    store.record_seed(
        run_id="demo_run",
        seed_scope="search",
        seed_value="0",
        recorded_by="demo.run",
    )
    store.upsert_worker_metadata(
        run_id="demo_run",
        worker_id="prototype.reducer.workflow",
        module_id="candidate_fitting",
        status="completed",
        details={"selected_family": "constant"},
    )
    store.save_step_state(
        run_id="demo_run",
        step_id="demo.run.workflow",
        module_id="manifest_registry",
        status="running",
        cursor="candidate_set_frozen",
        details={"resume_from": "candidate_set_frozen"},
    )
    store.save_step_state(
        run_id="demo_run",
        step_id="demo.run.workflow",
        module_id="catalog_publishing",
        status="completed",
        cursor="candidate_publication_completed",
        details={"result_mode": "candidate_publication"},
    )

    snapshot = store.load_run_snapshot("demo_run")

    assert [event.state_id for event in snapshot.stage_events] == [
        "run_declared",
        "candidate_set_frozen",
    ]
    assert snapshot.stage_events[0].details == {"entrypoint_id": "demo.run"}
    assert snapshot.freeze_markers[0].manifest_ref == TypedRef(
        "freeze_event_manifest@1.0.0",
        "demo_freeze_event",
    )
    assert snapshot.budget_counters[0].consumed == 4
    assert snapshot.budget_counters[0].limit == 4
    assert snapshot.seed_records[0].seed_scope == "search"
    assert snapshot.seed_records[0].recorded_by == "demo.run"
    assert snapshot.worker_metadata[0].details == {"selected_family": "constant"}
    assert snapshot.step_states[0].status == "completed"
    assert snapshot.step_states[0].cursor == "candidate_publication_completed"

