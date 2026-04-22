from __future__ import annotations

import time

from euclid.runtime.parallel import (
    ParallelExecutionPolicy,
    ParallelWorkItem,
    run_replay_safe_parallel,
)


def test_parallel_runtime_is_deterministic_across_worker_counts() -> None:
    items = (
        ParallelWorkItem(item_id="c", payload={"value": 3}),
        ParallelWorkItem(item_id="a", payload={"value": 1}),
        ParallelWorkItem(item_id="b", payload={"value": 2}),
    )

    serial = run_replay_safe_parallel(
        items,
        worker=lambda item: {"candidate_id": item.item_id, "score": item.payload["value"]},
        policy=ParallelExecutionPolicy(max_workers=1, timeout_seconds=1.0),
    )
    parallel = run_replay_safe_parallel(
        items,
        worker=lambda item: {"candidate_id": item.item_id, "score": item.payload["value"]},
        policy=ParallelExecutionPolicy(max_workers=3, timeout_seconds=1.0),
    )

    assert serial.status == "completed"
    assert parallel.status == "completed"
    assert serial.outputs == parallel.outputs
    assert serial.replay_identity == parallel.replay_identity
    assert parallel.policy["aggregation_rule"] == "sort_by_item_id"


def test_parallel_runtime_records_worker_failure_diagnostics_without_reordering_successes(
) -> None:
    items = (
        ParallelWorkItem(item_id="ok-2", payload=2),
        ParallelWorkItem(item_id="bad", payload=0),
        ParallelWorkItem(item_id="ok-1", payload=1),
    )

    def worker(item: ParallelWorkItem):
        if item.item_id == "bad":
            raise ValueError("boom")
        return {"value": item.payload}

    result = run_replay_safe_parallel(
        items,
        worker=worker,
        policy=ParallelExecutionPolicy(max_workers=3, timeout_seconds=1.0),
    )

    assert result.status == "degraded"
    assert [output.item_id for output in result.outputs] == ["ok-1", "ok-2"]
    assert result.diagnostics[0].reason_code == "worker_exception"
    assert result.diagnostics[0].recoverable is True


def test_parallel_runtime_enforces_timeout_as_recoverable_degradation() -> None:
    items = (ParallelWorkItem(item_id="slow", payload=None),)

    result = run_replay_safe_parallel(
        items,
        worker=lambda _item: (time.sleep(0.05), "late")[1],
        policy=ParallelExecutionPolicy(max_workers=1, timeout_seconds=0.001),
    )

    assert result.status == "timeout"
    assert result.outputs == ()
    assert result.diagnostics[0].reason_code == "worker_timeout"
    assert result.diagnostics[0].details["timeout_seconds"] == 0.001
