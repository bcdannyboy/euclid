from __future__ import annotations

from euclid.search.engines.pysr_engine import (
    PySrDiscovery,
    PySrEngine,
    PySrEngineConfig,
)
from euclid.search.engines.pysr_lowering import PySrHallOfFameRow
from euclid.search.orchestration import SearchEngineOrchestrator, run_search_engines


def test_pysr_backend_lowers_hall_of_fame_refits_and_blocks_direct_publication():
    context = _context()
    engine = PySrEngine(
        config=PySrEngineConfig(
            binary_operators=("add", "mul"),
            unary_operators=(),
            niterations=5,
            timeout_seconds=0.5,
            refit_enabled=True,
        ),
        runner=_DeterministicPySrRunner(),
    )

    first = run_search_engines(context=context, engines=(engine,))
    second = run_search_engines(context=context, engines=(engine,))

    assert first.engine_runs["pysr-engine-v1"].status == "completed"
    assert len(first.accepted_candidates) == 1
    candidate = first.accepted_candidates[0]
    assert candidate.structural_layer.expression_payload is not None
    assert candidate.evidence_layer.backend_origin_record.adapter_id == "pysr-engine-v1"
    assert candidate.evidence_layer.transient_diagnostics["pysr_trace"][
        "runtime_metadata"
    ]["julia_version"] == "1.10-test"
    assert first.engine_runs["pysr-engine-v1"].candidates[0].candidate_trace[
        "euclid_fit"
    ]["status"] == "converged"
    assert first.replay_identity == second.replay_identity
    assert first.claim_boundary["claim_publication_allowed"] is False


def _context():
    rows = tuple(
        _row(
            f"2026-01-{index + 1:02d}T00:00:00Z",
            target=2.0 * float(index) + 1.0,
            lag_1=float(index),
        )
        for index in range(1, 8)
    )
    return SearchEngineOrchestrator.context_from_rows(
        search_plan_id="pysr-integration-plan",
        search_class="bounded_heuristic",
        random_seed="17",
        proposal_limit=3,
        frontier_axes=("structure_code_bits", "fit_loss"),
        rows=rows,
        feature_names=("lag_1",),
        timeout_seconds=3.0,
        engine_ids=("pysr-engine-v1",),
    )


def _row(event_time: str, *, target: float, lag_1: float) -> dict[str, object]:
    return {
        "event_time": event_time,
        "available_at": event_time,
        "target": target,
        "lag_1": lag_1,
    }


class _DeterministicPySrRunner:
    def discover(self, *, context, config):
        return PySrDiscovery(
            status="completed",
            engine_version="pysr-test",
            hall_of_fame=(
                PySrHallOfFameRow(
                    equation="2.0 * lag_1 + 1.0",
                    complexity=5,
                    loss=0.0,
                    score=1.0,
                ),
            ),
            trace={"hall_of_fame_rows": 1},
            runtime_metadata={
                "pysr_version": "pysr-test",
                "symbolic_regression_jl_version": "test",
                "julia_version": "1.10-test",
            },
            omission_disclosure={"omitted_by_pareto_limit": 0},
            failure_diagnostics=(),
        )
