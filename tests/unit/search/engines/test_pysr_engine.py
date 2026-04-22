from __future__ import annotations

from euclid.search.engines.pysr_engine import (
    PySrDiscovery,
    PySrEngine,
    PySrEngineConfig,
    PySrRuntimeUnavailable,
)
from euclid.search.engines.pysr_lowering import PySrHallOfFameRow
from euclid.search.orchestration import SearchEngineOrchestrator, run_search_engines


def test_pysr_engine_returns_lowered_cir_runtime_metadata_and_no_claims() -> None:
    engine = PySrEngine(
        config=PySrEngineConfig(
            binary_operators=("add", "mul"),
            unary_operators=(),
            niterations=5,
            timeout_seconds=0.5,
        ),
        runner=_LinearPySrRunner(),
    )

    result = engine.run(_context())

    assert result.status == "completed"
    assert result.engine_id == "pysr-engine-v1"
    assert result.trace["operator_set"]["binary_operators"] == ["+", "*"]
    assert result.trace["runtime_metadata"]["julia_version"] == "1.10-test"
    assert result.claim_boundary["claim_publication_allowed"] is False
    assert len(result.candidates) == 1
    record = result.candidates[0]
    assert record.proposed_cir is not None
    assert record.proposed_cir.structural_layer.expression_payload is not None
    assert record.candidate_trace["euclid_fit"]["status"] == "converged"
    assert (
        record.candidate_trace["claim_boundary"]["claim_publication_allowed"]
        is False
    )
    assert "search_engine_not_claim_authority" in record.claim_boundary["reason_codes"]


def test_pysr_engine_reports_julia_runtime_unavailable_as_typed_omission() -> None:
    result = PySrEngine(runner=_UnavailablePySrRunner()).run(_context())

    assert result.status == "failed"
    assert result.candidates == ()
    assert result.failure_diagnostics[0].reason_code == "pysr_runtime_unavailable"
    assert result.omission_disclosure["omitted_due_to_runtime_unavailable"] is True


def test_pysr_engine_duplicate_canonical_candidates_are_deduped_by_orchestrator():
    context = _context()
    result = run_search_engines(
        context=context,
        engines=(PySrEngine(runner=_DuplicatePySrRunner()),),
    )

    assert result.engine_runs["pysr-engine-v1"].status == "partial"
    assert len(result.accepted_candidates) == 1
    assert result.duplicate_diagnostics[0].reason_code == "duplicate_canonical_output"
    assert result.replay_identity == run_search_engines(
        context=context,
        engines=(PySrEngine(runner=_DuplicatePySrRunner()),),
    ).replay_identity


def _context():
    rows = tuple(
        _row(
            f"2026-01-0{index + 1}T00:00:00Z",
            target=2.0 * float(index) + 1.0,
            lag_1=float(index),
        )
        for index in range(4)
    )
    return SearchEngineOrchestrator.context_from_rows(
        search_plan_id="pysr-plan",
        search_class="bounded_heuristic",
        random_seed="11",
        proposal_limit=5,
        frontier_axes=("structure_code_bits", "fit_loss"),
        rows=rows,
        feature_names=("lag_1",),
        timeout_seconds=1.0,
        engine_ids=("pysr-engine-v1",),
    )


def _row(event_time: str, *, target: float, lag_1: float) -> dict[str, object]:
    return {
        "event_time": event_time,
        "available_at": event_time,
        "target": target,
        "lag_1": lag_1,
    }


class _LinearPySrRunner:
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


class _UnavailablePySrRunner:
    def discover(self, *, context, config):
        raise PySrRuntimeUnavailable("julia unavailable")


class _DuplicatePySrRunner:
    def discover(self, *, context, config):
        return PySrDiscovery(
            status="partial",
            engine_version="pysr-test",
            hall_of_fame=(
                PySrHallOfFameRow(equation="2.0 * lag_1 + 1.0", complexity=5),
                PySrHallOfFameRow(equation="1.0 + lag_1 * 2.0", complexity=5),
            ),
            trace={"hall_of_fame_rows": 2},
            runtime_metadata={
                "pysr_version": "pysr-test",
                "julia_version": "1.10-test",
            },
            omission_disclosure={"omitted_due_to_timeout": True},
            failure_diagnostics=(
                {
                    "reason_code": "pysr_partial_result",
                    "message": "timeout returned a partial hall of fame",
                    "recoverable": True,
                },
            ),
        )
