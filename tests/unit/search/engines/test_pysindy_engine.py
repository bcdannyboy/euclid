from __future__ import annotations

import time

from euclid.search.engines.pysindy_engine import (
    PySindyDiscovery,
    PySindyEngine,
    PySindyEngineConfig,
    PySindyRuntimeUnavailable,
)
from euclid.search.engines.pysindy_lowering import (
    PySindyDiscoveredEquation,
    PySindyTerm,
)
from euclid.search.orchestration import SearchEngineOrchestrator, run_search_engines


def test_pysindy_engine_returns_lowered_cir_fit_evidence_and_no_claims() -> None:
    engine = PySindyEngine(
        config=PySindyEngineConfig(polynomial_degree=1, threshold=0.0),
        runner=_LinearPySindyRunner(),
    )

    result = engine.run(_context(timeout_seconds=1.0))

    assert result.status == "completed"
    assert result.engine_id == "pysindy-engine-v1"
    assert result.claim_boundary["claim_publication_allowed"] is False
    assert len(result.candidates) == 1
    record = result.candidates[0]
    assert record.proposed_cir is not None
    assert record.claim_boundary["claim_publication_allowed"] is False
    assert record.proposed_cir.structural_layer.expression_payload is not None
    assert record.candidate_trace["euclid_fit"]["status"] == "converged"
    assert record.candidate_trace["pysindy_trace"]["support_mask"] == [True, True]
    assert (
        record.candidate_trace["claim_boundary"]["claim_publication_allowed"]
        is False
    )
    assert "search_engine_not_claim_authority" in record.claim_boundary["reason_codes"]


def test_pysindy_engine_reports_runtime_unavailable_as_typed_omission() -> None:
    engine = PySindyEngine(runner=_UnavailablePySindyRunner())

    result = engine.run(_context(timeout_seconds=1.0))

    assert result.status == "failed"
    assert result.candidates == ()
    assert result.failure_diagnostics[0].reason_code == "pysindy_runtime_unavailable"
    assert result.omission_disclosure["omitted_due_to_runtime_unavailable"] is True


def test_pysindy_engine_crash_timeout_partial_and_duplicate_canonical_paths() -> None:
    context = _context(timeout_seconds=0.01)

    crashed = run_search_engines(
        context=context,
        engines=(PySindyEngine(runner=_CrashingPySindyRunner()),),
    )
    assert crashed.failure_diagnostics[0].reason_code == "engine_crash"

    timed_out = run_search_engines(
        context=context,
        engines=(PySindyEngine(runner=_SlowPySindyRunner()),),
    )
    assert timed_out.engine_runs["pysindy-engine-v1"].status == "timeout"
    assert timed_out.failure_diagnostics[0].reason_code == "engine_timeout"

    partial = run_search_engines(
        context=_context(timeout_seconds=1.0),
        engines=(PySindyEngine(runner=_PartialDuplicatePySindyRunner()),),
    )

    assert partial.engine_runs["pysindy-engine-v1"].status == "partial"
    assert len(partial.accepted_candidates) == 1
    assert partial.duplicate_diagnostics[0].reason_code == "duplicate_canonical_output"
    assert partial.failure_diagnostics[0].reason_code == "pysindy_partial_result"
    assert partial.replay_identity == run_search_engines(
        context=_context(timeout_seconds=1.0),
        engines=(PySindyEngine(runner=_PartialDuplicatePySindyRunner()),),
    ).replay_identity


def _context(*, timeout_seconds: float):
    rows = tuple(
        _row(
            f"2026-01-0{index + 1}T00:00:00Z",
            target=2.0 * float(index) + 1.0,
            lag_1=float(index),
        )
        for index in range(4)
    )
    return SearchEngineOrchestrator.context_from_rows(
        search_plan_id="pysindy-plan",
        search_class="bounded_heuristic",
        random_seed="7",
        proposal_limit=5,
        frontier_axes=("structure_code_bits", "fit_loss"),
        rows=rows,
        feature_names=("lag_1",),
        timeout_seconds=timeout_seconds,
        engine_ids=("pysindy-engine-v1",),
    )


def _row(event_time: str, *, target: float, lag_1: float) -> dict[str, object]:
    return {
        "event_time": event_time,
        "available_at": event_time,
        "target": target,
        "lag_1": lag_1,
    }


class _LinearPySindyRunner:
    def discover(self, *, context, config):
        return PySindyDiscovery(
            status="completed",
            engine_version="pysindy-test",
            equations=(
                PySindyDiscoveredEquation(
                    output_name="target",
                    terms=(
                        PySindyTerm("1", 1.0),
                        PySindyTerm("lag_1", 2.0),
                    ),
                    equation_text="1.000 1 + 2.000 lag_1",
                ),
            ),
            trace={
                "pysindy_version": "pysindy-test",
                "support_mask": [True, True],
                "coefficients": [1.0, 2.0],
            },
            omission_disclosure={"omitted_by_sparsity": 0},
            failure_diagnostics=(),
        )


class _UnavailablePySindyRunner:
    def discover(self, *, context, config):
        raise PySindyRuntimeUnavailable("pysindy import failed")


class _CrashingPySindyRunner:
    def discover(self, *, context, config):
        raise RuntimeError("boom")


class _SlowPySindyRunner(_LinearPySindyRunner):
    def discover(self, *, context, config):
        time.sleep(0.02)
        return super().discover(context=context, config=config)


class _PartialDuplicatePySindyRunner:
    def discover(self, *, context, config):
        equation = PySindyDiscoveredEquation(
            output_name="target",
            terms=(PySindyTerm("1", 1.0), PySindyTerm("lag_1", 2.0)),
            equation_text="1.000 1 + 2.000 lag_1",
        )
        return PySindyDiscovery(
            status="partial",
            engine_version="pysindy-test",
            equations=(equation, equation),
            trace={"support_mask": [True, True], "coefficients": [1.0, 2.0]},
            omission_disclosure={"omitted_due_to_budget": True},
            failure_diagnostics=(
                {
                    "reason_code": "pysindy_partial_result",
                    "message": "budget exhausted after partial support recovery",
                    "recoverable": True,
                },
            ),
        )
