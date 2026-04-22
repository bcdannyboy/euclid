from __future__ import annotations

import time

import pytest

from euclid.cir.models import (
    CIRBackendOriginRecord,
    CIRForecastOperator,
    CIRInputSignature,
    CIRModelCodeDecomposition,
    CIRReplayHook,
    CIRReplayHooks,
)
from euclid.cir.normalize import build_cir_candidate_from_expression
from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import BinaryOp, Feature, Parameter
from euclid.search.engine_contracts import (
    EngineCandidateRecord,
    EngineRunResult,
    SearchEngine,
)
from euclid.search.orchestration import (
    EngineRegistry,
    SearchEngineOrchestrator,
    run_search_engines,
)


def test_engine_registry_rejects_duplicate_engine_ids() -> None:
    registry = EngineRegistry()
    registry.register(_StaticEngine("fixture", candidates=()))

    with pytest.raises(ContractValidationError, match="duplicate"):
        registry.register(_StaticEngine("fixture", candidates=()))


def test_orchestrator_handles_crash_timeout_partial_and_malformed_results() -> None:
    candidate = _linear_expression_candidate("candidate")
    valid_record = _engine_candidate("candidate", proposed_cir=candidate)

    result = run_search_engines(
        context=_context(timeout_seconds=0.01),
        engines=(
            _CrashingEngine("crash"),
            _SleepingEngine("slow", sleep_seconds=0.02),
            _StaticEngine(
                "partial",
                status="partial",
                candidates=(valid_record,),
                trace={"partial_reason": "budget_exhausted"},
            ),
            _MalformedEngine("malformed"),
        ),
    )

    assert result.accepted_candidates == (candidate,)
    assert {failure.reason_code for failure in result.failure_diagnostics} == {
        "engine_crash",
        "engine_timeout",
        "malformed_engine_result",
    }
    assert result.engine_runs["partial"].status == "partial"
    assert result.claim_boundary["claim_publication_allowed"] is False


def test_orchestrator_deduplicates_canonical_outputs_and_records_replay_identity() -> None:
    candidate = _linear_expression_candidate("same")
    context = _context(timeout_seconds=1.0)

    first = run_search_engines(
        context=context,
        engines=(
            _StaticEngine(
                "engine-a",
                candidates=(_engine_candidate("same-a", proposed_cir=candidate),),
            ),
            _StaticEngine(
                "engine-b",
                candidates=(_engine_candidate("same-b", proposed_cir=candidate),),
            ),
        ),
    )
    second = run_search_engines(
        context=context,
        engines=(
            _StaticEngine(
                "engine-a",
                candidates=(_engine_candidate("same-a", proposed_cir=candidate),),
            ),
            _StaticEngine(
                "engine-b",
                candidates=(_engine_candidate("same-b", proposed_cir=candidate),),
            ),
        ),
    )

    assert len(first.accepted_candidates) == 1
    assert first.duplicate_diagnostics[0].reason_code == "duplicate_canonical_output"
    assert first.replay_identity == second.replay_identity
    assert first.replay_metadata == second.replay_metadata


def test_orchestrator_records_failed_lowering_without_blocking_other_engines() -> None:
    good = _engine_candidate("good", proposed_cir=_linear_expression_candidate("good"))
    bad = _engine_candidate(
        "bad",
        lowering_kind="unknown_lowering",
        lowerable_payload={"candidate_id": "bad"},
    )

    result = run_search_engines(
        context=_context(timeout_seconds=1.0),
        engines=(_StaticEngine("mixed", candidates=(bad, good)),),
    )

    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.accepted_candidates
    ] == ["good"]
    assert result.failure_diagnostics[0].reason_code == "failed_lowering"
    assert result.engine_runs["mixed"].status == "completed"


def _context(*, timeout_seconds: float):
    return SearchEngineOrchestrator.context_from_rows(
        search_plan_id="plan",
        search_class="bounded_heuristic",
        random_seed="5",
        proposal_limit=4,
        frontier_axes=("structure_code_bits", "fit_loss"),
        rows=(
            {
                "event_time": "2026-01-01T00:00:00Z",
                "available_at": "2026-01-01T00:00:00Z",
                "target": 1.0,
                "lag_1": 0.0,
            },
        ),
        feature_names=("lag_1",),
        timeout_seconds=timeout_seconds,
        engine_ids=("fixture",),
    )


def _engine_candidate(
    candidate_id: str,
    *,
    proposed_cir=None,
    lowering_kind: str = "proposed_cir",
    lowerable_payload=None,
) -> EngineCandidateRecord:
    return EngineCandidateRecord(
        candidate_id=candidate_id,
        engine_id="fixture",
        engine_version="1.0",
        search_class="bounded_heuristic",
        search_space_declaration="fixture-space",
        budget_declaration={"proposal_limit": 4},
        rows_used=("2026-01-01T00:00:00Z",),
        features_used=("lag_1",),
        random_seed="5",
        candidate_trace={"candidate_id": candidate_id},
        omission_disclosure={"omitted": 0},
        claim_boundary={
            "claim_publication_allowed": False,
            "reason_codes": ["search_engine_not_claim_authority"],
        },
        proposed_cir=proposed_cir,
        lowering_kind=lowering_kind,
        lowerable_payload=lowerable_payload or {},
    )


def _linear_expression_candidate(candidate_id: str):
    expression = BinaryOp(
        "add",
        BinaryOp("mul", Parameter("slope"), Feature("lag_1")),
        Parameter("intercept"),
    )
    return build_cir_candidate_from_expression(
        expression=expression,
        cir_family_id="analytic",
        cir_form_class="expression_ir",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("lag_1",),
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=CIRModelCodeDecomposition(
            L_family_bits=1.0,
            L_structure_bits=1.0,
            L_literals_bits=0.0,
            L_params_bits=2.0,
            L_state_bits=0.0,
        ),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="fixture-engine",
            adapter_class="unit_test",
            source_candidate_id=candidate_id,
            search_class="unit_test",
            proposal_rank=0,
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(CIRReplayHook(hook_name="test", hook_ref="unit:test"),)
        ),
    )


class _StaticEngine(SearchEngine):
    def __init__(
        self,
        engine_id: str,
        *,
        candidates: tuple[EngineCandidateRecord, ...],
        status: str = "completed",
        trace=None,
    ) -> None:
        self.engine_id = engine_id
        self.engine_version = "1.0"
        self.status = status
        self.candidates = candidates
        self.trace = trace or {}

    def run(self, context):
        return EngineRunResult(
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            status=self.status,
            candidates=self.candidates,
            failure_diagnostics=(),
            trace=self.trace,
            omission_disclosure={"omitted": 0},
            replay_metadata=context.replay_metadata(),
        )


class _CrashingEngine(SearchEngine):
    engine_version = "1.0"

    def __init__(self, engine_id: str) -> None:
        self.engine_id = engine_id

    def run(self, context):
        raise RuntimeError("boom")


class _SleepingEngine(_CrashingEngine):
    def __init__(self, engine_id: str, *, sleep_seconds: float) -> None:
        super().__init__(engine_id)
        self.sleep_seconds = sleep_seconds

    def run(self, context):
        time.sleep(self.sleep_seconds)
        return EngineRunResult(
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            status="completed",
            candidates=(),
            failure_diagnostics=(),
            trace={"slept": self.sleep_seconds},
            omission_disclosure={"omitted": 0},
            replay_metadata=context.replay_metadata(),
        )


class _MalformedEngine(_CrashingEngine):
    def run(self, context):
        return {"not": "an EngineRunResult"}
