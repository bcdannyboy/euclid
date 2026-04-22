from __future__ import annotations

from euclid.cir.models import (
    CIRBackendOriginRecord,
    CIRForecastOperator,
    CIRInputSignature,
    CIRModelCodeDecomposition,
    CIRReplayHook,
    CIRReplayHooks,
)
from euclid.cir.normalize import build_cir_candidate_from_expression
from euclid.expr.ast import Feature, Literal, NaryOp
from euclid.expr.serialization import expression_to_dict
from euclid.search import backends
from euclid.search.engines.egraph import EGraphEngine, StaticRewriteCandidateProvider
from euclid.search.orchestration import SearchEngineOrchestrator, run_search_engines


def test_egraph_engine_rewrites_cir_and_emits_replay_evidence() -> None:
    x = Feature("x")
    source = _candidate(
        NaryOp("mul", (NaryOp("add", (x, Literal(0))), Literal(1))),
        source_candidate_id="redundant_expression",
    )
    context = _context()
    engine = EGraphEngine(candidate_provider=StaticRewriteCandidateProvider((source,)))

    result = engine.run(context)

    assert result.status == "completed"
    assert len(result.candidates) == 1
    record = result.candidates[0]
    assert record.proposed_cir is not None
    payload = record.proposed_cir.structural_layer.expression_payload
    assert payload is not None
    assert payload.expression_tree == expression_to_dict(x)
    assert record.candidate_trace["rewrite_trace"]["status"] == "completed"
    equivalence = record.candidate_trace["rewrite_trace"]["equivalence_evidence"]
    assert equivalence["status"] == "verified"
    assert record.claim_boundary["claim_publication_allowed"] is False
    assert result.claim_boundary["claim_publication_allowed"] is False


def test_egraph_engine_duplicate_outputs_are_deduped_by_orchestrator() -> None:
    x = Feature("x")
    source_a = _candidate(NaryOp("add", (x, Literal(0))), source_candidate_id="a")
    source_b = _candidate(NaryOp("mul", (x, Literal(1))), source_candidate_id="b")
    context = _context()
    engine = EGraphEngine(
        candidate_provider=StaticRewriteCandidateProvider((source_a, source_b))
    )

    result = run_search_engines(context=context, engines=(engine,))

    assert len(result.accepted_candidates) == 1
    assert len(result.duplicate_diagnostics) == 1
    assert result.duplicate_diagnostics[0].reason_code == "duplicate_canonical_output"
    assert result.replay_identity.startswith("sha256:")


def test_sort_only_equality_saturation_is_not_a_production_path() -> None:
    assert not hasattr(backends, "_equality_extractor_sort_key")


def _context():
    return SearchEngineOrchestrator.context_from_rows(
        search_plan_id="rewrite-plan",
        search_class="equality_saturation_heuristic",
        random_seed="17",
        proposal_limit=4,
        frontier_axes=("structure_code_bits", "fit_loss"),
        rows=(
            {
                "event_time": "2026-01-01T00:00:00Z",
                "available_at": "2026-01-01T00:00:00Z",
                "target": 1.0,
                "x": 1.0,
            },
            {
                "event_time": "2026-01-02T00:00:00Z",
                "available_at": "2026-01-02T00:00:00Z",
                "target": 2.0,
                "x": 2.0,
            },
        ),
        feature_names=("x",),
        timeout_seconds=5.0,
        engine_ids=("egraph-engine-v1",),
    )


def _candidate(expression, *, source_candidate_id: str):
    return build_cir_candidate_from_expression(
        expression=expression,
        cir_family_id="analytic",
        cir_form_class="expression_ir",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("x",),
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="point_forecast",
            horizon=1,
        ),
        model_code_decomposition=CIRModelCodeDecomposition(
            L_family_bits=1.0,
            L_structure_bits=10.0,
            L_literals_bits=1.0,
            L_params_bits=0.0,
            L_state_bits=0.0,
        ),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="fixture-expression-adapter",
            adapter_class="fixture",
            source_candidate_id=source_candidate_id,
            search_class="bounded_heuristic",
            backend_family="fixture",
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(
                CIRReplayHook(
                    hook_name="fixture_expression",
                    hook_ref=source_candidate_id,
                ),
            )
        ),
    )
