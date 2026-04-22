from __future__ import annotations

import importlib.util

import pytest

from euclid.search.engines.pysindy_engine import PySindyEngine, PySindyEngineConfig
from euclid.search.orchestration import SearchEngineOrchestrator, run_search_engines


@pytest.mark.skipif(
    importlib.util.find_spec("pysindy") is None,
    reason="PySINDy package is not installed in this Python environment",
)
def test_real_pysindy_backend_lowers_refits_and_replays_sparse_candidate() -> None:
    context = _context()
    engine = PySindyEngine(
        config=PySindyEngineConfig(
            library_kind="polynomial",
            polynomial_degree=1,
            include_bias=True,
            optimizer_kind="stlsq",
            threshold=0.0,
            refit_enabled=True,
        )
    )

    first = run_search_engines(context=context, engines=(engine,))
    second = run_search_engines(context=context, engines=(engine,))

    assert first.engine_runs["pysindy-engine-v1"].status == "completed"
    assert len(first.accepted_candidates) == 1
    candidate = first.accepted_candidates[0]
    assert candidate.structural_layer.expression_payload is not None
    assert (
        candidate.evidence_layer.backend_origin_record.adapter_id
        == "pysindy-engine-v1"
    )
    assert (
        candidate.evidence_layer.transient_diagnostics["pysindy_trace"][
            "library_kind"
        ]
        == "polynomial"
    )
    assert first.engine_runs["pysindy-engine-v1"].candidates[0].candidate_trace[
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
        search_plan_id="pysindy-integration-plan",
        search_class="bounded_heuristic",
        random_seed="13",
        proposal_limit=3,
        frontier_axes=("structure_code_bits", "fit_loss"),
        rows=rows,
        feature_names=("lag_1",),
        timeout_seconds=30.0,
        engine_ids=("pysindy-engine-v1",),
    )


def _row(event_time: str, *, target: float, lag_1: float) -> dict[str, object]:
    return {
        "event_time": event_time,
        "available_at": event_time,
        "target": target,
        "lag_1": lag_1,
    }
