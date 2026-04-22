from __future__ import annotations

from euclid.search.engines.decomposition import DecompositionEngine
from euclid.search.engines.latent_state import LatentStateEngine
from euclid.search.engines.planned import PlannedEnginePlaceholder
from euclid.search.engines.sparse_regression import SparseRegressionEngine
from euclid.search.orchestration import SearchEngineOrchestrator, run_search_engines


def test_completed_phase_engines_emit_behavioral_non_claim_candidates() -> None:
    context = SearchEngineOrchestrator.context_from_rows(
        search_plan_id="completed-engine-plan",
        search_class="bounded_heuristic",
        random_seed="3",
        proposal_limit=2,
        frontier_axes=("structure_code_bits",),
        rows=(
            {
                "event_time": "2026-01-01T00:00:00Z",
                "available_at": "2026-01-01T00:00:00Z",
                "target": 1.0,
                "lag_1": 0.0,
                "trend": 0.0,
            },
            {
                "event_time": "2026-01-02T00:00:00Z",
                "available_at": "2026-01-02T00:00:00Z",
                "target": 3.0,
                "lag_1": 1.0,
                "trend": 1.0,
            },
            {
                "event_time": "2026-01-03T00:00:00Z",
                "available_at": "2026-01-03T00:00:00Z",
                "target": 5.0,
                "lag_1": 2.0,
                "trend": 2.0,
            },
            {
                "event_time": "2026-01-04T00:00:00Z",
                "available_at": "2026-01-04T00:00:00Z",
                "target": 7.0,
                "lag_1": 3.0,
                "trend": 3.0,
            },
        ),
        feature_names=("lag_1", "trend"),
        timeout_seconds=1.0,
        engine_ids=(
            "decomposition-engine-v1",
            "latent-state-engine-v1",
            "sparse-regression-engine-v1",
        ),
    )
    expected_rows = tuple(
        f"2026-01-0{index}T00:00:00Z" for index in range(1, 5)
    )

    for engine, trace_key, backend in (
        (
            DecompositionEngine(),
            "decomposition_backend",
            "scipy.linalg.lstsq",
        ),
        (
            LatentStateEngine(),
            "latent_state_backend",
            "sklearn.decomposition.PCA",
        ),
        (
            SparseRegressionEngine(),
            "sparse_regression_backend",
            "sklearn.linear_model.Lasso",
        ),
    ):
        assert not isinstance(engine, PlannedEnginePlaceholder)

        result = engine.run(context)

        assert result.status == "completed"
        assert result.failure_diagnostics == ()
        assert "planned_engine_placeholder" not in result.trace
        assert result.claim_boundary["claim_publication_allowed"] is False
        assert result.trace[trace_key] == backend

        record = result.candidates[0]
        assert record.rows_used == expected_rows
        assert record.proposed_cir is not None
        assert record.claim_boundary["claim_publication_allowed"] is False
        assert record.candidate_trace[trace_key] == backend
        assert record.proposed_cir.evidence_layer.transient_diagnostics[
            "claim_publication_allowed"
        ] is False

        orchestrated = run_search_engines(context=context, engines=(engine,))
        assert len(orchestrated.accepted_candidates) == 1
        assert orchestrated.failure_diagnostics == ()
