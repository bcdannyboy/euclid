from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.search.engine_contracts import (
    EngineCandidateRecord,
    EngineFailureDiagnostic,
    EngineInputContext,
    EngineRunResult,
    RowsFeaturesAccess,
)


def test_engine_context_declares_rows_features_seed_budget_and_claim_boundary() -> None:
    access = RowsFeaturesAccess.from_rows(
        rows=(
            {
                "event_time": "2026-01-01T00:00:00Z",
                "available_at": "2026-01-01T00:00:00Z",
                "target": 1.0,
                "lag_1": 0.0,
            },
        ),
        feature_names=("lag_1",),
    )
    context = EngineInputContext(
        search_plan_id="search-plan",
        search_class="bounded_heuristic",
        random_seed="17",
        proposal_limit=5,
        frontier_axes=("structure_code_bits", "fit_loss"),
        rows_features_access=access,
        timeout_seconds=0.25,
        engine_ids=("native-bounded",),
    )

    assert context.row_count == 1
    assert context.feature_names == ("lag_1",)
    assert context.claim_boundary == {
        "claim_publication_allowed": False,
        "reason_codes": ["search_engine_not_claim_authority"],
    }
    replay = context.replay_metadata()
    assert replay["random_seed"] == "17"
    assert replay["row_fingerprints"][0]["event_time"] == "2026-01-01T00:00:00Z"
    assert "target_values" not in replay


def test_engine_candidate_record_blocks_direct_claim_publication() -> None:
    with pytest.raises(ContractValidationError, match="claim"):
        EngineCandidateRecord(
            candidate_id="bad-claim",
            engine_id="external-engine",
            engine_version="1.0",
            search_class="bounded_heuristic",
            search_space_declaration="fixture-space",
            budget_declaration={"proposal_limit": 1},
            rows_used=("2026-01-01T00:00:00Z",),
            features_used=("lag_1",),
            random_seed="0",
            candidate_trace={"step": "unsafe"},
            omission_disclosure={"omitted": 0},
            claim_boundary={"claim_publication_allowed": True},
            published_claim_payload={"claim": "unsafe direct claim"},
        )


def test_engine_run_result_requires_typed_status_and_failure_diagnostics() -> None:
    with pytest.raises(ContractValidationError, match="status"):
        EngineRunResult(
            engine_id="native",
            engine_version="1.0",
            status="surprising",
            candidates=(),
            failure_diagnostics=(),
            trace={"events": []},
            omission_disclosure={"omitted": 0},
            replay_metadata={"random_seed": "0"},
        )

    result = EngineRunResult(
        engine_id="native",
        engine_version="1.0",
        status="failed",
        candidates=(),
        failure_diagnostics=(
            EngineFailureDiagnostic(
                engine_id="native",
                reason_code="engine_crash",
                message="boom",
                recoverable=True,
            ),
        ),
        trace={"events": ["crashed"]},
        omission_disclosure={"omitted": 1},
        replay_metadata={"random_seed": "0"},
    )

    assert result.failure_diagnostics[0].reason_code == "engine_crash"
    assert result.claim_boundary["claim_publication_allowed"] is False
