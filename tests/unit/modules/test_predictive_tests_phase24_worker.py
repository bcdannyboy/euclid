from __future__ import annotations

from typing import Any

import euclid.modules.predictive_tests as predictive_tests
from euclid.modules.predictive_tests import evaluate_predictive_promotion


def _paired_identity(pair_count: int) -> dict[str, object]:
    return {
        "row_set_id": "phase24_confirmatory_pairs",
        "origin_ids": [f"origin_{index}" for index in range(pair_count)],
        "horizons": [1],
        "entity_ids": ["series_a"],
    }


def test_nonstationarity_diagnostic_failure_blocks_automatic_promotion() -> None:
    pair_count = 80

    result = evaluate_predictive_promotion(
        candidate_losses=tuple(0.70 for _ in range(pair_count)),
        baseline_losses=tuple(1.10 for _ in range(pair_count)),
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
        paired_stream_identity=_paired_identity(pair_count),
        effective_sample_size=80,
        effective_block_count=80,
        nonstationarity_status="failed",
    )

    assert result.status == "failed"
    assert result.promotion_allowed is False
    assert result.reason_codes == ("nonstationarity_detected",)


def test_facade_downgrades_backend_pass_when_interval_crosses_margin(
    monkeypatch: Any,
) -> None:
    pair_count = 80

    class BackendResult:
        status = "passed"
        promotion_allowed = True
        reason_codes: tuple[str, ...] = ()
        mean_loss_differential = 0.08
        confidence_interval = (0.01, 0.15)
        p_value = 0.02
        confidence_interval_method = "dm_hln_hac_t_interval"

    def backend_pass_with_crossing_interval(**_: Any) -> BackendResult:
        return BackendResult()

    monkeypatch.setattr(
        predictive_tests,
        "_run_declared_paired_test",
        backend_pass_with_crossing_interval,
    )

    result = evaluate_predictive_promotion(
        candidate_losses=tuple(0.70 for _ in range(pair_count)),
        baseline_losses=tuple(1.10 for _ in range(pair_count)),
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
        paired_stream_identity=_paired_identity(pair_count),
        effective_sample_size=80,
        effective_block_count=80,
    )

    assert result.status == "downgraded"
    assert result.promotion_allowed is False
    assert result.reason_codes == ("uncertainty_interval_crosses_margin",)
