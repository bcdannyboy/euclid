from __future__ import annotations

from euclid.falsification.residuals import evaluate_residual_diagnostics


def test_structured_residuals_fail_with_replay_evidence() -> None:
    result = evaluate_residual_diagnostics(
        candidate_id="trend_candidate",
        residuals=(-3.0, -2.0, -1.0, 1.0, 2.0, 3.0),
    )

    assert result.status == "failed"
    assert "structured_residuals" in result.reason_codes
    assert result.claim_effect == "downgrade_predictive_claim"
    assert result.replay_identity.startswith("residual-diagnostics:")
    assert result.as_manifest()["schema_name"] == "residual_diagnostics@1.0.0"


def test_nonfinite_or_tiny_residual_sets_abstain() -> None:
    result = evaluate_residual_diagnostics(
        candidate_id="tiny_candidate",
        residuals=(0.1, float("nan")),
    )

    assert result.status == "abstained"
    assert result.claim_effect == "block_claim"
    assert "nonfinite_residual" in result.reason_codes
