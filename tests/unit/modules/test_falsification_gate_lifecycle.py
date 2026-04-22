from __future__ import annotations

from euclid.modules.claims import resolve_claim_publication
from euclid.modules.gate_lifecycle import resolve_scorecard_status


def test_failed_falsification_blocks_predictive_claim_and_downgrades_publication() -> None:
    scorecard = resolve_scorecard_status(
        candidate_admissible=True,
        robustness_status="passed",
        candidate_beats_baseline=True,
        confirmatory_promotion_allowed=True,
        point_score_comparison_status="comparable",
        time_safety_status="passed",
        calibration_status="not_applicable_for_forecast_type",
        falsification_status="failed",
        falsification_reason_codes=("structured_residuals",),
    )

    assert scorecard.descriptive_status == "passed"
    assert scorecard.predictive_status == "blocked"
    assert scorecard.predictive_reason_codes == (
        "falsification_failed",
        "structured_residuals",
    )

    decision = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": scorecard.descriptive_status,
            "descriptive_reason_codes": list(scorecard.descriptive_reason_codes),
            "predictive_status": scorecard.predictive_status,
            "predictive_reason_codes": list(scorecard.predictive_reason_codes),
        }
    )

    assert decision.publication_mode == "candidate_publication"
    assert decision.claim_type == "descriptive_structure"
    assert decision.predictive_support_status == "blocked"
