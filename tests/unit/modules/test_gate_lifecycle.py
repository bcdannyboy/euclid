from __future__ import annotations

from euclid.modules.gate_lifecycle import resolve_scorecard_status


def test_resolve_scorecard_status_blocks_publication_on_robustness_failure() -> None:
    decision = resolve_scorecard_status(
        candidate_admissible=True,
        robustness_status="failed",
        robustness_reason_codes=("leakage_canary_failed",),
        candidate_beats_baseline=True,
        confirmatory_promotion_allowed=True,
        point_score_comparison_status="comparable",
        time_safety_status="passed",
        calibration_status="not_applicable_for_forecast_type",
    )

    assert decision.descriptive_status == "blocked_robustness_failed"
    assert decision.descriptive_reason_codes == (
        "robustness_failed",
        "leakage_canary_failed",
    )
    assert decision.predictive_status == "not_requested"
    assert decision.predictive_reason_codes == ("predictive_not_requested",)


def test_resolve_scorecard_status_defaults_floor_failure_to_no_admissible_reducer() -> (
    None
):
    decision = resolve_scorecard_status(
        candidate_admissible=False,
        robustness_status="passed",
        candidate_beats_baseline=False,
        confirmatory_promotion_allowed=False,
        point_score_comparison_status="comparable",
        time_safety_status="passed",
        calibration_status="not_applicable_for_forecast_type",
    )

    assert decision.descriptive_status == "blocked"
    assert decision.descriptive_reason_codes == (
        "descriptive_gate_failed",
        "no_candidate_survived_search",
    )
    assert decision.predictive_status == "not_requested"
    assert decision.predictive_reason_codes == ("predictive_not_requested",)


def test_resolve_scorecard_status_preserves_codelength_comparability_block() -> None:
    decision = resolve_scorecard_status(
        candidate_admissible=False,
        robustness_status="passed",
        candidate_beats_baseline=False,
        confirmatory_promotion_allowed=False,
        point_score_comparison_status="comparable",
        time_safety_status="passed",
        calibration_status="not_applicable_for_forecast_type",
        descriptive_failure_reason_codes=("codelength_comparability_failed",),
    )

    assert decision.descriptive_status == "blocked_codelength_comparability_failed"
    assert decision.descriptive_reason_codes == ("codelength_comparability_failed",)


def test_predictive_block_preserves_descriptive_pass() -> None:
    decision = resolve_scorecard_status(
        candidate_admissible=True,
        robustness_status="passed",
        candidate_beats_baseline=False,
        confirmatory_promotion_allowed=False,
        point_score_comparison_status="comparable",
        time_safety_status="passed",
        calibration_status="not_applicable_for_forecast_type",
    )

    assert decision.descriptive_status == "passed"
    assert decision.descriptive_reason_codes == ()
    assert decision.predictive_status == "blocked"
    assert decision.predictive_reason_codes == ("baseline_rule_failed",)
