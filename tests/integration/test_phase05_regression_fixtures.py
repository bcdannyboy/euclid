from __future__ import annotations

from pathlib import Path

import euclid

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_phase05_regression_fixture_keeps_descriptive_and_predictive_evidence_separate(
    tmp_path: Path,
) -> None:
    manifest_path = (
        PROJECT_ROOT
        / "fixtures/runtime/phase05/descriptive-vs-predictive-disagreement.yaml"
    )

    result = euclid.run_demo_point_evaluation(
        manifest_path=manifest_path,
        output_root=tmp_path / "phase05-disagreement-output",
    )

    candidate_summaries = {
        summary.family_id: summary
        for summary in result.run.workflow_result.candidate_summaries
    }
    predictive_winner = min(
        result.run.workflow_result.candidate_summaries,
        key=lambda summary: (summary.confirmatory_primary_score, summary.family_id),
    )

    assert result.run.summary.selected_family == "linear_trend"
    assert predictive_winner.family_id == "seasonal_naive"
    assert result.run.summary.selected_family != predictive_winner.family_id
    assert (
        candidate_summaries["linear_trend"].description_gain_bits
        > candidate_summaries["seasonal_naive"].description_gain_bits
    )
    assert (
        candidate_summaries["seasonal_naive"].confirmatory_primary_score
        < candidate_summaries["seasonal_naive"].baseline_primary_score
    )
    assert (
        candidate_summaries["linear_trend"].confirmatory_primary_score
        > candidate_summaries["linear_trend"].baseline_primary_score
    )
    assert result.comparison.candidate_beats_baseline is False
    assert result.run.workflow_result.scorecard.manifest.body["descriptive_status"] == (
        "blocked_robustness_failed"
    )
    assert result.run.workflow_result.scorecard.manifest.body["predictive_status"] == (
        "not_requested"
    )
    assert result.run.summary.result_mode == "abstention_only_publication"
    assert result.run.workflow_result.scorecard.manifest.body[
        "descriptive_reason_codes"
    ] == ["robustness_failed", "perturbation_protocol_failed"]
    assert result.run.workflow_result.scorecard.manifest.body[
        "predictive_reason_codes"
    ] == ["predictive_not_requested"]
    assert result.run.workflow_result.claim_card is None
    assert result.run.workflow_result.abstention is not None
    assert result.run.workflow_result.abstention.manifest.body["abstention_type"] == (
        "robustness_failed"
    )
    assert all(
        "confirmatory_primary_score" not in record
        for record in result.run.workflow_result.search_ledger.manifest.body[
            "candidates"
        ]
    )
