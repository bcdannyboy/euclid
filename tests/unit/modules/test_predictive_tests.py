from __future__ import annotations

from euclid.modules.predictive_tests import evaluate_predictive_promotion


def test_statistical_promotion_requires_margin_and_uncertainty_evidence() -> None:
    result = evaluate_predictive_promotion(
        candidate_losses=(0.8, 0.9, 0.7, 0.8),
        baseline_losses=(1.2, 1.1, 1.0, 1.3),
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.1,
    )

    assert result.status == "passed"
    assert result.promotion_allowed is True
    assert result.raw_metric_comparison_role == "diagnostic_only"
    assert result.replay_identity.startswith("predictive-promotion:")
    manifest = result.as_manifest()
    assert manifest["schema_name"] == "paired_predictive_test_result@1.0.0"
    assert manifest["statistical_test_backend"] == (
        "statsmodels_hac_mean_loss_differential"
    )
    assert manifest["confidence_interval_method"] == "newey_west_hac_t_interval"


def test_ties_and_insignificant_improvements_downgrade() -> None:
    tie = evaluate_predictive_promotion(
        candidate_losses=(1.0, 1.0, 1.0),
        baseline_losses=(1.0, 1.0, 1.0),
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.01,
    )
    noisy = evaluate_predictive_promotion(
        candidate_losses=(0.99, 1.02, 0.98, 1.01),
        baseline_losses=(1.0, 1.0, 1.0, 1.0),
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
    )

    assert tie.status == "downgraded"
    assert "baseline_tie" in tie.reason_codes
    assert noisy.status == "downgraded"
    assert "insignificant_improvement" in noisy.reason_codes


def test_missing_baseline_unstable_split_leakage_and_calibration_fail_closed() -> None:
    result = evaluate_predictive_promotion(
        candidate_losses=(0.5, 0.5),
        baseline_losses=(),
        split_protocol_id="train_only",
        baseline_id=None,
        practical_margin=0.1,
        calibration_status="failed",
        leakage_status="failed",
    )

    assert result.status == "abstained"
    assert result.promotion_allowed is False
    assert set(result.reason_codes) >= {
        "missing_baseline",
        "unstable_split_protocol",
        "leakage_detected",
        "calibration_failed",
    }


def test_poor_coverage_and_train_only_overfit_abstain() -> None:
    result = evaluate_predictive_promotion(
        candidate_losses=(0.1, 0.1, 0.1),
        baseline_losses=(1.0, 1.0, 1.0),
        split_protocol_id="train_only",
        baseline_id="naive",
        practical_margin=0.1,
        calibration_status="poor_coverage",
    )

    assert result.status == "abstained"
    assert result.promotion_allowed is False
    assert set(result.reason_codes) >= {
        "unstable_split_protocol",
        "calibration_failed",
        "poor_coverage",
    }
