from __future__ import annotations

from euclid.modules.predictive_tests import evaluate_predictive_promotion


def test_statistical_promotion_gate_uses_statsmodels_uncertainty_evidence() -> None:
    result = evaluate_predictive_promotion(
        candidate_losses=(0.8, 0.9, 0.7, 0.8),
        baseline_losses=(1.2, 1.1, 1.0, 1.3),
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.1,
    )

    manifest = result.as_manifest()

    assert result.promotion_allowed is True
    assert manifest["statistical_test_backend"] == (
        "statsmodels_hac_mean_loss_differential"
    )
    assert manifest["confidence_interval_method"] == "newey_west_hac_t_interval"
