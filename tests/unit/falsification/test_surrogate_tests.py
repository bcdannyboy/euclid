from __future__ import annotations

from euclid.falsification.surrogate_tests import (
    evaluate_parameter_stability,
    evaluate_surrogate_residual_test,
)


def test_parameter_instability_fails() -> None:
    result = evaluate_parameter_stability(
        candidate_id="unstable_parametric_candidate",
        window_parameters=(
            {"alpha": 1.0, "beta": 0.5},
            {"alpha": 1.8, "beta": 0.52},
            {"alpha": 0.7, "beta": 0.49},
        ),
        max_relative_range=0.25,
    )

    assert result.status == "failed"
    assert result.reason_codes == ("parameter_instability",)
    assert result.claim_effect == "downgrade_predictive_claim"
    assert result.replay_identity.startswith("parameter-stability:")


def test_surrogate_residual_test_detects_structured_residuals() -> None:
    result = evaluate_surrogate_residual_test(
        candidate_id="structured_candidate",
        observed_statistic=0.9,
        surrogate_statistics=(0.1, 0.2, 0.3, 0.4),
        max_p_value=0.4,
    )

    assert result.status == "failed"
    assert "structured_residuals_vs_surrogate" in result.reason_codes
    assert result.monte_carlo_p_value <= 0.4
    assert result.replay_identity.startswith("surrogate-residual-test:")
