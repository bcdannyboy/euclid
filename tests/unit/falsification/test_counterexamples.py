from __future__ import annotations

from euclid.falsification.counterexamples import discover_counterexamples


def test_counterexample_and_extrapolation_failure_are_reported() -> None:
    result = discover_counterexamples(
        candidate_id="local_law",
        cases=(
            {
                "case_id": "holdout_near",
                "prediction": 10.0,
                "observed": 10.2,
                "domain_valid": True,
                "extrapolation_distance": 0.0,
            },
            {
                "case_id": "holdout_far",
                "prediction": 10.0,
                "observed": 17.0,
                "domain_valid": True,
                "extrapolation_distance": 2.5,
            },
        ),
        error_tolerance=1.0,
        max_extrapolation_distance=1.0,
    )

    assert result.status == "failed"
    assert set(result.reason_codes) >= {
        "counterexample_discovered",
        "extrapolation_failure",
    }
    assert result.claim_effect == "downgrade_predictive_claim"
    assert result.counterexamples[0]["case_id"] == "holdout_far"
    assert result.replay_identity.startswith("counterexamples:")


def test_domain_violation_blocks_all_claims() -> None:
    result = discover_counterexamples(
        candidate_id="domain_law",
        cases=(
            {
                "case_id": "negative_mass",
                "prediction": -1.0,
                "observed": 1.0,
                "domain_valid": False,
            },
        ),
    )

    assert result.status == "failed"
    assert "domain_violation" in result.reason_codes
    assert result.claim_effect == "block_claim"
