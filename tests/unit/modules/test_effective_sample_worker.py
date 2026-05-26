from __future__ import annotations

from euclid.modules.effective_sample import (
    bootstrap_effective_block_count,
    hac_effective_sample_size,
    minimum_pair_policy,
)


def test_hac_effective_sample_size_uses_newey_west_autocorrelation_penalty() -> None:
    result = hac_effective_sample_size((1.0, 2.0, 3.0, 4.0, 5.0), max_lag=1)

    assert result.raw_count == 5
    assert result.max_lag == 1
    assert result.autocorrelation_sum == 0.4
    assert result.n_eff == 2.777778
    assert result.as_manifest() == {
        "method": "hac_newey_west_effective_sample_size_v1",
        "raw_count": 5,
        "max_lag": 1,
        "autocorrelation_sum": 0.4,
        "n_eff": 2.777778,
    }


def test_bootstrap_effective_block_count_counts_only_complete_blocks() -> None:
    result = bootstrap_effective_block_count(raw_count=59, block_length=7)

    assert result.raw_count == 59
    assert result.block_length == 7
    assert result.effective_block_count == 8
    assert result.as_manifest() == {
        "method": "stationary_block_bootstrap_effective_block_count_v1",
        "raw_count": 59,
        "block_length": 7,
        "effective_block_count": 8,
    }


def test_minimum_pair_policy_abstains_below_effective_sample_threshold() -> None:
    decision = minimum_pair_policy(
        n_eff=24.999,
        raw_pair_count=80,
        declared_test_id="diebold_mariano_hln_v1",
    )

    assert decision.status == "abstained"
    assert decision.promotion_allowed is False
    assert decision.reason_codes == ("insufficient_effective_sample_size",)
    assert decision.as_manifest()["thresholds"]["minimum_n_eff"] == 25


def test_minimum_pair_policy_abstains_for_insufficient_bootstrap_blocks() -> None:
    decision = minimum_pair_policy(
        n_eff=60.0,
        raw_pair_count=49,
        declared_test_id="paired_stationary_block_bootstrap_v1",
        block_length=7,
    )

    assert decision.status == "abstained"
    assert decision.promotion_allowed is False
    assert decision.effective_block_count == 7
    assert decision.reason_codes == ("insufficient_effective_block_count",)
    assert decision.as_manifest()["thresholds"]["minimum_effective_block_count"] == 8


def test_minimum_pair_policy_marks_mid_range_effective_sample_human_review_only() -> None:
    decision = minimum_pair_policy(
        n_eff=32.0,
        raw_pair_count=64,
        declared_test_id="diebold_mariano_hln_v1",
    )

    assert decision.status == "human_review_only"
    assert decision.promotion_allowed is False
    assert decision.reason_codes == ("minimum_effective_sample_requires_human_review",)


def test_minimum_pair_policy_allows_promotion_when_effective_information_is_sufficient() -> None:
    decision = minimum_pair_policy(
        n_eff=50.0,
        raw_pair_count=64,
        declared_test_id="paired_stationary_block_bootstrap_v1",
        block_length=8,
    )

    assert decision.status == "passed"
    assert decision.promotion_allowed is True
    assert decision.effective_block_count == 8
    assert decision.reason_codes == ()
    assert decision.as_manifest() == {
        "status": "passed",
        "promotion_allowed": True,
        "reason_codes": [],
        "raw_pair_count": 64,
        "n_eff": 50.0,
        "declared_test_id": "paired_stationary_block_bootstrap_v1",
        "block_length": 8,
        "effective_block_count": 8,
        "thresholds": {
            "minimum_n_eff": 25,
            "human_review_n_eff": 50,
            "minimum_effective_block_count": 8,
        },
    }
