from __future__ import annotations

import pytest

from euclid.modules.predictive_tests import (
    evaluate_predictive_promotion,
    run_declared_predictive_test,
)


def _losses(
    *,
    pair_count: int,
    candidate_loss: float = 0.70,
    baseline_loss: float = 1.05,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        tuple(candidate_loss for _ in range(pair_count)),
        tuple(baseline_loss for _ in range(pair_count)),
    )


def test_statistical_promotion_requires_margin_and_uncertainty_evidence() -> None:
    result = evaluate_predictive_promotion(
        candidate_losses=(0.8, 0.9, 0.7, 0.8),
        baseline_losses=(1.2, 1.1, 1.0, 1.3),
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.1,
        declared_test_id="diebold_mariano_hln_v1",
        paired_stream_identity={
            "row_set_id": "confirmatory_pairs_v1",
            "origin_ids": [f"origin_{index}" for index in range(4)],
            "horizons": [1],
            "entity_ids": ["entity_a"],
        },
        effective_sample_size=80,
        effective_block_count=80,
    )

    assert result.status == "passed"
    assert result.promotion_allowed is True
    assert result.raw_metric_comparison_role == "diagnostic_only"
    assert result.replay_identity.startswith("predictive-promotion:")
    manifest = result.as_manifest()
    assert manifest["schema_name"] == "paired_predictive_test_result@1.0.0"
    assert manifest["declared_test_id"] == "diebold_mariano_hln_v1"
    assert manifest["actual_test_id"] == "diebold_mariano_hln_v1"
    assert manifest["statistical_test_backend"] == "diebold_mariano_hln_v1"
    assert manifest["confidence_interval_method"] == "dm_hln_hac_t_interval"


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


def test_separately_averaged_losses_cannot_be_promoted_without_paired_stream() -> None:
    candidate_losses, baseline_losses = _losses(pair_count=80)

    result = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
    )

    assert result.status == "abstained"
    assert result.promotion_allowed is False
    assert "unpaired_loss_stream" in result.reason_codes


def test_one_element_paired_stream_always_abstains() -> None:
    candidate_losses, baseline_losses = _losses(pair_count=1)

    result = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
        paired_stream_identity={
            "row_set_id": "confirmatory_pairs_v1",
            "origin_ids": ["origin_0"],
            "horizons": [1],
            "entity_ids": ["entity_a"],
        },
    )

    assert result.status == "abstained"
    assert result.promotion_allowed is False
    assert "insufficient_paired_count" in result.reason_codes


def test_declared_dm_hln_test_id_is_public_result_identity() -> None:
    candidate_losses, baseline_losses = _losses(pair_count=80)

    result = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
        declared_test_id="diebold_mariano_hln_v1",
        paired_stream_identity={
            "row_set_id": "confirmatory_pairs_v1",
            "origin_ids": [f"origin_{index}" for index in range(80)],
            "horizons": [1],
            "entity_ids": ["entity_a"],
        },
        effective_sample_size=80,
        effective_block_count=80,
    )

    manifest = result.as_manifest()
    assert manifest["declared_test_id"] == "diebold_mariano_hln_v1"
    assert manifest["statistical_test_backend"] == "diebold_mariano_hln_v1"
    assert manifest["confidence_interval_method"] != "newey_west_hac_t_interval"
    assert "hln_small_sample_correction" in manifest


def test_declared_stationary_block_bootstrap_records_test_configuration() -> None:
    candidate_losses, baseline_losses = _losses(pair_count=96)

    result = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
        declared_test_id="paired_stationary_block_bootstrap_v1",
        paired_stream_identity={
            "row_set_id": "confirmatory_pairs_v1",
            "origin_ids": [f"origin_{index}" for index in range(96)],
            "horizons": [1],
            "entity_ids": ["entity_a"],
        },
        effective_sample_size=96,
        effective_block_count=12,
        block_bootstrap_config={
            "block_length": 8,
            "seed": 20260426,
            "bootstrap_count": 999,
        },
    )

    manifest = result.as_manifest()
    assert manifest["declared_test_id"] == "paired_stationary_block_bootstrap_v1"
    assert manifest["block_bootstrap"]["block_length"] == 8
    assert manifest["block_bootstrap"]["seed"] == 20260426
    assert manifest["block_bootstrap"]["bootstrap_count"] == 999


def test_unsupported_declared_test_id_fails_closed() -> None:
    candidate_losses, baseline_losses = _losses(pair_count=80)

    result = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
        declared_test_id="made_up_predictive_test_v1",
        paired_stream_identity={
            "row_set_id": "confirmatory_pairs_v1",
            "origin_ids": [f"origin_{index}" for index in range(80)],
            "horizons": [1],
            "entity_ids": ["entity_a"],
        },
        effective_sample_size=80,
        effective_block_count=80,
    )

    assert result.status == "abstained"
    assert result.promotion_allowed is False
    assert "unsupported_declared_test_id" in result.reason_codes


def test_gw_conditional_test_requires_declared_instruments_for_promotion() -> None:
    candidate_losses, baseline_losses = _losses(pair_count=80)

    result = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
        declared_test_id="giacomini_white_conditional_predictive_ability_v1",
        paired_stream_identity={
            "row_set_id": "confirmatory_pairs_v1",
            "origin_ids": [f"origin_{index}" for index in range(80)],
            "horizons": [1],
            "entity_ids": ["entity_a"],
        },
        effective_sample_size=80,
        effective_block_count=80,
        conditional_instrument_declarations=(),
    )

    assert result.status == "abstained"
    assert result.promotion_allowed is False
    assert "missing_conditional_instrument_declarations" in result.reason_codes


@pytest.mark.parametrize(
    (
        "declared_test_id",
        "effective_sample_size",
        "effective_block_count",
        "expected_status",
        "expected_reason",
        "promotion_allowed",
    ),
    (
        (
            "diebold_mariano_hln_v1",
            24,
            24,
            "abstained",
            "insufficient_effective_sample_size",
            False,
        ),
        (
            "paired_stationary_block_bootstrap_v1",
            60,
            7,
            "abstained",
            "insufficient_effective_block_count",
            False,
        ),
        (
            "diebold_mariano_hln_v1",
            40,
            40,
            "human_review_only",
            "minimum_effective_sample_requires_human_review",
            False,
        ),
        (
            "paired_stationary_block_bootstrap_v1",
            60,
            10,
            "passed",
            None,
            True,
        ),
    ),
)
def test_effective_sample_and_block_count_threshold_statuses(
    declared_test_id: str,
    effective_sample_size: int,
    effective_block_count: int,
    expected_status: str,
    expected_reason: str | None,
    promotion_allowed: bool,
) -> None:
    candidate_losses, baseline_losses = _losses(pair_count=80)

    result = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
        declared_test_id=declared_test_id,
        paired_stream_identity={
            "row_set_id": "confirmatory_pairs_v1",
            "origin_ids": [f"origin_{index}" for index in range(80)],
            "horizons": [1],
            "entity_ids": ["entity_a"],
        },
        effective_sample_size=effective_sample_size,
        effective_block_count=effective_block_count,
        block_bootstrap_config={
            "block_length": 8,
            "seed": 20260426,
            "bootstrap_count": 499,
        },
    )

    assert result.status == expected_status
    assert result.promotion_allowed is promotion_allowed
    if expected_reason is not None:
        assert expected_reason in result.reason_codes
    manifest = result.as_manifest()
    assert manifest["minimum_pair_policy"] == {
        "minimum_effective_sample_size": 25,
        "human_review_effective_sample_size": 50,
        "minimum_effective_block_count": 8,
    }
    assert manifest["effective_sample_size"] == effective_sample_size
    assert manifest["effective_block_count"] == effective_block_count


def test_interval_crossing_practical_margin_blocks_automatic_promotion() -> None:
    differentials = tuple([0.4] * 40 + [-0.2] * 40)

    result = evaluate_predictive_promotion(
        candidate_losses=tuple(1.0 for _ in differentials),
        baseline_losses=tuple(1.0 + differential for differential in differentials),
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
        declared_test_id="diebold_mariano_hln_v1",
        paired_stream_identity=_paired_identity(len(differentials)),
        effective_sample_size=80,
        effective_block_count=80,
    )

    assert result.mean_loss_differential > result.practical_margin
    assert result.confidence_interval is not None
    assert result.confidence_interval[0] <= result.practical_margin
    assert result.status in {"downgraded", "abstained"}
    assert result.promotion_allowed is False
    assert "uncertainty_interval_crosses_margin" in result.reason_codes


def test_nonstationarity_diagnostic_failure_blocks_automatic_promotion() -> None:
    candidate_losses, baseline_losses = _losses(pair_count=80)

    result = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
        declared_test_id="diebold_mariano_hln_v1",
        paired_stream_identity=_paired_identity(80),
        effective_sample_size=80,
        effective_block_count=80,
        nonstationarity_diagnostic={
            "status": "failed",
            "reason_codes": ["structural_break_detected"],
        },
    )

    assert result.status in {"abstained", "downgraded"}
    assert result.promotion_allowed is False
    assert "nonstationarity_detected" in result.reason_codes


def test_unresolved_instability_blocks_promotion_until_lane_handles_it() -> None:
    candidate_losses, baseline_losses = _losses(pair_count=80)
    diagnostic = {
        "schema_name": "stability_diagnostic_artifact@1.0.0",
        "status": "failed",
        "reason_codes": [
            "stability_test_failed",
            "instability_evidence_unresolved",
        ],
    }

    unresolved = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
        declared_test_id="diebold_mariano_hln_v1",
        paired_stream_identity=_paired_identity(80),
        effective_sample_size=80,
        effective_block_count=80,
        nonstationarity_diagnostic=diagnostic,
    )
    handled = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.05,
        declared_test_id="diebold_mariano_hln_v1",
        paired_stream_identity=_paired_identity(80),
        effective_sample_size=80,
        effective_block_count=80,
        nonstationarity_diagnostic={
            **diagnostic,
            "nonstationarity_handling": {
                "status": "passed",
                "lane_id": "state_space_local_level_v1",
                "artifact_ref": {
                    "schema_name": "state_space_artifact@1.0.0",
                    "object_id": "state-space-fit-001",
                },
            },
        },
    )

    assert unresolved.promotion_allowed is False
    assert "nonstationarity_detected" in unresolved.reason_codes
    assert handled.status == "passed"
    assert handled.promotion_allowed is True
    assert "nonstationarity_detected" not in handled.reason_codes


@pytest.mark.parametrize(
    "declared_test_id",
    ("model_confidence_set_v1", "superior_predictive_ability_v1"),
)
def test_unavailable_many_model_tests_emit_superiority_not_tested(
    declared_test_id: str,
) -> None:
    result = run_declared_predictive_test(
        declared_test_id=declared_test_id,
        loss_differentials=(0.2,) * 80,
        practical_margin=0.05,
        optional_backend_overrides={"arch": None},
        effective_sample_size=80,
        effective_block_count=80,
    )

    manifest = result.as_manifest()

    assert manifest["status"] == "abstained"
    assert manifest["promotion_allowed"] is False
    assert "multi_model_superiority_not_tested" in manifest["reason_codes"]
    assert manifest["dependency_diagnostics"]["backend"] == "arch"
    assert (
        manifest["dependency_diagnostics"]["reason_code"]
        == "multi_model_test_backend_unavailable"
    )
    assert manifest["dependency_diagnostics"]["declared_test_id"] == declared_test_id


def _paired_identity(pair_count: int) -> dict[str, object]:
    return {
        "row_set_id": "confirmatory_pairs_v1",
        "origin_ids": [f"origin_{index}" for index in range(pair_count)],
        "horizons": [1],
        "entity_ids": ["entity_a"],
    }
