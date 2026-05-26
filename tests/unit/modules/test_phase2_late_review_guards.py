from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from euclid.contracts.loader import load_contract_catalog
from euclid.invariance.environments import construct_environments
from euclid.invariance.gates import evaluate_invariance
from euclid.manifests.base import ManifestEnvelope
from euclid.modules.evaluation_governance import (
    build_predictive_gate_policy,
    resolve_confirmatory_promotion_allowed,
)
from euclid.modules.gate_lifecycle import resolve_scorecard_status
from euclid.modules.predictive_tests import evaluate_predictive_promotion


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GENERIC_REASON_CODES = {"failed", "invalid", "error"}


def _losses(
    *,
    pair_count: int,
    candidate_loss: float = 0.65,
    baseline_loss: float = 1.0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        tuple(candidate_loss for _ in range(pair_count)),
        tuple(baseline_loss for _ in range(pair_count)),
    )


def _paired_stream_identity(
    pair_count: int,
    *,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "row_set_id": "confirmatory_holdout_pairs_v1",
        "origin_ids": [f"origin_{index}" for index in range(pair_count)],
        "horizons": [1],
        "entity_ids": ["entity_a"],
    }
    if extra_fields is not None:
        identity.update(dict(extra_fields))
    return identity


def _assert_specific_reason(
    reason_codes: tuple[str, ...] | list[str],
    expected_reason_code: str,
) -> None:
    observed = set(reason_codes)
    assert expected_reason_code in observed
    assert not (GENERIC_REASON_CODES & observed)


def _point_predictive_gate_policy() -> ManifestEnvelope:
    return build_predictive_gate_policy(
        allowed_forecast_object_types=("point",),
    ).to_manifest(load_contract_catalog(PROJECT_ROOT))


def _passing_predictive_test_manifest() -> dict[str, Any]:
    return {
        "schema_name": "paired_predictive_test_result@1.0.0",
        "declared_test_id": "diebold_mariano_hln_v1",
        "actual_test_id": "diebold_mariano_hln_v1",
        "status": "passed",
        "promotion_allowed": True,
        "reason_codes": [],
        "mean_loss_differential": 0.35,
        "confidence_interval": [0.30, 0.40],
        "confidence_interval_method": "dm_hln_hac_t_interval",
        "practical_margin": 0.05,
        "raw_metric_comparison_role": "diagnostic_only",
        "statistical_test_backend": "diebold_mariano_hln_v1",
        "raw_pair_count": 80,
        "effective_sample_size": 80,
        "effective_block_count": 80,
    }


def test_raw_average_deltas_without_pair_identity_are_diagnostic_only() -> None:
    candidate_losses, baseline_losses = _losses(pair_count=80)

    result = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_confirmatory_holdout",
        baseline_id="naive",
        practical_margin=0.05,
        effective_sample_size=80,
        effective_block_count=80,
    )

    assert result.status == "abstained"
    assert result.promotion_allowed is False
    assert result.reason_codes == ("unpaired_loss_stream",)
    assert result.raw_metric_comparison_role == "diagnostic_only"
    _assert_specific_reason(result.reason_codes, "unpaired_loss_stream")


def test_thin_effective_sample_blocks_with_publication_floor_reason() -> None:
    candidate_losses, baseline_losses = _losses(pair_count=80)

    result = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_confirmatory_holdout",
        baseline_id="naive",
        practical_margin=0.05,
        paired_stream_identity=_paired_stream_identity(80),
        effective_sample_size=24,
        effective_block_count=24,
    )

    assert result.status == "abstained"
    assert result.promotion_allowed is False
    assert result.reason_codes == ("insufficient_effective_sample_size",)
    _assert_specific_reason(result.reason_codes, "insufficient_effective_sample_size")


def test_pairwise_dm_does_not_promote_after_multi_model_selection() -> None:
    candidate_losses, baseline_losses = _losses(pair_count=80)

    result = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_confirmatory_holdout",
        baseline_id="naive",
        practical_margin=0.05,
        declared_test_id="diebold_mariano_hln_v1",
        paired_stream_identity=_paired_stream_identity(
            80,
            extra_fields={
                "comparison_regime_id": "multi_model_candidate_selection",
                "comparison_model_count": 8,
                "many_model_adjustment_id": "none",
            },
        ),
        effective_sample_size=80,
        effective_block_count=80,
    )

    assert result.promotion_allowed is False
    _assert_specific_reason(result.reason_codes, "many_model_correction_failed")


def test_zero_practical_margin_is_not_practical_effect_evidence() -> None:
    candidate_losses, baseline_losses = _losses(pair_count=80)

    result = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_confirmatory_holdout",
        baseline_id="naive",
        practical_margin=0.0,
        declared_test_id="diebold_mariano_hln_v1",
        paired_stream_identity=_paired_stream_identity(80),
        effective_sample_size=80,
        effective_block_count=80,
    )

    assert result.promotion_allowed is False
    _assert_specific_reason(result.reason_codes, "missing_practical_effect_margin")


@pytest.mark.parametrize("missing_field", ("confidence_interval", "practical_margin"))
def test_hand_authored_pass_requires_uncertainty_and_effect_evidence(
    missing_field: str,
) -> None:
    predictive_test_result = _passing_predictive_test_manifest()
    predictive_test_result.pop(missing_field)

    confirmatory_allowed = resolve_confirmatory_promotion_allowed(
        candidate_beats_baseline=True,
        predictive_gate_policy_manifest=_point_predictive_gate_policy(),
        predictive_test_result_manifest=predictive_test_result,
    )
    scorecard = resolve_scorecard_status(
        candidate_admissible=True,
        robustness_status="passed",
        candidate_beats_baseline=True,
        confirmatory_promotion_allowed=confirmatory_allowed,
        point_score_comparison_status="comparable",
        time_safety_status="passed",
        calibration_status="not_applicable_for_forecast_type",
        predictive_governance_reason_codes=("metric_prerequisite_missing",),
    )

    assert confirmatory_allowed is False
    assert scorecard.predictive_reason_codes == ("metric_prerequisite_missing",)
    _assert_specific_reason(
        scorecard.predictive_reason_codes,
        "metric_prerequisite_missing",
    )


def test_calibration_failure_blocks_predictive_publication_with_reason() -> None:
    scorecard = resolve_scorecard_status(
        candidate_admissible=True,
        robustness_status="passed",
        candidate_beats_baseline=True,
        confirmatory_promotion_allowed=True,
        point_score_comparison_status="comparable",
        time_safety_status="passed",
        calibration_status="failed",
    )

    assert scorecard.predictive_status == "blocked"
    assert scorecard.predictive_reason_codes == ("calibration_failed",)
    _assert_specific_reason(scorecard.predictive_reason_codes, "calibration_failed")


def test_nonstationarity_failure_blocks_predictive_publication_with_reason() -> None:
    environments = construct_environments(
        (
            {"environment": "stable", "target": 1.0},
            {"environment": "drifted", "target": 1.0},
        ),
        policy="explicit_label",
        label_field="environment",
    )
    invariance = evaluate_invariance(
        environments=environments,
        residuals_by_environment={
            "stable": (0.01, 0.02, 0.01),
            "drifted": (0.40, 0.45, 0.50),
        },
        parameters_by_environment={
            "stable": {"slope": 1.0},
            "drifted": {"slope": 1.0},
        },
        supports_by_environment={"stable": {"x"}, "drifted": {"x"}},
        residual_spread_threshold=0.05,
    )

    assert invariance.status == "failed"
    _assert_specific_reason(invariance.reason_codes, "residual_invariance_failed")

    scorecard = resolve_scorecard_status(
        candidate_admissible=True,
        robustness_status="passed",
        candidate_beats_baseline=True,
        confirmatory_promotion_allowed=True,
        point_score_comparison_status="comparable",
        time_safety_status="passed",
        calibration_status="not_applicable_for_forecast_type",
        falsification_status="failed",
        falsification_reason_codes=("invariance_check_failed",),
    )

    assert scorecard.predictive_status == "blocked"
    assert scorecard.predictive_reason_codes == (
        "falsification_failed",
        "invariance_check_failed",
    )
    _assert_specific_reason(
        scorecard.predictive_reason_codes,
        "invariance_check_failed",
    )
