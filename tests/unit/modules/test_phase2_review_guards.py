from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from euclid.contracts.loader import load_contract_catalog
from euclid.modules.evaluation_governance import (
    build_predictive_gate_policy,
    resolve_confirmatory_promotion_allowed,
)
from euclid.modules.predictive_tests import evaluate_predictive_promotion


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_raw_averaged_metric_delta_cannot_promote_without_declared_paired_test() -> (
    None
):
    catalog = load_contract_catalog(PROJECT_ROOT)
    predictive_gate_policy = build_predictive_gate_policy(
        allowed_forecast_object_types=("point",),
    ).to_manifest(catalog)
    raw_delta_only_result = {
        "schema_name": "paired_predictive_test_result@1.0.0",
        "status": "passed",
        "promotion_allowed": True,
        "reason_codes": [],
        "mean_loss_differential": 1.0,
        "confidence_interval": [0.8, 1.2],
        "confidence_interval_method": "not_declared",
        "practical_margin": 0.1,
        "raw_metric_comparison_role": "diagnostic_only",
        "statistical_test_backend": "raw_averaged_metric_delta",
    }

    assert (
        resolve_confirmatory_promotion_allowed(
            candidate_beats_baseline=True,
            predictive_gate_policy_manifest=predictive_gate_policy,
            predictive_test_result_manifest=raw_delta_only_result,
        )
        is False
    )


def test_one_pair_predictive_test_abstains_with_specific_reason() -> None:
    result = evaluate_predictive_promotion(
        candidate_losses=(0.0,),
        baseline_losses=(1.0,),
        split_protocol_id="declared_confirmatory_holdout",
        baseline_id="naive",
        practical_margin=0.1,
    )

    assert result.status == "abstained"
    assert result.promotion_allowed is False
    assert "insufficient_paired_count" in result.reason_codes


def test_declared_predictive_test_id_matches_actual_computation_label() -> None:
    result = evaluate_predictive_promotion(
        candidate_losses=(0.0,) * 60,
        baseline_losses=(1.0,) * 60,
        split_protocol_id="declared_confirmatory_holdout",
        baseline_id="naive",
        practical_margin=0.1,
    )
    manifest = result.as_manifest()

    assert manifest.get("declared_test_id") == "diebold_mariano_hln_v1"
    assert manifest.get("actual_test_id") == manifest["declared_test_id"]
    assert manifest["statistical_test_backend"] == "diebold_mariano_hln_v1"


@pytest.mark.parametrize(
    ("paired_count", "expected_status", "expected_reason_code"),
    (
        (24, "abstained", "insufficient_effective_sample_size"),
        (30, "human_review_only", "minimum_effective_information_human_review"),
    ),
)
def test_minimum_effective_information_thresholds_block_automatic_promotion(
    paired_count: int,
    expected_status: str,
    expected_reason_code: str,
) -> None:
    result = evaluate_predictive_promotion(
        candidate_losses=(0.0,) * paired_count,
        baseline_losses=(1.0,) * paired_count,
        split_protocol_id="declared_confirmatory_holdout",
        baseline_id="naive",
        practical_margin=0.1,
    )
    manifest = result.as_manifest()

    assert result.status == expected_status
    assert result.promotion_allowed is False
    assert expected_reason_code in result.reason_codes
    assert manifest["minimum_pair_policy"]["minimum_effective_sample_size"] == 25
    assert manifest["effective_sample_size"] == paired_count


def test_unsupported_optional_bootstrap_backend_fails_closed_with_stable_reason() -> (
    None
):
    import euclid.modules.predictive_tests as predictive_tests

    runner = getattr(predictive_tests, "run_declared_predictive_test", None)
    assert runner is not None, "Phase 2 must expose a declared predictive-test runner"

    result = runner(
        declared_test_id="paired_stationary_block_bootstrap_v1",
        loss_differentials=(1.0,) * 60,
        optional_backend_overrides={"arch": None},
        block_length=10,
        bootstrap_count=999,
        seed=123,
    )
    manifest: dict[str, Any] = result.as_manifest()

    assert manifest["status"] == "abstained"
    assert manifest["promotion_allowed"] is False
    assert manifest["reason_codes"] == ["bootstrap_test_backend_unavailable"]
    assert manifest["dependency_diagnostics"] == {
        "backend": "arch",
        "reason_code": "bootstrap_test_backend_unavailable",
    }
