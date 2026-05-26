from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.modules.claims import assert_claim_scope_publication
from euclid.modules.conformal import (
    FINITE_SAMPLE_DISTRIBUTION_FREE_SCOPE_REASON,
    resolve_conformal_method,
)


EXACT_FINITE_SAMPLE_WORDING = (
    "This interval has exact finite-sample distribution-free coverage."
)


def test_exact_finite_sample_wording_allows_passed_exchangeable_method() -> None:
    method = resolve_conformal_method(
        method_id="split_conformal_exchangeable_v1",
        calibration_split_ids=("calibration-split-a",),
        horizon_ids=(1,),
        assumption_declarations={"exchangeability": True},
    ).as_manifest()

    assert method["status"] == "passed"
    assert method["guarantee_tier"] == "finite_sample_exchangeable"
    assert method["finite_sample_distribution_free_claim_allowed"] is True

    assert_claim_scope_publication(_claim_card(method))


@pytest.mark.parametrize(
    (
        "case_id",
        "method_manifest",
        "expected_tier",
        "expected_status",
    ),
    (
        (
            "approximate_mixing",
            resolve_conformal_method(
                method_id="enbpi_time_series_v1",
                calibration_split_ids=("rolling-calibration-a",),
                horizon_ids=(1,),
                assumption_declarations={"weak_dependence_or_mixing": True},
            ).as_manifest(),
            "approximate_mixing_time_series",
            "passed",
        ),
        (
            "long_run",
            resolve_conformal_method(
                method_id="adaptive_conformal_time_series_v1",
                calibration_split_ids=("online-calibration-a",),
                horizon_ids=(1,),
                assumption_declarations={
                    "online_adaptation": True,
                    "long_run_frequency": True,
                },
            ).as_manifest(),
            "long_run_frequency_control",
            "passed",
        ),
        (
            "diagnostic_only",
            {
                "method_id": "diagnostic_residual_coverage_v1",
                "status": "passed",
                "reason_codes": [],
                "guarantee_tier": "diagnostic_only",
                "assumption_scope": "diagnostic_only",
                "assumption_ids": [],
                "calibration_split_ids": ["diagnostic-split-a"],
                "horizon_ids": [1],
                "finite_sample_distribution_free_claim_allowed": False,
                "fixed_time_finite_sample_claim_allowed": False,
            },
            "diagnostic_only",
            "passed",
        ),
        (
            "unknown",
            resolve_conformal_method(
                method_id="raw_residual_quantile_v0",
                calibration_split_ids=("calibration-split-a",),
                horizon_ids=(1,),
                assumption_declarations={"exchangeability": True},
            ).as_manifest(),
            "diagnostic_only",
            "failed",
        ),
        (
            "blocked_exchangeable",
            resolve_conformal_method(
                method_id="split_conformal_exchangeable_v1",
                calibration_split_ids=("calibration-split-a",),
                horizon_ids=(1,),
                assumption_declarations={},
            ).as_manifest(),
            "finite_sample_exchangeable",
            "blocked",
        ),
    ),
)
def test_exact_finite_sample_wording_blocks_non_exchangeable_guarantees(
    case_id: str,
    method_manifest: dict[str, object],
    expected_tier: str,
    expected_status: str,
) -> None:
    assert case_id
    assert method_manifest["guarantee_tier"] == expected_tier
    assert method_manifest["status"] == expected_status
    assert method_manifest["finite_sample_distribution_free_claim_allowed"] is False

    with pytest.raises(ContractValidationError) as exc_info:
        assert_claim_scope_publication(_claim_card(method_manifest))

    assert exc_info.value.code == "claim_scope_overstatement"
    assert FINITE_SAMPLE_DISTRIBUTION_FREE_SCOPE_REASON in (
        exc_info.value.details["reason_codes"]
    )


def _claim_card(method_manifest: dict[str, object]) -> dict[str, object]:
    return {
        "claim_type": "predictive_within_declared_scope",
        "claim_ceiling": "predictive_within_declared_scope",
        "claim_text": EXACT_FINITE_SAMPLE_WORDING,
        "invariance_support_status": "not_requested",
        "transport_support_status": "not_requested",
        "stochastic_support_status": "not_requested",
        "conformal_method": method_manifest,
    }
