from __future__ import annotations

import importlib
from typing import Any

import pytest

from euclid.contracts.errors import ContractValidationError


def _conformal() -> Any:
    try:
        return importlib.import_module("euclid.modules.conformal")
    except ModuleNotFoundError as exc:
        pytest.fail(f"conformal module is missing: {exc}")


def test_conformal_registry_declares_known_guarantee_tiers() -> None:
    module = _conformal()
    registry = module.conformal_method_registry()

    assert module.CONFORMAL_GUARANTEE_TIERS == (
        "finite_sample_exchangeable",
        "approximate_mixing_time_series",
        "asymptotic_time_series",
        "long_run_frequency_control",
        "diagnostic_only",
    )
    assert registry["split_conformal_exchangeable_v1"].guarantee_tier == (
        "finite_sample_exchangeable"
    )
    assert registry["enbpi_time_series_v1"].guarantee_tier == (
        "approximate_mixing_time_series"
    )
    assert registry["adaptive_conformal_time_series_v1"].guarantee_tier == (
        "long_run_frequency_control"
    )


def test_split_conformal_requires_exchangeability_declaration() -> None:
    module = _conformal()

    result = module.resolve_conformal_method(
        method_id="split_conformal_exchangeable_v1",
        calibration_split_ids=("calibration-split-a",),
        horizon_ids=(1, 3),
        declarations={"exchangeability": False},
    )

    assert result.status == "failed"
    assert result.reason_codes == ("exchangeability_declaration_required",)
    assert result.guarantee_tier == "finite_sample_exchangeable"
    assert result.finite_sample_distribution_free_allowed is False
    assert result.finite_sample_distribution_free_claim_allowed is False
    assert result.as_manifest()["calibration_split_ids"] == ["calibration-split-a"]
    assert result.as_manifest()["horizon_ids"] == [1, 3]


def test_registry_resolution_records_split_and_horizon_scope_for_valid_method() -> None:
    module = _conformal()

    result = module.resolve_conformal_method(
        method_id="split_conformal_exchangeable_v1",
        calibration_split_ids=("calibration-split-a", "calibration-split-b"),
        horizon_ids=(1, 3),
        declarations={"exchangeability": True},
    )

    assert result.status == "passed"
    assert result.reason_codes == ()
    assert result.guarantee_tier == "finite_sample_exchangeable"
    assert result.finite_sample_distribution_free_allowed is True
    assert result.finite_sample_distribution_free_claim_allowed is True
    assert result.fixed_time_finite_sample_allowed is True
    assert result.fixed_time_finite_sample_claim_allowed is True
    manifest = result.as_manifest()
    assert manifest["schema_name"] == "conformal_method_resolution@1.0.0"
    assert manifest["method_id"] == "split_conformal_exchangeable_v1"
    assert manifest["calibration_split_id"] == "calibration-split-a"
    assert manifest["calibration_split_ids"] == [
        "calibration-split-a",
        "calibration-split-b",
    ]
    assert manifest["horizon_ids"] == [1, 3]
    assert manifest["finite_sample_distribution_free_claim_allowed"] is True


def test_time_series_methods_do_not_claim_exact_finite_sample_validity() -> None:
    module = _conformal()

    enbpi = module.resolve_conformal_method(
        method_id="enbpi_time_series_v1",
        calibration_split_ids=("rolling-calibration-a",),
        horizon_ids=(1,),
        declarations={"weak_dependence_or_mixing": True},
    )
    adaptive = module.resolve_conformal_method(
        method_id="adaptive_conformal_time_series_v1",
        calibration_split_ids=("online-calibration-a",),
        horizon_ids=(1,),
        declarations={"online_adaptation": True, "long_run_frequency": True},
    )

    assert enbpi.status == "passed"
    assert enbpi.guarantee_tier == "approximate_mixing_time_series"
    assert enbpi.finite_sample_distribution_free_allowed is False
    assert enbpi.fixed_time_finite_sample_allowed is False
    assert enbpi.assumption_ids == ("weak_dependence_or_mixing",)
    assert adaptive.status == "passed"
    assert adaptive.guarantee_tier == "long_run_frequency_control"
    assert adaptive.finite_sample_distribution_free_allowed is False
    assert adaptive.fixed_time_finite_sample_allowed is False
    assert adaptive.assumption_ids == ("online_adaptation", "long_run_frequency")


def test_unknown_conformal_method_fails_closed() -> None:
    module = _conformal()

    result = module.resolve_conformal_method(
        method_id="raw_residual_quantile_v0",
        calibration_split_ids=("calibration-split-a",),
        horizon_ids=(1,),
        declarations={"exchangeability": True},
    )

    assert result.status == "failed"
    assert result.reason_codes == ("unknown_conformal_method",)
    assert result.guarantee_tier == "diagnostic_only"
    assert result.finite_sample_distribution_free_allowed is False
    assert result.fixed_time_finite_sample_allowed is False
    assert result.as_manifest()["method_id"] == "raw_residual_quantile_v0"


def test_claim_scope_blocks_finite_sample_distribution_free_language() -> None:
    module = _conformal()
    method = module.resolve_conformal_method(
        method_id="enbpi_time_series_v1",
        calibration_split_ids=("rolling-calibration-a",),
        horizon_ids=(1,),
        declarations={"weak_dependence_or_mixing": True},
    )

    with pytest.raises(ContractValidationError) as exc_info:
        module.assert_conformal_claim_scope(
            {
                "claim_text": (
                    "This interval has finite-sample distribution-free coverage."
                ),
                "conformal_method": method.as_manifest(),
            }
        )

    assert exc_info.value.code == "claim_scope_overstatement"
    assert exc_info.value.details["guarantee_tier"] == (
        "approximate_mixing_time_series"
    )
    assert (
        "finite_sample_distribution_free_language_requires_"
        "finite_sample_exchangeable_tier"
        in exc_info.value.details["reason_codes"]
    )
