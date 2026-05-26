from __future__ import annotations

import pytest

from euclid.nonstationarity.stability import run_stability_diagnostic


def _piecewise_shifted_series() -> tuple[float, ...]:
    first = tuple(0.02 * index for index in range(40))
    second = tuple(4.0 + 0.02 * index for index in range(40))
    return first + second


def test_cusum_recursive_residual_diagnostic_emits_instability_evidence() -> None:
    artifact = run_stability_diagnostic(
        series_id="shifted_mean_series",
        observations=_piecewise_shifted_series(),
        alpha=0.05,
    )

    manifest = artifact.as_manifest()

    assert manifest["schema_name"] == "stability_diagnostic_artifact@1.0.0"
    assert manifest["status"] == "failed"
    assert "stability_test_failed" in manifest["reason_codes"]
    assert "instability_evidence_unresolved" in manifest["reason_codes"]
    assert manifest["backend"] == "statsmodels"
    assert manifest["diagnostics"]["cusum_ols_residuals"]["p_value"] < 0.05
    assert manifest["diagnostics"]["recursive_residuals"]["max_abs_cusum"] > 0.0


def test_missing_statsmodels_emits_stable_unavailable_reason() -> None:
    artifact = run_stability_diagnostic(
        series_id="series_without_backend",
        observations=_piecewise_shifted_series(),
        optional_backend_overrides={"statsmodels": None},
    )

    manifest = artifact.as_manifest()

    assert manifest["status"] == "abstained"
    assert manifest["reason_codes"] == ["statsmodels_unavailable"]
    assert manifest["unavailable_reason"] == "statsmodels_unavailable"
    assert manifest["backend"] == "statsmodels"


@pytest.mark.parametrize(
    "observations",
    ((1.0,) * 32, _piecewise_shifted_series()),
)
def test_stability_diagnostics_are_not_law_claims(
    observations: tuple[float, ...],
) -> None:
    artifact = run_stability_diagnostic(
        series_id="diagnostic_only_series",
        observations=observations,
        optional_backend_overrides={"statsmodels": None},
    )

    manifest = artifact.as_manifest()

    assert manifest["evidence_role"] == "diagnostic_only"
    assert manifest["is_law_claim"] is False
    assert manifest["claim_scope"] == "not_a_law_claim"
