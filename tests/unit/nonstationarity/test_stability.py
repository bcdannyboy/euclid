from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


def _load_stability_diagnostic() -> Callable[..., Any]:
    try:
        from euclid.nonstationarity.stability import run_stability_diagnostic
    except ModuleNotFoundError:
        pytest.fail(
            "Phase 6 stability API is missing; expected "
            "euclid.nonstationarity.stability.run_stability_diagnostic"
        )
    return run_stability_diagnostic


def _linear_design(row_count: int) -> tuple[tuple[float, float], ...]:
    return tuple((1.0, float(index)) for index in range(row_count))


def test_cusum_recursive_residual_diagnostic_emits_instability_evidence() -> None:
    run_stability_diagnostic = _load_stability_diagnostic()
    observations = tuple(
        1.0 + 0.2 * index if index < 48 else 18.0 + 0.2 * index
        for index in range(96)
    )

    artifact = run_stability_diagnostic(
        observations=observations,
        design_matrix=_linear_design(len(observations)),
        method="cusum_recursive_residuals",
        significance_level=0.05,
        min_observations=24,
    )
    manifest = artifact.as_manifest()

    assert manifest["schema_name"] == "stability_diagnostic_artifact@1.0.0"
    assert manifest["method"] == "cusum_recursive_residuals"
    assert manifest["status"] == "failed"
    assert manifest["instability_detected"] is True
    assert "recursive_residual_instability_detected" in manifest["reason_codes"]
    assert manifest["diagnostic_statistic"] is not None
    assert manifest["p_value"] < 0.05


def test_stability_diagnostics_are_diagnostic_evidence_not_law_claims() -> None:
    run_stability_diagnostic = _load_stability_diagnostic()
    observations = tuple(
        2.0 + 0.5 * index + (0.01 if index % 2 else -0.01)
        for index in range(80)
    )

    artifact = run_stability_diagnostic(
        observations=observations,
        design_matrix=_linear_design(len(observations)),
        method="cusum_recursive_residuals",
        significance_level=0.05,
        min_observations=24,
    )
    manifest = artifact.as_manifest()

    assert manifest["evidence_role"] == "diagnostic_only"
    assert manifest["claim_scope"] == "diagnostic_only"
    assert manifest["is_law_claim"] is False
    assert manifest["may_publish_stationary_law_claim"] is False
