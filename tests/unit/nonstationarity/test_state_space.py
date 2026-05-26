from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import pytest


def _load_state_space_adapter() -> Callable[..., Any]:
    try:
        from euclid.nonstationarity.state_space import fit_local_level_state_space
    except ModuleNotFoundError:
        pytest.fail(
            "Phase 6 state-space API is missing; expected "
            "euclid.nonstationarity.state_space.fit_local_level_state_space"
        )
    return fit_local_level_state_space


def _local_level_series() -> tuple[tuple[float, ...], tuple[float, ...]]:
    latent: list[float] = []
    observations: list[float] = []
    seed = 17
    level = 0.0
    for index in range(72):
        level += 0.05
        if index in {24, 48}:
            level += 0.4
        seed = (1103515245 * seed + 12345) % (2**31)
        noise = ((seed / (2**31)) - 0.5) * 0.12
        latent.append(level)
        observations.append(level + noise)
    return tuple(observations), tuple(latent)


def _autocorrelated_innovation_series() -> (
    tuple[tuple[float, ...], tuple[float, ...]]
):
    latent: list[float] = []
    observations: list[float] = []
    correlated_error = 0.0
    for index in range(96):
        level = 0.02 * index
        shock = 0.12 if index % 4 else -0.02
        correlated_error = 0.85 * correlated_error + shock
        latent.append(level)
        observations.append(level + correlated_error)
    return tuple(observations), tuple(latent)


def _rmse(observed: Sequence[float], expected: Sequence[float]) -> float:
    assert len(observed) == len(expected)
    squared_error = [
        (float(actual) - float(target)) ** 2
        for actual, target in zip(observed, expected, strict=True)
    ]
    return math.sqrt(sum(squared_error) / len(squared_error))


def _assert_finite_sequence(values: Sequence[float], *, length: int) -> None:
    assert len(values) == length
    assert all(math.isfinite(float(value)) for value in values)


def test_local_level_synthetic_series_recovers_latent_state_within_tolerance() -> None:
    fit_local_level_state_space = _load_state_space_adapter()
    observations, latent_truth = _local_level_series()

    artifact = fit_local_level_state_space(
        observations=observations,
        series_id="synthetic-local-level",
        latent_truth=latent_truth,
    )
    manifest = artifact.as_manifest()

    assert manifest["schema_name"] == "state_space_artifact@1.0.0"
    assert manifest["artifact_type"] == "state_space"
    assert manifest["model"] == "local_level"
    assert manifest["latent_recovery"]["rmse"] <= 0.15
    assert _rmse(manifest["smoothed_state"], latent_truth) <= 0.15


def test_state_space_artifact_includes_filter_smoother_covariance_and_likelihood() -> (
    None
):
    fit_local_level_state_space = _load_state_space_adapter()
    observations, latent_truth = _local_level_series()

    artifact = fit_local_level_state_space(
        observations=observations,
        series_id="state-space-evidence-contract",
        latent_truth=latent_truth,
    )
    manifest = artifact.as_manifest()

    assert manifest["schema_name"] == "state_space_artifact@1.0.0"
    _assert_finite_sequence(manifest["filtered_state"], length=len(observations))
    _assert_finite_sequence(manifest["smoothed_state"], length=len(observations))
    assert sorted(manifest["state_covariance"]) == ["filtered", "smoothed"]
    _assert_finite_sequence(
        manifest["state_covariance"]["filtered"],
        length=len(observations),
    )
    _assert_finite_sequence(
        manifest["state_covariance"]["smoothed"],
        length=len(observations),
    )
    _assert_finite_sequence(manifest["innovations"], length=len(observations))
    assert all(
        float(value) >= 0.0 for value in manifest["state_covariance"]["filtered"]
    )
    assert all(
        float(value) >= 0.0 for value in manifest["state_covariance"]["smoothed"]
    )
    assert math.isfinite(float(manifest["log_likelihood"]))


def test_innovation_whiteness_failure_blocks_promotion() -> None:
    fit_local_level_state_space = _load_state_space_adapter()
    observations, latent_truth = _autocorrelated_innovation_series()

    artifact = fit_local_level_state_space(
        observations=observations,
        series_id="state-space-autocorrelated-innovation",
        latent_truth=latent_truth,
        whiteness_lag=4,
        whiteness_alpha=0.05,
    )
    manifest = artifact.as_manifest()

    assert manifest["status"] in {"failed", "abstained"}
    assert "state_space_innovation_whiteness_failed" in manifest["reason_codes"]
    assert manifest["promotion_allowed"] is False
    assert manifest["innovation_diagnostics"]["whiteness_passed"] is False


def test_missing_statsmodels_state_space_backend_fails_closed_with_specific_reason() -> (
    None
):
    fit_local_level_state_space = _load_state_space_adapter()
    observations, _ = _local_level_series()

    artifact = fit_local_level_state_space(
        observations=observations,
        series_id="state-space-without-backend",
        optional_backend_overrides={"statsmodels": None},
    )
    manifest = artifact.as_manifest()

    assert manifest["status"] == "adapter_unavailable"
    assert manifest["reason_codes"] == [
        "statsmodels_state_space_backend_unavailable"
    ]
    assert manifest["filtered_state"] == []
    assert manifest["smoothed_state"] == []
    assert manifest["promotion_allowed"] is False
