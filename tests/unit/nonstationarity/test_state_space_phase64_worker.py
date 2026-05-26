from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
import pytest


def _state_space_module() -> Any:
    try:
        return importlib.import_module("euclid.nonstationarity.state_space")
    except ModuleNotFoundError as exc:  # pragma: no cover - red-phase guard
        pytest.fail(f"missing state-space module: {exc}")


def _local_level_series() -> tuple[tuple[float, ...], tuple[float, ...]]:
    rng = np.random.default_rng(6404)
    latent = np.cumsum(rng.normal(0.0, 0.09, 96))
    observations = latent + rng.normal(0.0, 0.035, 96)
    return tuple(float(value) for value in observations), tuple(
        float(value) for value in latent
    )


def test_local_level_fit_recovers_latent_state_and_emits_full_artifact() -> None:
    state_space = _state_space_module()
    observations, latent_truth = _local_level_series()

    artifact = state_space.fit_local_level_state_space(
        observations=observations,
        latent_truth=latent_truth,
        whiteness_lag=10,
    )

    manifest = artifact.as_manifest()

    assert list(manifest) == [
        "schema_name",
        "artifact_type",
        "artifact_id",
        "series_id",
        "lane_id",
        "model",
        "backend",
        "status",
        "reason_codes",
        "promotion_allowed",
        "sample_count",
        "filtered_state",
        "smoothed_state",
        "state_covariance",
        "innovations",
        "log_likelihood",
        "innovation_diagnostics",
        "latent_recovery",
        "parameters",
        "metadata",
        "evidence_role",
        "claim_scope",
        "law_claim_allowed",
        "unavailable_reason",
    ]
    assert manifest["schema_name"] == "state_space_artifact@1.0.0"
    assert manifest["artifact_type"] == "state_space"
    assert manifest["lane_id"] == "state_space_local_level_v1"
    assert manifest["model"] == "local_level"
    assert manifest["backend"] == "statsmodels"
    assert manifest["status"] == "passed"
    assert manifest["reason_codes"] == []
    assert manifest["promotion_allowed"] is True
    assert manifest["sample_count"] == len(observations)
    assert len(manifest["filtered_state"]) == len(observations)
    assert len(manifest["smoothed_state"]) == len(observations)
    assert len(manifest["state_covariance"]["filtered"]) == len(observations)
    assert len(manifest["state_covariance"]["smoothed"]) == len(observations)
    assert len(manifest["innovations"]) == len(observations)
    assert math.isfinite(manifest["log_likelihood"])
    assert manifest["innovation_diagnostics"]["whiteness_passed"] is True
    assert manifest["innovation_diagnostics"]["ljung_box_p_value"] > 0.05
    assert manifest["latent_recovery"]["rmse"] < 0.08


def test_innovation_whiteness_failure_blocks_promotion() -> None:
    state_space = _state_space_module()
    t = np.arange(120, dtype=float)
    observations = tuple(float(value) for value in np.sin(t / 3.0))

    artifact = state_space.fit_local_level_state_space(
        observations=observations,
        whiteness_lag=10,
    )

    manifest = artifact.as_manifest()

    assert manifest["status"] == "failed"
    assert manifest["promotion_allowed"] is False
    assert manifest["reason_codes"] == ["state_space_innovation_whiteness_failed"]
    assert manifest["innovation_diagnostics"]["whiteness_passed"] is False
    assert manifest["innovation_diagnostics"]["ljung_box_p_value"] < 0.05


def test_missing_statsmodels_fails_closed_with_stable_reason() -> None:
    state_space = _state_space_module()

    artifact = state_space.fit_local_level_state_space(
        observations=(1.0, 1.1, 0.9, 1.2),
        optional_backend_overrides={"statsmodels": None},
    )

    manifest = artifact.as_manifest()

    assert manifest["schema_name"] == "state_space_artifact@1.0.0"
    assert manifest["status"] == "adapter_unavailable"
    assert manifest["promotion_allowed"] is False
    assert manifest["reason_codes"] == [
        "statsmodels_state_space_backend_unavailable"
    ]
    assert manifest["unavailable_reason"] == (
        "statsmodels_state_space_backend_unavailable"
    )
    assert manifest["filtered_state"] == []
    assert manifest["smoothed_state"] == []
    assert manifest["state_covariance"] == {}
    assert manifest["innovations"] == []
    assert manifest["log_likelihood"] is None
    assert manifest["innovation_diagnostics"]["whiteness_passed"] is False
