from __future__ import annotations

import importlib
import inspect
import math
from collections.abc import Callable, Sequence
from typing import Any

import pytest


def _load_regime_switching_adapter() -> Callable[..., Any]:
    try:
        module = importlib.import_module("euclid.nonstationarity.regime_switching")
    except ModuleNotFoundError:
        pytest.fail(
            "Phase 6 regime-switching API is missing; expected "
            "euclid.nonstationarity.regime_switching"
        )
    for function_name in (
        "fit_markov_switching_regimes",
        "fit_regime_switching",
        "run_regime_switching_diagnostic",
    ):
        adapter = getattr(module, function_name, None)
        if callable(adapter):
            return adapter
    pytest.fail(
        "Phase 6 regime-switching API is missing; expected a callable "
        "fit_markov_switching_regimes, fit_regime_switching, or "
        "run_regime_switching_diagnostic"
    )


def _strong_regime_series() -> tuple[tuple[float, ...], tuple[int, ...]]:
    values: list[float] = []
    truth: list[int] = []
    for regime_id, center, count in (
        (0, -2.0, 30),
        (1, 3.5, 30),
        (0, -2.0, 30),
        (1, 3.5, 30),
    ):
        for index in range(count):
            values.append(center + 0.04 * ((index % 5) - 2))
            truth.append(regime_id)
    return tuple(values), tuple(truth)


def _weak_regime_series() -> tuple[tuple[float, ...], tuple[int, ...]]:
    values: list[float] = []
    truth: list[int] = []
    for index in range(90):
        regime_id = index % 2
        truth.append(regime_id)
        values.append(1.0 + 0.03 * regime_id + 0.005 * ((index % 3) - 1))
    return tuple(values), tuple(truth)


def _assert_probability_rows(
    rows: Sequence[Sequence[float]],
    *,
    row_count: int,
    width: int,
) -> None:
    assert len(rows) == row_count
    for row in rows:
        assert len(row) == width
        assert sum(float(value) for value in row) == pytest.approx(1.0, abs=1e-6)
        assert all(0.0 <= float(value) <= 1.0 for value in row)


def _fit_regimes(
    adapter: Callable[..., Any],
    *,
    observations: Sequence[float],
    series_id: str,
    regime_count: int,
    truth_regimes: Sequence[int] | None = None,
    min_regime_mean_separation: float = 0.25,
    optional_backend_overrides: dict[str, Any] | None = None,
) -> Any:
    parameters = inspect.signature(adapter).parameters
    kwargs: dict[str, Any] = {
        "observations": observations,
        "series_id": series_id,
        "truth_regimes": truth_regimes,
        "optional_backend_overrides": optional_backend_overrides,
    }
    if "regime_count" in parameters:
        kwargs["regime_count"] = regime_count
    else:
        kwargs["n_regimes"] = regime_count
    if "method" in parameters:
        kwargs["method"] = "markov_regression"
    else:
        kwargs["model_type"] = "markov_regression"
    if "min_regime_mean_separation" in parameters:
        kwargs["min_regime_mean_separation"] = min_regime_mean_separation
    else:
        kwargs["min_separation"] = min_regime_mean_separation
    filtered_kwargs = {
        key: value for key, value in kwargs.items() if value is not None
    }
    return adapter(**filtered_kwargs)


def _fake_statsmodels_backend() -> dict[str, type]:
    class _FakeMarkovRegression:
        def __init__(self, values: Sequence[float], *, k_regimes: int, **_: Any):
            self.values = tuple(float(value) for value in values)
            self.k_regimes = int(k_regimes)

        def fit(self, **_: Any) -> Any:
            return _FakeMarkovResult(self.values, self.k_regimes)

    return {
        "MarkovRegression": _FakeMarkovRegression,
        "MarkovAutoregression": _FakeMarkovRegression,
    }


class _FakeMarkovResult:
    def __init__(self, values: Sequence[float], regime_count: int):
        midpoint = (min(values) + max(values)) / 2.0
        spread = max(values) - min(values)
        if spread < 0.25:
            self.smoothed_marginal_probabilities = [
                [0.52, 0.48] for _ in values
            ]
            self.regime_transition = [[0.55, 0.45], [0.45, 0.55]]
            self.expected_durations = [2.2222222222] * regime_count
        else:
            self.smoothed_marginal_probabilities = [
                [0.97, 0.03] if value <= midpoint else [0.03, 0.97]
                for value in values
            ]
            self.regime_transition = [[0.92, 0.08], [0.08, 0.92]]
            self.expected_durations = [12.5] * regime_count
        self.mle_retvals = {"converged": True, "iterations": 7}
        self.llf = -12.5


def test_markov_switching_adapter_emits_regime_evidence_contract() -> None:
    fit_markov_switching_regimes = _load_regime_switching_adapter()
    observations, truth = _strong_regime_series()

    artifact = _fit_regimes(
        fit_markov_switching_regimes,
        observations=observations,
        series_id="synthetic-markov-regime",
        regime_count=2,
        truth_regimes=truth,
        min_regime_mean_separation=1.0,
        optional_backend_overrides={"statsmodels": _fake_statsmodels_backend()},
    )
    manifest = artifact.as_manifest()

    assert manifest["schema_name"] == "regime_switching_artifact@1.0.0"
    assert manifest["artifact_type"] == "regime_switching"
    assert manifest["series_id"] == "synthetic-markov-regime"
    assert manifest.get("method", manifest.get("model_class")) in {
        "markov_regression",
        "MarkovRegression",
    }
    assert manifest["status"] == "passed"
    assert manifest["reason_codes"] == []
    assert manifest.get("regime_count", manifest.get("n_regimes")) == 2
    _assert_probability_rows(
        manifest["transition_matrix"],
        row_count=2,
        width=2,
    )
    _assert_probability_rows(
        manifest["smoothed_probabilities"],
        row_count=len(observations),
        width=2,
    )
    assert len(manifest["expected_durations"]) == 2
    assert all(float(duration) > 1.0 for duration in manifest["expected_durations"])
    assert manifest["convergence_diagnostics"]["converged"] is True
    assert isinstance(manifest["convergence_diagnostics"]["iterations"], int)
    assert manifest["convergence_diagnostics"]["iterations"] >= 0
    log_likelihood = manifest["convergence_diagnostics"].get(
        "log_likelihood",
        manifest["convergence_diagnostics"].get("llf"),
    )
    assert math.isfinite(float(log_likelihood))


def test_weakly_separated_regimes_fail_closed_with_identifiability_reason() -> None:
    fit_markov_switching_regimes = _load_regime_switching_adapter()
    observations, truth = _weak_regime_series()

    artifact = _fit_regimes(
        fit_markov_switching_regimes,
        observations=observations,
        series_id="weakly-separated-regime-series",
        regime_count=2,
        truth_regimes=truth,
        min_regime_mean_separation=0.25,
        optional_backend_overrides={"statsmodels": _fake_statsmodels_backend()},
    )
    manifest = artifact.as_manifest()

    assert manifest["status"] in {"failed", "abstained"}
    assert "weak_regime_identifiability" in manifest["reason_codes"]
    assert manifest["promotion_allowed"] is False
    assert manifest["claim_scope"] != "valid_given_regime"


def test_regime_conditioned_laws_are_scoped_to_valid_given_regime() -> None:
    fit_markov_switching_regimes = _load_regime_switching_adapter()
    observations, truth = _strong_regime_series()

    artifact = _fit_regimes(
        fit_markov_switching_regimes,
        observations=observations,
        series_id="regime-scoped-law-series",
        regime_count=2,
        truth_regimes=truth,
        min_regime_mean_separation=1.0,
        optional_backend_overrides={"statsmodels": _fake_statsmodels_backend()},
    )
    manifest = artifact.as_manifest()

    assert manifest["status"] == "passed"
    assert manifest["claim_scope"] == "valid_given_regime"
    assert manifest.get("regime_conditioned_scope") == "valid_given_regime"
    assert manifest.get("is_law_claim", False) is False
    assert manifest["claim_scope"] not in {"stationary_law", "universal_law"}
    assert manifest.get("may_publish_unconditional_law_claim", False) is False


def test_missing_statsmodels_markov_backend_fails_closed_with_specific_reason() -> None:
    fit_markov_switching_regimes = _load_regime_switching_adapter()
    observations, _ = _strong_regime_series()

    artifact = _fit_regimes(
        fit_markov_switching_regimes,
        observations=observations,
        series_id="regime-series-without-backend",
        regime_count=2,
        optional_backend_overrides={"statsmodels": None},
    )
    manifest = artifact.as_manifest()

    assert manifest["status"] == "adapter_unavailable"
    assert manifest["reason_codes"] == [
        "statsmodels_markov_switching_backend_unavailable"
    ]
    assert manifest["transition_matrix"] == []
    assert manifest["smoothed_probabilities"] == []
    assert manifest["promotion_allowed"] is False
