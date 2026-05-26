from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest


def _regime_switching_module() -> Any:
    try:
        return importlib.import_module("euclid.nonstationarity.regime_switching")
    except ModuleNotFoundError as exc:  # pragma: no cover - red-phase guard
        pytest.fail(f"missing regime-switching module: {exc}")


def test_markov_regression_artifact_emits_transition_probabilities_and_diagnostics() -> None:
    regime_switching = _regime_switching_module()
    fake_backend = _fake_statsmodels_backend(
        transition_matrix=((0.92, 0.08), (0.15, 0.85)),
        smoothed_probabilities=(
            (0.95, 0.05),
            (0.91, 0.09),
            (0.08, 0.92),
            (0.04, 0.96),
        ),
        expected_durations=(12.5, 6.666666666667),
        converged=True,
    )

    artifact = regime_switching.fit_regime_switching(
        series=(0.0, 0.1, 5.0, 5.2),
        series_id="two_state_synthetic",
        n_regimes=2,
        min_separation=0.25,
        optional_backend_overrides={"statsmodels": fake_backend},
    )
    manifest = artifact.as_manifest()

    assert manifest["schema_name"] == "regime_switching_artifact@1.0.0"
    assert manifest["artifact_type"] == "regime_switching"
    assert manifest["status"] == "passed"
    assert manifest["reason_codes"] == []
    assert manifest["backend"] == "statsmodels"
    assert manifest["model_class"] == "MarkovRegression"
    assert manifest["transition_matrix"] == [[0.92, 0.08], [0.15, 0.85]]
    assert manifest["expected_durations"] == [12.5, 6.666666666667]
    assert manifest["smoothed_probabilities"][0] == [0.95, 0.05]
    assert manifest["smoothed_probabilities"][-1] == [0.04, 0.96]
    assert manifest["convergence_diagnostics"]["converged"] is True
    assert manifest["convergence_diagnostics"]["iterations"] == 14
    assert manifest["regime_conditioned_scope"] == "valid_given_regime"


def test_missing_statsmodels_abstains_fail_closed_with_stable_reason() -> None:
    regime_switching = _regime_switching_module()

    artifact = regime_switching.fit_regime_switching(
        series=(0.0, 1.0, 0.0, 1.0),
        series_id="backend_missing",
        n_regimes=2,
        optional_backend_overrides={"statsmodels": None},
    )
    manifest = artifact.as_manifest()

    assert manifest["status"] == "abstained"
    assert manifest["reason_codes"] == ["statsmodels_unavailable"]
    assert manifest["unavailable_reason"] == "statsmodels_unavailable"
    assert manifest["transition_matrix"] == []
    assert manifest["smoothed_probabilities"] == []
    assert manifest["expected_durations"] == []
    assert manifest["convergence_diagnostics"]["converged"] is False


def test_weakly_separated_regimes_emit_identifiability_reason() -> None:
    regime_switching = _regime_switching_module()
    fake_backend = _fake_statsmodels_backend(
        transition_matrix=((0.55, 0.45), (0.45, 0.55)),
        smoothed_probabilities=(
            (0.53, 0.47),
            (0.52, 0.48),
            (0.48, 0.52),
            (0.47, 0.53),
        ),
        expected_durations=(2.222222222222, 2.222222222222),
        converged=True,
    )

    artifact = regime_switching.fit_regime_switching(
        series=(1.0, 1.01, 1.02, 1.03),
        series_id="weakly_separated",
        n_regimes=2,
        min_separation=1.0,
        optional_backend_overrides={"statsmodels": fake_backend},
    )
    manifest = artifact.as_manifest()

    assert manifest["status"] == "abstained"
    assert manifest["reason_codes"] == ["weak_regime_identifiability"]
    assert manifest["diagnostics"]["minimum_regime_separation"] < 1.0


def test_truth_regimes_emit_posterior_probability_brier_metrics() -> None:
    regime_switching = _regime_switching_module()
    fake_backend = _fake_statsmodels_backend(
        transition_matrix=((0.88, 0.12), (0.10, 0.90)),
        smoothed_probabilities=(
            (0.9, 0.1),
            (0.8, 0.2),
            (0.2, 0.8),
            (0.1, 0.9),
            (0.7, 0.3),
        ),
        expected_durations=(8.333333333333, 10.0),
        converged=True,
    )

    artifact = regime_switching.fit_regime_switching(
        series=(0.0, 0.2, 4.9, 5.1, 0.1),
        series_id="truth_labeled_synthetic",
        n_regimes=2,
        truth_regimes=(0, 0, 1, 1, 0),
        min_separation=0.25,
        optional_backend_overrides={"statsmodels": fake_backend},
    )
    calibration = artifact.as_manifest()["posterior_calibration"]

    assert calibration["status"] == "evaluated"
    assert calibration["n_observations"] == 5
    assert calibration["brier_score"] == pytest.approx(0.038)
    assert calibration["multiclass_brier_score"] == pytest.approx(0.076)
    assert calibration["accuracy"] == 1.0


def _fake_statsmodels_backend(
    *,
    transition_matrix: tuple[tuple[float, ...], ...],
    smoothed_probabilities: tuple[tuple[float, ...], ...],
    expected_durations: tuple[float, ...],
    converged: bool,
) -> Any:
    fake_transition_matrix = transition_matrix
    fake_smoothed_probabilities = smoothed_probabilities
    fake_expected_durations = expected_durations
    fake_converged = converged

    class _FakeResult:
        regime_transition = fake_transition_matrix
        smoothed_marginal_probabilities = fake_smoothed_probabilities
        expected_durations = fake_expected_durations
        llf = -11.25
        aic = 31.5
        bic = 34.0
        mle_retvals = {
            "converged": fake_converged,
            "iterations": 14,
            "warnflag": 0 if fake_converged else 1,
        }

    class _FakeMarkovRegression:
        def __init__(self, endog: Any, **kwargs: Any) -> None:
            self.endog = endog
            self.kwargs = kwargs

        def fit(self, **_: Any) -> _FakeResult:
            return _FakeResult()

    return SimpleNamespace(MarkovRegression=_FakeMarkovRegression)
