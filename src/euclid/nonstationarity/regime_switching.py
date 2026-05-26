from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

_PASSED = "passed"
_ABSTAINED = "abstained"
_ADAPTER_UNAVAILABLE = "adapter_unavailable"
_VALID_GIVEN_REGIME = "valid_given_regime"
_NOT_A_LAW_CLAIM = "not_a_law_claim"


@dataclass(frozen=True)
class RegimeSwitchingArtifact:
    artifact_id: str
    series_id: str
    status: str
    reason_codes: tuple[str, ...]
    backend: str
    method: str
    model_class: str
    n_regimes: int
    transition_matrix: tuple[tuple[float, ...], ...] = ()
    smoothed_probabilities: tuple[tuple[float, ...], ...] = ()
    expected_durations: tuple[float, ...] = ()
    convergence_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    posterior_calibration: Mapping[str, Any] = field(default_factory=dict)
    unavailable_reason: str | None = None
    evidence_role: str = "diagnostic_only"
    is_law_claim: bool = False
    regime_conditioned_scope: str = _VALID_GIVEN_REGIME
    claim_scope: str = _NOT_A_LAW_CLAIM

    @property
    def promotion_allowed(self) -> bool:
        return self.status == _PASSED and not self.reason_codes

    def as_manifest(self) -> dict[str, Any]:
        regime_ids = tuple(range(int(self.n_regimes)))
        regime_conditioned_claim_allowed = bool(
            self.promotion_allowed and self.claim_scope == _VALID_GIVEN_REGIME
        )
        manifest: dict[str, Any] = {
            "schema_name": "regime_switching_artifact@1.0.0",
            "artifact_type": "regime_switching",
            "artifact_id": self.artifact_id,
            "series_id": self.series_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "backend": self.backend,
            "method": self.method,
            "model_class": self.model_class,
            "n_regimes": self.n_regimes,
            "regime_count": self.n_regimes,
            "regime_ids": list(regime_ids),
            "transition_matrix": _jsonable(self.transition_matrix),
            "smoothed_probabilities": _jsonable(self.smoothed_probabilities),
            "expected_durations": _jsonable(self.expected_durations),
            "convergence_diagnostics": _jsonable(self.convergence_diagnostics),
            "diagnostics": _jsonable(self.diagnostics),
            "posterior_calibration": _jsonable(self.posterior_calibration),
            "promotion_allowed": self.promotion_allowed,
            "evidence_role": self.evidence_role,
            "is_law_claim": self.is_law_claim,
            "regime_conditioned_scope": self.regime_conditioned_scope,
            "claim_scope": self.claim_scope,
            "unconditional_law_claim_allowed": False,
            "regime_conditioned_law_claim_allowed": regime_conditioned_claim_allowed,
            "may_publish_regime_conditioned_law_claim": (
                regime_conditioned_claim_allowed
            ),
            "scope_qualifier": {
                "validity": self.claim_scope,
                "regime_ids": list(regime_ids),
            },
        }
        if self.unavailable_reason is not None:
            manifest["unavailable_reason"] = self.unavailable_reason
        return manifest


def fit_regime_switching(
    series: Sequence[float] | None = None,
    *,
    observations: Sequence[float] | None = None,
    series_id: str = "series",
    n_regimes: int = 2,
    truth_regimes: Sequence[int] | None = None,
    min_separation: float = 0.25,
    model_type: str = "markov_regression",
    maxiter: int = 100,
    trend: str = "c",
    order: int = 1,
    switching_variance: bool = True,
    optional_backend_overrides: Mapping[str, Any] | None = None,
    _backend_unavailable_status: str = _ABSTAINED,
    _backend_unavailable_reason_code: str = "statsmodels_unavailable",
) -> RegimeSwitchingArtifact:
    values = _finite_values(series if series is not None else observations)
    regime_count = int(n_regimes)
    backend = _statsmodels_backend(optional_backend_overrides)
    model_name = _model_class_name(model_type)
    method = _method_name(model_type)

    if backend is None:
        return _artifact(
            series_id=series_id,
            status=_backend_unavailable_status,
            reason_codes=(_backend_unavailable_reason_code,),
            backend="statsmodels",
            method=method,
            model_class=model_name,
            n_regimes=max(regime_count, 0),
            unavailable_reason=_backend_unavailable_reason_code,
            convergence_diagnostics={"converged": False},
        )
    if regime_count < 2:
        return _artifact(
            series_id=series_id,
            status=_ABSTAINED,
            reason_codes=("invalid_regime_count",),
            backend="statsmodels",
            method=method,
            model_class=model_name,
            n_regimes=max(regime_count, 0),
            convergence_diagnostics={"converged": False},
        )
    if len(values) < 2 * regime_count:
        return _artifact(
            series_id=series_id,
            status=_ABSTAINED,
            reason_codes=("insufficient_observations",),
            backend="statsmodels",
            method=method,
            model_class=model_name,
            n_regimes=regime_count,
            diagnostics={"n_observations": len(values)},
            convergence_diagnostics={"converged": False},
        )

    try:
        result = _fit_markov_model(
            backend=backend,
            values=values,
            n_regimes=regime_count,
            model_type=model_type,
            maxiter=maxiter,
            trend=trend,
            order=order,
            switching_variance=switching_variance,
        )
    except Exception as exc:  # pragma: no cover - backend-specific numerical guard.
        return _artifact(
            series_id=series_id,
            status=_ABSTAINED,
            reason_codes=("regime_switching_fit_failed",),
            backend="statsmodels",
            method=method,
            model_class=model_name,
            n_regimes=regime_count,
            diagnostics={"error": exc.__class__.__name__},
            convergence_diagnostics={"converged": False},
            unavailable_reason="regime_switching_fit_failed",
        )

    transition_matrix = _transition_matrix(result, regime_count)
    smoothed_probabilities = _smoothed_probabilities(result, regime_count, len(values))
    expected_durations = _expected_durations(result, transition_matrix, regime_count)
    convergence_diagnostics = _convergence_diagnostics(result)
    diagnostics = _identifiability_diagnostics(
        values=values,
        smoothed_probabilities=smoothed_probabilities,
        min_separation=float(min_separation),
    )
    posterior_calibration = _posterior_calibration(
        smoothed_probabilities=smoothed_probabilities,
        truth_regimes=truth_regimes,
        n_regimes=regime_count,
    )

    reason_codes: list[str] = []
    if not convergence_diagnostics.get("converged", False):
        reason_codes.append("regime_switching_not_converged")
    if not smoothed_probabilities:
        reason_codes.append("regime_switching_posterior_unavailable")
    if diagnostics.get("weak_regime_identifiability", False):
        reason_codes.append("weak_regime_identifiability")

    return _artifact(
        series_id=series_id,
        status=_ABSTAINED if reason_codes else _PASSED,
        reason_codes=tuple(reason_codes),
        backend="statsmodels",
        method=method,
        model_class=model_name,
        n_regimes=regime_count,
        transition_matrix=transition_matrix,
        smoothed_probabilities=smoothed_probabilities,
        expected_durations=expected_durations,
        convergence_diagnostics=convergence_diagnostics,
        diagnostics=diagnostics,
        posterior_calibration=posterior_calibration,
        claim_scope=_VALID_GIVEN_REGIME if not reason_codes else _NOT_A_LAW_CLAIM,
    )


def fit_markov_switching_regimes(
    *,
    observations: Sequence[float],
    series_id: str = "series",
    regime_count: int = 2,
    truth_regimes: Sequence[int] | None = None,
    method: str = "markov_regression",
    min_regime_mean_separation: float = 0.25,
    maxiter: int = 100,
    trend: str = "c",
    order: int = 1,
    switching_variance: bool = True,
    optional_backend_overrides: Mapping[str, Any] | None = None,
) -> RegimeSwitchingArtifact:
    return fit_regime_switching(
        observations=observations,
        series_id=series_id,
        n_regimes=regime_count,
        truth_regimes=truth_regimes,
        min_separation=min_regime_mean_separation,
        model_type=method,
        maxiter=maxiter,
        trend=trend,
        order=order,
        switching_variance=switching_variance,
        optional_backend_overrides=optional_backend_overrides,
        _backend_unavailable_status=_ADAPTER_UNAVAILABLE,
        _backend_unavailable_reason_code=(
            "statsmodels_markov_switching_backend_unavailable"
        ),
    )


def _fit_markov_model(
    *,
    backend: SimpleNamespace,
    values: tuple[float, ...],
    n_regimes: int,
    model_type: str,
    maxiter: int,
    trend: str,
    order: int,
    switching_variance: bool,
) -> Any:
    y = np.asarray(values, dtype=float)
    if _model_class_name(model_type) == "MarkovAutoregression":
        model_class = backend.MarkovAutoregression
        model = model_class(
            y,
            k_regimes=n_regimes,
            order=int(order),
            trend=trend,
            switching_ar=True,
            switching_variance=bool(switching_variance),
        )
    else:
        model_class = backend.MarkovRegression
        model = model_class(
            y,
            k_regimes=n_regimes,
            trend=trend,
            switching_variance=bool(switching_variance),
        )
    return model.fit(disp=False, maxiter=int(maxiter))


def _statsmodels_backend(
    optional_backend_overrides: Mapping[str, Any] | None,
) -> SimpleNamespace | None:
    if (
        optional_backend_overrides is not None
        and "statsmodels" in optional_backend_overrides
    ):
        override = optional_backend_overrides["statsmodels"]
        if override is None:
            return None
        return SimpleNamespace(
            MarkovRegression=_override_attr(override, "MarkovRegression"),
            MarkovAutoregression=_override_attr(override, "MarkovAutoregression"),
        )

    try:
        markov_regression = importlib.import_module(
            "statsmodels.tsa.regime_switching.markov_regression"
        )
    except ImportError:
        return None
    try:
        markov_autoregression = importlib.import_module(
            "statsmodels.tsa.regime_switching.markov_autoregression"
        )
    except ImportError:
        markov_autoregression = None

    return SimpleNamespace(
        MarkovRegression=getattr(markov_regression, "MarkovRegression"),
        MarkovAutoregression=(
            getattr(markov_autoregression, "MarkovAutoregression")
            if markov_autoregression is not None
            else None
        ),
    )


def _override_attr(override: Any, name: str) -> Any:
    if isinstance(override, Mapping) and name in override:
        return override[name]
    if hasattr(override, name):
        return getattr(override, name)
    if callable(override) and name == "MarkovRegression":
        return override
    return None


def _transition_matrix(result: Any, n_regimes: int) -> tuple[tuple[float, ...], ...]:
    raw = getattr(result, "regime_transition", None)
    if raw is None:
        model = getattr(result, "model", None)
        params = getattr(result, "params", None)
        matrix_fn = getattr(model, "regime_transition_matrix", None)
        if matrix_fn is not None and params is not None:
            raw = matrix_fn(params)
    if raw is None:
        return ()

    matrix = _coerce_matrix(raw, n_regimes)
    if matrix.size == 0:
        return ()
    matrix = _orient_row_stochastic(matrix)
    return tuple(tuple(_stable_float(item) for item in row) for row in matrix)


def _coerce_matrix(raw: Any, n_regimes: int) -> np.ndarray:
    matrix = np.asarray(raw, dtype=float)
    if matrix.ndim == 3:
        if matrix.shape[0] == n_regimes and matrix.shape[1] == n_regimes:
            matrix = matrix[:, :, -1]
        elif matrix.shape[1] == n_regimes and matrix.shape[2] == n_regimes:
            matrix = matrix[-1, :, :]
    if matrix.ndim != 2 or matrix.shape != (n_regimes, n_regimes):
        return np.asarray((), dtype=float)
    if not np.all(np.isfinite(matrix)):
        return np.asarray((), dtype=float)
    return matrix


def _orient_row_stochastic(matrix: np.ndarray) -> np.ndarray:
    row_error = float(np.mean(np.abs(matrix.sum(axis=1) - 1.0)))
    column_error = float(np.mean(np.abs(matrix.sum(axis=0) - 1.0)))
    oriented = matrix.T if column_error < row_error else matrix.copy()
    row_sums = oriented.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        return oriented
    return oriented / row_sums


def _smoothed_probabilities(
    result: Any,
    n_regimes: int,
    n_observations: int,
) -> tuple[tuple[float, ...], ...]:
    raw = getattr(result, "smoothed_marginal_probabilities", None)
    if raw is None:
        return ()
    probabilities = np.asarray(raw, dtype=float)
    if probabilities.ndim != 2:
        return ()
    if probabilities.shape[1] != n_regimes and probabilities.shape[0] == n_regimes:
        probabilities = probabilities.T
    if probabilities.shape[1] != n_regimes:
        return ()
    if probabilities.shape[0] > n_observations:
        probabilities = probabilities[-n_observations:, :]
    if not np.all(np.isfinite(probabilities)):
        return ()

    normalized_rows: list[tuple[float, ...]] = []
    for row in probabilities:
        row_sum = float(np.sum(row))
        if row_sum <= 0.0:
            return ()
        normalized_rows.append(tuple(_stable_float(item / row_sum) for item in row))
    return tuple(normalized_rows)


def _expected_durations(
    result: Any,
    transition_matrix: tuple[tuple[float, ...], ...],
    n_regimes: int,
) -> tuple[float, ...]:
    raw = getattr(result, "expected_durations", None)
    if raw is not None:
        durations = np.asarray(raw, dtype=float).reshape(-1)
        if durations.size == n_regimes and np.all(np.isfinite(durations)):
            return tuple(_stable_float(item) for item in durations)
    if not transition_matrix:
        return ()
    durations_from_transition: list[float] = []
    for index, row in enumerate(transition_matrix):
        stay_probability = float(row[index])
        if stay_probability >= 1.0:
            return ()
        durations_from_transition.append(1.0 / max(1.0 - stay_probability, 1e-12))
    return tuple(_stable_float(item) for item in durations_from_transition)


def _convergence_diagnostics(result: Any) -> dict[str, Any]:
    raw_retvals = getattr(result, "mle_retvals", {})
    retvals = dict(raw_retvals) if isinstance(raw_retvals, Mapping) else {}
    converged = bool(retvals.get("converged", getattr(result, "converged", False)))
    diagnostics: dict[str, Any] = {
        "converged": converged,
    }
    if "iterations" in retvals:
        diagnostics["iterations"] = int(retvals["iterations"])
    elif "iter" in retvals:
        diagnostics["iterations"] = int(retvals["iter"])
    else:
        diagnostics["iterations"] = 0
    if "warnflag" in retvals:
        diagnostics["warnflag"] = int(retvals["warnflag"])
    for name in ("llf", "aic", "bic"):
        value = getattr(result, name, None)
        if value is not None:
            diagnostics[name] = _stable_float(value)
    if "llf" in diagnostics:
        diagnostics["log_likelihood"] = diagnostics["llf"]
    return diagnostics


def _identifiability_diagnostics(
    *,
    values: tuple[float, ...],
    smoothed_probabilities: tuple[tuple[float, ...], ...],
    min_separation: float,
) -> dict[str, Any]:
    if not smoothed_probabilities:
        return {
            "minimum_regime_separation": None,
            "mean_posterior_confidence": None,
            "weak_regime_identifiability": True,
        }

    probabilities = np.asarray(smoothed_probabilities, dtype=float)
    y = np.asarray(values[-probabilities.shape[0] :], dtype=float)
    means: list[float] = []
    for regime_index in range(probabilities.shape[1]):
        weights = probabilities[:, regime_index]
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            return {
                "minimum_regime_separation": 0.0,
                "mean_posterior_confidence": _stable_float(
                    float(np.mean(np.max(probabilities, axis=1)))
                ),
                "weak_regime_identifiability": True,
            }
        means.append(float(np.sum(weights * y) / weight_sum))

    if len(means) < 2:
        minimum_separation = 0.0
    else:
        minimum_separation = min(
            abs(left - right)
            for left_index, left in enumerate(means)
            for right in means[left_index + 1 :]
        )
    mean_confidence = float(np.mean(np.max(probabilities, axis=1)))
    return {
        "minimum_regime_separation": _stable_float(minimum_separation),
        "mean_posterior_confidence": _stable_float(mean_confidence),
        "regime_weighted_means": [_stable_float(mean) for mean in means],
        "weak_regime_identifiability": bool(minimum_separation < min_separation),
    }


def _posterior_calibration(
    *,
    smoothed_probabilities: tuple[tuple[float, ...], ...],
    truth_regimes: Sequence[int] | None,
    n_regimes: int,
) -> dict[str, Any]:
    if truth_regimes is None:
        return {"status": "not_evaluated", "reason": "truth_regimes_not_provided"}
    if not smoothed_probabilities:
        return {"status": "not_evaluated", "reason": "posterior_unavailable"}

    probabilities = np.asarray(smoothed_probabilities, dtype=float)
    truth = np.asarray(tuple(int(value) for value in truth_regimes), dtype=int)
    if truth.size != probabilities.shape[0]:
        return {
            "status": "not_evaluated",
            "reason": "truth_regime_length_mismatch",
            "n_observations": int(probabilities.shape[0]),
            "n_truth_regimes": int(truth.size),
        }
    if np.any((truth < 0) | (truth >= n_regimes)):
        return {"status": "not_evaluated", "reason": "truth_regime_out_of_range"}

    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(truth.size), truth] = 1.0
    truth_probabilities = probabilities[np.arange(truth.size), truth]
    brier_score = float(np.mean((1.0 - truth_probabilities) ** 2))
    multiclass_brier_score = float(
        np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    )
    predictions = np.argmax(probabilities, axis=1)
    per_regime: list[dict[str, Any]] = []
    for regime_index in range(n_regimes):
        mask = truth == regime_index
        count = int(np.sum(mask))
        if count == 0:
            per_regime.append(
                {
                    "regime": regime_index,
                    "n_observations": 0,
                    "brier_score": None,
                    "mean_posterior_probability": None,
                }
            )
            continue
        regime_truth_probabilities = truth_probabilities[mask]
        per_regime.append(
            {
                "regime": regime_index,
                "n_observations": count,
                "brier_score": _stable_float(
                    float(np.mean((1.0 - regime_truth_probabilities) ** 2))
                ),
                "mean_posterior_probability": _stable_float(
                    float(np.mean(regime_truth_probabilities))
                ),
            }
        )

    return {
        "status": "evaluated",
        "n_observations": int(truth.size),
        "brier_score": _stable_float(brier_score),
        "multiclass_brier_score": _stable_float(multiclass_brier_score),
        "accuracy": _stable_float(float(np.mean(predictions == truth))),
        "mean_truth_probability": _stable_float(float(np.mean(truth_probabilities))),
        "per_regime": per_regime,
    }


def _artifact(
    *,
    series_id: str,
    status: str,
    reason_codes: tuple[str, ...],
    backend: str,
    method: str,
    model_class: str,
    n_regimes: int,
    transition_matrix: tuple[tuple[float, ...], ...] = (),
    smoothed_probabilities: tuple[tuple[float, ...], ...] = (),
    expected_durations: tuple[float, ...] = (),
    convergence_diagnostics: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    posterior_calibration: Mapping[str, Any] | None = None,
    unavailable_reason: str | None = None,
    claim_scope: str = _NOT_A_LAW_CLAIM,
) -> RegimeSwitchingArtifact:
    payload = {
        "backend": backend,
        "method": method,
        "model_class": model_class,
        "n_regimes": n_regimes,
        "reason_codes": list(reason_codes),
        "series_id": str(series_id),
        "status": status,
        "transition_matrix": transition_matrix,
        "unavailable_reason": unavailable_reason,
    }
    return RegimeSwitchingArtifact(
        artifact_id=f"regime-switching:{_digest(payload)}",
        series_id=str(series_id),
        status=status,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        backend=backend,
        method=method,
        model_class=model_class,
        n_regimes=n_regimes,
        transition_matrix=transition_matrix,
        smoothed_probabilities=smoothed_probabilities,
        expected_durations=expected_durations,
        convergence_diagnostics=dict(convergence_diagnostics or {}),
        diagnostics=dict(diagnostics or {}),
        posterior_calibration=dict(posterior_calibration or {}),
        unavailable_reason=unavailable_reason,
        claim_scope=claim_scope,
    )


def _finite_values(values: Sequence[float] | None) -> tuple[float, ...]:
    if values is None:
        return ()
    result: list[float] = []
    for value in values:
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return tuple(result)


def _model_class_name(model_type: str) -> str:
    normalized = str(model_type).strip().lower().replace("-", "_")
    if normalized in {"markov_autoregression", "autoregression", "ar"}:
        return "MarkovAutoregression"
    return "MarkovRegression"


def _method_name(model_type: str) -> str:
    normalized = str(model_type).strip().lower().replace("-", "_")
    if normalized in {"markov_autoregression", "autoregression", "ar"}:
        return "markov_autoregression"
    return "markov_regression"


def _stable_float(value: Any) -> float:
    return float(round(float(value), 12))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return _stable_float(value)
    return value


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(payload),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


run_regime_switching_diagnostic = fit_regime_switching

__all__ = [
    "RegimeSwitchingArtifact",
    "fit_regime_switching",
    "fit_markov_switching_regimes",
    "run_regime_switching_diagnostic",
]
