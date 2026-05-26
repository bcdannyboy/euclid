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

_SCHEMA_NAME = "state_space_artifact@1.0.0"
_LANE_ID = "state_space_local_level_v1"
_MODEL = "local_level"
_BACKEND = "statsmodels"
_PASSED = "passed"
_FAILED = "failed"
_ADAPTER_UNAVAILABLE = "adapter_unavailable"


@dataclass(frozen=True)
class StateSpaceArtifact:
    artifact_id: str
    series_id: str
    status: str
    reason_codes: tuple[str, ...]
    sample_count: int
    filtered_state: tuple[float, ...] = ()
    smoothed_state: tuple[float, ...] = ()
    state_covariance: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    innovations: tuple[float, ...] = ()
    log_likelihood: float | None = None
    innovation_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    latent_recovery: Mapping[str, Any] | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    unavailable_reason: str | None = None
    model: str = _MODEL
    backend: str = _BACKEND
    lane_id: str = _LANE_ID
    evidence_role: str = "nonstationarity_handling"
    claim_scope: str = "nonstationary_lane_evidence"
    law_claim_allowed: bool = False

    @property
    def promotion_allowed(self) -> bool:
        return self.status == _PASSED

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": _SCHEMA_NAME,
            "artifact_type": "state_space",
            "artifact_id": self.artifact_id,
            "series_id": self.series_id,
            "lane_id": self.lane_id,
            "model": self.model,
            "backend": self.backend,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "promotion_allowed": self.promotion_allowed,
            "sample_count": int(self.sample_count),
            "filtered_state": list(self.filtered_state),
            "smoothed_state": list(self.smoothed_state),
            "state_covariance": {
                key: list(value) for key, value in sorted(self.state_covariance.items())
            },
            "innovations": list(self.innovations),
            "log_likelihood": self.log_likelihood,
            "innovation_diagnostics": _stable_mapping(self.innovation_diagnostics),
            "latent_recovery": (
                None
                if self.latent_recovery is None
                else _stable_mapping(self.latent_recovery)
            ),
            "parameters": _stable_mapping(self.parameters),
            "metadata": _stable_mapping(self.metadata),
            "evidence_role": self.evidence_role,
            "claim_scope": self.claim_scope,
            "law_claim_allowed": self.law_claim_allowed,
            "unavailable_reason": self.unavailable_reason,
        }


def fit_local_level_state_space(
    *,
    observations: Sequence[float],
    latent_truth: Sequence[float] | None = None,
    series_id: str = "series",
    whiteness_lag: int = 10,
    whiteness_alpha: float = 0.05,
    optional_backend_overrides: Mapping[str, Any] | None = None,
) -> StateSpaceArtifact:
    values = _finite_values(observations)
    backend = _statsmodels_backend(optional_backend_overrides)
    resolved_lag = max(1, int(whiteness_lag))
    resolved_alpha = float(whiteness_alpha)
    if backend is None:
        return _artifact(
            series_id=series_id,
            status=_ADAPTER_UNAVAILABLE,
            reason_codes=("statsmodels_state_space_backend_unavailable",),
            sample_count=len(values),
            innovation_diagnostics={
                "whiteness_lag": resolved_lag,
                "alpha": _stable_float(resolved_alpha),
                "whiteness_passed": False,
                "ljung_box_statistic": None,
                "ljung_box_p_value": None,
            },
            parameters={
                "whiteness_alpha": _stable_float(resolved_alpha),
                "whiteness_lag": resolved_lag,
            },
            unavailable_reason="statsmodels_state_space_backend_unavailable",
        )

    y = np.asarray(values, dtype=float)
    try:
        model = backend.unobserved_components(y, level="local level")
        fit = model.fit(disp=False)
    except Exception as exc:  # pragma: no cover - backend-specific numerical guard.
        return _artifact(
            series_id=series_id,
            status=_FAILED,
            reason_codes=("state_space_fit_failed",),
            sample_count=len(values),
            innovation_diagnostics={
                "whiteness_lag": resolved_lag,
                "alpha": _stable_float(resolved_alpha),
                "whiteness_passed": False,
                "ljung_box_statistic": None,
                "ljung_box_p_value": None,
            },
            parameters={
                "whiteness_alpha": _stable_float(resolved_alpha),
                "whiteness_lag": resolved_lag,
            },
            metadata={"fit_error": exc.__class__.__name__},
        )

    filtered_state = _series_from_state(fit.filtered_state)
    smoothed_state = _series_from_state(fit.smoothed_state)
    filtered_covariance = _series_from_state_cov(fit.filtered_state_cov)
    smoothed_covariance = _series_from_state_cov(fit.smoothed_state_cov)
    innovations = _series_from_state(fit.filter_results.forecasts_error)
    standardized_innovations = _finite_array(
        _series_from_state(fit.filter_results.standardized_forecasts_error)
    )
    whiteness = _innovation_whiteness(
        backend=backend,
        standardized_innovations=standardized_innovations,
        lag=resolved_lag,
        alpha=resolved_alpha,
    )
    status = _PASSED if whiteness["whiteness_passed"] else _FAILED
    reason_codes = (
        () if status == _PASSED else ("state_space_innovation_whiteness_failed",)
    )
    latent_recovery = _latent_recovery(
        smoothed_state=smoothed_state,
        latent_truth=latent_truth,
    )
    parameters = {
        "fit_method": "statsmodels.UnobservedComponents",
        "level": "local level",
        "whiteness_alpha": _stable_float(resolved_alpha),
        "whiteness_lag": resolved_lag,
    }
    return _artifact(
        series_id=series_id,
        status=status,
        reason_codes=reason_codes,
        sample_count=len(values),
        filtered_state=filtered_state,
        smoothed_state=smoothed_state,
        state_covariance={
            "filtered": filtered_covariance,
            "smoothed": smoothed_covariance,
        },
        innovations=innovations,
        log_likelihood=_stable_float(float(fit.llf)),
        innovation_diagnostics=whiteness,
        latent_recovery=latent_recovery,
        parameters=parameters,
        metadata={
            "nobs": int(getattr(fit, "nobs", len(values))),
            "param_names": tuple(str(name) for name in getattr(fit, "param_names", ())),
            "params": _stable_float_tuple(tuple(float(value) for value in fit.params)),
        },
    )


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
            diagnostic=getattr(override, "diagnostic", override),
            unobserved_components=getattr(
                override,
                "unobserved_components",
                getattr(override, "UnobservedComponents", override),
            ),
        )
    try:
        structural = importlib.import_module("statsmodels.tsa.statespace.structural")
        diagnostic = importlib.import_module("statsmodels.stats.diagnostic")
    except ImportError:
        return None
    return SimpleNamespace(
        diagnostic=diagnostic,
        unobserved_components=structural.UnobservedComponents,
    )


def _innovation_whiteness(
    *,
    backend: SimpleNamespace,
    standardized_innovations: np.ndarray,
    lag: int,
    alpha: float,
) -> dict[str, Any]:
    finite = standardized_innovations[np.isfinite(standardized_innovations)]
    if finite.size <= lag:
        return {
            "whiteness_lag": int(lag),
            "alpha": _stable_float(alpha),
            "whiteness_passed": False,
            "ljung_box_statistic": None,
            "ljung_box_p_value": None,
            "effective_sample_size": int(finite.size),
        }
    diagnostic = backend.diagnostic.acorr_ljungbox(
        finite,
        lags=[int(lag)],
        return_df=True,
    )
    statistic = float(diagnostic["lb_stat"].iloc[-1])
    p_value = float(diagnostic["lb_pvalue"].iloc[-1])
    return {
        "whiteness_lag": int(lag),
        "alpha": _stable_float(alpha),
        "whiteness_passed": bool(p_value > alpha),
        "ljung_box_statistic": _stable_float(statistic),
        "ljung_box_p_value": _stable_float(p_value),
        "effective_sample_size": int(finite.size),
    }


def _latent_recovery(
    *,
    smoothed_state: tuple[float, ...],
    latent_truth: Sequence[float] | None,
) -> dict[str, Any] | None:
    if latent_truth is None:
        return None
    truth = _finite_values(latent_truth)
    pair_count = min(len(smoothed_state), len(truth))
    if pair_count == 0:
        return {"sample_count": 0, "rmse": None, "max_abs_error": None}
    estimate = np.asarray(smoothed_state[:pair_count], dtype=float)
    expected = np.asarray(truth[:pair_count], dtype=float)
    errors = estimate - expected
    return {
        "sample_count": int(pair_count),
        "rmse": _stable_float(float(np.sqrt(np.mean(errors * errors)))),
        "max_abs_error": _stable_float(float(np.max(np.abs(errors)))),
    }


def _artifact(
    *,
    series_id: str,
    status: str,
    reason_codes: tuple[str, ...],
    sample_count: int,
    filtered_state: tuple[float, ...] = (),
    smoothed_state: tuple[float, ...] = (),
    state_covariance: Mapping[str, tuple[float, ...]] | None = None,
    innovations: tuple[float, ...] = (),
    log_likelihood: float | None = None,
    innovation_diagnostics: Mapping[str, Any] | None = None,
    latent_recovery: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    unavailable_reason: str | None = None,
) -> StateSpaceArtifact:
    payload = {
        "backend": _BACKEND,
        "filtered_state": filtered_state,
        "innovation_diagnostics": innovation_diagnostics or {},
        "innovations": innovations,
        "log_likelihood": log_likelihood,
        "model": _MODEL,
        "reason_codes": reason_codes,
        "sample_count": int(sample_count),
        "schema_name": _SCHEMA_NAME,
        "series_id": str(series_id),
        "smoothed_state": smoothed_state,
        "state_covariance": state_covariance or {},
        "status": status,
        "unavailable_reason": unavailable_reason,
    }
    return StateSpaceArtifact(
        artifact_id=f"state-space:{_digest(_jsonable(payload))}",
        series_id=str(series_id),
        status=status,
        reason_codes=tuple(reason_codes),
        sample_count=int(sample_count),
        filtered_state=_stable_float_tuple(filtered_state),
        smoothed_state=_stable_float_tuple(smoothed_state),
        state_covariance={
            key: _stable_float_tuple(value)
            for key, value in (state_covariance or {}).items()
        },
        innovations=_stable_float_tuple(innovations),
        log_likelihood=log_likelihood,
        innovation_diagnostics=innovation_diagnostics or {},
        latent_recovery=latent_recovery,
        parameters=parameters or {},
        metadata=metadata or {},
        unavailable_reason=unavailable_reason,
    )


def _series_from_state(values: Any) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return (float(array),)
    if array.ndim == 1:
        return tuple(float(value) for value in array)
    return tuple(float(value) for value in array[0])


def _series_from_state_cov(values: Any) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return (float(array),)
    if array.ndim == 1:
        return tuple(float(value) for value in array)
    if array.ndim == 2:
        return tuple(float(value) for value in array[0])
    return tuple(float(value) for value in array[0, 0])


def _finite_values(values: Sequence[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return tuple(result)


def _finite_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=float,
    )


def _stable_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _stable_value(mapping[key]) for key in sorted(mapping)}


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _stable_mapping(value)
    if isinstance(value, tuple):
        return [_stable_value(item) for item in value]
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    if isinstance(value, float):
        return _stable_float(value)
    return value


def _stable_float_tuple(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(_stable_float(value) for value in values)


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


def _jsonable(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, default=str))


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = ["StateSpaceArtifact", "fit_local_level_state_space"]
