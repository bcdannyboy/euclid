from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class StabilityDiagnosticArtifact:
    artifact_id: str
    series_id: str
    status: str
    reason_codes: tuple[str, ...]
    backend: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    alpha: float = 0.05
    method: str = "cusum_recursive_residuals"
    diagnostic_statistic: float | None = None
    p_value: float | None = None
    instability_detected: bool = False
    unavailable_reason: str | None = None
    evidence_role: str = "diagnostic_only"
    is_law_claim: bool = False
    claim_scope: str = "diagnostic_only"

    def as_manifest(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "schema_name": "stability_diagnostic_artifact@1.0.0",
            "artifact_id": self.artifact_id,
            "series_id": self.series_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "backend": self.backend,
            "method": self.method,
            "instability_detected": self.instability_detected,
            "diagnostic_statistic": self.diagnostic_statistic,
            "p_value": self.p_value,
            "diagnostics": _jsonable(self.diagnostics),
            "alpha": _stable_float(self.alpha),
            "evidence_role": self.evidence_role,
            "is_law_claim": self.is_law_claim,
            "claim_scope": self.claim_scope,
            "may_publish_stationary_law_claim": False,
        }
        if self.unavailable_reason is not None:
            manifest["unavailable_reason"] = self.unavailable_reason
        return manifest


def run_stability_diagnostic(
    *,
    observations: Sequence[float],
    series_id: str = "series",
    design_matrix: Sequence[Sequence[float]] | None = None,
    method: str = "cusum_recursive_residuals",
    significance_level: float | None = None,
    min_observations: int = 8,
    alpha: float = 0.05,
    optional_backend_overrides: Mapping[str, Any] | None = None,
) -> StabilityDiagnosticArtifact:
    resolved_alpha = (
        float(significance_level) if significance_level is not None else float(alpha)
    )
    values = _finite_tuple(observations)
    backend = _statsmodels_backend(optional_backend_overrides)
    if backend is None:
        return _artifact(
            series_id=series_id,
            status="abstained",
            reason_codes=("statsmodels_unavailable",),
            backend="statsmodels",
            diagnostics={},
            alpha=resolved_alpha,
            unavailable_reason="statsmodels_unavailable",
            claim_scope="not_a_law_claim",
        )
    if len(values) < int(min_observations):
        return _artifact(
            series_id=series_id,
            status="abstained",
            reason_codes=("insufficient_observations",),
            backend="statsmodels",
            diagnostics={},
            alpha=resolved_alpha,
        )

    sm = backend.api
    diagnostic = backend.diagnostic
    y = np.asarray(values, dtype=float)
    exog = _design_matrix(
        observations=values,
        design_matrix=design_matrix,
        statsmodels_api=sm,
    )
    try:
        fit = sm.OLS(y, exog).fit()
        (
            cusum_statistic,
            cusum_p_value,
            critical_values,
        ) = diagnostic.breaks_cusumolsresid(fit.resid, ddof=int(fit.df_model) + 1)
        recursive = diagnostic.recursive_olsresiduals(
            fit,
            alpha=1.0 - float(resolved_alpha),
        )
    except Exception as exc:  # pragma: no cover - backend-specific numerical guard.
        return _artifact(
            series_id=series_id,
            status="abstained",
            reason_codes=("stability_diagnostic_unavailable",),
            backend="statsmodels",
            diagnostics={"error": exc.__class__.__name__},
            alpha=resolved_alpha,
            unavailable_reason="stability_diagnostic_unavailable",
        )

    recursive_cusum = np.asarray(recursive[-2], dtype=float)
    diagnostic_statistic = _stable_float(float(cusum_statistic))
    p_value = _stable_float(float(cusum_p_value))
    diagnostics = {
        "cusum_ols_residuals": {
            "statistic": diagnostic_statistic,
            "p_value": p_value,
            "critical_values": [
                {"level": int(level), "value": _stable_float(float(value))}
                for level, value in critical_values
            ],
        },
        "recursive_residuals": {
            "max_abs_cusum": _stable_float(
                float(np.nanmax(np.abs(recursive_cusum)))
            ),
            "point_count": int(recursive_cusum.size),
        },
    }
    if float(cusum_p_value) < float(resolved_alpha):
        return _artifact(
            series_id=series_id,
            status="failed",
            reason_codes=(
                "recursive_residual_instability_detected",
                "stability_test_failed",
                "instability_evidence_unresolved",
            ),
            backend="statsmodels",
            diagnostics=diagnostics,
            alpha=resolved_alpha,
            method=method,
            diagnostic_statistic=diagnostic_statistic,
            p_value=p_value,
            instability_detected=True,
        )
    return _artifact(
        series_id=series_id,
        status="passed",
        reason_codes=(),
        backend="statsmodels",
        diagnostics=diagnostics,
        alpha=resolved_alpha,
        method=method,
        diagnostic_statistic=diagnostic_statistic,
        p_value=p_value,
        instability_detected=False,
    )


def _design_matrix(
    *,
    observations: tuple[float, ...],
    design_matrix: Sequence[Sequence[float]] | None,
    statsmodels_api: Any,
) -> np.ndarray:
    if design_matrix is not None:
        exog = np.asarray(design_matrix, dtype=float)
        if exog.ndim == 2 and exog.shape[0] == len(observations):
            return exog
    trend = np.arange(len(observations), dtype=float)
    return statsmodels_api.add_constant(trend)


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
            api=getattr(override, "api", override),
            diagnostic=getattr(override, "diagnostic", override),
        )
    try:
        return SimpleNamespace(
            api=importlib.import_module("statsmodels.api"),
            diagnostic=importlib.import_module("statsmodels.stats.diagnostic"),
        )
    except ImportError:
        pass
    try:
        linear_model = importlib.import_module("statsmodels.regression.linear_model")
        tools = importlib.import_module("statsmodels.tools.tools")
        return SimpleNamespace(
            api=SimpleNamespace(
                OLS=linear_model.OLS,
                add_constant=tools.add_constant,
            ),
            diagnostic=importlib.import_module("statsmodels.stats.diagnostic"),
        )
    except ImportError:
        return None


def _artifact(
    *,
    series_id: str,
    status: str,
    reason_codes: tuple[str, ...],
    backend: str,
    diagnostics: Mapping[str, Any],
    alpha: float,
    unavailable_reason: str | None = None,
    method: str = "cusum_recursive_residuals",
    diagnostic_statistic: float | None = None,
    p_value: float | None = None,
    instability_detected: bool = False,
    claim_scope: str = "diagnostic_only",
) -> StabilityDiagnosticArtifact:
    payload = {
        "alpha": _stable_float(alpha),
        "backend": backend,
        "diagnostics": _jsonable(diagnostics),
        "reason_codes": list(reason_codes),
        "series_id": str(series_id),
        "status": status,
        "unavailable_reason": unavailable_reason,
    }
    return StabilityDiagnosticArtifact(
        artifact_id=f"stability-diagnostic:{_digest(payload)}",
        series_id=str(series_id),
        status=status,
        reason_codes=reason_codes,
        backend=backend,
        diagnostics=diagnostics,
        alpha=_stable_float(alpha),
        method=method,
        diagnostic_statistic=diagnostic_statistic,
        p_value=p_value,
        instability_detected=instability_detected,
        unavailable_reason=unavailable_reason,
        claim_scope=claim_scope,
    )


def _finite_tuple(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        return ()
    return result


def _jsonable(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, default=str))


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = ["StabilityDiagnosticArtifact", "run_stability_diagnostic"]
