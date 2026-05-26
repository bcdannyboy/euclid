from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS

from euclid.contracts.errors import ContractValidationError

_IDENTITY_FIELDS = ("origin_id", "horizon", "entity_id", "row_set_id")
_SUPPORTED_DECLARED_TEST_IDS = frozenset(
    {
        "diebold_mariano_hln_v1",
        "paired_stationary_block_bootstrap_v1",
        "giacomini_white_v1",
        "model_confidence_set_v1",
        "superior_predictive_ability_v1",
    }
)


@dataclass(frozen=True)
class PairedLossDifferentialStream:
    stream_id: str
    candidate_id: str
    baseline_id: str
    loss_id: str
    pairs: tuple[Mapping[str, Any], ...]
    loss_differentials: tuple[float, ...]
    raw_pair_count: int
    effective_sample_size: float
    block_count: int
    horizon_geometry: tuple[int, ...]
    replay_identity: str

    @classmethod
    def from_loss_rows(
        cls,
        *,
        stream_id: str,
        candidate_id: str,
        baseline_id: str,
        loss_id: str,
        candidate_rows: Sequence[Mapping[str, Any]],
        baseline_rows: Sequence[Mapping[str, Any]],
    ) -> "PairedLossDifferentialStream":
        candidate = tuple(candidate_rows)
        baseline = tuple(baseline_rows)
        if len(candidate) != len(baseline):
            raise ContractValidationError(
                code="paired_loss_identity_mismatch",
                message=(
                    "candidate and baseline loss streams must have the same row count"
                ),
                field_path="baseline_rows",
                details={
                    "field": "row_count",
                    "candidate_count": len(candidate),
                    "baseline_count": len(baseline),
                },
            )

        pairs: list[dict[str, Any]] = []
        differentials: list[float] = []
        for index, (candidate_row, baseline_row) in enumerate(
            zip(candidate, baseline, strict=True)
        ):
            _validate_pair_identity(
                candidate_row=candidate_row,
                baseline_row=baseline_row,
                index=index,
            )
            candidate_loss = _finite_float(
                candidate_row.get("loss"),
                field_path=f"candidate_rows[{index}].loss",
            )
            baseline_loss = _finite_float(
                baseline_row.get("loss"),
                field_path=f"baseline_rows[{index}].loss",
            )
            loss_differential = _stable_float(baseline_loss - candidate_loss)
            pair = {
                "origin_id": str(candidate_row["origin_id"]),
                "horizon": int(candidate_row["horizon"]),
                "entity_id": str(candidate_row["entity_id"]),
                "row_set_id": str(candidate_row["row_set_id"]),
                "candidate_loss": _stable_float(candidate_loss),
                "baseline_loss": _stable_float(baseline_loss),
                "loss_differential": loss_differential,
            }
            pairs.append(pair)
            differentials.append(loss_differential)

        raw_pair_count = len(pairs)
        horizon_geometry = tuple(sorted({int(pair["horizon"]) for pair in pairs}))
        payload = {
            "baseline_id": str(baseline_id),
            "candidate_id": str(candidate_id),
            "loss_id": str(loss_id),
            "pairs": pairs,
            "stream_id": str(stream_id),
        }
        return cls(
            stream_id=str(stream_id),
            candidate_id=str(candidate_id),
            baseline_id=str(baseline_id),
            loss_id=str(loss_id),
            pairs=tuple(pairs),
            loss_differentials=tuple(differentials),
            raw_pair_count=raw_pair_count,
            effective_sample_size=_stable_float(float(raw_pair_count)),
            block_count=_default_block_count(raw_pair_count),
            horizon_geometry=horizon_geometry,
            replay_identity=f"paired-loss-differential-stream:{_digest(payload)}",
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "paired_loss_differential_stream@1.0.0",
            "stream_id": self.stream_id,
            "candidate_id": self.candidate_id,
            "baseline_id": self.baseline_id,
            "loss_id": self.loss_id,
            "raw_pair_count": self.raw_pair_count,
            "effective_sample_size": self.effective_sample_size,
            "block_count": self.block_count,
            "horizon_geometry": list(self.horizon_geometry),
            "pairs": [dict(pair) for pair in self.pairs],
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class DeclaredPredictiveTestResult:
    declared_test_id: str
    status: str
    promotion_allowed: bool
    reason_codes: tuple[str, ...]
    mean_loss_differential: float
    confidence_interval: tuple[float, float] | None
    p_value: float | None
    practical_margin: float
    raw_pair_count: int
    effective_sample_size: float
    block_count: int
    raw_metric_comparison_role: str
    statistical_test_backend: str
    confidence_interval_method: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dependency_diagnostics: Mapping[str, Any] | None = None
    replay_identity: str = ""

    def as_manifest(self) -> dict[str, Any]:
        manifest = {
            "schema_name": "declared_predictive_test_result@1.0.0",
            "declared_test_id": self.declared_test_id,
            "status": self.status,
            "promotion_allowed": self.promotion_allowed,
            "reason_codes": list(self.reason_codes),
            "mean_loss_differential": self.mean_loss_differential,
            "confidence_interval": (
                list(self.confidence_interval)
                if self.confidence_interval is not None
                else None
            ),
            "p_value": self.p_value,
            "practical_margin": self.practical_margin,
            "raw_pair_count": self.raw_pair_count,
            "effective_sample_size": self.effective_sample_size,
            "block_count": self.block_count,
            "raw_metric_comparison_role": self.raw_metric_comparison_role,
            "statistical_test_backend": self.statistical_test_backend,
            "confidence_interval_method": self.confidence_interval_method,
            "metadata": {key: self.metadata[key] for key in sorted(self.metadata)},
            "replay_identity": self.replay_identity,
        }
        if self.dependency_diagnostics is not None:
            manifest["dependency_diagnostics"] = dict(self.dependency_diagnostics)
        return manifest


def run_declared_predictive_test(
    *,
    stream: PairedLossDifferentialStream,
    declared_test_id: str,
    practical_margin: float,
    block_length: int | None = None,
    bootstrap_count: int = 1000,
    seed: int | None = None,
    instruments: Sequence[Mapping[str, Any]] | None = None,
    state_declarations: Sequence[Mapping[str, Any]] | None = None,
    optional_backend_overrides: Mapping[str, Any] | None = None,
) -> DeclaredPredictiveTestResult:
    test_id = str(declared_test_id)
    if test_id not in _SUPPORTED_DECLARED_TEST_IDS:
        return _result(
            stream=stream,
            declared_test_id=test_id,
            status="failed",
            promotion_allowed=False,
            reason_codes=("unsupported_declared_predictive_test_id",),
            confidence_interval=None,
            p_value=None,
            practical_margin=practical_margin,
            statistical_test_backend="unavailable",
            confidence_interval_method="not_applicable",
            metadata={"requested_test_id": test_id},
        )
    registry: dict[str, Callable[..., DeclaredPredictiveTestResult]] = {
        "diebold_mariano_hln_v1": _run_dm_hln,
        "paired_stationary_block_bootstrap_v1": _run_stationary_block_bootstrap,
        "giacomini_white_v1": _run_giacomini_white,
        "model_confidence_set_v1": _run_model_confidence_set,
        "superior_predictive_ability_v1": _run_superior_predictive_ability,
    }
    return registry[test_id](
        stream=stream,
        declared_test_id=test_id,
        practical_margin=float(practical_margin),
        block_length=block_length,
        bootstrap_count=bootstrap_count,
        seed=seed,
        instruments=instruments,
        state_declarations=state_declarations,
        optional_backend_overrides=optional_backend_overrides,
    )


def _run_dm_hln(
    *,
    stream: PairedLossDifferentialStream,
    declared_test_id: str,
    practical_margin: float,
    block_length: int | None,
    bootstrap_count: int,
    seed: int | None,
    instruments: Sequence[Mapping[str, Any]] | None,
    state_declarations: Sequence[Mapping[str, Any]] | None,
    optional_backend_overrides: Mapping[str, Any] | None,
) -> DeclaredPredictiveTestResult:
    del (
        block_length,
        bootstrap_count,
        seed,
        instruments,
        state_declarations,
        optional_backend_overrides,
    )
    if stream.raw_pair_count < 2:
        return _insufficient_pair_result(
            stream=stream,
            declared_test_id=declared_test_id,
            practical_margin=practical_margin,
            statistical_test_backend="diebold_mariano_hln_v1",
        )

    confidence_interval, p_value = _hac_mean_interval_and_p_value(
        differentials=stream.loss_differentials,
        maxlags=_hac_maxlags(stream.raw_pair_count),
    )
    reason_codes = _uncertainty_reason_codes(
        stream=stream,
        practical_margin=practical_margin,
        confidence_interval=confidence_interval,
    )
    status = "passed" if not reason_codes else "downgraded"
    return _result(
        stream=stream,
        declared_test_id=declared_test_id,
        status=status,
        promotion_allowed=status == "passed",
        reason_codes=reason_codes,
        confidence_interval=confidence_interval,
        p_value=p_value,
        practical_margin=practical_margin,
        statistical_test_backend="diebold_mariano_hln_v1",
        confidence_interval_method="dm_hln_hac_t_interval",
        metadata={
            "hac_maxlags": _hac_maxlags(stream.raw_pair_count),
            "mean_interval_component": "statsmodels_hac_mean_interval_internal",
        },
    )


def _run_stationary_block_bootstrap(
    *,
    stream: PairedLossDifferentialStream,
    declared_test_id: str,
    practical_margin: float,
    block_length: int | None,
    bootstrap_count: int,
    seed: int | None,
    instruments: Sequence[Mapping[str, Any]] | None,
    state_declarations: Sequence[Mapping[str, Any]] | None,
    optional_backend_overrides: Mapping[str, Any] | None,
) -> DeclaredPredictiveTestResult:
    del instruments, state_declarations, optional_backend_overrides
    if stream.raw_pair_count < 2:
        return _insufficient_pair_result(
            stream=stream,
            declared_test_id=declared_test_id,
            practical_margin=practical_margin,
            statistical_test_backend="paired_stationary_block_bootstrap_v1",
        )

    resolved_block_length = _resolve_block_length(
        block_length=block_length,
        sample_size=stream.raw_pair_count,
    )
    resolved_bootstrap_count = max(1, int(bootstrap_count))
    confidence_interval, p_value = _stationary_block_bootstrap_interval(
        differentials=stream.loss_differentials,
        block_length=resolved_block_length,
        bootstrap_count=resolved_bootstrap_count,
        seed=seed,
        practical_margin=practical_margin,
    )
    reason_codes = _uncertainty_reason_codes(
        stream=stream,
        practical_margin=practical_margin,
        confidence_interval=confidence_interval,
    )
    status = "passed" if not reason_codes else "downgraded"
    return _result(
        stream=stream,
        declared_test_id=declared_test_id,
        status=status,
        promotion_allowed=status == "passed",
        reason_codes=reason_codes,
        confidence_interval=confidence_interval,
        p_value=p_value,
        practical_margin=practical_margin,
        statistical_test_backend="paired_stationary_block_bootstrap_v1",
        confidence_interval_method="stationary_block_bootstrap_percentile_interval",
        metadata={
            "block_length": resolved_block_length,
            "bootstrap_count": resolved_bootstrap_count,
            "seed": seed,
        },
    )


def _run_giacomini_white(
    *,
    stream: PairedLossDifferentialStream,
    declared_test_id: str,
    practical_margin: float,
    block_length: int | None,
    bootstrap_count: int,
    seed: int | None,
    instruments: Sequence[Mapping[str, Any]] | None,
    state_declarations: Sequence[Mapping[str, Any]] | None,
    optional_backend_overrides: Mapping[str, Any] | None,
) -> DeclaredPredictiveTestResult:
    del block_length, bootstrap_count, seed, optional_backend_overrides
    if not instruments and not state_declarations:
        return _result(
            stream=stream,
            declared_test_id=declared_test_id,
            status="failed",
            promotion_allowed=False,
            reason_codes=("gw_requires_instruments_or_state",),
            confidence_interval=None,
            p_value=None,
            practical_margin=practical_margin,
            statistical_test_backend="giacomini_white_v1",
            confidence_interval_method="not_applicable",
            metadata={"required_declarations": ["instruments", "state_declarations"]},
        )
    return _result(
        stream=stream,
        declared_test_id=declared_test_id,
        status="failed",
        promotion_allowed=False,
        reason_codes=("gw_backend_not_implemented",),
        confidence_interval=None,
        p_value=None,
        practical_margin=practical_margin,
        statistical_test_backend="giacomini_white_v1",
        confidence_interval_method="not_applicable",
        metadata={},
    )


def _run_model_confidence_set(
    *,
    stream: PairedLossDifferentialStream,
    declared_test_id: str,
    practical_margin: float,
    block_length: int | None,
    bootstrap_count: int,
    seed: int | None,
    instruments: Sequence[Mapping[str, Any]] | None,
    state_declarations: Sequence[Mapping[str, Any]] | None,
    optional_backend_overrides: Mapping[str, Any] | None,
) -> DeclaredPredictiveTestResult:
    del block_length, bootstrap_count, seed, instruments, state_declarations
    arch_backend = _arch_bootstrap_backend(
        optional_backend_overrides=optional_backend_overrides,
    )
    implementation = getattr(arch_backend, "MCS", None) if arch_backend is not None else None
    if implementation is None:
        return _multi_model_backend_unavailable_result(
            stream=stream,
            declared_test_id=declared_test_id,
            practical_margin=practical_margin,
            implementation="MCS",
        )
    return _multi_model_backend_unavailable_result(
        stream=stream,
        declared_test_id=declared_test_id,
        practical_margin=practical_margin,
        implementation="MCS",
    )


def _run_superior_predictive_ability(
    *,
    stream: PairedLossDifferentialStream,
    declared_test_id: str,
    practical_margin: float,
    block_length: int | None,
    bootstrap_count: int,
    seed: int | None,
    instruments: Sequence[Mapping[str, Any]] | None,
    state_declarations: Sequence[Mapping[str, Any]] | None,
    optional_backend_overrides: Mapping[str, Any] | None,
) -> DeclaredPredictiveTestResult:
    del block_length, bootstrap_count, seed, instruments, state_declarations
    arch_backend = _arch_bootstrap_backend(
        optional_backend_overrides=optional_backend_overrides,
    )
    implementation = getattr(arch_backend, "SPA", None) if arch_backend is not None else None
    if implementation is None:
        return _multi_model_backend_unavailable_result(
            stream=stream,
            declared_test_id=declared_test_id,
            practical_margin=practical_margin,
            implementation="SPA",
        )
    return _multi_model_backend_unavailable_result(
        stream=stream,
        declared_test_id=declared_test_id,
        practical_margin=practical_margin,
        implementation="SPA",
    )


def _arch_bootstrap_backend(
    *,
    optional_backend_overrides: Mapping[str, Any] | None,
) -> Any | None:
    if optional_backend_overrides is not None and "arch" in optional_backend_overrides:
        arch_override = optional_backend_overrides["arch"]
        if arch_override is None:
            return None
        return getattr(arch_override, "bootstrap", arch_override)
    try:
        return importlib.import_module("arch.bootstrap")
    except ImportError:
        return None


def _multi_model_backend_unavailable_result(
    *,
    stream: PairedLossDifferentialStream,
    declared_test_id: str,
    practical_margin: float,
    implementation: str,
) -> DeclaredPredictiveTestResult:
    diagnostics = {
        "backend": "arch",
        "implementation": implementation,
        "reason_code": "multi_model_test_backend_unavailable",
    }
    return _result(
        stream=stream,
        declared_test_id=declared_test_id,
        status="abstained",
        promotion_allowed=False,
        reason_codes=(
            "multi_model_test_backend_unavailable",
            "multi_model_superiority_not_tested",
        ),
        confidence_interval=None,
        p_value=None,
        practical_margin=practical_margin,
        statistical_test_backend="arch",
        confidence_interval_method="not_applicable",
        metadata={},
        dependency_diagnostics=diagnostics,
    )


def _validate_pair_identity(
    *,
    candidate_row: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
    index: int,
) -> None:
    for identity_field in _IDENTITY_FIELDS:
        candidate_value = candidate_row.get(identity_field)
        baseline_value = baseline_row.get(identity_field)
        if str(candidate_value) != str(baseline_value):
            raise ContractValidationError(
                code="paired_loss_identity_mismatch",
                message=(
                    "candidate and baseline losses must share origin, horizon, "
                    "entity, and row-set identity"
                ),
                field_path=f"baseline_rows[{index}].{identity_field}",
                details={
                    "field": identity_field,
                    "candidate_value": candidate_value,
                    "baseline_value": baseline_value,
                },
            )


def _insufficient_pair_result(
    *,
    stream: PairedLossDifferentialStream,
    declared_test_id: str,
    practical_margin: float,
    statistical_test_backend: str,
) -> DeclaredPredictiveTestResult:
    return _result(
        stream=stream,
        declared_test_id=declared_test_id,
        status="abstained",
        promotion_allowed=False,
        reason_codes=("insufficient_paired_count",),
        confidence_interval=None,
        p_value=None,
        practical_margin=practical_margin,
        statistical_test_backend=statistical_test_backend,
        confidence_interval_method="not_applicable",
        metadata={"minimum_pair_count": 2},
    )


def _result(
    *,
    stream: PairedLossDifferentialStream,
    declared_test_id: str,
    status: str,
    promotion_allowed: bool,
    reason_codes: tuple[str, ...],
    confidence_interval: tuple[float, float] | None,
    p_value: float | None,
    practical_margin: float,
    statistical_test_backend: str,
    confidence_interval_method: str,
    metadata: Mapping[str, Any],
    dependency_diagnostics: Mapping[str, Any] | None = None,
) -> DeclaredPredictiveTestResult:
    mean_loss_differential = _mean_loss_differential(stream.loss_differentials)
    normalized_metadata = {key: metadata[key] for key in sorted(metadata)}
    payload = {
        "confidence_interval": list(confidence_interval)
        if confidence_interval is not None
        else None,
        "declared_test_id": declared_test_id,
        "mean_loss_differential": mean_loss_differential,
        "metadata": normalized_metadata,
        "p_value": p_value,
        "practical_margin": _stable_float(float(practical_margin)),
        "reason_codes": list(reason_codes),
        "statistical_test_backend": statistical_test_backend,
        "status": status,
        "stream_replay_identity": stream.replay_identity,
    }
    if dependency_diagnostics is not None:
        payload["dependency_diagnostics"] = {
            key: dependency_diagnostics[key] for key in sorted(dependency_diagnostics)
        }
    return DeclaredPredictiveTestResult(
        declared_test_id=declared_test_id,
        status=status,
        promotion_allowed=promotion_allowed,
        reason_codes=reason_codes,
        mean_loss_differential=mean_loss_differential,
        confidence_interval=confidence_interval,
        p_value=_stable_float(p_value) if p_value is not None else None,
        practical_margin=_stable_float(float(practical_margin)),
        raw_pair_count=stream.raw_pair_count,
        effective_sample_size=stream.effective_sample_size,
        block_count=stream.block_count,
        raw_metric_comparison_role="diagnostic_only",
        statistical_test_backend=statistical_test_backend,
        confidence_interval_method=confidence_interval_method,
        metadata=normalized_metadata,
        dependency_diagnostics=dependency_diagnostics,
        replay_identity=f"declared-predictive-test:{_digest(payload)}",
    )


def _uncertainty_reason_codes(
    *,
    stream: PairedLossDifferentialStream,
    practical_margin: float,
    confidence_interval: tuple[float, float],
) -> tuple[str, ...]:
    reason_codes: list[str] = []
    mean_differential = _mean_loss_differential(stream.loss_differentials)
    if math.isclose(mean_differential, 0.0, abs_tol=1e-12):
        reason_codes.append("baseline_tie")
    if mean_differential <= float(practical_margin):
        reason_codes.append("insignificant_improvement")
    if confidence_interval[0] <= float(practical_margin):
        reason_codes.append("uncertainty_interval_crosses_margin")
    return _unique(reason_codes)


def _hac_mean_interval_and_p_value(
    *,
    differentials: tuple[float, ...],
    maxlags: int,
) -> tuple[tuple[float, float], float]:
    values = np.asarray(differentials, dtype=float)
    design = np.ones((len(values), 1), dtype=float)
    model = OLS(values, design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max(0, int(maxlags))},
    )
    mean = float(model.params[0])
    covariance = np.asarray(model.cov_params(), dtype=float)
    variance = max(0.0, float(covariance[0, 0]))
    standard_error = math.sqrt(variance)
    degrees_of_freedom = max(len(values) - 1, 1)
    if math.isclose(standard_error, 0.0, abs_tol=1e-15):
        p_value = 0.0 if not math.isclose(mean, 0.0, abs_tol=1e-15) else 1.0
        return (_stable_float(mean), _stable_float(mean)), p_value
    statistic = mean / standard_error
    critical_value = float(stats.t.ppf(0.975, degrees_of_freedom))
    p_value = float(2.0 * stats.t.sf(abs(statistic), degrees_of_freedom))
    return (
        (
            _stable_float(mean - (critical_value * standard_error)),
            _stable_float(mean + (critical_value * standard_error)),
        ),
        p_value,
    )


def _stationary_block_bootstrap_interval(
    *,
    differentials: tuple[float, ...],
    block_length: int,
    bootstrap_count: int,
    seed: int | None,
    practical_margin: float,
) -> tuple[tuple[float, float], float]:
    values = np.asarray(differentials, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [
            float(np.mean(_stationary_block_resample(values, block_length, rng)))
            for _ in range(bootstrap_count)
        ],
        dtype=float,
    )
    lower, upper = np.quantile(means, [0.025, 0.975])
    p_value = float(np.mean(means <= float(practical_margin)))
    return (
        (_stable_float(float(lower)), _stable_float(float(upper))),
        p_value,
    )


def _stationary_block_resample(
    values: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    size = int(values.size)
    result = np.empty(size, dtype=float)
    index = int(rng.integers(0, size))
    continuation_probability = 1.0 - (1.0 / max(1, int(block_length)))
    for position in range(size):
        if position > 0 and rng.random() >= continuation_probability:
            index = int(rng.integers(0, size))
        result[position] = values[index]
        index = (index + 1) % size
    return result


def _resolve_block_length(*, block_length: int | None, sample_size: int) -> int:
    if block_length is None:
        return max(1, int(round(math.sqrt(sample_size))))
    return max(1, min(int(block_length), int(sample_size)))


def _default_block_count(raw_pair_count: int) -> int:
    if raw_pair_count <= 0:
        return 0
    return raw_pair_count


def _hac_maxlags(sample_size: int) -> int:
    if sample_size < 8:
        return 0
    return min(sample_size - 1, max(1, int(math.sqrt(sample_size))))


def _mean_loss_differential(differentials: Sequence[float]) -> float:
    if not differentials:
        return 0.0
    return _stable_float(fmean(float(value) for value in differentials))


def _finite_float(value: Any, *, field_path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            code="nonfinite_loss_value",
            message="paired loss streams require finite loss values",
            field_path=field_path,
            details={"value": value},
        ) from exc
    if not math.isfinite(result):
        raise ContractValidationError(
            code="nonfinite_loss_value",
            message="paired loss streams require finite loss values",
            field_path=field_path,
            details={"value": value},
        )
    return result


def _unique(codes: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for code in codes:
        if code:
            seen.setdefault(str(code), None)
    return tuple(seen)


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "DeclaredPredictiveTestResult",
    "PairedLossDifferentialStream",
    "run_declared_predictive_test",
]
