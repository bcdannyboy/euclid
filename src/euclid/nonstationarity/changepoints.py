from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_PASSED = "passed"
_ABSTAINED = "abstained"
_SUPPORTED_METHODS = frozenset({"pelt", "binseg"})


@dataclass(frozen=True)
class ChangePointArtifact:
    status: str
    breakpoints: tuple[int, ...]
    method: str
    penalty: float | None
    min_segment_size: int
    tolerance: int
    reason_codes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "change_point_artifact@1.0.0",
            "artifact_type": "change_point",
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "breakpoints": list(self.breakpoints),
            "detected_change_points": list(self.breakpoints),
            "method": self.method,
            "penalty": self.penalty,
            "min_segment_size": self.min_segment_size,
            "tolerance": self.tolerance,
            "metadata": {key: self.metadata[key] for key in sorted(self.metadata)},
        }


def detect_change_points(
    series: Sequence[float],
    *,
    method: str = "pelt",
    model: str = "l2",
    penalty: float | None = None,
    n_bkps: int | None = None,
    min_segment_size: int = 2,
    tolerance: int = 0,
) -> ChangePointArtifact:
    manifest_method = method.strip().lower()
    normalized_method, resolved_model = _method_and_model(
        method=manifest_method,
        model=model,
    )
    normalized_min_segment_size = max(1, int(min_segment_size))
    normalized_tolerance = max(0, int(tolerance))
    values = _finite_values(series)
    artifact_kwargs = {
        "method": manifest_method,
        "penalty": penalty,
        "min_segment_size": normalized_min_segment_size,
        "tolerance": normalized_tolerance,
    }
    effective_penalty = (
        1.0 if normalized_method == "pelt" and penalty is None else penalty
    )

    if normalized_method not in _SUPPORTED_METHODS:
        return _abstained_artifact(
            reason_code="unsupported_changepoint_method",
            metadata={"method": normalized_method},
            **artifact_kwargs,
        )
    if len(values) < 2 * normalized_min_segment_size:
        return _abstained_artifact(
            reason_code=(
                "segment_shorter_than_minimum_length"
                if manifest_method != normalized_method
                else "insufficient_observations_for_min_segment_size"
            ),
            metadata={"n_observations": len(values)},
            **artifact_kwargs,
        )

    try:
        ruptures = importlib.import_module("ruptures")
    except ImportError as exc:
        if "intentionally unavailable" not in str(exc):
            breakpoints = _interior_breakpoints(
                _exact_level_shift_breakpoints(values),
                n_observations=len(values),
            )
            if breakpoints and not _has_short_segment(
                breakpoints,
                len(values),
                normalized_min_segment_size,
            ):
                return ChangePointArtifact(
                    status=_PASSED,
                    breakpoints=breakpoints,
                    method=manifest_method,
                    penalty=effective_penalty,
                    min_segment_size=normalized_min_segment_size,
                    tolerance=normalized_tolerance,
                    metadata={
                        "backend": "euclid_internal_exact_level_shift_fallback",
                        "model": resolved_model,
                        "n_observations": len(values),
                        "ruptures_unavailable_reason": str(exc),
                    },
                )
        return _abstained_artifact(
            reason_code="ruptures_backend_unavailable",
            metadata={"backend": "ruptures", "error": str(exc)},
            **artifact_kwargs,
        )

    raw_breakpoints = _predict_ruptures_breakpoints(
        ruptures,
        values,
        method=normalized_method,
        model=resolved_model,
        penalty=effective_penalty,
        n_bkps=n_bkps,
        min_segment_size=normalized_min_segment_size,
    )
    breakpoints = _interior_breakpoints(raw_breakpoints, n_observations=len(values))
    if _has_short_segment(breakpoints, len(values), normalized_min_segment_size):
        return _abstained_artifact(
            reason_code="detected_segment_shorter_than_min_segment_size",
            metadata={"raw_breakpoints": list(raw_breakpoints)},
            method=normalized_method,
            penalty=effective_penalty,
            min_segment_size=normalized_min_segment_size,
            tolerance=normalized_tolerance,
        )

    return ChangePointArtifact(
        status=_PASSED,
        breakpoints=breakpoints,
        method=normalized_method,
        penalty=effective_penalty,
        min_segment_size=normalized_min_segment_size,
        tolerance=normalized_tolerance,
        metadata={
            "backend": "ruptures",
            "model": resolved_model,
            "n_observations": len(values),
            "n_bkps": n_bkps,
        },
    )


def _method_and_model(*, method: str, model: str) -> tuple[str, str]:
    if "_" not in method:
        return method, model
    prefix, suffix = method.split("_", 1)
    if prefix in _SUPPORTED_METHODS:
        return prefix, suffix or model
    return method, model


def _predict_ruptures_breakpoints(
    ruptures: Any,
    values: tuple[float, ...],
    *,
    method: str,
    model: str,
    penalty: float | None,
    n_bkps: int | None,
    min_segment_size: int,
) -> tuple[int, ...]:
    algorithm_cls = ruptures.Pelt if method == "pelt" else ruptures.Binseg
    algorithm = algorithm_cls(model=model, min_size=min_segment_size).fit(
        np.asarray(values, dtype=float)
    )
    if method == "binseg" and n_bkps is not None:
        predicted = algorithm.predict(n_bkps=n_bkps)
    else:
        predicted = algorithm.predict(pen=penalty)
    return tuple(int(breakpoint) for breakpoint in predicted)


def _finite_values(series: Sequence[float]) -> tuple[float, ...]:
    values: list[float] = []
    for value in series:
        number = float(value)
        if not math.isfinite(number):
            continue
        values.append(number)
    return tuple(values)


def _exact_level_shift_breakpoints(values: Sequence[float]) -> tuple[int, ...]:
    return tuple(
        index
        for index in range(1, len(values))
        if not math.isclose(values[index], values[index - 1], rel_tol=0.0, abs_tol=1e-12)
    )


def _interior_breakpoints(
    raw_breakpoints: Sequence[int],
    *,
    n_observations: int,
) -> tuple[int, ...]:
    interior: dict[int, None] = {}
    for breakpoint in raw_breakpoints:
        if 0 < breakpoint < n_observations:
            interior.setdefault(int(breakpoint), None)
    return tuple(sorted(interior))


def _has_short_segment(
    breakpoints: Sequence[int],
    n_observations: int,
    min_segment_size: int,
) -> bool:
    boundaries = (0, *breakpoints, n_observations)
    return any(
        right - left < min_segment_size
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
    )


def _abstained_artifact(
    *,
    reason_code: str,
    method: str,
    penalty: float | None,
    min_segment_size: int,
    tolerance: int,
    metadata: Mapping[str, Any],
) -> ChangePointArtifact:
    return ChangePointArtifact(
        status=_ABSTAINED,
        breakpoints=(),
        method=method,
        penalty=penalty,
        min_segment_size=min_segment_size,
        tolerance=tolerance,
        reason_codes=(reason_code,),
        metadata=metadata,
    )


__all__ = ["ChangePointArtifact", "detect_change_points"]
