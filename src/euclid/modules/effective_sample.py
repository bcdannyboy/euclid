from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Sequence


MINIMUM_N_EFF = 25
HUMAN_REVIEW_N_EFF = 50
MINIMUM_EFFECTIVE_BLOCK_COUNT = 8
BLOCK_BOOTSTRAP_TEST_ID = "paired_stationary_block_bootstrap_v1"


@dataclass(frozen=True)
class HacEffectiveSampleSize:
    raw_count: int
    max_lag: int
    autocorrelation_sum: float
    n_eff: float

    def as_manifest(self) -> dict[str, Any]:
        return {
            "method": "hac_newey_west_effective_sample_size_v1",
            "raw_count": self.raw_count,
            "max_lag": self.max_lag,
            "autocorrelation_sum": self.autocorrelation_sum,
            "n_eff": self.n_eff,
        }


@dataclass(frozen=True)
class BootstrapEffectiveBlockCount:
    raw_count: int
    block_length: int
    effective_block_count: int

    def as_manifest(self) -> dict[str, Any]:
        return {
            "method": "stationary_block_bootstrap_effective_block_count_v1",
            "raw_count": self.raw_count,
            "block_length": self.block_length,
            "effective_block_count": self.effective_block_count,
        }


@dataclass(frozen=True)
class MinimumPairPolicyDecision:
    status: str
    promotion_allowed: bool
    reason_codes: tuple[str, ...]
    raw_pair_count: int
    n_eff: float
    declared_test_id: str
    block_length: int | None = None
    effective_block_count: int | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "promotion_allowed": self.promotion_allowed,
            "reason_codes": list(self.reason_codes),
            "raw_pair_count": self.raw_pair_count,
            "n_eff": self.n_eff,
            "declared_test_id": self.declared_test_id,
            "block_length": self.block_length,
            "effective_block_count": self.effective_block_count,
            "thresholds": _threshold_manifest(),
        }


def hac_effective_sample_size(
    values: Sequence[float],
    *,
    max_lag: int | None = None,
) -> HacEffectiveSampleSize:
    series = _finite_tuple(values)
    raw_count = len(series)
    if raw_count == 0:
        return HacEffectiveSampleSize(
            raw_count=0,
            max_lag=0,
            autocorrelation_sum=0.0,
            n_eff=0.0,
        )

    lag_count = _resolved_max_lag(raw_count, max_lag)
    centered = tuple(value - fmean(series) for value in series)
    variance = sum(value * value for value in centered) / raw_count
    if math.isclose(variance, 0.0, abs_tol=1e-15):
        return HacEffectiveSampleSize(
            raw_count=raw_count,
            max_lag=lag_count,
            autocorrelation_sum=0.0,
            n_eff=_stable_float(float(raw_count)),
        )

    autocorrelation_sum = 0.0
    for lag in range(1, lag_count + 1):
        autocovariance = (
            sum(centered[index] * centered[index - lag] for index in range(lag, raw_count))
            / raw_count
        )
        autocorrelation_sum += autocovariance / variance

    penalty = max(1.0, 1.0 + (2.0 * autocorrelation_sum))
    n_eff = min(float(raw_count), raw_count / penalty)
    return HacEffectiveSampleSize(
        raw_count=raw_count,
        max_lag=lag_count,
        autocorrelation_sum=_stable_float(autocorrelation_sum),
        n_eff=_stable_float(n_eff),
    )


def bootstrap_effective_block_count(
    *,
    raw_count: int,
    block_length: int,
) -> BootstrapEffectiveBlockCount:
    normalized_raw_count = max(0, int(raw_count))
    normalized_block_length = max(1, int(block_length))
    return BootstrapEffectiveBlockCount(
        raw_count=normalized_raw_count,
        block_length=normalized_block_length,
        effective_block_count=normalized_raw_count // normalized_block_length,
    )


def minimum_pair_policy(
    *,
    raw_pair_count: int,
    declared_test_id: str,
    n_eff: float | None = None,
    loss_differentials: Sequence[float] | None = None,
    block_length: int | None = None,
) -> MinimumPairPolicyDecision:
    normalized_n_eff = _resolved_n_eff(
        n_eff=n_eff,
        loss_differentials=loss_differentials,
    )
    normalized_raw_pair_count = max(0, int(raw_pair_count))
    normalized_test_id = str(declared_test_id)
    block_count = _effective_block_count_for_policy(
        raw_pair_count=normalized_raw_pair_count,
        declared_test_id=normalized_test_id,
        block_length=block_length,
    )

    reason_codes: list[str] = []
    if normalized_n_eff < MINIMUM_N_EFF:
        reason_codes.append("insufficient_effective_sample_size")
    if (
        normalized_test_id == BLOCK_BOOTSTRAP_TEST_ID
        and (block_count is None or block_count < MINIMUM_EFFECTIVE_BLOCK_COUNT)
    ):
        reason_codes.append("insufficient_effective_block_count")

    if reason_codes:
        status = "abstained"
        promotion_allowed = False
    elif normalized_n_eff < HUMAN_REVIEW_N_EFF:
        status = "human_review_only"
        promotion_allowed = False
        reason_codes.append("minimum_effective_sample_requires_human_review")
    else:
        status = "passed"
        promotion_allowed = True

    return MinimumPairPolicyDecision(
        status=status,
        promotion_allowed=promotion_allowed,
        reason_codes=tuple(reason_codes),
        raw_pair_count=normalized_raw_pair_count,
        n_eff=_stable_float(normalized_n_eff),
        declared_test_id=normalized_test_id,
        block_length=(max(1, int(block_length)) if block_length is not None else None),
        effective_block_count=block_count,
    )


def _effective_block_count_for_policy(
    *,
    raw_pair_count: int,
    declared_test_id: str,
    block_length: int | None,
) -> int | None:
    if declared_test_id != BLOCK_BOOTSTRAP_TEST_ID:
        return None
    if block_length is None:
        return 0
    return bootstrap_effective_block_count(
        raw_count=raw_pair_count,
        block_length=block_length,
    ).effective_block_count


def _resolved_n_eff(
    *,
    n_eff: float | None,
    loss_differentials: Sequence[float] | None,
) -> float:
    if n_eff is not None:
        candidate = float(n_eff)
        return candidate if math.isfinite(candidate) else 0.0
    if loss_differentials is None:
        return 0.0
    return hac_effective_sample_size(loss_differentials).n_eff


def _resolved_max_lag(raw_count: int, max_lag: int | None) -> int:
    if raw_count < 2:
        return 0
    if max_lag is None:
        return min(raw_count - 1, max(1, int(math.sqrt(raw_count))))
    return min(raw_count - 1, max(0, int(max_lag)))


def _finite_tuple(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        return ()
    return result


def _threshold_manifest() -> dict[str, int]:
    return {
        "minimum_n_eff": MINIMUM_N_EFF,
        "human_review_n_eff": HUMAN_REVIEW_N_EFF,
        "minimum_effective_block_count": MINIMUM_EFFECTIVE_BLOCK_COUNT,
    }


def _stable_float(value: float) -> float:
    return float(round(float(value), 6))


__all__ = [
    "BLOCK_BOOTSTRAP_TEST_ID",
    "BootstrapEffectiveBlockCount",
    "HUMAN_REVIEW_N_EFF",
    "HacEffectiveSampleSize",
    "MINIMUM_EFFECTIVE_BLOCK_COUNT",
    "MINIMUM_N_EFF",
    "MinimumPairPolicyDecision",
    "bootstrap_effective_block_count",
    "hac_effective_sample_size",
    "minimum_pair_policy",
]
