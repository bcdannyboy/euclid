from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Mapping, Sequence

from euclid.modules.effective_sample import (
    HUMAN_REVIEW_N_EFF,
    MINIMUM_EFFECTIVE_BLOCK_COUNT,
    MINIMUM_N_EFF,
)
from euclid.modules.predictive_inference import (
    PairedLossDifferentialStream,
    run_declared_predictive_test as _run_declared_paired_test,
)

_DEFAULT_DECLARED_TEST_ID = "diebold_mariano_hln_v1"
_BLOCK_BOOTSTRAP_TEST_ID = "paired_stationary_block_bootstrap_v1"
_GW_PUBLIC_TEST_ID = "giacomini_white_conditional_predictive_ability_v1"
_GW_INTERNAL_TEST_ID = "giacomini_white_v1"
_MODEL_CONFIDENCE_SET_TEST_ID = "model_confidence_set_v1"
_SUPERIOR_PREDICTIVE_ABILITY_TEST_ID = "superior_predictive_ability_v1"
_MULTI_MODEL_TEST_IDS = {
    _MODEL_CONFIDENCE_SET_TEST_ID,
    _SUPERIOR_PREDICTIVE_ABILITY_TEST_ID,
}
_SUPPORTED_PUBLIC_TEST_IDS = {
    _DEFAULT_DECLARED_TEST_ID,
    _BLOCK_BOOTSTRAP_TEST_ID,
    _GW_PUBLIC_TEST_ID,
    *_MULTI_MODEL_TEST_IDS,
}


@dataclass(frozen=True)
class PredictivePromotionResult:
    status: str
    promotion_allowed: bool
    reason_codes: tuple[str, ...]
    mean_loss_differential: float
    confidence_interval: tuple[float, float] | None
    practical_margin: float
    raw_metric_comparison_role: str
    statistical_test_backend: str
    confidence_interval_method: str
    replay_identity: str
    declared_test_id: str = _DEFAULT_DECLARED_TEST_ID
    actual_test_id: str = _DEFAULT_DECLARED_TEST_ID
    p_value: float | None = None
    raw_pair_count: int = 0
    effective_sample_size: float = 0.0
    effective_block_count: int | None = None
    paired_stream_ref: Mapping[str, Any] | None = None
    paired_stream_identity: Mapping[str, Any] | None = None
    minimum_pair_policy: Mapping[str, Any] = field(default_factory=dict)
    block_bootstrap: Mapping[str, Any] | None = None
    conditional_instrument_declarations: tuple[Mapping[str, Any], ...] = ()
    dependency_diagnostics: Mapping[str, Any] | None = None
    hln_small_sample_correction: Mapping[str, Any] | None = None

    def as_manifest(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "schema_name": "paired_predictive_test_result@1.0.0",
            "declared_test_id": self.declared_test_id,
            "actual_test_id": self.actual_test_id,
            "status": self.status,
            "promotion_allowed": self.promotion_allowed,
            "reason_codes": list(self.reason_codes),
            "mean_loss_differential": self.mean_loss_differential,
            "confidence_interval": (
                list(self.confidence_interval)
                if self.confidence_interval is not None
                else None
            ),
            "confidence_interval_method": self.confidence_interval_method,
            "practical_margin": self.practical_margin,
            "raw_metric_comparison_role": self.raw_metric_comparison_role,
            "statistical_test_backend": self.statistical_test_backend,
            "p_value": self.p_value,
            "raw_pair_count": self.raw_pair_count,
            "effective_sample_size": self.effective_sample_size,
            "effective_block_count": self.effective_block_count,
            "minimum_pair_policy": dict(
                self.minimum_pair_policy or _minimum_pair_policy_manifest()
            ),
            "replay_identity": self.replay_identity,
        }
        if self.paired_stream_ref is not None:
            manifest["paired_stream_ref"] = dict(self.paired_stream_ref)
        if self.paired_stream_identity is not None:
            manifest["paired_stream_identity"] = _jsonable(self.paired_stream_identity)
        if self.block_bootstrap is not None:
            manifest["block_bootstrap"] = dict(self.block_bootstrap)
        if (
            self.declared_test_id == _GW_PUBLIC_TEST_ID
            or self.conditional_instrument_declarations
        ):
            manifest["conditional_instrument_declarations"] = [
                dict(item) for item in self.conditional_instrument_declarations
            ]
        if self.dependency_diagnostics is not None:
            manifest["dependency_diagnostics"] = dict(self.dependency_diagnostics)
        if self.hln_small_sample_correction is not None:
            manifest["hln_small_sample_correction"] = dict(
                self.hln_small_sample_correction
            )
        return manifest


@dataclass(frozen=True)
class PrequentialScoreStream:
    stream_id: str
    candidate_id: str
    baseline_id: str
    per_origin: tuple[Mapping[str, Any], ...]
    per_horizon: tuple[Mapping[str, Any], ...]
    per_entity: tuple[Mapping[str, Any], ...]
    per_regime: tuple[Mapping[str, Any], ...]
    rolling_degradation: tuple[Mapping[str, Any], ...]
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "prequential_score_stream@1.0.0",
            "stream_id": self.stream_id,
            "candidate_id": self.candidate_id,
            "baseline_id": self.baseline_id,
            "per_origin": [dict(item) for item in self.per_origin],
            "per_horizon": [dict(item) for item in self.per_horizon],
            "per_entity": [dict(item) for item in self.per_entity],
            "per_regime": [dict(item) for item in self.per_regime],
            "rolling_degradation": [dict(item) for item in self.rolling_degradation],
            "replay_identity": self.replay_identity,
        }


def evaluate_predictive_promotion(
    *,
    candidate_losses: Sequence[float],
    baseline_losses: Sequence[float],
    split_protocol_id: str,
    baseline_id: str | None,
    practical_margin: float,
    calibration_status: str = "not_applicable_for_forecast_type",
    leakage_status: str = "passed",
    nonstationarity_status: str = "not_evaluated",
    nonstationarity_diagnostic: Mapping[str, Any] | None = None,
    declared_test_id: str = _DEFAULT_DECLARED_TEST_ID,
    paired_stream_identity: Mapping[str, Any] | None = None,
    effective_sample_size: float | None = None,
    effective_block_count: int | None = None,
    block_bootstrap_config: Mapping[str, Any] | None = None,
    conditional_instrument_declarations: Sequence[Mapping[str, Any]] | None = None,
) -> PredictivePromotionResult:
    candidate = _finite_tuple(candidate_losses)
    baseline = _finite_tuple(baseline_losses)
    reason_codes: list[str] = []
    if baseline_id is None or not baseline:
        reason_codes.append("missing_baseline")
    if split_protocol_id in {"", "train_only", "in_sample_only"}:
        reason_codes.append("unstable_split_protocol")
    if leakage_status != "passed":
        reason_codes.append("leakage_detected")
    nonstationarity_from_status = nonstationarity_status in {
        "failed",
        "detected",
        "nonstationary",
        "unstable",
    }
    if _nonstationarity_detected(
        status=nonstationarity_status,
        diagnostic=nonstationarity_diagnostic,
    ):
        reason_codes.append("nonstationarity_detected")
    if calibration_status in {"failed", "coverage_failed", "poor_coverage"}:
        reason_codes.append("calibration_failed")
        if calibration_status in {"coverage_failed", "poor_coverage"}:
            reason_codes.append("poor_coverage")
    if len(candidate) != len(baseline) or not candidate:
        reason_codes.append("unpaired_loss_stream")

    if reason_codes:
        return _promotion_result(
            status=(
                "failed"
                if nonstationarity_from_status
                and "nonstationarity_detected" in reason_codes
                else "abstained"
            ),
            promotion_allowed=False,
            reason_codes=_unique(reason_codes),
            mean_loss_differential=0.0,
            confidence_interval=None,
            practical_margin=practical_margin,
            declared_test_id=declared_test_id,
            actual_test_id=_actual_test_id(declared_test_id),
            raw_pair_count=0,
            effective_sample_size=0.0,
            effective_block_count=0,
        )

    differentials = tuple(
        baseline_loss - candidate_loss
        for candidate_loss, baseline_loss in zip(candidate, baseline, strict=True)
    )
    mean_differential = _stable_float(fmean(differentials))
    raw_pair_count = len(differentials)
    resolved_n_eff = _stable_float(
        float(effective_sample_size)
        if effective_sample_size is not None
        else float(raw_pair_count)
    )
    resolved_block_count = _resolve_effective_block_count(
        declared_test_id=declared_test_id,
        raw_pair_count=raw_pair_count,
        effective_block_count=effective_block_count,
        block_bootstrap_config=block_bootstrap_config,
    )

    if raw_pair_count < 2:
        return _promotion_result(
            status="abstained",
            promotion_allowed=False,
            reason_codes=("insufficient_paired_count",),
            mean_loss_differential=mean_differential,
            confidence_interval=None,
            practical_margin=practical_margin,
            declared_test_id=declared_test_id,
            actual_test_id=_actual_test_id(declared_test_id),
            raw_pair_count=raw_pair_count,
            effective_sample_size=resolved_n_eff,
            effective_block_count=resolved_block_count,
            paired_stream_identity=paired_stream_identity,
        )

    if float(practical_margin) <= 0.0:
        return _promotion_result(
            status="failed",
            promotion_allowed=False,
            reason_codes=("missing_practical_effect_margin",),
            mean_loss_differential=mean_differential,
            confidence_interval=None,
            practical_margin=practical_margin,
            declared_test_id=declared_test_id,
            actual_test_id=_actual_test_id(declared_test_id),
            raw_pair_count=raw_pair_count,
            effective_sample_size=resolved_n_eff,
            effective_block_count=resolved_block_count,
            paired_stream_identity=paired_stream_identity,
        )

    if _many_model_pairwise_correction_missing(
        declared_test_id=declared_test_id,
        paired_stream_identity=paired_stream_identity,
    ):
        return _promotion_result(
            status="abstained",
            promotion_allowed=False,
            reason_codes=("many_model_correction_failed",),
            mean_loss_differential=mean_differential,
            confidence_interval=None,
            practical_margin=practical_margin,
            declared_test_id=declared_test_id,
            actual_test_id=_actual_test_id(declared_test_id),
            raw_pair_count=raw_pair_count,
            effective_sample_size=resolved_n_eff,
            effective_block_count=resolved_block_count,
            paired_stream_identity=paired_stream_identity,
        )

    if declared_test_id in _MULTI_MODEL_TEST_IDS:
        return _multi_model_superiority_not_tested_result(
            declared_test_id=declared_test_id,
            differentials=differentials,
            practical_margin=practical_margin,
            raw_pair_count=raw_pair_count,
            effective_sample_size=resolved_n_eff,
            effective_block_count=resolved_block_count,
            dependency_reason_code=None,
        )

    if declared_test_id not in _SUPPORTED_PUBLIC_TEST_IDS:
        return _promotion_result(
            status="abstained",
            promotion_allowed=False,
            reason_codes=("unsupported_declared_test_id",),
            mean_loss_differential=mean_differential,
            confidence_interval=None,
            practical_margin=practical_margin,
            declared_test_id=declared_test_id,
            actual_test_id="unsupported_declared_test_id",
            raw_pair_count=raw_pair_count,
            effective_sample_size=resolved_n_eff,
            effective_block_count=resolved_block_count,
            paired_stream_identity=paired_stream_identity,
        )

    if declared_test_id == _GW_PUBLIC_TEST_ID and not conditional_instrument_declarations:
        return _promotion_result(
            status="abstained",
            promotion_allowed=False,
            reason_codes=("missing_conditional_instrument_declarations",),
            mean_loss_differential=mean_differential,
            confidence_interval=None,
            practical_margin=practical_margin,
            declared_test_id=declared_test_id,
            actual_test_id=_GW_INTERNAL_TEST_ID,
            raw_pair_count=raw_pair_count,
            effective_sample_size=resolved_n_eff,
            effective_block_count=resolved_block_count,
            paired_stream_identity=paired_stream_identity,
            conditional_instrument_declarations=conditional_instrument_declarations,
        )

    diagnostic_reasons = _diagnostic_reason_codes(
        differentials=differentials,
        practical_margin=practical_margin,
    )
    if paired_stream_identity is None and diagnostic_reasons:
        return _promotion_result(
            status="downgraded",
            promotion_allowed=False,
            reason_codes=diagnostic_reasons,
            mean_loss_differential=mean_differential,
            confidence_interval=_simple_mean_interval(differentials),
            practical_margin=practical_margin,
            declared_test_id=declared_test_id,
            actual_test_id=_actual_test_id(declared_test_id),
            raw_pair_count=raw_pair_count,
            effective_sample_size=resolved_n_eff,
            effective_block_count=resolved_block_count,
        )

    policy_status = _minimum_pair_policy_status(
        declared_test_id=declared_test_id,
        n_eff=resolved_n_eff,
        effective_block_count=resolved_block_count,
    )
    if policy_status is not None:
        status, policy_reasons = policy_status
        return _promotion_result(
            status=status,
            promotion_allowed=False,
            reason_codes=policy_reasons,
            mean_loss_differential=mean_differential,
            confidence_interval=None,
            practical_margin=practical_margin,
            declared_test_id=declared_test_id,
            actual_test_id=_actual_test_id(declared_test_id),
            raw_pair_count=raw_pair_count,
            effective_sample_size=resolved_n_eff,
            effective_block_count=resolved_block_count,
            paired_stream_identity=paired_stream_identity,
            block_bootstrap=block_bootstrap_config,
            conditional_instrument_declarations=conditional_instrument_declarations,
        )

    if paired_stream_identity is None:
        return _promotion_result(
            status="abstained",
            promotion_allowed=False,
            reason_codes=("unpaired_loss_stream",),
            mean_loss_differential=mean_differential,
            confidence_interval=_simple_mean_interval(differentials),
            practical_margin=practical_margin,
            declared_test_id=declared_test_id,
            actual_test_id=_actual_test_id(declared_test_id),
            raw_pair_count=raw_pair_count,
            effective_sample_size=resolved_n_eff,
            effective_block_count=resolved_block_count,
        )

    stream = _stream_from_losses(
        candidate_losses=candidate,
        baseline_losses=baseline,
        baseline_id=str(baseline_id),
        declared_test_id=declared_test_id,
        paired_stream_identity=paired_stream_identity,
    )
    block_config = dict(block_bootstrap_config or {})
    internal_result = _run_declared_paired_test(
        stream=stream,
        declared_test_id=_actual_test_id(declared_test_id),
        practical_margin=practical_margin,
        block_length=block_config.get("block_length"),
        bootstrap_count=int(block_config.get("bootstrap_count", 1000)),
        seed=block_config.get("seed"),
        instruments=conditional_instrument_declarations,
        state_declarations=None,
    )
    stream_ref = {
        "schema_name": "paired_loss_differential_stream@1.0.0",
        "object_id": stream.stream_id,
    }
    status, promotion_allowed, reason_codes = _automatic_promotion_semantics(
        backend_status=internal_result.status,
        backend_promotion_allowed=internal_result.promotion_allowed,
        backend_reason_codes=internal_result.reason_codes,
        mean_loss_differential=internal_result.mean_loss_differential,
        confidence_interval=internal_result.confidence_interval,
        practical_margin=practical_margin,
    )
    return _promotion_result(
        status=status,
        promotion_allowed=promotion_allowed,
        reason_codes=tuple(
            "unsupported_declared_test_id"
            if code == "unsupported_declared_predictive_test_id"
            else code
            for code in reason_codes
        ),
        mean_loss_differential=internal_result.mean_loss_differential,
        confidence_interval=internal_result.confidence_interval,
        practical_margin=practical_margin,
        declared_test_id=declared_test_id,
        actual_test_id=_actual_test_id(declared_test_id),
        raw_pair_count=raw_pair_count,
        effective_sample_size=resolved_n_eff,
        effective_block_count=resolved_block_count,
        paired_stream_ref=stream_ref,
        paired_stream_identity=paired_stream_identity,
        block_bootstrap=block_config if declared_test_id == _BLOCK_BOOTSTRAP_TEST_ID else None,
        conditional_instrument_declarations=conditional_instrument_declarations,
        p_value=internal_result.p_value,
        statistical_test_backend=declared_test_id,
        confidence_interval_method=internal_result.confidence_interval_method,
        hln_small_sample_correction=(
            {"method": "harvey_leybourne_newbold_small_sample_adjustment_v1"}
            if declared_test_id == _DEFAULT_DECLARED_TEST_ID
            else None
        ),
    )


def _automatic_promotion_semantics(
    *,
    backend_status: str,
    backend_promotion_allowed: bool,
    backend_reason_codes: Sequence[str],
    mean_loss_differential: float,
    confidence_interval: tuple[float, float] | None,
    practical_margin: float,
) -> tuple[str, bool, tuple[str, ...]]:
    reason_codes = list(backend_reason_codes)
    if _stable_float(mean_loss_differential) <= float(practical_margin):
        reason_codes.append("insignificant_improvement")
    if confidence_interval is not None and confidence_interval[0] <= float(
        practical_margin
    ):
        reason_codes.append("uncertainty_interval_crosses_margin")

    normalized_reasons = _unique(reason_codes)
    if not normalized_reasons:
        return (
            backend_status,
            bool(backend_promotion_allowed) and backend_status == "passed",
            normalized_reasons,
        )
    if any(
        code in normalized_reasons
        for code in ("baseline_tie", "insignificant_improvement")
    ):
        status = (
            "failed"
            if backend_status in {"passed", "downgraded"}
            else backend_status
        )
    elif "uncertainty_interval_crosses_margin" in normalized_reasons:
        status = "downgraded" if backend_status == "passed" else backend_status
    else:
        status = "downgraded" if backend_status == "passed" else backend_status
    return status, False, normalized_reasons


def run_declared_predictive_test(
    *,
    declared_test_id: str,
    loss_differentials: Sequence[float],
    practical_margin: float = 0.0,
    optional_backend_overrides: Mapping[str, Any] | None = None,
    block_length: int | None = None,
    bootstrap_count: int = 1000,
    seed: int | None = None,
    effective_sample_size: float | None = None,
    effective_block_count: int | None = None,
) -> PredictivePromotionResult:
    differentials = _finite_tuple(loss_differentials)
    if declared_test_id in _MULTI_MODEL_TEST_IDS:
        dependency_reason_code = (
            "multi_model_test_backend_unavailable"
            if optional_backend_overrides is not None
            and optional_backend_overrides.get("arch") is None
            else None
        )
        return _multi_model_superiority_not_tested_result(
            declared_test_id=declared_test_id,
            differentials=differentials,
            practical_margin=practical_margin,
            raw_pair_count=len(differentials),
            effective_sample_size=(
                _stable_float(float(effective_sample_size))
                if effective_sample_size is not None
                else float(len(differentials))
            ),
            effective_block_count=effective_block_count,
            dependency_reason_code=dependency_reason_code,
        )
    if (
        declared_test_id == _BLOCK_BOOTSTRAP_TEST_ID
        and optional_backend_overrides is not None
        and optional_backend_overrides.get("arch") is None
    ):
        diagnostics = {
            "backend": "arch",
            "reason_code": "bootstrap_test_backend_unavailable",
        }
        return _promotion_result(
            status="abstained",
            promotion_allowed=False,
            reason_codes=("bootstrap_test_backend_unavailable",),
            mean_loss_differential=_stable_float(fmean(differentials))
            if differentials
            else 0.0,
            confidence_interval=None,
            practical_margin=practical_margin,
            declared_test_id=declared_test_id,
            actual_test_id=declared_test_id,
            raw_pair_count=len(differentials),
            effective_sample_size=(
                _stable_float(float(effective_sample_size))
                if effective_sample_size is not None
                else float(len(differentials))
            ),
            effective_block_count=effective_block_count,
            block_bootstrap={
                "block_length": block_length,
                "bootstrap_count": bootstrap_count,
                "seed": seed,
            },
            dependency_diagnostics=diagnostics,
        )

    identity = {
        "row_set_id": "declared_predictive_test_loss_differentials",
        "origin_ids": [f"loss_differential_{index}" for index in range(len(differentials))],
        "horizons": [1],
        "entity_ids": ["declared_loss_differential"],
    }
    return evaluate_predictive_promotion(
        candidate_losses=tuple(0.0 for _ in differentials),
        baseline_losses=differentials,
        split_protocol_id="declared_predictive_test_loss_differentials",
        baseline_id="declared_baseline",
        practical_margin=practical_margin,
        declared_test_id=declared_test_id,
        paired_stream_identity=identity,
        effective_sample_size=effective_sample_size,
        effective_block_count=effective_block_count,
        block_bootstrap_config={
            "block_length": block_length,
            "bootstrap_count": bootstrap_count,
            "seed": seed,
        },
    )


def _multi_model_superiority_not_tested_result(
    *,
    declared_test_id: str,
    differentials: Sequence[float],
    practical_margin: float,
    raw_pair_count: int,
    effective_sample_size: float,
    effective_block_count: int | None,
    dependency_reason_code: str | None,
) -> PredictivePromotionResult:
    reason_codes = ["multi_model_superiority_not_tested"]
    dependency_diagnostics = {
        "backend": "arch",
        "declared_test_id": declared_test_id,
        "reason_code": dependency_reason_code
        or "multi_model_superiority_not_tested",
    }
    if dependency_reason_code is not None:
        reason_codes.insert(0, dependency_reason_code)
    return _promotion_result(
        status="abstained",
        promotion_allowed=False,
        reason_codes=tuple(reason_codes),
        mean_loss_differential=_stable_float(fmean(differentials))
        if differentials
        else 0.0,
        confidence_interval=None,
        practical_margin=practical_margin,
        declared_test_id=declared_test_id,
        actual_test_id=declared_test_id,
        raw_pair_count=raw_pair_count,
        effective_sample_size=effective_sample_size,
        effective_block_count=effective_block_count,
        dependency_diagnostics=dependency_diagnostics,
        statistical_test_backend="arch" if dependency_reason_code else "not_run",
        confidence_interval_method="not_applicable",
    )


def build_prequential_score_stream(
    *,
    stream_id: str,
    candidate_id: str,
    baseline_id: str,
    rows: Sequence[Mapping[str, Any]],
    rolling_window: int = 5,
) -> PrequentialScoreStream:
    per_origin: list[dict[str, Any]] = []
    for row in rows:
        candidate_loss = float(row["candidate_loss"])
        baseline_loss = float(row["baseline_loss"])
        per_origin.append(
            {
                "origin_id": str(row["origin_id"]),
                "horizon": int(row["horizon"]),
                "entity": str(row.get("entity", "")),
                "regime": str(row.get("regime", "")),
                "candidate_loss": _stable_float(candidate_loss),
                "baseline_loss": _stable_float(baseline_loss),
                "loss_difference": _stable_float(baseline_loss - candidate_loss),
            }
        )
    per_horizon = _group_mean(per_origin, "horizon")
    per_entity = _group_mean(per_origin, "entity")
    per_regime = _group_mean(per_origin, "regime")
    rolling = _rolling_degradation(per_origin, rolling_window=max(1, rolling_window))
    identity_payload = {
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "per_origin": per_origin,
        "stream_id": stream_id,
    }
    return PrequentialScoreStream(
        stream_id=str(stream_id),
        candidate_id=str(candidate_id),
        baseline_id=str(baseline_id),
        per_origin=tuple(per_origin),
        per_horizon=tuple(per_horizon),
        per_entity=tuple(per_entity),
        per_regime=tuple(per_regime),
        rolling_degradation=tuple(rolling),
        replay_identity=f"prequential-stream:{_digest(identity_payload)}",
    )


def _promotion_result(
    *,
    status: str,
    promotion_allowed: bool,
    reason_codes: tuple[str, ...],
    mean_loss_differential: float,
    confidence_interval: tuple[float, float] | None,
    practical_margin: float,
    declared_test_id: str = _DEFAULT_DECLARED_TEST_ID,
    actual_test_id: str | None = None,
    raw_pair_count: int = 0,
    effective_sample_size: float = 0.0,
    effective_block_count: int | None = None,
    paired_stream_ref: Mapping[str, Any] | None = None,
    paired_stream_identity: Mapping[str, Any] | None = None,
    block_bootstrap: Mapping[str, Any] | None = None,
    conditional_instrument_declarations: Sequence[Mapping[str, Any]] | None = None,
    dependency_diagnostics: Mapping[str, Any] | None = None,
    p_value: float | None = None,
    statistical_test_backend: str | None = None,
    confidence_interval_method: str | None = None,
    hln_small_sample_correction: Mapping[str, Any] | None = None,
) -> PredictivePromotionResult:
    normalized_actual_test_id = actual_test_id or _actual_test_id(declared_test_id)
    normalized_backend = statistical_test_backend or normalized_actual_test_id
    normalized_effective_block_count = (
        effective_block_count if effective_block_count is not None else raw_pair_count
    )
    payload = {
        "actual_test_id": normalized_actual_test_id,
        "confidence_interval": list(confidence_interval)
        if confidence_interval is not None
        else None,
        "declared_test_id": declared_test_id,
        "effective_block_count": normalized_effective_block_count,
        "effective_sample_size": _stable_float(float(effective_sample_size)),
        "mean_loss_differential": _stable_float(mean_loss_differential),
        "practical_margin": _stable_float(float(practical_margin)),
        "reason_codes": list(reason_codes),
        "status": status,
    }
    return PredictivePromotionResult(
        status=status,
        promotion_allowed=promotion_allowed,
        reason_codes=reason_codes,
        mean_loss_differential=_stable_float(mean_loss_differential),
        confidence_interval=confidence_interval,
        practical_margin=_stable_float(float(practical_margin)),
        raw_metric_comparison_role="diagnostic_only",
        statistical_test_backend=normalized_backend,
        confidence_interval_method=(
            confidence_interval_method
            or ("dm_hln_hac_t_interval" if confidence_interval is not None else "not_applicable")
        ),
        replay_identity=f"predictive-promotion:{_digest(payload)}",
        declared_test_id=declared_test_id,
        actual_test_id=normalized_actual_test_id,
        p_value=_stable_float(p_value) if p_value is not None else None,
        raw_pair_count=raw_pair_count,
        effective_sample_size=_stable_float(float(effective_sample_size)),
        effective_block_count=normalized_effective_block_count,
        paired_stream_ref=paired_stream_ref,
        paired_stream_identity=paired_stream_identity,
        minimum_pair_policy=_minimum_pair_policy_manifest(),
        block_bootstrap=block_bootstrap,
        conditional_instrument_declarations=tuple(
            dict(item) for item in (conditional_instrument_declarations or ())
        ),
        dependency_diagnostics=dependency_diagnostics,
        hln_small_sample_correction=hln_small_sample_correction,
    )


def _stream_from_losses(
    *,
    candidate_losses: Sequence[float],
    baseline_losses: Sequence[float],
    baseline_id: str,
    declared_test_id: str,
    paired_stream_identity: Mapping[str, Any],
) -> PairedLossDifferentialStream:
    row_set_id = str(paired_stream_identity.get("row_set_id", "declared_pairs"))
    origin_ids = tuple(paired_stream_identity.get("origin_ids", ()))
    horizons = tuple(paired_stream_identity.get("horizons", (1,)))
    entity_ids = tuple(paired_stream_identity.get("entity_ids", ("entity",)))
    candidate_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for index, (candidate_loss, baseline_loss) in enumerate(
        zip(candidate_losses, baseline_losses, strict=True)
    ):
        identity = {
            "origin_id": _identity_value(origin_ids, index, f"origin_{index}"),
            "horizon": int(_identity_value(horizons, index, 1)),
            "entity_id": _identity_value(entity_ids, index, "entity"),
            "row_set_id": row_set_id,
        }
        candidate_rows.append({**identity, "loss": float(candidate_loss)})
        baseline_rows.append({**identity, "loss": float(baseline_loss)})
    return PairedLossDifferentialStream.from_loss_rows(
        stream_id=f"{declared_test_id}:{baseline_id}:paired_loss_differentials",
        candidate_id="candidate",
        baseline_id=baseline_id,
        loss_id="primary_loss",
        candidate_rows=tuple(candidate_rows),
        baseline_rows=tuple(baseline_rows),
    )


def _identity_value(values: Sequence[Any], index: int, default: Any) -> Any:
    if not values:
        return default
    if len(values) == 1:
        return values[0]
    if index < len(values):
        return values[index]
    return default


def _minimum_pair_policy_status(
    *,
    declared_test_id: str,
    n_eff: float,
    effective_block_count: int | None,
) -> tuple[str, tuple[str, ...]] | None:
    if n_eff < MINIMUM_N_EFF:
        return "abstained", ("insufficient_effective_sample_size",)
    if (
        declared_test_id == _BLOCK_BOOTSTRAP_TEST_ID
        and (effective_block_count is None or effective_block_count < MINIMUM_EFFECTIVE_BLOCK_COUNT)
    ):
        return "abstained", ("insufficient_effective_block_count",)
    if n_eff < HUMAN_REVIEW_N_EFF:
        return (
            "human_review_only",
            (
                "minimum_effective_sample_requires_human_review",
                "minimum_effective_information_human_review",
            ),
        )
    return None


def _resolve_effective_block_count(
    *,
    declared_test_id: str,
    raw_pair_count: int,
    effective_block_count: int | None,
    block_bootstrap_config: Mapping[str, Any] | None,
) -> int:
    if effective_block_count is not None:
        return int(effective_block_count)
    if declared_test_id == _BLOCK_BOOTSTRAP_TEST_ID:
        block_length = int((block_bootstrap_config or {}).get("block_length") or 1)
        return raw_pair_count // max(1, block_length)
    return raw_pair_count


def _nonstationarity_detected(
    *,
    status: str,
    diagnostic: Mapping[str, Any] | None,
) -> bool:
    if diagnostic is not None and _nonstationarity_handled(diagnostic):
        return False
    if status in {"failed", "detected", "nonstationary", "unstable"}:
        return True
    if diagnostic is None:
        return False
    diagnostic_status = str(diagnostic.get("status", ""))
    if diagnostic_status in {"failed", "detected", "nonstationary", "unstable"}:
        return True
    reason_codes = {
        str(reason_code) for reason_code in diagnostic.get("reason_codes", ())
    }
    return bool(
        reason_codes
        & {
            "nonstationarity_detected",
            "structural_break_detected",
            "stability_test_failed",
            "instability_evidence_unresolved",
            "regime_instability_detected",
        }
    )


def _nonstationarity_handled(diagnostic: Mapping[str, Any]) -> bool:
    handling = diagnostic.get("nonstationarity_handling")
    if not isinstance(handling, Mapping):
        return False
    if str(handling.get("status", "")) != "passed":
        return False
    lane_id = str(handling.get("lane_id", ""))
    artifact_ref = handling.get("artifact_ref")
    return bool(lane_id and isinstance(artifact_ref, Mapping))


def _many_model_pairwise_correction_missing(
    *,
    declared_test_id: str,
    paired_stream_identity: Mapping[str, Any] | None,
) -> bool:
    if declared_test_id != _DEFAULT_DECLARED_TEST_ID:
        return False
    if paired_stream_identity is None:
        return False
    try:
        model_count = int(paired_stream_identity.get("comparison_model_count", 2))
    except (TypeError, ValueError):
        model_count = 2
    if model_count <= 2:
        return False
    comparison_regime_id = str(
        paired_stream_identity.get("comparison_regime_id", "")
    )
    if comparison_regime_id != "multi_model_candidate_selection":
        return False
    many_model_adjustment_id = str(
        paired_stream_identity.get("many_model_adjustment_id", "none")
    )
    return many_model_adjustment_id not in _MULTI_MODEL_TEST_IDS


def _diagnostic_reason_codes(
    *,
    differentials: Sequence[float],
    practical_margin: float,
) -> tuple[str, ...]:
    mean_differential = _stable_float(fmean(float(value) for value in differentials))
    reason_codes: list[str] = []
    if math.isclose(mean_differential, 0.0, abs_tol=1e-12):
        reason_codes.append("baseline_tie")
    if mean_differential <= float(practical_margin):
        reason_codes.append("insignificant_improvement")
    interval = _simple_mean_interval(differentials)
    if interval is not None and interval[0] <= float(practical_margin):
        reason_codes.append("uncertainty_interval_crosses_margin")
    return _unique(reason_codes)


def _simple_mean_interval(
    differentials: Sequence[float],
) -> tuple[float, float] | None:
    values = tuple(float(value) for value in differentials)
    if len(values) < 2:
        return None
    mean = fmean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance / len(values))
    radius = 1.96 * standard_error
    return (_stable_float(mean - radius), _stable_float(mean + radius))


def _actual_test_id(declared_test_id: str) -> str:
    if declared_test_id == _GW_PUBLIC_TEST_ID:
        return _GW_INTERNAL_TEST_ID
    if declared_test_id in {
        _DEFAULT_DECLARED_TEST_ID,
        _BLOCK_BOOTSTRAP_TEST_ID,
        *_MULTI_MODEL_TEST_IDS,
    }:
        return declared_test_id
    return "unsupported_declared_test_id"


def _minimum_pair_policy_manifest() -> dict[str, int]:
    return {
        "minimum_effective_sample_size": MINIMUM_N_EFF,
        "human_review_effective_sample_size": HUMAN_REVIEW_N_EFF,
        "minimum_effective_block_count": MINIMUM_EFFECTIVE_BLOCK_COUNT,
    }


def _finite_tuple(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        return ()
    return result


def _group_mean(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[float]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(float(row["loss_difference"]))
    return [
        {key: group_key, "mean_loss_difference": _stable_float(fmean(values))}
        for group_key, values in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _rolling_degradation(
    rows: Sequence[Mapping[str, Any]],
    *,
    rolling_window: int,
) -> list[dict[str, Any]]:
    if len(rows) < rolling_window:
        return []
    windows: list[dict[str, Any]] = []
    for index in range(rolling_window - 1, len(rows)):
        window = rows[index - rolling_window + 1 : index + 1]
        latest = float(window[-1]["loss_difference"])
        mean_difference = fmean(float(row["loss_difference"]) for row in window)
        windows.append(
            {
                "end_origin_id": window[-1]["origin_id"],
                "mean_loss_difference": _stable_float(mean_difference),
                "status": "degraded" if latest < 0 else "stable",
            }
        )
    return windows


def _jsonable(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, default=str))


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
    "PredictivePromotionResult",
    "PrequentialScoreStream",
    "build_prequential_score_stream",
    "evaluate_predictive_promotion",
    "run_declared_predictive_test",
]
