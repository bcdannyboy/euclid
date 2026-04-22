from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from euclid.invariance.environments import EnvironmentConstructionResult
from euclid.invariance.scoring import (
    holdout_stability_metrics,
    parameter_stability_metrics,
    residual_invariance_metrics,
    support_stability_metrics,
)


@dataclass(frozen=True)
class InvarianceEvaluation:
    status: str
    claim_lane_allowed: bool
    reason_codes: tuple[str, ...]
    metrics: Mapping[str, float]
    environment_construction_ref: str
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "invariance_evaluation@1.0.0",
            "status": self.status,
            "claim_lane_allowed": self.claim_lane_allowed,
            "reason_codes": list(self.reason_codes),
            "metrics": dict(self.metrics),
            "environment_construction_ref": self.environment_construction_ref,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class TransportEvaluation:
    status: str
    claim_lane_allowed: bool
    reason_codes: tuple[str, ...]
    metrics: Mapping[str, float]
    source_environment_ids: tuple[str, ...]
    target_environment_ids: tuple[str, ...]
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "transport_evaluation@1.0.0",
            "status": self.status,
            "claim_lane_allowed": self.claim_lane_allowed,
            "reason_codes": list(self.reason_codes),
            "metrics": dict(self.metrics),
            "source_environment_ids": list(self.source_environment_ids),
            "target_environment_ids": list(self.target_environment_ids),
            "replay_identity": self.replay_identity,
        }


def evaluate_invariance(
    *,
    environments: EnvironmentConstructionResult,
    residuals_by_environment: Mapping[str, Sequence[float]],
    parameters_by_environment: Mapping[str, Mapping[str, float]],
    supports_by_environment: Mapping[str, set[str] | frozenset[str]],
    holdout_losses_by_environment: Mapping[str, Mapping[str, float]] | None = None,
    min_environment_count: int = 2,
    residual_spread_threshold: float = 0.05,
    parameter_drift_threshold: float = 0.05,
    min_support_jaccard: float = 1.0,
    max_holdout_degradation: float = 0.1,
) -> InvarianceEvaluation:
    reason_codes: list[str] = []
    if environments.status != "constructed":
        reason_codes.extend(environments.reason_codes or ("environment_construction_failed",))
    if len(environments.slices) < int(min_environment_count):
        reason_codes.append("insufficient_environments")

    metrics: dict[str, float] = {}
    metrics.update(residual_invariance_metrics(residuals_by_environment))
    metrics.update(parameter_stability_metrics(parameters_by_environment))
    metrics.update(support_stability_metrics(supports_by_environment))
    metrics.update(holdout_stability_metrics(holdout_losses_by_environment))

    if metrics["residual_spread"] > float(residual_spread_threshold):
        reason_codes.append("residual_invariance_failed")
    if metrics["max_parameter_drift"] > float(parameter_drift_threshold):
        reason_codes.append("parameter_drift_failed")
    if metrics["min_support_jaccard"] < float(min_support_jaccard):
        reason_codes.append("support_stability_failed")
    if metrics["max_holdout_degradation"] > float(max_holdout_degradation):
        reason_codes.append("environment_holdout_failed")

    unique_reasons = _unique(reason_codes)
    status = "passed" if not unique_reasons else "failed"
    return _invariance_result(
        status=status,
        claim_lane_allowed=status == "passed",
        reason_codes=unique_reasons,
        metrics=metrics,
        environment_construction_ref=environments.replay_identity,
    )


def evaluate_transport(
    *,
    source_environment_ids: Sequence[str],
    target_environment_ids: Sequence[str],
    holdout_scores: Mapping[str, Mapping[str, float]],
    max_transport_degradation: float = 0.25,
) -> TransportEvaluation:
    source_ids = tuple(str(item) for item in source_environment_ids if str(item))
    target_ids = tuple(str(item) for item in target_environment_ids if str(item))
    reason_codes: list[str] = []
    if not source_ids:
        reason_codes.append("missing_source_environment")
    if not target_ids:
        reason_codes.append("missing_target_environment")

    max_degradation = 0.0
    missing_targets: list[str] = []
    for target in target_ids:
        scores = holdout_scores.get(target)
        if scores is None:
            missing_targets.append(target)
            continue
        source_loss = float(scores.get("source_loss", scores.get("train", 0.0)))
        target_loss = float(scores.get("target_loss", scores.get("holdout", 0.0)))
        max_degradation = max(max_degradation, target_loss - source_loss)
    if missing_targets:
        reason_codes.append("missing_target_holdout_score")
    if max_degradation > float(max_transport_degradation):
        reason_codes.append("transport_holdout_failed")

    metrics = {"max_transport_degradation": float(round(max_degradation, 12))}
    unique_reasons = _unique(reason_codes)
    status = "passed" if not unique_reasons else "failed"
    return _transport_result(
        status=status,
        claim_lane_allowed=status == "passed",
        reason_codes=unique_reasons,
        metrics=metrics,
        source_environment_ids=source_ids,
        target_environment_ids=target_ids,
    )


def _invariance_result(
    *,
    status: str,
    claim_lane_allowed: bool,
    reason_codes: tuple[str, ...],
    metrics: Mapping[str, float],
    environment_construction_ref: str,
) -> InvarianceEvaluation:
    identity_payload = {
        "environment_construction_ref": environment_construction_ref,
        "metrics": dict(metrics),
        "reason_codes": list(reason_codes),
        "status": status,
    }
    return InvarianceEvaluation(
        status=status,
        claim_lane_allowed=claim_lane_allowed,
        reason_codes=reason_codes,
        metrics=dict(metrics),
        environment_construction_ref=environment_construction_ref,
        replay_identity=f"invariance-evaluation:{_digest(identity_payload)}",
    )


def _transport_result(
    *,
    status: str,
    claim_lane_allowed: bool,
    reason_codes: tuple[str, ...],
    metrics: Mapping[str, float],
    source_environment_ids: tuple[str, ...],
    target_environment_ids: tuple[str, ...],
) -> TransportEvaluation:
    identity_payload = {
        "metrics": dict(metrics),
        "reason_codes": list(reason_codes),
        "source_environment_ids": list(source_environment_ids),
        "status": status,
        "target_environment_ids": list(target_environment_ids),
    }
    return TransportEvaluation(
        status=status,
        claim_lane_allowed=claim_lane_allowed,
        reason_codes=reason_codes,
        metrics=dict(metrics),
        source_environment_ids=source_environment_ids,
        target_environment_ids=target_environment_ids,
        replay_identity=f"transport-evaluation:{_digest(identity_payload)}",
    )


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _unique(codes: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for code in codes:
        text = str(code)
        if text:
            seen.setdefault(text, None)
    return tuple(seen)


__all__ = [
    "InvarianceEvaluation",
    "TransportEvaluation",
    "evaluate_invariance",
    "evaluate_transport",
]
