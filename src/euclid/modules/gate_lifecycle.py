from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

_NON_BLOCKING_CALIBRATION_STATUSES = {
    "not_applicable_for_forecast_type",
    "recorded_not_gating",
    "passed",
}


@dataclass(frozen=True)
class ScorecardStatusDecision:
    descriptive_status: str
    descriptive_reason_codes: tuple[str, ...]
    predictive_status: str
    predictive_reason_codes: tuple[str, ...]
    mechanistic_status: str = "not_requested"
    mechanistic_reason_codes: tuple[str, ...] = ()


def resolve_scorecard_status(
    *,
    candidate_admissible: bool,
    robustness_status: str,
    candidate_beats_baseline: bool,
    confirmatory_promotion_allowed: bool,
    point_score_comparison_status: str,
    time_safety_status: str,
    calibration_status: str,
    descriptive_failure_reason_codes: Sequence[str] = (),
    robustness_reason_codes: Sequence[str] = (),
    predictive_governance_reason_codes: Sequence[str] = (),
    falsification_status: str = "passed",
    falsification_reason_codes: Sequence[str] = (),
    mechanistic_requested: bool = False,
    mechanistic_evidence_status: str | None = None,
    mechanistic_reason_codes: Sequence[str] = (),
) -> ScorecardStatusDecision:
    descriptive_status, descriptive_reason_codes = _resolve_descriptive_status(
        candidate_admissible=candidate_admissible,
        robustness_status=robustness_status,
        descriptive_failure_reason_codes=descriptive_failure_reason_codes,
        robustness_reason_codes=robustness_reason_codes,
    )
    predictive_status, predictive_reason_codes = _resolve_predictive_status(
        descriptive_status=descriptive_status,
        candidate_beats_baseline=candidate_beats_baseline,
        confirmatory_promotion_allowed=confirmatory_promotion_allowed,
        point_score_comparison_status=point_score_comparison_status,
        time_safety_status=time_safety_status,
        calibration_status=calibration_status,
        predictive_governance_reason_codes=predictive_governance_reason_codes,
        falsification_status=falsification_status,
        falsification_reason_codes=falsification_reason_codes,
    )
    resolved_mechanistic_status, resolved_mechanistic_reason_codes = (
        _resolve_mechanistic_status(
            mechanistic_requested=mechanistic_requested,
            predictive_status=predictive_status,
            mechanistic_evidence_status=mechanistic_evidence_status,
            mechanistic_reason_codes=mechanistic_reason_codes,
        )
    )
    return ScorecardStatusDecision(
        descriptive_status=descriptive_status,
        descriptive_reason_codes=descriptive_reason_codes,
        predictive_status=predictive_status,
        predictive_reason_codes=predictive_reason_codes,
        mechanistic_status=resolved_mechanistic_status,
        mechanistic_reason_codes=resolved_mechanistic_reason_codes,
    )


def _resolve_descriptive_status(
    *,
    candidate_admissible: bool,
    robustness_status: str,
    descriptive_failure_reason_codes: Sequence[str],
    robustness_reason_codes: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    if not candidate_admissible:
        reason_codes = _normalized_codes(
            descriptive_failure_reason_codes
            or ("descriptive_gate_failed", "no_candidate_survived_search")
        )
        if "codelength_comparability_failed" in reason_codes:
            return "blocked_codelength_comparability_failed", reason_codes
        return "blocked", reason_codes

    if robustness_status != "passed":
        reason_codes = _normalized_codes(
            ("robustness_failed", *robustness_reason_codes)
        )
        return "blocked_robustness_failed", reason_codes

    return "passed", ()


def _resolve_predictive_status(
    *,
    descriptive_status: str,
    candidate_beats_baseline: bool,
    confirmatory_promotion_allowed: bool,
    point_score_comparison_status: str,
    time_safety_status: str,
    calibration_status: str,
    predictive_governance_reason_codes: Sequence[str],
    falsification_status: str,
    falsification_reason_codes: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    if descriptive_status != "passed":
        return "not_requested", ("predictive_not_requested",)
    if time_safety_status not in {"passed", "verified"}:
        return "blocked", ("time_safety_failed",)
    if point_score_comparison_status != "comparable":
        return "blocked", ("point_score_not_comparable",)
    if calibration_status == "failed":
        return "blocked", ("calibration_failed",)
    if calibration_status not in _NON_BLOCKING_CALIBRATION_STATUSES:
        return "blocked", ("calibration_record_missing_or_invalid",)
    if falsification_status not in {"passed", "not_requested"}:
        return (
            "blocked",
            _normalized_codes(
                ("falsification_failed", *falsification_reason_codes)
            ),
        )
    if not candidate_beats_baseline:
        return "blocked", ("baseline_rule_failed",)
    if not confirmatory_promotion_allowed:
        return (
            "blocked",
            _normalized_codes(
                predictive_governance_reason_codes or ("many_model_correction_failed",)
            ),
        )
    return "passed", ()


def _resolve_mechanistic_status(
    *,
    mechanistic_requested: bool,
    predictive_status: str,
    mechanistic_evidence_status: str | None,
    mechanistic_reason_codes: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    if not mechanistic_requested:
        return "not_requested", ()
    if predictive_status != "passed":
        return "blocked_predictive_floor", ("predictive_floor_required",)
    if mechanistic_evidence_status == "passed":
        return "passed", ()
    if mechanistic_evidence_status == "blocked_predictive_floor":
        return "blocked_predictive_floor", ("predictive_floor_required",)
    return (
        "downgraded_to_predictive_within_declared_scope",
        _normalized_codes(
            mechanistic_reason_codes or ("mechanistic_requirements_failed",)
        ),
    )


def _normalized_codes(codes: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for code in codes:
        text = str(code)
        if text:
            seen.setdefault(text, None)
    return tuple(seen)
