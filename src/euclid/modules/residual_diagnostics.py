from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AssessmentStatus = Literal["supported", "unsupported", "indeterminate"]
ResidualDiagnosticsStatus = Literal[
    "structured_residual_remains",
    "noise_like_residual",
    "indeterminate",
]

DEFAULT_MIN_SAMPLE_COUNT = 32


@dataclass(frozen=True)
class FiniteDimensionalityEvidence:
    sample_count: int
    dimension_upper_bound: int | None = None
    replicated_across_splits: bool = False
    explicit_infinite_dimensionality_evidence: bool = False


@dataclass(frozen=True)
class RecoverabilityEvidence:
    sample_count: int
    state_reconstruction_passed: bool | None = None
    replicated_across_splits: bool = False
    explicit_unrecoverable_evidence: bool = False


@dataclass(frozen=True)
class ResidualDiagnosticsEvidence:
    finite_dimensionality: FiniteDimensionalityEvidence
    recoverability: RecoverabilityEvidence


@dataclass(frozen=True)
class DiagnosticAssessment:
    status: AssessmentStatus
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class ResidualDiagnosticsResult:
    status: ResidualDiagnosticsStatus
    residual_law_search_eligible: bool
    finite_dimensionality_status: AssessmentStatus
    recoverability_status: AssessmentStatus
    reason_codes: tuple[str, ...]


def assess_finite_dimensionality(
    evidence: FiniteDimensionalityEvidence,
    *,
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT,
) -> DiagnosticAssessment:
    _validate_sample_count(evidence.sample_count)
    if evidence.sample_count < min_sample_count:
        return DiagnosticAssessment(
            status="indeterminate",
            reason_codes=("finite_dimensionality_insufficient_data",),
        )
    if evidence.explicit_infinite_dimensionality_evidence:
        return DiagnosticAssessment(
            status="unsupported",
            reason_codes=("finite_dimensionality_rejected",),
        )
    if evidence.dimension_upper_bound is None:
        return DiagnosticAssessment(
            status="indeterminate",
            reason_codes=("finite_dimensionality_unmeasured",),
        )
    if evidence.dimension_upper_bound < 1:
        return DiagnosticAssessment(
            status="unsupported",
            reason_codes=("finite_dimensionality_invalid_dimension",),
        )
    if not evidence.replicated_across_splits:
        return DiagnosticAssessment(
            status="indeterminate",
            reason_codes=("finite_dimensionality_unreplicated",),
        )
    return DiagnosticAssessment(status="supported", reason_codes=())


def assess_recoverability(
    evidence: RecoverabilityEvidence,
    *,
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT,
) -> DiagnosticAssessment:
    _validate_sample_count(evidence.sample_count)
    if evidence.sample_count < min_sample_count:
        return DiagnosticAssessment(
            status="indeterminate",
            reason_codes=("recoverability_insufficient_data",),
        )
    if evidence.explicit_unrecoverable_evidence:
        return DiagnosticAssessment(
            status="unsupported",
            reason_codes=("recoverability_rejected",),
        )
    if evidence.state_reconstruction_passed is None:
        return DiagnosticAssessment(
            status="indeterminate",
            reason_codes=("recoverability_unmeasured",),
        )
    if evidence.state_reconstruction_passed is False:
        return DiagnosticAssessment(
            status="unsupported",
            reason_codes=("recoverability_failed",),
        )
    if not evidence.replicated_across_splits:
        return DiagnosticAssessment(
            status="indeterminate",
            reason_codes=("recoverability_unreplicated",),
        )
    return DiagnosticAssessment(status="supported", reason_codes=())


def evaluate_residual_diagnostics(
    evidence: ResidualDiagnosticsEvidence,
    *,
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT,
) -> ResidualDiagnosticsResult:
    finite_dimensionality = assess_finite_dimensionality(
        evidence.finite_dimensionality,
        min_sample_count=min_sample_count,
    )
    recoverability = assess_recoverability(
        evidence.recoverability,
        min_sample_count=min_sample_count,
    )
    reason_codes = _stable_reason_codes(
        finite_dimensionality.reason_codes,
        recoverability.reason_codes,
    )

    if (
        finite_dimensionality.status == "supported"
        and recoverability.status == "supported"
    ):
        return ResidualDiagnosticsResult(
            status="structured_residual_remains",
            residual_law_search_eligible=True,
            finite_dimensionality_status=finite_dimensionality.status,
            recoverability_status=recoverability.status,
            reason_codes=reason_codes,
        )

    if (
        finite_dimensionality.status == "unsupported"
        and recoverability.status == "unsupported"
    ):
        return ResidualDiagnosticsResult(
            status="noise_like_residual",
            residual_law_search_eligible=False,
            finite_dimensionality_status=finite_dimensionality.status,
            recoverability_status=recoverability.status,
            reason_codes=reason_codes,
        )

    if {finite_dimensionality.status, recoverability.status} == {
        "supported",
        "unsupported",
    }:
        reason_codes = _stable_reason_codes(reason_codes, ("diagnostic_conflict",))

    return ResidualDiagnosticsResult(
        status="indeterminate",
        residual_law_search_eligible=False,
        finite_dimensionality_status=finite_dimensionality.status,
        recoverability_status=recoverability.status,
        reason_codes=reason_codes,
    )


def _validate_sample_count(sample_count: int) -> None:
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")


def _stable_reason_codes(*reason_groups: tuple[str, ...]) -> tuple[str, ...]:
    ordered_reason_codes: list[str] = []
    for reason_group in reason_groups:
        for reason_code in reason_group:
            if reason_code not in ordered_reason_codes:
                ordered_reason_codes.append(reason_code)
    return tuple(ordered_reason_codes)


__all__ = [
    "AssessmentStatus",
    "DEFAULT_MIN_SAMPLE_COUNT",
    "DiagnosticAssessment",
    "FiniteDimensionalityEvidence",
    "RecoverabilityEvidence",
    "ResidualDiagnosticsEvidence",
    "ResidualDiagnosticsResult",
    "ResidualDiagnosticsStatus",
    "assess_finite_dimensionality",
    "assess_recoverability",
    "evaluate_residual_diagnostics",
]
