from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from euclid.contracts.errors import ContractValidationError

_EVIDENCE_STATUSES = frozenset(
    {"passed", "failed", "abstained", "downgraded"}
)

_KNOWN_REASON_CODES = frozenset(
    {
        "baseline_rule_failed",
        "biased_residuals",
        "calibration_failed",
        "calibration_record_missing_or_invalid",
        "bootstrap_test_backend_unavailable",
        "codelength_comparability_failed",
        "counterexample_discovered",
        "descriptive_gate_failed",
        "domain_violation",
        "extrapolation_failure",
        "falsification_failed",
        "baseline_tie",
        "leakage_canary_failed",
        "leakage_detected",
        "candidate_refit_failed_on_surrogate",
        "downstream_artifact_created",
        "gw_backend_not_implemented",
        "gw_requires_instruments_or_state",
        "insignificant_improvement",
        "insufficient_effective_block_count",
        "insufficient_effective_sample_size",
        "insufficient_parameter_windows",
        "insufficient_paired_count",
        "insufficient_residual_support",
        "invariance_check_failed",
        "invariance_evidence_missing",
        "invalid_surrogate_residual_test",
        "many_model_correction_failed",
        "malformed_null_protocol",
        "mechanistic_requirements_failed",
        "mechanism_mapping_incomplete",
        "mechanism_mapping_missing",
        "metric_prerequisite_missing",
        "missing_stage_evidence",
        "missing_baseline",
        "missing_conditional_instrument_declarations",
        "missing_perturbation_runs",
        "missing_practical_effect_margin",
        "multi_model_superiority_not_tested",
        "multi_model_test_backend_unavailable",
        "no_candidate_survived_search",
        "no_valid_perturbation_runs",
        "nonstationarity_detected",
        "nonfinite_metric_value",
        "nonfinite_observation",
        "nonfinite_parameter_estimate",
        "nonfinite_residual",
        "nonfinite_statistic",
        "null_protocol_failed",
        "outside_law_eligible_scope",
        "parameter_instability",
        "point_score_not_comparable",
        "poor_coverage",
        "perturbation_protocol_failed",
        "perturbation_instability",
        "predictive_evidence_overlap",
        "predictive_floor_required",
        "predictive_governance_blocked",
        "predictive_not_requested",
        "required_metric_has_no_applicable_family",
        "robustness_failed",
        "structured_residual_remains",
        "structured_residuals",
        "structured_residuals_vs_surrogate",
        "surrogate_generation_failed",
        "time_safety_failed",
        "transport_failed",
        "residual_outlier",
        "stochastic_miscalibration",
        "uncertainty_interval_crosses_margin",
        "unpaired_loss_stream",
        "unsupported_declared_predictive_test_id",
        "unsupported_declared_test_id",
        "unexpected_canary_survival",
        "unstable_split_protocol",
        "units_check_incompatible",
        "units_check_incomplete",
        "units_check_missing",
        "unsupported_null_statistic",
        "wrong_block_reason_code",
        "wrong_block_stage",
    }
)


def _normalized_codes(
    reason_codes: Sequence[str],
    *,
    allowed_reason_codes: Iterable[str] | None,
    require_non_empty: bool,
    field_path: str,
) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    allowed = (
        frozenset(allowed_reason_codes)
        if allowed_reason_codes is not None
        else _KNOWN_REASON_CODES
    )
    for index, reason_code in enumerate(reason_codes):
        if not isinstance(reason_code, str) or not reason_code.strip():
            raise ContractValidationError(
                code="empty_evidence_reason_code",
                message="evidence reason codes must be non-empty strings",
                field_path=f"{field_path}[{index}]",
            )
        if reason_code not in allowed:
            raise ContractValidationError(
                code="unknown_evidence_reason_code",
                message=f"unknown evidence reason code {reason_code!r}",
                field_path=f"{field_path}[{index}]",
                details={"reason_code": reason_code},
            )
        seen.setdefault(reason_code, None)
    normalized = tuple(seen)
    if require_non_empty and not normalized:
        raise ContractValidationError(
            code="missing_evidence_reason_code",
            message="non-passed evidence statuses require at least one reason code",
            field_path=field_path,
        )
    return normalized


def _ref_as_manifest(ref: Any) -> Any:
    if hasattr(ref, "as_dict"):
        return ref.as_dict()
    if isinstance(ref, Mapping):
        return {key: ref[key] for key in sorted(ref)}
    return ref


@dataclass(frozen=True)
class EvidenceStatus:
    status: str
    reason_codes: tuple[str, ...] = ()
    evidence_refs: tuple[Any, ...] = ()
    artifacts_required: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    allowed_reason_codes: frozenset[str] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.status not in _EVIDENCE_STATUSES:
            raise ContractValidationError(
                code="unknown_evidence_status",
                message=f"unknown evidence status {self.status!r}",
                field_path="status",
                details={"status": self.status},
            )
        normalized_codes = _normalized_codes(
            self.reason_codes,
            allowed_reason_codes=self.allowed_reason_codes,
            require_non_empty=self.status != "passed",
            field_path="reason_codes",
        )
        if (
            self.status == "passed"
            and self.artifacts_required
            and not self.evidence_refs
        ):
            raise ContractValidationError(
                code="missing_required_evidence_refs",
                message=(
                    "passed evidence status requires evidence refs for "
                    "artifact-backed gates"
                ),
                field_path="evidence_refs",
            )
        object.__setattr__(self, "reason_codes", normalized_codes)
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def passed(
        cls,
        *,
        evidence_refs: Sequence[Any] = (),
        artifacts_required: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> "EvidenceStatus":
        return cls(
            status="passed",
            reason_codes=(),
            evidence_refs=tuple(evidence_refs),
            artifacts_required=artifacts_required,
            metadata=metadata or {},
        )

    @classmethod
    def failed(
        cls,
        reason_codes: Sequence[str],
        *,
        evidence_refs: Sequence[Any] = (),
        metadata: Mapping[str, Any] | None = None,
        allowed_reason_codes: Iterable[str] | None = None,
    ) -> "EvidenceStatus":
        return cls(
            status="failed",
            reason_codes=tuple(reason_codes),
            evidence_refs=tuple(evidence_refs),
            metadata=metadata or {},
            allowed_reason_codes=(
                frozenset(allowed_reason_codes)
                if allowed_reason_codes is not None
                else None
            ),
        )

    @classmethod
    def abstained(
        cls,
        reason_codes: Sequence[str],
        *,
        evidence_refs: Sequence[Any] = (),
        metadata: Mapping[str, Any] | None = None,
        allowed_reason_codes: Iterable[str] | None = None,
    ) -> "EvidenceStatus":
        return cls(
            status="abstained",
            reason_codes=tuple(reason_codes),
            evidence_refs=tuple(evidence_refs),
            metadata=metadata or {},
            allowed_reason_codes=(
                frozenset(allowed_reason_codes)
                if allowed_reason_codes is not None
                else None
            ),
        )

    @classmethod
    def downgraded(
        cls,
        reason_codes: Sequence[str],
        *,
        evidence_refs: Sequence[Any] = (),
        metadata: Mapping[str, Any] | None = None,
        allowed_reason_codes: Iterable[str] | None = None,
    ) -> "EvidenceStatus":
        return cls(
            status="downgraded",
            reason_codes=tuple(reason_codes),
            evidence_refs=tuple(evidence_refs),
            metadata=metadata or {},
            allowed_reason_codes=(
                frozenset(allowed_reason_codes)
                if allowed_reason_codes is not None
                else None
            ),
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "evidence_refs": [_ref_as_manifest(ref) for ref in self.evidence_refs],
            "metadata": {key: self.metadata[key] for key in sorted(self.metadata)},
        }


@dataclass(frozen=True)
class EvidenceGateDecision:
    gate_id: str
    status: EvidenceStatus

    def as_manifest(self) -> dict[str, Any]:
        status_manifest = self.status.as_manifest()
        return {
            "gate_id": self.gate_id,
            "status": status_manifest["status"],
            "reason_codes": status_manifest["reason_codes"],
            "evidence_refs": status_manifest["evidence_refs"],
            "metadata": status_manifest["metadata"],
        }


@dataclass(frozen=True)
class ClaimScopeDecision:
    claim_scope: str
    status: EvidenceStatus

    def as_manifest(self) -> dict[str, Any]:
        status_manifest = self.status.as_manifest()
        return {
            "claim_scope": self.claim_scope,
            "status": status_manifest["status"],
            "reason_codes": status_manifest["reason_codes"],
            "evidence_refs": status_manifest["evidence_refs"],
            "metadata": status_manifest["metadata"],
        }


@dataclass(frozen=True)
class PromotionDecision:
    promotion_id: str
    status: EvidenceStatus

    def as_manifest(self) -> dict[str, Any]:
        status_manifest = self.status.as_manifest()
        return {
            "promotion_id": self.promotion_id,
            "status": status_manifest["status"],
            "reason_codes": status_manifest["reason_codes"],
            "evidence_refs": status_manifest["evidence_refs"],
            "metadata": status_manifest["metadata"],
        }


def build_evidence_gate_decision(
    *,
    gate_id: str,
    legacy_status: str,
    reason_codes: Sequence[str],
    evidence_refs: Sequence[Any] = (),
    artifacts_required: bool = False,
) -> EvidenceGateDecision:
    metadata: dict[str, Any] = {}
    if legacy_status == "passed":
        status = EvidenceStatus.passed(
            evidence_refs=evidence_refs,
            artifacts_required=artifacts_required,
        )
    elif legacy_status in {
        "blocked",
        "blocked_codelength_comparability_failed",
        "blocked_robustness_failed",
        "blocked_predictive_floor",
    }:
        metadata["legacy_status"] = legacy_status
        status = EvidenceStatus.failed(reason_codes, metadata=metadata)
    elif legacy_status == "not_requested":
        status = EvidenceStatus.abstained(
            reason_codes or (f"{gate_id}_not_requested",),
            metadata={"legacy_status": legacy_status},
            allowed_reason_codes=(
                *sorted(_KNOWN_REASON_CODES),
                f"{gate_id}_not_requested",
            ),
        )
    elif legacy_status == "downgraded_to_predictive_within_declared_scope":
        metadata["legacy_status"] = legacy_status
        status = EvidenceStatus.downgraded(reason_codes, metadata=metadata)
    else:
        raise ContractValidationError(
            code="unknown_evidence_status",
            message=f"unknown legacy evidence status {legacy_status!r}",
            field_path=f"{gate_id}.status",
            details={"status": legacy_status},
        )
    return EvidenceGateDecision(gate_id=gate_id, status=status)


def build_scorecard_manifest_body(
    *,
    scorecard_id: str,
    candidate_ref: Any,
    descriptive: EvidenceStatus,
    predictive: EvidenceStatus,
    forecast_object_type: str,
    point_score_result_ref: Any | None = None,
    calibration_result_ref: Any | None = None,
    robustness_report_ref: Any | None = None,
    mechanistic: EvidenceStatus | None = None,
    extra_fields: Mapping[str, Any] | None = None,
    **refs: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "scorecard_id": scorecard_id,
        "candidate_ref": _ref_as_manifest(candidate_ref),
    }
    for field_name, ref in {
        **refs,
        "point_score_result_ref": point_score_result_ref,
        "calibration_result_ref": calibration_result_ref,
        "robustness_report_ref": robustness_report_ref,
    }.items():
        if ref is not None:
            body[field_name] = _ref_as_manifest(ref)
    body.update(
        {
            "forecast_object_type": forecast_object_type,
            "descriptive_status": descriptive.status,
            "descriptive_reason_codes": list(descriptive.reason_codes),
            "descriptive_evidence_refs": [
                _ref_as_manifest(ref) for ref in descriptive.evidence_refs
            ],
            "predictive_status": predictive.status,
            "predictive_reason_codes": list(predictive.reason_codes),
            "predictive_evidence_refs": [
                _ref_as_manifest(ref) for ref in predictive.evidence_refs
            ],
        }
    )
    if mechanistic is not None:
        body.update(
            {
                "mechanistic_status": mechanistic.status,
                "mechanistic_reason_codes": list(mechanistic.reason_codes),
                "mechanistic_evidence_refs": [
                    _ref_as_manifest(ref) for ref in mechanistic.evidence_refs
                ],
            }
        )
    if extra_fields:
        body.update(extra_fields)
    return body


def build_claim_card_manifest_body(
    *,
    claim_card_id: str,
    candidate_ref: Any,
    scorecard_ref: Any,
    validation_scope_ref: Any,
    claim_decision: Any,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "claim_card_id": claim_card_id,
        "candidate_ref": _ref_as_manifest(candidate_ref),
        "scorecard_ref": _ref_as_manifest(scorecard_ref),
        "validation_scope_ref": _ref_as_manifest(validation_scope_ref),
        "claim_type": claim_decision.claim_type,
        "claim_ceiling": claim_decision.claim_ceiling,
        "predictive_support_status": claim_decision.predictive_support_status,
        "allowed_interpretation_codes": list(
            claim_decision.allowed_interpretation_codes
        ),
        "forbidden_interpretation_codes": list(
            claim_decision.forbidden_interpretation_codes
        ),
    }
    if extra_fields:
        body.update(extra_fields)
    return body
