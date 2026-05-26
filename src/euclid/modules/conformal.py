from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from euclid.contracts.errors import ContractValidationError
from euclid.modules.calibration import (
    CalibrationPartition,
    CalibrationPartitionEvaluation,
    evaluate_calibration_partitions,
    run_mapie_time_series_adapter,
)

CONFORMAL_GUARANTEE_TIERS = (
    "finite_sample_exchangeable",
    "approximate_mixing_time_series",
    "asymptotic_time_series",
    "long_run_frequency_control",
    "diagnostic_only",
)
FINITE_SAMPLE_DISTRIBUTION_FREE_SCOPE_REASON = (
    "finite_sample_distribution_free_language_requires_"
    "finite_sample_exchangeable_tier"
)


@dataclass(frozen=True)
class ConformalMethodSpec:
    method_id: str
    guarantee_tier: str
    assumption_ids: tuple[str, ...] = ()
    assumption_scope: str = "none"
    finite_sample_distribution_free_allowed: bool = False
    fixed_time_finite_sample_allowed: bool = False
    approximate_coverage_allowed: bool = False
    long_run_coverage_allowed: bool = False

    def __post_init__(self) -> None:
        if self.guarantee_tier not in CONFORMAL_GUARANTEE_TIERS:
            raise ContractValidationError(
                code="unknown_conformal_guarantee_tier",
                message=f"unknown conformal guarantee tier {self.guarantee_tier!r}",
                field_path="guarantee_tier",
                details={"guarantee_tier": self.guarantee_tier},
            )

    @property
    def finite_sample_distribution_free_claim_allowed(self) -> bool:
        return self.finite_sample_distribution_free_allowed

    @property
    def fixed_time_finite_sample_claim_allowed(self) -> bool:
        return self.fixed_time_finite_sample_allowed


@dataclass(frozen=True)
class ConformalMethodResolution:
    method_id: str
    status: str
    reason_codes: tuple[str, ...]
    guarantee_tier: str
    assumption_ids: tuple[str, ...]
    assumption_scope: str
    calibration_split_ids: tuple[str, ...]
    horizon_ids: tuple[int, ...]
    declarations: Mapping[str, Any] = field(default_factory=dict)
    finite_sample_distribution_free_allowed: bool = False
    fixed_time_finite_sample_allowed: bool = False
    approximate_coverage_allowed: bool = False
    long_run_coverage_allowed: bool = False

    @property
    def finite_sample_distribution_free_claim_allowed(self) -> bool:
        return self.finite_sample_distribution_free_allowed

    @property
    def fixed_time_finite_sample_claim_allowed(self) -> bool:
        return self.fixed_time_finite_sample_allowed

    def as_manifest(self) -> dict[str, Any]:
        calibration_split_ids = list(self.calibration_split_ids)
        return {
            "schema_name": "conformal_method_resolution@1.0.0",
            "method_id": self.method_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "guarantee_tier": self.guarantee_tier,
            "assumption_scope": self.assumption_scope,
            "assumption_ids": list(self.assumption_ids),
            "calibration_split_id": (
                calibration_split_ids[0] if calibration_split_ids else None
            ),
            "calibration_split_ids": calibration_split_ids,
            "horizon_ids": list(self.horizon_ids),
            "finite_sample_distribution_free_allowed": (
                self.finite_sample_distribution_free_allowed
            ),
            "finite_sample_distribution_free_claim_allowed": (
                self.finite_sample_distribution_free_claim_allowed
            ),
            "fixed_time_finite_sample_allowed": (
                self.fixed_time_finite_sample_allowed
            ),
            "fixed_time_finite_sample_claim_allowed": (
                self.fixed_time_finite_sample_claim_allowed
            ),
            "approximate_coverage_allowed": self.approximate_coverage_allowed,
            "approximate_coverage_claim_allowed": self.approximate_coverage_allowed,
            "long_run_coverage_allowed": self.long_run_coverage_allowed,
            "long_run_coverage_claim_allowed": self.long_run_coverage_allowed,
            "declarations": _stable_mapping(self.declarations),
            "assumption_declarations": _stable_mapping(self.declarations),
        }


_REGISTRY = MappingProxyType(
    {
        "split_conformal_exchangeable_v1": ConformalMethodSpec(
            method_id="split_conformal_exchangeable_v1",
            guarantee_tier="finite_sample_exchangeable",
            assumption_ids=("exchangeability",),
            assumption_scope="exchangeable",
            finite_sample_distribution_free_allowed=True,
            fixed_time_finite_sample_allowed=True,
        ),
        "enbpi_time_series_v1": ConformalMethodSpec(
            method_id="enbpi_time_series_v1",
            guarantee_tier="approximate_mixing_time_series",
            assumption_ids=("weak_dependence_or_mixing",),
            assumption_scope="mixing_time_series",
            approximate_coverage_allowed=True,
        ),
        "adaptive_conformal_time_series_v1": ConformalMethodSpec(
            method_id="adaptive_conformal_time_series_v1",
            guarantee_tier="long_run_frequency_control",
            assumption_ids=("online_adaptation", "long_run_frequency"),
            assumption_scope="time_series_adaptive",
            long_run_coverage_allowed=True,
        ),
    }
)


def conformal_method_registry() -> Mapping[str, ConformalMethodSpec]:
    return _REGISTRY


def resolve_conformal_method(
    *,
    method_id: str,
    calibration_split_id: str | None = None,
    calibration_split_ids: Sequence[str] | None = None,
    horizon_ids: Sequence[int] = (),
    assumption_declarations: Mapping[str, Any] | None = None,
    declarations: Mapping[str, Any] | None = None,
) -> ConformalMethodResolution:
    resolved_declarations = {
        **dict(assumption_declarations or {}),
        **dict(declarations or {}),
    }
    resolved_split_ids = _calibration_split_ids(
        calibration_split_id=calibration_split_id,
        calibration_split_ids=calibration_split_ids,
    )
    resolved_horizon_ids = tuple(_unique(int(horizon) for horizon in horizon_ids))
    spec = _REGISTRY.get(str(method_id))
    if spec is None:
        return ConformalMethodResolution(
            method_id=str(method_id),
            status="failed",
            reason_codes=("unknown_conformal_method",),
            guarantee_tier="diagnostic_only",
            assumption_ids=(),
            assumption_scope="unknown",
            calibration_split_ids=resolved_split_ids,
            horizon_ids=resolved_horizon_ids,
            declarations=resolved_declarations,
        )

    if spec.method_id == "split_conformal_exchangeable_v1":
        return _resolve_exchangeable_split(
            spec=spec,
            calibration_split_ids=resolved_split_ids,
            horizon_ids=resolved_horizon_ids,
            declarations=resolved_declarations,
        )

    reason_codes: tuple[str, ...]
    if spec.method_id == "enbpi_time_series_v1":
        reason_codes = ("finite_sample_distribution_free_not_supported",)
    elif spec.method_id == "adaptive_conformal_time_series_v1":
        reason_codes = (
            "finite_sample_distribution_free_not_supported",
            "fixed_time_finite_sample_not_supported",
        )
    else:
        reason_codes = ()
    return _resolution_from_spec(
        spec=spec,
        status="passed",
        reason_codes=reason_codes,
        calibration_split_ids=resolved_split_ids,
        horizon_ids=resolved_horizon_ids,
        declarations=resolved_declarations,
        finite_sample_distribution_free_allowed=False,
        fixed_time_finite_sample_allowed=False,
    )


def assert_conformal_claim_scope(claim_card_body: Mapping[str, Any]) -> None:
    claim_text = str(claim_card_body.get("claim_text", ""))
    method_manifest = _extract_method_manifest(claim_card_body)
    guarantee_tier = str(method_manifest.get("guarantee_tier", "diagnostic_only"))
    method_status = str(method_manifest.get("status", "failed"))
    finite_sample_distribution_free_allowed = bool(
        method_manifest.get(
            "finite_sample_distribution_free_claim_allowed",
            method_manifest.get("finite_sample_distribution_free_allowed", False),
        )
    )

    reason_codes: list[str] = []
    if _uses_finite_sample_distribution_free_language(claim_text) and not (
        guarantee_tier == "finite_sample_exchangeable"
        and method_status == "passed"
        and finite_sample_distribution_free_allowed
    ):
        reason_codes.append(FINITE_SAMPLE_DISTRIBUTION_FREE_SCOPE_REASON)
    if reason_codes:
        raise ContractValidationError(
            code="claim_scope_overstatement",
            message="claim publication overstates conformal guarantee scope",
            field_path="claim_card.claim_text",
            details={
                "guarantee_tier": guarantee_tier,
                "method_status": method_status,
                "reason_codes": reason_codes,
            },
        )


def _resolve_exchangeable_split(
    *,
    spec: ConformalMethodSpec,
    calibration_split_ids: tuple[str, ...],
    horizon_ids: tuple[int, ...],
    declarations: Mapping[str, Any],
) -> ConformalMethodResolution:
    if "exchangeability" not in declarations:
        return _resolution_from_spec(
            spec=spec,
            status="blocked",
            reason_codes=("missing_exchangeability_declaration",),
            calibration_split_ids=calibration_split_ids,
            horizon_ids=horizon_ids,
            declarations=declarations,
            finite_sample_distribution_free_allowed=False,
            fixed_time_finite_sample_allowed=False,
        )
    if not _declared(declarations.get("exchangeability")):
        return _resolution_from_spec(
            spec=spec,
            status="failed",
            reason_codes=("exchangeability_declaration_required",),
            calibration_split_ids=calibration_split_ids,
            horizon_ids=horizon_ids,
            declarations=declarations,
            finite_sample_distribution_free_allowed=False,
            fixed_time_finite_sample_allowed=False,
        )
    return _resolution_from_spec(
        spec=spec,
        status="passed",
        reason_codes=(),
        calibration_split_ids=calibration_split_ids,
        horizon_ids=horizon_ids,
        declarations=declarations,
        finite_sample_distribution_free_allowed=True,
        fixed_time_finite_sample_allowed=True,
    )


def _resolution_from_spec(
    *,
    spec: ConformalMethodSpec,
    status: str,
    reason_codes: tuple[str, ...],
    calibration_split_ids: tuple[str, ...],
    horizon_ids: tuple[int, ...],
    declarations: Mapping[str, Any],
    finite_sample_distribution_free_allowed: bool,
    fixed_time_finite_sample_allowed: bool,
) -> ConformalMethodResolution:
    return ConformalMethodResolution(
        method_id=spec.method_id,
        status=status,
        reason_codes=reason_codes,
        guarantee_tier=spec.guarantee_tier,
        assumption_ids=spec.assumption_ids,
        assumption_scope=spec.assumption_scope,
        calibration_split_ids=calibration_split_ids,
        horizon_ids=horizon_ids,
        declarations=declarations,
        finite_sample_distribution_free_allowed=finite_sample_distribution_free_allowed,
        fixed_time_finite_sample_allowed=fixed_time_finite_sample_allowed,
        approximate_coverage_allowed=spec.approximate_coverage_allowed,
        long_run_coverage_allowed=spec.long_run_coverage_allowed,
    )


def _calibration_split_ids(
    *,
    calibration_split_id: str | None,
    calibration_split_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    split_ids: list[str] = []
    if calibration_split_id is not None:
        split_ids.append(str(calibration_split_id))
    if calibration_split_ids is not None:
        split_ids.extend(str(split_id) for split_id in calibration_split_ids)
    return tuple(_unique(split_id for split_id in split_ids if split_id))


def _extract_method_manifest(claim_card_body: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = (
        claim_card_body.get("conformal_method")
        or claim_card_body.get("conformal_method_manifest")
        or claim_card_body.get("calibration_method_manifest")
        or {}
    )
    if hasattr(raw, "as_manifest"):
        return raw.as_manifest()
    if isinstance(raw, Mapping):
        body = raw.get("body")
        if isinstance(body, Mapping) and "guarantee_tier" in body:
            return body
        return raw
    return {}


def _uses_finite_sample_distribution_free_language(claim_text: str) -> bool:
    normalized = claim_text.lower().replace("-", " ")
    return "finite sample" in normalized and "distribution free" in normalized


def _declared(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in {
            "false",
            "no",
            "none",
            "not_declared",
        }
    return bool(value)


def _stable_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): mapping[key] for key in sorted(mapping)}


def _unique(values: Any) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


__all__ = [
    "CONFORMAL_GUARANTEE_TIERS",
    "CalibrationPartition",
    "CalibrationPartitionEvaluation",
    "ConformalMethodResolution",
    "ConformalMethodSpec",
    "FINITE_SAMPLE_DISTRIBUTION_FREE_SCOPE_REASON",
    "assert_conformal_claim_scope",
    "conformal_method_registry",
    "evaluate_calibration_partitions",
    "resolve_conformal_method",
    "run_mapie_time_series_adapter",
]
