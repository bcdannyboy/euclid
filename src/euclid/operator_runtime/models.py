from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from euclid.contracts.refs import TypedRef

ADMITTED_FORECAST_OBJECT_TYPES = (
    "point",
    "distribution",
    "interval",
    "quantile",
    "event_probability",
)
DEFAULT_RUN_SUPPORT_OBJECT_IDS = (
    "target_transform_object",
    "quantization_object",
    "observation_model_object",
    "reference_description_object",
    "codelength_policy_object",
)
DEFAULT_ADMISSIBILITY_RULE_IDS = (
    "family_membership",
    "composition_closure",
    "observation_model_compatibility",
    "valid_state_semantics",
    "codelength_comparability",
)
RETAINED_POINT_FAMILY_IDS = (
    "constant",
    "drift",
    "linear_trend",
    "seasonal_naive",
)


def _typed_ref_from_payload(payload: Mapping[str, Any] | None) -> TypedRef | None:
    if payload is None:
        return None
    schema_name = payload.get("schema_name")
    object_id = payload.get("object_id")
    if not isinstance(schema_name, str) or not isinstance(object_id, str):
        raise ValueError("typed ref payload must include schema_name and object_id")
    return TypedRef(schema_name=schema_name, object_id=object_id)


def _typed_ref_payload(ref: TypedRef | None) -> dict[str, str] | None:
    return None if ref is None else ref.as_dict()


def derive_extension_lane_ids(
    *,
    search_family_ids: tuple[str, ...],
    forecast_object_type: str,
    external_evidence_enabled: bool = False,
    mechanistic_evidence_enabled: bool = False,
    robustness_override_enabled: bool = False,
) -> tuple[str, ...]:
    extension_lane_ids: list[str] = []
    for family_id in search_family_ids:
        if family_id not in RETAINED_POINT_FAMILY_IDS:
            extension_lane_ids.append(family_id)
    if forecast_object_type != "point":
        extension_lane_ids.append(forecast_object_type)
    if external_evidence_enabled:
        extension_lane_ids.append("external_evidence")
    if mechanistic_evidence_enabled:
        extension_lane_ids.append("mechanistic_evidence")
    if robustness_override_enabled:
        extension_lane_ids.append("robustness")
    return tuple(dict.fromkeys(extension_lane_ids))


@dataclass(frozen=True)
class OperatorRequest:
    request_id: str
    manifest_path: Path
    dataset_csv: Path
    cutoff_available_at: str | None
    quantization_step: str
    minimum_description_gain_bits: float
    min_train_size: int = 3
    horizon: int = 1
    search_family_ids: tuple[str, ...] = RETAINED_POINT_FAMILY_IDS
    search_class: str = "bounded_heuristic"
    search_seed: str = "0"
    proposal_limit: int | None = None
    seasonal_period: int = 2
    forecast_object_type: str = "point"
    primary_score_id: str | None = None
    calibration_thresholds: Mapping[str, float] | None = None
    external_evidence_payload: Mapping[str, Any] | None = None
    mechanistic_evidence_payload: Mapping[str, Any] | None = None
    robustness_payload: Mapping[str, Any] | None = None
    declared_entity_panel: tuple[str, ...] = ()
    run_support_object_ids: tuple[str, ...] = DEFAULT_RUN_SUPPORT_OBJECT_IDS
    admissibility_rule_ids: tuple[str, ...] = DEFAULT_ADMISSIBILITY_RULE_IDS
    extension_lane_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.forecast_object_type not in ADMITTED_FORECAST_OBJECT_TYPES:
            raise ValueError(
                "forecast_object_type must be one of "
                + ", ".join(ADMITTED_FORECAST_OBJECT_TYPES)
            )
        object.__setattr__(
            self,
            "calibration_thresholds",
            dict(self.calibration_thresholds or {}),
        )
        object.__setattr__(
            self,
            "external_evidence_payload",
            (
                None
                if self.external_evidence_payload is None
                else dict(self.external_evidence_payload)
            ),
        )
        object.__setattr__(
            self,
            "mechanistic_evidence_payload",
            (
                None
                if self.mechanistic_evidence_payload is None
                else dict(self.mechanistic_evidence_payload)
            ),
        )
        object.__setattr__(
            self,
            "robustness_payload",
            None if self.robustness_payload is None else dict(self.robustness_payload),
        )
        if not self.extension_lane_ids:
            object.__setattr__(
                self,
                "extension_lane_ids",
                derive_extension_lane_ids(
                    search_family_ids=self.search_family_ids,
                    forecast_object_type=self.forecast_object_type,
                    external_evidence_enabled=self.external_evidence_payload is not None,
                    mechanistic_evidence_enabled=(
                        self.mechanistic_evidence_payload is not None
                    ),
                    robustness_override_enabled=(
                        self.robustness_payload is not None
                    ),
                ),
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "manifest_path": str(self.manifest_path),
            "dataset_csv": str(self.dataset_csv),
            "cutoff_available_at": self.cutoff_available_at,
            "quantization_step": self.quantization_step,
            "minimum_description_gain_bits": self.minimum_description_gain_bits,
            "min_train_size": self.min_train_size,
            "horizon": self.horizon,
            "search_family_ids": list(self.search_family_ids),
            "search_class": self.search_class,
            "search_seed": self.search_seed,
            "proposal_limit": self.proposal_limit,
            "seasonal_period": self.seasonal_period,
            "forecast_object_type": self.forecast_object_type,
            "primary_score_id": self.primary_score_id,
            "calibration_thresholds": dict(self.calibration_thresholds or {}),
            "external_evidence_payload": (
                None
                if self.external_evidence_payload is None
                else dict(self.external_evidence_payload)
            ),
            "mechanistic_evidence_payload": (
                None
                if self.mechanistic_evidence_payload is None
                else dict(self.mechanistic_evidence_payload)
            ),
            "robustness_payload": (
                None if self.robustness_payload is None else dict(self.robustness_payload)
            ),
            "declared_entity_panel": list(self.declared_entity_panel),
            "run_support_object_ids": list(self.run_support_object_ids),
            "admissibility_rule_ids": list(self.admissibility_rule_ids),
            "extension_lane_ids": list(self.extension_lane_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorRequest":
        return cls(
            request_id=str(payload["request_id"]),
            manifest_path=Path(str(payload["manifest_path"])),
            dataset_csv=Path(str(payload["dataset_csv"])),
            cutoff_available_at=(
                None
                if payload.get("cutoff_available_at") is None
                else str(payload["cutoff_available_at"])
            ),
            quantization_step=str(payload["quantization_step"]),
            minimum_description_gain_bits=float(
                payload["minimum_description_gain_bits"]
            ),
            min_train_size=int(payload["min_train_size"]),
            horizon=int(payload["horizon"]),
            search_family_ids=tuple(str(item) for item in payload["search_family_ids"]),
            search_class=str(payload["search_class"]),
            search_seed=str(payload["search_seed"]),
            proposal_limit=(
                None
                if payload.get("proposal_limit") is None
                else int(payload["proposal_limit"])
            ),
            seasonal_period=int(payload["seasonal_period"]),
            forecast_object_type=str(payload["forecast_object_type"]),
            primary_score_id=(
                None
                if payload.get("primary_score_id") is None
                else str(payload["primary_score_id"])
            ),
            calibration_thresholds={
                str(key): float(value)
                for key, value in dict(payload.get("calibration_thresholds", {})).items()
            },
            external_evidence_payload=(
                None
                if payload.get("external_evidence_payload") is None
                else dict(payload["external_evidence_payload"])
            ),
            mechanistic_evidence_payload=(
                None
                if payload.get("mechanistic_evidence_payload") is None
                else dict(payload["mechanistic_evidence_payload"])
            ),
            robustness_payload=(
                None
                if payload.get("robustness_payload") is None
                else dict(payload["robustness_payload"])
            ),
            declared_entity_panel=tuple(
                str(item) for item in payload.get("declared_entity_panel", ())
            ),
            run_support_object_ids=tuple(
                str(item) for item in payload["run_support_object_ids"]
            ),
            admissibility_rule_ids=tuple(
                str(item) for item in payload["admissibility_rule_ids"]
            ),
            extension_lane_ids=tuple(str(item) for item in payload["extension_lane_ids"]),
        )


@dataclass(frozen=True)
class OperatorPaths:
    output_root: Path
    active_run_root: Path
    sealed_run_root: Path
    artifact_root: Path
    metadata_store_path: Path
    control_plane_store_path: Path
    run_summary_path: Path
    cache_root: Path
    temp_root: Path
    run_lock_path: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "output_root": str(self.output_root),
            "active_run_root": str(self.active_run_root),
            "sealed_run_root": str(self.sealed_run_root),
            "artifact_root": str(self.artifact_root),
            "metadata_store_path": str(self.metadata_store_path),
            "control_plane_store_path": str(self.control_plane_store_path),
            "run_summary_path": str(self.run_summary_path),
            "cache_root": str(self.cache_root),
            "temp_root": str(self.temp_root),
            "run_lock_path": str(self.run_lock_path),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorPaths":
        return cls(
            output_root=Path(str(payload["output_root"])),
            active_run_root=Path(str(payload["active_run_root"])),
            sealed_run_root=Path(str(payload["sealed_run_root"])),
            artifact_root=Path(str(payload["artifact_root"])),
            metadata_store_path=Path(str(payload["metadata_store_path"])),
            control_plane_store_path=Path(str(payload["control_plane_store_path"])),
            run_summary_path=Path(str(payload["run_summary_path"])),
            cache_root=Path(str(payload["cache_root"])),
            temp_root=Path(str(payload["temp_root"])),
            run_lock_path=Path(str(payload["run_lock_path"])),
        )


@dataclass(frozen=True)
class OperatorRunSummary:
    selected_candidate_id: str
    selected_family: str
    forecast_object_type: str
    result_mode: str
    bundle_ref: TypedRef
    run_result_ref: TypedRef
    scope_ledger_ref: TypedRef
    selected_candidate_ref: TypedRef | None
    publication_record_ref: TypedRef | None
    comparison_universe_ref: TypedRef | None
    scorecard_ref: TypedRef | None
    claim_card_ref: TypedRef | None
    abstention_ref: TypedRef | None
    prediction_artifact_ref: TypedRef | None
    primary_score_result_ref: TypedRef | None
    primary_calibration_result_ref: TypedRef | None
    confirmatory_primary_score: float
    calibration_status: str | None = None
    run_support_object_ids: tuple[str, ...] = DEFAULT_RUN_SUPPORT_OBJECT_IDS
    admissibility_rule_ids: tuple[str, ...] = DEFAULT_ADMISSIBILITY_RULE_IDS
    extension_lane_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "selected_family": self.selected_family,
            "forecast_object_type": self.forecast_object_type,
            "result_mode": self.result_mode,
            "bundle_ref": self.bundle_ref.as_dict(),
            "run_result_ref": self.run_result_ref.as_dict(),
            "scope_ledger_ref": self.scope_ledger_ref.as_dict(),
            "selected_candidate_ref": _typed_ref_payload(self.selected_candidate_ref),
            "publication_record_ref": _typed_ref_payload(self.publication_record_ref),
            "comparison_universe_ref": _typed_ref_payload(self.comparison_universe_ref),
            "scorecard_ref": _typed_ref_payload(self.scorecard_ref),
            "claim_card_ref": _typed_ref_payload(self.claim_card_ref),
            "abstention_ref": _typed_ref_payload(self.abstention_ref),
            "prediction_artifact_ref": _typed_ref_payload(self.prediction_artifact_ref),
            "primary_score_result_ref": _typed_ref_payload(
                self.primary_score_result_ref
            ),
            "primary_calibration_result_ref": _typed_ref_payload(
                self.primary_calibration_result_ref
            ),
            "confirmatory_primary_score": self.confirmatory_primary_score,
            "calibration_status": self.calibration_status,
            "run_support_object_ids": list(self.run_support_object_ids),
            "admissibility_rule_ids": list(self.admissibility_rule_ids),
            "extension_lane_ids": list(self.extension_lane_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorRunSummary":
        return cls(
            selected_candidate_id=str(payload["selected_candidate_id"]),
            selected_family=str(payload["selected_family"]),
            forecast_object_type=str(payload["forecast_object_type"]),
            result_mode=str(payload["result_mode"]),
            bundle_ref=_typed_ref_from_payload(
                dict(payload["bundle_ref"])
            ),
            run_result_ref=_typed_ref_from_payload(
                dict(payload["run_result_ref"])
            ),
            scope_ledger_ref=_typed_ref_from_payload(
                dict(payload["scope_ledger_ref"])
            ),
            selected_candidate_ref=_typed_ref_from_payload(
                payload.get("selected_candidate_ref")
            ),
            publication_record_ref=_typed_ref_from_payload(
                payload.get("publication_record_ref")
            ),
            comparison_universe_ref=_typed_ref_from_payload(
                payload.get("comparison_universe_ref")
            ),
            scorecard_ref=_typed_ref_from_payload(payload.get("scorecard_ref")),
            claim_card_ref=_typed_ref_from_payload(payload.get("claim_card_ref")),
            abstention_ref=_typed_ref_from_payload(payload.get("abstention_ref")),
            prediction_artifact_ref=_typed_ref_from_payload(
                payload.get("prediction_artifact_ref")
            ),
            primary_score_result_ref=_typed_ref_from_payload(
                payload.get("primary_score_result_ref")
            ),
            primary_calibration_result_ref=_typed_ref_from_payload(
                payload.get("primary_calibration_result_ref")
            ),
            confirmatory_primary_score=float(payload["confirmatory_primary_score"]),
            calibration_status=(
                None
                if payload.get("calibration_status") is None
                else str(payload["calibration_status"])
            ),
            run_support_object_ids=tuple(
                str(item) for item in payload["run_support_object_ids"]
            ),
            admissibility_rule_ids=tuple(
                str(item) for item in payload["admissibility_rule_ids"]
            ),
            extension_lane_ids=tuple(str(item) for item in payload["extension_lane_ids"]),
        )


@dataclass(frozen=True)
class OperatorRunResult:
    request: OperatorRequest
    paths: OperatorPaths
    summary: OperatorRunSummary

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.as_dict(),
            "paths": self.paths.as_dict(),
            "summary": self.summary.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorRunResult":
        return cls(
            request=OperatorRequest.from_dict(dict(payload["request"])),
            paths=OperatorPaths.from_dict(dict(payload["paths"])),
            summary=OperatorRunSummary.from_dict(dict(payload["summary"])),
        )


@dataclass(frozen=True)
class OperatorReplaySummary:
    bundle_ref: TypedRef
    run_result_ref: TypedRef
    selected_candidate_ref: TypedRef | None
    selected_family: str
    result_mode: str
    forecast_object_type: str
    replay_verification_status: str
    confirmatory_primary_score: float
    failure_reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_ref": self.bundle_ref.as_dict(),
            "run_result_ref": self.run_result_ref.as_dict(),
            "selected_candidate_ref": _typed_ref_payload(self.selected_candidate_ref),
            "selected_family": self.selected_family,
            "result_mode": self.result_mode,
            "forecast_object_type": self.forecast_object_type,
            "replay_verification_status": self.replay_verification_status,
            "confirmatory_primary_score": self.confirmatory_primary_score,
            "failure_reason_codes": list(self.failure_reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorReplaySummary":
        return cls(
            bundle_ref=_typed_ref_from_payload(dict(payload["bundle_ref"])),
            run_result_ref=_typed_ref_from_payload(dict(payload["run_result_ref"])),
            selected_candidate_ref=_typed_ref_from_payload(
                payload.get("selected_candidate_ref")
            ),
            selected_family=str(payload["selected_family"]),
            result_mode=str(payload["result_mode"]),
            forecast_object_type=str(payload["forecast_object_type"]),
            replay_verification_status=str(payload["replay_verification_status"]),
            confirmatory_primary_score=float(payload["confirmatory_primary_score"]),
            failure_reason_codes=tuple(
                str(item) for item in payload.get("failure_reason_codes", ())
            ),
        )


@dataclass(frozen=True)
class OperatorReplayResult:
    paths: OperatorPaths
    summary: OperatorReplaySummary

    def as_dict(self) -> dict[str, Any]:
        return {
            "paths": self.paths.as_dict(),
            "summary": self.summary.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorReplayResult":
        return cls(
            paths=OperatorPaths.from_dict(dict(payload["paths"])),
            summary=OperatorReplaySummary.from_dict(dict(payload["summary"])),
        )


__all__ = [
    "ADMITTED_FORECAST_OBJECT_TYPES",
    "DEFAULT_ADMISSIBILITY_RULE_IDS",
    "DEFAULT_RUN_SUPPORT_OBJECT_IDS",
    "OperatorPaths",
    "OperatorReplayResult",
    "OperatorReplaySummary",
    "OperatorRequest",
    "OperatorRunResult",
    "OperatorRunSummary",
    "RETAINED_POINT_FAMILY_IDS",
    "derive_extension_lane_ids",
]
