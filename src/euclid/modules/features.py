from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.modules.snapshotting import FrozenDatasetSnapshot
from euclid.modules.timeguard import CanaryFailure, TimeSafetyAudit

_REUSABLE_STAGE_IDS = ("search", "candidate_fitting", "evaluation")
_LAG_ALLOWED_KEYS = frozenset({"feature_id", "kind", "lag_steps"})
_ROLLING_ALLOWED_KEYS = frozenset({"feature_id", "kind", "window"})


@dataclass(frozen=True)
class FeatureCoordinateAudit:
    coordinate_index: int
    event_time: str
    available_at: str
    legal_history_indices: tuple[int, ...]
    source_row_indices: Mapping[str, tuple[int, ...]]
    status: str
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "available_at": self.available_at,
            "coordinate_index": self.coordinate_index,
            "event_time": self.event_time,
            "legal_history_indices": list(self.legal_history_indices),
            "reason_codes": list(self.reason_codes),
            "source_row_indices": {
                feature_name: list(indices)
                for feature_name, indices in sorted(self.source_row_indices.items())
            },
            "status": self.status,
        }


@dataclass(frozen=True)
class FeatureMaterializationReport:
    status: str
    history_access_law: str
    reusable_stage_ids: tuple[str, ...]
    included_coordinate_count: int
    excluded_coordinate_count: int
    coordinate_audits: tuple[FeatureCoordinateAudit, ...]
    canary_failures: tuple[CanaryFailure, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "canary_failures": [
                canary_failure.as_dict()
                for canary_failure in self.canary_failures
            ],
            "coordinate_audits": [
                coordinate_audit.as_dict()
                for coordinate_audit in self.coordinate_audits
            ],
            "excluded_coordinate_count": self.excluded_coordinate_count,
            "history_access_law": self.history_access_law,
            "included_coordinate_count": self.included_coordinate_count,
            "reusable_stage_ids": list(self.reusable_stage_ids),
            "status": self.status,
        }


@dataclass(frozen=True)
class FeatureSpec:
    feature_spec_id: str
    features: tuple[Mapping[str, Any], ...]

    def to_manifest(self, catalog: ContractCatalog) -> ManifestEnvelope:
        return ManifestEnvelope.build(
            schema_name="feature_spec@1.0.0",
            module_id="features",
            body={
                "feature_spec_id": self.feature_spec_id,
                "owner_id": "module.features-v1",
                "scope_id": "euclid_v1_binding_scope@1.0.0",
                "features": [dict(feature) for feature in self.features],
            },
            catalog=catalog,
        )


@dataclass(frozen=True)
class FeatureView:
    series_id: str
    feature_names: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    entity_panel: tuple[str, ...] = ()
    materialization_report: FeatureMaterializationReport | None = None

    def require_stage_reuse(self, stage_id: str) -> "FeatureView":
        report = self.materialization_report
        if report is None:
            raise ContractValidationError(
                code="missing_feature_contract",
                message="feature view is missing the materialization contract",
                field_path="feature_view.materialization_report",
            )
        if report.status != "passed":
            raise ContractValidationError(
                code="illegal_feature_reuse",
                message="feature view cannot be reused because materialization failed",
                field_path="feature_view.materialization_report.status",
            )
        if stage_id not in report.reusable_stage_ids:
            raise ContractValidationError(
                code="illegal_feature_reuse",
                message=f"{stage_id!r} is not an admitted feature-view reuse stage",
                field_path="feature_view.materialization_report.reusable_stage_ids",
            )
        return self

    def to_manifest(
        self,
        catalog: ContractCatalog,
        *,
        snapshot_ref: TypedRef | None = None,
        feature_spec_ref: TypedRef | None = None,
        time_safety_audit_ref: TypedRef | None = None,
    ) -> ManifestEnvelope:
        body: dict[str, Any] = {
            "feature_view_id": f"{self.series_id}_feature_view_v1",
            "owner_id": "module.features-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "series_id": self.series_id,
            "feature_names": list(self.feature_names),
            "row_count": len(self.rows),
            "rows": list(self.rows),
        }
        if self.entity_panel:
            body["entity_panel"] = list(self.entity_panel)
        if self.materialization_report is not None:
            body["materialization_report"] = self.materialization_report.as_dict()
        if snapshot_ref is not None:
            body["snapshot_ref"] = snapshot_ref.as_dict()
        if feature_spec_ref is not None:
            body["feature_spec_ref"] = feature_spec_ref.as_dict()
        if time_safety_audit_ref is not None:
            body["time_safety_audit_ref"] = time_safety_audit_ref.as_dict()
        return ManifestEnvelope.build(
            schema_name="feature_view_manifest@1.0.0",
            module_id="features",
            body=body,
            catalog=catalog,
        )


def default_feature_spec() -> FeatureSpec:
    return FeatureSpec(
        feature_spec_id="prototype_lag_feature_spec_v1",
        features=(
            {"feature_id": "lag_1", "kind": "lag", "lag_steps": 1},
        ),
    )


def materialize_feature_view(
    *,
    snapshot: FrozenDatasetSnapshot,
    audit: TimeSafetyAudit,
    feature_spec: FeatureSpec,
) -> FeatureView:
    if audit.status != "passed":
        raise ContractValidationError(
            code="time_safety_blocked",
            message="cannot materialize a feature view from a blocked snapshot",
            field_path="audit.status",
        )

    feature_defs = list(feature_spec.features)
    if not feature_defs:
        raise ContractValidationError(
            code="illegal_feature_spec",
            message="at least one feature is required",
            field_path="feature_spec.features",
        )

    feature_names: list[str] = []
    normalized_defs: list[tuple[str, str, int]] = []
    warmup_requirement = 0
    for feature in feature_defs:
        kind = str(feature.get("kind", ""))
        feature_id = str(feature.get("feature_id", "")).strip()
        if not feature_id:
            raise ContractValidationError(
                code="illegal_feature_spec",
                message="every feature requires a non-empty feature_id",
                field_path="feature_spec.features",
            )
        if kind == "lag":
            extra_keys = sorted(set(feature) - _LAG_ALLOWED_KEYS)
            if extra_keys:
                raise ContractValidationError(
                    code="illegal_feature_spec",
                    message=(
                        "lag features may not depend on cached transforms "
                        "or derived-column inputs in retained scope"
                    ),
                    field_path="feature_spec.features",
                    details={"extra_keys": extra_keys, "feature_id": feature_id},
                )
            lag_steps = int(feature.get("lag_steps", 0))
            if lag_steps < 1:
                raise ContractValidationError(
                    code="illegal_feature_spec",
                    message="lag features require lag_steps >= 1",
                    field_path="feature_spec.features",
                )
            normalized_defs.append((kind, feature_id, lag_steps))
            warmup_requirement = max(warmup_requirement, lag_steps)
        elif kind == "rolling_mean":
            extra_keys = sorted(set(feature) - _ROLLING_ALLOWED_KEYS)
            if extra_keys:
                raise ContractValidationError(
                    code="illegal_feature_spec",
                    message=(
                        "rolling features may not depend on cached transforms "
                        "or derived-column inputs in retained scope"
                    ),
                    field_path="feature_spec.features",
                    details={"extra_keys": extra_keys, "feature_id": feature_id},
                )
            window = int(feature.get("window", 0))
            if window < 1:
                raise ContractValidationError(
                    code="illegal_feature_spec",
                    message="rolling_mean features require window >= 1",
                    field_path="feature_spec.features",
                )
            normalized_defs.append((kind, feature_id, window))
            warmup_requirement = max(warmup_requirement, window)
        else:
            raise ContractValidationError(
                code="illegal_feature_spec",
                message=(
                    "retained scope admits lag and trailing rolling_mean "
                    "features only in this slice"
                ),
                field_path="feature_spec.features",
            )
        feature_names.append(feature_id)

    rows: list[dict[str, Any]] = []
    coordinate_audits: list[FeatureCoordinateAudit] = []
    canary_failures: list[CanaryFailure] = []
    rows_by_entity: dict[str, tuple[tuple[int, Any], ...]] = {
        entity: tuple(
            (index, row)
            for index, row in enumerate(snapshot.rows)
            if (row.entity or snapshot.series_id) == entity
        )
        for entity in snapshot.entity_panel
    }
    for entity in snapshot.entity_panel:
        entity_rows = rows_by_entity[entity]
        for entity_row_index in range(warmup_requirement, len(entity_rows)):
            current_index, current = entity_rows[entity_row_index]
            legal_history_indices = tuple(
                history_index
                for history_index, (_, history_row) in enumerate(
                    entity_rows[:entity_row_index]
                )
                if history_row.event_time < current.event_time
                and history_row.available_at <= current.available_at
            )

            row_reasons: set[str] = set()
            relevant_late_source_indices: set[int] = set()
            source_row_indices: dict[str, tuple[int, ...]] = {}
            row = {
                "entity": entity,
                "entity_row_index": entity_row_index,
                "event_time": current.event_time,
                "available_at": current.available_at,
                "target": current.observed_value,
            }
            for kind, feature_id, parameter in normalized_defs:
                if kind == "lag":
                    naive_local_indices = (entity_row_index - parameter,)
                else:
                    naive_local_indices = tuple(
                        range(entity_row_index - parameter, entity_row_index)
                    )
                relevant_late_source_indices.update(
                    entity_rows[source_index][0]
                    for source_index in naive_local_indices
                    if entity_rows[source_index][1].available_at > current.available_at
                )
                if len(legal_history_indices) < parameter:
                    row_reasons.add("insufficient_legal_history")
                    continue
                if kind == "lag":
                    selected_local_indices = (legal_history_indices[-parameter],)
                    selected_indices = tuple(
                        entity_rows[selected_local_indices[0]][0] for _ in (0,)
                    )
                    source_row_indices[feature_id] = selected_indices
                    row[feature_id] = entity_rows[selected_local_indices[0]][
                        1
                    ].observed_value
                    continue

                selected_local_indices = legal_history_indices[-parameter:]
                selected_indices = tuple(
                    entity_rows[selected_index][0]
                    for selected_index in selected_local_indices
                )
                source_row_indices[feature_id] = selected_indices
                values = [
                    entity_rows[selected_index][1].observed_value
                    for selected_index in selected_local_indices
                ]
                row[feature_id] = sum(values) / len(values)

            if relevant_late_source_indices:
                canary_failures.append(
                    CanaryFailure(
                        canary_id=(
                            f"features:{entity}:{current_index}:late_source_availability"
                        ),
                        stage_id="feature_materialization",
                        coordinate_index=current_index,
                        entity=entity,
                        event_time=current.event_time,
                        available_at=current.available_at,
                        reason_code="late_source_availability",
                        details={
                            "blocked_source_indices": sorted(
                                relevant_late_source_indices
                            )
                        },
                    )
                )

            normalized_reasons = tuple(sorted(row_reasons))
            coordinate_audits.append(
                FeatureCoordinateAudit(
                    coordinate_index=current_index,
                    event_time=current.event_time,
                    available_at=current.available_at,
                    legal_history_indices=tuple(
                        entity_rows[index][0] for index in legal_history_indices
                    ),
                    source_row_indices=source_row_indices,
                    status="excluded" if normalized_reasons else "materialized",
                    reason_codes=normalized_reasons,
                )
            )
            if normalized_reasons:
                continue
            rows.append(row)

    materialization_report = FeatureMaterializationReport(
        status="passed",
        history_access_law="past_only_visible_history_by_availability_time",
        reusable_stage_ids=_REUSABLE_STAGE_IDS,
        included_coordinate_count=len(rows),
        excluded_coordinate_count=len(coordinate_audits) - len(rows),
        coordinate_audits=tuple(coordinate_audits),
        canary_failures=tuple(canary_failures),
    )
    return FeatureView(
        series_id=snapshot.series_id,
        feature_names=tuple(feature_names),
        rows=tuple(rows),
        entity_panel=snapshot.entity_panel,
        materialization_report=materialization_report,
    )
