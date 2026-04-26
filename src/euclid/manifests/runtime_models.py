from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog, parse_schema_name
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.runtime.hashing import normalize_json_value

_DEFAULT_SCOPE_ID = "euclid_v1_binding_scope@1.0.0"


def _dedupe_refs(refs: tuple[TypedRef, ...]) -> tuple[TypedRef, ...]:
    seen: set[tuple[str, str]] = set()
    ordered: list[TypedRef] = []
    for ref in refs:
        key = (ref.schema_name, ref.object_id)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(ref)
    return tuple(ordered)


def _typed_ref_from_payload(payload: Any, *, field_path: str) -> TypedRef:
    if not isinstance(payload, Mapping):
        raise ContractValidationError(
            code="invalid_typed_ref_shape",
            message="typed refs must be mappings with schema_name and object_id",
            field_path=field_path,
        )
    schema_name = payload.get("schema_name")
    object_id = payload.get("object_id")
    if not isinstance(schema_name, str):
        raise ContractValidationError(
            code="invalid_typed_ref_shape",
            message="typed refs require schema_name to be a string",
            field_path=f"{field_path}.schema_name",
        )
    if not isinstance(object_id, str):
        raise ContractValidationError(
            code="invalid_typed_ref_shape",
            message="typed refs require object_id to be a string",
            field_path=f"{field_path}.object_id",
        )
    parse_schema_name(schema_name)
    return TypedRef(schema_name=schema_name, object_id=object_id)


def _typed_ref_list_from_payload(
    payload: Any, *, field_path: str
) -> tuple[TypedRef, ...]:
    if not isinstance(payload, list):
        raise ContractValidationError(
            code="invalid_ref_collection",
            message=f"{field_path} must be a list",
            field_path=field_path,
        )
    return tuple(
        _typed_ref_from_payload(item, field_path=f"{field_path}[{index}]")
        for index, item in enumerate(payload)
    )


def _string_list_from_payload(payload: Any, *, field_path: str) -> tuple[str, ...]:
    if not isinstance(payload, list):
        raise ContractValidationError(
            code="invalid_ref_collection",
            message=f"{field_path} must be a list",
            field_path=field_path,
        )
    values: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, str):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path}[{index}] must be a string",
                field_path=f"{field_path}[{index}]",
            )
        values.append(item)
    return tuple(values)


def _string_mapping_from_payload(payload: Any, *, field_path: str) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise ContractValidationError(
            code="invalid_manifest_model_field",
            message=f"{field_path} must be a mapping",
            field_path=field_path,
        )
    values: dict[str, str] = {}
    for key, item in payload.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path} must map string keys to string values",
                field_path=field_path,
            )
        values[key] = item
    return values


@dataclass(frozen=True, kw_only=True)
class RuntimeManifestModel:
    object_id: str | None = None
    parent_refs: tuple[TypedRef, ...] = ()

    schema_name: ClassVar[str]
    module_id: ClassVar[str]

    @property
    def schema_family(self) -> str:
        return parse_schema_name(self.schema_name)[0]

    @property
    def schema_version(self) -> str:
        return parse_schema_name(self.schema_name)[1]

    @property
    def ref(self) -> TypedRef:
        if self.object_id is None:
            raise ContractValidationError(
                code="invalid_object_id",
                message="manifest model ref requires a non-empty object_id",
                field_path="object_id",
            )
        return TypedRef(schema_name=self.schema_name, object_id=self.object_id)

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return ()

    @property
    def lineage_refs(self) -> tuple[TypedRef, ...]:
        return _dedupe_refs(self.body_refs + self.parent_refs)

    def body(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_manifest(self, catalog: ContractCatalog) -> ManifestEnvelope:
        return ManifestEnvelope.build(
            schema_name=self.schema_name,
            module_id=self.module_id,
            body=self.body(),
            catalog=catalog,
            object_id=self.object_id,
        )

    @classmethod
    def _validate_manifest(cls, manifest: ManifestEnvelope) -> None:
        if manifest.schema_name != cls.schema_name:
            raise ContractValidationError(
                code="manifest_model_schema_mismatch",
                message=f"{cls.__name__} expects {cls.schema_name}",
                field_path="schema_name",
                details={
                    "expected_schema_name": cls.schema_name,
                    "schema_name": manifest.schema_name,
                },
            )


@dataclass(frozen=True)
class ArtifactHashRecord:
    artifact_role: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {"artifact_role": self.artifact_role, "sha256": self.sha256}

    @classmethod
    def from_payload(cls, payload: Any, *, field_path: str) -> "ArtifactHashRecord":
        if not isinstance(payload, Mapping):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path} must be a mapping",
                field_path=field_path,
            )
        artifact_role = payload.get("artifact_role")
        sha256 = payload.get("sha256")
        if not isinstance(artifact_role, str) or not isinstance(sha256, str):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path} requires string artifact_role and sha256",
                field_path=field_path,
            )
        return cls(artifact_role=artifact_role, sha256=sha256)


@dataclass(frozen=True)
class SeedRecord:
    seed_scope: str
    seed_value: str

    def as_dict(self) -> dict[str, str]:
        return {"seed_scope": self.seed_scope, "seed_value": self.seed_value}

    @classmethod
    def from_payload(cls, payload: Any, *, field_path: str) -> "SeedRecord":
        if not isinstance(payload, Mapping):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path} must be a mapping",
                field_path=field_path,
            )
        seed_scope = payload.get("seed_scope")
        seed_value = payload.get("seed_value")
        if not isinstance(seed_scope, str) or not isinstance(seed_value, str):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path} requires string seed_scope and seed_value",
                field_path=field_path,
            )
        return cls(seed_scope=seed_scope, seed_value=seed_value)


@dataclass(frozen=True)
class ReplayStageRecord:
    stage_id: str
    manifest_ref: TypedRef

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "manifest_ref": self.manifest_ref.as_dict(),
        }

    @classmethod
    def from_payload(cls, payload: Any, *, field_path: str) -> "ReplayStageRecord":
        if not isinstance(payload, Mapping):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path} must be a mapping",
                field_path=field_path,
            )
        stage_id = payload.get("stage_id")
        if not isinstance(stage_id, str):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path}.stage_id must be a string",
                field_path=f"{field_path}.stage_id",
            )
        return cls(
            stage_id=stage_id,
            manifest_ref=_typed_ref_from_payload(
                payload.get("manifest_ref"),
                field_path=f"{field_path}.manifest_ref",
            ),
        )


@dataclass(frozen=True)
class PredictionRow:
    origin_time: str
    available_at: str
    horizon: int
    point_forecast: float
    realized_observation: float
    entity: str | None = None

    def as_dict(self) -> dict[str, Any]:
        body = {
            "origin_time": self.origin_time,
            "available_at": self.available_at,
            "horizon": self.horizon,
            "point_forecast": self.point_forecast,
            "realized_observation": self.realized_observation,
        }
        if self.entity is not None:
            body["entity"] = self.entity
        return body

    @classmethod
    def from_payload(cls, payload: Any, *, field_path: str) -> "PredictionRow":
        if not isinstance(payload, Mapping):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path} must be a mapping",
                field_path=field_path,
            )
        return cls(
            origin_time=str(payload["origin_time"]),
            available_at=str(payload["available_at"]),
            horizon=int(payload["horizon"]),
            point_forecast=float(payload["point_forecast"]),
            realized_observation=float(payload["realized_observation"]),
            entity=(
                str(payload["entity"]) if payload.get("entity") is not None else None
            ),
        )


@dataclass(frozen=True)
class DistributionPredictionRow:
    origin_time: str
    available_at: str
    horizon: int
    distribution_family: str
    location: float
    scale: float
    support_kind: str
    realized_observation: float
    entity: str | None = None
    distribution_parameters: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        body = {
            "origin_time": self.origin_time,
            "available_at": self.available_at,
            "horizon": self.horizon,
            "distribution_family": self.distribution_family,
            "location": self.location,
            "scale": self.scale,
            "support_kind": self.support_kind,
            "distribution_parameters": {
                str(key): float(value)
                for key, value in self.distribution_parameters.items()
            },
            "realized_observation": self.realized_observation,
        }
        if self.entity is not None:
            body["entity"] = self.entity
        return body


@dataclass(frozen=True)
class IntervalValue:
    nominal_coverage: float
    lower_bound: float
    upper_bound: float

    def as_dict(self) -> dict[str, float]:
        return {
            "nominal_coverage": float(self.nominal_coverage),
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
        }


@dataclass(frozen=True)
class IntervalPredictionRow:
    origin_time: str
    available_at: str
    horizon: int
    nominal_coverage: float
    lower_bound: float
    upper_bound: float
    realized_observation: float
    entity: str | None = None
    distribution_family: str | None = None
    distribution_parameters: Mapping[str, float] = field(default_factory=dict)
    intervals: tuple[IntervalValue | Mapping[str, float], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        body = {
            "origin_time": self.origin_time,
            "available_at": self.available_at,
            "horizon": self.horizon,
            "nominal_coverage": self.nominal_coverage,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "realized_observation": self.realized_observation,
        }
        if self.distribution_family is not None:
            body["distribution_family"] = self.distribution_family
            body["distribution_parameters"] = {
                str(key): float(value)
                for key, value in self.distribution_parameters.items()
            }
        if self.intervals:
            body["intervals"] = [
                item.as_dict() if isinstance(item, IntervalValue) else dict(item)
                for item in self.intervals
            ]
        if self.entity is not None:
            body["entity"] = self.entity
        return body


@dataclass(frozen=True)
class QuantileValue:
    level: float
    value: float

    def as_dict(self) -> dict[str, float]:
        return {"level": self.level, "value": self.value}


@dataclass(frozen=True)
class QuantilePredictionRow:
    origin_time: str
    available_at: str
    horizon: int
    quantiles: tuple[QuantileValue, ...]
    realized_observation: float
    entity: str | None = None
    distribution_family: str | None = None
    distribution_parameters: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        body = {
            "origin_time": self.origin_time,
            "available_at": self.available_at,
            "horizon": self.horizon,
            "quantiles": [item.as_dict() for item in self.quantiles],
            "realized_observation": self.realized_observation,
        }
        if self.distribution_family is not None:
            body["distribution_family"] = self.distribution_family
            body["distribution_parameters"] = {
                str(key): float(value)
                for key, value in self.distribution_parameters.items()
            }
        if self.entity is not None:
            body["entity"] = self.entity
        return body


@dataclass(frozen=True)
class EventProbabilityPredictionRow:
    origin_time: str
    available_at: str
    horizon: int
    event_definition: Mapping[str, Any]
    event_probability: float
    realized_observation: float
    realized_event: bool
    entity: str | None = None
    distribution_family: str | None = None
    distribution_parameters: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        body = {
            "origin_time": self.origin_time,
            "available_at": self.available_at,
            "horizon": self.horizon,
            "event_definition": dict(self.event_definition),
            "event_probability": self.event_probability,
            "realized_observation": self.realized_observation,
            "realized_event": self.realized_event,
        }
        if self.distribution_family is not None:
            body["distribution_family"] = self.distribution_family
            body["distribution_parameters"] = {
                str(key): float(value)
                for key, value in self.distribution_parameters.items()
            }
        if self.entity is not None:
            body["entity"] = self.entity
        return body


PredictionArtifactRow = (
    PredictionRow
    | DistributionPredictionRow
    | IntervalPredictionRow
    | QuantilePredictionRow
    | EventProbabilityPredictionRow
)


@dataclass(frozen=True)
class PerHorizonScore:
    horizon: int
    valid_origin_count: int
    mean_point_loss: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "valid_origin_count": self.valid_origin_count,
            "mean_point_loss": self.mean_point_loss,
        }

    @classmethod
    def from_payload(cls, payload: Any, *, field_path: str) -> "PerHorizonScore":
        if not isinstance(payload, Mapping):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path} must be a mapping",
                field_path=field_path,
            )
        return cls(
            horizon=int(payload["horizon"]),
            valid_origin_count=int(payload["valid_origin_count"]),
            mean_point_loss=float(payload["mean_point_loss"]),
        )


@dataclass(frozen=True)
class PerHorizonPrimaryScore:
    horizon: int
    valid_origin_count: int
    mean_primary_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "valid_origin_count": self.valid_origin_count,
            "mean_primary_score": self.mean_primary_score,
        }


@dataclass(frozen=True, kw_only=True)
class RunDeclarationManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "run_manifest@1.0.0"
    module_id: ClassVar[str] = "manifest_registry"

    run_id: str
    entrypoint_id: str
    requested_at: str
    lifecycle_state: str = "run_declared"
    scope_id: str = _DEFAULT_SCOPE_ID
    forecast_object_type: str = "point"
    requested_manifest_refs: tuple[TypedRef, ...] = ()
    seed_records: tuple[SeedRecord, ...] = ()

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return self.requested_manifest_refs

    def body(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "entrypoint_id": self.entrypoint_id,
            "requested_at": self.requested_at,
            "lifecycle_state": self.lifecycle_state,
            "scope_id": self.scope_id,
            "forecast_object_type": self.forecast_object_type,
            "requested_manifest_refs": [
                ref.as_dict() for ref in self.requested_manifest_refs
            ],
            "seed_records": [record.as_dict() for record in self.seed_records],
        }

    @classmethod
    def from_manifest(cls, manifest: ManifestEnvelope) -> "RunDeclarationManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            run_id=str(manifest.body["run_id"]),
            entrypoint_id=str(manifest.body["entrypoint_id"]),
            requested_at=str(manifest.body["requested_at"]),
            lifecycle_state=str(manifest.body["lifecycle_state"]),
            scope_id=str(manifest.body["scope_id"]),
            forecast_object_type=str(
                manifest.body.get("forecast_object_type", "point")
            ),
            requested_manifest_refs=_typed_ref_list_from_payload(
                manifest.body.get("requested_manifest_refs", []),
                field_path="body.requested_manifest_refs",
            ),
            seed_records=tuple(
                SeedRecord.from_payload(
                    item,
                    field_path=f"body.seed_records[{index}]",
                )
                for index, item in enumerate(manifest.body.get("seed_records", []))
            ),
        )


@dataclass(frozen=True, kw_only=True)
class DatasetSnapshotManifestModel(RuntimeManifestModel):
    schema_name: ClassVar[str] = "dataset_snapshot_manifest@1.0.0"
    module_id: ClassVar[str] = "snapshotting"

    series_id: str
    cutoff_available_at: str
    revision_policy: str
    rows: tuple[Mapping[str, Any], ...]
    entity_panel: tuple[str, ...] = ()
    raw_observations: tuple[Mapping[str, Any], ...] = ()
    sampling_metadata: Mapping[str, Any] | None = None
    source_provenance: tuple[Mapping[str, Any], ...] = ()
    admitted_side_information_fields: tuple[str, ...] = ()
    raw_observation_hash: str | None = None
    coded_target_hash: str | None = None
    lineage_payload_hash: str | None = None
    lineage_payload_hashes: tuple[str, ...] = ()
    snapshot_id: str | None = None
    owner_id: str = "module.snapshotting-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    def body(self) -> dict[str, Any]:
        event_rows = self.raw_observations or self.rows
        return {
            "snapshot_id": self.snapshot_id or f"{self.series_id}_dataset_snapshot_v1",
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "series_id": self.series_id,
            "entity_panel": list(self.entity_panel),
            "cutoff_available_at": self.cutoff_available_at,
            "revision_policy": self.revision_policy,
            "row_count": len(self.rows),
            "first_event_time": event_rows[0]["event_time"] if event_rows else None,
            "last_event_time": event_rows[-1]["event_time"] if event_rows else None,
            "rows": [dict(row) for row in self.rows],
            "coded_targets": [dict(row) for row in self.rows],
            "raw_observations": [dict(row) for row in self.raw_observations],
            "sampling_metadata": dict(self.sampling_metadata or {}),
            "source_provenance": [dict(record) for record in self.source_provenance],
            "admitted_side_information_fields": list(
                self.admitted_side_information_fields
            ),
            "raw_observation_hash": self.raw_observation_hash,
            "coded_target_hash": self.coded_target_hash,
            "lineage_payload_hash": self.lineage_payload_hash,
            "lineage_payload_hashes": list(self.lineage_payload_hashes),
        }

    @classmethod
    def from_manifest(
        cls, manifest: ManifestEnvelope
    ) -> "DatasetSnapshotManifestModel":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            series_id=str(manifest.body["series_id"]),
            entity_panel=_string_list_from_payload(
                manifest.body.get("entity_panel", []),
                field_path="body.entity_panel",
            ),
            cutoff_available_at=str(manifest.body["cutoff_available_at"]),
            revision_policy=str(manifest.body["revision_policy"]),
            rows=tuple(
                dict(row)
                for row in manifest.body.get(
                    "coded_targets",
                    manifest.body.get("rows", []),
                )
            ),
            raw_observations=tuple(
                dict(row) for row in manifest.body.get("raw_observations", [])
            ),
            sampling_metadata=dict(manifest.body.get("sampling_metadata", {})),
            source_provenance=tuple(
                dict(record) for record in manifest.body.get("source_provenance", [])
            ),
            admitted_side_information_fields=_string_list_from_payload(
                manifest.body.get("admitted_side_information_fields", []),
                field_path="body.admitted_side_information_fields",
            ),
            raw_observation_hash=(
                str(manifest.body["raw_observation_hash"])
                if manifest.body.get("raw_observation_hash") is not None
                else None
            ),
            coded_target_hash=(
                str(manifest.body["coded_target_hash"])
                if manifest.body.get("coded_target_hash") is not None
                else None
            ),
            lineage_payload_hash=(
                str(manifest.body["lineage_payload_hash"])
                if manifest.body.get("lineage_payload_hash") is not None
                else None
            ),
            lineage_payload_hashes=_string_list_from_payload(
                manifest.body.get("lineage_payload_hashes", []),
                field_path="body.lineage_payload_hashes",
            ),
            snapshot_id=(
                str(manifest.body["snapshot_id"])
                if manifest.body.get("snapshot_id") is not None
                else None
            ),
            owner_id=str(manifest.body.get("owner_id", "module.snapshotting-v1")),
            scope_id=str(manifest.body.get("scope_id", _DEFAULT_SCOPE_ID)),
        )


@dataclass(frozen=True, kw_only=True)
class EvaluationPlanManifestModel(RuntimeManifestModel):
    schema_name: ClassVar[str] = "evaluation_plan_manifest@1.1.0"
    module_id: ClassVar[str] = "split_planning"

    folds: tuple[Mapping[str, Any], ...]
    time_safety_audit_ref: TypedRef
    feature_view_ref: TypedRef | None = None
    evaluation_plan_id: str = "prototype_nested_walk_forward_plan_v1"
    owner_prompt_id: str = "prompt.predictive-validation-v1"
    scope_id: str = _DEFAULT_SCOPE_ID
    forecast_object_type: str = "point"
    split_strategy: str = "nested_walk_forward_only"
    inner_search_policy: str = "fold_local_only"
    confirmatory_holdout_policy: str = "single_use_sealed"
    max_confirmatory_accesses: int = 1
    reusable_holdout_policy: str = "not_supported"
    entity_panel: tuple[str, ...] = ()

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        refs = [self.time_safety_audit_ref]
        if self.feature_view_ref is not None:
            refs.append(self.feature_view_ref)
        return tuple(refs)

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "evaluation_plan_id": self.evaluation_plan_id,
            "owner_prompt_id": self.owner_prompt_id,
            "scope_id": self.scope_id,
            "forecast_object_type": self.forecast_object_type,
            "split_strategy": self.split_strategy,
            "inner_search_policy": self.inner_search_policy,
            "outer_fold_count": len(self.folds),
            "confirmatory_holdout_policy": self.confirmatory_holdout_policy,
            "max_confirmatory_accesses": self.max_confirmatory_accesses,
            "reusable_holdout_policy": self.reusable_holdout_policy,
            "time_safety_audit_ref": self.time_safety_audit_ref.as_dict(),
            "folds": [dict(fold) for fold in self.folds],
        }
        if self.entity_panel:
            body["entity_panel"] = list(self.entity_panel)
        if self.feature_view_ref is not None:
            body["feature_view_ref"] = self.feature_view_ref.as_dict()
        return body


@dataclass(frozen=True, kw_only=True)
class CanonicalizationPolicyManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "canonicalization_policy_manifest@1.0.0"
    module_id: ClassVar[str] = "search_planning"

    canonicalization_policy_id: str
    owner_id: str = "module.search-planning-v1"
    scope_id: str = _DEFAULT_SCOPE_ID
    canonical_form_id: str = "search_normal_form_v1"
    commutative_child_sorting: str = "lexicographic_by_canonical_subtree"
    literal_normalization: str = "lattice_quantized_decimal_string"
    piecewise_branch_order: str = "ascending_split_literal"
    additive_residual_child_order: str = "base_then_residual"

    def body(self) -> dict[str, Any]:
        return {
            "canonicalization_policy_id": self.canonicalization_policy_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "canonical_form_id": self.canonical_form_id,
            "commutative_child_sorting": self.commutative_child_sorting,
            "literal_normalization": self.literal_normalization,
            "piecewise_branch_order": self.piecewise_branch_order,
            "additive_residual_child_order": self.additive_residual_child_order,
        }


@dataclass(frozen=True, kw_only=True)
class SearchPlanManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "search_plan_manifest@1.0.0"
    module_id: ClassVar[str] = "search_planning"

    search_plan_id: str
    canonicalization_policy_ref: TypedRef
    codelength_policy_ref: TypedRef
    reference_description_policy_ref: TypedRef
    observation_model_ref: TypedRef
    predictive_mode: str
    forecast_object_type: str = "point"
    search_class: str = "bounded_heuristic"
    primitive_families: tuple[str, ...] = ("analytic", "recursive", "spectral")
    composition_operators: tuple[str, ...] = (
        "piecewise",
        "additive_residual",
        "regime_conditioned",
    )
    candidate_family_ids: tuple[str, ...] = ()
    fold_local_search_required: bool = True
    max_candidate_count: int = 1
    random_seed: str = "0"
    proposal_limit: int = 1
    frontier_width: int = 1
    shortlist_limit: int = 1
    wall_clock_budget_seconds: int = 1
    budget_accounting_rule: str = "proposal_count_then_candidate_id_tie_break"
    parallel_max_worker_count: int = 1
    parallel_candidate_batch_size: int = 1
    parallel_aggregation_rule: str = "deterministic_candidate_id_order"
    seed_derivation_rule: str = "deterministic_scope_hash"
    seed_scopes: tuple[str, ...] = ("search", "candidate_generation", "tie_break")
    frontier_id: str = "retained_scope_search_frontier_v1"
    frontier_axes: tuple[str, ...] = (
        "structure_code_bits",
        "description_gain_bits",
        "inner_primary_score",
    )
    predictive_axis_rule: str = "inner_primary_score_allowed_only_when_fold_local"
    forbidden_frontier_axes: tuple[str, ...] = (
        "holdout_results",
        "outer_fold_results",
        "null_results",
        "robustness_results",
    )
    search_time_predictive_policy: str = "fold_local_only"
    fit_strategy: Mapping[str, Any] | None = None
    quantization_policy: Mapping[str, Any] | None = None
    reference_policy: Mapping[str, Any] | None = None
    data_code_family: str | None = None
    owner_id: str = "module.search-planning-v1"
    scope_id: str = _DEFAULT_SCOPE_ID
    seasonal_period: int | None = None
    minimum_description_gain_bits: float | None = None

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (
            self.canonicalization_policy_ref,
            self.codelength_policy_ref,
            self.reference_description_policy_ref,
            self.observation_model_ref,
        )

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "search_plan_id": self.search_plan_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "primitive_families": list(self.primitive_families),
            "composition_operators": list(self.composition_operators),
            "predictive_mode": self.predictive_mode,
            "forecast_object_type": self.forecast_object_type,
            "search_class": self.search_class,
            "fold_local_search_required": self.fold_local_search_required,
            "canonicalization_policy_ref": (self.canonicalization_policy_ref.as_dict()),
            "codelength_policy_ref": self.codelength_policy_ref.as_dict(),
            "reference_description_policy_ref": (
                self.reference_description_policy_ref.as_dict()
            ),
            "observation_model_ref": self.observation_model_ref.as_dict(),
            "max_candidate_count": self.max_candidate_count,
            "random_seed": self.random_seed,
            "search_budget": {
                "proposal_limit": self.proposal_limit,
                "frontier_width": self.frontier_width,
                "shortlist_limit": self.shortlist_limit,
                "wall_clock_budget_seconds": self.wall_clock_budget_seconds,
                "budget_accounting_rule": self.budget_accounting_rule,
            },
            "parallel_budget": {
                "max_worker_count": self.parallel_max_worker_count,
                "candidate_batch_size": self.parallel_candidate_batch_size,
                "aggregation_rule": self.parallel_aggregation_rule,
            },
            "seed_policy": {
                "root_seed": self.random_seed,
                "seed_derivation_rule": self.seed_derivation_rule,
                "seed_scopes": list(self.seed_scopes),
            },
            "frontier_policy": {
                "frontier_id": self.frontier_id,
                "axes": list(self.frontier_axes),
                "predictive_axis_rule": self.predictive_axis_rule,
                "forbidden_axes": list(self.forbidden_frontier_axes),
            },
            "search_time_predictive_policy": self.search_time_predictive_policy,
        }
        if self.candidate_family_ids:
            body["candidate_family_ids"] = list(self.candidate_family_ids)
        if self.seasonal_period is not None:
            body["seasonal_period"] = self.seasonal_period
        if self.minimum_description_gain_bits is not None:
            body["minimum_description_gain_bits"] = self.minimum_description_gain_bits
        if self.fit_strategy is not None:
            body["fit_strategy"] = dict(self.fit_strategy)
        if self.quantization_policy is not None:
            body["quantization_policy"] = dict(self.quantization_policy)
        if self.reference_policy is not None:
            body["reference_policy"] = dict(self.reference_policy)
        if self.data_code_family is not None:
            body["data_code_family"] = self.data_code_family
        return body


@dataclass(frozen=True, kw_only=True)
class SearchLedgerManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "search_ledger_manifest@1.0.0"
    module_id: ClassVar[str] = "search_planning"

    search_ledger_id: str
    selected_candidate_id: str
    selection_rule: str
    candidates: tuple[Mapping[str, Any], ...]
    budget_accounting: Mapping[str, Any]
    owner_id: str = "module.search-planning-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    def body(self) -> dict[str, Any]:
        return {
            "search_ledger_id": self.search_ledger_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "candidate_count": len(self.candidates),
            "selected_candidate_id": self.selected_candidate_id,
            "selection_rule": self.selection_rule,
            "candidates": [dict(candidate) for candidate in self.candidates],
            "budget_accounting": dict(self.budget_accounting),
        }


@dataclass(frozen=True, kw_only=True)
class FrontierManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "frontier_manifest@1.0.0"
    module_id: ClassVar[str] = "search_planning"

    frontier_id: str
    frontier_axes: tuple[str, ...]
    frontier_candidate_ids: tuple[str, ...]
    frontier_records: tuple[Mapping[str, Any], ...]
    owner_id: str = "module.search-planning-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    def body(self) -> dict[str, Any]:
        return {
            "frontier_id": self.frontier_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "frontier_axes": list(self.frontier_axes),
            "frontier_cardinality": len(self.frontier_candidate_ids),
            "frontier_candidate_ids": list(self.frontier_candidate_ids),
            "frontier_records": [dict(record) for record in self.frontier_records],
        }


@dataclass(frozen=True, kw_only=True)
class RejectedDiagnosticsManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "rejected_diagnostics_manifest@1.0.0"
    module_id: ClassVar[str] = "search_planning"

    rejected_diagnostics_id: str
    rejected_records: tuple[Mapping[str, Any], ...]
    owner_id: str = "module.search-planning-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    def body(self) -> dict[str, Any]:
        return {
            "rejected_diagnostics_id": self.rejected_diagnostics_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "rejected_records": [dict(record) for record in self.rejected_records],
        }


@dataclass(frozen=True, kw_only=True)
class FrozenShortlistManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "frozen_shortlist_manifest@1.0.0"
    module_id: ClassVar[str] = "search_planning"

    frozen_shortlist_id: str
    search_plan_ref: TypedRef
    candidate_refs: tuple[TypedRef, ...]
    selection_rule: str = "single_lowest_total_bits_admissible_candidate"
    tie_break_rule: str = "lexicographic_candidate_id"
    shortlist_cardinality: int = 1
    owner_id: str = "module.search-planning-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (self.search_plan_ref, *self.candidate_refs)

    def body(self) -> dict[str, Any]:
        return {
            "frozen_shortlist_id": self.frozen_shortlist_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "search_plan_ref": self.search_plan_ref.as_dict(),
            "selection_rule": self.selection_rule,
            "tie_break_rule": self.tie_break_rule,
            "shortlist_cardinality": self.shortlist_cardinality,
            "candidate_refs": [ref.as_dict() for ref in self.candidate_refs],
        }


@dataclass(frozen=True, kw_only=True)
class FreezeEventManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "freeze_event_manifest@1.0.0"
    module_id: ClassVar[str] = "search_planning"

    freeze_event_id: str
    frozen_candidate_ref: TypedRef
    frozen_shortlist_ref: TypedRef
    confirmatory_baseline_id: str
    freeze_stage: str = "global_pair_freeze_pre_holdout"
    baseline_selection_rule: str = "ex_ante_fixed"
    holdout_materialized_before_freeze: bool = False
    post_freeze_candidate_mutation_count: int = 0
    owner_id: str = "module.search-planning-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (self.frozen_candidate_ref, self.frozen_shortlist_ref)

    def body(self) -> dict[str, Any]:
        return {
            "freeze_event_id": self.freeze_event_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "freeze_stage": self.freeze_stage,
            "frozen_candidate_ref": self.frozen_candidate_ref.as_dict(),
            "frozen_shortlist_ref": self.frozen_shortlist_ref.as_dict(),
            "confirmatory_baseline_id": self.confirmatory_baseline_id,
            "baseline_selection_rule": self.baseline_selection_rule,
            "holdout_materialized_before_freeze": (
                self.holdout_materialized_before_freeze
            ),
            "post_freeze_candidate_mutation_count": (
                self.post_freeze_candidate_mutation_count
            ),
        }


@dataclass(frozen=True, kw_only=True)
class CandidateSpecManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "candidate_spec@1.0.0"
    module_id: ClassVar[str] = "candidate_fitting"

    candidate_spec_id: str
    family_id: str
    parameter_summary: Mapping[str, float | int]
    selection_floor_bits: float
    candidate_id: str | None = None
    fit_window_id: str | None = None
    optimizer_backend_id: str | None = None
    owner_id: str = "module.candidate-fitting-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "candidate_spec_id": self.candidate_spec_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "family_id": self.family_id,
            "parameter_summary": dict(self.parameter_summary),
            "selection_floor_bits": self.selection_floor_bits,
        }
        if self.candidate_id is not None:
            body["candidate_id"] = self.candidate_id
        if self.fit_window_id is not None:
            body["fit_window_id"] = self.fit_window_id
        if self.optimizer_backend_id is not None:
            body["optimizer_backend_id"] = self.optimizer_backend_id
        return body


@dataclass(frozen=True, kw_only=True)
class CandidateStateManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "candidate_state_manifest@1.0.0"
    module_id: ClassVar[str] = "candidate_fitting"

    candidate_state_id: str
    candidate_id: str
    fit_window_id: str
    stage_id: str
    lifecycle_state: str
    optimizer_backend_id: str
    optimizer_objective_id: str
    optimizer_seed: str
    converged: bool
    iteration_count: int
    final_loss: float
    training_row_count: int
    initial_state: Mapping[str, Any]
    final_state: Mapping[str, Any]
    state_transitions: tuple[Mapping[str, Any], ...]
    candidate_spec_ref: TypedRef | None = None
    search_plan_ref: TypedRef | None = None
    owner_id: str = "module.candidate-fitting-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        refs: list[TypedRef] = []
        if self.candidate_spec_ref is not None:
            refs.append(self.candidate_spec_ref)
        if self.search_plan_ref is not None:
            refs.append(self.search_plan_ref)
        return tuple(refs)

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "candidate_state_id": self.candidate_state_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "candidate_id": self.candidate_id,
            "fit_window_id": self.fit_window_id,
            "stage_id": self.stage_id,
            "lifecycle_state": self.lifecycle_state,
            "optimizer_backend_id": self.optimizer_backend_id,
            "optimizer_objective_id": self.optimizer_objective_id,
            "optimizer_seed": self.optimizer_seed,
            "converged": self.converged,
            "iteration_count": self.iteration_count,
            "final_loss": self.final_loss,
            "training_row_count": self.training_row_count,
            "initial_state": dict(self.initial_state),
            "final_state": dict(self.final_state),
            "state_transition_count": len(self.state_transitions),
            "state_transitions": [
                dict(transition) for transition in self.state_transitions
            ],
        }
        if self.candidate_spec_ref is not None:
            body["candidate_spec_ref"] = self.candidate_spec_ref.as_dict()
        if self.search_plan_ref is not None:
            body["search_plan_ref"] = self.search_plan_ref.as_dict()
        return body


@dataclass(frozen=True, kw_only=True)
class ResidualHistoryManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "residual_history_manifest@1.0.0"
    module_id: ClassVar[str] = "candidate_fitting"

    residual_history_id: str
    candidate_id: str
    fit_window_id: str
    residual_rows: tuple[Mapping[str, Any], ...]
    residual_history_digest: str
    residual_basis: str
    construction_policy: str
    replay_identity: str
    source_row_set_digest: str | None = None
    owner_id: str = "module.candidate-fitting-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    def body(self) -> dict[str, Any]:
        rows = tuple(dict(row) for row in self.residual_rows)
        for index, row in enumerate(rows):
            if "split_role" not in row:
                raise ContractValidationError(
                    code="missing_split_role_metadata",
                    message="residual history rows require split_role metadata",
                    field_path=f"residual_rows[{index}].split_role",
                )
            for field_name in ("origin_available_at", "target_available_at"):
                if field_name not in row:
                    raise ContractValidationError(
                        code="missing_origin_or_target_availability",
                        message=(
                            "residual history rows require origin and target "
                            "availability metadata"
                        ),
                        field_path=f"residual_rows[{index}].{field_name}",
                    )
        payload: dict[str, Any] = {
            "residual_history_id": self.residual_history_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "candidate_id": self.candidate_id,
            "fit_window_id": self.fit_window_id,
            "residual_rows": rows,
            "residual_row_count": len(rows),
            "residual_history_digest": self.residual_history_digest,
            "residual_basis": self.residual_basis,
            "construction_policy": self.construction_policy,
            "replay_identity": self.replay_identity,
        }
        if self.source_row_set_digest is not None:
            payload["source_row_set_digest"] = self.source_row_set_digest
        return payload

    @classmethod
    def from_manifest(cls, manifest: ManifestEnvelope) -> "ResidualHistoryManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            residual_history_id=str(manifest.body["residual_history_id"]),
            candidate_id=str(manifest.body["candidate_id"]),
            fit_window_id=str(manifest.body["fit_window_id"]),
            residual_rows=tuple(
                dict(item) for item in manifest.body.get("residual_rows", [])
            ),
            residual_history_digest=str(manifest.body["residual_history_digest"]),
            residual_basis=str(manifest.body["residual_basis"]),
            construction_policy=str(manifest.body["construction_policy"]),
            replay_identity=str(manifest.body["replay_identity"]),
            source_row_set_digest=(
                str(manifest.body["source_row_set_digest"])
                if manifest.body.get("source_row_set_digest") is not None
                else None
            ),
            owner_id=str(
                manifest.body.get("owner_id", "module.candidate-fitting-v1")
            ),
            scope_id=str(manifest.body.get("scope_id", _DEFAULT_SCOPE_ID)),
        )


@dataclass(frozen=True, kw_only=True)
class StochasticModelManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "stochastic_model_manifest@1.0.0"
    module_id: ClassVar[str] = "probabilistic_evaluation"

    stochastic_model_id: str
    candidate_id: str
    observation_family: str
    residual_family: str
    support_kind: str
    horizon_scale_law: str
    fitted_parameters: Mapping[str, float | int]
    residual_count: int
    min_count_policy: Mapping[str, Any]
    evidence_status: str
    heuristic_gaussian_support: bool
    replay_identity: str
    residual_history_ref: TypedRef | None = None
    residual_location: float | None = None
    residual_scale: float | None = None
    owner_id: str = "module.probabilistic-evaluation-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        if self.residual_history_ref is None:
            return ()
        return (self.residual_history_ref,)

    def body(self) -> dict[str, Any]:
        if self.evidence_status not in {"production", "compatibility"}:
            raise ContractValidationError(
                code="invalid_stochastic_evidence_status",
                message="stochastic evidence status must be production or compatibility",
                field_path="evidence_status",
                details={"evidence_status": self.evidence_status},
            )
        if self.evidence_status == "production" and self.residual_history_ref is None:
            raise ContractValidationError(
                code="missing_residual_history_evidence",
                message=(
                    "production stochastic evidence requires a residual history ref"
                ),
                field_path="residual_history_ref",
            )
        if self.evidence_status == "production" and self.heuristic_gaussian_support:
            raise ContractValidationError(
                code="heuristic_gaussian_support_not_production",
                message=(
                    "heuristic Gaussian support is compatibility evidence only"
                ),
                field_path="heuristic_gaussian_support",
            )
        minimum_count = int(self.min_count_policy.get("minimum_residual_count", 0))
        if self.evidence_status == "production" and self.residual_count < minimum_count:
            raise ContractValidationError(
                code="insufficient_stochastic_training_support",
                message=(
                    "production stochastic evidence requires enough residual rows"
                ),
                field_path="residual_count",
                details={
                    "residual_count": self.residual_count,
                    "minimum_residual_count": minimum_count,
                },
            )
        fitted_parameters = {
            str(key): float(value) for key, value in self.fitted_parameters.items()
        }
        residual_location = (
            float(self.residual_location)
            if self.residual_location is not None
            else float(fitted_parameters.get("location", 0.0))
        )
        residual_scale = (
            float(self.residual_scale)
            if self.residual_scale is not None
            else float(fitted_parameters.get("scale", 0.0))
        )
        payload: dict[str, Any] = {
            "stochastic_model_id": self.stochastic_model_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "candidate_id": self.candidate_id,
            "observation_family": self.observation_family,
            "residual_family": self.residual_family,
            "support_kind": self.support_kind,
            "residual_location": residual_location,
            "residual_scale": residual_scale,
            "residual_parameter_summary": dict(fitted_parameters),
            "fitted_parameters": dict(fitted_parameters),
            "residual_count": int(self.residual_count),
            "min_count_policy": dict(self.min_count_policy),
            "horizon_scale_law": self.horizon_scale_law,
            "evidence_status": self.evidence_status,
            "production_evidence": self.evidence_status == "production",
            "heuristic_gaussian_support": self.heuristic_gaussian_support,
            "replay_identity": self.replay_identity,
        }
        if self.residual_history_ref is not None:
            payload["residual_history_ref"] = self.residual_history_ref.as_dict()
        return payload

    @classmethod
    def from_manifest(cls, manifest: ManifestEnvelope) -> "StochasticModelManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            stochastic_model_id=str(manifest.body["stochastic_model_id"]),
            candidate_id=str(manifest.body["candidate_id"]),
            residual_history_ref=(
                _typed_ref_from_payload(
                    manifest.body["residual_history_ref"],
                    field_path="body.residual_history_ref",
                )
                if manifest.body.get("residual_history_ref") is not None
                else None
            ),
            observation_family=str(manifest.body["observation_family"]),
            residual_family=str(
                manifest.body.get(
                    "residual_family",
                    manifest.body["observation_family"],
                )
            ),
            support_kind=str(manifest.body.get("support_kind", "all_real")),
            horizon_scale_law=str(manifest.body["horizon_scale_law"]),
            fitted_parameters=dict(
                manifest.body.get(
                    "fitted_parameters",
                    manifest.body.get("residual_parameter_summary", {}),
                )
            ),
            residual_count=int(manifest.body.get("residual_count", 0)),
            min_count_policy=dict(manifest.body.get("min_count_policy", {})),
            evidence_status=str(
                manifest.body.get(
                    "evidence_status",
                    (
                        "production"
                        if manifest.body.get("production_evidence") is True
                        else "compatibility"
                    ),
                )
            ),
            heuristic_gaussian_support=bool(
                manifest.body.get("heuristic_gaussian_support", False)
            ),
            replay_identity=str(manifest.body["replay_identity"]),
            residual_location=(
                float(manifest.body["residual_location"])
                if manifest.body.get("residual_location") is not None
                else None
            ),
            residual_scale=(
                float(manifest.body["residual_scale"])
                if manifest.body.get("residual_scale") is not None
                else None
            ),
            owner_id=str(
                manifest.body.get(
                    "owner_id",
                    "module.probabilistic-evaluation-v1",
                )
            ),
            scope_id=str(manifest.body.get("scope_id", _DEFAULT_SCOPE_ID)),
        )


@dataclass(frozen=True, kw_only=True)
class PredictionArtifactManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "prediction_artifact_manifest@1.1.0"
    module_id: ClassVar[str] = "evaluation"

    prediction_artifact_id: str
    candidate_id: str
    stage_id: str
    fit_window_id: str
    test_window_id: str
    model_freeze_status: str
    refit_rule_applied: str
    score_policy_ref: TypedRef
    rows: tuple[PredictionArtifactRow, ...]
    owner_prompt_id: str = "prompt.predictive-validation-v1"
    scope_id: str = _DEFAULT_SCOPE_ID
    forecast_object_type: str = "point"
    outer_fold_id: str | None = None
    search_mutation_after_stage_freeze: bool = False
    score_law_id: str | None = None
    horizon_weights: tuple[Mapping[str, Any], ...] = ()
    entity_panel: tuple[str, ...] = ()
    entity_weights: tuple[Mapping[str, Any], ...] = ()
    scored_origin_panel: tuple[Mapping[str, Any], ...] = ()
    scored_origin_set_id: str | None = None
    comparison_key: Mapping[str, Any] | None = None
    missing_scored_origins: tuple[Mapping[str, Any], ...] = ()
    timeguard_checks: tuple[Mapping[str, Any], ...] = ()
    composition_graph: Mapping[str, Any] | None = None
    composition_runtime_evidence: Mapping[str, Any] | None = None
    residual_history_refs: tuple[TypedRef, ...] = ()
    stochastic_model_refs: tuple[TypedRef, ...] = ()
    stochastic_support_status: str | None = None
    stochastic_support_reason_codes: tuple[str, ...] = ()
    effective_probabilistic_config: Mapping[str, Any] | None = None

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (
            self.score_policy_ref,
            *self.residual_history_refs,
            *self.stochastic_model_refs,
        )

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prediction_artifact_id": self.prediction_artifact_id,
            "owner_prompt_id": self.owner_prompt_id,
            "scope_id": self.scope_id,
            "candidate_id": self.candidate_id,
            "forecast_object_type": self.forecast_object_type,
            "stage_id": self.stage_id,
            "fit_window_id": self.fit_window_id,
            "test_window_id": self.test_window_id,
            "model_freeze_status": self.model_freeze_status,
            "refit_rule_applied": self.refit_rule_applied,
            "search_mutation_after_stage_freeze": (
                self.search_mutation_after_stage_freeze
            ),
            "score_policy_ref": self.score_policy_ref.as_dict(),
            "rows": [row.as_dict() for row in self.rows],
        }
        if self.outer_fold_id is not None:
            body["outer_fold_id"] = self.outer_fold_id
        if self.score_law_id is not None:
            body["score_law_id"] = self.score_law_id
        if self.horizon_weights:
            body["horizon_weights"] = [dict(weight) for weight in self.horizon_weights]
        if self.entity_panel:
            body["entity_panel"] = list(self.entity_panel)
        if self.entity_weights:
            body["entity_weights"] = [dict(weight) for weight in self.entity_weights]
        if self.scored_origin_panel:
            body["scored_origin_panel"] = [
                dict(origin) for origin in self.scored_origin_panel
            ]
        if self.scored_origin_set_id is not None:
            body["scored_origin_set_id"] = self.scored_origin_set_id
        if self.comparison_key is not None:
            body["comparison_key"] = dict(self.comparison_key)
        if self.composition_graph is not None:
            body["composition_graph"] = dict(self.composition_graph)
        if self.composition_runtime_evidence is not None:
            body["composition_runtime_evidence"] = dict(
                self.composition_runtime_evidence
            )
        if self.effective_probabilistic_config is not None:
            body["effective_probabilistic_config"] = dict(
                self.effective_probabilistic_config
            )
        if self.stochastic_support_status is not None:
            body["stochastic_support_status"] = self.stochastic_support_status
            body["stochastic_support_reason_codes"] = list(
                self.stochastic_support_reason_codes
            )
            body["residual_history_refs"] = [
                ref.as_dict() for ref in self.residual_history_refs
            ]
            body["stochastic_model_refs"] = [
                ref.as_dict() for ref in self.stochastic_model_refs
            ]
        if self.missing_scored_origins:
            body["missing_scored_origins"] = [
                dict(item) for item in self.missing_scored_origins
            ]
        else:
            body["missing_scored_origins"] = []
        if self.timeguard_checks:
            body["timeguard_checks"] = [dict(check) for check in self.timeguard_checks]
        else:
            body["timeguard_checks"] = []
        return body


@dataclass(frozen=True, kw_only=True)
class PointScoreResultManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "point_score_result_manifest@1.0.0"
    module_id: ClassVar[str] = "scoring"

    score_result_id: str
    score_policy_ref: TypedRef
    prediction_artifact_ref: TypedRef
    per_horizon: tuple[PerHorizonScore, ...]
    aggregated_primary_score: float
    comparison_status: str
    failure_reason_code: str | None
    owner_prompt_id: str = "prompt.scoring-calibration-v1"
    scope_id: str = _DEFAULT_SCOPE_ID
    forecast_object_type: str = "point"

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (self.score_policy_ref, self.prediction_artifact_ref)

    def body(self) -> dict[str, Any]:
        return {
            "score_result_id": self.score_result_id,
            "owner_prompt_id": self.owner_prompt_id,
            "scope_id": self.scope_id,
            "score_policy_ref": self.score_policy_ref.as_dict(),
            "prediction_artifact_ref": self.prediction_artifact_ref.as_dict(),
            "forecast_object_type": self.forecast_object_type,
            "per_horizon": [item.as_dict() for item in self.per_horizon],
            "aggregated_primary_score": self.aggregated_primary_score,
            "comparison_status": self.comparison_status,
            "failure_reason_code": self.failure_reason_code,
        }


@dataclass(frozen=True, kw_only=True)
class ProbabilisticScoreResultManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "probabilistic_score_result_manifest@1.0.0"
    module_id: ClassVar[str] = "scoring"

    score_result_id: str
    score_policy_ref: TypedRef
    prediction_artifact_ref: TypedRef
    per_horizon: tuple[PerHorizonPrimaryScore, ...]
    aggregated_primary_score: float
    comparison_status: str
    failure_reason_code: str | None
    forecast_object_type: str
    owner_prompt_id: str = "prompt.scoring-calibration-v1"
    scope_id: str = _DEFAULT_SCOPE_ID
    effective_probabilistic_config: Mapping[str, Any] | None = None

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (self.score_policy_ref, self.prediction_artifact_ref)

    def body(self) -> dict[str, Any]:
        body = {
            "score_result_id": self.score_result_id,
            "owner_prompt_id": self.owner_prompt_id,
            "scope_id": self.scope_id,
            "score_policy_ref": self.score_policy_ref.as_dict(),
            "prediction_artifact_ref": self.prediction_artifact_ref.as_dict(),
            "forecast_object_type": self.forecast_object_type,
            "per_horizon": [item.as_dict() for item in self.per_horizon],
            "aggregated_primary_score": self.aggregated_primary_score,
            "comparison_status": self.comparison_status,
            "failure_reason_code": self.failure_reason_code,
        }
        if self.effective_probabilistic_config is not None:
            body["effective_probabilistic_config"] = dict(
                self.effective_probabilistic_config
            )
        return body


@dataclass(frozen=True, kw_only=True)
class CalibrationContractManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "calibration_contract_manifest@1.0.0"
    module_id: ClassVar[str] = "scoring"

    calibration_contract_id: str
    forecast_object_type: str
    calibration_mode: str
    required_diagnostic_ids: tuple[str, ...]
    optional_diagnostic_ids: tuple[str, ...]
    pass_rule: str
    gate_effect: str
    thresholds: Mapping[str, float] = field(default_factory=dict)
    reliability_bins: Mapping[str, Any] = field(default_factory=dict)
    pit_config: Mapping[str, Any] = field(default_factory=dict)
    interval_levels: tuple[float, ...] = ()
    quantile_levels: tuple[float, ...] = ()
    calibration_lane: str = "evaluation_only"
    owner_prompt_id: str = "prompt.scoring-calibration-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "calibration_contract_id": self.calibration_contract_id,
            "owner_prompt_id": self.owner_prompt_id,
            "scope_id": self.scope_id,
            "forecast_object_type": self.forecast_object_type,
            "calibration_mode": self.calibration_mode,
            "required_diagnostic_ids": list(self.required_diagnostic_ids),
            "optional_diagnostic_ids": list(self.optional_diagnostic_ids),
            "pass_rule": self.pass_rule,
            "gate_effect": self.gate_effect,
        }
        if self.thresholds:
            body["thresholds"] = {
                key: float(value) for key, value in self.thresholds.items()
            }
        if self.reliability_bins:
            body["reliability_bins"] = dict(self.reliability_bins)
        if self.pit_config:
            body["pit"] = dict(self.pit_config)
        if self.interval_levels:
            body["interval_levels"] = [float(level) for level in self.interval_levels]
        if self.quantile_levels:
            body["quantile_levels"] = [float(level) for level in self.quantile_levels]
        body["calibration_lane"] = self.calibration_lane
        return body


@dataclass(frozen=True, kw_only=True)
class CalibrationResultManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "calibration_result_manifest@1.0.0"
    module_id: ClassVar[str] = "scoring"

    calibration_result_id: str
    calibration_contract_ref: TypedRef
    prediction_artifact_ref: TypedRef
    forecast_object_type: str
    status: str
    gate_effect: str
    diagnostics: tuple[Mapping[str, Any], ...]
    failure_reason_code: str | None = None
    pass_value: bool | None = None
    effective_calibration_config: Mapping[str, Any] | None = None
    calibration_identity: Mapping[str, Any] | None = None
    lane_status: str | None = None
    owner_prompt_id: str = "prompt.scoring-calibration-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (self.calibration_contract_ref, self.prediction_artifact_ref)

    def body(self) -> dict[str, Any]:
        body = {
            "calibration_result_id": self.calibration_result_id,
            "owner_prompt_id": self.owner_prompt_id,
            "scope_id": self.scope_id,
            "calibration_contract_ref": self.calibration_contract_ref.as_dict(),
            "prediction_artifact_ref": self.prediction_artifact_ref.as_dict(),
            "forecast_object_type": self.forecast_object_type,
            "status": self.status,
            "failure_reason_code": self.failure_reason_code,
            "pass": self.pass_value,
            "gate_effect": self.gate_effect,
            "diagnostics": [dict(item) for item in self.diagnostics],
        }
        if self.effective_calibration_config is not None:
            body["effective_calibration_config"] = dict(
                self.effective_calibration_config
            )
        if self.calibration_identity is not None:
            body["calibration_identity"] = dict(self.calibration_identity)
        if self.lane_status is not None:
            body["lane_status"] = self.lane_status
        return body


@dataclass(frozen=True, kw_only=True)
class NullResultManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "null_result_manifest@1.0.0"
    module_id: ClassVar[str] = "robustness"

    null_result_id: str
    status: str
    failure_reason_code: str | None
    observed_statistic: float | None
    surrogate_statistics: tuple[float, ...]
    monte_carlo_p_value: float | None
    max_p_value: float | None
    resample_count: int
    statistic_id: str = "description_gain_bits"
    statistic_orientation: str = "larger_is_more_structure"
    candidate_id: str | None = None
    null_protocol_ref: TypedRef | None = None
    owner_prompt_id: str = "prompt.nulls-stability-leakage-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        if self.null_protocol_ref is None:
            return ()
        return (self.null_protocol_ref,)

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "null_result_id": self.null_result_id,
            "owner_prompt_id": self.owner_prompt_id,
            "scope_id": self.scope_id,
            "status": self.status,
            "failure_reason_code": self.failure_reason_code,
            "observed_statistic": self.observed_statistic,
            "surrogate_statistics": list(self.surrogate_statistics),
            "monte_carlo_p_value": self.monte_carlo_p_value,
            "max_p_value": self.max_p_value,
            "resample_count": self.resample_count,
            "statistic_id": self.statistic_id,
            "statistic_orientation": self.statistic_orientation,
        }
        if self.candidate_id is not None:
            body["candidate_id"] = self.candidate_id
        if self.null_protocol_ref is not None:
            body["null_protocol_ref"] = self.null_protocol_ref.as_dict()
        return body

    @classmethod
    def from_manifest(cls, manifest: ManifestEnvelope) -> "NullResultManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            null_result_id=str(manifest.body["null_result_id"]),
            status=str(manifest.body["status"]),
            failure_reason_code=(
                str(manifest.body["failure_reason_code"])
                if manifest.body.get("failure_reason_code") is not None
                else None
            ),
            observed_statistic=(
                float(manifest.body["observed_statistic"])
                if manifest.body.get("observed_statistic") is not None
                else None
            ),
            surrogate_statistics=tuple(
                float(item) for item in manifest.body.get("surrogate_statistics", [])
            ),
            monte_carlo_p_value=(
                float(manifest.body["monte_carlo_p_value"])
                if manifest.body.get("monte_carlo_p_value") is not None
                else None
            ),
            max_p_value=(
                float(manifest.body["max_p_value"])
                if manifest.body.get("max_p_value") is not None
                else None
            ),
            resample_count=int(manifest.body["resample_count"]),
            statistic_id=str(
                manifest.body.get("statistic_id", "description_gain_bits")
            ),
            statistic_orientation=str(
                manifest.body.get(
                    "statistic_orientation",
                    "larger_is_more_structure",
                )
            ),
            candidate_id=(
                str(manifest.body["candidate_id"])
                if manifest.body.get("candidate_id") is not None
                else None
            ),
            null_protocol_ref=(
                _typed_ref_from_payload(
                    manifest.body["null_protocol_ref"],
                    field_path="body.null_protocol_ref",
                )
                if manifest.body.get("null_protocol_ref") is not None
                else None
            ),
            owner_prompt_id=str(manifest.body["owner_prompt_id"]),
            scope_id=str(manifest.body["scope_id"]),
        )


@dataclass(frozen=True, kw_only=True)
class PerturbationFamilyResultManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "perturbation_family_result_manifest@1.0.0"
    module_id: ClassVar[str] = "robustness"

    perturbation_family_result_id: str
    perturbation_protocol_ref: TypedRef
    family_id: str
    status: str
    valid_run_count: int
    invalid_run_count: int
    metric_results: tuple[Mapping[str, Any], ...]
    perturbation_runs: tuple[Mapping[str, Any], ...] = ()
    candidate_id: str | None = None
    owner_prompt_id: str = "prompt.nulls-stability-leakage-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        refs = [self.perturbation_protocol_ref]
        for index, item in enumerate(self.metric_results):
            metric_ref = item.get("metric_ref") if isinstance(item, Mapping) else None
            if isinstance(metric_ref, Mapping):
                refs.append(
                    _typed_ref_from_payload(
                        metric_ref,
                        field_path=f"body.metric_results[{index}].metric_ref",
                    )
                )
        return tuple(refs)

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "perturbation_family_result_id": self.perturbation_family_result_id,
            "owner_prompt_id": self.owner_prompt_id,
            "scope_id": self.scope_id,
            "perturbation_protocol_ref": self.perturbation_protocol_ref.as_dict(),
            "family_id": self.family_id,
            "status": self.status,
            "valid_run_count": self.valid_run_count,
            "invalid_run_count": self.invalid_run_count,
            "metric_results": [dict(item) for item in self.metric_results],
            "perturbation_runs": [dict(item) for item in self.perturbation_runs],
        }
        if self.candidate_id is not None:
            body["candidate_id"] = self.candidate_id
        return body

    @classmethod
    def from_manifest(
        cls, manifest: ManifestEnvelope
    ) -> "PerturbationFamilyResultManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            perturbation_family_result_id=str(
                manifest.body["perturbation_family_result_id"]
            ),
            perturbation_protocol_ref=_typed_ref_from_payload(
                manifest.body["perturbation_protocol_ref"],
                field_path="body.perturbation_protocol_ref",
            ),
            family_id=str(manifest.body["family_id"]),
            status=str(manifest.body["status"]),
            valid_run_count=int(manifest.body["valid_run_count"]),
            invalid_run_count=int(manifest.body["invalid_run_count"]),
            metric_results=tuple(
                dict(item) for item in manifest.body.get("metric_results", [])
            ),
            perturbation_runs=tuple(
                dict(item) for item in manifest.body.get("perturbation_runs", [])
            ),
            candidate_id=(
                str(manifest.body["candidate_id"])
                if manifest.body.get("candidate_id") is not None
                else None
            ),
            owner_prompt_id=str(manifest.body["owner_prompt_id"]),
            scope_id=str(manifest.body["scope_id"]),
        )


@dataclass(frozen=True, kw_only=True)
class SensitivityAnalysisManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "sensitivity_analysis_manifest@1.0.0"
    module_id: ClassVar[str] = "robustness"

    sensitivity_analysis_id: str
    perturbation_family_result_ref: TypedRef
    family_id: str
    candidate_id: str | None = None
    perturbation_id: str | None = None
    canonical_form_matches: bool | None = None
    description_gain_bits: float | None = None
    outer_candidate_score: float | None = None
    outer_baseline_score: float | None = None
    failure_reason_code: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    owner_prompt_id: str = "prompt.nulls-stability-leakage-v1"
    scope_id: str = _DEFAULT_SCOPE_ID

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (self.perturbation_family_result_ref,)

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "sensitivity_analysis_id": self.sensitivity_analysis_id,
            "analysis_id": self.sensitivity_analysis_id,
            "owner_prompt_id": self.owner_prompt_id,
            "scope_id": self.scope_id,
            "perturbation_family_result_ref": (
                self.perturbation_family_result_ref.as_dict()
            ),
            "family_id": self.family_id,
            "metadata": dict(self.metadata),
        }
        if self.candidate_id is not None:
            body["candidate_id"] = self.candidate_id
        if self.perturbation_id is not None:
            body["perturbation_id"] = self.perturbation_id
        if self.canonical_form_matches is not None:
            body["canonical_form_matches"] = self.canonical_form_matches
        if self.description_gain_bits is not None:
            body["description_gain_bits"] = self.description_gain_bits
        if self.outer_candidate_score is not None:
            body["outer_candidate_score"] = self.outer_candidate_score
        if self.outer_baseline_score is not None:
            body["outer_baseline_score"] = self.outer_baseline_score
        if self.failure_reason_code is not None:
            body["failure_reason_code"] = self.failure_reason_code
        return body

    @classmethod
    def from_manifest(cls, manifest: ManifestEnvelope) -> "SensitivityAnalysisManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            sensitivity_analysis_id=str(manifest.body["sensitivity_analysis_id"]),
            perturbation_family_result_ref=_typed_ref_from_payload(
                manifest.body["perturbation_family_result_ref"],
                field_path="body.perturbation_family_result_ref",
            ),
            family_id=str(manifest.body["family_id"]),
            candidate_id=(
                str(manifest.body["candidate_id"])
                if manifest.body.get("candidate_id") is not None
                else None
            ),
            perturbation_id=(
                str(manifest.body["perturbation_id"])
                if manifest.body.get("perturbation_id") is not None
                else None
            ),
            canonical_form_matches=(
                bool(manifest.body["canonical_form_matches"])
                if manifest.body.get("canonical_form_matches") is not None
                else None
            ),
            description_gain_bits=(
                float(manifest.body["description_gain_bits"])
                if manifest.body.get("description_gain_bits") is not None
                else None
            ),
            outer_candidate_score=(
                float(manifest.body["outer_candidate_score"])
                if manifest.body.get("outer_candidate_score") is not None
                else None
            ),
            outer_baseline_score=(
                float(manifest.body["outer_baseline_score"])
                if manifest.body.get("outer_baseline_score") is not None
                else None
            ),
            failure_reason_code=(
                str(manifest.body["failure_reason_code"])
                if manifest.body.get("failure_reason_code") is not None
                else None
            ),
            metadata=_json_ready_mapping(manifest.body.get("metadata", {})),
            owner_prompt_id=str(manifest.body["owner_prompt_id"]),
            scope_id=str(manifest.body["scope_id"]),
        )


@dataclass(frozen=True, kw_only=True)
class RobustnessReportManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "robustness_report_manifest@1.1.0"
    module_id: ClassVar[str] = "robustness"

    robustness_report_id: str
    perturbation_protocol_ref: TypedRef
    leakage_canary_result_refs: tuple[TypedRef, ...]
    owner_prompt_id: str = "prompt.nulls-stability-leakage-v1"
    scope_id: str = _DEFAULT_SCOPE_ID
    null_protocol_ref: TypedRef | None = None
    null_result_ref: TypedRef | None = None
    perturbation_family_result_refs: tuple[TypedRef, ...] = ()
    status: str | None = None
    candidate_id: str | None = None
    null_result: Mapping[str, Any] | None = None
    perturbation_family_results: tuple[Mapping[str, Any], ...] = ()
    aggregate_metric_results: tuple[Mapping[str, Any], ...] = ()
    stability_status: str | None = None
    required_canary_type_coverage: tuple[str, ...] = ()
    final_robustness_status: str | None = None
    candidate_context: Mapping[str, Any] | None = None
    sensitivity_analysis_refs: tuple[TypedRef, ...] = ()
    sensitivity_analyses: tuple[Mapping[str, Any], ...] = ()

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        refs = [
            self.perturbation_protocol_ref,
            *self.perturbation_family_result_refs,
            *self.leakage_canary_result_refs,
            *self.sensitivity_analysis_refs,
        ]
        if self.null_protocol_ref is not None:
            refs.insert(0, self.null_protocol_ref)
        if self.null_result_ref is not None:
            refs.insert(
                1 if self.null_protocol_ref is not None else 0,
                self.null_result_ref,
            )
        return tuple(refs)

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "robustness_report_id": self.robustness_report_id,
            "owner_prompt_id": self.owner_prompt_id,
            "scope_id": self.scope_id,
            "perturbation_protocol_ref": self.perturbation_protocol_ref.as_dict(),
            "leakage_canary_result_refs": [
                ref.as_dict() for ref in self.leakage_canary_result_refs
            ],
        }
        if self.null_protocol_ref is not None:
            body["null_protocol_ref"] = self.null_protocol_ref.as_dict()
        if self.null_result_ref is not None:
            body["null_result_ref"] = self.null_result_ref.as_dict()
        if self.candidate_id is not None:
            body["candidate_id"] = self.candidate_id
        if self.null_result_ref is None and self.null_result is not None:
            body["null_result"] = dict(self.null_result)
        if self.perturbation_family_result_refs:
            body["perturbation_family_result_refs"] = [
                ref.as_dict() for ref in self.perturbation_family_result_refs
            ]
        elif self.perturbation_family_results:
            body["perturbation_family_results"] = [
                dict(item) for item in self.perturbation_family_results
            ]
        if self.aggregate_metric_results:
            body["aggregate_metric_results"] = [
                dict(item) for item in self.aggregate_metric_results
            ]
        if self.stability_status is not None:
            body["stability_status"] = self.stability_status
        if self.required_canary_type_coverage:
            body["required_canary_type_coverage"] = list(
                self.required_canary_type_coverage
            )
        if self.final_robustness_status is not None:
            body["final_robustness_status"] = self.final_robustness_status
            body.setdefault("status", self.final_robustness_status)
        if self.candidate_context is not None:
            body["candidate_context"] = dict(self.candidate_context)
        if self.sensitivity_analysis_refs:
            body["sensitivity_analysis_refs"] = [
                ref.as_dict() for ref in self.sensitivity_analysis_refs
            ]
        elif self.sensitivity_analyses:
            body["sensitivity_analyses"] = [
                dict(item) for item in self.sensitivity_analyses
            ]
        if self.status is not None:
            body["status"] = self.status
        return body

    @classmethod
    def from_manifest(cls, manifest: ManifestEnvelope) -> "RobustnessReportManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            robustness_report_id=str(manifest.body["robustness_report_id"]),
            owner_prompt_id=str(manifest.body["owner_prompt_id"]),
            scope_id=str(manifest.body["scope_id"]),
            null_protocol_ref=(
                _typed_ref_from_payload(
                    manifest.body["null_protocol_ref"],
                    field_path="body.null_protocol_ref",
                )
                if manifest.body.get("null_protocol_ref") is not None
                else None
            ),
            null_result_ref=(
                _typed_ref_from_payload(
                    manifest.body["null_result_ref"],
                    field_path="body.null_result_ref",
                )
                if manifest.body.get("null_result_ref") is not None
                else None
            ),
            perturbation_protocol_ref=_typed_ref_from_payload(
                manifest.body["perturbation_protocol_ref"],
                field_path="body.perturbation_protocol_ref",
            ),
            perturbation_family_result_refs=_typed_ref_list_from_payload(
                manifest.body.get("perturbation_family_result_refs", []),
                field_path="body.perturbation_family_result_refs",
            ),
            leakage_canary_result_refs=_typed_ref_list_from_payload(
                manifest.body.get("leakage_canary_result_refs", []),
                field_path="body.leakage_canary_result_refs",
            ),
            status=(
                str(manifest.body["status"])
                if manifest.body.get("status") is not None
                else None
            ),
            candidate_id=(
                str(manifest.body["candidate_id"])
                if manifest.body.get("candidate_id") is not None
                else None
            ),
            null_result=(
                dict(manifest.body["null_result"])
                if manifest.body.get("null_result") is not None
                else None
            ),
            perturbation_family_results=tuple(
                dict(item)
                for item in manifest.body.get("perturbation_family_results", [])
            ),
            aggregate_metric_results=tuple(
                dict(item) for item in manifest.body.get("aggregate_metric_results", [])
            ),
            stability_status=(
                str(manifest.body["stability_status"])
                if manifest.body.get("stability_status") is not None
                else None
            ),
            required_canary_type_coverage=_string_list_from_payload(
                manifest.body.get("required_canary_type_coverage", []),
                field_path="body.required_canary_type_coverage",
            ),
            final_robustness_status=(
                str(
                    manifest.body.get(
                        "final_robustness_status",
                        manifest.body.get("status"),
                    )
                )
                if (
                    manifest.body.get("final_robustness_status") is not None
                    or manifest.body.get("status") is not None
                )
                else None
            ),
            candidate_context=(
                dict(manifest.body.get("candidate_context", {}))
                if manifest.body.get("candidate_context") is not None
                else None
            ),
            sensitivity_analysis_refs=_typed_ref_list_from_payload(
                manifest.body.get("sensitivity_analysis_refs", []),
                field_path="body.sensitivity_analysis_refs",
            ),
            sensitivity_analyses=tuple(
                dict(item) for item in manifest.body.get("sensitivity_analyses", [])
            ),
        )


@dataclass(frozen=True, kw_only=True)
class SourceDigestManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "source_digest_manifest@1.0.0"
    module_id: ClassVar[str] = "external_evidence_ingestion"

    source_digest_id: str
    source_id: str
    domain_id: str
    acquired_at: str
    digest_sha256: str
    digest_algorithm: str = "sha256"
    evidence_kind: str | None = None

    def body(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_digest_id": self.source_digest_id,
            "source_id": self.source_id,
            "domain_id": self.domain_id,
            "acquired_at": self.acquired_at,
            "digest_algorithm": self.digest_algorithm,
            "digest_sha256": self.digest_sha256,
        }
        if self.evidence_kind is not None:
            payload["evidence_kind"] = self.evidence_kind
        return payload

    @classmethod
    def from_manifest(cls, manifest: ManifestEnvelope) -> "SourceDigestManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            source_digest_id=str(manifest.body["source_digest_id"]),
            source_id=str(manifest.body["source_id"]),
            domain_id=str(manifest.body["domain_id"]),
            acquired_at=str(manifest.body["acquired_at"]),
            digest_sha256=str(manifest.body["digest_sha256"]),
            digest_algorithm=str(manifest.body.get("digest_algorithm", "sha256")),
            evidence_kind=(
                str(manifest.body["evidence_kind"])
                if "evidence_kind" in manifest.body
                else None
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ExternalEvidenceRecordManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "external_evidence_record_manifest@1.0.0"
    module_id: ClassVar[str] = "external_evidence_ingestion"

    external_evidence_record_id: str
    bundle_id: str
    source_id: str
    domain_id: str
    acquired_at: str
    evidence_kind: str
    source_digest_ref: TypedRef
    citation: str
    content: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    independence_mode: str = "external_domain_source"

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (self.source_digest_ref,)

    def body(self) -> dict[str, Any]:
        return {
            "external_evidence_record_id": self.external_evidence_record_id,
            "bundle_id": self.bundle_id,
            "source_id": self.source_id,
            "domain_id": self.domain_id,
            "acquired_at": self.acquired_at,
            "evidence_kind": self.evidence_kind,
            "source_digest_ref": self.source_digest_ref.as_dict(),
            "citation": self.citation,
            "content": normalize_json_value(self.content),
            "provenance": normalize_json_value(self.provenance),
            "independence_mode": self.independence_mode,
        }

    @classmethod
    def from_manifest(
        cls, manifest: ManifestEnvelope
    ) -> "ExternalEvidenceRecordManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            external_evidence_record_id=str(
                manifest.body["external_evidence_record_id"]
            ),
            bundle_id=str(manifest.body["bundle_id"]),
            source_id=str(manifest.body["source_id"]),
            domain_id=str(manifest.body["domain_id"]),
            acquired_at=str(manifest.body["acquired_at"]),
            evidence_kind=str(manifest.body["evidence_kind"]),
            source_digest_ref=_typed_ref_from_payload(
                manifest.body["source_digest_ref"],
                field_path="body.source_digest_ref",
            ),
            citation=str(manifest.body["citation"]),
            content=dict(manifest.body.get("content", {})),
            provenance=dict(manifest.body.get("provenance", {})),
            independence_mode=str(
                manifest.body.get("independence_mode", "external_domain_source")
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ExternalEvidenceManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "external_evidence_manifest@1.0.0"
    module_id: ClassVar[str] = "external_evidence_ingestion"

    external_evidence_id: str
    bundle_id: str
    domain_id: str
    acquisition_window: Mapping[str, Any]
    ordered_source_ids: tuple[str, ...]
    record_refs: tuple[TypedRef, ...]
    source_digest_refs: tuple[TypedRef, ...]
    source_count: int

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (*self.record_refs, *self.source_digest_refs)

    def body(self) -> dict[str, Any]:
        return {
            "external_evidence_id": self.external_evidence_id,
            "bundle_id": self.bundle_id,
            "domain_id": self.domain_id,
            "acquisition_window": normalize_json_value(self.acquisition_window),
            "ordered_source_ids": list(self.ordered_source_ids),
            "record_refs": [ref.as_dict() for ref in self.record_refs],
            "source_digest_refs": [ref.as_dict() for ref in self.source_digest_refs],
            "source_count": self.source_count,
        }

    @classmethod
    def from_manifest(cls, manifest: ManifestEnvelope) -> "ExternalEvidenceManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            external_evidence_id=str(manifest.body["external_evidence_id"]),
            bundle_id=str(manifest.body["bundle_id"]),
            domain_id=str(manifest.body["domain_id"]),
            acquisition_window=dict(manifest.body["acquisition_window"]),
            ordered_source_ids=_string_list_from_payload(
                manifest.body["ordered_source_ids"],
                field_path="body.ordered_source_ids",
            ),
            record_refs=_typed_ref_list_from_payload(
                manifest.body["record_refs"],
                field_path="body.record_refs",
            ),
            source_digest_refs=_typed_ref_list_from_payload(
                manifest.body["source_digest_refs"],
                field_path="body.source_digest_refs",
            ),
            source_count=int(manifest.body["source_count"]),
        )


@dataclass(frozen=True, kw_only=True)
class DomainSpecificMechanismMappingManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "domain_specific_mechanism_mapping_manifest@1.0.0"
    module_id: ClassVar[str] = "mechanistic_evidence"

    mechanism_mapping_id: str
    candidate_ref: TypedRef
    prediction_artifact_ref: TypedRef
    external_evidence_ref: TypedRef
    status: str
    term_bindings: tuple[Mapping[str, Any], ...] = ()
    reason_codes: tuple[str, ...] = ()

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (
            self.candidate_ref,
            self.prediction_artifact_ref,
            self.external_evidence_ref,
        )

    def body(self) -> dict[str, Any]:
        return {
            "mechanism_mapping_id": self.mechanism_mapping_id,
            "candidate_ref": self.candidate_ref.as_dict(),
            "prediction_artifact_ref": self.prediction_artifact_ref.as_dict(),
            "external_evidence_ref": self.external_evidence_ref.as_dict(),
            "status": self.status,
            "term_bindings": [
                normalize_json_value(item) for item in self.term_bindings
            ],
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_manifest(
        cls, manifest: ManifestEnvelope
    ) -> "DomainSpecificMechanismMappingManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            mechanism_mapping_id=str(manifest.body["mechanism_mapping_id"]),
            candidate_ref=_typed_ref_from_payload(
                manifest.body["candidate_ref"],
                field_path="body.candidate_ref",
            ),
            prediction_artifact_ref=_typed_ref_from_payload(
                manifest.body["prediction_artifact_ref"],
                field_path="body.prediction_artifact_ref",
            ),
            external_evidence_ref=_typed_ref_from_payload(
                manifest.body["external_evidence_ref"],
                field_path="body.external_evidence_ref",
            ),
            status=str(manifest.body["status"]),
            term_bindings=tuple(
                dict(item) for item in manifest.body.get("term_bindings", [])
            ),
            reason_codes=_string_list_from_payload(
                manifest.body.get("reason_codes", []),
                field_path="body.reason_codes",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class UnitsCheckManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "units_check_manifest@1.0.0"
    module_id: ClassVar[str] = "mechanistic_evidence"

    units_check_id: str
    mechanism_mapping_ref: TypedRef
    status: str
    term_units: tuple[Mapping[str, Any], ...] = ()
    reason_codes: tuple[str, ...] = ()

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (self.mechanism_mapping_ref,)

    def body(self) -> dict[str, Any]:
        return {
            "units_check_id": self.units_check_id,
            "mechanism_mapping_ref": self.mechanism_mapping_ref.as_dict(),
            "status": self.status,
            "term_units": [normalize_json_value(item) for item in self.term_units],
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_manifest(cls, manifest: ManifestEnvelope) -> "UnitsCheckManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            units_check_id=str(manifest.body["units_check_id"]),
            mechanism_mapping_ref=_typed_ref_from_payload(
                manifest.body["mechanism_mapping_ref"],
                field_path="body.mechanism_mapping_ref",
            ),
            status=str(manifest.body["status"]),
            term_units=tuple(
                dict(item) for item in manifest.body.get("term_units", [])
            ),
            reason_codes=_string_list_from_payload(
                manifest.body.get("reason_codes", []),
                field_path="body.reason_codes",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class InvarianceTestManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "invariance_test_manifest@1.0.0"
    module_id: ClassVar[str] = "mechanistic_evidence"

    invariance_test_id: str
    mechanism_mapping_ref: TypedRef
    external_evidence_ref: TypedRef
    status: str
    checks: tuple[Mapping[str, Any], ...] = ()
    reason_codes: tuple[str, ...] = ()

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (self.mechanism_mapping_ref, self.external_evidence_ref)

    def body(self) -> dict[str, Any]:
        return {
            "invariance_test_id": self.invariance_test_id,
            "mechanism_mapping_ref": self.mechanism_mapping_ref.as_dict(),
            "external_evidence_ref": self.external_evidence_ref.as_dict(),
            "status": self.status,
            "checks": [normalize_json_value(item) for item in self.checks],
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_manifest(cls, manifest: ManifestEnvelope) -> "InvarianceTestManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            invariance_test_id=str(manifest.body["invariance_test_id"]),
            mechanism_mapping_ref=_typed_ref_from_payload(
                manifest.body["mechanism_mapping_ref"],
                field_path="body.mechanism_mapping_ref",
            ),
            external_evidence_ref=_typed_ref_from_payload(
                manifest.body["external_evidence_ref"],
                field_path="body.external_evidence_ref",
            ),
            status=str(manifest.body["status"]),
            checks=tuple(dict(item) for item in manifest.body.get("checks", [])),
            reason_codes=_string_list_from_payload(
                manifest.body.get("reason_codes", []),
                field_path="body.reason_codes",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class EvidenceIndependenceProtocolManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "evidence_independence_protocol_manifest@1.0.0"
    module_id: ClassVar[str] = "mechanistic_evidence"

    evidence_independence_id: str
    external_evidence_ref: TypedRef
    prediction_artifact_ref: TypedRef
    status: str
    predictive_evidence_refs: tuple[TypedRef, ...] = ()
    overlap_refs: tuple[TypedRef, ...] = ()
    reason_codes: tuple[str, ...] = ()

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (
            self.external_evidence_ref,
            self.prediction_artifact_ref,
            *self.predictive_evidence_refs,
            *self.overlap_refs,
        )

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "evidence_independence_id": self.evidence_independence_id,
            "external_evidence_ref": self.external_evidence_ref.as_dict(),
            "prediction_artifact_ref": self.prediction_artifact_ref.as_dict(),
            "status": self.status,
            "predictive_evidence_refs": [
                ref.as_dict() for ref in self.predictive_evidence_refs
            ],
            "reason_codes": list(self.reason_codes),
        }
        if self.overlap_refs:
            body["overlap_refs"] = [ref.as_dict() for ref in self.overlap_refs]
        return body

    @classmethod
    def from_manifest(
        cls, manifest: ManifestEnvelope
    ) -> "EvidenceIndependenceProtocolManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            evidence_independence_id=str(manifest.body["evidence_independence_id"]),
            external_evidence_ref=_typed_ref_from_payload(
                manifest.body["external_evidence_ref"],
                field_path="body.external_evidence_ref",
            ),
            prediction_artifact_ref=_typed_ref_from_payload(
                manifest.body["prediction_artifact_ref"],
                field_path="body.prediction_artifact_ref",
            ),
            status=str(manifest.body["status"]),
            predictive_evidence_refs=_typed_ref_list_from_payload(
                manifest.body.get("predictive_evidence_refs", []),
                field_path="body.predictive_evidence_refs",
            ),
            overlap_refs=_typed_ref_list_from_payload(
                manifest.body.get("overlap_refs", []),
                field_path="body.overlap_refs",
            ),
            reason_codes=_string_list_from_payload(
                manifest.body.get("reason_codes", []),
                field_path="body.reason_codes",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class MechanisticEvidenceDossierManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "mechanistic_evidence_dossier_manifest@1.0.0"
    module_id: ClassVar[str] = "mechanistic_evidence"

    mechanistic_evidence_id: str
    candidate_ref: TypedRef
    prediction_artifact_ref: TypedRef
    external_evidence_ref: TypedRef
    mechanism_mapping_ref: TypedRef
    units_check_ref: TypedRef
    invariance_test_ref: TypedRef
    evidence_independence_ref: TypedRef
    status: str
    lower_claim_ceiling: str
    resolved_claim_ceiling: str
    reason_codes: tuple[str, ...] = ()

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (
            self.candidate_ref,
            self.prediction_artifact_ref,
            self.external_evidence_ref,
            self.mechanism_mapping_ref,
            self.units_check_ref,
            self.invariance_test_ref,
            self.evidence_independence_ref,
        )

    def body(self) -> dict[str, Any]:
        return {
            "mechanistic_evidence_id": self.mechanistic_evidence_id,
            "candidate_ref": self.candidate_ref.as_dict(),
            "prediction_artifact_ref": self.prediction_artifact_ref.as_dict(),
            "external_evidence_ref": self.external_evidence_ref.as_dict(),
            "mechanism_mapping_ref": self.mechanism_mapping_ref.as_dict(),
            "units_check_ref": self.units_check_ref.as_dict(),
            "invariance_test_ref": self.invariance_test_ref.as_dict(),
            "evidence_independence_ref": self.evidence_independence_ref.as_dict(),
            "status": self.status,
            "lower_claim_ceiling": self.lower_claim_ceiling,
            "resolved_claim_ceiling": self.resolved_claim_ceiling,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_manifest(
        cls, manifest: ManifestEnvelope
    ) -> "MechanisticEvidenceDossierManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            mechanistic_evidence_id=str(manifest.body["mechanistic_evidence_id"]),
            candidate_ref=_typed_ref_from_payload(
                manifest.body["candidate_ref"],
                field_path="body.candidate_ref",
            ),
            prediction_artifact_ref=_typed_ref_from_payload(
                manifest.body["prediction_artifact_ref"],
                field_path="body.prediction_artifact_ref",
            ),
            external_evidence_ref=_typed_ref_from_payload(
                manifest.body["external_evidence_ref"],
                field_path="body.external_evidence_ref",
            ),
            mechanism_mapping_ref=_typed_ref_from_payload(
                manifest.body["mechanism_mapping_ref"],
                field_path="body.mechanism_mapping_ref",
            ),
            units_check_ref=_typed_ref_from_payload(
                manifest.body["units_check_ref"],
                field_path="body.units_check_ref",
            ),
            invariance_test_ref=_typed_ref_from_payload(
                manifest.body["invariance_test_ref"],
                field_path="body.invariance_test_ref",
            ),
            evidence_independence_ref=_typed_ref_from_payload(
                manifest.body["evidence_independence_ref"],
                field_path="body.evidence_independence_ref",
            ),
            status=str(manifest.body["status"]),
            lower_claim_ceiling=str(manifest.body["lower_claim_ceiling"]),
            resolved_claim_ceiling=str(manifest.body["resolved_claim_ceiling"]),
            reason_codes=_string_list_from_payload(
                manifest.body.get("reason_codes", []),
                field_path="body.reason_codes",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class AbstentionManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "abstention_manifest@1.1.0"
    module_id: ClassVar[str] = "claims"

    abstention_id: str
    abstention_type: str
    blocked_ceiling: str
    reason_codes: tuple[str, ...]
    governing_refs: tuple[TypedRef, ...]

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return self.governing_refs

    def body(self) -> dict[str, Any]:
        return {
            "abstention_id": self.abstention_id,
            "abstention_type": self.abstention_type,
            "blocked_ceiling": self.blocked_ceiling,
            "reason_codes": list(self.reason_codes),
            "governing_refs": [ref.as_dict() for ref in self.governing_refs],
        }


@dataclass(frozen=True, kw_only=True)
class ReproducibilityBundleManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "reproducibility_bundle_manifest@1.0.0"
    module_id: ClassVar[str] = "replay"

    bundle_id: str
    bundle_mode: str
    dataset_snapshot_ref: TypedRef
    feature_view_ref: TypedRef
    search_plan_ref: TypedRef
    evaluation_plan_ref: TypedRef
    comparison_universe_ref: TypedRef
    evaluation_event_log_ref: TypedRef
    evaluation_governance_ref: TypedRef
    run_result_ref: TypedRef
    required_manifest_refs: tuple[TypedRef, ...]
    artifact_hash_records: tuple[ArtifactHashRecord, ...]
    seed_records: tuple[SeedRecord, ...]
    replay_verification_status: str
    failure_reason_codes: tuple[str, ...]
    environment_metadata: dict[str, str] = field(default_factory=dict)
    stage_order_records: tuple[ReplayStageRecord, ...] = ()
    scope_id: str = _DEFAULT_SCOPE_ID
    replay_entrypoint_id: str = "retained_scope_replay_v1"

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (
            self.dataset_snapshot_ref,
            self.feature_view_ref,
            self.search_plan_ref,
            self.evaluation_plan_ref,
            self.comparison_universe_ref,
            self.evaluation_event_log_ref,
            self.evaluation_governance_ref,
            self.run_result_ref,
            *self.required_manifest_refs,
            *(record.manifest_ref for record in self.stage_order_records),
        )

    def body(self) -> dict[str, Any]:
        body = {
            "bundle_id": self.bundle_id,
            "scope_id": self.scope_id,
            "bundle_mode": self.bundle_mode,
            "dataset_snapshot_ref": self.dataset_snapshot_ref.as_dict(),
            "feature_view_ref": self.feature_view_ref.as_dict(),
            "search_plan_ref": self.search_plan_ref.as_dict(),
            "evaluation_plan_ref": self.evaluation_plan_ref.as_dict(),
            "comparison_universe_ref": self.comparison_universe_ref.as_dict(),
            "evaluation_event_log_ref": self.evaluation_event_log_ref.as_dict(),
            "evaluation_governance_ref": self.evaluation_governance_ref.as_dict(),
            "run_result_ref": self.run_result_ref.as_dict(),
            "required_manifest_refs": [
                ref.as_dict() for ref in self.required_manifest_refs
            ],
            "artifact_hash_records": [
                record.as_dict() for record in self.artifact_hash_records
            ],
            "seed_records": [record.as_dict() for record in self.seed_records],
            "replay_entrypoint_id": self.replay_entrypoint_id,
            "replay_verification_status": self.replay_verification_status,
            "failure_reason_codes": list(self.failure_reason_codes),
        }
        if self.environment_metadata:
            body["environment_metadata"] = dict(self.environment_metadata)
        if self.stage_order_records:
            body["stage_order_records"] = [
                record.as_dict() for record in self.stage_order_records
            ]
        return body

    @classmethod
    def from_manifest(
        cls, manifest: ManifestEnvelope
    ) -> "ReproducibilityBundleManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            bundle_id=str(manifest.body["bundle_id"]),
            scope_id=str(manifest.body["scope_id"]),
            bundle_mode=str(manifest.body["bundle_mode"]),
            dataset_snapshot_ref=_typed_ref_from_payload(
                manifest.body["dataset_snapshot_ref"],
                field_path="body.dataset_snapshot_ref",
            ),
            feature_view_ref=_typed_ref_from_payload(
                manifest.body["feature_view_ref"],
                field_path="body.feature_view_ref",
            ),
            search_plan_ref=_typed_ref_from_payload(
                manifest.body["search_plan_ref"],
                field_path="body.search_plan_ref",
            ),
            evaluation_plan_ref=_typed_ref_from_payload(
                manifest.body["evaluation_plan_ref"],
                field_path="body.evaluation_plan_ref",
            ),
            comparison_universe_ref=_typed_ref_from_payload(
                manifest.body["comparison_universe_ref"],
                field_path="body.comparison_universe_ref",
            ),
            evaluation_event_log_ref=_typed_ref_from_payload(
                manifest.body["evaluation_event_log_ref"],
                field_path="body.evaluation_event_log_ref",
            ),
            evaluation_governance_ref=_typed_ref_from_payload(
                manifest.body["evaluation_governance_ref"],
                field_path="body.evaluation_governance_ref",
            ),
            run_result_ref=_typed_ref_from_payload(
                manifest.body["run_result_ref"],
                field_path="body.run_result_ref",
            ),
            required_manifest_refs=_typed_ref_list_from_payload(
                manifest.body["required_manifest_refs"],
                field_path="body.required_manifest_refs",
            ),
            artifact_hash_records=tuple(
                ArtifactHashRecord.from_payload(
                    item,
                    field_path=f"body.artifact_hash_records[{index}]",
                )
                for index, item in enumerate(manifest.body["artifact_hash_records"])
            ),
            seed_records=tuple(
                SeedRecord.from_payload(
                    item,
                    field_path=f"body.seed_records[{index}]",
                )
                for index, item in enumerate(manifest.body["seed_records"])
            ),
            replay_entrypoint_id=str(manifest.body["replay_entrypoint_id"]),
            replay_verification_status=str(manifest.body["replay_verification_status"]),
            failure_reason_codes=_string_list_from_payload(
                manifest.body["failure_reason_codes"],
                field_path="body.failure_reason_codes",
            ),
            environment_metadata=_string_mapping_from_payload(
                manifest.body.get("environment_metadata", {}),
                field_path="body.environment_metadata",
            ),
            stage_order_records=tuple(
                ReplayStageRecord.from_payload(
                    item,
                    field_path=f"body.stage_order_records[{index}]",
                )
                for index, item in enumerate(
                    manifest.body.get("stage_order_records", [])
                )
            ),
        )


@dataclass(frozen=True, kw_only=True)
class RunResultManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "run_result_manifest@1.1.0"
    module_id: ClassVar[str] = "catalog_publishing"

    run_id: str
    scope_ledger_ref: TypedRef
    search_plan_ref: TypedRef
    evaluation_plan_ref: TypedRef
    comparison_universe_ref: TypedRef
    evaluation_event_log_ref: TypedRef
    evaluation_governance_ref: TypedRef
    result_mode: str
    reproducibility_bundle_ref: TypedRef
    forecast_object_type: str = "point"
    primary_validation_scope_ref: TypedRef | None = None
    prediction_artifact_refs: tuple[TypedRef, ...] = ()
    residual_history_refs: tuple[TypedRef, ...] = ()
    stochastic_model_refs: tuple[TypedRef, ...] = ()
    stochastic_support_status: str | None = None
    stochastic_support_reason_codes: tuple[str, ...] = ()
    primary_score_result_ref: TypedRef | None = None
    primary_calibration_result_ref: TypedRef | None = None
    primary_external_evidence_ref: TypedRef | None = None
    primary_mechanistic_evidence_ref: TypedRef | None = None
    robustness_report_refs: tuple[TypedRef, ...] = ()
    deferred_scope_policy_refs: tuple[TypedRef, ...] = ()
    primary_reducer_artifact_ref: TypedRef | None = None
    primary_scorecard_ref: TypedRef | None = None
    primary_claim_card_ref: TypedRef | None = None
    primary_abstention_ref: TypedRef | None = None

    def __post_init__(self) -> None:
        if self.result_mode == "candidate_publication":
            required = (
                self.primary_reducer_artifact_ref,
                self.primary_scorecard_ref,
                self.primary_claim_card_ref,
            )
            if any(ref is None for ref in required) or self.primary_abstention_ref:
                raise ContractValidationError(
                    code="invalid_result_mode_payload",
                    message=(
                        "candidate_publication requires candidate refs and "
                        "forbids abstention refs"
                    ),
                    field_path="result_mode",
                )
            return
        if self.result_mode == "abstention_only_publication":
            if self.primary_abstention_ref is None:
                raise ContractValidationError(
                    code="invalid_result_mode_payload",
                    message=(
                        "abstention_only_publication requires primary_abstention_ref"
                    ),
                    field_path="primary_abstention_ref",
                )
            if any(
                ref is not None
                for ref in (
                    self.primary_reducer_artifact_ref,
                    self.primary_scorecard_ref,
                    self.primary_claim_card_ref,
                )
            ):
                raise ContractValidationError(
                    code="invalid_result_mode_payload",
                    message=(
                        "abstention_only_publication forbids candidate placeholders"
                    ),
                    field_path="result_mode",
                )
            return
        raise ContractValidationError(
            code="invalid_result_mode_payload",
            message=f"unsupported result_mode {self.result_mode!r}",
            field_path="result_mode",
        )

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        refs: list[TypedRef] = [
            self.scope_ledger_ref,
            self.search_plan_ref,
            self.evaluation_plan_ref,
            self.comparison_universe_ref,
            self.evaluation_event_log_ref,
            self.evaluation_governance_ref,
            *(
                ()
                if self.primary_validation_scope_ref is None
                else (self.primary_validation_scope_ref,)
            ),
            *self.prediction_artifact_refs,
            *self.residual_history_refs,
            *self.stochastic_model_refs,
            *(
                ()
                if self.primary_score_result_ref is None
                else (self.primary_score_result_ref,)
            ),
            *(
                ()
                if self.primary_calibration_result_ref is None
                else (self.primary_calibration_result_ref,)
            ),
            *(
                ()
                if self.primary_external_evidence_ref is None
                else (self.primary_external_evidence_ref,)
            ),
            *(
                ()
                if self.primary_mechanistic_evidence_ref is None
                else (self.primary_mechanistic_evidence_ref,)
            ),
            *self.robustness_report_refs,
            self.reproducibility_bundle_ref,
            *self.deferred_scope_policy_refs,
        ]
        for optional_ref in (
            self.primary_reducer_artifact_ref,
            self.primary_scorecard_ref,
            self.primary_claim_card_ref,
            self.primary_abstention_ref,
        ):
            if optional_ref is not None:
                refs.append(optional_ref)
        return tuple(refs)

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "run_id": self.run_id,
            "scope_ledger_ref": self.scope_ledger_ref.as_dict(),
            "search_plan_ref": self.search_plan_ref.as_dict(),
            "evaluation_plan_ref": self.evaluation_plan_ref.as_dict(),
            "comparison_universe_ref": self.comparison_universe_ref.as_dict(),
            "evaluation_event_log_ref": self.evaluation_event_log_ref.as_dict(),
            "evaluation_governance_ref": self.evaluation_governance_ref.as_dict(),
            "result_mode": self.result_mode,
            "prediction_artifact_refs": [
                ref.as_dict() for ref in self.prediction_artifact_refs
            ],
            "residual_history_refs": [
                ref.as_dict() for ref in self.residual_history_refs
            ],
            "stochastic_model_refs": [
                ref.as_dict() for ref in self.stochastic_model_refs
            ],
            "robustness_report_refs": [
                ref.as_dict() for ref in self.robustness_report_refs
            ],
            "reproducibility_bundle_ref": self.reproducibility_bundle_ref.as_dict(),
        }
        body["forecast_object_type"] = self.forecast_object_type
        if self.stochastic_support_status is not None:
            body["stochastic_support_status"] = self.stochastic_support_status
            body["stochastic_support_reason_codes"] = list(
                self.stochastic_support_reason_codes
            )
        if self.primary_validation_scope_ref is not None:
            body["primary_validation_scope_ref"] = (
                self.primary_validation_scope_ref.as_dict()
            )
        if self.primary_score_result_ref is not None:
            body["primary_score_result_ref"] = self.primary_score_result_ref.as_dict()
        if self.primary_calibration_result_ref is not None:
            body["primary_calibration_result_ref"] = (
                self.primary_calibration_result_ref.as_dict()
            )
        if self.primary_external_evidence_ref is not None:
            body["primary_external_evidence_ref"] = (
                self.primary_external_evidence_ref.as_dict()
            )
        if self.primary_mechanistic_evidence_ref is not None:
            body["primary_mechanistic_evidence_ref"] = (
                self.primary_mechanistic_evidence_ref.as_dict()
            )
        if self.primary_reducer_artifact_ref is not None:
            body["primary_reducer_artifact_ref"] = (
                self.primary_reducer_artifact_ref.as_dict()
            )
        if self.primary_scorecard_ref is not None:
            body["primary_scorecard_ref"] = self.primary_scorecard_ref.as_dict()
        if self.primary_claim_card_ref is not None:
            body["primary_claim_card_ref"] = self.primary_claim_card_ref.as_dict()
        if self.primary_abstention_ref is not None:
            body["primary_abstention_ref"] = self.primary_abstention_ref.as_dict()
        if self.deferred_scope_policy_refs:
            body["deferred_scope_policy_refs"] = [
                ref.as_dict() for ref in self.deferred_scope_policy_refs
            ]
        return body

    @classmethod
    def from_manifest(cls, manifest: ManifestEnvelope) -> "RunResultManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            run_id=str(manifest.body["run_id"]),
            scope_ledger_ref=_typed_ref_from_payload(
                manifest.body["scope_ledger_ref"],
                field_path="body.scope_ledger_ref",
            ),
            search_plan_ref=_typed_ref_from_payload(
                manifest.body["search_plan_ref"],
                field_path="body.search_plan_ref",
            ),
            evaluation_plan_ref=_typed_ref_from_payload(
                manifest.body["evaluation_plan_ref"],
                field_path="body.evaluation_plan_ref",
            ),
            comparison_universe_ref=_typed_ref_from_payload(
                manifest.body["comparison_universe_ref"],
                field_path="body.comparison_universe_ref",
            ),
            evaluation_event_log_ref=_typed_ref_from_payload(
                manifest.body["evaluation_event_log_ref"],
                field_path="body.evaluation_event_log_ref",
            ),
            evaluation_governance_ref=_typed_ref_from_payload(
                manifest.body["evaluation_governance_ref"],
                field_path="body.evaluation_governance_ref",
            ),
            result_mode=str(manifest.body["result_mode"]),
            forecast_object_type=str(
                manifest.body.get("forecast_object_type", "point")
            ),
            primary_validation_scope_ref=(
                _typed_ref_from_payload(
                    manifest.body["primary_validation_scope_ref"],
                    field_path="body.primary_validation_scope_ref",
                )
                if "primary_validation_scope_ref" in manifest.body
                else None
            ),
            primary_reducer_artifact_ref=(
                _typed_ref_from_payload(
                    manifest.body["primary_reducer_artifact_ref"],
                    field_path="body.primary_reducer_artifact_ref",
                )
                if "primary_reducer_artifact_ref" in manifest.body
                else None
            ),
            primary_scorecard_ref=(
                _typed_ref_from_payload(
                    manifest.body["primary_scorecard_ref"],
                    field_path="body.primary_scorecard_ref",
                )
                if "primary_scorecard_ref" in manifest.body
                else None
            ),
            primary_claim_card_ref=(
                _typed_ref_from_payload(
                    manifest.body["primary_claim_card_ref"],
                    field_path="body.primary_claim_card_ref",
                )
                if "primary_claim_card_ref" in manifest.body
                else None
            ),
            primary_abstention_ref=(
                _typed_ref_from_payload(
                    manifest.body["primary_abstention_ref"],
                    field_path="body.primary_abstention_ref",
                )
                if "primary_abstention_ref" in manifest.body
                else None
            ),
            prediction_artifact_refs=_typed_ref_list_from_payload(
                manifest.body.get("prediction_artifact_refs", []),
                field_path="body.prediction_artifact_refs",
            ),
            residual_history_refs=_typed_ref_list_from_payload(
                manifest.body.get("residual_history_refs", []),
                field_path="body.residual_history_refs",
            ),
            stochastic_model_refs=_typed_ref_list_from_payload(
                manifest.body.get("stochastic_model_refs", []),
                field_path="body.stochastic_model_refs",
            ),
            stochastic_support_status=(
                str(manifest.body["stochastic_support_status"])
                if manifest.body.get("stochastic_support_status") is not None
                else None
            ),
            stochastic_support_reason_codes=_string_list_from_payload(
                manifest.body.get("stochastic_support_reason_codes", []),
                field_path="body.stochastic_support_reason_codes",
            ),
            primary_score_result_ref=(
                _typed_ref_from_payload(
                    manifest.body["primary_score_result_ref"],
                    field_path="body.primary_score_result_ref",
                )
                if "primary_score_result_ref" in manifest.body
                else None
            ),
            primary_calibration_result_ref=(
                _typed_ref_from_payload(
                    manifest.body["primary_calibration_result_ref"],
                    field_path="body.primary_calibration_result_ref",
                )
                if "primary_calibration_result_ref" in manifest.body
                else None
            ),
            primary_external_evidence_ref=(
                _typed_ref_from_payload(
                    manifest.body["primary_external_evidence_ref"],
                    field_path="body.primary_external_evidence_ref",
                )
                if "primary_external_evidence_ref" in manifest.body
                else None
            ),
            primary_mechanistic_evidence_ref=(
                _typed_ref_from_payload(
                    manifest.body["primary_mechanistic_evidence_ref"],
                    field_path="body.primary_mechanistic_evidence_ref",
                )
                if "primary_mechanistic_evidence_ref" in manifest.body
                else None
            ),
            robustness_report_refs=_typed_ref_list_from_payload(
                manifest.body.get("robustness_report_refs", []),
                field_path="body.robustness_report_refs",
            ),
            reproducibility_bundle_ref=_typed_ref_from_payload(
                manifest.body["reproducibility_bundle_ref"],
                field_path="body.reproducibility_bundle_ref",
            ),
            deferred_scope_policy_refs=_typed_ref_list_from_payload(
                manifest.body.get("deferred_scope_policy_refs", []),
                field_path="body.deferred_scope_policy_refs",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ReadinessGateRecord:
    gate_id: str
    status: str
    required: bool
    summary: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "status": self.status,
            "required": self.required,
            "summary": self.summary,
            "evidence": _json_ready_mapping(self.evidence),
        }

    @classmethod
    def from_payload(cls, payload: Any, *, field_path: str) -> "ReadinessGateRecord":
        if not isinstance(payload, Mapping):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path} must be a mapping",
                field_path=field_path,
            )
        gate_id = payload.get("gate_id")
        status = payload.get("status")
        required = payload.get("required")
        summary = payload.get("summary")
        if not isinstance(gate_id, str):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path}.gate_id must be a string",
                field_path=f"{field_path}.gate_id",
            )
        if not isinstance(status, str):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path}.status must be a string",
                field_path=f"{field_path}.status",
            )
        if not isinstance(required, bool):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path}.required must be a boolean",
                field_path=f"{field_path}.required",
            )
        if not isinstance(summary, str):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path}.summary must be a string",
                field_path=f"{field_path}.summary",
            )
        evidence_payload = payload.get("evidence", {})
        if not isinstance(evidence_payload, Mapping):
            raise ContractValidationError(
                code="invalid_manifest_model_field",
                message=f"{field_path}.evidence must be a mapping",
                field_path=f"{field_path}.evidence",
            )
        return cls(
            gate_id=gate_id,
            status=status,
            required=required,
            summary=summary,
            evidence=dict(evidence_payload),
        )


def _json_ready_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): normalize_json_value(value) for key, value in payload.items()}


@dataclass(frozen=True, kw_only=True)
class ReadinessJudgmentManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "readiness_judgment_manifest@1.0.0"
    module_id: ClassVar[str] = "catalog_publishing"

    judgment_id: str
    final_verdict: str
    catalog_scope: str
    verdict_summary: str = ""
    reason_codes: tuple[str, ...] = ()
    judged_at: str | None = None
    required_gate_count: int = 0
    passed_gate_count: int = 0
    failed_gate_count: int = 0
    missing_gate_count: int = 0
    gate_records: tuple[ReadinessGateRecord, ...] = ()

    def body(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "judgment_id": self.judgment_id,
            "final_verdict": self.final_verdict,
            "catalog_scope": self.catalog_scope,
            "verdict_summary": self.verdict_summary,
            "reason_codes": list(self.reason_codes),
            "required_gate_count": self.required_gate_count,
            "passed_gate_count": self.passed_gate_count,
            "failed_gate_count": self.failed_gate_count,
            "missing_gate_count": self.missing_gate_count,
            "gate_records": [record.as_dict() for record in self.gate_records],
        }
        if self.judged_at is not None:
            payload["judged_at"] = self.judged_at
        return payload

    @classmethod
    def from_manifest(cls, manifest: ManifestEnvelope) -> "ReadinessJudgmentManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            judgment_id=str(manifest.body["judgment_id"]),
            final_verdict=str(manifest.body["final_verdict"]),
            catalog_scope=str(manifest.body["catalog_scope"]),
            verdict_summary=str(manifest.body.get("verdict_summary", "")),
            reason_codes=_string_list_from_payload(
                manifest.body.get("reason_codes", []),
                field_path="body.reason_codes",
            ),
            judged_at=(
                str(manifest.body["judged_at"])
                if "judged_at" in manifest.body
                else None
            ),
            required_gate_count=int(manifest.body.get("required_gate_count", 0)),
            passed_gate_count=int(manifest.body.get("passed_gate_count", 0)),
            failed_gate_count=int(manifest.body.get("failed_gate_count", 0)),
            missing_gate_count=int(manifest.body.get("missing_gate_count", 0)),
            gate_records=tuple(
                ReadinessGateRecord.from_payload(
                    item,
                    field_path=f"body.gate_records[{index}]",
                )
                for index, item in enumerate(manifest.body.get("gate_records", []))
            ),
        )


@dataclass(frozen=True, kw_only=True)
class SchemaLifecycleIntegrationClosureManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "schema_lifecycle_integration_closure_manifest@1.0.0"
    module_id: ClassVar[str] = "catalog_publishing"

    closure_id: str
    status: str

    def body(self) -> dict[str, Any]:
        return {"closure_id": self.closure_id, "status": self.status}


@dataclass(frozen=True, kw_only=True)
class PublicationRecordManifest(RuntimeManifestModel):
    schema_name: ClassVar[str] = "publication_record_manifest@1.1.0"
    module_id: ClassVar[str] = "catalog_publishing"

    publication_id: str
    run_result_ref: TypedRef
    catalog_scope: str
    publication_mode: str
    replay_verification_status: str
    comparator_exposure_status: str
    reproducibility_bundle_ref: TypedRef
    readiness_judgment_ref: TypedRef
    schema_lifecycle_integration_closure_ref: TypedRef
    published_at: str

    @property
    def body_refs(self) -> tuple[TypedRef, ...]:
        return (
            self.run_result_ref,
            self.reproducibility_bundle_ref,
            self.readiness_judgment_ref,
            self.schema_lifecycle_integration_closure_ref,
        )

    def body(self) -> dict[str, Any]:
        return {
            "publication_id": self.publication_id,
            "run_result_ref": self.run_result_ref.as_dict(),
            "catalog_scope": self.catalog_scope,
            "publication_mode": self.publication_mode,
            "replay_verification_status": self.replay_verification_status,
            "comparator_exposure_status": self.comparator_exposure_status,
            "reproducibility_bundle_ref": (self.reproducibility_bundle_ref.as_dict()),
            "readiness_judgment_ref": self.readiness_judgment_ref.as_dict(),
            "schema_lifecycle_integration_closure_ref": (
                self.schema_lifecycle_integration_closure_ref.as_dict()
            ),
            "published_at": self.published_at,
        }

    @classmethod
    def from_manifest(cls, manifest: ManifestEnvelope) -> "PublicationRecordManifest":
        cls._validate_manifest(manifest)
        return cls(
            object_id=manifest.object_id,
            publication_id=str(manifest.body["publication_id"]),
            run_result_ref=_typed_ref_from_payload(
                manifest.body["run_result_ref"],
                field_path="body.run_result_ref",
            ),
            catalog_scope=str(manifest.body["catalog_scope"]),
            publication_mode=str(manifest.body["publication_mode"]),
            replay_verification_status=str(manifest.body["replay_verification_status"]),
            comparator_exposure_status=str(manifest.body["comparator_exposure_status"]),
            reproducibility_bundle_ref=_typed_ref_from_payload(
                manifest.body["reproducibility_bundle_ref"],
                field_path="body.reproducibility_bundle_ref",
            ),
            readiness_judgment_ref=_typed_ref_from_payload(
                manifest.body["readiness_judgment_ref"],
                field_path="body.readiness_judgment_ref",
            ),
            schema_lifecycle_integration_closure_ref=_typed_ref_from_payload(
                manifest.body["schema_lifecycle_integration_closure_ref"],
                field_path="body.schema_lifecycle_integration_closure_ref",
            ),
            published_at=str(manifest.body["published_at"]),
        )


__all__ = [
    "AbstentionManifest",
    "ArtifactHashRecord",
    "CalibrationContractManifest",
    "CalibrationResultManifest",
    "CandidateStateManifest",
    "CandidateSpecManifest",
    "CanonicalizationPolicyManifest",
    "DatasetSnapshotManifestModel",
    "DomainSpecificMechanismMappingManifest",
    "DistributionPredictionRow",
    "EvidenceIndependenceProtocolManifest",
    "EventProbabilityPredictionRow",
    "EvaluationPlanManifestModel",
    "ExternalEvidenceManifest",
    "ExternalEvidenceRecordManifest",
    "FreezeEventManifest",
    "FrozenShortlistManifest",
    "FrontierManifest",
    "InvarianceTestManifest",
    "IntervalPredictionRow",
    "IntervalValue",
    "MechanisticEvidenceDossierManifest",
    "NullResultManifest",
    "PerHorizonPrimaryScore",
    "PerHorizonScore",
    "PerturbationFamilyResultManifest",
    "PredictionArtifactManifest",
    "PredictionArtifactRow",
    "PredictionRow",
    "ProbabilisticScoreResultManifest",
    "PublicationRecordManifest",
    "QuantilePredictionRow",
    "QuantileValue",
    "ReadinessGateRecord",
    "ReadinessJudgmentManifest",
    "ReplayStageRecord",
    "RejectedDiagnosticsManifest",
    "ReproducibilityBundleManifest",
    "RobustnessReportManifest",
    "RunDeclarationManifest",
    "RunResultManifest",
    "RuntimeManifestModel",
    "SchemaLifecycleIntegrationClosureManifest",
    "SearchLedgerManifest",
    "SearchPlanManifest",
    "SeedRecord",
    "SensitivityAnalysisManifest",
    "SourceDigestManifest",
    "PointScoreResultManifest",
    "UnitsCheckManifest",
]
