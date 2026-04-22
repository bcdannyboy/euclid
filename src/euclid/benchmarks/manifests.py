from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from euclid.contracts.errors import ContractValidationError

TRACK_IDS = (
    "rediscovery",
    "predictive_generalization",
    "adversarial_honesty",
)

BENCHMARK_DIRECTORY_NAMES = (
    "tracks",
    "tasks",
    "manifests",
    "baselines",
    "results",
    "reports",
)

_FORECAST_OBJECT_TYPES = {
    "point",
    "distribution",
    "interval",
    "quantile",
    "event_probability",
}
_GENERATOR_STATUSES = {
    "known_generator",
    "near_equivalent_generator",
    "unknown_real_world",
}
_SEARCH_CLASS_IDS = {
    "exact_finite_enumeration",
    "bounded_heuristic",
    "equality_saturation_heuristic",
    "stochastic_heuristic",
}
_COMPOSITION_OPERATOR_IDS = {
    "piecewise",
    "additive_residual",
    "regime_conditioned",
    "shared_plus_local_decomposition",
}
_FULL_VISION_REQUIRED_SURFACE_IDS = {
    "retained_core_release",
    "probabilistic_forecast_surface",
    "algorithmic_backend",
    "search_class_honesty",
    "composition_operator_semantics",
    "shared_plus_local_decomposition",
    "mechanistic_lane",
    "external_evidence_ingestion",
    "robustness_lane",
    "portfolio_orchestration",
}
_FULL_VISION_REQUIRED_CASE_TAGS = {
    "positive_case",
    "negative_case",
    "abstention_case",
}
_CERTIFICATION_FIXTURE_ROOT_PREFIX = "fixtures/runtime/full_vision_certification/"
_EXPECTED_AUTHORITY_SNAPSHOT_ID = "euclid-authority-2026-04-15-b"
_EXPECTED_FIXTURE_SPEC_ID = "euclid-certification-fixtures-v1"
_PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID = "portfolio_orchestrator"


def _raise_validation_error(
    *,
    code: str,
    message: str,
    field_path: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    raise ContractValidationError(
        code=code,
        message=message,
        field_path=field_path,
        details=dict(details or {}),
    )


def _require_mapping(payload: Any, *, field_path: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message=f"{field_path} must be a mapping",
            field_path=field_path,
        )
    return {str(key): value for key, value in payload.items()}


def _require_string(payload: Any, *, field_path: str) -> str:
    if not isinstance(payload, str) or not payload.strip():
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message=f"{field_path} must be a non-empty string",
            field_path=field_path,
        )
    return payload.strip()


def _require_allowed_string(payload: Any, *, field_path: str, allowed: set[str]) -> str:
    value = _require_string(payload, field_path=field_path)
    if value not in allowed:
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message=f"{field_path} must be one of {sorted(allowed)}",
            field_path=field_path,
            details={"allowed_values": sorted(allowed), "value": value},
        )
    return value


def _require_float(payload: Any, *, field_path: str) -> float:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message=f"{field_path} must be a number",
            field_path=field_path,
        )
    return float(payload)


def _require_string_list(payload: Any, *, field_path: str) -> tuple[str, ...]:
    if not isinstance(payload, list):
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message=f"{field_path} must be a list",
            field_path=field_path,
        )
    values = tuple(
        _require_string(item, field_path=f"{field_path}[{index}]")
        for index, item in enumerate(payload)
    )
    if not values:
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message=f"{field_path} must not be empty",
            field_path=field_path,
        )
    return values


def _optional_mapping(payload: Any, *, field_path: str) -> dict[str, Any] | None:
    if payload is None:
        return None
    return _require_mapping(payload, field_path=field_path)


def _optional_string(payload: Any, *, field_path: str) -> str | None:
    if payload is None:
        return None
    return _require_string(payload, field_path=field_path)


def _optional_string_list(payload: Any, *, field_path: str) -> tuple[str, ...]:
    if payload is None:
        return ()
    return _require_string_list(payload, field_path=field_path)


@dataclass(frozen=True)
class BenchmarkRegistryEntry:
    entry_id: str
    payload: dict[str, Any]

    @classmethod
    def from_payload(
        cls, payload: Any, *, field_path: str, id_field: str
    ) -> "BenchmarkRegistryEntry":
        data = _require_mapping(payload, field_path=field_path)
        return cls(
            entry_id=_require_string(
                data.get(id_field),
                field_path=f"{field_path}.{id_field}",
            ),
            payload=data,
        )


def _require_registry(
    payload: Any,
    *,
    field_path: str,
    id_field: str,
) -> tuple[BenchmarkRegistryEntry, ...]:
    if not isinstance(payload, list):
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message=f"{field_path} must be a list",
            field_path=field_path,
        )
    entries = tuple(
        BenchmarkRegistryEntry.from_payload(
            item,
            field_path=f"{field_path}[{index}]",
            id_field=id_field,
        )
        for index, item in enumerate(payload)
    )
    if not entries:
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message=f"{field_path} must not be empty",
            field_path=field_path,
        )
    return entries


@dataclass(frozen=True)
class FrozenBenchmarkProtocol:
    dataset_ref: str
    snapshot_policy: dict[str, Any]
    target_transform_policy: dict[str, Any]
    quantization_policy: dict[str, Any]
    observation_model_policy: dict[str, Any]
    split_policy: dict[str, Any]
    forecast_object_type: str
    score_policy: dict[str, Any]
    calibration_policy: dict[str, Any] | None
    budget_policy: dict[str, Any]
    seed_policy: dict[str, Any]
    replay_policy: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkTaskManifest:
    task_id: str
    track_id: str
    task_family: str
    search_class: str | None
    search_class_honesty: dict[str, Any]
    composition_operators: tuple[str, ...]
    regime_tags: tuple[str, ...]
    generator_status: str
    practical_significance_margin: float
    adversarial_tags: tuple[str, ...]
    forbidden_shortcuts: tuple[str, ...]
    abstention_policy: dict[str, Any]
    baseline_registry: tuple[BenchmarkRegistryEntry, ...]
    submitter_registry: tuple[BenchmarkRegistryEntry, ...]
    frozen_protocol: FrozenBenchmarkProtocol
    metric_thresholds: dict[str, Any]
    expected_claim_ceiling: str
    false_claim_expectations: tuple[str, ...]
    engine_requirements: tuple[str, ...]
    semantic_readiness_row_ids: tuple[str, ...]
    fixture_spec_id: str | None
    fixture_family_id: str | None
    source_path: Path

    @property
    def dataset_ref(self) -> str:
        return self.frozen_protocol.dataset_ref

    @property
    def submitter_ids(self) -> tuple[str, ...]:
        return tuple(entry.entry_id for entry in self.submitter_registry)

    @property
    def baseline_ids(self) -> tuple[str, ...]:
        return tuple(entry.entry_id for entry in self.baseline_registry)

    @property
    def score_law(self) -> str:
        metric_id = self.frozen_protocol.score_policy.get("metric_id")
        if isinstance(metric_id, str) and metric_id.strip():
            return metric_id.strip()
        return "unknown_score_law"

    @property
    def calibration_required(self) -> bool:
        calibration_policy = self.frozen_protocol.calibration_policy
        if not isinstance(calibration_policy, Mapping):
            return False
        return bool(calibration_policy.get("required"))

    @property
    def calibration_expectation(self) -> str | None:
        calibration_policy = self.frozen_protocol.calibration_policy
        if not isinstance(calibration_policy, Mapping):
            return None
        return "required" if bool(calibration_policy.get("required")) else "optional"

    @property
    def abstention_mode(self) -> str:
        expected_mode = self.abstention_policy.get("expected_mode")
        if isinstance(expected_mode, str) and expected_mode.strip():
            return expected_mode.strip()
        return "not_declared"

    @property
    def replay_obligation(self) -> str:
        verification_mode = self.frozen_protocol.replay_policy.get("verification_mode")
        if isinstance(verification_mode, str) and verification_mode.strip():
            return verification_mode.strip()
        return "ledger_only"


@dataclass(frozen=True)
class RediscoveryTaskManifest(BenchmarkTaskManifest):
    target_structure_ref: str
    equivalence_policy: dict[str, Any]
    parameter_tolerance_policy: dict[str, Any] | None
    predictive_adequacy_floor: dict[str, Any]


@dataclass(frozen=True)
class PredictiveGeneralizationTaskManifest(BenchmarkTaskManifest):
    origin_policy: dict[str, Any]
    horizon_policy: dict[str, Any]
    baseline_comparison_policy: dict[str, Any]


@dataclass(frozen=True)
class AdversarialHonestyTaskManifest(BenchmarkTaskManifest):
    trap_class: str
    expected_safe_outcome: str
    failure_severity: str


@dataclass(frozen=True)
class BenchmarkRepositoryTree:
    root: Path
    tracks_dir: Path
    tasks_dir: Path
    manifests_dir: Path
    baselines_dir: Path
    results_dir: Path
    reports_dir: Path

    def track_directory(self, category: str, track_id: str) -> Path:
        if category not in BENCHMARK_DIRECTORY_NAMES:
            _raise_validation_error(
                code="invalid_benchmark_directory",
                message=f"{category!r} is not a benchmark directory category",
                field_path="category",
                details={"category": category},
            )
        if track_id not in TRACK_IDS:
            _raise_validation_error(
                code="invalid_benchmark_manifest_field",
                message=f"{track_id!r} is not a supported track id",
                field_path="track_id",
                details={"track_id": track_id},
            )
        return getattr(self, f"{category}_dir") / track_id


@dataclass(frozen=True)
class BenchmarkSuiteSurfaceRequirement:
    surface_id: str
    task_ids: tuple[str, ...]
    replay_required: bool = True


@dataclass(frozen=True)
class BenchmarkSuiteManifest:
    suite_id: str
    description: str
    task_manifest_paths: tuple[Path, ...]
    required_tracks: tuple[str, ...]
    surface_requirements: tuple[BenchmarkSuiteSurfaceRequirement, ...]
    authority_snapshot_id: str | None
    fixture_spec_id: str | None
    source_path: Path


def ensure_benchmark_repository_tree(root: Path | str) -> BenchmarkRepositoryTree:
    benchmark_root = Path(root)
    benchmark_root.mkdir(parents=True, exist_ok=True)

    directories = {name: benchmark_root / name for name in BENCHMARK_DIRECTORY_NAMES}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
        for track_id in TRACK_IDS:
            (directory / track_id).mkdir(parents=True, exist_ok=True)

    return BenchmarkRepositoryTree(
        root=benchmark_root,
        tracks_dir=directories["tracks"],
        tasks_dir=directories["tasks"],
        manifests_dir=directories["manifests"],
        baselines_dir=directories["baselines"],
        results_dir=directories["results"],
        reports_dir=directories["reports"],
    )


def _validate_manifest_track_path(source_path: Path, *, track_id: str) -> None:
    parts = source_path.parts
    if "tasks" not in parts:
        return
    tasks_index = parts.index("tasks")
    if tasks_index + 1 >= len(parts):
        return
    track_directory = parts[tasks_index + 1]
    if track_directory in TRACK_IDS and track_directory != track_id:
        _raise_validation_error(
            code="benchmark_track_path_mismatch",
            message="task manifest path does not match declared track_id",
            field_path="track_id",
            details={
                "path_track_id": track_directory,
                "track_id": track_id,
                "source_path": str(source_path),
            },
        )


def _build_common_fields(
    payload: Mapping[str, Any], *, source_path: Path
) -> dict[str, Any]:
    track_id = _require_allowed_string(
        payload.get("track_id"),
        field_path="track_id",
        allowed=set(TRACK_IDS),
    )
    _validate_manifest_track_path(source_path, track_id=track_id)

    frozen_protocol = FrozenBenchmarkProtocol(
        dataset_ref=_require_string(
            payload.get("dataset_ref"),
            field_path="dataset_ref",
        ),
        snapshot_policy=_require_mapping(
            payload.get("snapshot_policy"),
            field_path="snapshot_policy",
        ),
        target_transform_policy=_require_mapping(
            payload.get("target_transform_policy"),
            field_path="target_transform_policy",
        ),
        quantization_policy=_require_mapping(
            payload.get("quantization_policy"),
            field_path="quantization_policy",
        ),
        observation_model_policy=_require_mapping(
            payload.get("observation_model_policy"),
            field_path="observation_model_policy",
        ),
        split_policy=_require_mapping(
            payload.get("split_policy"),
            field_path="split_policy",
        ),
        forecast_object_type=_require_allowed_string(
            payload.get("forecast_object_type"),
            field_path="forecast_object_type",
            allowed=_FORECAST_OBJECT_TYPES,
        ),
        score_policy=_require_mapping(
            payload.get("score_policy"),
            field_path="score_policy",
        ),
        calibration_policy=_optional_mapping(
            payload.get("calibration_policy"),
            field_path="calibration_policy",
        ),
        budget_policy=_require_mapping(
            payload.get("budget_policy"),
            field_path="budget_policy",
        ),
        seed_policy=_require_mapping(
            payload.get("seed_policy"),
            field_path="seed_policy",
        ),
        replay_policy=_require_mapping(
            payload.get("replay_policy"),
            field_path="replay_policy",
        ),
    )

    if (
        frozen_protocol.forecast_object_type != "point"
        and frozen_protocol.calibration_policy is None
    ):
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message=(
                "calibration_policy is required for probabilistic forecast objects"
            ),
            field_path="calibration_policy",
        )

    search_class_payload = payload.get("search_class")
    search_class = (
        None
        if search_class_payload is None
        else _require_allowed_string(
            search_class_payload,
            field_path="search_class",
            allowed=_SEARCH_CLASS_IDS,
        )
    )
    search_class_honesty_payload = payload.get("search_class_honesty")
    if search_class_honesty_payload is None:
        search_class_honesty: dict[str, Any] = {}
    else:
        if search_class is None:
            _raise_validation_error(
                code="invalid_benchmark_manifest_field",
                message=(
                    "search_class_honesty requires an accompanying search_class"
                ),
                field_path="search_class_honesty",
            )
        search_class_honesty = _require_mapping(
            search_class_honesty_payload,
            field_path="search_class_honesty",
        )

    composition_operators_payload = payload.get("composition_operators")
    composition_operators = ()
    if composition_operators_payload is not None:
        composition_operators = tuple(
            _require_allowed_string(
                item,
                field_path=f"composition_operators[{index}]",
                allowed=_COMPOSITION_OPERATOR_IDS,
            )
            for index, item in enumerate(
                _require_string_list(
                    composition_operators_payload,
                    field_path="composition_operators",
                )
            )
        )

    task_id = _require_string(payload.get("task_id"), field_path="task_id")
    task_family = _require_string(payload.get("task_family"), field_path="task_family")
    submitter_registry = _require_registry(
        payload.get("submitter_registry"),
        field_path="submitter_registry",
        id_field="submitter_id",
    )
    submitter_ids = tuple(entry.entry_id for entry in submitter_registry)

    return {
        "task_id": task_id,
        "track_id": track_id,
        "task_family": task_family,
        "search_class": search_class,
        "search_class_honesty": search_class_honesty,
        "composition_operators": composition_operators,
        "regime_tags": _require_string_list(
            payload.get("regime_tags"), field_path="regime_tags"
        ),
        "generator_status": _require_allowed_string(
            payload.get("generator_status"),
            field_path="generator_status",
            allowed=_GENERATOR_STATUSES,
        ),
        "practical_significance_margin": _require_float(
            payload.get("practical_significance_margin"),
            field_path="practical_significance_margin",
        ),
        "adversarial_tags": _require_string_list(
            payload.get("adversarial_tags"), field_path="adversarial_tags"
        ),
        "forbidden_shortcuts": _require_string_list(
            payload.get("forbidden_shortcuts"),
            field_path="forbidden_shortcuts",
        ),
        "abstention_policy": _require_mapping(
            payload.get("abstention_policy"),
            field_path="abstention_policy",
        ),
        "baseline_registry": _require_registry(
            payload.get("baseline_registry"),
            field_path="baseline_registry",
            id_field="baseline_id",
        ),
        "submitter_registry": submitter_registry,
        "frozen_protocol": frozen_protocol,
        "metric_thresholds": _semantic_metric_thresholds(
            payload,
            frozen_protocol=frozen_protocol,
        ),
        "expected_claim_ceiling": _semantic_claim_ceiling(
            payload,
            track_id=track_id,
            forecast_object_type=frozen_protocol.forecast_object_type,
        ),
        "false_claim_expectations": _semantic_false_claim_expectations(
            payload,
            track_id=track_id,
        ),
        "engine_requirements": _semantic_engine_requirements(
            payload,
            submitter_ids=submitter_ids,
        ),
        "semantic_readiness_row_ids": _semantic_readiness_row_ids(
            payload,
            task_id=task_id,
            task_family=task_family,
        ),
        "fixture_spec_id": _optional_string(
            payload.get("fixture_spec_id"),
            field_path="fixture_spec_id",
        ),
        "fixture_family_id": _optional_string(
            payload.get("fixture_family_id"),
            field_path="fixture_family_id",
        ),
        "source_path": source_path,
    }


def _semantic_metric_thresholds(
    payload: Mapping[str, Any],
    *,
    frozen_protocol: FrozenBenchmarkProtocol,
) -> dict[str, Any]:
    explicit = _optional_mapping(
        payload.get("metric_thresholds"),
        field_path="metric_thresholds",
    )
    if explicit is not None:
        return explicit
    thresholds: dict[str, Any] = {
        "practical_significance_margin": {
            "metric_id": "practical_significance_margin",
            "comparator": ">=",
            "threshold": _require_float(
                payload.get("practical_significance_margin"),
                field_path="practical_significance_margin",
            ),
        }
    }
    calibration_policy = frozen_protocol.calibration_policy
    if isinstance(calibration_policy, Mapping):
        for key, value in sorted(calibration_policy.items()):
            if key == "required" or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                thresholds[f"calibration:{key}"] = {
                    "metric_id": key,
                    "comparator": "<=",
                    "threshold": float(value),
                }
    return thresholds


def _add_predictive_floor_threshold(
    thresholds: Mapping[str, Any],
    predictive_adequacy_floor: Mapping[str, Any],
) -> dict[str, Any]:
    updated = dict(thresholds)
    metric_id = predictive_adequacy_floor.get("metric_id")
    max_value = predictive_adequacy_floor.get("max_value")
    if isinstance(metric_id, str) and metric_id.strip() and isinstance(
        max_value,
        (int, float),
    ):
        updated["predictive_adequacy_floor"] = {
            "metric_id": metric_id.strip(),
            "comparator": "<=",
            "threshold": float(max_value),
        }
    return updated


def _semantic_claim_ceiling(
    payload: Mapping[str, Any],
    *,
    track_id: str,
    forecast_object_type: str,
) -> str:
    explicit = _optional_string(
        payload.get("expected_claim_ceiling"),
        field_path="expected_claim_ceiling",
    )
    if explicit is not None:
        return explicit
    expected_safe_outcome = payload.get("expected_safe_outcome")
    if track_id == "adversarial_honesty" or expected_safe_outcome == "abstain":
        return "abstention_only"
    if forecast_object_type == "point":
        return "descriptive_benchmark_only"
    return f"{forecast_object_type}_benchmark_only"


def _semantic_false_claim_expectations(
    payload: Mapping[str, Any],
    *,
    track_id: str,
) -> tuple[str, ...]:
    explicit = _optional_string_list(
        payload.get("false_claim_expectations"),
        field_path="false_claim_expectations",
    )
    if explicit:
        return explicit
    expected_safe_outcome = payload.get("expected_safe_outcome")
    adversarial_tags = payload.get("adversarial_tags")
    if (
        track_id == "adversarial_honesty"
        or expected_safe_outcome == "abstain"
        or (
            isinstance(adversarial_tags, list)
            and any("false" in str(tag) or "leak" in str(tag) for tag in adversarial_tags)
        )
    ):
        return ("no_false_publication",)
    return ("no_claim_promotion_from_benchmark_success",)


def _semantic_engine_requirements(
    payload: Mapping[str, Any],
    *,
    submitter_ids: tuple[str, ...],
) -> tuple[str, ...]:
    explicit = _optional_string_list(
        payload.get("engine_requirements"),
        field_path="engine_requirements",
    )
    return explicit or submitter_ids


def _semantic_readiness_row_ids(
    payload: Mapping[str, Any],
    *,
    task_id: str,
    task_family: str,
) -> tuple[str, ...]:
    explicit = _optional_string_list(
        payload.get("semantic_readiness_row_ids"),
        field_path="semantic_readiness_row_ids",
    )
    return explicit or (
        f"benchmark_task_semantics:{task_id}",
        f"benchmark_family_semantics:{task_family}",
    )


def load_benchmark_task_manifest(
    source: Path | str,
) -> BenchmarkTaskManifest:
    source_path = Path(source)
    with source_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        _raise_validation_error(
            code="invalid_benchmark_manifest_file",
            message="benchmark task files must deserialize to a mapping",
            field_path=str(source_path),
        )

    common_fields = _build_common_fields(payload, source_path=source_path)
    track_id = common_fields["track_id"]

    if track_id == "rediscovery":
        predictive_adequacy_floor = _require_mapping(
            payload.get("predictive_adequacy_floor"),
            field_path="predictive_adequacy_floor",
        )
        rediscovery_fields = {
            **common_fields,
            "metric_thresholds": _add_predictive_floor_threshold(
                common_fields["metric_thresholds"],
                predictive_adequacy_floor,
            ),
        }
        return RediscoveryTaskManifest(
            **rediscovery_fields,
            target_structure_ref=_require_string(
                payload.get("target_structure_ref"),
                field_path="target_structure_ref",
            ),
            equivalence_policy=_require_mapping(
                payload.get("equivalence_policy"),
                field_path="equivalence_policy",
            ),
            parameter_tolerance_policy=_optional_mapping(
                payload.get("parameter_tolerance_policy"),
                field_path="parameter_tolerance_policy",
            ),
            predictive_adequacy_floor=predictive_adequacy_floor,
        )

    if track_id == "predictive_generalization":
        return PredictiveGeneralizationTaskManifest(
            **common_fields,
            origin_policy=_require_mapping(
                payload.get("origin_policy"),
                field_path="origin_policy",
            ),
            horizon_policy=_require_mapping(
                payload.get("horizon_policy"),
                field_path="horizon_policy",
            ),
            baseline_comparison_policy=_require_mapping(
                payload.get("baseline_comparison_policy"),
                field_path="baseline_comparison_policy",
            ),
        )

    return AdversarialHonestyTaskManifest(
        **common_fields,
        trap_class=_require_string(payload.get("trap_class"), field_path="trap_class"),
        expected_safe_outcome=_require_string(
            payload.get("expected_safe_outcome"),
            field_path="expected_safe_outcome",
        ),
        failure_severity=_require_string(
            payload.get("failure_severity"),
            field_path="failure_severity",
        ),
    )


def load_benchmark_task_manifests(
    root: Path | str | Iterable[Path | str],
) -> tuple[BenchmarkTaskManifest, ...]:
    if isinstance(root, (str, Path)):
        root_path = Path(root)
        sources = (
            sorted(root_path.rglob("*.yaml")) if root_path.is_dir() else [root_path]
        )
    else:
        sources = sorted(Path(item) for item in root)

    return tuple(load_benchmark_task_manifest(source) for source in sources)


def load_benchmark_suite_manifest(
    source: Path | str,
) -> BenchmarkSuiteManifest:
    source_path = Path(source)
    with source_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        _raise_validation_error(
            code="invalid_benchmark_manifest_file",
            message="benchmark suite files must deserialize to a mapping",
            field_path=str(source_path),
        )

    task_manifest_paths = tuple(
        (
            source_path.parent.parent.parent
            / _require_string(
                item,
                field_path=f"task_manifest_paths[{index}]",
            )
        ).resolve()
        for index, item in enumerate(
            payload.get("task_manifest_paths", [])
            if isinstance(payload.get("task_manifest_paths"), list)
            else []
        )
    )
    if not task_manifest_paths:
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message="task_manifest_paths must not be empty",
            field_path="task_manifest_paths",
        )

    required_tracks = tuple(
        _require_allowed_string(
            track_id,
            field_path=f"required_tracks[{index}]",
            allowed=set(TRACK_IDS),
        )
        for index, track_id in enumerate(
            _require_string_list(
                payload.get("required_tracks"),
                field_path="required_tracks",
            )
        )
    )
    surface_requirements_payload = payload.get("surface_requirements")
    if (
        not isinstance(surface_requirements_payload, list)
        or not surface_requirements_payload
    ):
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message="surface_requirements must be a non-empty list",
            field_path="surface_requirements",
        )
    surface_requirements = tuple(
        BenchmarkSuiteSurfaceRequirement(
            surface_id=_require_string(
                _require_mapping(
                    item,
                    field_path=f"surface_requirements[{index}]",
                ).get("surface_id"),
                field_path=f"surface_requirements[{index}].surface_id",
            ),
            task_ids=_require_string_list(
                _require_mapping(
                    item,
                    field_path=f"surface_requirements[{index}]",
                ).get("task_ids"),
                field_path=f"surface_requirements[{index}].task_ids",
            ),
            replay_required=bool(
                _require_mapping(item, field_path=f"surface_requirements[{index}]").get(
                    "replay_required",
                    True,
                )
            ),
        )
        for index, item in enumerate(surface_requirements_payload)
    )

    suite_manifest = BenchmarkSuiteManifest(
        suite_id=_require_string(payload.get("suite_id"), field_path="suite_id"),
        description=_require_string(
            payload.get("description"), field_path="description"
        ),
        task_manifest_paths=task_manifest_paths,
        required_tracks=required_tracks,
        surface_requirements=surface_requirements,
        authority_snapshot_id=_optional_string(
            payload.get("authority_snapshot_id"),
            field_path="authority_snapshot_id",
        ),
        fixture_spec_id=_optional_string(
            payload.get("fixture_spec_id"),
            field_path="fixture_spec_id",
        ),
        source_path=source_path,
    )
    _validate_suite_manifest_contracts(suite_manifest)
    return suite_manifest


def _validate_suite_manifest_contracts(suite_manifest: BenchmarkSuiteManifest) -> None:
    task_manifests = tuple(
        load_benchmark_task_manifest(path)
        for path in suite_manifest.task_manifest_paths
    )
    task_manifest_by_id = {
        task_manifest.task_id: task_manifest for task_manifest in task_manifests
    }
    if len(task_manifest_by_id) != len(task_manifests):
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message="suite task ids must be unique",
            field_path="task_manifest_paths",
            details={
                "task_ids": [task_manifest.task_id for task_manifest in task_manifests]
            },
        )

    for requirement in suite_manifest.surface_requirements:
        missing_task_ids = [
            task_id
            for task_id in requirement.task_ids
            if task_id not in task_manifest_by_id
        ]
        if missing_task_ids:
            _raise_validation_error(
                code="invalid_benchmark_manifest_field",
                message=(
                    "surface requirements must reference declared suite task ids only"
                ),
                field_path=f"surface_requirements.{requirement.surface_id}.task_ids",
                details={
                    "surface_id": requirement.surface_id,
                    "missing_task_ids": missing_task_ids,
                },
            )
        if requirement.surface_id == "portfolio_orchestration":
            _validate_portfolio_surface_requirement(
                requirement=requirement,
                task_manifest_by_id=task_manifest_by_id,
            )

    if suite_manifest.suite_id == "full_vision":
        _validate_full_vision_suite(
            suite_manifest=suite_manifest,
            task_manifests=task_manifests,
        )
    if suite_manifest.suite_id == "current_release":
        _validate_current_release_suite(
            suite_manifest=suite_manifest,
            task_manifests=task_manifests,
        )


def _validate_portfolio_surface_requirement(
    *,
    requirement: BenchmarkSuiteSurfaceRequirement,
    task_manifest_by_id: Mapping[str, BenchmarkTaskManifest],
) -> None:
    matched_manifests = tuple(
        task_manifest_by_id[task_id] for task_id in requirement.task_ids
    )
    if not matched_manifests:
        _raise_validation_error(
            code="dishonest_portfolio_surface",
            message="portfolio_orchestration requires at least one portfolio task",
            field_path="surface_requirements",
            details={"surface_id": requirement.surface_id},
        )
    for task_manifest in matched_manifests:
        non_portfolio_submitter_ids = tuple(
            submitter_id
            for submitter_id in task_manifest.submitter_ids
            if submitter_id != _PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
        )
        if (
            task_manifest.task_family != "portfolio_selection_surface"
            or _PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID not in task_manifest.submitter_ids
            or len(non_portfolio_submitter_ids) < 3
        ):
            _raise_validation_error(
                code="dishonest_portfolio_surface",
                message=(
                    "portfolio_orchestration requires a dedicated portfolio "
                    "benchmark task with at least three real backend participants"
                ),
                field_path="surface_requirements",
                details={
                    "surface_id": requirement.surface_id,
                    "task_id": task_manifest.task_id,
                    "task_family": task_manifest.task_family,
                    "submitter_ids": list(task_manifest.submitter_ids),
                },
            )


def _validate_full_vision_suite(
    *,
    suite_manifest: BenchmarkSuiteManifest,
    task_manifests: tuple[BenchmarkTaskManifest, ...],
) -> None:
    if suite_manifest.authority_snapshot_id != _EXPECTED_AUTHORITY_SNAPSHOT_ID:
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message="full_vision suite must bind the frozen authority snapshot id",
            field_path="authority_snapshot_id",
            details={
                "expected": _EXPECTED_AUTHORITY_SNAPSHOT_ID,
                "actual": suite_manifest.authority_snapshot_id,
            },
        )
    if suite_manifest.fixture_spec_id != _EXPECTED_FIXTURE_SPEC_ID:
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message="full_vision suite must bind the frozen fixture spec id",
            field_path="fixture_spec_id",
            details={
                "expected": _EXPECTED_FIXTURE_SPEC_ID,
                "actual": suite_manifest.fixture_spec_id,
            },
        )
    surface_ids = {surface.surface_id for surface in suite_manifest.surface_requirements}
    if surface_ids != _FULL_VISION_REQUIRED_SURFACE_IDS:
        _raise_validation_error(
            code="dishonest_full_vision_suite",
            message="full_vision suite must declare every required capability surface",
            field_path="surface_requirements",
            details={
                "expected_surface_ids": sorted(_FULL_VISION_REQUIRED_SURFACE_IDS),
                "actual_surface_ids": sorted(surface_ids),
            },
        )
    forecast_object_types = {
        task_manifest.frozen_protocol.forecast_object_type
        for task_manifest in task_manifests
    }
    if forecast_object_types != _FORECAST_OBJECT_TYPES:
        _raise_validation_error(
            code="dishonest_full_vision_suite",
            message=(
                "full_vision suite must cover every admitted forecast object type"
            ),
            field_path="task_manifest_paths",
            details={
                "expected_forecast_object_types": sorted(_FORECAST_OBJECT_TYPES),
                "actual_forecast_object_types": sorted(forecast_object_types),
            },
        )
    if forecast_object_types == {"point"}:
        _raise_validation_error(
            code="dishonest_full_vision_suite",
            message="full_vision suite cannot fall back to point-only benchmark proof",
            field_path="task_manifest_paths",
        )
    all_case_tags = {
        tag
        for task_manifest in task_manifests
        for tag in (*task_manifest.regime_tags, *task_manifest.adversarial_tags)
    }
    if not _FULL_VISION_REQUIRED_CASE_TAGS <= all_case_tags:
        _raise_validation_error(
            code="dishonest_full_vision_suite",
            message=(
                "full_vision suite must include positive, negative, and abstention "
                "proof cases"
            ),
            field_path="task_manifest_paths",
            details={
                "required_case_tags": sorted(_FULL_VISION_REQUIRED_CASE_TAGS),
                "actual_case_tags": sorted(all_case_tags),
            },
        )
    for task_manifest in task_manifests:
        if task_manifest.fixture_spec_id != suite_manifest.fixture_spec_id:
            _raise_validation_error(
                code="dishonest_full_vision_suite",
                message=(
                    "full_vision benchmark tasks must bind the suite fixture spec id"
                ),
                field_path=f"task_manifest_paths.{task_manifest.task_id}.fixture_spec_id",
                details={
                    "task_id": task_manifest.task_id,
                    "expected_fixture_spec_id": suite_manifest.fixture_spec_id,
                    "actual_fixture_spec_id": task_manifest.fixture_spec_id,
                },
            )
        if not task_manifest.fixture_family_id:
            _raise_validation_error(
                code="dishonest_full_vision_suite",
                message=(
                    "full_vision benchmark tasks must declare a certification "
                    "fixture family"
                ),
                field_path=(
                    f"task_manifest_paths.{task_manifest.task_id}.fixture_family_id"
                ),
                details={"task_id": task_manifest.task_id},
            )
        if not task_manifest.dataset_ref.startswith(_CERTIFICATION_FIXTURE_ROOT_PREFIX):
            _raise_validation_error(
                code="dishonest_full_vision_suite",
                message=(
                    "full_vision benchmark tasks must default to certification "
                    "fixture corpora"
                ),
                field_path=f"task_manifest_paths.{task_manifest.task_id}.dataset_ref",
                details={
                    "task_id": task_manifest.task_id,
                    "dataset_ref": task_manifest.dataset_ref,
                },
            )


def _validate_current_release_suite(
    *,
    suite_manifest: BenchmarkSuiteManifest,
    task_manifests: tuple[BenchmarkTaskManifest, ...],
) -> None:
    if suite_manifest.authority_snapshot_id != _EXPECTED_AUTHORITY_SNAPSHOT_ID:
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message="current_release suite must bind the frozen authority snapshot id",
            field_path="authority_snapshot_id",
            details={
                "expected": _EXPECTED_AUTHORITY_SNAPSHOT_ID,
                "actual": suite_manifest.authority_snapshot_id,
            },
        )
    if suite_manifest.fixture_spec_id != _EXPECTED_FIXTURE_SPEC_ID:
        _raise_validation_error(
            code="invalid_benchmark_manifest_field",
            message="current_release suite must bind the frozen fixture spec id",
            field_path="fixture_spec_id",
            details={
                "expected": _EXPECTED_FIXTURE_SPEC_ID,
                "actual": suite_manifest.fixture_spec_id,
            },
        )
    surface_ids = {surface.surface_id for surface in suite_manifest.surface_requirements}
    if _FULL_VISION_REQUIRED_SURFACE_IDS <= surface_ids or (
        "portfolio_orchestration" in surface_ids
    ):
        _raise_validation_error(
            code="dishonest_current_release_suite",
            message=(
                "current_release must remain a truthful subset and cannot present "
                "portfolio proof as current-release closure"
            ),
            field_path="surface_requirements",
            details={"surface_ids": sorted(surface_ids)},
        )
    for task_manifest in task_manifests:
        if (
            task_manifest.fixture_spec_id is not None
            and task_manifest.fixture_spec_id != suite_manifest.fixture_spec_id
        ):
            _raise_validation_error(
                code="dishonest_current_release_suite",
                message=(
                    "current_release tasks with fixture metadata must agree with the "
                    "suite fixture spec id"
                ),
                field_path=f"task_manifest_paths.{task_manifest.task_id}.fixture_spec_id",
                details={
                    "task_id": task_manifest.task_id,
                    "expected_fixture_spec_id": suite_manifest.fixture_spec_id,
                    "actual_fixture_spec_id": task_manifest.fixture_spec_id,
                },
            )
