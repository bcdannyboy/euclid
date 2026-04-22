from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import yaml

from euclid.artifacts import FilesystemArtifactStore
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog, load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.control_plane import (
    FileLock,
    RuntimeWorkspace,
    RunWorkspacePaths,
    SQLiteExecutionStateStore,
    SQLiteMetadataStore,
)
from euclid.manifest_registry import ManifestRegistry
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    AbstentionManifest,
    ReadinessGateRecord,
    ReadinessJudgmentManifest,
    SchemaLifecycleIntegrationClosureManifest,
)
from euclid.math.observation_models import PointObservationModel
from euclid.modules.calibration import (
    build_calibration_contract,
    evaluate_prediction_calibration,
)
from euclid.modules.candidate_fitting import (
    build_candidate_fit_artifacts,
    fit_candidate_window,
)
from euclid.modules.catalog_publishing import (
    build_publication_record_manifest,
    build_run_result_manifest,
)
from euclid.modules.claims import resolve_claim_publication
from euclid.modules.evaluation import emit_point_prediction_artifact
from euclid.modules.evaluation_governance import (
    BaselineRegistry,
    ComparatorDeclaration,
    build_comparison_key,
    build_comparison_universe,
    build_evaluation_event_log,
    build_evaluation_governance,
    build_forecast_comparison_policy,
    build_predictive_gate_policy,
    resolve_confirmatory_promotion_allowed,
)
from euclid.modules.gate_lifecycle import resolve_scorecard_status
from euclid.modules.probabilistic_evaluation import (
    emit_probabilistic_prediction_artifact,
)
from euclid.modules.predictive_tests import evaluate_predictive_promotion
from euclid.modules.replay import (
    ReplayedOutcome,
    build_artifact_hash_records,
    build_replay_seed_records,
    build_replay_stage_order,
    build_reproducibility_bundle_manifest,
    inspect_reproducibility_bundle,
    required_manifest_refs_for_publication,
    resolve_primary_score_result,
    resolve_scorecard,
    selected_candidate_ref,
    verify_replayed_outcome,
)
from euclid.modules.robustness import (
    AggregateMetricEvaluation,
    NullComparisonEvaluation,
    PerturbationFamilyEvaluation,
    PerturbationMetricEvaluation,
    build_leakage_canary_manifest,
    build_leakage_canary_result_manifest,
    build_null_protocol_manifest,
    build_null_result_manifest,
    build_perturbation_family_result_manifest,
    build_perturbation_protocol_manifest,
    build_retention_metric_manifest,
    build_robustness_report,
    build_sensitivity_analysis_manifest,
    build_surrogate_generator_manifest,
)
from euclid.modules.scoring import (
    evaluate_point_comparators,
    score_probabilistic_prediction_artifact,
)
from euclid.modules.search_planning import (
    SearchCandidateRecord,
    build_freeze_event,
    build_frontier,
    build_frozen_shortlist,
    build_rejected_diagnostics,
    build_search_ledger,
)
from euclid.operator_runtime.models import (
    DEFAULT_ADMISSIBILITY_RULE_IDS,
    DEFAULT_RUN_SUPPORT_OBJECT_IDS,
)
from euclid.operator_runtime.resources import (
    default_run_output_root,
    resolve_asset_root,
    resolve_fixture_path,
)
from euclid.operator_runtime.workflow import replay_operator_run
from euclid.performance import (
    PerformanceTelemetryArtifact,
    TelemetryRecorder,
    write_performance_telemetry,
)
from euclid.prototype.workflow import (
    CandidateSummary,
    PrototypeReducerWorkflowResult,
    PrototypeReplayResult,
    replay_prototype_run,
    run_prototype_reducer_workflow,
)
from euclid.readiness import ReadinessGateResult, judge_readiness
from euclid.reducers.models import BoundObservationModel
from euclid.search import run_descriptive_search_backends
from euclid.search.backends import (
    AlgorithmicSearchBackendAdapter,
    AnalyticSearchBackendAdapter,
    DescriptiveSearchProposal,
    RecursiveSearchBackendAdapter,
    SpectralSearchBackendAdapter,
)

if TYPE_CHECKING:
    from euclid.inspection import (
        BaselineComparisonInspection,
        CalibrationInspection,
        PointPredictionInspection,
        ProbabilisticPredictionInspection,
    )

_DEFAULT_SEARCH_FAMILY_IDS = (
    "constant",
    "drift",
    "linear_trend",
    "seasonal_naive",
)
_PROBABILISTIC_SCORE_POLICY_SCHEMAS = {
    "distribution": "probabilistic_score_policy_manifest@1.0.0",
    "interval": "interval_score_policy_manifest@1.0.0",
    "quantile": "quantile_score_policy_manifest@1.0.0",
    "event_probability": "event_probability_score_policy_manifest@1.0.0",
}
_PROBABILISTIC_PRIMARY_SCORES = {
    "distribution": "continuous_ranked_probability_score",
    "interval": "interval_score",
    "quantile": "pinball_loss",
    "event_probability": "brier_score",
}
_PROBABILISTIC_BASELINE_FALLBACK_CANDIDATE_IDS = (
    "analytic_intercept",
    "recursive_running_mean",
    "recursive_level_smoother",
    "algorithmic_last_observation",
    "spectral_harmonic_1",
)


@dataclass(frozen=True)
class DemoRequest:
    request_id: str
    manifest_path: Path
    dataset_csv: Path
    cutoff_available_at: str | None
    quantization_step: str
    minimum_description_gain_bits: float
    min_train_size: int = 3
    horizon: int = 1
    search_family_ids: tuple[str, ...] = _DEFAULT_SEARCH_FAMILY_IDS
    search_class: str = "bounded_heuristic"
    search_seed: str = "0"
    proposal_limit: int | None = None
    seasonal_period: int = 2
    forecast_object_type: str = "point"
    primary_score_id: str | None = None
    calibration_thresholds: Mapping[str, float] | None = None
    run_support_object_ids: tuple[str, ...] = DEFAULT_RUN_SUPPORT_OBJECT_IDS
    admissibility_rule_ids: tuple[str, ...] = DEFAULT_ADMISSIBILITY_RULE_IDS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "calibration_thresholds",
            dict(self.calibration_thresholds or {}),
        )


@dataclass(frozen=True)
class DemoPaths:
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


@dataclass(frozen=True)
class DemoRunSummary:
    selected_family: str
    result_mode: str
    bundle_ref: TypedRef
    run_result_ref: TypedRef
    selected_candidate_ref: TypedRef
    confirmatory_primary_score: float


@dataclass(frozen=True)
class DemoRunResult:
    request: DemoRequest
    paths: DemoPaths
    summary: DemoRunSummary
    workflow_result: PrototypeReducerWorkflowResult


@dataclass(frozen=True)
class DemoPointEvaluationResult:
    run: DemoRunResult
    prediction: PointPredictionInspection
    comparison: BaselineComparisonInspection


@dataclass(frozen=True)
class DemoAlgorithmicSearchSummary:
    selected_candidate_id: str
    selected_family: str
    accepted_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    search_class: str
    coverage_statement: str
    exactness_ceiling: str
    scope_declaration: str


@dataclass(frozen=True)
class DemoAlgorithmicSearchResult:
    request: DemoRequest
    paths: DemoPaths
    summary: DemoAlgorithmicSearchSummary


@dataclass(frozen=True)
class DemoProbabilisticRunSummary:
    selected_candidate_id: str
    selected_family: str
    forecast_object_type: str
    bundle_ref: TypedRef
    run_result_ref: TypedRef
    publication_record_ref: TypedRef
    comparison_universe_ref: TypedRef
    scorecard_ref: TypedRef
    claim_card_ref: TypedRef | None
    abstention_ref: TypedRef | None
    prediction_artifact_ref: TypedRef
    score_result_ref: TypedRef
    calibration_result_ref: TypedRef
    aggregated_primary_score: float
    calibration_status: str


@dataclass(frozen=True)
class DemoProbabilisticEvaluationResult:
    request: DemoRequest
    paths: DemoPaths
    summary: DemoProbabilisticRunSummary
    prediction: ProbabilisticPredictionInspection
    calibration: CalibrationInspection


@dataclass(frozen=True)
class DemoReplaySummary:
    bundle_ref: TypedRef
    run_result_ref: TypedRef
    selected_candidate_ref: TypedRef
    selected_family: str
    result_mode: str
    replay_verification_status: str
    confirmatory_primary_score: float
    failure_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DemoReplayResult:
    paths: DemoPaths
    summary: DemoReplaySummary
    replay_result: PrototypeReplayResult


@dataclass(frozen=True)
class ProfiledDemoRunResult:
    run: DemoRunResult
    telemetry: PerformanceTelemetryArtifact
    telemetry_path: Path


def default_demo_manifest_path() -> Path:
    return resolve_fixture_path("runtime", "prototype-demo.yaml")


def default_probabilistic_demo_manifest_path() -> Path:
    return resolve_fixture_path(
        "runtime",
        "phase06",
        "probabilistic-distribution-demo.yaml",
    )


def default_algorithmic_demo_manifest_path() -> Path:
    return resolve_fixture_path(
        "runtime",
        "phase06",
        "algorithmic-search-demo.yaml",
    )


def load_demo_request(manifest_path: Path | None = None) -> DemoRequest:
    resolved_manifest_path = (manifest_path or default_demo_manifest_path()).resolve()
    payload = yaml.safe_load(resolved_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("demo manifest must deserialize to a mapping")

    request_id = payload.get("request_id")
    dataset_csv = payload.get("dataset_csv")

    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("demo manifest requires a non-empty request_id")
    if not isinstance(dataset_csv, str) or not dataset_csv.strip():
        raise ValueError("demo manifest requires a dataset_csv path")

    cutoff_available_at = payload.get("cutoff_available_at")
    if cutoff_available_at is not None and not isinstance(cutoff_available_at, str):
        raise ValueError("cutoff_available_at must be a string or null")

    quantization_step = payload.get("quantization_step", "0.5")
    if not isinstance(quantization_step, str) or not quantization_step.strip():
        raise ValueError("quantization_step must be a non-empty string")

    minimum_description_gain_bits = payload.get("minimum_description_gain_bits", 0.0)
    if not isinstance(minimum_description_gain_bits, int | float):
        raise ValueError("minimum_description_gain_bits must be numeric")

    min_train_size = payload.get("min_train_size", 3)
    if not isinstance(min_train_size, int) or min_train_size < 1:
        raise ValueError("min_train_size must be a positive integer")

    horizon = payload.get("horizon", 1)
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")

    search_payload = _mapping_or_empty(payload.get("search"), field_path="search")
    search_family_ids = _tuple_of_strings(
        search_payload.get(
            "family_ids", payload.get("search_family_ids", _DEFAULT_SEARCH_FAMILY_IDS)
        ),
        field_path="search.family_ids",
    )
    search_class = _string_field(
        search_payload.get("class", payload.get("search_class", "bounded_heuristic")),
        field_path="search.class",
    )
    search_seed = _string_field(
        search_payload.get("seed", payload.get("search_seed", "0")),
        field_path="search.seed",
    )
    proposal_limit = _optional_positive_int(
        search_payload.get("proposal_limit", payload.get("proposal_limit")),
        field_path="search.proposal_limit",
    )
    seasonal_period = payload.get(
        "seasonal_period",
        search_payload.get("seasonal_period", 2),
    )
    if not isinstance(seasonal_period, int) or seasonal_period < 1:
        raise ValueError("seasonal_period must be a positive integer")

    probabilistic_payload = _mapping_or_empty(
        payload.get("probabilistic"),
        field_path="probabilistic",
    )
    forecast_object_type = _string_field(
        probabilistic_payload.get(
            "forecast_object_type",
            payload.get("forecast_object_type", "point"),
        ),
        field_path="probabilistic.forecast_object_type",
    )
    primary_score_id = probabilistic_payload.get(
        "primary_score",
        payload.get("primary_score_id"),
    )
    if primary_score_id is not None:
        primary_score_id = _string_field(
            primary_score_id,
            field_path="probabilistic.primary_score",
        )
    calibration_thresholds = _float_mapping(
        probabilistic_payload.get(
            "calibration_thresholds",
            payload.get("calibration_thresholds", {}),
        ),
        field_path="probabilistic.calibration_thresholds",
    )

    return DemoRequest(
        request_id=request_id,
        manifest_path=resolved_manifest_path,
        dataset_csv=(resolved_manifest_path.parent / dataset_csv).resolve(),
        cutoff_available_at=cutoff_available_at,
        quantization_step=quantization_step,
        minimum_description_gain_bits=float(minimum_description_gain_bits),
        min_train_size=min_train_size,
        horizon=horizon,
        search_family_ids=search_family_ids,
        search_class=search_class,
        search_seed=search_seed,
        proposal_limit=proposal_limit,
        seasonal_period=seasonal_period,
        forecast_object_type=forecast_object_type,
        primary_score_id=primary_score_id,
        calibration_thresholds=calibration_thresholds,
    )


def _uses_shared_local_operator_workflow(request: DemoRequest) -> bool:
    return (
        request.forecast_object_type == "point"
        and "shared_plus_local_decomposition" in request.search_family_ids
    )


def run_demo(
    *,
    manifest_path: Path | None = None,
    output_root: Path | None = None,
    telemetry: TelemetryRecorder | None = None,
) -> DemoRunResult:
    request = load_demo_request(manifest_path)
    paths = _demo_paths(
        request=request,
        output_root=output_root,
        reset_existing_run=True,
    )
    catalog = _load_catalog()
    execution_state = SQLiteExecutionStateStore(paths.control_plane_store_path)

    with FileLock(paths.run_lock_path):
        execution_state.save_step_state(
            run_id=request.request_id,
            step_id="demo.run.workflow",
            module_id="manifest_registry",
            status="running",
            cursor="run_declared",
            details={
                "manifest_path": str(request.manifest_path),
                "dataset_csv": str(request.dataset_csv),
            },
        )
        execution_state.upsert_worker_metadata(
            run_id=request.request_id,
            worker_id="prototype.reducer.workflow",
            module_id="manifest_registry",
            status="running",
            details={"entrypoint_id": "demo.run"},
        )
        registry = _build_registry(catalog=catalog, paths=paths, telemetry=telemetry)
        try:
            if _uses_shared_local_operator_workflow(request):
                workflow_result = _run_shared_local_point_workflow(
                    request=request,
                    catalog=catalog,
                    registry=registry,
                )
            else:
                workflow_result = run_prototype_reducer_workflow(
                    csv_path=request.dataset_csv,
                    catalog=catalog,
                    registry=registry,
                    cutoff_available_at=request.cutoff_available_at,
                    quantization_step=request.quantization_step,
                    min_train_size=request.min_train_size,
                    horizon=request.horizon,
                    search_family_ids=request.search_family_ids,
                    search_class=request.search_class,
                    search_seed=request.search_seed,
                    proposal_limit=request.proposal_limit,
                    minimum_description_gain_bits=request.minimum_description_gain_bits,
                    seasonal_period=request.seasonal_period,
                    telemetry=telemetry,
                )
        except Exception as exc:
            execution_state.save_step_state(
                run_id=request.request_id,
                step_id="demo.run.workflow",
                module_id="catalog_publishing",
                status="failed",
                cursor="failed",
                details={"error": str(exc)},
            )
            execution_state.upsert_worker_metadata(
                run_id=request.request_id,
                worker_id="prototype.reducer.workflow",
                module_id="catalog_publishing",
                status="failed",
                details={"error": str(exc)},
            )
            raise

        summary = DemoRunSummary(
            selected_family=str(
                workflow_result.selected_candidate.manifest.body["family_id"]
            ),
            result_mode=str(workflow_result.run_result.manifest.body["result_mode"]),
            bundle_ref=workflow_result.reproducibility_bundle.manifest.ref,
            run_result_ref=workflow_result.run_result.manifest.ref,
            selected_candidate_ref=workflow_result.selected_candidate.manifest.ref,
            confirmatory_primary_score=workflow_result.confirmatory_primary_score,
        )
        _record_control_plane_state(
            request=request,
            execution_state=execution_state,
            workflow_result=workflow_result,
            summary=summary,
        )
        _write_run_summary(request=request, paths=paths, summary=summary)
        return DemoRunResult(
            request=request,
            paths=paths,
            summary=summary,
            workflow_result=workflow_result,
        )


def profile_demo_run(
    *,
    manifest_path: Path | None = None,
    output_root: Path | None = None,
) -> ProfiledDemoRunResult:
    request = load_demo_request(manifest_path)
    telemetry = TelemetryRecorder()
    with telemetry.span(
        "demo.run.profile",
        category="run",
        attributes={"request_id": request.request_id},
    ):
        result = run_demo(
            manifest_path=request.manifest_path,
            output_root=output_root,
            telemetry=telemetry,
        )
    artifact = telemetry.build_artifact(
        profile_kind="demo_run",
        subject_id=result.request.request_id,
        attributes={
            "manifest_path": str(result.request.manifest_path),
            "output_root": str(result.paths.output_root),
        },
    )
    telemetry_path = write_performance_telemetry(
        result.paths.output_root / "telemetry" / "demo-run-profile.json",
        artifact,
    )
    return ProfiledDemoRunResult(
        run=result,
        telemetry=artifact,
        telemetry_path=telemetry_path,
    )


def replay_demo(
    *,
    output_root: Path | None = None,
    run_id: str | None = None,
    bundle_ref: str | TypedRef | None = None,
) -> DemoReplayResult:
    paths, summary_payload = _resolve_replay_paths_and_summary(
        output_root=output_root,
        run_id=run_id,
    )
    catalog = _load_catalog()
    registry = _build_registry(catalog=catalog, paths=paths)
    effective_bundle_ref = _coerce_typed_ref(
        bundle_ref if bundle_ref is not None else summary_payload["bundle_ref"]
    )
    replay_result = _replay_registered_run(
        bundle_ref=effective_bundle_ref,
        catalog=catalog,
        registry=registry,
    )
    summary = DemoReplaySummary(
        bundle_ref=effective_bundle_ref,
        run_result_ref=replay_result.run_result_ref,
        selected_candidate_ref=replay_result.selected_candidate_ref,
        selected_family=str(summary_payload["selected_family"]),
        result_mode=str(summary_payload["result_mode"]),
        replay_verification_status=replay_result.replay_verification_status,
        confirmatory_primary_score=replay_result.confirmatory_primary_score,
        failure_reason_codes=replay_result.failure_reason_codes,
    )
    return DemoReplayResult(paths=paths, summary=summary, replay_result=replay_result)


def _replay_registered_run(
    *,
    bundle_ref: TypedRef,
    catalog: ContractCatalog,
    registry: ManifestRegistry,
) -> PrototypeReplayResult:
    bundle = registry.resolve(bundle_ref)
    run_result = registry.resolve(
        _coerce_typed_ref(bundle.manifest.body["run_result_ref"])
    )
    if (
        str(bundle.manifest.body.get("replay_entrypoint_id", ""))
        == "operator_native_replay_v1"
    ):
        return replay_operator_run(
            bundle_ref=bundle_ref,
            catalog=catalog,
            registry=registry,
        )
    forecast_object_type = str(
        run_result.manifest.body.get("forecast_object_type", "point")
    )
    if forecast_object_type == "point" and not isinstance(
        run_result.manifest.body.get("primary_score_result_ref"), Mapping
    ):
        return replay_prototype_run(
            bundle_ref=bundle_ref,
            catalog=catalog,
            registry=registry,
        )

    scorecard = resolve_scorecard(run_result.manifest.body, registry)
    primary_score_result = resolve_primary_score_result(run_result, registry)
    selected_ref = selected_candidate_ref(run_result.manifest.body, registry)
    selected_candidate = registry.resolve(selected_ref)
    bundle_inspection = inspect_reproducibility_bundle(bundle)
    verification = verify_replayed_outcome(
        bundle=bundle,
        registry=registry,
        outcome=ReplayedOutcome(
            selected_candidate_id=str(selected_candidate.manifest.body["candidate_id"]),
            confirmatory_primary_score=float(
                primary_score_result.manifest.body["aggregated_primary_score"]
            ),
            publication_mode=str(run_result.manifest.body["result_mode"]),
            descriptive_status=str(scorecard.manifest.body["descriptive_status"]),
            descriptive_reason_codes=tuple(
                str(item)
                for item in scorecard.manifest.body.get("descriptive_reason_codes", ())
            ),
            predictive_status=str(scorecard.manifest.body["predictive_status"]),
            predictive_reason_codes=tuple(
                str(item)
                for item in scorecard.manifest.body.get("predictive_reason_codes", ())
            ),
            replayed_stage_order=bundle_inspection.recorded_stage_order,
            mechanistic_status=str(
                scorecard.manifest.body.get("mechanistic_status", "not_requested")
            ),
            mechanistic_reason_codes=tuple(
                str(item)
                for item in scorecard.manifest.body.get(
                    "mechanistic_reason_codes",
                    (),
                )
            ),
        ),
    )
    return PrototypeReplayResult(
        bundle_ref=verification.bundle_ref,
        run_result_ref=verification.run_result_ref,
        selected_candidate_ref=verification.selected_candidate_ref,
        replay_verification_status=verification.replay_verification_status,
        confirmatory_primary_score=verification.confirmatory_primary_score,
        failure_reason_codes=verification.failure_reason_codes,
    )


def run_demo_point_evaluation(
    *,
    manifest_path: Path | None = None,
    output_root: Path | None = None,
) -> DemoPointEvaluationResult:
    run_result = run_demo(manifest_path=manifest_path, output_root=output_root)
    from euclid.inspection import (
        compare_demo_baseline,
        inspect_demo_point_prediction,
    )

    prediction = inspect_demo_point_prediction(
        output_root=run_result.paths.output_root,
        run_id=run_result.summary.run_result_ref.object_id,
    )
    comparison = compare_demo_baseline(
        output_root=run_result.paths.output_root,
        run_id=run_result.summary.run_result_ref.object_id,
    )
    return DemoPointEvaluationResult(
        run=run_result,
        prediction=prediction,
        comparison=comparison,
    )


def run_demo_algorithmic_search(
    *,
    manifest_path: Path | None = None,
    output_root: Path | None = None,
) -> DemoAlgorithmicSearchResult:
    request = load_demo_request(
        manifest_path or default_algorithmic_demo_manifest_path()
    )
    if not request.search_family_ids or not all(
        family_id.startswith("algorithmic_") for family_id in request.search_family_ids
    ):
        raise ContractValidationError(
            code="invalid_algorithmic_search_request",
            message=(
                "algorithmic workflow surfaces require algorithmic candidate ids only"
            ),
            field_path="search_family_ids",
            details={"search_family_ids": list(request.search_family_ids)},
        )

    paths = _demo_paths(
        request=request,
        output_root=output_root,
        reset_existing_run=True,
    )
    catalog = _load_catalog()

    with FileLock(paths.run_lock_path):
        registry = _build_registry(catalog=catalog, paths=paths)
        intake = _build_search_intake(
            request=request,
            catalog=catalog,
            registry=registry,
        )
        search_result = run_descriptive_search_backends(
            search_plan=intake.search_plan_object,
            feature_view=intake.feature_view_object,
        )
        selected_candidate = _select_frontier_candidate(search_result)
        summary = DemoAlgorithmicSearchSummary(
            selected_candidate_id=(
                selected_candidate.evidence_layer.backend_origin_record.source_candidate_id
            ),
            selected_family=selected_candidate.structural_layer.cir_family_id,
            accepted_candidate_ids=tuple(
                candidate.evidence_layer.backend_origin_record.source_candidate_id
                for candidate in search_result.accepted_candidates
            ),
            rejected_candidate_ids=tuple(
                diagnostic.candidate_id
                for diagnostic in search_result.rejected_diagnostics
            ),
            search_class=search_result.coverage.search_class,
            coverage_statement=search_result.coverage.coverage_statement,
            exactness_ceiling=search_result.coverage.exactness_ceiling,
            scope_declaration=search_result.coverage.scope_declaration,
        )
        _write_summary_payload(
            paths=paths,
            payload={
                "workflow_surface": "algorithmic_search",
                "request_id": request.request_id,
                "manifest_path": str(request.manifest_path),
                "dataset_csv": str(request.dataset_csv),
                "search_plan_ref": intake.search_plan.manifest.ref.as_dict(),
                "selected_candidate_id": summary.selected_candidate_id,
                "selected_family": summary.selected_family,
                "accepted_candidate_ids": list(summary.accepted_candidate_ids),
                "rejected_candidate_ids": list(summary.rejected_candidate_ids),
                "search_class": summary.search_class,
                "coverage_statement": summary.coverage_statement,
                "exactness_ceiling": summary.exactness_ceiling,
                "scope_declaration": summary.scope_declaration,
            },
        )
        return DemoAlgorithmicSearchResult(
            request=request,
            paths=paths,
            summary=summary,
        )


def run_demo_probabilistic_evaluation(
    *,
    manifest_path: Path | None = None,
    output_root: Path | None = None,
) -> DemoProbabilisticEvaluationResult:
    request = load_demo_request(
        manifest_path or default_probabilistic_demo_manifest_path()
    )
    if request.forecast_object_type == "point":
        raise ContractValidationError(
            code="invalid_probabilistic_demo_request",
            message=(
                "probabilistic workflow surfaces require a non-point "
                "forecast_object_type"
            ),
            field_path="forecast_object_type",
        )

    paths = _demo_paths(
        request=request,
        output_root=output_root,
        reset_existing_run=True,
    )
    catalog = _load_catalog()

    with FileLock(paths.run_lock_path):
        registry = _build_registry(catalog=catalog, paths=paths)
        intake = _build_search_intake(
            request=request,
            catalog=catalog,
            registry=registry,
        )
        search_result = run_descriptive_search_backends(
            search_plan=intake.search_plan_object,
            feature_view=intake.feature_view_object,
        )
        selected_candidate = _select_frontier_candidate(search_result)
        selected_candidate_id = _candidate_source_id(selected_candidate)
        selected_candidate_admissible, descriptive_failure_reason_codes = (
            _selected_candidate_law_semantics(
                search_result=search_result,
                selected_candidate=selected_candidate,
            )
        )
        baseline_candidate = _select_probabilistic_baseline_candidate(
            search_result=search_result,
            selected_candidate_id=selected_candidate_id,
            feature_view=intake.feature_view_object,
            search_plan=intake.search_plan_object,
        )
        fit_result = fit_candidate_window(
            candidate=selected_candidate,
            feature_view=intake.feature_view_object,
            fit_window=intake.evaluation_plan_object.confirmatory_segment,
            search_plan=intake.search_plan_object,
            stage_id="confirmatory_holdout",
        )
        baseline_fit_result = fit_candidate_window(
            candidate=baseline_candidate,
            feature_view=intake.feature_view_object,
            fit_window=intake.evaluation_plan_object.confirmatory_segment,
            search_plan=intake.search_plan_object,
            stage_id="confirmatory_holdout",
        )
        description_gain_bits = _description_gain_for_candidate(
            search_result=search_result,
            candidate=selected_candidate,
        )
        baseline_description_gain_bits = _description_gain_for_candidate(
            search_result=search_result,
            candidate=baseline_candidate,
        )
        fit_artifacts = build_candidate_fit_artifacts(
            catalog=catalog,
            fit_result=fit_result,
            search_plan_ref=intake.search_plan.manifest.ref,
            selection_floor_bits=request.minimum_description_gain_bits,
            description_gain_bits=description_gain_bits,
        )
        baseline_fit_artifacts = build_candidate_fit_artifacts(
            catalog=catalog,
            fit_result=baseline_fit_result,
            search_plan_ref=intake.search_plan.manifest.ref,
            selection_floor_bits=request.minimum_description_gain_bits,
            description_gain_bits=baseline_description_gain_bits,
        )
        candidate_spec = registry.register(
            fit_artifacts.candidate_spec,
            parent_refs=(intake.search_plan.manifest.ref,),
        )
        candidate_state = registry.register(
            fit_artifacts.candidate_state,
            parent_refs=(intake.search_plan.manifest.ref, candidate_spec.manifest.ref),
        )
        reducer_artifact = registry.register(
            fit_artifacts.reducer_artifact,
            parent_refs=(
                intake.search_plan.manifest.ref,
                candidate_spec.manifest.ref,
                candidate_state.manifest.ref,
            ),
        )
        baseline_candidate_spec = registry.register(
            baseline_fit_artifacts.candidate_spec,
            parent_refs=(intake.search_plan.manifest.ref,),
        )
        baseline_candidate_state = registry.register(
            baseline_fit_artifacts.candidate_state,
            parent_refs=(
                intake.search_plan.manifest.ref,
                baseline_candidate_spec.manifest.ref,
            ),
        )
        baseline_reducer_artifact = registry.register(
            baseline_fit_artifacts.reducer_artifact,
            parent_refs=(
                intake.search_plan.manifest.ref,
                baseline_candidate_spec.manifest.ref,
                baseline_candidate_state.manifest.ref,
            ),
        )
        score_policy = registry.register(
            _build_probabilistic_score_policy_manifest(
                catalog=catalog,
                request=request,
                horizon_set=intake.evaluation_plan_object.horizon_set,
            )
        )
        scope_ledger = registry.register(
            ManifestEnvelope.build(
                schema_name="scope_ledger_manifest@1.0.0",
                module_id="manifest_registry",
                body={
                    "scope_ledger_id": f"{request.request_id}_scope_ledger",
                    "scope_id": "euclid_v1_binding_scope@1.0.0",
                    "forecast_object_type": request.forecast_object_type,
                    "candidate_family_ids": list(request.search_family_ids),
                    "run_support_object_ids": list(request.run_support_object_ids),
                    "admissibility_rule_ids": list(request.admissibility_rule_ids),
                    "deferred_scope_annotations": [
                        "shared_plus_local_decomposition",
                        "mechanistic_evidence",
                        "algorithmic_publication",
                    ],
                },
                catalog=catalog,
            )
        )
        best_overall_candidate = getattr(search_result, "best_overall_candidate", None)
        if best_overall_candidate is None:
            raise ContractValidationError(
                code="missing_best_overall_candidate",
                message=(
                    "runtime ledger reconstruction requires an explicit "
                    "best_overall_candidate from descriptive_scope"
                ),
                field_path="search_result.best_overall_candidate",
            )
        descriptive_candidate_id = _candidate_source_id(best_overall_candidate)
        accepted_candidate_id = (
            _candidate_source_id(search_result.accepted_candidate)
            if getattr(search_result, "accepted_candidate", None) is not None
            else None
        )
        search_candidate_records = _build_probabilistic_search_candidate_records(
            search_result=search_result,
            descriptive_candidate_id=descriptive_candidate_id,
        )
        search_ledger = registry.register(
            build_search_ledger(
                search_plan=intake.search_plan_object,
                candidate_records=search_candidate_records,
                selected_candidate_id=descriptive_candidate_id,
                accepted_candidate_id=accepted_candidate_id,
            ).to_manifest(catalog),
            parent_refs=(intake.search_plan.manifest.ref,),
        )
        frontier = registry.register(
            build_frontier(
                search_plan=intake.search_plan_object,
                candidate_records=search_candidate_records,
            ).to_manifest(catalog),
            parent_refs=(search_ledger.manifest.ref,),
        )
        rejected_diagnostics = registry.register(
            build_rejected_diagnostics(
                candidate_records=search_candidate_records,
            ).to_manifest(catalog),
            parent_refs=(search_ledger.manifest.ref,),
        )
        frozen_shortlist = registry.register(
            build_frozen_shortlist(
                search_plan_ref=intake.search_plan.manifest.ref,
                candidate_ref=reducer_artifact.manifest.ref,
            ).to_manifest(catalog),
            parent_refs=(
                intake.search_plan.manifest.ref,
                frontier.manifest.ref,
                rejected_diagnostics.manifest.ref,
                reducer_artifact.manifest.ref,
            ),
        )
        baseline_id = _candidate_source_id(baseline_candidate)
        freeze_event = registry.register(
            build_freeze_event(
                frozen_candidate_ref=reducer_artifact.manifest.ref,
                frozen_shortlist_ref=frozen_shortlist.manifest.ref,
                confirmatory_baseline_id=baseline_id,
            ).to_manifest(catalog),
            parent_refs=(frozen_shortlist.manifest.ref,),
        )
        baseline_registry = registry.register(
            BaselineRegistry(
                compatible_point_score_policy_ref=score_policy.manifest.ref,
                declarations=(
                    ComparatorDeclaration(
                        comparator_declaration_id=f"{baseline_id}_declaration",
                        baseline_id=baseline_id,
                        comparator_kind="baseline",
                        forecast_object_type=request.forecast_object_type,
                        family_id=_candidate_family_id(baseline_candidate),
                        freeze_rule="frozen_before_confirmatory_access",
                    ),
                ),
            ).to_manifest(catalog),
            parent_refs=(score_policy.manifest.ref,),
        )
        forecast_comparison_policy = registry.register(
            build_forecast_comparison_policy(
                primary_score_policy_ref=score_policy.manifest.ref,
                primary_baseline_id=baseline_id,
            ).to_manifest(catalog),
            parent_refs=(score_policy.manifest.ref,),
        )
        prediction_artifact = registry.register(
            emit_probabilistic_prediction_artifact(
                catalog=catalog,
                feature_view=intake.feature_view_object,
                evaluation_plan=intake.evaluation_plan_object,
                evaluation_segment=intake.evaluation_plan_object.confirmatory_segment,
                fit_result=fit_result,
                score_policy_manifest=score_policy.manifest,
                stage_id="confirmatory_holdout",
                forecast_object_type=request.forecast_object_type,
            ),
            parent_refs=(
                reducer_artifact.manifest.ref,
                intake.evaluation_plan.manifest.ref,
                score_policy.manifest.ref,
            ),
        )
        baseline_prediction_artifact = registry.register(
            emit_probabilistic_prediction_artifact(
                catalog=catalog,
                feature_view=intake.feature_view_object,
                evaluation_plan=intake.evaluation_plan_object,
                evaluation_segment=intake.evaluation_plan_object.confirmatory_segment,
                fit_result=baseline_fit_result,
                score_policy_manifest=score_policy.manifest,
                stage_id="confirmatory_holdout",
                forecast_object_type=request.forecast_object_type,
            ),
            parent_refs=(
                baseline_reducer_artifact.manifest.ref,
                intake.evaluation_plan.manifest.ref,
                score_policy.manifest.ref,
            ),
        )
        score_result = registry.register(
            score_probabilistic_prediction_artifact(
                catalog=catalog,
                prediction_artifact_manifest=prediction_artifact.manifest,
                score_policy_manifest=score_policy.manifest,
            ),
            parent_refs=(prediction_artifact.manifest.ref, score_policy.manifest.ref),
        )
        baseline_score_result = registry.register(
            score_probabilistic_prediction_artifact(
                catalog=catalog,
                prediction_artifact_manifest=baseline_prediction_artifact.manifest,
                score_policy_manifest=score_policy.manifest,
            ),
            parent_refs=(
                baseline_prediction_artifact.manifest.ref,
                score_policy.manifest.ref,
            ),
        )
        calibration_contract = registry.register(
            build_calibration_contract(
                catalog=catalog,
                forecast_object_type=request.forecast_object_type,
                thresholds=request.calibration_thresholds,
            )
        )
        calibration_result = registry.register(
            evaluate_prediction_calibration(
                catalog=catalog,
                calibration_contract_manifest=calibration_contract.manifest,
                prediction_artifact_manifest=prediction_artifact.manifest,
            ),
            parent_refs=(
                calibration_contract.manifest.ref,
                prediction_artifact.manifest.ref,
            ),
        )
        comparison_key = build_comparison_key(
            evaluation_plan=intake.evaluation_plan_object,
            score_policy_ref=score_policy.manifest.ref,
            forecast_object_type=request.forecast_object_type,
        )
        comparison_universe = registry.register(
            build_comparison_universe(
                selected_candidate_id=selected_candidate_id,
                baseline_id=baseline_id,
                candidate_primary_score=float(
                    score_result.manifest.body["aggregated_primary_score"]
                ),
                baseline_primary_score=float(
                    baseline_score_result.manifest.body["aggregated_primary_score"]
                ),
                candidate_comparison_key=comparison_key,
                baseline_comparison_key=comparison_key,
                candidate_score_result_ref=score_result.manifest.ref,
                baseline_score_result_ref=baseline_score_result.manifest.ref,
                comparator_score_result_refs=(baseline_score_result.manifest.ref,),
                paired_comparison_records=(
                    _probabilistic_paired_comparison_record(
                        comparator_id=baseline_id,
                        candidate_score_result=score_result.manifest,
                        comparator_score_result=baseline_score_result.manifest,
                    ),
                ),
            ).to_manifest(catalog),
            parent_refs=(
                freeze_event.manifest.ref,
                frozen_shortlist.manifest.ref,
                score_policy.manifest.ref,
                forecast_comparison_policy.manifest.ref,
            ),
        )
        run_result_id = f"{request.request_id}_run_result"
        bundle_id = f"{request.request_id}_bundle"
        publication_id = f"{request.request_id}_publication_record"
        run_result_ref = TypedRef(
            schema_name="run_result_manifest@1.1.0",
            object_id=run_result_id,
        )
        bundle_ref = TypedRef(
            schema_name="reproducibility_bundle_manifest@1.0.0",
            object_id=bundle_id,
        )
        evaluation_event_log = registry.register(
            build_evaluation_event_log(
                search_plan_ref=intake.search_plan.manifest.ref,
                frozen_shortlist_ref=frozen_shortlist.manifest.ref,
                freeze_event_ref=freeze_event.manifest.ref,
                freeze_event_manifest=freeze_event.manifest,
                comparison_universe_ref=comparison_universe.manifest.ref,
                search_local_segment_ids=tuple(
                    segment.segment_id
                    for segment in intake.evaluation_plan_object.development_segments
                ),
                confirmatory_segment_id=(
                    intake.evaluation_plan_object.confirmatory_segment.segment_id
                ),
                holdout_access_count=1,
                prediction_artifact_ref=prediction_artifact.manifest.ref,
                run_result_ref=run_result_ref,
            ).to_manifest(catalog),
            parent_refs=(
                intake.search_plan.manifest.ref,
                frozen_shortlist.manifest.ref,
                freeze_event.manifest.ref,
                comparison_universe.manifest.ref,
                prediction_artifact.manifest.ref,
            ),
        )
        predictive_gate_policy = registry.register(
            build_predictive_gate_policy(
                allowed_forecast_object_types=(request.forecast_object_type,),
            ).to_manifest(catalog)
        )
        evaluation_governance = registry.register(
            build_evaluation_governance(
                comparison_universe_ref=comparison_universe.manifest.ref,
                event_log_ref=evaluation_event_log.manifest.ref,
                freeze_event_ref=freeze_event.manifest.ref,
                frozen_shortlist_ref=frozen_shortlist.manifest.ref,
                confirmatory_promotion_allowed=resolve_confirmatory_promotion_allowed(
                    candidate_beats_baseline=bool(
                        comparison_universe.manifest.body["candidate_beats_baseline"]
                    ),
                    predictive_gate_policy_manifest=predictive_gate_policy.manifest,
                    calibration_result_manifest=calibration_result.manifest,
                    comparison_universe_manifest=comparison_universe.manifest,
                ),
            ).to_manifest(catalog),
            parent_refs=(
                comparison_universe.manifest.ref,
                evaluation_event_log.manifest.ref,
                freeze_event.manifest.ref,
                frozen_shortlist.manifest.ref,
            ),
        )
        (
            null_protocol,
            perturbation_protocol,
            robustness_report,
        ) = _register_probabilistic_robustness_artifacts(
            catalog=catalog,
            registry=registry,
            intake=intake,
            request=request,
            candidate_id=selected_candidate_id,
            description_gain_bits=description_gain_bits,
            baseline_id=baseline_id,
            baseline_registry_ref=baseline_registry.manifest.ref,
            score_policy_ref=score_policy.manifest.ref,
        )
        scorecard_decision = resolve_scorecard_status(
            candidate_admissible=selected_candidate_admissible,
            robustness_status=str(
                robustness_report.manifest.body.get("final_robustness_status", "failed")
            ),
            candidate_beats_baseline=bool(
                comparison_universe.manifest.body["candidate_beats_baseline"]
            ),
            confirmatory_promotion_allowed=bool(
                evaluation_governance.manifest.body["confirmatory_promotion_allowed"]
            ),
            point_score_comparison_status=str(
                score_result.manifest.body["comparison_status"]
            ),
            time_safety_status=str(intake.time_safety_audit.manifest.body["status"]),
            calibration_status=str(calibration_result.manifest.body["status"]),
            descriptive_failure_reason_codes=descriptive_failure_reason_codes,
            robustness_reason_codes=(),
            predictive_governance_reason_codes=(),
        )
        scorecard = registry.register(
            ManifestEnvelope.build(
                schema_name="scorecard_manifest@1.1.0",
                module_id="gate_lifecycle",
                body={
                    "scorecard_id": f"{request.request_id}_scorecard",
                    "candidate_ref": reducer_artifact.manifest.ref.as_dict(),
                    "point_score_policy_ref": score_policy.manifest.ref.as_dict(),
                    "point_score_result_ref": score_result.manifest.ref.as_dict(),
                    "calibration_contract_ref": (
                        calibration_contract.manifest.ref.as_dict()
                    ),
                    "calibration_result_ref": (
                        calibration_result.manifest.ref.as_dict()
                    ),
                    "evaluation_plan_ref": (
                        intake.evaluation_plan.manifest.ref.as_dict()
                    ),
                    "baseline_registry_ref": baseline_registry.manifest.ref.as_dict(),
                    "forecast_comparison_policy_ref": (
                        forecast_comparison_policy.manifest.ref.as_dict()
                    ),
                    "comparison_universe_ref": (
                        comparison_universe.manifest.ref.as_dict()
                    ),
                    "evaluation_event_log_ref": (
                        evaluation_event_log.manifest.ref.as_dict()
                    ),
                    "evaluation_governance_ref": (
                        evaluation_governance.manifest.ref.as_dict()
                    ),
                    "predictive_gate_policy_ref": (
                        predictive_gate_policy.manifest.ref.as_dict()
                    ),
                    "null_protocol_ref": null_protocol.manifest.ref.as_dict(),
                    "perturbation_protocol_ref": (
                        perturbation_protocol.manifest.ref.as_dict()
                    ),
                    "robustness_report_ref": robustness_report.manifest.ref.as_dict(),
                    "time_safety_audit_ref": (
                        intake.time_safety_audit.manifest.ref.as_dict()
                    ),
                    "forecast_object_type": request.forecast_object_type,
                    "description_gain_bits": _stable_demo_float(description_gain_bits),
                    "descriptive_status": scorecard_decision.descriptive_status,
                    "descriptive_reason_codes": list(
                        scorecard_decision.descriptive_reason_codes
                    ),
                    "predictive_status": scorecard_decision.predictive_status,
                    "predictive_reason_codes": list(
                        scorecard_decision.predictive_reason_codes
                    ),
                },
                catalog=catalog,
            ),
            parent_refs=(
                reducer_artifact.manifest.ref,
                score_policy.manifest.ref,
                score_result.manifest.ref,
                calibration_contract.manifest.ref,
                calibration_result.manifest.ref,
                comparison_universe.manifest.ref,
                evaluation_event_log.manifest.ref,
                evaluation_governance.manifest.ref,
                predictive_gate_policy.manifest.ref,
                null_protocol.manifest.ref,
                perturbation_protocol.manifest.ref,
                robustness_report.manifest.ref,
                intake.time_safety_audit.manifest.ref,
            ),
        )
        validation_scope = registry.register(
            ManifestEnvelope.build(
                schema_name="validation_scope_manifest@1.0.0",
                module_id="claims",
                body={
                    "validation_scope_id": f"{request.request_id}_validation_scope",
                    "scope_ledger_ref": scope_ledger.manifest.ref.as_dict(),
                    "evaluation_plan_ref": (
                        intake.evaluation_plan.manifest.ref.as_dict()
                    ),
                    "baseline_registry_ref": baseline_registry.manifest.ref.as_dict(),
                    "forecast_comparison_policy_ref": (
                        forecast_comparison_policy.manifest.ref.as_dict()
                    ),
                    "comparison_universe_ref": (
                        comparison_universe.manifest.ref.as_dict()
                    ),
                    "evaluation_event_log_ref": (
                        evaluation_event_log.manifest.ref.as_dict()
                    ),
                    "evaluation_governance_ref": (
                        evaluation_governance.manifest.ref.as_dict()
                    ),
                    "predictive_gate_policy_ref": (
                        predictive_gate_policy.manifest.ref.as_dict()
                    ),
                    "null_protocol_ref": null_protocol.manifest.ref.as_dict(),
                    "perturbation_protocol_ref": (
                        perturbation_protocol.manifest.ref.as_dict()
                    ),
                    "time_safety_audit_ref": (
                        intake.time_safety_audit.manifest.ref.as_dict()
                    ),
                    "point_score_policy_ref": score_policy.manifest.ref.as_dict(),
                    "calibration_contract_ref": (
                        calibration_contract.manifest.ref.as_dict()
                    ),
                    "public_forecast_object_type": request.forecast_object_type,
                    "run_support_object_ids": list(request.run_support_object_ids),
                    "admissibility_rule_ids": list(request.admissibility_rule_ids),
                    "entity_scope": "single_entity_only",
                    "horizon_scope": (
                        "single_horizon"
                        if len(intake.evaluation_plan_object.horizon_set) == 1
                        else "multi_horizon"
                    ),
                },
                catalog=catalog,
            ),
            parent_refs=(
                scope_ledger.manifest.ref,
                intake.evaluation_plan.manifest.ref,
                baseline_registry.manifest.ref,
                forecast_comparison_policy.manifest.ref,
                comparison_universe.manifest.ref,
                evaluation_event_log.manifest.ref,
                evaluation_governance.manifest.ref,
                predictive_gate_policy.manifest.ref,
                null_protocol.manifest.ref,
                perturbation_protocol.manifest.ref,
                intake.time_safety_audit.manifest.ref,
                score_policy.manifest.ref,
                calibration_contract.manifest.ref,
            ),
        )
        claim_decision = resolve_claim_publication(
            scorecard_body=scorecard.manifest.body
        )
        claim_card = None
        abstention = None
        if claim_decision.publication_mode == "candidate_publication":
            claim_card = registry.register(
                ManifestEnvelope.build(
                    schema_name="claim_card_manifest@1.1.0",
                    module_id="claims",
                    body={
                        "claim_card_id": f"{request.request_id}_claim_card",
                        "candidate_ref": reducer_artifact.manifest.ref.as_dict(),
                        "scorecard_ref": scorecard.manifest.ref.as_dict(),
                        "validation_scope_ref": validation_scope.manifest.ref.as_dict(),
                        "claim_type": claim_decision.claim_type,
                        "claim_ceiling": claim_decision.claim_ceiling,
                        "predictive_support_status": (
                            claim_decision.predictive_support_status
                        ),
                        "allowed_interpretation_codes": list(
                            claim_decision.allowed_interpretation_codes
                        ),
                        "forbidden_interpretation_codes": list(
                            claim_decision.forbidden_interpretation_codes
                        ),
                    },
                    catalog=catalog,
                ),
                parent_refs=(
                    reducer_artifact.manifest.ref,
                    scorecard.manifest.ref,
                    validation_scope.manifest.ref,
                ),
            )
        else:
            abstention = registry.register(
                AbstentionManifest(
                    abstention_id=f"{request.request_id}_abstention",
                    abstention_type=str(claim_decision.abstention_type),
                    blocked_ceiling=str(claim_decision.blocked_ceiling),
                    reason_codes=claim_decision.abstention_reason_codes,
                    governing_refs=(scorecard.manifest.ref,),
                ).to_manifest(catalog),
                parent_refs=(scorecard.manifest.ref,),
            )
        publication_mode = claim_decision.publication_mode
        run_result_manifest = build_run_result_manifest(
            object_id=run_result_id,
            run_id=run_result_id,
            scope_ledger_ref=scope_ledger.manifest.ref,
            search_plan_ref=intake.search_plan.manifest.ref,
            evaluation_plan_ref=intake.evaluation_plan.manifest.ref,
            comparison_universe_ref=comparison_universe.manifest.ref,
            evaluation_event_log_ref=evaluation_event_log.manifest.ref,
            evaluation_governance_ref=evaluation_governance.manifest.ref,
            reproducibility_bundle_ref=bundle_ref,
            forecast_object_type=request.forecast_object_type,
            primary_validation_scope_ref=validation_scope.manifest.ref,
            publication_mode=publication_mode,
            selected_candidate_ref=(
                reducer_artifact.manifest.ref
                if publication_mode == "candidate_publication"
                else None
            ),
            scorecard_ref=(
                scorecard.manifest.ref
                if publication_mode == "candidate_publication"
                else None
            ),
            claim_card_ref=claim_card.manifest.ref if claim_card is not None else None,
            abstention_ref=abstention.manifest.ref if abstention is not None else None,
            prediction_artifact_refs=(prediction_artifact.manifest.ref,),
            primary_score_result_ref=score_result.manifest.ref,
            primary_calibration_result_ref=calibration_result.manifest.ref,
            robustness_report_refs=(robustness_report.manifest.ref,),
        )
        publication_artifact = claim_card or abstention
        assert publication_artifact is not None
        reproducibility_bundle = registry.register(
            build_reproducibility_bundle_manifest(
                object_id=bundle_id,
                bundle_id=bundle_id,
                bundle_mode=publication_mode,
                dataset_snapshot_ref=intake.snapshot.manifest.ref,
                feature_view_ref=intake.feature_view.manifest.ref,
                search_plan_ref=intake.search_plan.manifest.ref,
                evaluation_plan_ref=intake.evaluation_plan.manifest.ref,
                comparison_universe_ref=comparison_universe.manifest.ref,
                evaluation_event_log_ref=evaluation_event_log.manifest.ref,
                evaluation_governance_ref=evaluation_governance.manifest.ref,
                run_result_ref=run_result_ref,
                required_manifest_refs=required_manifest_refs_for_publication(
                    publication_mode=publication_mode,
                    candidate_ref=reducer_artifact.manifest.ref,
                    scorecard_ref=scorecard.manifest.ref,
                    claim_ref=claim_card.manifest.ref if claim_card else None,
                    abstention_ref=abstention.manifest.ref if abstention else None,
                    supporting_refs=(
                        prediction_artifact.manifest.ref,
                        validation_scope.manifest.ref,
                        score_result.manifest.ref,
                        calibration_result.manifest.ref,
                    ),
                ),
                artifact_hash_records=build_artifact_hash_records(
                    snapshot=intake.snapshot,
                    feature_view=intake.feature_view,
                    search_plan=intake.search_plan,
                    evaluation_plan=intake.evaluation_plan,
                    run_result_manifest=run_result_manifest.to_manifest(catalog),
                    candidate_or_abstention=publication_artifact,
                    scorecard=scorecard,
                    prediction_artifact=prediction_artifact,
                    validation_scope=validation_scope,
                    robustness_report=robustness_report,
                    score_result=score_result,
                    calibration_result=calibration_result,
                ),
                seed_records=build_replay_seed_records(request.search_seed),
                stage_order_records=build_replay_stage_order(
                    dataset_snapshot_ref=intake.snapshot.manifest.ref,
                    feature_view_ref=intake.feature_view.manifest.ref,
                    search_plan_ref=intake.search_plan.manifest.ref,
                    evaluation_plan_ref=intake.evaluation_plan.manifest.ref,
                    comparison_universe_ref=comparison_universe.manifest.ref,
                    evaluation_event_log_ref=evaluation_event_log.manifest.ref,
                    evaluation_governance_ref=evaluation_governance.manifest.ref,
                    scorecard_ref=scorecard.manifest.ref,
                    candidate_or_abstention_ref=publication_artifact.manifest.ref,
                    run_result_ref=run_result_ref,
                ),
                replay_verification_status="verified",
                failure_reason_codes=(),
            ).to_manifest(catalog),
        )
        run_result = registry.register(
            build_run_result_manifest(
                object_id=run_result_manifest.object_id,
                run_id=run_result_manifest.run_id,
                scope_ledger_ref=run_result_manifest.scope_ledger_ref,
                search_plan_ref=run_result_manifest.search_plan_ref,
                evaluation_plan_ref=run_result_manifest.evaluation_plan_ref,
                comparison_universe_ref=run_result_manifest.comparison_universe_ref,
                evaluation_event_log_ref=run_result_manifest.evaluation_event_log_ref,
                evaluation_governance_ref=run_result_manifest.evaluation_governance_ref,
                reproducibility_bundle_ref=reproducibility_bundle.manifest.ref,
                forecast_object_type=run_result_manifest.forecast_object_type,
                primary_validation_scope_ref=(
                    run_result_manifest.primary_validation_scope_ref
                ),
                publication_mode=run_result_manifest.result_mode,
                selected_candidate_ref=run_result_manifest.primary_reducer_artifact_ref,
                scorecard_ref=run_result_manifest.primary_scorecard_ref,
                claim_card_ref=run_result_manifest.primary_claim_card_ref,
                abstention_ref=run_result_manifest.primary_abstention_ref,
                prediction_artifact_refs=run_result_manifest.prediction_artifact_refs,
                primary_score_result_ref=(run_result_manifest.primary_score_result_ref),
                primary_calibration_result_ref=(
                    run_result_manifest.primary_calibration_result_ref
                ),
                robustness_report_refs=run_result_manifest.robustness_report_refs,
                deferred_scope_policy_refs=run_result_manifest.deferred_scope_policy_refs,
            ).to_manifest(catalog),
            parent_refs=run_result_manifest.lineage_refs,
        )
        readiness_judgment = registry.register(
            _build_runtime_readiness_manifest(
                catalog=catalog,
                judgment_id=f"{request.request_id}_readiness_judgment",
                replay_verification_status="verified",
                schema_closure_status="passed",
            )
        )
        schema_lifecycle_integration_closure = registry.register(
            SchemaLifecycleIntegrationClosureManifest(
                closure_id=f"{request.request_id}_schema_lifecycle_closure",
                status="passed",
            ).to_manifest(catalog)
        )
        publication_record = registry.register(
            build_publication_record_manifest(
                object_id=publication_id,
                publication_id=publication_id,
                run_result_manifest=run_result.manifest,
                comparison_universe_manifest=comparison_universe.manifest,
                reproducibility_bundle_manifest=reproducibility_bundle.manifest,
                readiness_judgment_manifest=readiness_judgment.manifest,
                schema_lifecycle_integration_closure_ref=(
                    schema_lifecycle_integration_closure.manifest.ref
                ),
                catalog_scope="public",
                published_at="2026-04-14T00:00:00Z",
            ).to_manifest(catalog),
            parent_refs=(
                run_result.manifest.ref,
                reproducibility_bundle.manifest.ref,
                readiness_judgment.manifest.ref,
                schema_lifecycle_integration_closure.manifest.ref,
            ),
        )
        _write_summary_payload(
            paths=paths,
            payload={
                "workflow_surface": "probabilistic_evaluation",
                "request_id": request.request_id,
                "manifest_path": str(request.manifest_path),
                "dataset_csv": str(request.dataset_csv),
                "bundle_ref": reproducibility_bundle.manifest.ref.as_dict(),
                "run_result_ref": run_result.manifest.ref.as_dict(),
                "publication_record_ref": publication_record.manifest.ref.as_dict(),
                "comparison_universe_ref": comparison_universe.manifest.ref.as_dict(),
                "scorecard_ref": scorecard.manifest.ref.as_dict(),
                "claim_card_ref": (
                    claim_card.manifest.ref.as_dict()
                    if claim_card is not None
                    else None
                ),
                "abstention_ref": (
                    abstention.manifest.ref.as_dict()
                    if abstention is not None
                    else None
                ),
                "result_mode": publication_mode,
                "selected_candidate_id": selected_candidate_id,
                "selected_family": fit_result.family_id,
                "forecast_object_type": request.forecast_object_type,
                "prediction_artifact_ref": prediction_artifact.manifest.ref.as_dict(),
                "score_result_ref": score_result.manifest.ref.as_dict(),
                "calibration_contract_ref": (
                    calibration_contract.manifest.ref.as_dict()
                ),
                "calibration_result_ref": (calibration_result.manifest.ref.as_dict()),
            },
        )

    from euclid.inspection import (
        inspect_demo_calibration,
        inspect_demo_probabilistic_prediction,
    )

    prediction = inspect_demo_probabilistic_prediction(output_root=paths.output_root)
    calibration = inspect_demo_calibration(output_root=paths.output_root)
    summary = DemoProbabilisticRunSummary(
        selected_candidate_id=selected_candidate_id,
        selected_family=fit_result.family_id,
        forecast_object_type=request.forecast_object_type,
        bundle_ref=reproducibility_bundle.manifest.ref,
        run_result_ref=run_result.manifest.ref,
        publication_record_ref=publication_record.manifest.ref,
        comparison_universe_ref=comparison_universe.manifest.ref,
        scorecard_ref=scorecard.manifest.ref,
        claim_card_ref=claim_card.manifest.ref if claim_card is not None else None,
        abstention_ref=abstention.manifest.ref if abstention is not None else None,
        prediction_artifact_ref=prediction.prediction_artifact_ref,
        score_result_ref=prediction.score_result_ref,
        calibration_result_ref=calibration.calibration_result_ref,
        aggregated_primary_score=prediction.aggregated_primary_score,
        calibration_status=calibration.status,
    )
    return DemoProbabilisticEvaluationResult(
        request=request,
        paths=paths,
        summary=summary,
        prediction=prediction,
        calibration=calibration,
    )


def _run_shared_local_point_workflow(
    *,
    request: DemoRequest,
    catalog: ContractCatalog,
    registry: ManifestRegistry,
) -> PrototypeReducerWorkflowResult:
    intake = _build_search_intake(
        request=request,
        catalog=catalog,
        registry=registry,
    )
    selected_candidate = _realize_shared_local_panel_candidate(
        search_plan=intake.search_plan_object,
        feature_view=intake.feature_view_object,
    )
    selected_candidate_id = _candidate_source_id(selected_candidate)
    baseline_candidate = _realize_probabilistic_baseline_candidate(
        feature_view=intake.feature_view_object,
        search_plan=intake.search_plan_object,
        selected_candidate_id=selected_candidate_id,
    )
    if baseline_candidate is None:
        raise ContractValidationError(
            code="missing_shared_local_baseline_candidate",
            message=(
                "shared-plus-local operator runs require a retained comparator "
                "baseline under the frozen confirmatory geometry"
            ),
            field_path="search_family_ids",
        )

    fit_result = fit_candidate_window(
        candidate=selected_candidate,
        feature_view=intake.feature_view_object,
        fit_window=intake.evaluation_plan_object.confirmatory_segment,
        search_plan=intake.search_plan_object,
        stage_id="confirmatory_holdout",
    )
    baseline_fit_result = fit_candidate_window(
        candidate=baseline_candidate,
        feature_view=intake.feature_view_object,
        fit_window=intake.evaluation_plan_object.confirmatory_segment,
        search_plan=intake.search_plan_object,
        stage_id="confirmatory_holdout",
    )
    description_gain_bits = 0.0
    baseline_description_gain_bits = 0.0
    fit_artifacts = build_candidate_fit_artifacts(
        catalog=catalog,
        fit_result=fit_result,
        search_plan_ref=intake.search_plan.manifest.ref,
        selection_floor_bits=request.minimum_description_gain_bits,
        description_gain_bits=description_gain_bits,
    )
    baseline_fit_artifacts = build_candidate_fit_artifacts(
        catalog=catalog,
        fit_result=baseline_fit_result,
        search_plan_ref=intake.search_plan.manifest.ref,
        selection_floor_bits=request.minimum_description_gain_bits,
        description_gain_bits=baseline_description_gain_bits,
    )
    candidate_spec = registry.register(
        fit_artifacts.candidate_spec,
        parent_refs=(intake.search_plan.manifest.ref,),
    )
    candidate_state = registry.register(
        fit_artifacts.candidate_state,
        parent_refs=(intake.search_plan.manifest.ref, candidate_spec.manifest.ref),
    )
    reducer_artifact = registry.register(
        fit_artifacts.reducer_artifact,
        parent_refs=(
            intake.search_plan.manifest.ref,
            candidate_spec.manifest.ref,
            candidate_state.manifest.ref,
        ),
    )
    baseline_candidate_spec = registry.register(
        baseline_fit_artifacts.candidate_spec,
        parent_refs=(intake.search_plan.manifest.ref,),
    )
    baseline_candidate_state = registry.register(
        baseline_fit_artifacts.candidate_state,
        parent_refs=(
            intake.search_plan.manifest.ref,
            baseline_candidate_spec.manifest.ref,
        ),
    )
    baseline_reducer_artifact = registry.register(
        baseline_fit_artifacts.reducer_artifact,
        parent_refs=(
            intake.search_plan.manifest.ref,
            baseline_candidate_spec.manifest.ref,
            baseline_candidate_state.manifest.ref,
        ),
    )
    point_score_policy = registry.register(
        _build_point_score_policy_manifest(
            catalog=catalog,
            request=request,
            horizon_set=intake.evaluation_plan_object.horizon_set,
            entity_panel=intake.evaluation_plan_object.entity_panel,
        )
    )
    shared_local_policy = _register_shared_local_policy_artifacts(
        catalog=catalog,
        registry=registry,
        request=request,
        entity_panel=intake.evaluation_plan_object.entity_panel,
    )
    scope_ledger = registry.register(
        ManifestEnvelope.build(
            schema_name="scope_ledger_manifest@1.0.0",
            module_id="manifest_registry",
            body={
                "scope_ledger_id": f"{request.request_id}_scope_ledger",
                "scope_id": "euclid_v1_binding_scope@1.0.0",
                "forecast_object_type": request.forecast_object_type,
                "candidate_family_ids": list(request.search_family_ids),
                "run_support_object_ids": list(request.run_support_object_ids),
                "admissibility_rule_ids": list(request.admissibility_rule_ids),
                "deferred_scope_annotations": [
                    "shared_plus_local_decomposition",
                    "mechanistic_evidence",
                    "algorithmic_publication",
                ],
            },
            catalog=catalog,
        )
    )
    exploratory_primary_score = _stable_demo_float(
        float(fit_result.optimizer_diagnostics["final_loss"])
        / max(fit_result.training_row_count, 1)
    )
    ledger_records = (
        SearchCandidateRecord(
            candidate_id=selected_candidate_id,
            family_id=fit_result.family_id,
            total_code_bits=0.0,
            structure_code_bits=0.0,
            description_gain_bits=description_gain_bits,
            inner_primary_score=exploratory_primary_score,
            admissible=True,
            ranked=True,
            law_eligible=True,
            canonical_byte_length=len(selected_candidate_id.encode("utf-8")),
        ),
    )
    search_ledger = registry.register(
        build_search_ledger(
            search_plan=intake.search_plan_object,
            candidate_records=ledger_records,
            selected_candidate_id=selected_candidate_id,
        ).to_manifest(catalog),
        parent_refs=(intake.search_plan.manifest.ref,),
    )
    frontier = registry.register(
        build_frontier(
            search_plan=intake.search_plan_object,
            candidate_records=ledger_records,
        ).to_manifest(catalog),
        parent_refs=(search_ledger.manifest.ref,),
    )
    rejected_diagnostics = registry.register(
        build_rejected_diagnostics(candidate_records=ledger_records).to_manifest(
            catalog
        ),
        parent_refs=(search_ledger.manifest.ref,),
    )
    frozen_shortlist = registry.register(
        build_frozen_shortlist(
            search_plan_ref=intake.search_plan.manifest.ref,
            candidate_ref=reducer_artifact.manifest.ref,
        ).to_manifest(catalog),
        parent_refs=(
            intake.search_plan.manifest.ref,
            frontier.manifest.ref,
            rejected_diagnostics.manifest.ref,
            reducer_artifact.manifest.ref,
        ),
    )
    baseline_id = _candidate_source_id(baseline_candidate)
    freeze_event = registry.register(
        build_freeze_event(
            frozen_candidate_ref=reducer_artifact.manifest.ref,
            frozen_shortlist_ref=frozen_shortlist.manifest.ref,
            confirmatory_baseline_id=baseline_id,
        ).to_manifest(catalog),
        parent_refs=(frozen_shortlist.manifest.ref,),
    )
    baseline_registry = registry.register(
        BaselineRegistry(
            compatible_point_score_policy_ref=point_score_policy.manifest.ref,
            declarations=(
                ComparatorDeclaration(
                    comparator_declaration_id=f"{baseline_id}_declaration",
                    baseline_id=baseline_id,
                    comparator_kind="baseline",
                    forecast_object_type="point",
                    family_id=_candidate_family_id(baseline_candidate),
                    freeze_rule="frozen_before_confirmatory_access",
                ),
            ),
        ).to_manifest(catalog),
        parent_refs=(point_score_policy.manifest.ref,),
    )
    forecast_comparison_policy = registry.register(
        build_forecast_comparison_policy(
            primary_score_policy_ref=point_score_policy.manifest.ref,
            primary_baseline_id=baseline_id,
        ).to_manifest(catalog),
        parent_refs=(point_score_policy.manifest.ref,),
    )
    prediction_artifact = registry.register(
        emit_point_prediction_artifact(
            catalog=catalog,
            feature_view=intake.feature_view_object,
            evaluation_plan=intake.evaluation_plan_object,
            evaluation_segment=intake.evaluation_plan_object.confirmatory_segment,
            fit_result=fit_result,
            score_policy_manifest=point_score_policy.manifest,
            stage_id="confirmatory_holdout",
        ),
        parent_refs=(
            reducer_artifact.manifest.ref,
            intake.evaluation_plan.manifest.ref,
            point_score_policy.manifest.ref,
        ),
    )
    baseline_prediction_artifact = registry.register(
        emit_point_prediction_artifact(
            catalog=catalog,
            feature_view=intake.feature_view_object,
            evaluation_plan=intake.evaluation_plan_object,
            evaluation_segment=intake.evaluation_plan_object.confirmatory_segment,
            fit_result=baseline_fit_result,
            score_policy_manifest=point_score_policy.manifest,
            stage_id="confirmatory_holdout",
        ),
        parent_refs=(
            baseline_reducer_artifact.manifest.ref,
            intake.evaluation_plan.manifest.ref,
            point_score_policy.manifest.ref,
        ),
    )
    comparator_result = evaluate_point_comparators(
        catalog=catalog,
        score_policy_manifest=point_score_policy.manifest,
        candidate_prediction_artifact=prediction_artifact.manifest,
        baseline_registry_manifest=baseline_registry.manifest,
        comparator_prediction_artifacts={
            baseline_id: baseline_prediction_artifact.manifest,
        },
    )
    point_score_result = registry.register(
        comparator_result.candidate_score_result,
        parent_refs=(prediction_artifact.manifest.ref, point_score_policy.manifest.ref),
    )
    baseline_score_result = registry.register(
        comparator_result.comparator_score_results[0],
        parent_refs=(
            baseline_prediction_artifact.manifest.ref,
            point_score_policy.manifest.ref,
        ),
    )
    comparison_universe = registry.register(
        comparator_result.comparison_universe,
        parent_refs=(
            freeze_event.manifest.ref,
            frozen_shortlist.manifest.ref,
            point_score_policy.manifest.ref,
            forecast_comparison_policy.manifest.ref,
            point_score_result.manifest.ref,
            baseline_score_result.manifest.ref,
        ),
    )
    calibration_contract = registry.register(
        build_calibration_contract(
            catalog=catalog,
            forecast_object_type=request.forecast_object_type,
            thresholds=request.calibration_thresholds,
        )
    )
    calibration_result = registry.register(
        evaluate_prediction_calibration(
            catalog=catalog,
            calibration_contract_manifest=calibration_contract.manifest,
            prediction_artifact_manifest=prediction_artifact.manifest,
        ),
        parent_refs=(
            calibration_contract.manifest.ref,
            prediction_artifact.manifest.ref,
        ),
    )
    run_result_id = f"{request.request_id}_run_result"
    bundle_id = f"{request.request_id}_bundle"
    run_result_ref = TypedRef(
        schema_name="run_result_manifest@1.1.0",
        object_id=run_result_id,
    )
    bundle_ref = TypedRef(
        schema_name="reproducibility_bundle_manifest@1.0.0",
        object_id=bundle_id,
    )
    evaluation_event_log = registry.register(
        build_evaluation_event_log(
            search_plan_ref=intake.search_plan.manifest.ref,
            frozen_shortlist_ref=frozen_shortlist.manifest.ref,
            freeze_event_ref=freeze_event.manifest.ref,
            freeze_event_manifest=freeze_event.manifest,
            comparison_universe_ref=comparison_universe.manifest.ref,
            search_local_segment_ids=tuple(
                segment.segment_id
                for segment in intake.evaluation_plan_object.development_segments
            ),
            confirmatory_segment_id=(
                intake.evaluation_plan_object.confirmatory_segment.segment_id
            ),
            holdout_access_count=1,
            prediction_artifact_ref=prediction_artifact.manifest.ref,
            run_result_ref=run_result_ref,
        ).to_manifest(catalog),
        parent_refs=(
            intake.search_plan.manifest.ref,
            frozen_shortlist.manifest.ref,
            freeze_event.manifest.ref,
            comparison_universe.manifest.ref,
            prediction_artifact.manifest.ref,
        ),
    )
    predictive_gate_policy = registry.register(
        build_predictive_gate_policy(
            allowed_forecast_object_types=(request.forecast_object_type,),
        ).to_manifest(catalog)
    )
    evaluation_governance = registry.register(
        build_evaluation_governance(
            comparison_universe_ref=comparison_universe.manifest.ref,
            event_log_ref=evaluation_event_log.manifest.ref,
            freeze_event_ref=freeze_event.manifest.ref,
            frozen_shortlist_ref=frozen_shortlist.manifest.ref,
            confirmatory_promotion_allowed=resolve_confirmatory_promotion_allowed(
                candidate_beats_baseline=bool(
                    comparison_universe.manifest.body["candidate_beats_baseline"]
                ),
                predictive_gate_policy_manifest=predictive_gate_policy.manifest,
                calibration_result_manifest=calibration_result.manifest,
                comparison_universe_manifest=comparison_universe.manifest,
            ),
        ).to_manifest(catalog),
        parent_refs=(
            comparison_universe.manifest.ref,
            evaluation_event_log.manifest.ref,
            freeze_event.manifest.ref,
            frozen_shortlist.manifest.ref,
        ),
    )
    (
        null_protocol,
        perturbation_protocol,
        robustness_report,
    ) = _register_probabilistic_robustness_artifacts(
        catalog=catalog,
        registry=registry,
        intake=intake,
        request=request,
        candidate_id=selected_candidate_id,
        description_gain_bits=description_gain_bits,
        baseline_id=baseline_id,
        baseline_registry_ref=baseline_registry.manifest.ref,
        score_policy_ref=point_score_policy.manifest.ref,
    )
    scorecard_decision = resolve_scorecard_status(
        candidate_admissible=True,
        robustness_status=str(
            robustness_report.manifest.body.get("final_robustness_status", "failed")
        ),
        candidate_beats_baseline=bool(
            comparison_universe.manifest.body["candidate_beats_baseline"]
        ),
        confirmatory_promotion_allowed=bool(
            evaluation_governance.manifest.body["confirmatory_promotion_allowed"]
        ),
        point_score_comparison_status=str(
            point_score_result.manifest.body["comparison_status"]
        ),
        time_safety_status=str(intake.time_safety_audit.manifest.body["status"]),
        calibration_status=str(calibration_result.manifest.body["status"]),
        descriptive_failure_reason_codes=(),
        robustness_reason_codes=(),
        predictive_governance_reason_codes=(),
    )
    scorecard = registry.register(
        ManifestEnvelope.build(
            schema_name="scorecard_manifest@1.1.0",
            module_id="gate_lifecycle",
            body={
                "scorecard_id": f"{request.request_id}_scorecard",
                "candidate_ref": reducer_artifact.manifest.ref.as_dict(),
                "point_score_policy_ref": point_score_policy.manifest.ref.as_dict(),
                "point_score_result_ref": point_score_result.manifest.ref.as_dict(),
                "calibration_contract_ref": (
                    calibration_contract.manifest.ref.as_dict()
                ),
                "calibration_result_ref": calibration_result.manifest.ref.as_dict(),
                "evaluation_plan_ref": intake.evaluation_plan.manifest.ref.as_dict(),
                "baseline_registry_ref": baseline_registry.manifest.ref.as_dict(),
                "forecast_comparison_policy_ref": (
                    forecast_comparison_policy.manifest.ref.as_dict()
                ),
                "comparison_universe_ref": comparison_universe.manifest.ref.as_dict(),
                "evaluation_event_log_ref": (
                    evaluation_event_log.manifest.ref.as_dict()
                ),
                "evaluation_governance_ref": (
                    evaluation_governance.manifest.ref.as_dict()
                ),
                "predictive_gate_policy_ref": (
                    predictive_gate_policy.manifest.ref.as_dict()
                ),
                "null_protocol_ref": null_protocol.manifest.ref.as_dict(),
                "perturbation_protocol_ref": (
                    perturbation_protocol.manifest.ref.as_dict()
                ),
                "robustness_report_ref": robustness_report.manifest.ref.as_dict(),
                "time_safety_audit_ref": (
                    intake.time_safety_audit.manifest.ref.as_dict()
                ),
                "forecast_object_type": request.forecast_object_type,
                "description_gain_bits": _stable_demo_float(description_gain_bits),
                "descriptive_status": scorecard_decision.descriptive_status,
                "descriptive_reason_codes": list(
                    scorecard_decision.descriptive_reason_codes
                ),
                "predictive_status": scorecard_decision.predictive_status,
                "predictive_reason_codes": list(
                    scorecard_decision.predictive_reason_codes
                ),
                "entity_panel": list(
                    prediction_artifact.manifest.body.get("entity_panel", ())
                ),
            },
            catalog=catalog,
        ),
        parent_refs=(
            reducer_artifact.manifest.ref,
            point_score_policy.manifest.ref,
            point_score_result.manifest.ref,
            calibration_contract.manifest.ref,
            calibration_result.manifest.ref,
            comparison_universe.manifest.ref,
            evaluation_event_log.manifest.ref,
            evaluation_governance.manifest.ref,
            predictive_gate_policy.manifest.ref,
            null_protocol.manifest.ref,
            perturbation_protocol.manifest.ref,
            robustness_report.manifest.ref,
            intake.time_safety_audit.manifest.ref,
            shared_local_policy.manifest.ref,
        ),
    )
    validation_scope = registry.register(
        ManifestEnvelope.build(
            schema_name="validation_scope_manifest@1.0.0",
            module_id="claims",
            body={
                "validation_scope_id": f"{request.request_id}_validation_scope",
                "scope_ledger_ref": scope_ledger.manifest.ref.as_dict(),
                "evaluation_plan_ref": intake.evaluation_plan.manifest.ref.as_dict(),
                "baseline_registry_ref": baseline_registry.manifest.ref.as_dict(),
                "forecast_comparison_policy_ref": (
                    forecast_comparison_policy.manifest.ref.as_dict()
                ),
                "comparison_universe_ref": comparison_universe.manifest.ref.as_dict(),
                "evaluation_event_log_ref": (
                    evaluation_event_log.manifest.ref.as_dict()
                ),
                "evaluation_governance_ref": (
                    evaluation_governance.manifest.ref.as_dict()
                ),
                "predictive_gate_policy_ref": (
                    predictive_gate_policy.manifest.ref.as_dict()
                ),
                "null_protocol_ref": null_protocol.manifest.ref.as_dict(),
                "perturbation_protocol_ref": (
                    perturbation_protocol.manifest.ref.as_dict()
                ),
                "time_safety_audit_ref": (
                    intake.time_safety_audit.manifest.ref.as_dict()
                ),
                "point_score_policy_ref": point_score_policy.manifest.ref.as_dict(),
                "calibration_contract_ref": (
                    calibration_contract.manifest.ref.as_dict()
                ),
                "public_forecast_object_type": "point",
                "run_support_object_ids": list(request.run_support_object_ids),
                "admissibility_rule_ids": list(request.admissibility_rule_ids),
                "entity_scope": "declared_entity_panel",
                "entity_panel": list(intake.evaluation_plan_object.entity_panel),
                "horizon_scope": (
                    "single_horizon"
                    if len(intake.evaluation_plan_object.horizon_set) == 1
                    else "multi_horizon"
                ),
            },
            catalog=catalog,
        ),
        parent_refs=(
            scope_ledger.manifest.ref,
            intake.evaluation_plan.manifest.ref,
            baseline_registry.manifest.ref,
            forecast_comparison_policy.manifest.ref,
            comparison_universe.manifest.ref,
            evaluation_event_log.manifest.ref,
            evaluation_governance.manifest.ref,
            predictive_gate_policy.manifest.ref,
            null_protocol.manifest.ref,
            perturbation_protocol.manifest.ref,
            intake.time_safety_audit.manifest.ref,
            point_score_policy.manifest.ref,
            calibration_contract.manifest.ref,
            shared_local_policy.manifest.ref,
        ),
    )
    claim_decision = resolve_claim_publication(scorecard_body=scorecard.manifest.body)
    claim_card = None
    abstention = None
    if claim_decision.publication_mode == "candidate_publication":
        claim_card = registry.register(
            ManifestEnvelope.build(
                schema_name="claim_card_manifest@1.1.0",
                module_id="claims",
                body={
                    "claim_card_id": f"{request.request_id}_claim_card",
                    "candidate_ref": reducer_artifact.manifest.ref.as_dict(),
                    "scorecard_ref": scorecard.manifest.ref.as_dict(),
                    "validation_scope_ref": validation_scope.manifest.ref.as_dict(),
                    "claim_type": claim_decision.claim_type,
                    "claim_ceiling": claim_decision.claim_ceiling,
                    "predictive_support_status": (
                        claim_decision.predictive_support_status
                    ),
                    "allowed_interpretation_codes": list(
                        claim_decision.allowed_interpretation_codes
                    ),
                    "forbidden_interpretation_codes": list(
                        claim_decision.forbidden_interpretation_codes
                    ),
                },
                catalog=catalog,
            ),
            parent_refs=(
                reducer_artifact.manifest.ref,
                scorecard.manifest.ref,
                validation_scope.manifest.ref,
                shared_local_policy.manifest.ref,
            ),
        )
    else:
        abstention = registry.register(
            AbstentionManifest(
                abstention_id=f"{request.request_id}_abstention",
                abstention_type=str(claim_decision.abstention_type),
                blocked_ceiling=str(claim_decision.blocked_ceiling),
                reason_codes=claim_decision.abstention_reason_codes,
                governing_refs=(scorecard.manifest.ref,),
            ).to_manifest(catalog),
            parent_refs=(scorecard.manifest.ref,),
        )
    publication_mode = claim_decision.publication_mode
    run_result_manifest = build_run_result_manifest(
        object_id=run_result_id,
        run_id=run_result_id,
        scope_ledger_ref=scope_ledger.manifest.ref,
        search_plan_ref=intake.search_plan.manifest.ref,
        evaluation_plan_ref=intake.evaluation_plan.manifest.ref,
        comparison_universe_ref=comparison_universe.manifest.ref,
        evaluation_event_log_ref=evaluation_event_log.manifest.ref,
        evaluation_governance_ref=evaluation_governance.manifest.ref,
        reproducibility_bundle_ref=bundle_ref,
        forecast_object_type=request.forecast_object_type,
        primary_validation_scope_ref=validation_scope.manifest.ref,
        publication_mode=publication_mode,
        selected_candidate_ref=(
            reducer_artifact.manifest.ref
            if publication_mode == "candidate_publication"
            else None
        ),
        scorecard_ref=(
            scorecard.manifest.ref
            if publication_mode == "candidate_publication"
            else None
        ),
        claim_card_ref=claim_card.manifest.ref if claim_card is not None else None,
        abstention_ref=abstention.manifest.ref if abstention is not None else None,
        prediction_artifact_refs=(prediction_artifact.manifest.ref,),
        primary_score_result_ref=point_score_result.manifest.ref,
        primary_calibration_result_ref=calibration_result.manifest.ref,
        robustness_report_refs=(robustness_report.manifest.ref,),
        deferred_scope_policy_refs=(shared_local_policy.manifest.ref,),
    )
    publication_artifact = claim_card or abstention
    assert publication_artifact is not None
    reproducibility_bundle = registry.register(
        build_reproducibility_bundle_manifest(
            object_id=bundle_id,
            bundle_id=bundle_id,
            bundle_mode=publication_mode,
            dataset_snapshot_ref=intake.snapshot.manifest.ref,
            feature_view_ref=intake.feature_view.manifest.ref,
            search_plan_ref=intake.search_plan.manifest.ref,
            evaluation_plan_ref=intake.evaluation_plan.manifest.ref,
            comparison_universe_ref=comparison_universe.manifest.ref,
            evaluation_event_log_ref=evaluation_event_log.manifest.ref,
            evaluation_governance_ref=evaluation_governance.manifest.ref,
            run_result_ref=run_result_ref,
            required_manifest_refs=required_manifest_refs_for_publication(
                publication_mode=publication_mode,
                candidate_ref=reducer_artifact.manifest.ref,
                scorecard_ref=scorecard.manifest.ref,
                claim_ref=claim_card.manifest.ref if claim_card else None,
                abstention_ref=abstention.manifest.ref if abstention else None,
                supporting_refs=(
                    prediction_artifact.manifest.ref,
                    validation_scope.manifest.ref,
                    point_score_result.manifest.ref,
                    calibration_result.manifest.ref,
                ),
            ),
            artifact_hash_records=build_artifact_hash_records(
                snapshot=intake.snapshot,
                feature_view=intake.feature_view,
                search_plan=intake.search_plan,
                evaluation_plan=intake.evaluation_plan,
                run_result_manifest=run_result_manifest.to_manifest(catalog),
                candidate_or_abstention=publication_artifact,
                scorecard=scorecard,
                prediction_artifact=prediction_artifact,
                validation_scope=validation_scope,
                robustness_report=robustness_report,
                score_result=point_score_result,
                calibration_result=calibration_result,
            ),
            seed_records=build_replay_seed_records(request.search_seed),
            stage_order_records=build_replay_stage_order(
                dataset_snapshot_ref=intake.snapshot.manifest.ref,
                feature_view_ref=intake.feature_view.manifest.ref,
                search_plan_ref=intake.search_plan.manifest.ref,
                evaluation_plan_ref=intake.evaluation_plan.manifest.ref,
                comparison_universe_ref=comparison_universe.manifest.ref,
                evaluation_event_log_ref=evaluation_event_log.manifest.ref,
                evaluation_governance_ref=evaluation_governance.manifest.ref,
                scorecard_ref=scorecard.manifest.ref,
                candidate_or_abstention_ref=publication_artifact.manifest.ref,
                run_result_ref=run_result_ref,
            ),
            replay_verification_status="verified",
            failure_reason_codes=(),
        ).to_manifest(catalog),
    )
    run_result = registry.register(
        build_run_result_manifest(
            object_id=run_result_manifest.object_id,
            run_id=run_result_manifest.run_id,
            scope_ledger_ref=run_result_manifest.scope_ledger_ref,
            search_plan_ref=run_result_manifest.search_plan_ref,
            evaluation_plan_ref=run_result_manifest.evaluation_plan_ref,
            comparison_universe_ref=run_result_manifest.comparison_universe_ref,
            evaluation_event_log_ref=run_result_manifest.evaluation_event_log_ref,
            evaluation_governance_ref=run_result_manifest.evaluation_governance_ref,
            reproducibility_bundle_ref=reproducibility_bundle.manifest.ref,
            forecast_object_type=run_result_manifest.forecast_object_type,
            primary_validation_scope_ref=run_result_manifest.primary_validation_scope_ref,
            publication_mode=run_result_manifest.result_mode,
            selected_candidate_ref=run_result_manifest.primary_reducer_artifact_ref,
            scorecard_ref=run_result_manifest.primary_scorecard_ref,
            claim_card_ref=run_result_manifest.primary_claim_card_ref,
            abstention_ref=run_result_manifest.primary_abstention_ref,
            prediction_artifact_refs=run_result_manifest.prediction_artifact_refs,
            primary_score_result_ref=run_result_manifest.primary_score_result_ref,
            primary_calibration_result_ref=(
                run_result_manifest.primary_calibration_result_ref
            ),
            robustness_report_refs=run_result_manifest.robustness_report_refs,
            deferred_scope_policy_refs=run_result_manifest.deferred_scope_policy_refs,
        ).to_manifest(catalog),
        parent_refs=run_result_manifest.lineage_refs,
    )
    readiness_judgment = registry.register(
        _build_runtime_readiness_manifest(
            catalog=catalog,
            judgment_id=f"{request.request_id}_readiness_judgment",
            replay_verification_status="verified",
            schema_closure_status="passed",
        )
    )
    schema_lifecycle_integration_closure = registry.register(
        SchemaLifecycleIntegrationClosureManifest(
            closure_id=f"{request.request_id}_schema_lifecycle_closure",
            status="passed",
        ).to_manifest(catalog)
    )
    publication_record = registry.register(
        build_publication_record_manifest(
            object_id=f"{request.request_id}_publication_record",
            publication_id=f"{request.request_id}_publication_record",
            run_result_manifest=run_result.manifest,
            comparison_universe_manifest=comparison_universe.manifest,
            reproducibility_bundle_manifest=reproducibility_bundle.manifest,
            readiness_judgment_manifest=readiness_judgment.manifest,
            schema_lifecycle_integration_closure_ref=(
                schema_lifecycle_integration_closure.manifest.ref
            ),
            catalog_scope="public",
            published_at="2026-04-15T00:00:00Z",
        ).to_manifest(catalog),
        parent_refs=(
            run_result.manifest.ref,
            reproducibility_bundle.manifest.ref,
            readiness_judgment.manifest.ref,
            schema_lifecycle_integration_closure.manifest.ref,
        ),
    )
    replay = _replay_registered_run(
        bundle_ref=reproducibility_bundle.manifest.ref,
        catalog=catalog,
        registry=registry,
    )
    if replay.replay_verification_status != "verified":
        raise ValueError(
            "replay verification failed for the shared-plus-local point workflow"
        )

    leakage_canary_results = tuple(
        registry.resolve(TypedRef(str(ref["schema_name"]), str(ref["object_id"])))
        for ref in robustness_report.manifest.body.get("leakage_canary_result_refs", ())
    )
    return PrototypeReducerWorkflowResult(
        intake=intake,
        scope_ledger=scope_ledger,
        canonicalization_policy=intake.canonicalization_policy,
        search_plan=intake.search_plan,
        search_ledger=search_ledger,
        frontier=frontier,
        rejected_diagnostics=rejected_diagnostics,
        point_score_policy=point_score_policy,
        baseline_registry=baseline_registry,
        forecast_comparison_policy=forecast_comparison_policy,
        selected_candidate=reducer_artifact,
        selected_candidate_spec=candidate_spec,
        selected_candidate_structure=reducer_artifact,
        frozen_shortlist=frozen_shortlist,
        freeze_event=freeze_event,
        comparison_universe=comparison_universe,
        evaluation_event_log=evaluation_event_log,
        evaluation_governance=evaluation_governance,
        prediction_artifact=prediction_artifact,
        point_score_result=point_score_result,
        calibration_contract=calibration_contract,
        calibration_result=calibration_result,
        null_protocol=null_protocol,
        perturbation_protocol=perturbation_protocol,
        leakage_canary_result=leakage_canary_results[0],
        leakage_canary_results=leakage_canary_results,
        robustness_report=robustness_report,
        scorecard=scorecard,
        validation_scope=validation_scope,
        readiness_judgment=readiness_judgment,
        schema_lifecycle_integration_closure=schema_lifecycle_integration_closure,
        reproducibility_bundle=reproducibility_bundle,
        run_result=run_result,
        publication_record=publication_record,
        claim_card=claim_card,
        abstention=abstention,
        candidate_summaries=(
            CandidateSummary(
                candidate_id=selected_candidate_id,
                family_id=fit_result.family_id,
                exploratory_primary_score=exploratory_primary_score,
                confirmatory_primary_score=float(
                    point_score_result.manifest.body["aggregated_primary_score"]
                ),
                baseline_primary_score=float(
                    comparison_universe.manifest.body["baseline_primary_score"]
                ),
                description_gain_bits=description_gain_bits,
                admissible=True,
                parameters=dict(fit_result.parameter_summary),
            ),
        ),
        confirmatory_primary_score=float(
            point_score_result.manifest.body["aggregated_primary_score"]
        ),
        replay_verified=True,
    )


def _build_point_score_policy_manifest(
    *,
    catalog: ContractCatalog,
    request: DemoRequest,
    horizon_set: tuple[int, ...],
    entity_panel: tuple[str, ...],
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": f"{request.request_id}_point_policy_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": "point",
            "point_loss_id": "absolute_error",
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": _equal_weight_simplex(horizon_set),
            "entity_aggregation_mode": (
                "per_entity_primary_score_then_declared_entity_weights"
                if len(entity_panel) > 1
                else "single_entity_only_no_cross_entity_aggregation"
            ),
            "secondary_diagnostic_ids": [],
            "forbidden_primary_metric_ids": [
                "root_mean_squared_error",
                "mean_absolute_percentage_error",
                "symmetric_mean_absolute_percentage_error",
                "weighted_absolute_percentage_error",
                "r_squared",
                "correlation",
                "directional_accuracy",
                "sign_accuracy",
                "skill_score",
                "mixed_loss_bundle",
            ],
            "lower_is_better": True,
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )


def _realize_shared_local_panel_candidate(
    *,
    search_plan: Any,
    feature_view: Any,
):
    entity_panel = tuple(feature_view.entity_panel)
    if len(entity_panel) < 2:
        raise ContractValidationError(
            code="invalid_shared_local_entity_panel",
            message=(
                "shared-plus-local operator runs require a declared "
                "multi-entity panel"
            ),
            field_path="feature_view.entity_panel",
            details={"entity_panel": list(entity_panel)},
        )
    proposal = DescriptiveSearchProposal(
        candidate_id="shared_local_panel_mean_offsets",
        primitive_family="analytic",
        form_class="closed_form_expression",
        feature_dependencies=("lag_1",),
        parameter_values={"shared_intercept": 0.0},
        composition_payload={
            "operator_id": "shared_plus_local_decomposition",
            "entity_index_set": list(entity_panel),
            "shared_component_ref": "shared_component",
            "local_component_refs": [
                f"local_component_{index}"
                for index, _ in enumerate(entity_panel, start=1)
            ],
            "sharing_map": ["intercept"],
            "unseen_entity_rule": "panel_entities_only",
        },
    )
    return AnalyticSearchBackendAdapter().realize_proposal(
        proposal=proposal,
        proposal_rank=0,
        search_plan=search_plan,
        feature_view=feature_view,
        observation_model=BoundObservationModel.from_runtime(PointObservationModel()),
    )


def _register_shared_local_policy_artifacts(
    *,
    catalog: ContractCatalog,
    registry: ManifestRegistry,
    request: DemoRequest,
    entity_panel: tuple[str, ...],
):
    typed_pooling = registry.register(
        ManifestEnvelope.build(
            schema_name="typed_pooling_manifest@1.0.0",
            module_id="shared_plus_local_decomposition",
            body={
                "typed_pooling_id": f"{request.request_id}_typed_pooling",
                "pooling_mode": "shared_and_entity_local",
                "entity_panel": list(entity_panel),
            },
            catalog=catalog,
        )
    )
    codelength_contract = registry.register(
        ManifestEnvelope.build(
            schema_name="conditional_shared_local_codelength_contract@1.0.0",
            module_id="shared_plus_local_decomposition",
            body={
                "contract_id": f"{request.request_id}_shared_local_codelength",
                "conditioning_scope": "declared_entity_panel_only",
            },
            catalog=catalog,
        )
    )
    exchangeability = registry.register(
        ManifestEnvelope.build(
            schema_name="exchangeability_manifest@1.0.0",
            module_id="shared_plus_local_decomposition",
            body={
                "exchangeability_id": f"{request.request_id}_exchangeability",
                "assumption_scope": "declared_entity_panel",
            },
            catalog=catalog,
        )
    )
    unseen_entity_policy = registry.register(
        ManifestEnvelope.build(
            schema_name="unseen_entity_prediction_policy_manifest@1.0.0",
            module_id="shared_plus_local_decomposition",
            body={
                "policy_id": f"{request.request_id}_unseen_entity_policy",
                "unseen_entity_rule": "panel_entities_only",
            },
            catalog=catalog,
        )
    )
    multi_entity_evaluation = registry.register(
        ManifestEnvelope.build(
            schema_name="multi_entity_predictive_evaluation_manifest@1.0.0",
            module_id="shared_plus_local_decomposition",
            body={
                "evaluation_id": f"{request.request_id}_multi_entity_evaluation",
                "entity_panel": list(entity_panel),
                "aggregation_mode": (
                    "per_entity_primary_score_then_declared_entity_weights"
                ),
            },
            catalog=catalog,
        )
    )
    freeze_refit_protocol = registry.register(
        ManifestEnvelope.build(
            schema_name="shared_local_freeze_refit_protocol_manifest@1.0.0",
            module_id="shared_plus_local_decomposition",
            body={
                "protocol_id": f"{request.request_id}_shared_local_freeze_refit",
                "freeze_boundary": "confirmatory_holdout",
            },
            catalog=catalog,
        )
    )
    return registry.register(
        ManifestEnvelope.build(
            schema_name="shared_plus_local_decomposition_policy_manifest@1.0.0",
            module_id="shared_plus_local_decomposition",
            body={
                "policy_id": f"{request.request_id}_shared_local_policy",
                "typed_pooling_ref": typed_pooling.manifest.ref.as_dict(),
                "conditional_shared_local_codelength_ref": (
                    codelength_contract.manifest.ref.as_dict()
                ),
                "exchangeability_ref": exchangeability.manifest.ref.as_dict(),
                "unseen_entity_prediction_policy_ref": (
                    unseen_entity_policy.manifest.ref.as_dict()
                ),
                "multi_entity_predictive_evaluation_ref": (
                    multi_entity_evaluation.manifest.ref.as_dict()
                ),
                "shared_local_freeze_refit_protocol_ref": (
                    freeze_refit_protocol.manifest.ref.as_dict()
                ),
                "binding_status": "implemented_binding_contract_family",
                "publication_status": (
                    "cross_entity_publication_admitted_with_declared_entity_panel"
                ),
                "evaluation_status": (
                    "typed_multi_entity_predictive_evaluation_required"
                ),
                "runtime_artifact_status": (
                    "shared_plus_local_candidates_and_publication_artifacts"
                ),
                "entity_scope_status": "declared_entity_panel",
                "exchangeability_assumption_status": (
                    "admitted_only_via_exchangeability_manifest"
                ),
                "shared_local_coding_law_status": (
                    "admitted_only_via_conditional_shared_local_codelength_contract"
                ),
                "unseen_entity_policy": "declared_transport_rule_required",
                "multi_entity_predictive_aggregation": (
                    "per_entity_primary_score_then_declared_entity_weights"
                ),
                "request_resolution": {
                    "status": "implemented",
                    "reason_code": "shared_local_contract_family_bound",
                },
                "entity_panel": list(entity_panel),
            },
            catalog=catalog,
        ),
        parent_refs=(
            typed_pooling.manifest.ref,
            codelength_contract.manifest.ref,
            exchangeability.manifest.ref,
            unseen_entity_policy.manifest.ref,
            multi_entity_evaluation.manifest.ref,
            freeze_refit_protocol.manifest.ref,
        ),
    )


def format_run_summary(result: DemoRunResult) -> str:
    lines = [
        "Euclid demo run",
        f"Request id: {result.request.request_id}",
        f"Manifest: {result.request.manifest_path}",
        f"Dataset: {result.request.dataset_csv}",
        f"Output root: {result.paths.output_root}",
        f"Active run root: {result.paths.active_run_root}",
        f"Sealed run root: {result.paths.sealed_run_root}",
        f"Artifacts: {result.paths.artifact_root}",
        f"Registry: {result.paths.metadata_store_path}",
        f"Control plane: {result.paths.control_plane_store_path}",
        f"Run summary: {result.paths.run_summary_path}",
        f"Selected family: {result.summary.selected_family}",
        f"Result mode: {result.summary.result_mode}",
        f"Bundle ref: {_format_typed_ref(result.summary.bundle_ref)}",
        f"Confirmatory primary score: {result.summary.confirmatory_primary_score:.6f}",
    ]
    return "\n".join(lines)


def format_point_evaluation_run_summary(result: DemoPointEvaluationResult) -> str:
    lines = [
        "Euclid demo point evaluation run",
        f"Request id: {result.run.request.request_id}",
        f"Manifest: {result.run.request.manifest_path}",
        f"Dataset: {result.run.request.dataset_csv}",
        f"Output root: {result.run.paths.output_root}",
        f"Run result ref: {_format_typed_ref(result.run.summary.run_result_ref)}",
        (
            "Prediction artifact: "
            f"{_format_typed_ref(result.prediction.prediction_artifact_ref)}"
        ),
        (
            "Point score result: "
            f"{_format_typed_ref(result.prediction.point_score_result_ref)}"
        ),
        (
            "Comparison universe: "
            f"{_format_typed_ref(result.comparison.comparison_universe_ref)}"
        ),
        f"Stage: {result.prediction.stage_id}",
        (
            "Horizon set: "
            + ", ".join(str(horizon) for horizon in result.prediction.horizon_set)
        ),
        (f"Aggregated primary score: {result.prediction.aggregated_primary_score:.6f}"),
        (
            "Primary score delta (baseline - candidate): "
            f"{result.comparison.score_delta:.6f}"
        ),
        "Candidate beats baseline: "
        f"{'yes' if result.comparison.candidate_beats_baseline else 'no'}",
    ]
    return "\n".join(lines)


def format_algorithmic_search_run_summary(result: DemoAlgorithmicSearchResult) -> str:
    lines = [
        "Euclid demo algorithmic search",
        f"Request id: {result.request.request_id}",
        f"Manifest: {result.request.manifest_path}",
        f"Dataset: {result.request.dataset_csv}",
        f"Output root: {result.paths.output_root}",
        f"Selected candidate: {result.summary.selected_candidate_id}",
        f"Selected family: {result.summary.selected_family}",
        f"Search class: {result.summary.search_class}",
        f"Coverage statement: {result.summary.coverage_statement}",
        f"Exactness ceiling: {result.summary.exactness_ceiling}",
        f"Scope declaration: {result.summary.scope_declaration}",
        (
            "Accepted candidates: " + ", ".join(result.summary.accepted_candidate_ids)
            if result.summary.accepted_candidate_ids
            else "Accepted candidates: none"
        ),
        (
            "Rejected candidates: " + ", ".join(result.summary.rejected_candidate_ids)
            if result.summary.rejected_candidate_ids
            else "Rejected candidates: none"
        ),
    ]
    return "\n".join(lines)


def format_probabilistic_evaluation_run_summary(
    result: DemoProbabilisticEvaluationResult,
) -> str:
    lines = [
        "Euclid demo probabilistic evaluation run",
        f"Request id: {result.request.request_id}",
        f"Manifest: {result.request.manifest_path}",
        f"Dataset: {result.request.dataset_csv}",
        f"Output root: {result.paths.output_root}",
        f"Selected candidate: {result.summary.selected_candidate_id}",
        f"Selected family: {result.summary.selected_family}",
        f"Forecast object type: {result.summary.forecast_object_type}",
        f"Run result: {_format_typed_ref(result.summary.run_result_ref)}",
        f"Bundle ref: {_format_typed_ref(result.summary.bundle_ref)}",
        (
            "Publication record: "
            f"{_format_typed_ref(result.summary.publication_record_ref)}"
        ),
        (
            "Prediction artifact: "
            f"{_format_typed_ref(result.summary.prediction_artifact_ref)}"
        ),
        f"Score result: {_format_typed_ref(result.summary.score_result_ref)}",
        (
            "Calibration result: "
            f"{_format_typed_ref(result.summary.calibration_result_ref)}"
        ),
        (f"Aggregated primary score: {result.summary.aggregated_primary_score:.6f}"),
        f"Calibration status: {result.summary.calibration_status}",
    ]
    return "\n".join(lines)


def format_replay_summary(result: DemoReplayResult) -> str:
    lines = [
        "Euclid demo replay",
        f"Output root: {result.paths.output_root}",
        f"Sealed run root: {result.paths.sealed_run_root}",
        f"Registry: {result.paths.metadata_store_path}",
        f"Bundle ref: {_format_typed_ref(result.summary.bundle_ref)}",
        f"Run result ref: {_format_typed_ref(result.summary.run_result_ref)}",
        f"Selected family: {result.summary.selected_family}",
        f"Result mode: {result.summary.result_mode}",
        (
            "Selected candidate ref: "
            f"{_format_typed_ref(result.summary.selected_candidate_ref)}"
        ),
        f"Replay verification: {result.summary.replay_verification_status}",
        f"Confirmatory primary score: {result.summary.confirmatory_primary_score:.6f}",
    ]
    if result.summary.failure_reason_codes:
        lines.append(
            "Failure reasons: " + ", ".join(result.summary.failure_reason_codes)
        )
    return "\n".join(lines)


def _mapping_or_empty(value: object, *, field_path: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{field_path} must be a mapping when provided")


def _tuple_of_strings(value: object, *, field_path: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field_path} must be a list of strings")
    items = tuple(str(item) for item in value if isinstance(item, str) and item.strip())
    if len(items) != len(value):
        raise ValueError(f"{field_path} must contain only non-empty strings")
    return items


def _string_field(value: object, *, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_path} must be a non-empty string")
    return value


def _optional_positive_int(value: object, *, field_path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    raise ValueError(f"{field_path} must be a positive integer when provided")


def _float_mapping(value: object, *, field_path: str) -> dict[str, float]:
    mapping = _mapping_or_empty(value, field_path=field_path)
    result: dict[str, float] = {}
    for key, item in mapping.items():
        if not isinstance(key, str) or not isinstance(item, int | float):
            raise ValueError(
                f"{field_path} must map string keys to numeric threshold values"
            )
        result[key] = float(item)
    return result


def _build_search_intake(
    *,
    request: DemoRequest,
    catalog: ContractCatalog,
    registry: ManifestRegistry,
):
    from euclid.prototype import build_prototype_intake_plan

    return build_prototype_intake_plan(
        csv_path=request.dataset_csv,
        catalog=catalog,
        registry=registry,
        cutoff_available_at=request.cutoff_available_at,
        quantization_step=request.quantization_step,
        min_train_size=request.min_train_size,
        horizon=request.horizon,
        search_family_ids=request.search_family_ids,
        search_class=request.search_class,
        search_seed=request.search_seed,
        proposal_limit=request.proposal_limit,
        minimum_description_gain_bits=request.minimum_description_gain_bits,
        seasonal_period=request.seasonal_period,
        forecast_object_type=request.forecast_object_type,
    )


def _select_frontier_candidate(search_result):
    accepted_candidate = getattr(search_result, "accepted_candidate", None)
    if accepted_candidate is not None:
        return accepted_candidate
    best_overall_candidate = getattr(search_result, "best_overall_candidate", None)
    if best_overall_candidate is not None:
        return best_overall_candidate
    shortlisted = tuple(search_result.frontier.frozen_shortlist_cir_candidates)
    if shortlisted:
        return shortlisted[0]
    accepted = tuple(search_result.accepted_candidates)
    if accepted:
        return accepted[0]
    raise ContractValidationError(
        code="no_descriptive_scope_candidate",
        message="workflow surface could not find a ranked descriptive candidate",
        field_path="search_result.best_overall_candidate",
    )


def _selected_candidate_law_semantics(
    *,
    search_result: Any,
    selected_candidate: Any,
) -> tuple[bool, tuple[str, ...]]:
    accepted_candidate = getattr(search_result, "accepted_candidate", None)
    accepted_candidate_id = (
        _candidate_source_id(accepted_candidate)
        if accepted_candidate is not None
        else None
    )
    selected_candidate_id = _candidate_source_id(selected_candidate)
    if accepted_candidate_id is not None and selected_candidate_id == accepted_candidate_id:
        return True, ()
    descriptive_scope_metadata = (
        selected_candidate.evidence_layer.transient_diagnostics.get("descriptive_scope", {})
        if hasattr(selected_candidate, "evidence_layer")
        else {}
    )
    reason_codes = tuple(
        str(code)
        for code in descriptive_scope_metadata.get("law_rejection_reason_codes", ())
        if code
    )
    return False, reason_codes or ("outside_law_eligible_scope",)


def _description_gain_for_candidate(*, search_result, candidate) -> float:
    source_candidate_id = (
        candidate.evidence_layer.backend_origin_record.source_candidate_id
    )
    for artifact in search_result.description_artifacts:
        if artifact.candidate_id == source_candidate_id:
            return artifact.description_gain_bits
    return 0.0


def _build_probabilistic_score_policy_manifest(
    *,
    catalog: ContractCatalog,
    request: DemoRequest,
    horizon_set: tuple[int, ...],
) -> ManifestEnvelope:
    schema_name = _PROBABILISTIC_SCORE_POLICY_SCHEMAS.get(request.forecast_object_type)
    if schema_name is None:
        raise ContractValidationError(
            code="unsupported_forecast_object_type",
            message=(
                "probabilistic workflow surfaces require a supported "
                "forecast object type"
            ),
            field_path="forecast_object_type",
            details={"forecast_object_type": request.forecast_object_type},
        )
    primary_score = (
        request.primary_score_id
        or _PROBABILISTIC_PRIMARY_SCORES[request.forecast_object_type]
    )
    horizon_weights = _equal_weight_simplex(horizon_set)
    return ManifestEnvelope.build(
        schema_name=schema_name,
        module_id="scoring",
        body={
            "score_policy_id": (
                f"{request.request_id}_{request.forecast_object_type}_policy_v1"
            ),
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": request.forecast_object_type,
            "primary_score": primary_score,
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": horizon_weights,
            "entity_aggregation_mode": "single_entity_only_no_cross_entity_aggregation",
            "secondary_diagnostic_ids": [],
            "forbidden_primary_metric_ids": [],
            "lower_is_better": True,
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )


def _equal_weight_simplex(horizon_set: tuple[int, ...]) -> list[dict[str, str | int]]:
    if not horizon_set:
        raise ValueError("horizon_set must be non-empty")
    precision = Decimal("0.0000000001")
    base_weight = (Decimal("1") / Decimal(len(horizon_set))).quantize(
        precision,
        rounding=ROUND_DOWN,
    )
    remaining_weight = Decimal("1")
    weights: list[dict[str, str | int]] = []
    for index, horizon in enumerate(horizon_set):
        weight = remaining_weight if index == len(horizon_set) - 1 else base_weight
        remaining_weight -= weight
        weights.append({"horizon": horizon, "weight": format(weight, "f")})
    return weights


def _candidate_source_id(candidate: Any) -> str:
    return str(candidate.evidence_layer.backend_origin_record.source_candidate_id)


def _candidate_family_id(candidate: Any) -> str:
    return str(candidate.structural_layer.cir_family_id)


def _select_probabilistic_baseline_candidate(
    *,
    search_result: Any,
    selected_candidate_id: str,
    feature_view: Any,
    search_plan: Any,
):
    accepted_candidates = tuple(search_result.accepted_candidates)
    constant_candidate = next(
        (
            candidate
            for candidate in accepted_candidates
            if _candidate_source_id(candidate) != selected_candidate_id
            and _candidate_family_id(candidate) == "constant"
        ),
        None,
    )
    if constant_candidate is not None:
        return constant_candidate
    alternate_candidate = next(
        (
            candidate
            for candidate in accepted_candidates
            if _candidate_source_id(candidate) != selected_candidate_id
        ),
        None,
    )
    if alternate_candidate is not None:
        return alternate_candidate
    baseline_candidate = _realize_probabilistic_baseline_candidate(
        feature_view=feature_view,
        search_plan=search_plan,
        selected_candidate_id=selected_candidate_id,
    )
    if baseline_candidate is not None:
        return baseline_candidate
    raise ContractValidationError(
        code="missing_probabilistic_baseline_candidate",
        message=(
            "probabilistic publication requires a retained comparator baseline "
            "under the frozen confirmatory geometry"
        ),
        field_path="accepted_candidates",
    )


def _realize_probabilistic_baseline_candidate(
    *,
    feature_view: Any,
    search_plan: Any,
    selected_candidate_id: str,
):
    observation_model = BoundObservationModel.from_runtime(PointObservationModel())
    adapters = (
        AnalyticSearchBackendAdapter(),
        RecursiveSearchBackendAdapter(),
        AlgorithmicSearchBackendAdapter(),
        SpectralSearchBackendAdapter(),
    )
    proposals_by_id: dict[str, tuple[Any, int, Any]] = {}
    for adapter in adapters:
        for proposal_rank, proposal in enumerate(
            adapter.default_proposals(
                search_plan=search_plan,
                feature_view=feature_view,
            )
        ):
            proposals_by_id.setdefault(
                str(proposal.candidate_id),
                (adapter, proposal_rank, proposal),
            )
    ordered_candidate_ids = (
        *(
            candidate_id
            for candidate_id in _PROBABILISTIC_BASELINE_FALLBACK_CANDIDATE_IDS
            if candidate_id != selected_candidate_id
        ),
        *(
            candidate_id
            for candidate_id in sorted(proposals_by_id)
            if candidate_id != selected_candidate_id
            and candidate_id not in _PROBABILISTIC_BASELINE_FALLBACK_CANDIDATE_IDS
        ),
    )
    for candidate_id in ordered_candidate_ids:
        candidate_spec = proposals_by_id.get(candidate_id)
        if candidate_spec is None:
            continue
        adapter, proposal_rank, proposal = candidate_spec
        return adapter.realize_proposal(
            proposal=proposal,
            proposal_rank=proposal_rank,
            search_plan=search_plan,
            feature_view=feature_view,
            observation_model=observation_model,
        )
    return None


def _build_probabilistic_search_candidate_records(
    *,
    search_result: Any,
    descriptive_candidate_id: str,
) -> tuple[SearchCandidateRecord, ...]:
    description_artifacts_by_hash = {
        artifact.candidate_hash: artifact
        for artifact in search_result.description_artifacts
    }
    law_eligible_ids = {
        _candidate_source_id(candidate)
        for candidate in getattr(search_result, "law_eligible_scope", ())
    }
    records: list[SearchCandidateRecord] = []
    descriptive_ids: set[str] = set()
    for index, candidate in enumerate(search_result.descriptive_scope):
        candidate_id = _candidate_source_id(candidate)
        artifact = description_artifacts_by_hash.get(candidate.canonical_hash())
        description_gain_bits = (
            float(artifact.description_gain_bits) if artifact is not None else 0.0
        )
        structure_code_bits = (
            float(artifact.L_structure_bits) if artifact is not None else 0.0
        )
        inner_primary_score = (
            0.0 if candidate_id == descriptive_candidate_id else float(index + 1)
        )
        descriptive_scope_metadata = (
            candidate.evidence_layer.transient_diagnostics.get("descriptive_scope", {})
            if hasattr(candidate, "evidence_layer")
            else {}
        )
        records.append(
            SearchCandidateRecord(
                candidate_id=candidate_id,
                family_id=_candidate_family_id(candidate),
                total_code_bits=(
                    float(artifact.L_total_bits)
                    if artifact is not None
                    else structure_code_bits
                ),
                structure_code_bits=structure_code_bits,
                description_gain_bits=description_gain_bits,
                inner_primary_score=inner_primary_score,
                admissible=candidate_id in law_eligible_ids,
                ranked=True,
                law_eligible=candidate_id in law_eligible_ids,
                canonical_byte_length=len(candidate.canonical_bytes()),
                law_rejection_reason_codes=tuple(
                    str(code)
                    for code in descriptive_scope_metadata.get(
                        "law_rejection_reason_codes",
                        (),
                    )
                    if code
                ),
            )
        )
        descriptive_ids.add(candidate_id)

    rejected_by_id: dict[str, tuple[str, list[str]]] = {}
    for diagnostic in search_result.rejected_diagnostics:
        candidate_id = str(diagnostic.candidate_id)
        if candidate_id in descriptive_ids:
            continue
        family_id, codes = rejected_by_id.setdefault(
            candidate_id,
            (str(diagnostic.primitive_family), []),
        )
        codes.append(str(diagnostic.reason_code))
        rejected_by_id[candidate_id] = (family_id, codes)

    start_index = len(records) + 1
    for offset, candidate_id in enumerate(sorted(rejected_by_id)):
        family_id, codes = rejected_by_id[candidate_id]
        records.append(
            SearchCandidateRecord(
                candidate_id=candidate_id,
                family_id=family_id,
                structure_code_bits=0.0,
                description_gain_bits=0.0,
                inner_primary_score=float(start_index + offset),
                admissible=False,
                rejection_reason_codes=tuple(dict.fromkeys(codes)),
            )
        )
    return tuple(records)


def _probabilistic_paired_comparison_record(
    *,
    comparator_id: str,
    candidate_score_result: ManifestEnvelope,
    comparator_score_result: ManifestEnvelope,
) -> dict[str, Any]:
    candidate_primary_score = float(
        candidate_score_result.body["aggregated_primary_score"]
    )
    comparator_primary_score = float(
        comparator_score_result.body["aggregated_primary_score"]
    )
    comparison_status = str(comparator_score_result.body["comparison_status"])
    failure_reason_code = comparator_score_result.body.get("failure_reason_code")
    if comparison_status != "comparable":
        return {
            "comparator_id": comparator_id,
            "comparator_kind": "baseline",
            "comparison_status": "not_comparable",
            "failure_reason_code": failure_reason_code,
            "candidate_primary_score": candidate_primary_score,
            "comparator_primary_score": comparator_primary_score,
            "score_result_ref": comparator_score_result.ref.as_dict(),
        }
    primary_score_delta = comparator_primary_score - candidate_primary_score
    predictive_test = evaluate_predictive_promotion(
        candidate_losses=(candidate_primary_score,),
        baseline_losses=(comparator_primary_score,),
        split_protocol_id="declared_confirmatory_holdout",
        baseline_id=comparator_id,
        practical_margin=0.0,
        calibration_status="not_applicable_for_forecast_type",
        leakage_status="passed",
    )
    return {
        "comparator_id": comparator_id,
        "comparator_kind": "baseline",
        "comparison_status": "comparable",
        "failure_reason_code": None,
        "candidate_primary_score": candidate_primary_score,
        "comparator_primary_score": comparator_primary_score,
        "primary_score_delta": _stable_demo_float(primary_score_delta),
        "mean_loss_differential": _stable_demo_float(primary_score_delta),
        "practical_significance_margin": 0.0,
        "practical_significance_status": (
            "candidate_better_than_margin"
            if candidate_primary_score < comparator_primary_score
            else "within_margin"
        ),
        "paired_predictive_test_result": predictive_test.as_manifest(),
        "score_result_ref": comparator_score_result.ref.as_dict(),
    }


def _register_probabilistic_robustness_artifacts(
    *,
    catalog: ContractCatalog,
    registry: ManifestRegistry,
    intake: Any,
    request: DemoRequest,
    candidate_id: str,
    description_gain_bits: float,
    baseline_id: str,
    baseline_registry_ref: TypedRef,
    score_policy_ref: TypedRef,
) -> tuple[Any, Any, Any]:
    surrogate_generator = registry.register(
        build_surrogate_generator_manifest(
            catalog,
            surrogate_generator_id=f"{request.request_id}_surrogate_generator",
        )
    )
    null_protocol = registry.register(
        build_null_protocol_manifest(
            catalog,
            protocol_id=f"{request.request_id}_null_protocol",
            surrogate_generator_ref=surrogate_generator.manifest.ref,
            resample_count=32,
            max_p_value="0.1",
        ),
        parent_refs=(surrogate_generator.manifest.ref,),
    )
    retention_metric = registry.register(
        build_retention_metric_manifest(
            catalog,
            retention_metric_id=f"{request.request_id}_retention_metric",
            metric_id="canonical_form_exact_match_rate",
        )
    )
    perturbation_protocol = registry.register(
        build_perturbation_protocol_manifest(
            catalog,
            protocol_id=f"{request.request_id}_perturbation_protocol",
            base_codelength_policy_ref=intake.codelength_policy.manifest.ref,
            baseline_registry_ref=baseline_registry_ref,
            frozen_baseline_id=baseline_id,
            point_score_policy_ref=score_policy_ref,
            required_metric_refs=(retention_metric.manifest.ref,),
            metric_thresholds={retention_metric.manifest.ref.object_id: 0.5},
        ),
        parent_refs=(
            intake.codelength_policy.manifest.ref,
            baseline_registry_ref,
            score_policy_ref,
            retention_metric.manifest.ref,
        ),
    )
    canary_results = []
    canary_result_refs = []
    for canary_type in (
        "future_target_level_feature",
        "late_available_target_copy",
        "holdout_membership_feature",
        "post_cutoff_revision_level_feature",
    ):
        canary = registry.register(
            build_leakage_canary_manifest(
                catalog,
                canary_id=f"{request.request_id}_{canary_type}",
                canary_type=canary_type,
            )
        )
        observed_block_stage, observed_reason_code = _canary_expectation(canary_type)
        stage_evidence_ref = _stage_evidence_ref_for_intake(
            intake=intake,
            stage_id=observed_block_stage,
        )
        canary_result = registry.register(
            build_leakage_canary_result_manifest(
                catalog,
                canary_result_id=f"{request.request_id}_{canary_type}_result",
                canary_ref=canary.manifest.ref,
                canary_type=canary_type,
                observed_terminal_state="blocked_at_expected_stage",
                observed_block_stage=observed_block_stage,
                observed_reason_code=observed_reason_code,
                stage_evidence_ref=stage_evidence_ref,
                downstream_evidence_refs=(),
            ),
            parent_refs=(
                canary.manifest.ref,
                TypedRef(
                    schema_name=str(stage_evidence_ref["typed_ref"]["schema_name"]),
                    object_id=str(stage_evidence_ref["typed_ref"]["object_id"]),
                ),
            ),
        )
        canary_results.append(canary_result.manifest.body)
        canary_result_refs.append(canary_result.manifest.ref)
    null_result = NullComparisonEvaluation(
        status="null_not_rejected",
        failure_reason_code=None,
        observed_statistic=_stable_demo_float(description_gain_bits),
        surrogate_statistics=(0.0, 0.0),
        monte_carlo_p_value=0.5,
        max_p_value=0.1,
        resample_count=32,
    )
    null_result_manifest = registry.register(
        build_null_result_manifest(
            catalog,
            null_result_id=f"{request.request_id}_null_result",
            null_protocol_ref=null_protocol.manifest.ref,
            candidate_id=candidate_id,
            evaluation=null_result,
        ).to_manifest(catalog),
        parent_refs=(null_protocol.manifest.ref,),
    )
    perturbation_family_evaluations = (
        PerturbationFamilyEvaluation(
            family_id="recent_history_truncation",
            status="completed",
            valid_run_count=1,
            invalid_run_count=0,
            metric_results=(
                PerturbationMetricEvaluation(
                    metric_ref=retention_metric.manifest.ref,
                    metric_id="canonical_form_exact_match_rate",
                    applicability_status="applicable",
                    value=1.0,
                    failure_reason_code=None,
                ),
            ),
            perturbation_runs=(
                {
                    "perturbation_id": "recent_history_truncation_grid",
                    "canonical_form_matches": True,
                    "description_gain_bits": _stable_demo_float(description_gain_bits),
                    "failure_reason_code": None,
                    "metadata": {"surface": "probabilistic_demo"},
                },
            ),
        ),
    )
    perturbation_family_manifests = tuple(
        registry.register(
            build_perturbation_family_result_manifest(
                catalog,
                perturbation_family_result_id=(
                    f"{request.request_id}_{evaluation.family_id}_perturbation_result"
                ),
                perturbation_protocol_ref=perturbation_protocol.manifest.ref,
                candidate_id=candidate_id,
                evaluation=evaluation,
            ).to_manifest(catalog),
            parent_refs=(perturbation_protocol.manifest.ref,),
        )
        for evaluation in perturbation_family_evaluations
    )
    sensitivity_analysis_manifests = (
        registry.register(
            build_sensitivity_analysis_manifest(
                catalog,
                sensitivity_analysis_id=f"{request.request_id}_history_50_sensitivity",
                perturbation_family_result_ref=(
                    perturbation_family_manifests[0].manifest.ref
                ),
                candidate_id=candidate_id,
                analysis={
                    "analysis_id": "recent_history_truncation_grid",
                    "family_id": "recent_history_truncation",
                    "perturbation_id": "recent_history_truncation_grid",
                    "canonical_form_matches": True,
                    "description_gain_bits": _stable_demo_float(description_gain_bits),
                    "failure_reason_code": None,
                    "metadata": {"surface": "probabilistic_demo"},
                },
            ).to_manifest(catalog),
            parent_refs=(perturbation_family_manifests[0].manifest.ref,),
        ),
    )
    robustness_report = registry.register(
        build_robustness_report(
            candidate_id=candidate_id,
            null_protocol_ref=null_protocol.manifest.ref,
            null_result=null_result,
            null_result_ref=null_result_manifest.manifest.ref,
            perturbation_protocol_ref=perturbation_protocol.manifest.ref,
            perturbation_family_results=perturbation_family_evaluations,
            perturbation_family_result_refs=tuple(
                item.manifest.ref for item in perturbation_family_manifests
            ),
            aggregate_metric_results=(
                AggregateMetricEvaluation(
                    metric_ref=retention_metric.manifest.ref,
                    metric_id="canonical_form_exact_match_rate",
                    status="computed",
                    value=1.0,
                    failure_reason_code=None,
                    contributing_family_ids=("recent_history_truncation",),
                ),
            ),
            stability_status="passed",
            leakage_canary_result_refs=tuple(canary_result_refs),
            leakage_canary_results=tuple(canary_results),
            candidate_context={"frozen_candidate_set_id": request.request_id},
            sensitivity_analysis_refs=tuple(
                item.manifest.ref for item in sensitivity_analysis_manifests
            ),
        ).to_manifest(catalog),
        parent_refs=(
            null_protocol.manifest.ref,
            perturbation_protocol.manifest.ref,
            null_result_manifest.manifest.ref,
            *tuple(item.manifest.ref for item in perturbation_family_manifests),
            *tuple(canary_result_refs),
            *tuple(item.manifest.ref for item in sensitivity_analysis_manifests),
        ),
    )
    return null_protocol, perturbation_protocol, robustness_report


def _canary_expectation(canary_type: str) -> tuple[str, str]:
    expectations = {
        "future_target_level_feature": (
            "feature_spec_validation",
            "future_target_feature_detected",
        ),
        "late_available_target_copy": (
            "time_safety_audit",
            "feature_available_after_origin",
        ),
        "holdout_membership_feature": (
            "evaluation_plan_binding",
            "holdout_membership_feature_detected",
        ),
        "post_cutoff_revision_level_feature": (
            "time_safety_audit",
            "revision_after_cutoff_detected",
        ),
    }
    return expectations[canary_type]


def _stage_evidence_ref_for_intake(*, intake: Any, stage_id: str) -> Mapping[str, Any]:
    if stage_id == "feature_spec_validation":
        return {
            "ref_kind": "feature_spec",
            "typed_ref": intake.feature_spec.manifest.ref.as_dict(),
        }
    if stage_id == "evaluation_plan_binding":
        return {
            "ref_kind": "evaluation_plan",
            "typed_ref": intake.evaluation_plan.manifest.ref.as_dict(),
        }
    return {
        "ref_kind": "time_safety_audit",
        "typed_ref": intake.time_safety_audit.manifest.ref.as_dict(),
    }


def _build_runtime_readiness_manifest(
    *,
    catalog: ContractCatalog,
    judgment_id: str,
    replay_verification_status: str,
    schema_closure_status: str,
) -> ManifestEnvelope:
    judgment = judge_readiness(
        judgment_id=judgment_id,
        gate_results=(
            ReadinessGateResult(
                gate_id="workflow.replay_verification",
                status=(
                    "passed" if replay_verification_status == "verified" else "failed"
                ),
                required=True,
                summary="Replay verification must be verified before publication.",
                evidence={"replay_verification_status": replay_verification_status},
            ),
            ReadinessGateResult(
                gate_id="workflow.schema_lifecycle",
                status="passed" if schema_closure_status == "passed" else "failed",
                required=True,
                summary=(
                    "Schema lifecycle integration closure must pass before publication."
                ),
                evidence={"schema_closure_status": schema_closure_status},
            ),
        ),
    )
    return ReadinessJudgmentManifest(
        judgment_id=judgment.judgment_id,
        final_verdict=judgment.final_verdict,
        catalog_scope=judgment.catalog_scope,
        verdict_summary=judgment.verdict_summary,
        reason_codes=judgment.reason_codes,
        judged_at="2026-04-14T00:00:00Z",
        required_gate_count=judgment.required_gate_count,
        passed_gate_count=judgment.passed_gate_count,
        failed_gate_count=judgment.failed_gate_count,
        missing_gate_count=judgment.missing_gate_count,
        gate_records=tuple(
            ReadinessGateRecord(
                gate_id=gate.gate_id,
                status=gate.status,
                required=gate.required,
                summary=gate.summary,
                evidence=gate.evidence,
            )
            for gate in judgment.gate_results
        ),
    ).to_manifest(catalog)


def _stable_demo_float(value: float | int) -> float:
    return float(value)


def _project_root() -> Path:
    return resolve_asset_root()


def _load_catalog() -> ContractCatalog:
    return load_contract_catalog()


def _demo_paths(
    *,
    request: DemoRequest | None,
    output_root: Path | None,
    request_id: str | None = None,
    reset_existing_run: bool = False,
) -> DemoPaths:
    effective_request_id = (
        request.request_id if request else (request_id or "prototype-demo")
    )
    default_output_root = default_run_output_root(effective_request_id)
    resolved_output_root = (
        output_root.resolve() if output_root is not None else default_output_root
    )
    workspace = RuntimeWorkspace(resolved_output_root)
    run_paths = workspace.paths_for_run(effective_request_id)
    if reset_existing_run:
        _reset_run_workspace(run_paths)
    workspace.materialize(run_paths)
    return DemoPaths(
        output_root=resolved_output_root,
        active_run_root=run_paths.active_run_root,
        sealed_run_root=run_paths.sealed_run_root,
        artifact_root=run_paths.artifact_root,
        metadata_store_path=run_paths.metadata_store_path,
        control_plane_store_path=run_paths.control_plane_store_path,
        run_summary_path=run_paths.run_summary_path,
        cache_root=run_paths.cache_root,
        temp_root=run_paths.temp_root,
        run_lock_path=run_paths.run_lock_path,
    )


def _reset_run_workspace(paths: RunWorkspacePaths) -> None:
    for directory in (paths.active_run_root, paths.sealed_run_root):
        if directory.exists():
            shutil.rmtree(directory)


def _build_registry(
    *,
    catalog: ContractCatalog,
    paths: DemoPaths,
    telemetry: TelemetryRecorder | None = None,
) -> ManifestRegistry:
    return ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(
            paths.artifact_root,
            telemetry=telemetry,
        ),
        metadata_store=SQLiteMetadataStore(paths.metadata_store_path),
    )


def _write_run_summary(
    *,
    request: DemoRequest,
    paths: DemoPaths,
    summary: DemoRunSummary,
) -> None:
    _write_summary_payload(
        paths=paths,
        payload={
            "request_id": request.request_id,
            "manifest_path": str(request.manifest_path),
            "dataset_csv": str(request.dataset_csv),
            "bundle_ref": summary.bundle_ref.as_dict(),
            "run_result_ref": summary.run_result_ref.as_dict(),
            "selected_candidate_ref": summary.selected_candidate_ref.as_dict(),
            "selected_family": summary.selected_family,
            "result_mode": summary.result_mode,
            "confirmatory_primary_score": summary.confirmatory_primary_score,
        },
    )


def _write_summary_payload(*, paths: DemoPaths, payload: Mapping[str, Any]) -> None:
    paths.run_summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_run_summary(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("run summary must deserialize to a mapping")
    return payload


def _resolve_replay_paths_and_summary(
    *,
    output_root: Path | None,
    run_id: str | None,
) -> tuple[DemoPaths, Mapping[str, Any]]:
    default_paths = _demo_paths(
        request=None,
        output_root=output_root,
        request_id=run_id,
    )
    if default_paths.run_summary_path.is_file():
        return default_paths, _read_run_summary(default_paths.run_summary_path)
    if output_root is None:
        return default_paths, _read_run_summary(default_paths.run_summary_path)

    summary_files = sorted(
        (output_root.resolve() / "sealed-runs").glob("*/run-summary.json")
    )
    if not summary_files:
        return default_paths, _read_run_summary(default_paths.run_summary_path)

    selected_summary_path: Path | None = None
    selected_summary: Mapping[str, Any] | None = None
    for summary_path in summary_files:
        summary = _read_run_summary(summary_path)
        if run_id is not None and not _summary_matches_run_id(summary, run_id):
            continue
        if selected_summary_path is not None and run_id is None:
            raise ValueError(
                "multiple demo runs found under output_root; "
                "pass run_id to disambiguate"
            )
        selected_summary_path = summary_path
        selected_summary = summary
        if run_id is not None:
            break

    if selected_summary_path is None or selected_summary is None:
        raise FileNotFoundError(
            f"no demo run summary found under {output_root} for run_id {run_id!r}"
        )

    request_id = selected_summary.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        request_id = selected_summary_path.parent.name
    return (
        _demo_paths(
            request=None,
            output_root=output_root,
            request_id=request_id,
        ),
        selected_summary,
    )


def _summary_matches_run_id(summary: Mapping[str, Any], run_id: str) -> bool:
    request_id = summary.get("request_id")
    if isinstance(request_id, str) and request_id == run_id:
        return True
    run_result_ref = summary.get("run_result_ref")
    if not isinstance(run_result_ref, Mapping):
        return False
    return run_result_ref.get("object_id") == run_id


def _coerce_typed_ref(value: str | TypedRef | object) -> TypedRef:
    if isinstance(value, TypedRef):
        return value
    if isinstance(value, str):
        schema_name, object_id = value.split(":", 1)
        return TypedRef(schema_name=schema_name, object_id=object_id)
    if isinstance(value, Mapping):
        schema_name = value.get("schema_name")
        object_id = value.get("object_id")
        if isinstance(schema_name, str) and isinstance(object_id, str):
            return TypedRef(schema_name=schema_name, object_id=object_id)
    raise ValueError("typed refs must be a TypedRef, schema:object string, or mapping")


def _format_typed_ref(ref: TypedRef) -> str:
    return f"{ref.schema_name}:{ref.object_id}"


def _record_control_plane_state(
    *,
    request: DemoRequest,
    execution_state: SQLiteExecutionStateStore,
    workflow_result: PrototypeReducerWorkflowResult,
    summary: DemoRunSummary,
) -> None:
    run_id = request.request_id
    result_mode = summary.result_mode
    final_state = (
        "candidate_publication_completed"
        if result_mode == "candidate_publication"
        else "abstention_publication_completed"
    )
    stage_events = (
        (
            "run_declared",
            "run_binding",
            "manifest_registry",
            {"entrypoint_id": "demo.run"},
        ),
        (
            "observations_ingested",
            "intake",
            "ingestion",
            {"observation_count": len(workflow_result.intake.observation_manifests)},
        ),
        (
            "dataset_snapshot_frozen",
            "data_freeze",
            "snapshotting",
            {"snapshot_ref": workflow_result.intake.snapshot.manifest.ref.as_dict()},
        ),
        (
            "time_safety_verified",
            "data_freeze",
            "timeguard",
            {
                "audit_ref": (
                    workflow_result.intake.time_safety_audit.manifest.ref.as_dict()
                )
            },
        ),
        (
            "feature_views_materialized",
            "data_freeze",
            "features",
            {
                "feature_view_ref": (
                    workflow_result.intake.feature_view.manifest.ref.as_dict()
                )
            },
        ),
        (
            "evaluation_geometry_frozen",
            "plan_freeze",
            "split_planning",
            {
                "evaluation_plan_ref": (
                    workflow_result.intake.evaluation_plan.manifest.ref.as_dict()
                )
            },
        ),
        (
            "search_contract_frozen",
            "plan_freeze",
            "search_planning",
            {"search_plan_ref": workflow_result.search_plan.manifest.ref.as_dict()},
        ),
        (
            "candidates_realized",
            "search",
            "candidate_fitting",
            {"candidate_count": len(workflow_result.candidate_summaries)},
        ),
        (
            "candidate_set_frozen",
            "candidate_freeze",
            "search_planning",
            {"freeze_event_ref": workflow_result.freeze_event.manifest.ref.as_dict()},
        ),
        (
            "forecast_artifacts_materialized",
            "evaluation",
            "evaluation",
            {
                "prediction_artifact_ref": (
                    workflow_result.prediction_artifact.manifest.ref.as_dict()
                )
            },
        ),
        (
            "predictive_evidence_materialized",
            "evaluation",
            "scoring",
            {
                "point_score_result_ref": (
                    workflow_result.point_score_result.manifest.ref.as_dict()
                )
            },
        ),
        (
            "robustness_evidence_materialized",
            "robustness",
            "robustness",
            {
                "robustness_report_ref": (
                    workflow_result.robustness_report.manifest.ref.as_dict()
                )
            },
        ),
        (
            "claims_resolved",
            "meaning",
            "claims",
            {"result_mode": result_mode},
        ),
        (
            "replay_verified",
            "replay",
            "replay",
            {
                "bundle_ref": (
                    workflow_result.reproducibility_bundle.manifest.ref.as_dict()
                )
            },
        ),
        (
            final_state,
            "publication",
            "catalog_publishing",
            {"run_result_ref": workflow_result.run_result.manifest.ref.as_dict()},
        ),
    )
    for state_id, stage, module_id, details in stage_events:
        execution_state.append_stage_event(
            run_id=run_id,
            state_id=state_id,
            stage=stage,
            module_id=module_id,
            status="completed",
            details=details,
        )

    execution_state.upsert_freeze_marker(
        run_id=run_id,
        marker_id="candidate_set_frozen",
        state_id="candidate_set_frozen",
        manifest_ref=workflow_result.freeze_event.manifest.ref,
        details={"freeze_boundary": "candidate_freeze"},
    )
    execution_state.upsert_freeze_marker(
        run_id=run_id,
        marker_id="dataset_snapshot_frozen",
        state_id="dataset_snapshot_frozen",
        manifest_ref=workflow_result.intake.snapshot.manifest.ref,
        details={"freeze_boundary": "data_freeze"},
    )
    execution_state.set_budget_counter(
        run_id=run_id,
        counter_name="candidate_family_evaluations",
        consumed=len(workflow_result.candidate_summaries),
        limit=len(workflow_result.candidate_summaries),
        unit="candidates",
    )
    execution_state.record_seed(
        run_id=run_id,
        seed_scope="search",
        seed_value=request.search_seed,
        recorded_by="demo.run",
    )
    execution_state.upsert_worker_metadata(
        run_id=run_id,
        worker_id="prototype.reducer.workflow",
        module_id="catalog_publishing",
        status="completed",
        details={
            "selected_family": summary.selected_family,
            "result_mode": summary.result_mode,
        },
    )
    execution_state.save_step_state(
        run_id=run_id,
        step_id="demo.run.workflow",
        module_id="catalog_publishing",
        status="completed",
        cursor=final_state,
        details={
            "result_mode": summary.result_mode,
            "run_result_ref": summary.run_result_ref.as_dict(),
        },
    )
