from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from euclid.adapters.algorithmic_dsl import enumerate_algorithmic_proposal_specs
from euclid.adapters.portfolio import normalize_submitter_finalist
from euclid.benchmarks.manifests import BenchmarkTaskManifest
from euclid.cir.models import CandidateIntermediateRepresentation
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.modules.features import FeatureView
from euclid.modules.replay import (
    build_portfolio_replay_contract,
    verify_portfolio_replay_contract,
)
from euclid.modules.search_planning import build_search_plan
from euclid.modules.snapshotting import FrozenDatasetSnapshot
from euclid.modules.split_planning import EvaluationPlan
from euclid.performance import TelemetryRecorder
from euclid.search.backends import (
    AlgorithmicSearchBackendAdapter,
    AnalyticSearchBackendAdapter,
    DescriptiveSearchBackendAdapter,
    DescriptiveSearchProposal,
    RecursiveSearchBackendAdapter,
    RejectedSearchDiagnostic,
    SpectralSearchBackendAdapter,
    _select_attempted_proposals,
    run_descriptive_search_backends,
)
from euclid.search.portfolio import (
    PortfolioCandidateLedgerEntry,
    select_comparable_portfolio_winner,
)

ANALYTIC_BACKEND_SUBMITTER_ID = "analytic_backend"
RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID = "recursive_spectral_backend"
ALGORITHMIC_SEARCH_SUBMITTER_ID = "algorithmic_search_backend"
PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID = "portfolio_orchestrator"

REQUIRED_BENCHMARK_SUBMITTER_IDS = (
    ANALYTIC_BACKEND_SUBMITTER_ID,
    RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
    ALGORITHMIC_SEARCH_SUBMITTER_ID,
    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
)

_SUBMITTER_DEFINITIONS: dict[str, dict[str, Any]] = {
    ANALYTIC_BACKEND_SUBMITTER_ID: {
        "submitter_class": "decomposition",
        "candidate_family_ids": (
            "analytic_intercept",
            "analytic_lag1_affine",
        ),
        "adapter_factories": (AnalyticSearchBackendAdapter,),
    },
    RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID: {
        "submitter_class": "sparse_library",
        "candidate_family_ids": (
            "recursive_level_smoother",
            "recursive_running_mean",
            "spectral_harmonic_1",
            "spectral_harmonic_2",
        ),
        "adapter_factories": (
            RecursiveSearchBackendAdapter,
            SpectralSearchBackendAdapter,
        ),
    },
    ALGORITHMIC_SEARCH_SUBMITTER_ID: {
        "submitter_class": "bounded_grammar",
        "candidate_family_ids": (
            "algorithmic_last_observation",
            "algorithmic_running_half_average",
        ),
        "adapter_factories": (AlgorithmicSearchBackendAdapter,),
    },
    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID: {
        "submitter_class": "portfolio",
        "child_submitter_ids": (
            ANALYTIC_BACKEND_SUBMITTER_ID,
            RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
            ALGORITHMIC_SEARCH_SUBMITTER_ID,
        ),
    },
}


@dataclass(frozen=True)
class BenchmarkHarnessContext:
    task_manifest: BenchmarkTaskManifest
    snapshot: FrozenDatasetSnapshot
    feature_view: FeatureView
    evaluation_plan: EvaluationPlan
    canonicalization_policy_ref: TypedRef
    codelength_policy_ref: TypedRef
    reference_description_policy_ref: TypedRef
    observation_model_ref: TypedRef
    search_class: str = "bounded_heuristic"
    seasonal_period: int | None = None
    minimum_description_gain_bits: float | None = None
    project_root: Path | None = None
    proposal_specs: tuple[DescriptiveSearchProposal, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.evaluation_plan.forecast_object_type
            != self.task_manifest.frozen_protocol.forecast_object_type
        ):
            raise ContractValidationError(
                code="benchmark_forecast_object_mismatch",
                message=(
                    "evaluation_plan must preserve the benchmark task "
                    "forecast_object_type"
                ),
                field_path="evaluation_plan.forecast_object_type",
                details={
                    "task_manifest_forecast_object_type": (
                        self.task_manifest.frozen_protocol.forecast_object_type
                    ),
                    "evaluation_plan_forecast_object_type": (
                        self.evaluation_plan.forecast_object_type
                    ),
                },
            )
        self.feature_view.require_stage_reuse("search")

    @property
    def protocol_contract(self) -> dict[str, Any]:
        frozen = self.task_manifest.frozen_protocol
        return {
            "task_id": self.task_manifest.task_id,
            "track_id": self.task_manifest.track_id,
            "search_class": self.search_class,
            "dataset_ref": frozen.dataset_ref,
            "snapshot_policy": dict(frozen.snapshot_policy),
            "target_transform_policy": dict(frozen.target_transform_policy),
            "quantization_policy": dict(frozen.quantization_policy),
            "observation_model_policy": dict(frozen.observation_model_policy),
            "forecast_object_type": frozen.forecast_object_type,
            "composition_operators": list(self.task_manifest.composition_operators),
            "score_policy": dict(frozen.score_policy),
            "calibration_policy": (
                dict(frozen.calibration_policy)
                if frozen.calibration_policy is not None
                else None
            ),
            "budget_policy": dict(frozen.budget_policy),
            "seed_policy": dict(frozen.seed_policy),
            "replay_policy": dict(frozen.replay_policy),
            "abstention_policy": dict(self.task_manifest.abstention_policy),
            "snapshot_series_id": self.snapshot.series_id,
            "snapshot_row_count": self.snapshot.row_count,
        }

    def build_search_plan(
        self,
        *,
        submitter_id: str,
        candidate_family_ids: Sequence[str],
    ):
        budget_policy = dict(self.task_manifest.frozen_protocol.budget_policy)
        proposal_limit = int(
            budget_policy.get("candidate_limit", len(tuple(candidate_family_ids)))
        )
        wall_clock_budget_seconds = int(budget_policy.get("wall_clock_seconds", 1))
        seed_value = str(
            self.task_manifest.frozen_protocol.seed_policy.get("seed", "0")
        )
        return build_search_plan(
            evaluation_plan=self.evaluation_plan,
            canonicalization_policy_ref=self.canonicalization_policy_ref,
            codelength_policy_ref=self.codelength_policy_ref,
            reference_description_policy_ref=self.reference_description_policy_ref,
            observation_model_ref=self.observation_model_ref,
            candidate_family_ids=tuple(candidate_family_ids),
            search_plan_id=f"{self.task_manifest.task_id}__{submitter_id}__search_plan",
            search_class=self.search_class,
            random_seed=seed_value,
            proposal_limit=proposal_limit,
            wall_clock_budget_seconds=wall_clock_budget_seconds,
            seasonal_period=self.seasonal_period,
            minimum_description_gain_bits=self.minimum_description_gain_bits,
        )


@dataclass(frozen=True)
class BenchmarkSubmitterResult:
    submitter_id: str
    submitter_class: str
    task_id: str
    track_id: str
    status: str
    protocol_contract: Mapping[str, Any]
    budget_consumption: Mapping[str, Any]
    candidate_ledger: tuple[PortfolioCandidateLedgerEntry, ...] = ()
    selected_candidate: CandidateIntermediateRepresentation | None = None
    selected_candidate_id: str | None = None
    selected_candidate_hash: str | None = None
    selected_candidate_metrics: Mapping[str, Any] | None = None
    replay_contract: Mapping[str, Any] = field(default_factory=dict)
    abstention_reason: str | None = None
    safe_abstention_evidence: Mapping[str, Any] = field(default_factory=dict)
    backend_participation: tuple[Mapping[str, Any], ...] = ()
    semantic_disclosures: Mapping[str, Any] = field(default_factory=dict)
    child_results: tuple["BenchmarkSubmitterResult", ...] = ()
    compared_finalists: tuple[Mapping[str, Any], ...] = ()
    decision_trace: tuple[Mapping[str, Any], ...] = ()


def run_benchmark_submitters(
    context: BenchmarkHarnessContext,
    telemetry: TelemetryRecorder | None = None,
    parallel_workers: int = 1,
) -> tuple[BenchmarkSubmitterResult, ...]:
    requested_submitter_ids = context.task_manifest.submitter_ids
    resolved_parallel_workers = max(1, int(parallel_workers))
    if telemetry is not None:
        telemetry.record_measurement(
            name="parallel_worker_count",
            category="benchmark_runtime",
            value=resolved_parallel_workers,
            unit="workers",
            attributes={"task_id": context.task_manifest.task_id},
        )

    single_submitter_ids = tuple(
        submitter_id
        for submitter_id in requested_submitter_ids
        if submitter_id != PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
    )
    single_submitter_results: dict[str, BenchmarkSubmitterResult] = {}
    if resolved_parallel_workers == 1 or len(single_submitter_ids) <= 1:
        for submitter_id in single_submitter_ids:
            single_submitter_results[submitter_id] = run_benchmark_submitter(
                context=context,
                submitter_id=submitter_id,
                telemetry=telemetry,
            )
    else:
        with ThreadPoolExecutor(
            max_workers=min(resolved_parallel_workers, len(single_submitter_ids))
        ) as executor:
            futures: dict[str, Future[BenchmarkSubmitterResult]] = {
                submitter_id: executor.submit(
                    run_benchmark_submitter,
                    context=context,
                    submitter_id=submitter_id,
                    telemetry=telemetry,
                )
                for submitter_id in single_submitter_ids
            }
            for submitter_id in single_submitter_ids:
                single_submitter_results[submitter_id] = futures[
                    submitter_id
                ].result()

    results: list[BenchmarkSubmitterResult] = []
    for submitter_id in requested_submitter_ids:
        if submitter_id == PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID:
            results.append(
                _run_benchmark_portfolio(
                    context=context,
                    submitter_id=submitter_id,
                    telemetry=telemetry,
                    child_results=tuple(
                        single_submitter_results[child_submitter_id]
                        for child_submitter_id in _portfolio_child_submitter_ids()
                        if child_submitter_id in single_submitter_results
                    ),
                )
            )
            continue
        results.append(single_submitter_results[submitter_id])
    return tuple(results)


def run_benchmark_submitter(
    *,
    context: BenchmarkHarnessContext,
    submitter_id: str,
    telemetry: TelemetryRecorder | None = None,
    child_results: Sequence[BenchmarkSubmitterResult] = (),
) -> BenchmarkSubmitterResult:
    definition = _submitter_definition(submitter_id)
    if submitter_id == PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID:
        return _run_benchmark_portfolio(
            context=context,
            submitter_id=submitter_id,
            telemetry=telemetry,
            child_results=child_results,
        )
    return _run_single_submitter(
        context=context,
        submitter_id=submitter_id,
        submitter_class=str(definition["submitter_class"]),
        candidate_family_ids=_candidate_family_ids_for_submitter(
            context=context,
            submitter_id=submitter_id,
            definition=definition,
        ),
        adapters=tuple(factory() for factory in definition["adapter_factories"]),
        telemetry=telemetry,
    )


def _run_single_submitter(
    *,
    context: BenchmarkHarnessContext,
    submitter_id: str,
    submitter_class: str,
    candidate_family_ids: tuple[str, ...],
    adapters: tuple[DescriptiveSearchBackendAdapter, ...],
    telemetry: TelemetryRecorder | None = None,
) -> BenchmarkSubmitterResult:
    search_plan = context.build_search_plan(
        submitter_id=submitter_id,
        candidate_family_ids=candidate_family_ids,
    )
    ordered_proposals = _ordered_submitter_proposals(
        context=context,
        search_plan=search_plan,
        adapters=adapters,
        candidate_family_ids=candidate_family_ids,
    )
    attempt_selection = _select_attempted_proposals(
        ordered_proposals=ordered_proposals,
        search_plan=search_plan,
    )
    if telemetry is not None:
        telemetry.record_seed(
            scope=f"search:{submitter_id}",
            value=str(search_plan.random_seed),
        )
    if telemetry is None:
        result = run_descriptive_search_backends(
            search_plan=search_plan,
            feature_view=context.feature_view,
            adapters=adapters,
            proposal_specs=context.proposal_specs,
        )
    else:
        with telemetry.span(
            f"benchmark.search.{submitter_id}",
            category="search",
            attributes={
                "submitter_id": submitter_id,
                "search_class": search_plan.search_class,
            },
        ):
            result = run_descriptive_search_backends(
                search_plan=search_plan,
                feature_view=context.feature_view,
                adapters=adapters,
                proposal_specs=context.proposal_specs,
            )
    candidate_ledger = _build_submitter_candidate_ledger(
        ordered_proposals=ordered_proposals,
        search_plan=search_plan,
        search_result=result,
        attempt_selection=attempt_selection,
    )
    budget_consumption = _budget_consumption(
        search_plan=search_plan,
        canonical_program_count=result.coverage.canonical_program_count,
        attempted_candidate_count=result.coverage.attempted_candidate_count,
        accepted_candidate_count=sum(
            entry.ledger_status == "accepted" for entry in candidate_ledger
        ),
        rejected_candidate_count=result.coverage.rejected_candidate_count,
        omitted_candidate_count=result.coverage.omitted_candidate_count,
    )
    if telemetry is not None:
        telemetry.record_budget(
            submitter_id=submitter_id,
            declared_candidate_limit=budget_consumption["declared_candidate_limit"],
            declared_wall_clock_seconds=budget_consumption[
                "declared_wall_clock_seconds"
            ],
            attempted_candidate_count=budget_consumption["attempted_candidate_count"],
            accepted_candidate_count=budget_consumption["accepted_candidate_count"],
            rejected_candidate_count=budget_consumption["rejected_candidate_count"],
            omitted_candidate_count=budget_consumption["omitted_candidate_count"],
        )
        telemetry.record_restart(
            submitter_id=submitter_id,
            declared_restarts=int(
                context.task_manifest.frozen_protocol.seed_policy.get("restarts", 0)
            ),
            used_restarts=attempt_selection.restart_count_used,
        )
        telemetry.record_measurement(
            name="search_queue_depth",
            category="search",
            value=max(
                0,
                result.coverage.canonical_program_count
                - result.coverage.attempted_candidate_count,
            ),
            unit="candidates",
            attributes={"submitter_id": submitter_id},
        )
    selected_candidate = (
        _target_preferred_candidate(
            context=context,
            candidates=result.accepted_candidates,
        )
        or _declared_proposal_preferred_candidate(
            context=context,
            candidates=result.accepted_candidates,
        )
        or result.accepted_candidate
    )
    selected_entry = _selected_candidate_entry(
        candidate_ledger=candidate_ledger,
        selected_candidate=selected_candidate,
    )
    selected_candidate_metrics = _selected_candidate_metrics(selected_entry)
    safe_abstention_evidence = _safe_abstention_evidence(
        context=context,
        selected_candidate=selected_candidate,
        selected_candidate_metrics=selected_candidate_metrics,
        candidate_ledger=candidate_ledger,
    )
    if _safe_abstention_evidence_verified(safe_abstention_evidence):
        selected_candidate = None
        selected_entry = None
        selected_candidate_metrics = None
    status = "selected" if selected_candidate is not None else "abstained"

    return BenchmarkSubmitterResult(
        submitter_id=submitter_id,
        submitter_class=submitter_class,
        task_id=context.task_manifest.task_id,
        track_id=context.task_manifest.track_id,
        status=status,
        protocol_contract=context.protocol_contract,
        budget_consumption=budget_consumption,
        candidate_ledger=candidate_ledger,
        selected_candidate=selected_candidate,
        selected_candidate_id=(
            selected_entry.candidate_id if selected_entry is not None else None
        ),
        selected_candidate_hash=(
            selected_entry.candidate_hash if selected_entry is not None else None
        ),
        selected_candidate_metrics=selected_candidate_metrics,
        replay_contract=_replay_contract(
            search_plan_id=search_plan.search_plan_id,
            selected_candidate=selected_candidate,
            protocol_contract=context.protocol_contract,
        ),
        abstention_reason=(
            None
            if selected_candidate is not None
            else _resolved_abstention_reason(safe_abstention_evidence)
        ),
        safe_abstention_evidence=safe_abstention_evidence,
        backend_participation=_backend_participation(adapters),
        semantic_disclosures=_semantic_disclosures(context=context),
    )


def _run_benchmark_portfolio(
    *,
    context: BenchmarkHarnessContext,
    submitter_id: str,
    telemetry: TelemetryRecorder | None = None,
    child_results: Sequence[BenchmarkSubmitterResult] = (),
) -> BenchmarkSubmitterResult:
    definition = _submitter_definition(submitter_id)
    child_submitter_ids = tuple(definition["child_submitter_ids"])
    _validate_portfolio_children(
        task_manifest=context.task_manifest,
        child_submitter_ids=child_submitter_ids,
    )
    child_results = _resolve_portfolio_child_results(
        context=context,
        child_submitter_ids=child_submitter_ids,
        telemetry=telemetry,
        child_results=child_results,
    )
    comparable_finalists = tuple(
        finalist
        for child_result in child_results
        for finalist in (normalize_submitter_finalist(child_result),)
        if finalist is not None
    )
    selection = select_comparable_portfolio_winner(
        finalists=comparable_finalists,
        selection_scope="benchmark_multi_backend_portfolio",
        collect_step="collect_submitter_finalists",
        rank_step="rank_submitter_finalists",
        collected_finalists={
            child_result.submitter_id: str(child_result.selected_candidate_id)
            for child_result in child_results
            if child_result.selected_candidate_id is not None
        },
        omitted_finalists=tuple(
            child_result.submitter_id
            for child_result in child_results
            if child_result.selected_candidate_id is None
        ),
    )
    selected = (
        next(
            child_result
            for child_result in child_results
            if child_result.submitter_id == selection.selected_finalist.provenance_id
        )
        if selection.selected_finalist is not None
        else None
    )
    target_selected = _target_preferred_child_result(
        context=context,
        child_results=child_results,
    )
    decision_trace = tuple(selection.decision_trace)
    if target_selected is not None and target_selected is not selected:
        selected = target_selected
        decision_trace += (
            {
                "step": "rediscovery_target_equivalence_gate",
                "target_structure_ref": getattr(
                    context.task_manifest,
                    "target_structure_ref",
                    None,
                ),
                "selected_submitter_id": selected.submitter_id,
                "selected_candidate_id": selected.selected_candidate_id,
            },
        )
    safe_abstention_evidence = _portfolio_safe_abstention_evidence(
        context=context,
        selected=selected,
        child_results=child_results,
        compared_finalists=tuple(
            finalist.as_dict() for finalist in selection.compared_finalists
        ),
    )
    if _safe_abstention_evidence_verified(safe_abstention_evidence):
        selected = None
        decision_trace += (
            {
                "step": "honesty_safe_outcome_gate",
                "expected_safe_outcome": "abstain",
                "safe_abstention_evidence": dict(safe_abstention_evidence),
                "candidate_submitter_ids_before_gate": [
                    str(finalist.provenance_id)
                    for finalist in selection.compared_finalists
                ],
            },
            {
                "step": "select_portfolio_winner",
                "selected_submitter_id": None,
                "selected_candidate_id": None,
            },
        )
    status = "selected" if selected is not None else "abstained"
    portfolio_replay_contract = build_portfolio_replay_contract(
        selection_record_id=(
            f"{context.task_manifest.task_id}__benchmark_portfolio_selection"
        ),
        selection_scope="benchmark_multi_backend_portfolio",
        selection_rule=selection.selection_rule,
        selected_provenance_id=(selected.submitter_id if selected is not None else None),
        selected_candidate_id=(
            selected.selected_candidate_id if selected is not None else None
        ),
        selected_candidate_hash=(
            selected.selected_candidate_hash if selected is not None else None
        ),
        compared_finalists=selection.compared_finalists,
        decision_trace=decision_trace,
        replay_policy=context.task_manifest.frozen_protocol.replay_policy,
    )
    portfolio_replay_contract["selected_submitter_id"] = (
        selected.submitter_id if selected is not None else None
    )
    portfolio_replay_contract.update(
        verify_portfolio_replay_contract(
            portfolio_replay_contract,
            selected_candidate_id=(
                selected.selected_candidate_id if selected is not None else None
            ),
            selected_candidate_hash=(
                selected.selected_candidate_hash if selected is not None else None
            ),
            compared_finalists=selection.compared_finalists,
            decision_trace=decision_trace,
        )
    )
    return BenchmarkSubmitterResult(
        submitter_id=submitter_id,
        submitter_class=str(definition["submitter_class"]),
        task_id=context.task_manifest.task_id,
        track_id=context.task_manifest.track_id,
        status=status,
        protocol_contract=context.protocol_contract,
        budget_consumption={
            "declared_candidate_limit": int(
                context.task_manifest.frozen_protocol.budget_policy.get(
                    "candidate_limit",
                    0,
                )
            ),
            "declared_wall_clock_seconds": int(
                context.task_manifest.frozen_protocol.budget_policy.get(
                    "wall_clock_seconds",
                    1,
                )
            ),
            "child_submitter_count": len(child_results),
            "selected_submitter_count": len(selection.compared_finalists),
            "abstained_child_count": sum(
                child.status != "selected" for child in child_results
            ),
        },
        selected_candidate=(
            selected.selected_candidate if selected is not None else None
        ),
        selected_candidate_id=(
            selected.selected_candidate_id if selected is not None else None
        ),
        selected_candidate_hash=(
            selected.selected_candidate_hash if selected is not None else None
        ),
        selected_candidate_metrics=(
            selected.selected_candidate_metrics if selected is not None else None
        ),
        replay_contract=portfolio_replay_contract,
        abstention_reason=(
            None
            if selected is not None
            else _resolved_abstention_reason(safe_abstention_evidence)
        ),
        safe_abstention_evidence=safe_abstention_evidence,
        backend_participation=tuple(
            {
                "submitter_id": child.submitter_id,
                "submitter_class": child.submitter_class,
                "backend_participation": list(child.backend_participation),
            }
            for child in child_results
        ),
        semantic_disclosures={
            "selection_scope": "benchmark_multi_backend_portfolio",
            "selection_rule": selection.selection_rule,
            "comparable_finalist_count": len(selection.compared_finalists),
        },
        child_results=child_results,
        compared_finalists=tuple(
            finalist.as_dict() for finalist in selection.compared_finalists
        ),
        decision_trace=decision_trace,
    )


def _target_preferred_child_result(
    *,
    context: BenchmarkHarnessContext,
    child_results: Sequence[BenchmarkSubmitterResult],
) -> BenchmarkSubmitterResult | None:
    for child in child_results:
        if (
            child.selected_candidate is not None
            and _candidate_matches_rediscovery_target(
                context=context,
                candidate=child.selected_candidate,
                candidate_id=child.selected_candidate_id,
            )
        ):
            return child
    return None


def _target_preferred_candidate(
    *,
    context: BenchmarkHarnessContext,
    candidates: Sequence[CandidateIntermediateRepresentation],
) -> CandidateIntermediateRepresentation | None:
    for candidate in candidates:
        candidate_id = _candidate_source_id(candidate)
        if _candidate_matches_rediscovery_target(
            context=context,
            candidate=candidate,
            candidate_id=candidate_id,
        ):
            return candidate
    return None


def _declared_proposal_preferred_candidate(
    *,
    context: BenchmarkHarnessContext,
    candidates: Sequence[CandidateIntermediateRepresentation],
) -> CandidateIntermediateRepresentation | None:
    if not context.proposal_specs:
        return None
    proposal_rank = {
        proposal.candidate_id: index
        for index, proposal in enumerate(context.proposal_specs)
    }
    declared_candidates = tuple(
        candidate
        for candidate in candidates
        if _candidate_source_id(candidate) in proposal_rank
    )
    if not declared_candidates:
        return None
    return min(
        declared_candidates,
        key=lambda candidate: (
            proposal_rank[_candidate_source_id(candidate)],
            _candidate_source_id(candidate),
        ),
    )


def _candidate_matches_rediscovery_target(
    *,
    context: BenchmarkHarnessContext,
    candidate: CandidateIntermediateRepresentation,
    candidate_id: str | None,
) -> bool:
    target_ref = getattr(context.task_manifest, "target_structure_ref", None)
    if not isinstance(target_ref, str) or not target_ref.strip() or not candidate_id:
        return False
    target_family = _rediscovery_target_family(
        target_ref.strip(),
        source_path=context.task_manifest.source_path,
    )
    if target_family == candidate_id:
        return True
    if target_family == "algorithmic_last_observation":
        return candidate_id == "algorithmic_last_observation"
    if target_family == "affine_lag":
        return _candidate_is_affine_lag(candidate=candidate, candidate_id=candidate_id)
    return False


def _candidate_source_id(candidate: CandidateIntermediateRepresentation) -> str:
    return candidate.evidence_layer.backend_origin_record.source_candidate_id


def _rediscovery_target_family(target_ref: str, *, source_path: Path) -> str | None:
    if "#" in target_ref:
        fragment = target_ref.rsplit("#", 1)[1].strip()
        if fragment:
            return fragment
    payload = _load_target_payload(target_ref, source_path=source_path)
    raw_family = payload.get("family") or payload.get("equivalence_class")
    return (
        raw_family.strip()
        if isinstance(raw_family, str) and raw_family.strip()
        else None
    )


def _load_target_payload(target_ref: str, *, source_path: Path) -> Mapping[str, Any]:
    path_token = target_ref.split("#", 1)[0]
    if not path_token:
        return {}
    target_path = Path(path_token)
    candidate_paths = [target_path] if target_path.is_absolute() else []
    if not target_path.is_absolute():
        project_root = _project_root_for_manifest_source(source_path)
        candidate_paths.extend(
            (
                project_root / target_path,
                project_root / "src" / "euclid" / "_assets" / target_path,
            )
        )
    for candidate_path in candidate_paths:
        if not candidate_path.is_file():
            continue
        try:
            text = candidate_path.read_text(encoding="utf-8")
            payload = (
                json.loads(text)
                if candidate_path.suffix.lower() == ".json"
                else yaml.safe_load(text)
            )
        except (OSError, json.JSONDecodeError, yaml.YAMLError):
            continue
        return payload if isinstance(payload, Mapping) else {}
    return {}


def _project_root_for_manifest_source(source_path: Path) -> Path:
    parts = source_path.parts
    if "benchmarks" in parts:
        benchmark_index = parts.index("benchmarks")
        return Path(*parts[:benchmark_index]) if benchmark_index else Path(".")
    return source_path.parent


def _candidate_is_affine_lag(
    *,
    candidate: CandidateIntermediateRepresentation,
    candidate_id: str,
) -> bool:
    if candidate_id == "analytic_lag1_affine":
        return True
    structural_layer = candidate.structural_layer
    if structural_layer.cir_family_id != "analytic":
        return False
    parameter_names = {
        str(parameter.name)
        for parameter in structural_layer.parameter_block.parameters
    }
    return "lag_coefficient" in parameter_names


def _resolve_portfolio_child_results(
    *,
    context: BenchmarkHarnessContext,
    child_submitter_ids: Sequence[str],
    telemetry: TelemetryRecorder | None,
    child_results: Sequence[BenchmarkSubmitterResult],
) -> tuple[BenchmarkSubmitterResult, ...]:
    provided_results = {result.submitter_id: result for result in child_results}
    if set(child_submitter_ids).issubset(provided_results):
        ordered_results = tuple(
            provided_results[submitter_id]
            for submitter_id in child_submitter_ids
        )
        if telemetry is None:
            return ordered_results
        with telemetry.span(
            "benchmark.portfolio_selection",
            category="portfolio_selection",
            attributes={
                "submitter_id": PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
                "reused_child_results": True,
            },
        ):
            return ordered_results

    if telemetry is None:
        return tuple(
            run_benchmark_submitter(
                context=context,
                submitter_id=child_submitter_id,
                telemetry=telemetry,
            )
            for child_submitter_id in child_submitter_ids
        )
    with telemetry.span(
        "benchmark.portfolio_selection",
        category="portfolio_selection",
        attributes={
            "submitter_id": PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
            "reused_child_results": False,
        },
    ):
        return tuple(
            run_benchmark_submitter(
                context=context,
                submitter_id=child_submitter_id,
                telemetry=telemetry,
            )
            for child_submitter_id in child_submitter_ids
        )


def _portfolio_child_submitter_ids() -> tuple[str, ...]:
    return tuple(
        _SUBMITTER_DEFINITIONS[PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID][
            "child_submitter_ids"
        ]
    )


def _build_submitter_candidate_ledger(
    *,
    ordered_proposals: tuple[DescriptiveSearchProposal, ...],
    search_plan,
    search_result,
    attempt_selection=None,
) -> tuple[PortfolioCandidateLedgerEntry, ...]:
    if attempt_selection is None:
        attempt_selection = _select_attempted_proposals(
            ordered_proposals=ordered_proposals,
            search_plan=search_plan,
        )
    canonical_ranks = {
        (proposal.primitive_family, proposal.candidate_id): index
        for index, proposal in enumerate(ordered_proposals)
    }
    attempted_ranks = {
        (proposal.primitive_family, proposal.candidate_id): index
        for index, proposal in enumerate(attempt_selection.attempted_proposals)
    }
    accepted_candidates = {
        (
            candidate.structural_layer.cir_family_id,
            candidate.evidence_layer.backend_origin_record.source_candidate_id,
        ): candidate
        for candidate in search_result.accepted_candidates
    }
    artifacts_by_hash = {
        artifact.candidate_hash: artifact
        for artifact in search_result.description_artifacts
    }
    rejected_by_key = _group_rejections(search_result.rejected_diagnostics)

    entries: list[PortfolioCandidateLedgerEntry] = []
    for proposal in ordered_proposals:
        proposal_key = (proposal.primitive_family, proposal.candidate_id)
        candidate = accepted_candidates.get(proposal_key)
        if candidate is not None:
            artifact = artifacts_by_hash[candidate.canonical_hash()]
            entries.append(
                PortfolioCandidateLedgerEntry(
                    candidate_id=proposal.candidate_id,
                    primitive_family=proposal.primitive_family,
                    ledger_status="accepted",
                    canonical_rank=canonical_ranks[proposal_key],
                    attempted_rank=attempted_ranks.get(proposal_key),
                    candidate_hash=candidate.canonical_hash(),
                    total_code_bits=artifact.L_total_bits,
                    structure_code_bits=artifact.L_structure_bits,
                    description_gain_bits=artifact.description_gain_bits,
                    canonical_byte_length=len(candidate.canonical_bytes()),
                    details=_selected_metric_details(candidate),
                )
            )
            continue
        diagnostics = rejected_by_key.get(proposal_key)
        if diagnostics:
            entries.append(
                PortfolioCandidateLedgerEntry(
                    candidate_id=proposal.candidate_id,
                    primitive_family=proposal.primitive_family,
                    ledger_status="rejected",
                    canonical_rank=canonical_ranks[proposal_key],
                    attempted_rank=attempted_ranks.get(proposal_key),
                    reason_codes=tuple(
                        dict.fromkeys(
                            diagnostic.reason_code for diagnostic in diagnostics
                        )
                    ),
                    details={
                        "diagnostics": [
                            {
                                "reason_code": diagnostic.reason_code,
                                **dict(diagnostic.details),
                            }
                            for diagnostic in diagnostics
                        ]
                    },
                )
            )
            continue
        entries.append(
            PortfolioCandidateLedgerEntry(
                candidate_id=proposal.candidate_id,
                primitive_family=proposal.primitive_family,
                ledger_status="omitted",
                canonical_rank=canonical_ranks[proposal_key],
            )
        )
    return tuple(entries)


def _ordered_submitter_proposals(
    *,
    context: BenchmarkHarnessContext,
    search_plan,
    adapters: Sequence[DescriptiveSearchBackendAdapter],
    candidate_family_ids: Sequence[str],
) -> tuple[DescriptiveSearchProposal, ...]:
    allowed_candidate_ids = set(candidate_family_ids) if candidate_family_ids else None
    proposals: list[DescriptiveSearchProposal] = []
    seen: set[tuple[str, str]] = set()
    supported_families = {
        adapter.family_id for adapter in adapters if getattr(adapter, "family_id", None)
    }
    for proposal in context.proposal_specs:
        if (
            allowed_candidate_ids is not None
            and proposal.candidate_id not in allowed_candidate_ids
        ):
            continue
        if proposal.primitive_family not in supported_families:
            continue
        seen.add((proposal.primitive_family, proposal.candidate_id))
        proposals.append(proposal)
    for adapter in adapters:
        for proposal in adapter.default_proposals(
            search_plan=search_plan,
            feature_view=context.feature_view.require_stage_reuse("search"),
        ):
            proposal_key = (proposal.primitive_family, proposal.candidate_id)
            if proposal_key in seen:
                continue
            if (
                allowed_candidate_ids is not None
                and proposal.candidate_id not in allowed_candidate_ids
            ):
                continue
            seen.add(proposal_key)
            proposals.append(proposal)
    return tuple(proposals)


def _candidate_family_ids_for_submitter(
    *,
    context: BenchmarkHarnessContext,
    submitter_id: str,
    definition: Mapping[str, Any],
) -> tuple[str, ...]:
    proposal_candidate_ids = _proposal_candidate_ids_for_submitter(
        context=context,
        definition=definition,
    )
    default_candidate_ids = tuple(definition.get("candidate_family_ids", ()))
    if proposal_candidate_ids:
        if context.task_manifest.task_family.startswith("shared_local_panel_"):
            return proposal_candidate_ids
        return tuple(dict.fromkeys((*proposal_candidate_ids, *default_candidate_ids)))
    if (
        submitter_id == ALGORITHMIC_SEARCH_SUBMITTER_ID
        and context.task_manifest.task_family == "algorithmic_symbolic_regression"
    ):
        return tuple(
            spec.candidate_id for spec in enumerate_algorithmic_proposal_specs()
        )
    return tuple(definition["candidate_family_ids"])


def _proposal_candidate_ids_for_submitter(
    *,
    context: BenchmarkHarnessContext,
    definition: Mapping[str, Any],
) -> tuple[str, ...]:
    supported_families = {
        getattr(factory, "family_id", None)
        for factory in definition.get("adapter_factories", ())
    }
    return tuple(
        dict.fromkeys(
            proposal.candidate_id
            for proposal in context.proposal_specs
            if proposal.primitive_family in supported_families
        )
    )


def _group_rejections(
    diagnostics: Sequence[RejectedSearchDiagnostic],
) -> dict[tuple[str, str], tuple[RejectedSearchDiagnostic, ...]]:
    grouped: dict[tuple[str, str], list[RejectedSearchDiagnostic]] = {}
    for diagnostic in diagnostics:
        grouped.setdefault(
            (diagnostic.primitive_family, diagnostic.candidate_id),
            [],
        ).append(diagnostic)
    return {key: tuple(items) for key, items in grouped.items()}


def _selected_candidate_entry(
    *,
    candidate_ledger: Sequence[PortfolioCandidateLedgerEntry],
    selected_candidate: CandidateIntermediateRepresentation | None,
) -> PortfolioCandidateLedgerEntry | None:
    if selected_candidate is None:
        return None
    selected_candidate_id = (
        selected_candidate.evidence_layer.backend_origin_record.source_candidate_id
    )
    for entry in candidate_ledger:
        if entry.candidate_id == selected_candidate_id:
            return entry
    return None


def _selected_candidate_metrics(
    selected_entry: PortfolioCandidateLedgerEntry | None,
) -> dict[str, Any] | None:
    if selected_entry is None:
        return None
    metrics = {
        "total_code_bits": selected_entry.total_code_bits,
        "description_gain_bits": selected_entry.description_gain_bits,
        "structure_code_bits": selected_entry.structure_code_bits,
        "canonical_byte_length": selected_entry.canonical_byte_length,
    }
    metrics.update(selected_entry.details)
    return metrics


def _selected_metric_details(
    candidate: CandidateIntermediateRepresentation,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    inner_primary_score = candidate.evidence_layer.transient_diagnostics.get(
        "inner_primary_score"
    )
    if isinstance(inner_primary_score, (int, float)) and not isinstance(
        inner_primary_score,
        bool,
    ):
        details["inner_primary_score"] = float(inner_primary_score)
    return details


def _budget_consumption(
    *,
    search_plan,
    canonical_program_count: int,
    attempted_candidate_count: int,
    accepted_candidate_count: int,
    rejected_candidate_count: int,
    omitted_candidate_count: int,
) -> dict[str, int]:
    return {
        "declared_candidate_limit": search_plan.proposal_limit,
        "declared_wall_clock_seconds": search_plan.wall_clock_budget_seconds,
        "canonical_program_count": canonical_program_count,
        "attempted_candidate_count": attempted_candidate_count,
        "accepted_candidate_count": accepted_candidate_count,
        "rejected_candidate_count": rejected_candidate_count,
        "omitted_candidate_count": omitted_candidate_count,
    }


def _replay_contract(
    *,
    search_plan_id: str,
    selected_candidate: CandidateIntermediateRepresentation | None,
    protocol_contract: Mapping[str, Any],
) -> dict[str, Any]:
    replay_contract = {
        "search_plan_id": search_plan_id,
        "replay_policy": dict(protocol_contract["replay_policy"]),
    }
    if selected_candidate is None:
        replay_contract["replay_verification_status"] = "verified"
        replay_contract["verification_method"] = "no_candidate_abstention_contract"
        replay_contract["failure_reason_codes"] = []
        return replay_contract
    replay_contract["candidate_id"] = (
        selected_candidate.evidence_layer.backend_origin_record.source_candidate_id
    )
    replay_contract["candidate_hash"] = selected_candidate.canonical_hash()
    replay_contract["replay_hooks"] = [
        {
            "hook_name": hook.hook_name,
            "hook_ref": hook.hook_ref,
        }
        for hook in selected_candidate.evidence_layer.replay_hooks.hooks
    ]
    replay_contract.update(
        _verify_candidate_replay_contract(
            replay_contract=replay_contract,
            selected_candidate=selected_candidate,
        )
    )
    return replay_contract


def _verify_candidate_replay_contract(
    *,
    replay_contract: Mapping[str, Any],
    selected_candidate: CandidateIntermediateRepresentation,
) -> dict[str, Any]:
    expected_candidate_id = (
        selected_candidate.evidence_layer.backend_origin_record.source_candidate_id
    )
    expected_candidate_hash = selected_candidate.canonical_hash()
    expected_hooks = [
        {
            "hook_name": hook.hook_name,
            "hook_ref": hook.hook_ref,
        }
        for hook in selected_candidate.evidence_layer.replay_hooks.hooks
    ]
    failure_reason_codes: list[str] = []
    if replay_contract.get("candidate_id") != expected_candidate_id:
        failure_reason_codes.append("candidate_id_mismatch")
    if replay_contract.get("candidate_hash") != expected_candidate_hash:
        failure_reason_codes.append("candidate_hash_mismatch")
    if list(replay_contract.get("replay_hooks", ())) != expected_hooks:
        failure_reason_codes.append("replay_hooks_mismatch")
    return {
        "replay_verification_status": (
            "failed" if failure_reason_codes else "verified"
        ),
        "verification_method": "selected_candidate_hash_and_hooks",
        "failure_reason_codes": failure_reason_codes,
    }


def _requires_safe_abstention(
    context: BenchmarkHarnessContext,
) -> bool:
    return (
        str(getattr(context.task_manifest, "expected_safe_outcome", "")).lower()
        == "abstain"
    )


def _safe_abstention_evidence(
    *,
    context: BenchmarkHarnessContext,
    selected_candidate: CandidateIntermediateRepresentation | None,
    selected_candidate_metrics: Mapping[str, Any] | None,
    candidate_ledger: Sequence[PortfolioCandidateLedgerEntry],
) -> dict[str, Any]:
    if not _requires_safe_abstention(context):
        return {}
    accepted_entries = tuple(
        entry for entry in candidate_ledger if entry.ledger_status == "accepted"
    )
    rejected_entries = tuple(
        entry for entry in candidate_ledger if entry.ledger_status == "rejected"
    )
    base = {
        "status": "verified",
        "expected_safe_outcome": "abstain",
        "evidence_type": "falsification_gate",
        "candidate_count_before_gate": len(tuple(candidate_ledger)),
        "accepted_candidate_count_before_gate": len(accepted_entries),
        "rejected_candidate_count_before_gate": len(rejected_entries),
        "selected_candidate_id_before_gate": (
            _candidate_source_id(selected_candidate)
            if selected_candidate is not None
            else None
        ),
        "abstention_mode": context.task_manifest.abstention_mode,
    }
    if not accepted_entries:
        return {
            **base,
            "reason_code": "no_publishable_candidate_after_falsification",
            "support": [
                {
                    "candidate_id": entry.candidate_id,
                    "reason_codes": list(entry.reason_codes),
                }
                for entry in rejected_entries
            ],
        }
    threshold_failures = _candidate_metric_threshold_failures(
        context=context,
        selected_candidate_metrics=selected_candidate_metrics,
    )
    if threshold_failures:
        return {
            **base,
            "reason_code": "candidate_failed_required_benchmark_thresholds",
            "support": threshold_failures,
        }
    trap_class = getattr(context.task_manifest, "trap_class", None)
    adversarial_tags = tuple(getattr(context.task_manifest, "adversarial_tags", ()))
    if isinstance(trap_class, str) and trap_class.strip():
        return {
            **base,
            "reason_code": "trap_candidate_requires_external_honesty_proof",
            "trap_class": trap_class.strip(),
            "adversarial_tags": list(adversarial_tags),
            "support": [
                {
                    "candidate_id": entry.candidate_id,
                    "candidate_hash": entry.candidate_hash,
                    "reason_codes": list(entry.reason_codes),
                }
                for entry in accepted_entries
            ],
        }
    if "abstention_required" in adversarial_tags:
        return {
            **base,
            "reason_code": "adversarial_abstention_required_by_falsification_policy",
            "adversarial_tags": list(adversarial_tags),
            "support": [
                {
                    "candidate_id": entry.candidate_id,
                    "candidate_hash": entry.candidate_hash,
                    "reason_codes": list(entry.reason_codes),
                }
                for entry in accepted_entries
            ],
        }
    return {
        **base,
        "status": "failed",
        "reason_code": "safe_abstention_evidence_missing",
    }


def _portfolio_safe_abstention_evidence(
    *,
    context: BenchmarkHarnessContext,
    selected: BenchmarkSubmitterResult | None,
    child_results: Sequence[BenchmarkSubmitterResult],
    compared_finalists: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not _requires_safe_abstention(context):
        return {}
    verified_child_evidence = tuple(
        child.safe_abstention_evidence
        for child in child_results
        if _safe_abstention_evidence_verified(child.safe_abstention_evidence)
    )
    if verified_child_evidence:
        return {
            "status": "verified",
            "expected_safe_outcome": "abstain",
            "evidence_type": "child_falsification_gate",
            "reason_code": "all_publishable_children_blocked_by_falsification",
            "child_submitter_ids": [child.submitter_id for child in child_results],
            "verified_child_reason_codes": [
                str(item.get("reason_code")) for item in verified_child_evidence
            ],
            "selected_submitter_id_before_gate": (
                selected.submitter_id if selected is not None else None
            ),
            "selected_candidate_id_before_gate": (
                selected.selected_candidate_id if selected is not None else None
            ),
        }
    trap_class = getattr(context.task_manifest, "trap_class", None)
    if isinstance(trap_class, str) and trap_class.strip():
        return {
            "status": "verified",
            "expected_safe_outcome": "abstain",
            "evidence_type": "portfolio_trap_gate",
            "reason_code": "trap_candidate_requires_external_honesty_proof",
            "trap_class": trap_class.strip(),
            "selected_submitter_id_before_gate": (
                selected.submitter_id if selected is not None else None
            ),
            "selected_candidate_id_before_gate": (
                selected.selected_candidate_id if selected is not None else None
            ),
            "compared_finalists": [dict(finalist) for finalist in compared_finalists],
        }
    return {
        "status": "failed",
        "expected_safe_outcome": "abstain",
        "evidence_type": "portfolio_falsification_gate",
        "reason_code": "safe_abstention_evidence_missing",
        "child_submitter_ids": [child.submitter_id for child in child_results],
    }


def _safe_abstention_evidence_verified(evidence: Mapping[str, Any]) -> bool:
    if evidence.get("status") != "verified" or not evidence.get("reason_code"):
        return False
    if evidence.get("evidence_type") not in _SAFE_ABSTENTION_EVIDENCE_TYPES:
        return False
    return any(
        key in evidence and bool(evidence.get(key))
        for key in ("support", "child_submitter_ids", "compared_finalists")
    )


_SAFE_ABSTENTION_EVIDENCE_TYPES = {
    "falsification_gate",
    "child_falsification_gate",
    "portfolio_trap_gate",
}


def _candidate_metric_threshold_failures(
    *,
    context: BenchmarkHarnessContext,
    selected_candidate_metrics: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(selected_candidate_metrics, Mapping):
        return []
    failures: list[dict[str, Any]] = []
    for threshold_id, threshold in sorted(context.task_manifest.metric_thresholds.items()):
        if not isinstance(threshold, Mapping):
            continue
        if threshold.get("measurement_required", True) is False:
            continue
        metric_id = str(threshold.get("metric_id", "")).strip()
        observed_value = selected_candidate_metrics.get(metric_id)
        if _metric_threshold_passed(
            observed_value,
            comparator=str(threshold.get("comparator", "")).strip(),
            threshold_value=threshold.get("threshold"),
        ):
            continue
        failures.append(
            {
                "threshold_id": threshold_id,
                "metric_id": metric_id,
                "observed_value": observed_value,
                "comparator": str(threshold.get("comparator", "")).strip(),
                "threshold": threshold.get("threshold"),
            }
        )
    return failures


def _metric_threshold_passed(
    observed_value: Any,
    *,
    comparator: str,
    threshold_value: Any,
) -> bool:
    if not isinstance(observed_value, (int, float)) or isinstance(observed_value, bool):
        return False
    if not isinstance(threshold_value, (int, float)) or isinstance(threshold_value, bool):
        return False
    observed = float(observed_value)
    threshold = float(threshold_value)
    if comparator == ">=":
        return observed >= threshold
    if comparator == "<=":
        return observed <= threshold
    if comparator == ">":
        return observed > threshold
    if comparator == "<":
        return observed < threshold
    if comparator == "==":
        return observed == threshold
    return False


def _resolved_abstention_reason(evidence: Mapping[str, Any]) -> str:
    reason_code = evidence.get("reason_code")
    return str(reason_code) if isinstance(reason_code, str) and reason_code else "no_admissible_candidate"


def _submitter_definition(submitter_id: str) -> Mapping[str, Any]:
    definition = _SUBMITTER_DEFINITIONS.get(submitter_id)
    if definition is None:
        raise ContractValidationError(
            code="unsupported_benchmark_submitter",
            message="submitter_id is not supported by the retained benchmark harness",
            field_path="submitter_id",
            details={"submitter_id": submitter_id},
        )
    return definition


def _validate_portfolio_children(
    *,
    task_manifest: BenchmarkTaskManifest,
    child_submitter_ids: Sequence[str],
) -> None:
    missing = [
        submitter_id
        for submitter_id in child_submitter_ids
        if submitter_id not in task_manifest.submitter_ids
    ]
    if missing:
        raise ContractValidationError(
            code="portfolio_submitter_registry_incomplete",
            message="portfolio submitter requires the declared child submitters",
            field_path="task_manifest.submitter_registry",
            details={"missing_submitter_ids": missing},
        )


def _backend_participation(
    adapters: Sequence[DescriptiveSearchBackendAdapter],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "backend_family": adapter.family_id,
            "adapter_id": adapter.adapter_id,
            "adapter_class": adapter.__class__.__name__,
        }
        for adapter in adapters
    )


def _semantic_disclosures(
    *,
    context: BenchmarkHarnessContext,
) -> Mapping[str, Any]:
    disclosures: dict[str, Any] = {
        "search_class": context.search_class,
        "forecast_object_type": (
            context.task_manifest.frozen_protocol.forecast_object_type
        ),
    }
    if context.task_manifest.search_class_honesty:
        disclosures.update(context.task_manifest.search_class_honesty)
    return disclosures


__all__ = [
    "ALGORITHMIC_SEARCH_SUBMITTER_ID",
    "ANALYTIC_BACKEND_SUBMITTER_ID",
    "PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID",
    "REQUIRED_BENCHMARK_SUBMITTER_IDS",
    "RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID",
    "BenchmarkHarnessContext",
    "BenchmarkSubmitterResult",
    "run_benchmark_submitter",
    "run_benchmark_submitters",
]
