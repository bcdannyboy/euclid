from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from euclid.adapters.algorithmic_dsl import enumerate_algorithmic_proposal_specs
from euclid.adapters.portfolio import normalize_submitter_finalist
from euclid.benchmarks.manifests import BenchmarkTaskManifest
from euclid.cir.models import CandidateIntermediateRepresentation
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.modules.features import FeatureView
from euclid.modules.replay import build_portfolio_replay_contract
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
        accepted_candidate_count=result.coverage.accepted_candidate_count,
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
        result.frontier.frozen_shortlist_cir_candidates[0]
        if result.frontier.frozen_shortlist_cir_candidates
        else None
    )
    selected_entry = _selected_candidate_entry(
        candidate_ledger=candidate_ledger,
        selected_candidate=selected_candidate,
    )
    requires_safe_abstention = _requires_safe_abstention(context)
    if requires_safe_abstention:
        selected_candidate = None
        selected_entry = None
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
        selected_candidate_metrics=_selected_candidate_metrics(selected_entry),
        replay_contract=_replay_contract(
            search_plan_id=search_plan.search_plan_id,
            selected_candidate=selected_candidate,
            protocol_contract=context.protocol_contract,
        ),
        abstention_reason=(
            None
            if selected_candidate is not None
            else (
                "safe_outcome_forced_abstention"
                if requires_safe_abstention
                else "no_admissible_candidate"
            )
        ),
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
    requires_safe_abstention = _requires_safe_abstention(context)
    decision_trace = tuple(selection.decision_trace)
    if requires_safe_abstention:
        selected = None
    if requires_safe_abstention:
        decision_trace += (
            {
                "step": "honesty_safe_outcome_gate",
                "expected_safe_outcome": "abstain",
                "forced_abstention": True,
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
        replay_contract={
            **build_portfolio_replay_contract(
                selection_record_id=(
                    f"{context.task_manifest.task_id}__benchmark_portfolio_selection"
                ),
                selection_scope="benchmark_multi_backend_portfolio",
                selection_rule=selection.selection_rule,
                selected_provenance_id=(
                    selected.submitter_id if selected is not None else None
                ),
                selected_candidate_id=(
                    selected.selected_candidate_id if selected is not None else None
                ),
                selected_candidate_hash=(
                    selected.selected_candidate_hash if selected is not None else None
                ),
                compared_finalists=selection.compared_finalists,
                decision_trace=decision_trace,
                replay_policy=context.task_manifest.frozen_protocol.replay_policy,
            ),
            "selected_submitter_id": (
                selected.submitter_id if selected is not None else None
            ),
        },
        abstention_reason=None if selected is not None else "no_admissible_candidate",
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
    for adapter in adapters:
        proposals.extend(
            proposal
            for proposal in adapter.default_proposals(
                search_plan=search_plan,
                feature_view=context.feature_view.require_stage_reuse("search"),
            )
            if (
                allowed_candidate_ids is None
                or proposal.candidate_id in allowed_candidate_ids
            )
        )
    for proposal in context.proposal_specs:
        if (
            allowed_candidate_ids is not None
            and proposal.candidate_id not in allowed_candidate_ids
        ):
            continue
        if proposal.primitive_family not in {adapter.family_id for adapter in adapters}:
            continue
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
    return {
        "total_code_bits": selected_entry.total_code_bits,
        "description_gain_bits": selected_entry.description_gain_bits,
        "structure_code_bits": selected_entry.structure_code_bits,
        "canonical_byte_length": selected_entry.canonical_byte_length,
    }


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
    return replay_contract


def _requires_safe_abstention(
    context: BenchmarkHarnessContext,
) -> bool:
    return (
        context.task_manifest.track_id == "adversarial_honesty"
        and str(getattr(context.task_manifest, "expected_safe_outcome", "")).lower()
        == "abstain"
    )


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
