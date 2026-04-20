from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from euclid.adapters.portfolio import ComparableBackendFinalist, normalize_cir_finalist
from euclid.cir.models import CandidateIntermediateRepresentation
from euclid.manifests.runtime_models import SearchPlanManifest
from euclid.modules.features import FeatureView
from euclid.search.backends import (
    DescriptiveSearchBackendAdapter,
    DescriptiveSearchProposal,
    DescriptiveSearchRunResult,
    RejectedSearchDiagnostic,
    SearchCoverageAccounting,
    _default_adapters,
    _select_attempted_proposals,
    run_descriptive_search_backends,
)

_SELECTION_SCOPE = "shared_planning_cir_only"
_SELECTION_RULE = (
    "min_total_code_bits_then_max_description_gain_then_"
    "min_structure_code_bits_then_min_canonical_byte_length_then_candidate_id"
)


@dataclass(frozen=True)
class PortfolioCandidateLedgerEntry:
    candidate_id: str
    primitive_family: str
    ledger_status: str
    canonical_rank: int
    attempted_rank: int | None = None
    candidate_hash: str | None = None
    total_code_bits: float | None = None
    structure_code_bits: float | None = None
    description_gain_bits: float | None = None
    canonical_byte_length: int | None = None
    reason_codes: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "primitive_family": self.primitive_family,
            "ledger_status": self.ledger_status,
            "canonical_rank": self.canonical_rank,
            "reason_codes": list(self.reason_codes),
            "details": dict(self.details),
        }
        if self.attempted_rank is not None:
            payload["attempted_rank"] = self.attempted_rank
        if self.candidate_hash is not None:
            payload["candidate_hash"] = self.candidate_hash
        if self.total_code_bits is not None:
            payload["total_code_bits"] = self.total_code_bits
        if self.structure_code_bits is not None:
            payload["structure_code_bits"] = self.structure_code_bits
        if self.description_gain_bits is not None:
            payload["description_gain_bits"] = self.description_gain_bits
        if self.canonical_byte_length is not None:
            payload["canonical_byte_length"] = self.canonical_byte_length
        return payload


@dataclass(frozen=True)
class PortfolioBackendLedger:
    family_id: str
    adapter_id: str
    search_class: str
    coverage_statement: str
    exactness_ceiling: str
    scope_declaration: str
    budget_consumption: Mapping[str, int]
    candidate_ledger: tuple[PortfolioCandidateLedgerEntry, ...]
    finalist_candidate_id: str | None = None
    finalist_candidate_hash: str | None = None
    finalist_selection_rule: str = _SELECTION_RULE

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "family_id": self.family_id,
            "adapter_id": self.adapter_id,
            "search_class": self.search_class,
            "coverage_statement": self.coverage_statement,
            "exactness_ceiling": self.exactness_ceiling,
            "scope_declaration": self.scope_declaration,
            "budget_consumption": dict(self.budget_consumption),
            "candidate_ledger": [entry.as_dict() for entry in self.candidate_ledger],
            "finalist_selection_rule": self.finalist_selection_rule,
        }
        if self.finalist_candidate_id is not None:
            payload["finalist_candidate_id"] = self.finalist_candidate_id
        if self.finalist_candidate_hash is not None:
            payload["finalist_candidate_hash"] = self.finalist_candidate_hash
        return payload


@dataclass(frozen=True)
class PortfolioSelectionRecord:
    selection_record_id: str
    selection_scope: str
    selection_rule: str
    forecast_object_type: str
    selected_candidate_id: str | None
    selected_candidate_hash: str | None
    selected_backend_family: str | None
    compared_finalists: tuple[Mapping[str, Any], ...]
    decision_trace: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "selection_record_id": self.selection_record_id,
            "selection_scope": self.selection_scope,
            "selection_rule": self.selection_rule,
            "forecast_object_type": self.forecast_object_type,
            "compared_finalists": [dict(item) for item in self.compared_finalists],
            "decision_trace": [dict(item) for item in self.decision_trace],
        }
        if self.selected_candidate_id is not None:
            payload["selected_candidate_id"] = self.selected_candidate_id
        if self.selected_candidate_hash is not None:
            payload["selected_candidate_hash"] = self.selected_candidate_hash
        if self.selected_backend_family is not None:
            payload["selected_backend_family"] = self.selected_backend_family
        return payload


@dataclass(frozen=True)
class DescriptiveSearchPortfolioResult:
    search_result: DescriptiveSearchRunResult
    backend_ledgers: tuple[PortfolioBackendLedger, ...]
    selected_candidate: CandidateIntermediateRepresentation | None
    selection_record: PortfolioSelectionRecord


@dataclass(frozen=True)
class ComparablePortfolioSelection:
    selection_scope: str
    selection_rule: str
    selected_finalist: ComparableBackendFinalist | None
    compared_finalists: tuple[ComparableBackendFinalist, ...]
    decision_trace: tuple[Mapping[str, Any], ...]


def run_descriptive_search_portfolio(
    *,
    search_plan: SearchPlanManifest,
    feature_view: FeatureView,
    proposal_specs: Sequence[DescriptiveSearchProposal] = (),
    include_default_grammar: bool = True,
    adapters: Sequence[DescriptiveSearchBackendAdapter] | None = None,
    observation_model=None,
) -> DescriptiveSearchPortfolioResult:
    adapter_list = tuple(adapters or _default_adapters())
    grouped_proposals = _group_proposals(
        adapters=adapter_list,
        search_plan=search_plan,
        feature_view=feature_view,
        proposal_specs=proposal_specs,
        include_default_grammar=include_default_grammar,
    )
    ordered_proposals = tuple(
        proposal
        for adapter in adapter_list
        for proposal in grouped_proposals[adapter.family_id]
    )
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

    search_result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        proposal_specs=proposal_specs,
        include_default_grammar=include_default_grammar,
        adapters=adapter_list,
        observation_model=observation_model,
    )
    accepted_candidates = {
        (
            candidate.evidence_layer.backend_origin_record.backend_family
            or candidate.structural_layer.cir_family_id,
            candidate.evidence_layer.backend_origin_record.source_candidate_id,
        ): candidate
        for candidate in search_result.accepted_candidates
    }
    description_artifacts = {
        artifact.candidate_hash: artifact
        for artifact in search_result.description_artifacts
    }
    rejected_diagnostics = _group_rejections(search_result.rejected_diagnostics)

    ledgers: list[PortfolioBackendLedger] = []
    finalists: list[
        tuple[PortfolioCandidateLedgerEntry, CandidateIntermediateRepresentation]
    ] = []

    for family_result in search_result.family_results:
        candidate_entries: list[PortfolioCandidateLedgerEntry] = []
        for proposal in grouped_proposals[family_result.family_id]:
            proposal_key = (proposal.primitive_family, proposal.candidate_id)
            candidate = accepted_candidates.get(proposal_key)
            if candidate is not None:
                artifact = description_artifacts[candidate.canonical_hash()]
                candidate_entries.append(
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

            diagnostics = rejected_diagnostics.get(proposal_key)
            if diagnostics:
                candidate_entries.append(
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

            candidate_entries.append(
                PortfolioCandidateLedgerEntry(
                    candidate_id=proposal.candidate_id,
                    primitive_family=proposal.primitive_family,
                    ledger_status="omitted",
                    canonical_rank=canonical_ranks[proposal_key],
                )
            )

        finalist_entry = _select_family_finalist(candidate_entries)
        finalist_candidate = None
        if finalist_entry is not None:
            finalist_candidate = accepted_candidates[
                (family_result.family_id, finalist_entry.candidate_id)
            ]
            finalists.append((finalist_entry, finalist_candidate))
        ledgers.append(
            PortfolioBackendLedger(
                family_id=family_result.family_id,
                adapter_id=family_result.adapter_id,
                search_class=family_result.coverage.search_class,
                coverage_statement=family_result.coverage.coverage_statement,
                exactness_ceiling=family_result.coverage.exactness_ceiling,
                scope_declaration=family_result.coverage.scope_declaration,
                budget_consumption=_budget_consumption(family_result.coverage),
                candidate_ledger=tuple(candidate_entries),
                finalist_candidate_id=(
                    finalist_entry.candidate_id if finalist_entry is not None else None
                ),
                finalist_candidate_hash=(
                    finalist_entry.candidate_hash
                    if finalist_entry is not None
                    else None
                ),
            )
        )

    finalist_map = {
        finalist_candidate.canonical_hash(): finalist_candidate
        for _, finalist_candidate in finalists
    }
    comparable_finalists = tuple(
        normalize_cir_finalist(
            finalist_candidate,
            total_code_bits=float(finalist_entry.total_code_bits),
            description_gain_bits=float(finalist_entry.description_gain_bits),
            structure_code_bits=float(finalist_entry.structure_code_bits),
            coverage_statement=family_result.coverage.coverage_statement,
            exactness_ceiling=family_result.coverage.exactness_ceiling,
            scope_declaration=family_result.coverage.scope_declaration,
            provenance_id=family_result.family_id,
        )
        for family_result in search_result.family_results
        for finalist_entry, finalist_candidate in finalists
        if (
            family_result.family_id
            == finalist_candidate.evidence_layer.backend_origin_record.backend_family
        )
    )
    selection = select_comparable_portfolio_winner(
        finalists=comparable_finalists,
        selection_scope=_SELECTION_SCOPE,
        collect_step="collect_family_finalists",
        rank_step="rank_family_finalists",
        collected_finalists={
            ledger.family_id: ledger.finalist_candidate_id
            for ledger in ledgers
            if ledger.finalist_candidate_id is not None
        },
        omitted_finalists=tuple(
            ledger.family_id
            for ledger in ledgers
            if ledger.finalist_candidate_id is None
        ),
    )
    selected_candidate = (
        finalist_map.get(selection.selected_finalist.candidate_hash)
        if selection.selected_finalist is not None
        else None
    )
    selected_entry = (
        next(
            entry
            for entry, candidate in finalists
            if candidate.canonical_hash() == selection.selected_finalist.candidate_hash
        )
        if selection.selected_finalist is not None
        else None
    )
    forecast_object_type = (
        selected_candidate.execution_layer.forecast_operator.forecast_object_type
        if selected_candidate is not None
        else "point"
    )
    selection_record = PortfolioSelectionRecord(
        selection_record_id=f"{search_plan.search_plan_id}__portfolio_selection",
        selection_scope=_SELECTION_SCOPE,
        selection_rule=_SELECTION_RULE,
        forecast_object_type=forecast_object_type,
        selected_candidate_id=(
            selected_entry.candidate_id if selected_entry is not None else None
        ),
        selected_candidate_hash=(
            selected_entry.candidate_hash if selected_entry is not None else None
        ),
        selected_backend_family=(
            selected_candidate.evidence_layer.backend_origin_record.backend_family
            if selected_candidate is not None
            else None
        ),
        compared_finalists=tuple(
            finalist.as_dict() for finalist in selection.compared_finalists
        ),
        decision_trace=selection.decision_trace,
    )
    return DescriptiveSearchPortfolioResult(
        search_result=search_result,
        backend_ledgers=tuple(ledgers),
        selected_candidate=selected_candidate,
        selection_record=selection_record,
    )


def _group_proposals(
    *,
    adapters: tuple[DescriptiveSearchBackendAdapter, ...],
    search_plan: SearchPlanManifest,
    feature_view: FeatureView,
    proposal_specs: Sequence[DescriptiveSearchProposal],
    include_default_grammar: bool,
) -> dict[str, list[DescriptiveSearchProposal]]:
    allowed_candidate_ids = (
        set(search_plan.candidate_family_ids)
        if search_plan.candidate_family_ids
        else None
    )
    grouped_proposals: dict[str, list[DescriptiveSearchProposal]] = {
        adapter.family_id: [] for adapter in adapters
    }
    if include_default_grammar:
        for adapter in adapters:
            for proposal in adapter.default_proposals(
                search_plan=search_plan,
                feature_view=feature_view.require_stage_reuse("search"),
            ):
                if (
                    allowed_candidate_ids is not None
                    and proposal.candidate_id not in allowed_candidate_ids
                ):
                    continue
                grouped_proposals[adapter.family_id].append(proposal)
    adapter_by_family = {adapter.family_id: adapter for adapter in adapters}
    for proposal in proposal_specs:
        if (
            allowed_candidate_ids is not None
            and proposal.candidate_id not in allowed_candidate_ids
        ):
            continue
        if adapter_by_family.get(proposal.primitive_family) is None:
            continue
        grouped_proposals[proposal.primitive_family].append(proposal)
    return grouped_proposals


def _group_rejections(
    diagnostics: Sequence[RejectedSearchDiagnostic],
) -> dict[tuple[str, str], list[RejectedSearchDiagnostic]]:
    grouped: dict[tuple[str, str], list[RejectedSearchDiagnostic]] = {}
    for diagnostic in diagnostics:
        grouped.setdefault(
            (diagnostic.primitive_family, diagnostic.candidate_id),
            [],
        ).append(diagnostic)
    return grouped


def _select_family_finalist(
    candidate_entries: Sequence[PortfolioCandidateLedgerEntry],
) -> PortfolioCandidateLedgerEntry | None:
    accepted = [
        entry for entry in candidate_entries if entry.ledger_status == "accepted"
    ]
    if not accepted:
        return None
    return sorted(accepted, key=_portfolio_sort_key)[0]


def _portfolio_sort_key(
    entry: PortfolioCandidateLedgerEntry,
) -> tuple[float, float, float, int, str]:
    return (
        float(entry.total_code_bits),
        -float(entry.description_gain_bits),
        float(entry.structure_code_bits),
        int(entry.canonical_byte_length),
        entry.candidate_id,
    )


def _budget_consumption(coverage: SearchCoverageAccounting) -> dict[str, int]:
    return {
        "canonical_program_count": coverage.canonical_program_count,
        "attempted_candidate_count": coverage.attempted_candidate_count,
        "accepted_candidate_count": coverage.accepted_candidate_count,
        "rejected_candidate_count": coverage.rejected_candidate_count,
        "omitted_candidate_count": coverage.omitted_candidate_count,
    }


def select_comparable_portfolio_winner(
    *,
    finalists: Sequence[ComparableBackendFinalist],
    selection_scope: str,
    collect_step: str,
    rank_step: str,
    collected_finalists: Mapping[str, str],
    omitted_finalists: Sequence[str] = (),
) -> ComparablePortfolioSelection:
    ranked_finalists = tuple(sorted(finalists, key=_comparable_portfolio_sort_key))
    selected = ranked_finalists[0] if ranked_finalists else None
    decision_trace: tuple[dict[str, Any], ...] = (
        {
            "step": collect_step,
            "family_finalists": dict(collected_finalists),
            "non_admitted_families": list(omitted_finalists),
        },
        {
            "step": rank_step,
            "ordered_candidate_ids": [
                finalist.candidate_id for finalist in ranked_finalists
            ],
        },
        {
            "step": "select_portfolio_winner",
            "selected_candidate_id": (
                selected.candidate_id if selected is not None else None
            ),
            "selected_candidate_hash": (
                selected.candidate_hash if selected is not None else None
            ),
            "selected_backend_family": (
                selected.backend_family if selected is not None else None
            ),
        },
    )
    return ComparablePortfolioSelection(
        selection_scope=selection_scope,
        selection_rule=_SELECTION_RULE,
        selected_finalist=selected,
        compared_finalists=ranked_finalists,
        decision_trace=decision_trace,
    )


def _comparable_portfolio_sort_key(
    finalist: ComparableBackendFinalist,
) -> tuple[float, float, float, int, str]:
    return (
        float(finalist.total_code_bits),
        -float(finalist.description_gain_bits),
        float(finalist.structure_code_bits),
        int(finalist.canonical_byte_length),
        finalist.candidate_id,
    )


__all__ = [
    "DescriptiveSearchPortfolioResult",
    "PortfolioBackendLedger",
    "PortfolioCandidateLedgerEntry",
    "ComparablePortfolioSelection",
    "PortfolioSelectionRecord",
    "select_comparable_portfolio_winner",
    "run_descriptive_search_portfolio",
]
