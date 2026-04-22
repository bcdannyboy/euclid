from __future__ import annotations

from typing import Any, Sequence

from euclid.manifests.runtime_models import SearchPlanManifest
from euclid.modules.features import FeatureView
from euclid.search.engine_contracts import (
    EngineCandidateRecord,
    EngineInputContext,
    EngineRunResult,
    SearchEngine,
    engine_claim_boundary,
)
from euclid.search.orchestration import SearchEngineOrchestrator


class NativeFragmentEngine(SearchEngine):
    engine_version = "1.0"

    def __init__(self, *, search_class: str) -> None:
        self.search_class = search_class
        suffix = {
            "exact_finite_enumeration": "exact",
            "bounded_heuristic": "bounded",
            "stochastic_heuristic": "stochastic",
            "equality_saturation_heuristic": "egraph",
        }.get(search_class, search_class.replace("_", "-"))
        self.engine_id = f"native-{suffix}-fragment-v1"

    @staticmethod
    def context_from_plan(
        *,
        search_plan: SearchPlanManifest,
        feature_view: FeatureView,
        timeout_seconds: float,
        engine_ids: Sequence[str] | None = None,
    ) -> EngineInputContext:
        engine_id = NativeFragmentEngine(
            search_class=search_plan.search_class
        ).engine_id
        return SearchEngineOrchestrator.context_from_rows(
            search_plan_id=search_plan.search_plan_id,
            search_class=search_plan.search_class,
            random_seed=search_plan.random_seed,
            proposal_limit=search_plan.proposal_limit,
            frontier_axes=search_plan.frontier_axes,
            rows=feature_view.rows,
            feature_names=feature_view.feature_names,
            timeout_seconds=timeout_seconds,
            engine_ids=tuple(engine_ids or (engine_id,)),
            allowed_candidate_ids=search_plan.candidate_family_ids,
            runtime_search_plan=search_plan,
            runtime_feature_view=feature_view,
        )

    def run(self, context: EngineInputContext) -> EngineRunResult:
        from euclid.search import backends

        search_plan = context.runtime_search_plan
        feature_view = context.runtime_feature_view
        if search_plan is None or feature_view is None:
            raise ValueError(
                "native engine requires runtime search plan and feature view"
            )

        adapters = backends._default_adapters()
        allowed = set(context.allowed_candidate_ids)
        proposals = []
        omitted_by_filter = 0
        for adapter in adapters:
            for proposal in adapter.default_proposals(
                search_plan=search_plan,
                feature_view=feature_view,
            ):
                if allowed and proposal.candidate_id not in allowed:
                    omitted_by_filter += 1
                    continue
                proposals.append(proposal)
        selection = backends._select_attempted_proposals(
            ordered_proposals=tuple(proposals),
            search_plan=search_plan,
        )
        attempted = selection.attempted_proposals
        candidate_records = tuple(
            self._record_from_proposal(
                proposal=proposal,
                proposal_rank=rank,
                context=context,
                canonical_program_count=len(proposals),
                restart_count_used=selection.restart_count_used,
            )
            for rank, proposal in enumerate(attempted)
        )
        omitted_by_budget = max(len(proposals) - len(attempted), 0)
        return EngineRunResult(
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            status="completed",
            candidates=candidate_records,
            failure_diagnostics=(),
            trace={
                "legacy_adapter_mode": "non_claim_structure_proposal",
                "search_class": self.search_class,
                "candidate_ids": [proposal.candidate_id for proposal in attempted],
            },
            omission_disclosure={
                "canonical_program_count": len(proposals),
                "attempted_candidate_count": len(attempted),
                "omitted_by_candidate_filter": omitted_by_filter,
                "omitted_by_budget": omitted_by_budget,
                "restart_count_used": selection.restart_count_used,
            },
            replay_metadata={
                **context.replay_metadata(),
                "engine_id": self.engine_id,
                "engine_version": self.engine_version,
            },
        )

    def _record_from_proposal(
        self,
        *,
        proposal,
        proposal_rank: int,
        context: EngineInputContext,
        canonical_program_count: int,
        restart_count_used: int,
    ) -> EngineCandidateRecord:
        return EngineCandidateRecord(
            candidate_id=proposal.candidate_id,
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            search_class=context.search_class,
            search_space_declaration="native_retained_fragment_space_v1",
            budget_declaration={
                "proposal_limit": context.proposal_limit,
                "timeout_seconds": context.timeout_seconds,
            },
            rows_used=tuple(
                row["event_time"]
                for row in context.rows_features_access.row_fingerprints
            ),
            features_used=tuple(proposal.feature_dependencies),
            random_seed=context.random_seed,
            candidate_trace={
                "proposal_rank": proposal_rank,
                "primitive_family": proposal.primitive_family,
                "form_class": proposal.form_class,
                "legacy_adapter_mode": "non_claim_structure_proposal",
            },
            omission_disclosure={
                "canonical_program_count": canonical_program_count,
                "restart_count_used": restart_count_used,
            },
            claim_boundary=engine_claim_boundary(),
            lowering_kind="descriptive_proposal",
            lowerable_payload=_proposal_payload(proposal),
        )


def _proposal_payload(proposal) -> dict[str, Any]:
    return {
        "candidate_id": proposal.candidate_id,
        "primitive_family": proposal.primitive_family,
        "form_class": proposal.form_class,
        "feature_dependencies": list(proposal.feature_dependencies),
        "parameter_values": dict(proposal.parameter_values),
        "literal_values": dict(proposal.literal_values),
        "persistent_state": dict(proposal.persistent_state),
        "composition_payload": proposal.composition_payload,
        "history_access_mode": proposal.history_access_mode,
        "max_lag": proposal.max_lag,
        "required_observation_model_family": proposal.required_observation_model_family,
    }


__all__ = ["NativeFragmentEngine"]
