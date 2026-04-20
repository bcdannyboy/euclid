from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from euclid.cir.models import (
    CandidateIntermediateRepresentation,
    CIRBackendOriginRecord,
)
from euclid.cir.normalize import rebind_cir_backend_origin
from euclid.reducers.models import BoundObservationModel, ScalarValue
from euclid.search.backends import (
    AnalyticSearchBackendAdapter,
    DescriptiveSearchProposal,
)


@dataclass(frozen=True)
class DecompositionAdapterCandidate:
    candidate_id: str
    form_class: str = "closed_form_expression"
    feature_dependencies: tuple[str, ...] = ()
    parameter_values: Mapping[str, ScalarValue] = field(default_factory=dict)
    literal_values: Mapping[str, ScalarValue] = field(default_factory=dict)
    composition_payload: Mapping[str, Any] | None = None
    history_access_mode: str = "full_prefix"
    max_lag: int | None = None


def normalize_decomposition_candidate(
    *,
    spec: DecompositionAdapterCandidate,
    search_plan,
    feature_view,
    observation_model: BoundObservationModel,
    proposal_rank: int = 0,
) -> CandidateIntermediateRepresentation:
    adapter = AnalyticSearchBackendAdapter()
    proposal = DescriptiveSearchProposal(
        candidate_id=spec.candidate_id,
        primitive_family=adapter.family_id,
        form_class=spec.form_class,
        feature_dependencies=spec.feature_dependencies,
        parameter_values=spec.parameter_values,
        literal_values=spec.literal_values,
        composition_payload=spec.composition_payload,
        history_access_mode=spec.history_access_mode,
        max_lag=spec.max_lag,
    )
    candidate = adapter.realize_proposal(
        proposal=proposal,
        proposal_rank=proposal_rank,
        search_plan=search_plan,
        feature_view=feature_view,
        observation_model=observation_model,
    )
    return rebind_cir_backend_origin(
        candidate,
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="ai_feynman-decomposition",
            adapter_class="decomposition",
            source_candidate_id=spec.candidate_id,
            search_class=search_plan.search_class,
            backend_family=adapter.family_id,
            proposal_rank=proposal_rank,
            backend_private_fields=("expression_trace", "search_trace"),
        ),
        transient_diagnostics={
            "backend_adapter_contract": {
                "adapter_class": "decomposition",
                "source_candidate_id": spec.candidate_id,
            }
        },
    )


__all__ = [
    "DecompositionAdapterCandidate",
    "normalize_decomposition_candidate",
]
