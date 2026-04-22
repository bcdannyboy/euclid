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
class LegacyDecompositionProposal:
    candidate_id: str
    form_class: str = "closed_form_expression"
    feature_dependencies: tuple[str, ...] = ()
    parameter_values: Mapping[str, ScalarValue] = field(default_factory=dict)
    literal_values: Mapping[str, ScalarValue] = field(default_factory=dict)
    composition_payload: Mapping[str, Any] | None = None
    history_access_mode: str = "full_prefix"
    max_lag: int | None = None


def normalize_legacy_decomposition_candidate(
    *,
    spec: LegacyDecompositionProposal,
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
            adapter_id="legacy-non-claim-decomposition-compat",
            adapter_class="legacy_non_claim_decomposition_adapter",
            source_candidate_id=spec.candidate_id,
            search_class=search_plan.search_class,
            backend_family=adapter.family_id,
            proposal_rank=proposal_rank,
            comparability_scope="legacy_compatibility_only_not_production_evidence",
            backend_private_fields=(
                "legacy_compatibility_trace",
                "replacement_engine_trace",
            ),
        ),
        transient_diagnostics={
            "legacy_non_claim_adapter": {
                "adapter_class": "legacy_non_claim_decomposition_adapter",
                "source_candidate_id": spec.candidate_id,
                "production_evidence_allowed": False,
                "replacement_engine_id": "decomposition-engine-v1",
                "reason_codes": [
                    "legacy_relabel_adapter_not_production_evidence",
                    "real_decomposition_trace_required_for_decomposition_claims",
                ],
            }
        },
    )


__all__ = [
    "LegacyDecompositionProposal",
    "normalize_legacy_decomposition_candidate",
]
