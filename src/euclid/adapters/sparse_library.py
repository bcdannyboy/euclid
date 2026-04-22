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
    DescriptiveSearchProposal,
    RecursiveSearchBackendAdapter,
    SpectralSearchBackendAdapter,
)

_LEGACY_ADAPTERS = {
    "recursive": RecursiveSearchBackendAdapter,
    "spectral": SpectralSearchBackendAdapter,
}


@dataclass(frozen=True)
class LegacySparseProposal:
    candidate_id: str
    primitive_family: str
    form_class: str
    feature_dependencies: tuple[str, ...] = ()
    parameter_values: Mapping[str, ScalarValue] = field(default_factory=dict)
    literal_values: Mapping[str, ScalarValue] = field(default_factory=dict)
    persistent_state: Mapping[str, ScalarValue] = field(default_factory=dict)
    composition_payload: Mapping[str, Any] | None = None
    history_access_mode: str = "full_prefix"
    max_lag: int | None = None


def normalize_legacy_sparse_candidate(
    *,
    spec: LegacySparseProposal,
    search_plan,
    feature_view,
    observation_model: BoundObservationModel,
    proposal_rank: int = 0,
) -> CandidateIntermediateRepresentation:
    try:
        adapter = _LEGACY_ADAPTERS[spec.primitive_family]()
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"unsupported legacy sparse primitive family: {spec.primitive_family!r}"
        ) from exc
    proposal = DescriptiveSearchProposal(
        candidate_id=spec.candidate_id,
        primitive_family=spec.primitive_family,
        form_class=spec.form_class,
        feature_dependencies=spec.feature_dependencies,
        parameter_values=spec.parameter_values,
        literal_values=spec.literal_values,
        persistent_state=spec.persistent_state,
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
            adapter_id="legacy-non-claim-sparse-compat",
            adapter_class="legacy_non_claim_sparse_adapter",
            source_candidate_id=spec.candidate_id,
            search_class=search_plan.search_class,
            backend_family=spec.primitive_family,
            proposal_rank=proposal_rank,
            comparability_scope="legacy_compatibility_only_not_production_evidence",
            backend_private_fields=(
                "legacy_compatibility_trace",
                "replacement_engine_trace",
            ),
        ),
        transient_diagnostics={
            "legacy_non_claim_adapter": {
                "adapter_class": "legacy_non_claim_sparse_adapter",
                "source_candidate_id": spec.candidate_id,
                "production_evidence_allowed": False,
                "replacement_engine_id": "pysindy-engine-v1",
                "reason_codes": [
                    "legacy_relabel_adapter_not_production_evidence",
                    "external_engine_trace_required_for_sindy_claims",
                ],
            }
        },
    )


__all__ = [
    "LegacySparseProposal",
    "normalize_legacy_sparse_candidate",
]
