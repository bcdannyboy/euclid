from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from euclid.adapters.algorithmic_dsl import enumerate_algorithmic_proposal_specs
from euclid.algorithmic_dsl import parse_algorithmic_program
from euclid.cir.models import (
    CandidateIntermediateRepresentation,
    CIRBackendOriginRecord,
    CIRForecastOperator,
    CIRHistoryAccessContract,
    CIRInputSignature,
    CIRLiteral,
    CIRLiteralBlock,
    CIRModelCodeDecomposition,
    CIRReplayHook,
    CIRReplayHooks,
)
from euclid.cir.normalize import (
    build_cir_candidate_from_algorithmic_program,
    build_cir_candidate_from_reducer,
    rebind_cir_backend_origin,
)
from euclid.contracts.errors import ContractValidationError
from euclid.manifests.runtime_models import SearchPlanManifest
from euclid.math.observation_models import PointObservationModel
from euclid.modules.candidate_fitting import fit_candidate_window
from euclid.modules.features import FeatureView
from euclid.modules.split_planning import EvaluationSegment
from euclid.reducers.models import (
    BoundObservationModel,
    ReducerAdmissibilityObject,
    ReducerFamilyId,
    ReducerObject,
    ReducerParameter,
    ReducerParameterObject,
    ReducerStateObject,
    ReducerStateSemantics,
    ReducerStateSlot,
    ReducerStateUpdateContext,
    ReducerStateUpdateRule,
    ScalarValue,
    parse_reducer_composition,
)
from euclid.search.descriptive_coding import (
    DescriptionGainArtifact,
    DescriptiveAdmissibilityDiagnostic,
    evaluate_descriptive_candidates,
)
from euclid.search.frontier import (
    FrontierCandidateMetrics,
    StageLocalFrontierResult,
    construct_stage_local_frontier,
)
from euclid.search.policies import (
    BANNED_DESCRIPTIVE_SCOPE_COMPOSITION_OPERATORS,
    BANNED_DESCRIPTIVE_SCOPE_FORM_CLASSES,
    COMPACT_DESCRIPTIVE_FALLBACK_CANDIDATE_IDS,
    DESCRIPTIVE_SCOPE_SELECTION_RULE,
    REQUIRED_CANDIDATE_SEARCH_EVIDENCE,
    REQUIRED_SEARCH_COVERAGE_DISCLOSURES,
    RETAINED_PRIMITIVE_FAMILIES,
)

_SUPPORTED_SEARCH_CLASSES = frozenset(
    {
        "exact_finite_enumeration",
        "bounded_heuristic",
        "equality_saturation_heuristic",
        "stochastic_heuristic",
    }
)
_FAMILY_PRIORITY = {
    family_id: index for index, family_id in enumerate(RETAINED_PRIMITIVE_FAMILIES)
}


@dataclass(frozen=True)
class _SearchClassSemantics:
    coverage_statement: str
    exactness_ceiling: str
    scope_declaration: str


_SEARCH_CLASS_SEMANTICS = {
    "exact_finite_enumeration": _SearchClassSemantics(
        coverage_statement="complete_over_declared_canonical_program_space",
        exactness_ceiling="exact_over_declared_fragment_only",
        scope_declaration="finite_exactness_limited_to_declared_canonical_program_space",
    ),
    "bounded_heuristic": _SearchClassSemantics(
        coverage_statement="incomplete_search_disclosed",
        exactness_ceiling="no_global_exactness_claim",
        scope_declaration="heuristic_prefix_over_declared_candidate_space",
    ),
    "equality_saturation_heuristic": _SearchClassSemantics(
        coverage_statement="incomplete_search_disclosed",
        exactness_ceiling="no_global_exactness_claim",
        scope_declaration="heuristic_rewrite_neighborhood_with_cost_extraction",
    ),
    "stochastic_heuristic": _SearchClassSemantics(
        coverage_statement="incomplete_search_disclosed",
        exactness_ceiling="no_global_exactness_claim",
        scope_declaration="heuristic_seeded_search_with_declared_restart_policy",
    ),
}


def _require_identifier(value: str, *, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(
            code="invalid_descriptive_search_value",
            message=f"{field_path} must be a non-empty string",
            field_path=field_path,
        )
    return value


def _normalize_scalar(value: ScalarValue, *, field_path: str) -> ScalarValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(
                code="invalid_descriptive_search_value",
                message=f"{field_path} must be finite",
                field_path=field_path,
            )
        return value
    raise ContractValidationError(
        code="invalid_descriptive_search_value",
        message=f"{field_path} must be a scalar literal",
        field_path=field_path,
    )


def _history_access_contract_id(proposal: "DescriptiveSearchProposal") -> str:
    lag_component = (
        "unbounded" if proposal.max_lag is None else f"lag_{proposal.max_lag}"
    )
    dependency_component = (
        "no_features"
        if not proposal.feature_dependencies
        else "__".join(proposal.feature_dependencies)
    )
    return (
        f"history_access__{proposal.history_access_mode}"
        f"__{lag_component}__{dependency_component}"
    )


@dataclass(frozen=True)
class DescriptiveSearchProposal:
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
    required_observation_model_family: str = "gaussian_location_scale"

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, field_path="candidate_id")
        _require_identifier(self.primitive_family, field_path="primitive_family")
        _require_identifier(self.form_class, field_path="form_class")
        _require_identifier(
            self.history_access_mode,
            field_path="history_access_mode",
        )
        _require_identifier(
            self.required_observation_model_family,
            field_path="required_observation_model_family",
        )
        if self.max_lag is not None and self.max_lag < 0:
            raise ContractValidationError(
                code="invalid_descriptive_search_value",
                message="max_lag must be non-negative when provided",
                field_path="max_lag",
            )
        object.__setattr__(
            self,
            "feature_dependencies",
            tuple(
                _require_identifier(
                    dependency,
                    field_path=f"feature_dependencies[{index}]",
                )
                for index, dependency in enumerate(self.feature_dependencies)
            ),
        )
        object.__setattr__(
            self,
            "parameter_values",
            {
                _require_identifier(
                    name,
                    field_path=f"parameter_values[{index}]",
                ): _normalize_scalar(
                    value,
                    field_path=f"parameter_values[{name}]",
                )
                for index, (name, value) in enumerate(
                    sorted(self.parameter_values.items())
                )
            },
        )
        object.__setattr__(
            self,
            "literal_values",
            {
                _require_identifier(
                    name,
                    field_path=f"literal_values[{index}]",
                ): _normalize_scalar(
                    value,
                    field_path=f"literal_values[{name}]",
                )
                for index, (name, value) in enumerate(
                    sorted(self.literal_values.items())
                )
            },
        )
        object.__setattr__(
            self,
            "persistent_state",
            {
                _require_identifier(
                    name,
                    field_path=f"persistent_state[{index}]",
                ): _normalize_scalar(
                    value,
                    field_path=f"persistent_state[{name}]",
                )
                for index, (name, value) in enumerate(
                    sorted(self.persistent_state.items())
                )
            },
        )


@dataclass(frozen=True)
class RejectedSearchDiagnostic:
    candidate_id: str
    primitive_family: str
    reason_code: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchCoverageAccounting:
    search_class: str
    canonical_program_count: int
    attempted_candidate_count: int
    accepted_candidate_count: int
    rejected_candidate_count: int
    omitted_candidate_count: int
    coverage_statement: str
    exactness_ceiling: str
    scope_declaration: str
    disclosures: Mapping[str, Any] = field(default_factory=dict)
    ranked_candidate_count: int = 0
    law_eligible_candidate_count: int = 0
    diagnostic_only_candidate_count: int = 0
    fallback_candidate_count: int = 0
    gap_report_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FamilySearchBackendResult:
    family_id: str
    adapter_id: str
    accepted_candidates: tuple[CandidateIntermediateRepresentation, ...]
    rejected_diagnostics: tuple[RejectedSearchDiagnostic, ...]
    coverage: SearchCoverageAccounting
    descriptive_scope: tuple[CandidateIntermediateRepresentation, ...] = ()
    law_eligible_scope: tuple[CandidateIntermediateRepresentation, ...] = ()
    best_overall_candidate: CandidateIntermediateRepresentation | None = None
    accepted_candidate: CandidateIntermediateRepresentation | None = None


@dataclass(frozen=True)
class DescriptiveSearchRunResult:
    accepted_candidates: tuple[CandidateIntermediateRepresentation, ...]
    description_artifacts: tuple[DescriptionGainArtifact, ...]
    admissibility_diagnostics: tuple[DescriptiveAdmissibilityDiagnostic, ...]
    rejected_diagnostics: tuple[RejectedSearchDiagnostic, ...]
    family_results: tuple[FamilySearchBackendResult, ...]
    coverage: SearchCoverageAccounting
    frontier: DescriptiveSearchFrontierResult
    descriptive_scope: tuple[CandidateIntermediateRepresentation, ...] = ()
    law_eligible_scope: tuple[CandidateIntermediateRepresentation, ...] = ()
    best_overall_candidate: CandidateIntermediateRepresentation | None = None
    accepted_candidate: CandidateIntermediateRepresentation | None = None
    gap_report_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DescriptiveSearchFrontierResult:
    state: StageLocalFrontierResult
    retained_frontier_cir_candidates: tuple[CandidateIntermediateRepresentation, ...]
    frozen_shortlist_cir_candidates: tuple[CandidateIntermediateRepresentation, ...]

    @property
    def frontier_candidates(self) -> tuple[FrontierCandidateMetrics, ...]:
        return self.state.frontier_candidates

    @property
    def retained_frontier_candidates(self) -> tuple[FrontierCandidateMetrics, ...]:
        return self.state.retained_frontier_candidates

    @property
    def frozen_shortlist_candidates(self) -> tuple[FrontierCandidateMetrics, ...]:
        return self.state.frozen_shortlist_candidates

    @property
    def dominance_records(self):
        return self.state.dominance_records

    @property
    def coverage(self):
        return self.state.coverage


@dataclass(frozen=True)
class _AttemptSelection:
    attempted_proposals: tuple[DescriptiveSearchProposal, ...]
    restart_count_used: int = 0
    rewrite_space_candidate_ids: tuple[str, ...] = ()
    restart_records: tuple[Mapping[str, Any], ...] = ()
    declared_stochastic_surfaces: tuple[str, ...] = ()


def _candidate_source_id(candidate: CandidateIntermediateRepresentation) -> str:
    return candidate.evidence_layer.backend_origin_record.source_candidate_id


def _descriptive_scope_sort_key(
    candidate: CandidateIntermediateRepresentation,
    artifact: DescriptionGainArtifact,
) -> tuple[float, float, float, int, str]:
    return (
        float(artifact.L_total_bits),
        -float(artifact.description_gain_bits),
        float(artifact.L_structure_bits),
        len(candidate.canonical_bytes()),
        _candidate_source_id(candidate),
    )


def _descriptive_scope_exclusion_reason(
    candidate: CandidateIntermediateRepresentation,
) -> str | None:
    operator_id = candidate.structural_layer.composition_graph.operator_id
    if operator_id in BANNED_DESCRIPTIVE_SCOPE_COMPOSITION_OPERATORS:
        return "requires_lookup_residual_wrapper"
    form_class = candidate.structural_layer.cir_form_class
    if form_class in BANNED_DESCRIPTIVE_SCOPE_FORM_CLASSES:
        return BANNED_DESCRIPTIVE_SCOPE_FORM_CLASSES[form_class]
    return None


def _partition_descriptive_scope_candidates(
    candidates: Sequence[CandidateIntermediateRepresentation],
    *,
    fallback_bank: bool = False,
) -> tuple[
    list[CandidateIntermediateRepresentation],
    list[RejectedSearchDiagnostic],
]:
    descriptive_scope_candidates: list[CandidateIntermediateRepresentation] = []
    rejected: list[RejectedSearchDiagnostic] = []
    for candidate in candidates:
        exclusion_reason = _descriptive_scope_exclusion_reason(candidate)
        if exclusion_reason is None:
            descriptive_scope_candidates.append(candidate)
            continue
        details: dict[str, Any] = {
            "candidate_hash": candidate.canonical_hash(),
            "reason_codes": [exclusion_reason],
        }
        if fallback_bank:
            details["fallback_bank"] = True
        rejected.append(
            RejectedSearchDiagnostic(
                candidate_id=_candidate_source_id(candidate),
                primitive_family=candidate.structural_layer.cir_family_id,
                reason_code="descriptive_scope_excluded",
                details=details,
            )
        )
    return descriptive_scope_candidates, rejected


def _descriptive_scope_private_fields(
    candidate: CandidateIntermediateRepresentation,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *candidate.evidence_layer.backend_origin_record.backend_private_fields,
                "descriptive_scope",
            )
        )
    )


def _with_descriptive_scope_metadata(
    candidate: CandidateIntermediateRepresentation,
    *,
    scope_rank: int,
    source: str,
    law_eligible: bool,
    law_rejection_reason_codes: Sequence[str],
) -> CandidateIntermediateRepresentation:
    origin = candidate.evidence_layer.backend_origin_record
    return rebind_cir_backend_origin(
        candidate,
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id=origin.adapter_id,
            adapter_class=origin.adapter_class,
            source_candidate_id=origin.source_candidate_id,
            search_class=origin.search_class,
            backend_family=origin.backend_family,
            proposal_rank=origin.proposal_rank,
            normalization_scope=origin.normalization_scope,
            comparability_scope=origin.comparability_scope,
            backend_private_fields=_descriptive_scope_private_fields(candidate),
        ),
        transient_diagnostics={
            "descriptive_scope": {
                "scope_rank": scope_rank,
                "source": source,
                "selection_rule": DESCRIPTIVE_SCOPE_SELECTION_RULE,
                "law_eligible": law_eligible,
                "law_rejection_reason_codes": list(law_rejection_reason_codes),
            }
        },
    )


def _compact_descriptive_fallback_proposals(
    *,
    adapters: Sequence[DescriptiveSearchBackendAdapter],
    search_plan: SearchPlanManifest,
    feature_view: FeatureView,
    allowed_candidate_ids: set[str] | None,
) -> tuple[DescriptiveSearchProposal, ...]:
    proposal_by_id: dict[str, DescriptiveSearchProposal] = {}
    for adapter in adapters:
        for proposal in adapter.default_proposals(
            search_plan=search_plan,
            feature_view=feature_view,
        ):
            proposal_by_id.setdefault(proposal.candidate_id, proposal)
    return tuple(
        proposal_by_id[candidate_id]
        for candidate_id in COMPACT_DESCRIPTIVE_FALLBACK_CANDIDATE_IDS
        if candidate_id in proposal_by_id
        and (
            allowed_candidate_ids is None
            or candidate_id in allowed_candidate_ids
        )
    )


def _canonicalize_proposal_semantics(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (key, _canonicalize_proposal_semantics(nested_value))
            for key, nested_value in sorted(value.items())
        )
    if isinstance(value, tuple):
        return tuple(_canonicalize_proposal_semantics(item) for item in value)
    if isinstance(value, list):
        return tuple(_canonicalize_proposal_semantics(item) for item in value)
    return value


def _proposal_semantic_identity(
    proposal: DescriptiveSearchProposal,
) -> tuple[Any, ...]:
    return (
        proposal.primitive_family,
        proposal.candidate_id,
        proposal.form_class,
        proposal.feature_dependencies,
        tuple(sorted(proposal.parameter_values.items())),
        tuple(sorted(proposal.literal_values.items())),
        tuple(sorted(proposal.persistent_state.items())),
        _canonicalize_proposal_semantics(proposal.composition_payload),
        proposal.history_access_mode,
        proposal.max_lag,
        proposal.required_observation_model_family,
    )


def _unattempted_compact_fallback_proposals(
    *,
    attempted_proposals: Sequence[DescriptiveSearchProposal],
    fallback_proposals: Sequence[DescriptiveSearchProposal],
) -> tuple[DescriptiveSearchProposal, ...]:
    attempted_semantic_identities = {
        _proposal_semantic_identity(proposal) for proposal in attempted_proposals
    }
    return tuple(
        proposal
        for proposal in fallback_proposals
        if _proposal_semantic_identity(proposal) not in attempted_semantic_identities
    )


def _proposal_count_by_family(
    *,
    adapter_families: Sequence[str],
    proposals: Sequence[DescriptiveSearchProposal],
) -> dict[str, int]:
    counts = {family_id: 0 for family_id in adapter_families}
    for proposal in proposals:
        if proposal.primitive_family in counts:
            counts[proposal.primitive_family] += 1
    return counts


def _exact_enumeration_sort_key(
    proposal: DescriptiveSearchProposal,
) -> tuple[int, str]:
    return (
        _FAMILY_PRIORITY.get(proposal.primitive_family, len(_FAMILY_PRIORITY)),
        proposal.candidate_id,
    )


def _ordered_exact_enumeration_proposals(
    proposals: Sequence[DescriptiveSearchProposal],
) -> tuple[DescriptiveSearchProposal, ...]:
    return tuple(sorted(proposals, key=_exact_enumeration_sort_key))


class DescriptiveSearchBackendAdapter:
    family_id: str
    adapter_id: str
    adapter_class: str = "bounded_grammar"

    def default_proposals(
        self,
        *,
        search_plan: SearchPlanManifest,
        feature_view: FeatureView,
    ) -> tuple[DescriptiveSearchProposal, ...]:
        raise NotImplementedError

    def realize_proposal(
        self,
        *,
        proposal: DescriptiveSearchProposal,
        proposal_rank: int,
        search_plan: SearchPlanManifest,
        feature_view: FeatureView,
        observation_model: BoundObservationModel,
        coverage_disclosures: Mapping[str, Any] | None = None,
    ) -> CandidateIntermediateRepresentation:
        if proposal.primitive_family != self.family_id:
            raise _SearchRealizationError(
                "family_membership_failed",
                {
                    "candidate_id": proposal.candidate_id,
                    "adapter_family_id": self.family_id,
                    "proposal_family_id": proposal.primitive_family,
                },
            )
        if proposal.required_observation_model_family != observation_model.family:
            raise _SearchRealizationError(
                "observation_model_incompatible",
                {
                    "candidate_id": proposal.candidate_id,
                    "required_family": proposal.required_observation_model_family,
                    "bound_family": observation_model.family,
                },
            )
        self._validate_bounds(
            proposal=proposal,
            search_plan=search_plan,
            feature_view=feature_view,
        )
        try:
            composition_object = (
                parse_reducer_composition(dict(proposal.composition_payload))
                if proposal.composition_payload is not None
                else parse_reducer_composition({})
            )
        except ContractValidationError as exc:
            raise _SearchRealizationError(
                "syntax_invalid",
                {
                    "candidate_id": proposal.candidate_id,
                    "error_code": exc.code,
                    "field_path": exc.field_path,
                },
            ) from exc

        reducer = ReducerObject(
            family=ReducerFamilyId(self.family_id),
            composition_object=composition_object,
            fitted_parameters=ReducerParameterObject(
                parameters=tuple(
                    ReducerParameter(name=name, value=value)
                    for name, value in proposal.parameter_values.items()
                )
            ),
            state_semantics=ReducerStateSemantics(
                persistent_state=ReducerStateObject(
                    slots=tuple(
                        ReducerStateSlot(name=name, value=value)
                        for name, value in proposal.persistent_state.items()
                    )
                ),
                update_rule=self._update_rule(proposal, search_plan),
            ),
            observation_model=observation_model,
            admissibility=ReducerAdmissibilityObject(
                family_membership=True,
                composition_closure=True,
                observation_model_compatibility=True,
                valid_state_semantics=True,
                codelength_comparability=True,
            ),
        )
        return build_cir_candidate_from_reducer(
            reducer=reducer,
            cir_form_class=proposal.form_class,
            input_signature=CIRInputSignature(
                target_series="target",
                side_information_fields=feature_view.feature_names,
            ),
            history_access_contract=CIRHistoryAccessContract(
                contract_id=_history_access_contract_id(proposal),
                access_mode=proposal.history_access_mode,
                max_lag=proposal.max_lag,
                allowed_side_information=proposal.feature_dependencies,
            ),
            literal_block=CIRLiteralBlock(
                literals=tuple(
                    CIRLiteral(name=name, value=value)
                    for name, value in proposal.literal_values.items()
                )
            ),
            forecast_operator=CIRForecastOperator(
                operator_id="one_step_point_forecast",
                horizon=1,
            ),
            model_code_decomposition=_model_code_decomposition(proposal),
            backend_origin_record=CIRBackendOriginRecord(
                adapter_id=self.adapter_id,
                adapter_class=self.adapter_class,
                source_candidate_id=proposal.candidate_id,
                search_class=search_plan.search_class,
                backend_family=self.family_id,
                proposal_rank=proposal_rank,
            ),
            replay_hooks=_build_replay_hooks(
                search_plan=search_plan,
                coverage_disclosures=coverage_disclosures
                or _coverage_disclosures(
                    search_plan=search_plan,
                    canonical_program_count=0,
                    restart_count_used=0,
                ),
            ),
            transient_diagnostics={
                "feature_dependencies": list(proposal.feature_dependencies),
            },
        )

    def _validate_bounds(
        self,
        *,
        proposal: DescriptiveSearchProposal,
        search_plan: SearchPlanManifest,
        feature_view: FeatureView,
    ) -> None:
        missing_dependencies = [
            feature_name
            for feature_name in proposal.feature_dependencies
            if feature_name not in feature_view.feature_names
        ]
        if missing_dependencies:
            raise _SearchRealizationError(
                "bounds_invalid",
                {
                    "candidate_id": proposal.candidate_id,
                    "missing_feature_dependencies": missing_dependencies,
                },
            )
        if proposal.max_lag is not None and proposal.max_lag > max(
            len(feature_view.rows) - 1, 0
        ):
            raise _SearchRealizationError(
                "bounds_invalid",
                {
                    "candidate_id": proposal.candidate_id,
                    "max_lag": proposal.max_lag,
                    "available_history_length": max(len(feature_view.rows) - 1, 0),
                },
            )

    def _update_rule(
        self,
        proposal: DescriptiveSearchProposal,
        search_plan: SearchPlanManifest,
    ) -> ReducerStateUpdateRule:
        raise NotImplementedError


class AnalyticSearchBackendAdapter(DescriptiveSearchBackendAdapter):
    family_id = "analytic"
    adapter_id = "analytic-search"

    def default_proposals(
        self,
        *,
        search_plan: SearchPlanManifest,
        feature_view: FeatureView,
    ) -> tuple[DescriptiveSearchProposal, ...]:
        proposals = [
            DescriptiveSearchProposal(
                candidate_id="analytic_intercept",
                primitive_family=self.family_id,
                form_class="closed_form_expression",
                parameter_values={"intercept": 13.0},
            )
        ]
        if "lag_1" in feature_view.feature_names:
            proposals.append(
                DescriptiveSearchProposal(
                    candidate_id="analytic_lag1_affine",
                    primitive_family=self.family_id,
                    form_class="closed_form_expression",
                    feature_dependencies=("lag_1",),
                    parameter_values={
                        "intercept": 1.0,
                        "lag_coefficient": 1.0,
                    },
                    history_access_mode="bounded_lag_window",
                    max_lag=1,
                )
            )
        return tuple(proposals)

    def _update_rule(
        self,
        proposal: DescriptiveSearchProposal,
        search_plan: SearchPlanManifest,
    ) -> ReducerStateUpdateRule:
        return ReducerStateUpdateRule(
            update_rule_id="analytic_identity_update",
            implementation=_identity_update,
        )


class RecursiveSearchBackendAdapter(DescriptiveSearchBackendAdapter):
    family_id = "recursive"
    adapter_id = "recursive-search"

    def default_proposals(
        self,
        *,
        search_plan: SearchPlanManifest,
        feature_view: FeatureView,
    ) -> tuple[DescriptiveSearchProposal, ...]:
        targets = _target_values(feature_view)
        first_target = targets[0]
        return (
            DescriptiveSearchProposal(
                candidate_id="recursive_level_smoother",
                primitive_family=self.family_id,
                form_class="state_recurrence",
                literal_values={"alpha": 0.5},
                persistent_state={"level": first_target, "step_count": 0},
                history_access_mode="bounded_lag_window",
                max_lag=1,
            ),
            DescriptiveSearchProposal(
                candidate_id="recursive_running_mean",
                primitive_family=self.family_id,
                form_class="state_recurrence",
                literal_values={"window_mass": len(targets)},
                persistent_state={
                    "running_mean": first_target,
                    "step_count": 1,
                },
                history_access_mode="bounded_lag_window",
                max_lag=1,
            ),
        )

    def _update_rule(
        self,
        proposal: DescriptiveSearchProposal,
        search_plan: SearchPlanManifest,
    ) -> ReducerStateUpdateRule:
        if proposal.candidate_id == "recursive_level_smoother":
            alpha = float(proposal.literal_values["alpha"])

            def update(
                state: ReducerStateObject,
                context: ReducerStateUpdateContext,
            ) -> ReducerStateObject:
                previous_level = float(state.get("level"))
                observed = (
                    previous_level
                    if not context.history
                    else float(context.history[-1])
                )
                next_level = (alpha * observed) + ((1.0 - alpha) * previous_level)
                return ReducerStateObject(
                    slots=(
                        ReducerStateSlot(name="level", value=_stable_float(next_level)),
                        ReducerStateSlot(
                            name="step_count",
                            value=int(state.get("step_count")) + 1,
                        ),
                    )
                )

            return ReducerStateUpdateRule(
                update_rule_id="recursive_level_smoother_update",
                implementation=update,
            )

        def update(
            state: ReducerStateObject,
            context: ReducerStateUpdateContext,
        ) -> ReducerStateObject:
            previous_mean = float(state.get("running_mean"))
            previous_steps = int(state.get("step_count"))
            observed = (
                previous_mean if not context.history else float(context.history[-1])
            )
            next_steps = previous_steps + 1
            next_mean = ((previous_mean * previous_steps) + observed) / next_steps
            return ReducerStateObject(
                slots=(
                    ReducerStateSlot(
                        name="running_mean",
                        value=_stable_float(next_mean),
                    ),
                    ReducerStateSlot(name="step_count", value=next_steps),
                )
            )

        return ReducerStateUpdateRule(
            update_rule_id="recursive_running_mean_update",
            implementation=update,
        )


class SpectralSearchBackendAdapter(DescriptiveSearchBackendAdapter):
    family_id = "spectral"
    adapter_id = "spectral-search"

    def default_proposals(
        self,
        *,
        search_plan: SearchPlanManifest,
        feature_view: FeatureView,
    ) -> tuple[DescriptiveSearchProposal, ...]:
        season_length = search_plan.seasonal_period
        if season_length is None:
            return ()
        amplitude = _stable_float(
            (_max_target(feature_view) - _min_target(feature_view)) / 2.0
        )
        proposals = [
            DescriptiveSearchProposal(
                candidate_id="spectral_harmonic_1",
                primitive_family=self.family_id,
                form_class="spectral_basis_expansion",
                literal_values={"harmonic": 1, "season_length": season_length},
                parameter_values={
                    "cosine_coefficient": 0.0,
                    "sine_coefficient": amplitude,
                },
                persistent_state={"last_basis_value": 0.0, "phase_index": 0},
            )
        ]
        if season_length >= 4:
            proposals.append(
                DescriptiveSearchProposal(
                    candidate_id="spectral_harmonic_2",
                    primitive_family=self.family_id,
                    form_class="spectral_basis_expansion",
                    literal_values={"harmonic": 2, "season_length": season_length},
                    parameter_values={
                        "cosine_coefficient": _stable_float(amplitude / 2.0),
                        "sine_coefficient": 0.0,
                    },
                    persistent_state={"last_basis_value": 0.0, "phase_index": 0},
                )
            )
        return tuple(proposals)

    def _validate_bounds(
        self,
        *,
        proposal: DescriptiveSearchProposal,
        search_plan: SearchPlanManifest,
        feature_view: FeatureView,
    ) -> None:
        super()._validate_bounds(
            proposal=proposal,
            search_plan=search_plan,
            feature_view=feature_view,
        )
        season_length = int(proposal.literal_values.get("season_length", 0))
        harmonic = int(proposal.literal_values.get("harmonic", 0))
        if season_length < 2:
            raise _SearchRealizationError(
                "bounds_invalid",
                {
                    "candidate_id": proposal.candidate_id,
                    "season_length": season_length,
                },
            )
        if season_length > len(feature_view.rows):
            raise _SearchRealizationError(
                "bounds_invalid",
                {
                    "candidate_id": proposal.candidate_id,
                    "season_length": season_length,
                    "feature_row_count": len(feature_view.rows),
                },
            )
        if harmonic < 1 or harmonic > max(season_length // 2, 1):
            raise _SearchRealizationError(
                "bounds_invalid",
                {
                    "candidate_id": proposal.candidate_id,
                    "harmonic": harmonic,
                    "season_length": season_length,
                },
            )

    def _update_rule(
        self,
        proposal: DescriptiveSearchProposal,
        search_plan: SearchPlanManifest,
    ) -> ReducerStateUpdateRule:
        season_length = int(proposal.literal_values["season_length"])
        harmonic = int(proposal.literal_values["harmonic"])

        def update(
            state: ReducerStateObject,
            context: ReducerStateUpdateContext,
        ) -> ReducerStateObject:
            phase_index = (int(state.get("phase_index")) + 1) % season_length
            angle = (2.0 * math.pi * harmonic * phase_index) / season_length
            basis_value = math.sin(angle)
            return ReducerStateObject(
                slots=(
                    ReducerStateSlot(
                        name="last_basis_value",
                        value=_stable_float(basis_value),
                    ),
                    ReducerStateSlot(name="phase_index", value=phase_index),
                )
            )

        return ReducerStateUpdateRule(
            update_rule_id="spectral_harmonic_basis_update",
            implementation=update,
        )


class AlgorithmicSearchBackendAdapter(DescriptiveSearchBackendAdapter):
    family_id = "algorithmic"
    adapter_id = "algorithmic-search"

    def default_proposals(
        self,
        *,
        search_plan: SearchPlanManifest,
        feature_view: FeatureView,
    ) -> tuple[DescriptiveSearchProposal, ...]:
        del search_plan, feature_view
        return tuple(
            DescriptiveSearchProposal(
                candidate_id=spec.candidate_id,
                primitive_family=self.family_id,
                form_class="bounded_program",
                literal_values={
                    "algorithmic_program": spec.program.canonical_source,
                    "algorithmic_state_slot_count": spec.program.state_slot_count,
                    "program_node_count": spec.program.node_count,
                },
                history_access_mode=(
                    "causal_current_observation"
                    if max(spec.program.allowed_observation_lags) == 0
                    else "bounded_lag_window"
                ),
                max_lag=max(spec.program.allowed_observation_lags),
            )
            for spec in enumerate_algorithmic_proposal_specs()
        )

    def realize_proposal(
        self,
        *,
        proposal: DescriptiveSearchProposal,
        proposal_rank: int,
        search_plan: SearchPlanManifest,
        feature_view: FeatureView,
        observation_model: BoundObservationModel,
        coverage_disclosures: Mapping[str, Any] | None = None,
    ) -> CandidateIntermediateRepresentation:
        if proposal.primitive_family != self.family_id:
            raise _SearchRealizationError(
                "family_membership_failed",
                {
                    "candidate_id": proposal.candidate_id,
                    "adapter_family_id": self.family_id,
                    "proposal_family_id": proposal.primitive_family,
                },
            )
        if proposal.required_observation_model_family != observation_model.family:
            raise _SearchRealizationError(
                "observation_model_incompatible",
                {
                    "candidate_id": proposal.candidate_id,
                    "required_family": proposal.required_observation_model_family,
                    "bound_family": observation_model.family,
                },
            )
        self._validate_bounds(
            proposal=proposal,
            search_plan=search_plan,
            feature_view=feature_view,
        )
        program_source = proposal.literal_values.get("algorithmic_program")
        if not isinstance(program_source, str):
            raise _SearchRealizationError(
                "parse_failed",
                {
                    "candidate_id": proposal.candidate_id,
                    "field_path": "literal_values.algorithmic_program",
                },
            )
        state_slot_count = int(
            proposal.literal_values.get("algorithmic_state_slot_count", 1)
        )
        if state_slot_count < 1:
            raise _SearchRealizationError(
                "bound_error",
                {
                    "candidate_id": proposal.candidate_id,
                    "field_path": "literal_values.algorithmic_state_slot_count",
                    "state_slot_count": state_slot_count,
                },
            )
        max_program_nodes = int(proposal.literal_values.get("program_node_count", 8))
        if max_program_nodes < 1:
            raise _SearchRealizationError(
                "bound_error",
                {
                    "candidate_id": proposal.candidate_id,
                    "field_path": "literal_values.program_node_count",
                    "program_node_count": max_program_nodes,
                },
            )
        allowed_observation_lags = tuple(range((proposal.max_lag or 0) + 1))
        try:
            program = parse_algorithmic_program(
                program_source,
                state_slot_count=state_slot_count,
                max_program_nodes=max_program_nodes,
                allowed_observation_lags=allowed_observation_lags,
            )
        except ContractValidationError as exc:
            raise _SearchRealizationError(
                exc.code,
                {
                    "candidate_id": proposal.candidate_id,
                    "field_path": exc.field_path,
                },
            ) from exc
        return build_cir_candidate_from_algorithmic_program(
            program=program,
            cir_form_class=proposal.form_class,
            input_signature=CIRInputSignature(
                target_series="target",
                side_information_fields=feature_view.feature_names,
            ),
            observation_model=observation_model,
            forecast_operator=CIRForecastOperator(
                operator_id="one_step_point_forecast",
                horizon=1,
            ),
            model_code_decomposition=CIRModelCodeDecomposition(
                L_family_bits=2.0,
                L_structure_bits=float(program.node_count),
                L_literals_bits=2.0,
                L_params_bits=0.0,
                L_state_bits=float(program.state_slot_count),
            ),
            backend_origin_record=CIRBackendOriginRecord(
                adapter_id=self.adapter_id,
                adapter_class=self.adapter_class,
                source_candidate_id=proposal.candidate_id,
                search_class=search_plan.search_class,
                backend_family=self.family_id,
                proposal_rank=proposal_rank,
            ),
            replay_hooks=_build_replay_hooks(
                search_plan=search_plan,
                coverage_disclosures=coverage_disclosures
                or _coverage_disclosures(
                    search_plan=search_plan,
                    canonical_program_count=0,
                    restart_count_used=0,
                ),
                extra_hooks=(
                    CIRReplayHook(
                        hook_name="algorithmic_dsl",
                        hook_ref="dsl:canonical_algorithmic_reducer_fragment",
                    ),
                ),
            ),
            transient_diagnostics={},
        )

    def _update_rule(
        self,
        proposal: DescriptiveSearchProposal,
        search_plan: SearchPlanManifest,
    ) -> ReducerStateUpdateRule:
        del proposal, search_plan
        raise NotImplementedError("algorithmic proposals override realize_proposal")


def run_descriptive_search_backends(
    *,
    search_plan: SearchPlanManifest,
    feature_view: FeatureView,
    proposal_specs: Sequence[DescriptiveSearchProposal] = (),
    include_default_grammar: bool = True,
    adapters: Sequence[DescriptiveSearchBackendAdapter] | None = None,
    observation_model: BoundObservationModel | None = None,
) -> DescriptiveSearchRunResult:
    if search_plan.search_class not in _SUPPORTED_SEARCH_CLASSES:
        raise ContractValidationError(
            code="unsupported_search_class",
            message=(
                "descriptive search backends support exact_finite_enumeration, "
                "bounded_heuristic, equality_saturation_heuristic, and "
                "stochastic_heuristic"
            ),
            field_path="search_plan.search_class",
        )
    legal_feature_view = feature_view.require_stage_reuse("search")
    bound_observation_model = observation_model or BoundObservationModel.from_runtime(
        PointObservationModel()
    )
    adapter_list = tuple(adapters or _default_adapters())
    adapter_by_family = {adapter.family_id: adapter for adapter in adapter_list}
    allowed_candidate_ids = (
        set(search_plan.candidate_family_ids)
        if search_plan.candidate_family_ids
        else None
    )

    grouped_proposals: dict[str, list[DescriptiveSearchProposal]] = {
        adapter.family_id: [] for adapter in adapter_list
    }
    unknown_family_proposals: list[DescriptiveSearchProposal] = []

    if include_default_grammar:
        for adapter in adapter_list:
            for proposal in adapter.default_proposals(
                search_plan=search_plan,
                feature_view=legal_feature_view,
            ):
                if (
                    allowed_candidate_ids is not None
                    and proposal.candidate_id not in allowed_candidate_ids
                ):
                    continue
                grouped_proposals[adapter.family_id].append(proposal)

    for proposal in proposal_specs:
        if (
            allowed_candidate_ids is not None
            and proposal.candidate_id not in allowed_candidate_ids
        ):
            continue
        adapter = adapter_by_family.get(proposal.primitive_family)
        if adapter is None:
            unknown_family_proposals.append(proposal)
            continue
        grouped_proposals[proposal.primitive_family].append(proposal)

    ordered_proposals = tuple(
        proposal
        for adapter in adapter_list
        for proposal in grouped_proposals[adapter.family_id]
    ) + tuple(unknown_family_proposals)
    fallback_proposals = _compact_descriptive_fallback_proposals(
        adapters=adapter_list,
        search_plan=search_plan,
        feature_view=legal_feature_view,
        allowed_candidate_ids=allowed_candidate_ids,
    )
    canonical_program_count = len(ordered_proposals)
    if (
        search_plan.search_class == "exact_finite_enumeration"
        and search_plan.proposal_limit < canonical_program_count
    ):
        raise ContractValidationError(
            code="exact_search_budget_too_small",
            message=(
                "proposal_limit must cover the full declared canonical program "
                "space for exact_finite_enumeration"
            ),
            field_path="search_plan.proposal_limit",
            details={
                "proposal_limit": search_plan.proposal_limit,
                "canonical_program_count": canonical_program_count,
            },
        )

    attempt_selection = _select_attempted_proposals(
        ordered_proposals=ordered_proposals,
        search_plan=search_plan,
    )
    attempted_proposals = attempt_selection.attempted_proposals
    coverage_disclosures = _coverage_disclosures(
        search_plan=search_plan,
        canonical_program_count=canonical_program_count,
        restart_count_used=attempt_selection.restart_count_used,
    )

    seen_hashes: set[str] = set()
    realized_candidates: list[CandidateIntermediateRepresentation] = []
    rejected_diagnostics: list[RejectedSearchDiagnostic] = []
    rejected_by_family: dict[str, list[RejectedSearchDiagnostic]] = {
        adapter.family_id: [] for adapter in adapter_list
    }
    attempted_by_family: dict[str, int] = {
        adapter.family_id: 0 for adapter in adapter_list
    }
    fallback_accounted_by_family: dict[str, int] = {
        adapter.family_id: 0 for adapter in adapter_list
    }
    fallback_attempted_by_family: dict[str, int] = {
        adapter.family_id: 0 for adapter in adapter_list
    }
    fallback_attempted_candidate_count = 0

    for rank, proposal in enumerate(attempted_proposals):
        adapter = adapter_by_family.get(proposal.primitive_family)
        if (
            adapter is None
            or proposal.primitive_family not in RETAINED_PRIMITIVE_FAMILIES
        ):
            diagnostic = RejectedSearchDiagnostic(
                candidate_id=proposal.candidate_id,
                primitive_family=proposal.primitive_family,
                reason_code="family_membership_failed",
                details={
                    "candidate_id": proposal.candidate_id,
                    "primitive_family": proposal.primitive_family,
                },
            )
            rejected_diagnostics.append(diagnostic)
            continue
        attempted_by_family[proposal.primitive_family] += 1
        try:
            candidate = adapter.realize_proposal(
                proposal=proposal,
                proposal_rank=rank,
                search_plan=search_plan,
                feature_view=legal_feature_view,
                observation_model=bound_observation_model,
                coverage_disclosures=coverage_disclosures,
            )
        except _SearchRealizationError as exc:
            diagnostic = RejectedSearchDiagnostic(
                candidate_id=proposal.candidate_id,
                primitive_family=proposal.primitive_family,
                reason_code=exc.reason_code,
                details=exc.details,
            )
            rejected_diagnostics.append(diagnostic)
            rejected_by_family[proposal.primitive_family].append(diagnostic)
            continue
        candidate_hash = candidate.canonical_hash()
        if candidate_hash in seen_hashes:
            diagnostic = RejectedSearchDiagnostic(
                candidate_id=proposal.candidate_id,
                primitive_family=proposal.primitive_family,
                reason_code="duplicate_canonical_candidate",
                details={"candidate_hash": candidate_hash},
            )
            rejected_diagnostics.append(diagnostic)
            rejected_by_family[proposal.primitive_family].append(diagnostic)
            continue
        seen_hashes.add(candidate_hash)
        realized_candidates.append(candidate)

    descriptive_scope_candidates, excluded_candidates = (
        _partition_descriptive_scope_candidates(realized_candidates)
    )
    for rejected in excluded_candidates:
        rejected_diagnostics.append(rejected)
        rejected_by_family[rejected.primitive_family].append(rejected)

    descriptive_result = evaluate_descriptive_candidates(
        tuple(descriptive_scope_candidates),
        feature_view=legal_feature_view,
        minimum_description_gain_bits=search_plan.minimum_description_gain_bits,
    )
    candidate_by_hash = {
        candidate.canonical_hash(): candidate
        for candidate in descriptive_scope_candidates
    }
    diagnostic_by_hash = {
        diagnostic.candidate_hash: diagnostic
        for diagnostic in descriptive_result.admissibility_diagnostics
    }
    rankable_candidates: list[CandidateIntermediateRepresentation] = []
    rankable_artifacts: list[DescriptionGainArtifact] = []
    for artifact in descriptive_result.description_artifacts:
        candidate = candidate_by_hash.get(artifact.candidate_hash)
        if candidate is None:
            raise ContractValidationError(
                code="missing_ranked_descriptive_candidate",
                message=(
                    "descriptive artifacts must map back to realized CIR "
                    "candidates before descriptive_scope ranking"
                ),
                field_path="description_artifacts",
                details={"candidate_hash": artifact.candidate_hash},
            )
        rankable_candidates.append(candidate)
        rankable_artifacts.append(artifact)
    for diagnostic in descriptive_result.admissibility_diagnostics:
        if diagnostic.is_admissible:
            continue
        rejected = RejectedSearchDiagnostic(
            candidate_id=diagnostic.candidate_id,
            primitive_family=diagnostic.primitive_family,
            reason_code="descriptive_admissibility_failed",
            details={
                "candidate_hash": diagnostic.candidate_hash,
                "reason_codes": list(diagnostic.reason_codes),
                **dict(diagnostic.details),
            },
        )
        rejected_diagnostics.append(rejected)
        rejected_by_family[diagnostic.primitive_family].append(rejected)
    all_description_artifacts = list(descriptive_result.description_artifacts)
    all_admissibility_diagnostics = list(descriptive_result.admissibility_diagnostics)
    descriptive_scope_source = "primary_search"
    if not rankable_candidates:
        if not fallback_proposals:
            if rejected_diagnostics and len(attempted_proposals) == 1:
                return _diagnostic_only_search_result(
                    search_plan=search_plan,
                    adapter_list=adapter_list,
                    grouped_proposals=grouped_proposals,
                    attempted_proposals=attempted_proposals,
                    attempted_by_family=attempted_by_family,
                    rejected_diagnostics=rejected_diagnostics,
                    rejected_by_family=rejected_by_family,
                    coverage_disclosures=coverage_disclosures,
                    canonical_program_count=canonical_program_count,
                    description_artifacts=tuple(all_description_artifacts),
                    admissibility_diagnostics=tuple(all_admissibility_diagnostics),
                )
            raise ContractValidationError(
                code="descriptive_fallback_bank_unavailable",
                message=(
                    "descriptive search must expose at least one compact "
                    "fallback candidate or fail as a configuration error"
                ),
                field_path="search_plan.candidate_family_ids",
                details={
                    "required_fallback_candidate_ids": list(
                        COMPACT_DESCRIPTIVE_FALLBACK_CANDIDATE_IDS
                    ),
                    "declared_candidate_family_ids": list(
                        search_plan.candidate_family_ids
                    ),
                },
            )
        fallback_attempt_proposals = _unattempted_compact_fallback_proposals(
            attempted_proposals=attempted_proposals,
            fallback_proposals=fallback_proposals,
        )
        if search_plan.search_class == "exact_finite_enumeration":
            fallback_attempt_proposals = _ordered_exact_enumeration_proposals(
                fallback_attempt_proposals
            )
        canonical_program_count += len(fallback_attempt_proposals)
        fallback_accounted_by_family = _proposal_count_by_family(
            adapter_families=tuple(grouped_proposals),
            proposals=fallback_attempt_proposals,
        )
        if (
            search_plan.search_class == "exact_finite_enumeration"
            and search_plan.proposal_limit < canonical_program_count
        ):
            raise ContractValidationError(
                code="exact_search_budget_too_small",
                message=(
                    "proposal_limit must cover the full declared canonical "
                    "program space for exact_finite_enumeration"
                ),
                field_path="search_plan.proposal_limit",
                details={
                    "proposal_limit": search_plan.proposal_limit,
                    "canonical_program_count": canonical_program_count,
                },
            )
        coverage_disclosures = _coverage_disclosures(
            search_plan=search_plan,
            canonical_program_count=canonical_program_count,
            restart_count_used=attempt_selection.restart_count_used,
        )
        fallback_candidates: list[CandidateIntermediateRepresentation] = []
        fallback_seen_hashes = set(candidate_by_hash)
        for fallback_rank, proposal in enumerate(
            fallback_attempt_proposals,
            start=len(attempted_proposals),
        ):
            adapter = adapter_by_family[proposal.primitive_family]
            fallback_attempted_candidate_count += 1
            fallback_attempted_by_family[proposal.primitive_family] += 1
            try:
                candidate = adapter.realize_proposal(
                    proposal=proposal,
                    proposal_rank=fallback_rank,
                    search_plan=search_plan,
                    feature_view=legal_feature_view,
                    observation_model=bound_observation_model,
                    coverage_disclosures=coverage_disclosures,
                )
            except _SearchRealizationError as exc:
                rejected = RejectedSearchDiagnostic(
                    candidate_id=proposal.candidate_id,
                    primitive_family=proposal.primitive_family,
                    reason_code=exc.reason_code,
                    details={**exc.details, "fallback_bank": True},
                )
                rejected_diagnostics.append(rejected)
                rejected_by_family[proposal.primitive_family].append(rejected)
                continue
            candidate_hash = candidate.canonical_hash()
            if candidate_hash in fallback_seen_hashes:
                continue
            fallback_seen_hashes.add(candidate_hash)
            fallback_candidates.append(candidate)
        fallback_descriptive_scope_candidates, excluded_fallback_candidates = (
            _partition_descriptive_scope_candidates(
                fallback_candidates,
                fallback_bank=True,
            )
        )
        for rejected in excluded_fallback_candidates:
            rejected_diagnostics.append(rejected)
            rejected_by_family[rejected.primitive_family].append(rejected)

        fallback_result = evaluate_descriptive_candidates(
            tuple(fallback_descriptive_scope_candidates),
            feature_view=legal_feature_view,
            minimum_description_gain_bits=search_plan.minimum_description_gain_bits,
        )
        all_description_artifacts.extend(fallback_result.description_artifacts)
        all_admissibility_diagnostics.extend(
            fallback_result.admissibility_diagnostics
        )
        fallback_candidate_by_hash = {
            candidate.canonical_hash(): candidate
            for candidate in fallback_descriptive_scope_candidates
        }
        for artifact in fallback_result.description_artifacts:
            candidate = fallback_candidate_by_hash.get(artifact.candidate_hash)
            if candidate is None:
                continue
            rankable_candidates.append(candidate)
            rankable_artifacts.append(artifact)
        for diagnostic in fallback_result.admissibility_diagnostics:
            if diagnostic.is_admissible:
                continue
            rejected = RejectedSearchDiagnostic(
                candidate_id=diagnostic.candidate_id,
                primitive_family=diagnostic.primitive_family,
                reason_code="descriptive_admissibility_failed",
                details={
                    "candidate_hash": diagnostic.candidate_hash,
                    "reason_codes": list(diagnostic.reason_codes),
                    "fallback_bank": True,
                    **dict(diagnostic.details),
                },
            )
            rejected_diagnostics.append(rejected)
            rejected_by_family[diagnostic.primitive_family].append(rejected)
        if not rankable_candidates:
            raise ContractValidationError(
                code="descriptive_scope_unavailable",
                message=(
                    "valid descriptive runs must yield at least one compact "
                    "rankable candidate instead of abstaining"
                ),
                field_path="search_result.descriptive_scope",
            )
        descriptive_scope_source = "fallback_bank"

    descriptive_scope = list(
        _decorate_search_candidates(
            scope_candidates=rankable_candidates,
            search_plan=search_plan,
            feature_view=feature_view,
            coverage_disclosures=coverage_disclosures,
            attempt_selection=attempt_selection,
        )
    )
    artifact_by_hash = {
        artifact.candidate_hash: artifact for artifact in rankable_artifacts
    }
    accepted_candidate_hashes = tuple(
        candidate.canonical_hash()
        for candidate in descriptive_result.accepted_candidates
    )
    law_eligible_hashes = (
        set()
        if descriptive_scope_source == "fallback_bank"
        else set(accepted_candidate_hashes)
    )
    descriptive_scope.sort(
        key=lambda candidate: _descriptive_scope_sort_key(
            candidate,
            artifact_by_hash[candidate.canonical_hash()],
        )
    )
    annotated_descriptive_scope: list[CandidateIntermediateRepresentation] = []
    for scope_rank, candidate in enumerate(descriptive_scope, start=1):
        diagnostic = diagnostic_by_hash.get(candidate.canonical_hash())
        law_eligible = candidate.canonical_hash() in law_eligible_hashes
        law_rejection_reason_codes = (
            ()
            if law_eligible
            else (
                ("descriptive_only_fallback_bank",)
                if descriptive_scope_source == "fallback_bank"
                else (diagnostic.reason_codes if diagnostic is not None else ())
            )
        )
        annotated_descriptive_scope.append(
            _with_descriptive_scope_metadata(
                candidate,
                scope_rank=scope_rank,
                source=descriptive_scope_source,
                law_eligible=law_eligible,
                law_rejection_reason_codes=law_rejection_reason_codes,
            )
        )
    descriptive_scope = annotated_descriptive_scope
    descriptive_scope_by_hash = {
        candidate.canonical_hash(): candidate for candidate in descriptive_scope
    }
    law_eligible_scope = tuple(
        candidate
        for candidate in descriptive_scope
        if candidate.canonical_hash() in law_eligible_hashes
    )
    accepted_candidates = _order_accepted_candidates_for_output(
        tuple(
            descriptive_scope_by_hash[candidate_hash]
            for candidate_hash in accepted_candidate_hashes
            if candidate_hash in descriptive_scope_by_hash
        ),
        search_plan=search_plan,
    )
    best_overall_candidate = descriptive_scope[0] if descriptive_scope else None
    accepted_candidate = law_eligible_scope[0] if law_eligible_scope else None
    gap_report_reason_codes = (
        ()
        if accepted_candidate is not None or best_overall_candidate is None
        else tuple(
            best_overall_candidate.evidence_layer.transient_diagnostics[
                "descriptive_scope"
            ]["law_rejection_reason_codes"]
        )
    )
    accepted_by_family: dict[str, list[CandidateIntermediateRepresentation]] = {
        adapter.family_id: [] for adapter in adapter_list
    }
    descriptive_scope_by_family: dict[
        str, list[CandidateIntermediateRepresentation]
    ] = {adapter.family_id: [] for adapter in adapter_list}
    law_eligible_by_family: dict[str, list[CandidateIntermediateRepresentation]] = {
        adapter.family_id: [] for adapter in adapter_list
    }
    for candidate in descriptive_scope:
        descriptive_scope_by_family[candidate.structural_layer.cir_family_id].append(
            candidate
        )
    for candidate in accepted_candidates:
        accepted_by_family[candidate.structural_layer.cir_family_id].append(candidate)
    for candidate in law_eligible_scope:
        law_eligible_by_family[candidate.structural_layer.cir_family_id].append(
            candidate
        )

    family_results = tuple(
        FamilySearchBackendResult(
            family_id=adapter.family_id,
            adapter_id=adapter.adapter_id,
            accepted_candidates=tuple(accepted_by_family[adapter.family_id]),
            rejected_diagnostics=tuple(rejected_by_family[adapter.family_id]),
            coverage=SearchCoverageAccounting(
                search_class=search_plan.search_class,
                canonical_program_count=(
                    len(grouped_proposals[adapter.family_id])
                    + fallback_accounted_by_family[adapter.family_id]
                ),
                attempted_candidate_count=(
                    attempted_by_family[adapter.family_id]
                    + fallback_attempted_by_family[adapter.family_id]
                ),
                accepted_candidate_count=(
                    1 if law_eligible_by_family[adapter.family_id] else 0
                ),
                rejected_candidate_count=len(rejected_by_family[adapter.family_id]),
                omitted_candidate_count=(
                    len(grouped_proposals[adapter.family_id])
                    + fallback_accounted_by_family[adapter.family_id]
                    - attempted_by_family[adapter.family_id]
                    - fallback_attempted_by_family[adapter.family_id]
                ),
                coverage_statement=_coverage_statement(search_plan.search_class),
                exactness_ceiling=_exactness_ceiling(search_plan.search_class),
                scope_declaration=_scope_declaration(search_plan.search_class),
                disclosures=_coverage_disclosures(
                    search_plan=search_plan,
                    canonical_program_count=(
                        len(grouped_proposals[adapter.family_id])
                        + fallback_accounted_by_family[adapter.family_id]
                    ),
                    restart_count_used=attempt_selection.restart_count_used,
                ),
                ranked_candidate_count=len(
                    descriptive_scope_by_family[adapter.family_id]
                ),
                law_eligible_candidate_count=len(
                    law_eligible_by_family[adapter.family_id]
                ),
                diagnostic_only_candidate_count=len(
                    rejected_by_family[adapter.family_id]
                ),
                fallback_candidate_count=(
                    len(descriptive_scope_by_family[adapter.family_id])
                    if descriptive_scope_source == "fallback_bank"
                    else 0
                ),
            ),
            descriptive_scope=tuple(descriptive_scope_by_family[adapter.family_id]),
            law_eligible_scope=tuple(law_eligible_by_family[adapter.family_id]),
            best_overall_candidate=(
                descriptive_scope_by_family[adapter.family_id][0]
                if descriptive_scope_by_family[adapter.family_id]
                else None
            ),
            accepted_candidate=(
                law_eligible_by_family[adapter.family_id][0]
                if law_eligible_by_family[adapter.family_id]
                else None
            ),
        )
        for adapter in adapter_list
    )
    coverage = SearchCoverageAccounting(
        search_class=search_plan.search_class,
        canonical_program_count=canonical_program_count,
        attempted_candidate_count=(
            len(attempted_proposals) + fallback_attempted_candidate_count
        ),
        accepted_candidate_count=1 if accepted_candidate is not None else 0,
        rejected_candidate_count=len(rejected_diagnostics),
        omitted_candidate_count=(
            canonical_program_count
            - len(attempted_proposals)
            - fallback_attempted_candidate_count
        ),
        coverage_statement=_coverage_statement(search_plan.search_class),
        exactness_ceiling=_exactness_ceiling(search_plan.search_class),
        scope_declaration=_scope_declaration(search_plan.search_class),
        disclosures=coverage_disclosures,
        ranked_candidate_count=len(descriptive_scope),
        law_eligible_candidate_count=len(law_eligible_scope),
        diagnostic_only_candidate_count=len(
            {
                (diagnostic.primitive_family, diagnostic.candidate_id)
                for diagnostic in rejected_diagnostics
            }
        ),
        fallback_candidate_count=(
            len(descriptive_scope) if descriptive_scope_source == "fallback_bank" else 0
        ),
        gap_report_reason_codes=gap_report_reason_codes,
    )
    frontier = _build_descriptive_frontier(
        search_plan=search_plan,
        accepted_candidates=descriptive_scope,
        description_artifacts=tuple(all_description_artifacts),
    )
    return DescriptiveSearchRunResult(
        accepted_candidates=tuple(accepted_candidates),
        description_artifacts=tuple(all_description_artifacts),
        admissibility_diagnostics=tuple(all_admissibility_diagnostics),
        rejected_diagnostics=tuple(rejected_diagnostics),
        family_results=family_results,
        coverage=coverage,
        frontier=frontier,
        descriptive_scope=tuple(descriptive_scope),
        law_eligible_scope=law_eligible_scope,
        best_overall_candidate=best_overall_candidate,
        accepted_candidate=accepted_candidate,
        gap_report_reason_codes=gap_report_reason_codes,
    )


def _order_accepted_candidates_for_output(
    candidates: Sequence[CandidateIntermediateRepresentation],
    *,
    search_plan: SearchPlanManifest,
) -> tuple[CandidateIntermediateRepresentation, ...]:
    if (
        search_plan.search_class != "exact_finite_enumeration"
        or not search_plan.candidate_family_ids
    ):
        return tuple(candidates)
    declaration_rank = {
        candidate_id: index
        for index, candidate_id in enumerate(search_plan.candidate_family_ids)
    }
    return tuple(
        candidate
        for _, candidate in sorted(
            enumerate(candidates),
            key=lambda item: (
                declaration_rank.get(
                    _candidate_source_id(item[1]),
                    len(declaration_rank) + item[0],
                ),
                item[1].evidence_layer.backend_origin_record.proposal_rank,
                _candidate_source_id(item[1]),
            ),
        )
    )


def _diagnostic_only_search_result(
    *,
    search_plan: SearchPlanManifest,
    adapter_list: Sequence[DescriptiveSearchBackendAdapter],
    grouped_proposals: Mapping[str, Sequence[DescriptiveSearchProposal]],
    attempted_proposals: Sequence[DescriptiveSearchProposal],
    attempted_by_family: Mapping[str, int],
    rejected_diagnostics: Sequence[RejectedSearchDiagnostic],
    rejected_by_family: Mapping[str, Sequence[RejectedSearchDiagnostic]],
    coverage_disclosures: Mapping[str, Any],
    canonical_program_count: int,
    description_artifacts: Sequence[DescriptionGainArtifact],
    admissibility_diagnostics: Sequence[DescriptiveAdmissibilityDiagnostic],
) -> DescriptiveSearchRunResult:
    family_results = tuple(
        FamilySearchBackendResult(
            family_id=adapter.family_id,
            adapter_id=adapter.adapter_id,
            accepted_candidates=(),
            rejected_diagnostics=tuple(rejected_by_family.get(adapter.family_id, ())),
            coverage=SearchCoverageAccounting(
                search_class=search_plan.search_class,
                canonical_program_count=len(
                    grouped_proposals.get(adapter.family_id, ())
                ),
                attempted_candidate_count=attempted_by_family.get(adapter.family_id, 0),
                accepted_candidate_count=0,
                rejected_candidate_count=len(
                    rejected_by_family.get(adapter.family_id, ())
                ),
                omitted_candidate_count=max(
                    len(grouped_proposals.get(adapter.family_id, ()))
                    - attempted_by_family.get(adapter.family_id, 0),
                    0,
                ),
                coverage_statement=_coverage_statement(search_plan.search_class),
                exactness_ceiling=_exactness_ceiling(search_plan.search_class),
                scope_declaration=_scope_declaration(search_plan.search_class),
                disclosures=_coverage_disclosures(
                    search_plan=search_plan,
                    canonical_program_count=len(
                        grouped_proposals.get(adapter.family_id, ())
                    ),
                    restart_count_used=0,
                ),
                diagnostic_only_candidate_count=len(
                    {
                        (diagnostic.primitive_family, diagnostic.candidate_id)
                        for diagnostic in rejected_by_family.get(adapter.family_id, ())
                    }
                ),
            ),
        )
        for adapter in adapter_list
    )
    coverage = SearchCoverageAccounting(
        search_class=search_plan.search_class,
        canonical_program_count=canonical_program_count,
        attempted_candidate_count=len(attempted_proposals),
        accepted_candidate_count=0,
        rejected_candidate_count=len(rejected_diagnostics),
        omitted_candidate_count=max(
            canonical_program_count - len(attempted_proposals),
            0,
        ),
        coverage_statement=_coverage_statement(search_plan.search_class),
        exactness_ceiling=_exactness_ceiling(search_plan.search_class),
        scope_declaration=_scope_declaration(search_plan.search_class),
        disclosures=coverage_disclosures,
        diagnostic_only_candidate_count=len(
            {
                (diagnostic.primitive_family, diagnostic.candidate_id)
                for diagnostic in rejected_diagnostics
            }
        ),
    )
    return DescriptiveSearchRunResult(
        accepted_candidates=(),
        description_artifacts=tuple(description_artifacts),
        admissibility_diagnostics=tuple(admissibility_diagnostics),
        rejected_diagnostics=tuple(rejected_diagnostics),
        family_results=family_results,
        coverage=coverage,
        frontier=_build_descriptive_frontier(
            search_plan=search_plan,
            accepted_candidates=(),
            description_artifacts=(),
        ),
        descriptive_scope=(),
        law_eligible_scope=(),
        best_overall_candidate=None,
        accepted_candidate=None,
        gap_report_reason_codes=(),
    )


@dataclass(frozen=True)
class _SearchRealizationError(Exception):
    reason_code: str
    details: Mapping[str, Any]


def _default_adapters() -> tuple[DescriptiveSearchBackendAdapter, ...]:
    return (
        AnalyticSearchBackendAdapter(),
        RecursiveSearchBackendAdapter(),
        SpectralSearchBackendAdapter(),
        AlgorithmicSearchBackendAdapter(),
    )


def _search_class_semantics(search_class: str) -> _SearchClassSemantics:
    try:
        return _SEARCH_CLASS_SEMANTICS[search_class]
    except KeyError as exc:
        raise ContractValidationError(
            code="unsupported_search_class",
            message=f"unsupported search_class: {search_class}",
            field_path="search_class",
        ) from exc


def _coverage_statement(search_class: str) -> str:
    return _search_class_semantics(search_class).coverage_statement


def _exactness_ceiling(search_class: str) -> str:
    return _search_class_semantics(search_class).exactness_ceiling


def _scope_declaration(search_class: str) -> str:
    return _search_class_semantics(search_class).scope_declaration


def _coverage_disclosures(
    *,
    search_plan: SearchPlanManifest,
    canonical_program_count: int,
    restart_count_used: int,
) -> Mapping[str, Any]:
    if search_plan.search_class == "exact_finite_enumeration":
        stop_rule = "exhaust_declared_canonical_program_space"
        if canonical_program_count > 0:
            stop_rule += f"(cardinality={canonical_program_count})"
        return {
            "fragment_bounds": f"proposal_limit={search_plan.proposal_limit}",
            "canonical_enumerator": "declared_adapter_family_order_then_candidate_id",
            "enumeration_cardinality": canonical_program_count,
            "stop_rule": stop_rule,
        }
    if search_plan.search_class == "bounded_heuristic":
        return {
            "proposer_mechanism": (
                "declared_adapter_default_or_user_supplied_proposals"
            ),
            "pruning_rules": (
                "proposal_limit_prefix_then_canonical_duplicate_screen"
                "_then_descriptive_admissibility"
            ),
            "stop_rule": f"proposal_limit={search_plan.proposal_limit}",
        }
    if search_plan.search_class == "equality_saturation_heuristic":
        return {
            "rewrite_system": "egraph_engine_required_for_expression_cir_rewrites",
            "extractor_cost": "declared_by_egraph_engine_rewrite_trace",
            "legacy_fragment_backend_mode": "no_sort_only_equality_saturation",
            "stop_rule": f"proposal_limit={search_plan.proposal_limit}",
        }
    return {
        "proposal_distribution": "seeded_sha256_permutation_without_replacement",
        "seed_policy": (
            f"root_seed={search_plan.random_seed};"
            f"derivation={search_plan.seed_derivation_rule}"
        ),
        "restart_policy": (
            f"batch_size={max(search_plan.parallel_candidate_batch_size, 1)};"
            "restart_until_budget_or_exhaustion;"
            f"restarts_used={restart_count_used}"
        ),
        "stop_rule": f"proposal_limit={search_plan.proposal_limit}",
    }


def _decorate_search_candidates(
    *,
    scope_candidates: Sequence[CandidateIntermediateRepresentation],
    search_plan: SearchPlanManifest,
    feature_view: FeatureView,
    coverage_disclosures: Mapping[str, Any],
    attempt_selection: _AttemptSelection,
) -> tuple[CandidateIntermediateRepresentation, ...]:
    if not scope_candidates:
        return ()
    inner_primary_scores = _search_inner_primary_scores(
        accepted_candidates=scope_candidates,
        search_plan=search_plan,
        feature_view=feature_view,
    )
    decorated = tuple(
        _with_search_honesty_evidence(
            candidate,
            search_plan=search_plan,
            coverage_disclosures=coverage_disclosures,
            attempt_selection=attempt_selection,
            inner_primary_score=inner_primary_scores.get(candidate.canonical_hash()),
        )
        for candidate in scope_candidates
    )
    _validate_search_honesty_evidence(
        search_plan=search_plan,
        coverage_disclosures=coverage_disclosures,
        accepted_candidates=decorated,
    )
    return decorated


def _search_inner_primary_scores(
    *,
    accepted_candidates: Sequence[CandidateIntermediateRepresentation],
    search_plan: SearchPlanManifest,
    feature_view: FeatureView,
) -> Mapping[str, float]:
    if "inner_primary_score" not in search_plan.frontier_axes:
        return {}
    legal_feature_view = feature_view.require_stage_reuse("candidate_fitting")
    fit_window = _search_fit_window(legal_feature_view)
    scores: dict[str, float] = {}
    for candidate in accepted_candidates:
        fit_result = fit_candidate_window(
            candidate=candidate,
            feature_view=legal_feature_view,
            fit_window=fit_window,
            search_plan=search_plan,
            stage_id="inner_search",
        )
        training_row_count = max(int(fit_result.training_row_count), 1)
        scores[candidate.canonical_hash()] = _stable_float(
            float(fit_result.optimizer_diagnostics["final_loss"]) / training_row_count
        )
    return scores


def _search_fit_window(feature_view: FeatureView) -> EvaluationSegment:
    if not feature_view.rows:
        raise ContractValidationError(
            code="empty_search_feature_view",
            message=(
                "search-time frontier scoring requires at least one feature row"
            ),
            field_path="feature_view.rows",
        )
    last_row = feature_view.rows[-1]
    row_count = len(feature_view.rows)
    last_event_time = str(last_row["event_time"])
    return EvaluationSegment(
        segment_id="search_frontier_fold_local",
        outer_fold_id="search_frontier_fold_local",
        role="development",
        train_start_index=0,
        train_end_index=row_count - 1,
        test_start_index=row_count - 1,
        test_end_index=row_count - 1,
        train_row_count=row_count,
        test_row_count=1,
        origin_index=row_count - 1,
        origin_time=last_event_time,
        train_end_event_time=last_event_time,
        test_end_event_time=last_event_time,
        horizon_set=(1,),
        scored_origin_ids=(),
        entity_panel=feature_view.entity_panel,
    )


def _with_search_honesty_evidence(
    candidate: CandidateIntermediateRepresentation,
    *,
    search_plan: SearchPlanManifest,
    coverage_disclosures: Mapping[str, Any],
    attempt_selection: _AttemptSelection,
    inner_primary_score: float | None,
) -> CandidateIntermediateRepresentation:
    backend_origin = candidate.evidence_layer.backend_origin_record
    private_fields = tuple(
        dict.fromkeys((*backend_origin.backend_private_fields, "search_evidence"))
    )
    transient_diagnostics: dict[str, Any] = {
        "search_evidence": _candidate_search_evidence(
            candidate_id=backend_origin.source_candidate_id,
            search_plan=search_plan,
            coverage_disclosures=coverage_disclosures,
            attempt_selection=attempt_selection,
        )
    }
    if inner_primary_score is not None:
        transient_diagnostics["inner_primary_score"] = _stable_float(
            inner_primary_score
        )
    return rebind_cir_backend_origin(
        candidate,
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id=backend_origin.adapter_id,
            adapter_class=backend_origin.adapter_class,
            source_candidate_id=backend_origin.source_candidate_id,
            search_class=backend_origin.search_class,
            backend_family=backend_origin.backend_family,
            proposal_rank=backend_origin.proposal_rank,
            normalization_scope=backend_origin.normalization_scope,
            comparability_scope=backend_origin.comparability_scope,
            backend_private_fields=private_fields,
        ),
        transient_diagnostics=transient_diagnostics,
    )


def _validate_search_honesty_evidence(
    *,
    search_plan: SearchPlanManifest,
    coverage_disclosures: Mapping[str, Any],
    accepted_candidates: Sequence[CandidateIntermediateRepresentation],
) -> None:
    required_disclosures = REQUIRED_SEARCH_COVERAGE_DISCLOSURES.get(
        search_plan.search_class,
        (),
    )
    missing_disclosures = tuple(
        disclosure
        for disclosure in required_disclosures
        if disclosure not in coverage_disclosures
    )
    if missing_disclosures:
        raise ContractValidationError(
            code="search_honesty_evidence_missing",
            message=(
                "search_class is missing required backend coverage disclosures"
            ),
            field_path="coverage.disclosures",
            details={
                "search_class": search_plan.search_class,
                "missing_disclosures": list(missing_disclosures),
            },
        )
    required_candidate_keys = REQUIRED_CANDIDATE_SEARCH_EVIDENCE.get(
        search_plan.search_class,
        (),
    )
    for candidate in accepted_candidates:
        search_evidence = candidate.evidence_layer.transient_diagnostics.get(
            "search_evidence"
        )
        if not isinstance(search_evidence, Mapping):
            raise ContractValidationError(
                code="search_honesty_evidence_missing",
                message="accepted candidates must carry declared search evidence",
                field_path="candidate.evidence_layer.transient_diagnostics",
                details={
                    "candidate_id": (
                        candidate.evidence_layer.backend_origin_record.source_candidate_id
                    ),
                    "search_class": search_plan.search_class,
                },
            )
        missing_keys = tuple(
            key for key in required_candidate_keys if key not in search_evidence
        )
        if missing_keys:
            raise ContractValidationError(
                code="search_honesty_evidence_missing",
                message=(
                    "accepted candidate is missing search evidence required by "
                    "the declared search class"
                ),
                field_path="candidate.evidence_layer.transient_diagnostics.search_evidence",
                details={
                    "candidate_id": (
                        candidate.evidence_layer.backend_origin_record.source_candidate_id
                    ),
                    "search_class": search_plan.search_class,
                    "missing_keys": list(missing_keys),
                },
            )


def _build_replay_hooks(
    *,
    search_plan: SearchPlanManifest,
    coverage_disclosures: Mapping[str, Any],
    extra_hooks: Sequence[CIRReplayHook] = (),
) -> CIRReplayHooks:
    hooks = [
        CIRReplayHook(
            hook_name="budget_record",
            hook_ref=f"budget:{search_plan.search_plan_id}:{search_plan.proposal_limit}",
        ),
        CIRReplayHook(
            hook_name="search_seed",
            hook_ref=f"seed:{search_plan.random_seed}",
        ),
        CIRReplayHook(
            hook_name="search_scope",
            hook_ref=_scope_declaration(search_plan.search_class),
        ),
    ]
    if search_plan.search_class == "bounded_heuristic":
        hooks.extend(
            (
                CIRReplayHook(
                    hook_name="search_proposer_mechanism",
                    hook_ref=str(coverage_disclosures["proposer_mechanism"]),
                ),
                CIRReplayHook(
                    hook_name="search_pruning_rules",
                    hook_ref=str(coverage_disclosures["pruning_rules"]),
                ),
                CIRReplayHook(
                    hook_name="search_stop_rule",
                    hook_ref=str(coverage_disclosures["stop_rule"]),
                ),
            )
        )
    elif search_plan.search_class == "equality_saturation_heuristic":
        hooks.extend(
            (
                CIRReplayHook(
                    hook_name="rewrite_system",
                    hook_ref=str(coverage_disclosures["rewrite_system"]),
                ),
                CIRReplayHook(
                    hook_name="extractor_cost",
                    hook_ref=str(coverage_disclosures["extractor_cost"]),
                ),
                CIRReplayHook(
                    hook_name="search_stop_rule",
                    hook_ref=str(coverage_disclosures["stop_rule"]),
                ),
            )
        )
    elif search_plan.search_class == "stochastic_heuristic":
        hooks.extend(
            (
                CIRReplayHook(
                    hook_name="proposal_distribution",
                    hook_ref=str(coverage_disclosures["proposal_distribution"]),
                ),
                CIRReplayHook(
                    hook_name="restart_policy",
                    hook_ref=str(coverage_disclosures["restart_policy"]),
                ),
                CIRReplayHook(
                    hook_name="search_stop_rule",
                    hook_ref=str(coverage_disclosures["stop_rule"]),
                ),
            )
        )
    else:
        hooks.append(
            CIRReplayHook(
                hook_name="search_stop_rule",
                hook_ref=str(coverage_disclosures["stop_rule"]),
            )
        )
    hooks.extend(extra_hooks)
    return CIRReplayHooks(hooks=tuple(hooks))


def _select_attempted_proposals(
    *,
    ordered_proposals: Sequence[DescriptiveSearchProposal],
    search_plan: SearchPlanManifest,
) -> _AttemptSelection:
    canonical_program_count = len(ordered_proposals)
    if search_plan.search_class == "exact_finite_enumeration":
        return _AttemptSelection(
            attempted_proposals=_ordered_exact_enumeration_proposals(
                ordered_proposals
            )
        )

    attempt_limit = min(search_plan.proposal_limit, canonical_program_count)
    if search_plan.search_class == "bounded_heuristic":
        return _AttemptSelection(
            attempted_proposals=tuple(ordered_proposals[:attempt_limit])
        )
    if search_plan.search_class == "equality_saturation_heuristic":
        ranked = tuple(ordered_proposals)
        return _AttemptSelection(
            attempted_proposals=tuple(ranked[:attempt_limit]),
            rewrite_space_candidate_ids=tuple(
                proposal.candidate_id for proposal in ranked
            ),
        )
    return _select_stochastic_attempts(
        ordered_proposals=ordered_proposals,
        search_plan=search_plan,
        attempt_limit=attempt_limit,
    )


def _select_stochastic_attempts(
    *,
    ordered_proposals: Sequence[DescriptiveSearchProposal],
    search_plan: SearchPlanManifest,
    attempt_limit: int,
) -> _AttemptSelection:
    if attempt_limit <= 0:
        return _AttemptSelection(attempted_proposals=(), restart_count_used=0)
    remaining = list(ordered_proposals)
    selected: list[DescriptiveSearchProposal] = []
    batch_size = max(1, min(search_plan.parallel_candidate_batch_size, attempt_limit))
    restart_count = 0
    restart_records: list[Mapping[str, Any]] = []
    while remaining and len(selected) < attempt_limit:
        ranked = sorted(
            remaining,
            key=lambda proposal: (
                _seeded_order_token(
                    root_seed=search_plan.random_seed,
                    restart_index=restart_count,
                    candidate_id=proposal.candidate_id,
                ),
                proposal.candidate_id,
            ),
        )
        take_count = min(batch_size, attempt_limit - len(selected))
        chosen = ranked[:take_count]
        selected.extend(chosen)
        restart_records.append(
            {
                "restart_index": restart_count,
                "batch_size": batch_size,
                "candidate_ids": [proposal.candidate_id for proposal in chosen],
            }
        )
        chosen_ids = {proposal.candidate_id for proposal in chosen}
        remaining = [
            proposal
            for proposal in remaining
            if proposal.candidate_id not in chosen_ids
        ]
        restart_count += 1
    return _AttemptSelection(
        attempted_proposals=tuple(selected),
        restart_count_used=restart_count,
        restart_records=tuple(restart_records),
        declared_stochastic_surfaces=(
            "proposal_distribution",
            "candidate_attempt_order",
            "restart_records",
        ),
    )


def _candidate_search_evidence(
    *,
    candidate_id: str,
    search_plan: SearchPlanManifest,
    coverage_disclosures: Mapping[str, Any],
    attempt_selection: _AttemptSelection,
) -> dict[str, Any]:
    search_evidence: dict[str, Any] = {
        "search_class": search_plan.search_class,
        "exactness_scope": _scope_declaration(search_plan.search_class),
        **dict(coverage_disclosures),
    }
    if search_plan.search_class == "equality_saturation_heuristic":
        search_evidence.update(
            {
                "rewrite_space_candidate_ids": list(
                    attempt_selection.rewrite_space_candidate_ids
                ),
                "selected_candidate_id": candidate_id,
            }
        )
    elif search_plan.search_class == "stochastic_heuristic":
        search_evidence.update(
            {
                "seed_scopes": list(search_plan.seed_scopes),
                "restart_records": [
                    dict(record) for record in attempt_selection.restart_records
                ],
                "declared_stochastic_surfaces": list(
                    attempt_selection.declared_stochastic_surfaces
                ),
            }
        )
    return search_evidence


def _seeded_order_token(
    *,
    root_seed: str,
    restart_index: int,
    candidate_id: str,
) -> str:
    payload = f"{root_seed}:{restart_index}:{candidate_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_descriptive_frontier(
    *,
    search_plan: SearchPlanManifest,
    accepted_candidates: Sequence[CandidateIntermediateRepresentation],
    description_artifacts: Sequence[DescriptionGainArtifact],
) -> DescriptiveSearchFrontierResult:
    artifact_by_hash = {
        artifact.candidate_hash: artifact for artifact in description_artifacts
    }
    candidate_by_hash: dict[str, CandidateIntermediateRepresentation] = {}
    metrics: list[FrontierCandidateMetrics] = []

    for candidate in accepted_candidates:
        candidate_hash = candidate.canonical_hash()
        artifact = artifact_by_hash.get(candidate_hash)
        if artifact is None:
            raise ContractValidationError(
                code="missing_frontier_description_artifact",
                message=(
                    "accepted descriptive candidates must retain matching "
                    "description-gain artifacts before frontier construction"
                ),
                field_path="description_artifacts",
                details={"candidate_hash": candidate_hash},
            )
        candidate_id = (
            candidate.evidence_layer.backend_origin_record.source_candidate_id
        )
        axis_values: dict[str, float] = {
            "structure_code_bits": artifact.L_structure_bits,
            "description_gain_bits": artifact.description_gain_bits,
        }
        inner_primary_score = candidate.evidence_layer.transient_diagnostics.get(
            "inner_primary_score"
        )
        if isinstance(inner_primary_score, (int, float)) and math.isfinite(
            float(inner_primary_score)
        ):
            axis_values["inner_primary_score"] = float(inner_primary_score)
        metrics.append(
            FrontierCandidateMetrics(
                candidate_id=candidate_id,
                primitive_family=candidate.structural_layer.cir_family_id,
                candidate_hash=candidate_hash,
                total_code_bits=artifact.L_total_bits,
                structure_code_bits=artifact.L_structure_bits,
                description_gain_bits=artifact.description_gain_bits,
                axis_values=axis_values,
            )
        )
        candidate_by_hash[candidate_hash] = candidate

    state = construct_stage_local_frontier(
        candidate_metrics=tuple(metrics),
        requested_axes=search_plan.frontier_axes,
        frontier_width=search_plan.frontier_width,
        shortlist_limit=search_plan.shortlist_limit,
        search_class=search_plan.search_class,
    )
    return DescriptiveSearchFrontierResult(
        state=state,
        retained_frontier_cir_candidates=tuple(
            candidate_by_hash[record.candidate_hash]
            for record in state.retained_frontier_candidates
        ),
        frozen_shortlist_cir_candidates=tuple(
            candidate_by_hash[record.candidate_hash]
            for record in state.frozen_shortlist_candidates
        ),
    )


def _model_code_decomposition(
    proposal: DescriptiveSearchProposal,
) -> CIRModelCodeDecomposition:
    return CIRModelCodeDecomposition(
        L_family_bits=2.0,
        L_structure_bits=1.0 + float(bool(proposal.feature_dependencies)),
        L_literals_bits=float(len(proposal.literal_values)),
        L_params_bits=float(len(proposal.parameter_values)),
        L_state_bits=float(len(proposal.persistent_state)),
    )


def _identity_update(
    state: ReducerStateObject,
    context: ReducerStateUpdateContext,
) -> ReducerStateObject:
    return state


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


def _target_values(feature_view: FeatureView) -> tuple[float, ...]:
    return tuple(float(row["target"]) for row in feature_view.rows)


def _max_target(feature_view: FeatureView) -> float:
    return max(_target_values(feature_view))


def _min_target(feature_view: FeatureView) -> float:
    return min(_target_values(feature_view))


__all__ = [
    "AnalyticSearchBackendAdapter",
    "AlgorithmicSearchBackendAdapter",
    "DescriptiveSearchProposal",
    "DescriptiveSearchFrontierResult",
    "DescriptiveSearchRunResult",
    "FamilySearchBackendResult",
    "DescriptionGainArtifact",
    "DescriptiveAdmissibilityDiagnostic",
    "RecursiveSearchBackendAdapter",
    "RejectedSearchDiagnostic",
    "SearchCoverageAccounting",
    "SpectralSearchBackendAdapter",
    "run_descriptive_search_backends",
]
