from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.manifests.runtime_models import (
    CanonicalizationPolicyManifest,
    FreezeEventManifest,
    FrontierManifest,
    FrozenShortlistManifest,
    RejectedDiagnosticsManifest,
    SearchLedgerManifest,
    SearchPlanManifest,
)
from euclid.modules.split_planning import EvaluationPlan
from euclid.search.policies import (
    DESCRIPTIVE_SCOPE_SELECTION_RULE,
    DEFAULT_FORBIDDEN_FRONTIER_AXES,
    DEFAULT_FRONTIER_AXES,
    DEFAULT_SEARCH_CLASS,
    RETAINED_COMPOSITION_OPERATORS,
    RETAINED_PRIMITIVE_FAMILIES,
    SUPPORTED_FRONTIER_AXES,
    FrontierPolicy,
    ParallelBudgetPolicy,
    SearchBudgetPolicy,
    SeedPolicy,
)

_ADMITTED_SEARCH_CLASSES = (
    "exact_finite_enumeration",
    "bounded_heuristic",
    "equality_saturation_heuristic",
    "stochastic_heuristic",
)
_NON_POINT_ADMITTED_SEARCH_CLASSES = (
    "exact_finite_enumeration",
    "bounded_heuristic",
)


@dataclass(frozen=True)
class SearchCandidateRecord:
    candidate_id: str
    family_id: str
    structure_code_bits: float
    description_gain_bits: float
    inner_primary_score: float
    admissible: bool
    total_code_bits: float | None = None
    canonical_byte_length: int = 0
    ranked: bool | None = None
    law_eligible: bool | None = None
    rejection_reason_codes: tuple[str, ...] = ()
    law_rejection_reason_codes: tuple[str, ...] = ()

    @property
    def ranked_for_descriptive_scope(self) -> bool:
        return self.ranked is True

    @property
    def law_rejection_codes(self) -> tuple[str, ...]:
        if self.law_rejection_reason_codes:
            return self.law_rejection_reason_codes
        if self.law_eligible_for_claims:
            return ()
        return self.rejection_reason_codes

    @property
    def law_eligible_for_claims(self) -> bool:
        return self.law_eligible is True

    def as_ledger_dict(
        self,
        *,
        descriptive_scope_rank: int | None = None,
        law_eligible_scope_rank: int | None = None,
    ) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "structure_code_bits": self.structure_code_bits,
            "total_code_bits": self.total_code_bits,
            "description_gain_bits": self.description_gain_bits,
            "inner_primary_score": self.inner_primary_score,
            "canonical_byte_length": self.canonical_byte_length,
            "admissible": self.admissible,
            "ranked": self.ranked_for_descriptive_scope,
            "descriptive_scope_rank": descriptive_scope_rank,
            "law_eligible": self.law_eligible_for_claims,
            "law_eligible_scope_rank": law_eligible_scope_rank,
            "law_rejection_reason_codes": list(self.law_rejection_codes),
        }

    def as_frontier_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "structure_code_bits": self.structure_code_bits,
            "total_code_bits": self.total_code_bits,
            "description_gain_bits": self.description_gain_bits,
            "inner_primary_score": self.inner_primary_score,
            "ranked": self.ranked_for_descriptive_scope,
            "law_eligible": self.law_eligible_for_claims,
            "law_rejection_reason_codes": list(self.law_rejection_codes),
        }

    def as_rejected_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "rejection_reason_codes": list(self.rejection_reason_codes),
            "law_rejection_reason_codes": list(self.law_rejection_codes),
            "law_eligible": self.law_eligible_for_claims,
        }


def build_canonicalization_policy(
    *,
    canonicalization_policy_id: str = "prototype_search_canonicalization_v1",
) -> CanonicalizationPolicyManifest:
    return CanonicalizationPolicyManifest(
        canonicalization_policy_id=canonicalization_policy_id
    )


def build_search_plan(
    *,
    evaluation_plan: EvaluationPlan,
    canonicalization_policy_ref: TypedRef,
    codelength_policy_ref: TypedRef,
    reference_description_policy_ref: TypedRef,
    observation_model_ref: TypedRef,
    candidate_family_ids: tuple[str, ...],
    predictive_mode: str = "predictive_requested",
    search_plan_id: str = "prototype_retained_scope_search_plan_v1",
    search_class: str = DEFAULT_SEARCH_CLASS,
    frontier_axes: tuple[str, ...] = DEFAULT_FRONTIER_AXES,
    forbidden_frontier_axes: tuple[str, ...] = DEFAULT_FORBIDDEN_FRONTIER_AXES,
    random_seed: str = "0",
    proposal_limit: int | None = None,
    frontier_width: int | None = None,
    shortlist_limit: int = 1,
    wall_clock_budget_seconds: int = 1,
    parallel_worker_count: int = 1,
    candidate_batch_size: int = 1,
    minimum_description_gain_bits: float | None = None,
    seasonal_period: int | None = None,
    fit_strategy: Mapping[str, Any] | None = None,
    quantization_policy: Mapping[str, Any] | None = None,
    reference_policy: Mapping[str, Any] | None = None,
    data_code_family: str | None = None,
    parent_refs: tuple[TypedRef, ...] = (),
) -> SearchPlanManifest:
    if not candidate_family_ids:
        raise ContractValidationError(
            code="empty_search_family_vocabulary",
            message="candidate_family_ids must declare at least one retained family",
            field_path="candidate_family_ids",
        )
    if predictive_mode == "descriptive_only":
        predictive_mode = "descriptive_structure"
    if predictive_mode not in {"descriptive_structure", "predictive_requested"}:
        raise ContractValidationError(
            code="invalid_predictive_mode",
            message=(
                "predictive_mode must be descriptive_structure or "
                "predictive_requested"
            ),
            field_path="predictive_mode",
        )
    _validate_object_type_search_class_pair(
        forecast_object_type=evaluation_plan.forecast_object_type,
        search_class=search_class,
    )
    _validate_frontier_axes(
        frontier_axes=frontier_axes,
        forbidden_frontier_axes=forbidden_frontier_axes,
    )
    if (
        "inner_primary_score" in frontier_axes
        and predictive_mode != "predictive_requested"
    ):
        raise ContractValidationError(
            code="predictive_axis_requires_predictive_mode",
            message=(
                "inner_primary_score is legal only when predictive "
                "search is requested"
            ),
            field_path="frontier_axes",
        )

    budget_policy = SearchBudgetPolicy(
        proposal_limit=proposal_limit or len(candidate_family_ids),
        frontier_width=frontier_width or len(candidate_family_ids),
        shortlist_limit=shortlist_limit,
        wall_clock_budget_seconds=wall_clock_budget_seconds,
    )
    parallel_budget = ParallelBudgetPolicy(
        max_worker_count=parallel_worker_count,
        candidate_batch_size=candidate_batch_size,
    )
    seed_policy = SeedPolicy(root_seed=random_seed)
    frontier_policy = FrontierPolicy(
        axes=frontier_axes,
        forbidden_axes=forbidden_frontier_axes,
    )
    _validate_u64_decimal(seed_policy.root_seed)

    return SearchPlanManifest(
        parent_refs=parent_refs,
        search_plan_id=search_plan_id,
        search_class=search_class,
        canonicalization_policy_ref=canonicalization_policy_ref,
        codelength_policy_ref=codelength_policy_ref,
        reference_description_policy_ref=reference_description_policy_ref,
        observation_model_ref=observation_model_ref,
        predictive_mode=predictive_mode,
        forecast_object_type=evaluation_plan.forecast_object_type,
        primitive_families=RETAINED_PRIMITIVE_FAMILIES,
        composition_operators=(
            tuple(RETAINED_COMPOSITION_OPERATORS)
            + (
                ("shared_plus_local_decomposition",)
                if len(evaluation_plan.entity_panel) > 1
                else ()
            )
        ),
        candidate_family_ids=candidate_family_ids,
        fold_local_search_required=True,
        max_candidate_count=budget_policy.proposal_limit,
        random_seed=seed_policy.root_seed,
        proposal_limit=budget_policy.proposal_limit,
        frontier_width=budget_policy.frontier_width,
        shortlist_limit=budget_policy.shortlist_limit,
        wall_clock_budget_seconds=budget_policy.wall_clock_budget_seconds,
        budget_accounting_rule=budget_policy.budget_accounting_rule,
        parallel_max_worker_count=parallel_budget.max_worker_count,
        parallel_candidate_batch_size=parallel_budget.candidate_batch_size,
        parallel_aggregation_rule=parallel_budget.aggregation_rule,
        seed_derivation_rule=seed_policy.seed_derivation_rule,
        seed_scopes=seed_policy.seed_scopes,
        frontier_id=frontier_policy.frontier_id,
        frontier_axes=frontier_policy.axes,
        predictive_axis_rule=frontier_policy.predictive_axis_rule,
        forbidden_frontier_axes=frontier_policy.forbidden_axes,
        search_time_predictive_policy="fold_local_only",
        minimum_description_gain_bits=minimum_description_gain_bits,
        seasonal_period=seasonal_period,
        fit_strategy=dict(fit_strategy) if fit_strategy is not None else None,
        quantization_policy=(
            dict(quantization_policy) if quantization_policy is not None else None
        ),
        reference_policy=dict(reference_policy) if reference_policy is not None else None,
        data_code_family=data_code_family,
    )


def build_search_ledger(
    *,
    search_plan: SearchPlanManifest,
    candidate_records: tuple[SearchCandidateRecord, ...],
    selected_candidate_id: str,
    accepted_candidate_id: str | None = None,
    search_ledger_id: str = "prototype_search_ledger_v1",
    parent_refs: tuple[TypedRef, ...] = (),
) -> SearchLedgerManifest:
    attempted_candidate_count = len(candidate_records)
    if attempted_candidate_count > search_plan.proposal_limit:
        raise ContractValidationError(
            code="proposal_budget_exhausted",
            message="candidate_records exceeds the frozen proposal_limit",
            field_path="candidate_records",
            details={
                "attempted_candidate_count": attempted_candidate_count,
                "proposal_limit": search_plan.proposal_limit,
            },
        )
    _validate_ranked_candidate_metrics(candidate_records)
    descriptive_scope_records = _ordered_descriptive_scope_records(candidate_records)
    if not descriptive_scope_records:
        raise ContractValidationError(
            code="missing_descriptive_scope_candidate",
            message=(
                "search ledgers must carry at least one ranked descriptive-scope "
                "candidate"
            ),
            field_path="candidate_records",
        )
    law_eligible_scope_records = tuple(
        record
        for record in descriptive_scope_records
        if record.law_eligible_for_claims
    )
    best_overall_candidate_id = descriptive_scope_records[0].candidate_id
    accepted_scope_candidate_id = (
        law_eligible_scope_records[0].candidate_id
        if law_eligible_scope_records
        else None
    )
    _validate_selection_scope_match(
        field_path="selected_candidate_id",
        provided_candidate_id=selected_candidate_id,
        expected_candidate_id=best_overall_candidate_id,
        scope_name="descriptive_scope",
    )
    if accepted_candidate_id is not None:
        _validate_selection_scope_match(
            field_path="accepted_candidate_id",
            provided_candidate_id=accepted_candidate_id,
            expected_candidate_id=accepted_scope_candidate_id,
            scope_name="law_eligible_scope",
        )

    descriptive_scope_ranks = {
        record.candidate_id: rank
        for rank, record in enumerate(descriptive_scope_records, start=1)
    }
    law_eligible_scope_ranks = {
        record.candidate_id: rank
        for rank, record in enumerate(law_eligible_scope_records, start=1)
    }
    ranked_candidate_count = sum(
        record.ranked_for_descriptive_scope for record in candidate_records
    )
    admissible_candidate_count = sum(record.admissible for record in candidate_records)
    law_eligible_candidate_count = len(law_eligible_scope_records)
    accepted_candidate_count = 1 if accepted_scope_candidate_id is not None else 0
    rejected_candidate_count = sum(
        _is_rejected_record(record) for record in candidate_records
    )
    remaining_budget_count = search_plan.proposal_limit - attempted_candidate_count
    return SearchLedgerManifest(
        parent_refs=parent_refs,
        search_ledger_id=search_ledger_id,
        selected_candidate_id=best_overall_candidate_id,
        selection_rule=DESCRIPTIVE_SCOPE_SELECTION_RULE,
        candidates=tuple(
            record.as_ledger_dict(
                descriptive_scope_rank=descriptive_scope_ranks.get(
                    record.candidate_id
                ),
                law_eligible_scope_rank=law_eligible_scope_ranks.get(
                    record.candidate_id
                ),
            )
            for record in candidate_records
        ),
        budget_accounting={
            "proposal_limit": search_plan.proposal_limit,
            "attempted_candidate_count": attempted_candidate_count,
            "ranked_candidate_count": ranked_candidate_count,
            "admissible_candidate_count": admissible_candidate_count,
            "law_eligible_candidate_count": law_eligible_candidate_count,
            "accepted_candidate_count": accepted_candidate_count,
            "rejected_candidate_count": rejected_candidate_count,
            "remaining_budget_count": remaining_budget_count,
            "best_overall_candidate_id": best_overall_candidate_id,
            "accepted_candidate_id": accepted_scope_candidate_id,
            "selected_candidate_scope": "descriptive_scope",
            "accepted_candidate_scope": "law_eligible_scope",
            "descriptive_scope_candidate_ids": [
                record.candidate_id for record in descriptive_scope_records
            ],
            "law_eligible_scope_candidate_ids": [
                record.candidate_id for record in law_eligible_scope_records
            ],
            "accounting_status": "within_budget",
            "stop_reason": (
                "proposal_limit_reached"
                if remaining_budget_count == 0
                else "candidate_list_exhausted"
            ),
        },
    )


def build_frontier(
    *,
    search_plan: SearchPlanManifest,
    candidate_records: tuple[SearchCandidateRecord, ...],
    frontier_id: str = "prototype_search_frontier_v1",
    parent_refs: tuple[TypedRef, ...] = (),
) -> FrontierManifest:
    ranked_records = tuple(
        record
        for record in candidate_records
        if record.ranked_for_descriptive_scope
    )
    frontier_records = tuple(
        record
        for record in ranked_records
        if _is_frontier_member(
            record,
            ranked_records,
            axes=search_plan.frontier_axes,
        )
    )
    return FrontierManifest(
        parent_refs=parent_refs,
        frontier_id=frontier_id,
        frontier_axes=search_plan.frontier_axes,
        frontier_candidate_ids=tuple(
            record.candidate_id for record in frontier_records
        ),
        frontier_records=tuple(
            record.as_frontier_dict() for record in frontier_records
        ),
    )


def build_rejected_diagnostics(
    *,
    candidate_records: tuple[SearchCandidateRecord, ...],
    rejected_diagnostics_id: str = "prototype_rejected_diagnostics_v1",
    parent_refs: tuple[TypedRef, ...] = (),
) -> RejectedDiagnosticsManifest:
    rejected_records = tuple(
        record.as_rejected_dict()
        for record in candidate_records
        if _is_rejected_record(record)
    )
    return RejectedDiagnosticsManifest(
        parent_refs=parent_refs,
        rejected_diagnostics_id=rejected_diagnostics_id,
        rejected_records=rejected_records,
    )


def build_frozen_shortlist(
    *,
    search_plan_ref: TypedRef,
    candidate_ref: TypedRef,
    frozen_shortlist_id: str = "prototype_frozen_shortlist_v1",
    parent_refs: tuple[TypedRef, ...] = (),
) -> FrozenShortlistManifest:
    return FrozenShortlistManifest(
        parent_refs=parent_refs,
        frozen_shortlist_id=frozen_shortlist_id,
        search_plan_ref=search_plan_ref,
        candidate_refs=(candidate_ref,),
    )


def build_freeze_event(
    *,
    frozen_candidate_ref: TypedRef,
    frozen_shortlist_ref: TypedRef,
    confirmatory_baseline_id: str,
    freeze_event_id: str = "prototype_freeze_event_v1",
    parent_refs: tuple[TypedRef, ...] = (),
) -> FreezeEventManifest:
    return FreezeEventManifest(
        parent_refs=parent_refs,
        freeze_event_id=freeze_event_id,
        frozen_candidate_ref=frozen_candidate_ref,
        frozen_shortlist_ref=frozen_shortlist_ref,
        confirmatory_baseline_id=confirmatory_baseline_id,
    )


def _validate_frontier_axes(
    *,
    frontier_axes: Iterable[str],
    forbidden_frontier_axes: Iterable[str],
) -> None:
    forbidden = set(forbidden_frontier_axes)
    for index, axis in enumerate(frontier_axes):
        if axis in forbidden:
            raise ContractValidationError(
                code="forbidden_frontier_axis",
                message=(
                    "holdout and downstream evidence may not appear on the "
                    "search frontier"
                ),
                field_path=f"frontier_axes[{index}]",
                details={"axis": axis},
            )
        if axis not in SUPPORTED_FRONTIER_AXES:
            raise ContractValidationError(
                code="unsupported_frontier_axis",
                message=(
                    "frontier_axes must be drawn from the search-stage "
                    "candidate metrics"
                ),
                field_path=f"frontier_axes[{index}]",
                details={"axis": axis},
            )


def _validate_u64_decimal(value: str) -> None:
    if not value.isdigit():
        raise ContractValidationError(
            code="invalid_search_seed",
            message="root seed must be an unsigned 64-bit decimal string",
            field_path="random_seed",
        )
    parsed = int(value)
    if parsed < 0 or parsed > (2**64 - 1):
        raise ContractValidationError(
            code="invalid_search_seed",
            message="root seed must fit in an unsigned 64-bit integer",
            field_path="random_seed",
        )


def _validate_object_type_search_class_pair(
    *,
    forecast_object_type: str,
    search_class: str,
) -> None:
    if search_class not in _ADMITTED_SEARCH_CLASSES:
        raise ContractValidationError(
            code="unsupported_search_class",
            message=(
                "search_class must be one of "
                f"{', '.join(_ADMITTED_SEARCH_CLASSES)}"
            ),
            field_path="search_class",
            details={"search_class": search_class},
        )
    if (
        forecast_object_type != "point"
        and search_class not in _NON_POINT_ADMITTED_SEARCH_CLASSES
    ):
        raise ContractValidationError(
            code="illegal_object_type_search_class_pair",
            message=(
                "non-point forecast objects require a search_class with a sealed "
                "runtime contract; use exact_finite_enumeration or "
                "bounded_heuristic until other classes are bound"
            ),
            field_path="search_class",
            details={
                "forecast_object_type": forecast_object_type,
                "search_class": search_class,
            },
        )


def _is_frontier_member(
    candidate: SearchCandidateRecord,
    candidates: tuple[SearchCandidateRecord, ...],
    *,
    axes: tuple[str, ...],
) -> bool:
    for competing in candidates:
        if competing.candidate_id == candidate.candidate_id:
            continue
        if _dominates(competing, candidate, axes=axes):
            return False
    return True


def _dominates(
    left: SearchCandidateRecord,
    right: SearchCandidateRecord,
    *,
    axes: tuple[str, ...],
) -> bool:
    if not axes:
        return False
    weakly_better_all_axes = True
    strictly_better_one_axis = False
    for axis in axes:
        left_value = _frontier_axis_value(left, axis)
        right_value = _frontier_axis_value(right, axis)
        prefers_smaller = _frontier_axis_prefers_smaller(axis)
        if prefers_smaller:
            weakly_better = left_value <= right_value
            strictly_better = left_value < right_value
        else:
            weakly_better = left_value >= right_value
            strictly_better = left_value > right_value
        weakly_better_all_axes = weakly_better_all_axes and weakly_better
        strictly_better_one_axis = strictly_better_one_axis or strictly_better
        if not weakly_better_all_axes:
            return False
    return weakly_better_all_axes and strictly_better_one_axis


def _ordered_descriptive_scope_records(
    candidate_records: tuple[SearchCandidateRecord, ...],
) -> tuple[SearchCandidateRecord, ...]:
    return tuple(
        sorted(
            (
                record
                for record in candidate_records
                if record.ranked_for_descriptive_scope
            ),
            key=_descriptive_scope_sort_key,
        )
    )


def _descriptive_scope_sort_key(
    record: SearchCandidateRecord,
) -> tuple[float, float, int, str]:
    return (
        _descriptive_total_code_bits(record),
        -float(record.description_gain_bits),
        float(record.structure_code_bits),
        int(record.canonical_byte_length),
        record.candidate_id,
    )


def _descriptive_total_code_bits(record: SearchCandidateRecord) -> float:
    if record.total_code_bits is None:
        raise ContractValidationError(
            code="missing_total_code_bits",
            message=(
                "ranked descriptive-scope candidates must carry total_code_bits "
                "for auditable canonical ranking"
            ),
            field_path="candidate_records",
            details={"candidate_id": record.candidate_id},
        )
    return float(record.total_code_bits)


def _validate_ranked_candidate_metrics(
    candidate_records: tuple[SearchCandidateRecord, ...],
) -> None:
    for index, record in enumerate(candidate_records):
        if not record.ranked_for_descriptive_scope:
            continue
        if record.total_code_bits is None:
            raise ContractValidationError(
                code="missing_total_code_bits",
                message=(
                    "ranked descriptive-scope candidates must carry "
                    "total_code_bits for auditable canonical ranking"
                ),
                field_path=f"candidate_records[{index}].total_code_bits",
                details={"candidate_id": record.candidate_id},
            )
        _require_finite_ranked_metric(
            candidate_id=record.candidate_id,
            metric_name="total_code_bits",
            metric_value=record.total_code_bits,
            field_path=f"candidate_records[{index}].total_code_bits",
        )
        _require_finite_ranked_metric(
            candidate_id=record.candidate_id,
            metric_name="description_gain_bits",
            metric_value=record.description_gain_bits,
            field_path=f"candidate_records[{index}].description_gain_bits",
        )
        _require_finite_ranked_metric(
            candidate_id=record.candidate_id,
            metric_name="structure_code_bits",
            metric_value=record.structure_code_bits,
            field_path=f"candidate_records[{index}].structure_code_bits",
        )
        if int(record.canonical_byte_length) < 1:
            raise ContractValidationError(
                code="missing_canonical_byte_length",
                message=(
                    "ranked descriptive-scope candidates must carry "
                    "canonical_byte_length for auditable canonical ranking"
                ),
                field_path=f"candidate_records[{index}].canonical_byte_length",
                details={"candidate_id": record.candidate_id},
            )


def _require_finite_ranked_metric(
    *,
    candidate_id: str,
    metric_name: str,
    metric_value: float | None,
    field_path: str,
) -> None:
    if not isinstance(metric_value, (int, float)) or not math.isfinite(metric_value):
        raise ContractValidationError(
            code="non_finite_ranked_candidate_metric",
            message=(
                "ranked descriptive-scope candidates must carry finite "
                f"{metric_name} for auditable ranking"
            ),
            field_path=field_path,
            details={
                "candidate_id": candidate_id,
                "metric_name": metric_name,
                "metric_value": metric_value,
            },
        )


def _is_rejected_record(record: SearchCandidateRecord) -> bool:
    return (
        not record.admissible
        or not record.law_eligible_for_claims
        or bool(record.rejection_reason_codes)
        or bool(record.law_rejection_codes)
    )


def _validate_selection_scope_match(
    *,
    field_path: str,
    provided_candidate_id: str,
    expected_candidate_id: str | None,
    scope_name: str,
) -> None:
    if provided_candidate_id == expected_candidate_id:
        return
    raise ContractValidationError(
        code="ledger_selection_scope_mismatch",
        message=(
            f"{field_path} must match the rank-1 candidate derived from "
            f"{scope_name}"
        ),
        field_path=field_path,
        details={
            "provided_candidate_id": provided_candidate_id,
            "expected_candidate_id": expected_candidate_id,
            "scope_name": scope_name,
        },
    )


def _frontier_axis_value(record: SearchCandidateRecord, axis: str) -> float:
    if axis == "structure_code_bits":
        return float(record.structure_code_bits)
    if axis == "description_gain_bits":
        return float(record.description_gain_bits)
    if axis == "inner_primary_score":
        return float(record.inner_primary_score)
    raise ContractValidationError(
        code="unsupported_frontier_axis",
        message=(
            "frontier_axes must be drawn from the search-stage candidate metrics"
        ),
        field_path="frontier_axes",
        details={"axis": axis},
    )


def _frontier_axis_prefers_smaller(axis: str) -> bool:
    return axis != "description_gain_bits"
