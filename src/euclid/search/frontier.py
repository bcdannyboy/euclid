from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from euclid.contracts.errors import ContractValidationError
from euclid.search.policies import FULL_COMPARABILITY_SEARCH_CLASSES

_MINIMIZE_AXES = frozenset({"structure_code_bits", "inner_primary_score"})
_MAXIMIZE_AXES = frozenset({"description_gain_bits"})
_SUPPORTED_AXES = _MINIMIZE_AXES | _MAXIMIZE_AXES
_DOMINANCE_RULE = "weakly_better_all_axes_strictly_better_one"


@dataclass(frozen=True)
class FrontierCandidateMetrics:
    candidate_id: str
    primitive_family: str
    candidate_hash: str
    total_code_bits: float
    structure_code_bits: float
    description_gain_bits: float
    axis_values: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ContractValidationError(
                code="invalid_frontier_candidate",
                message="candidate_id must be non-empty",
                field_path="candidate_id",
            )
        if not self.primitive_family:
            raise ContractValidationError(
                code="invalid_frontier_candidate",
                message="primitive_family must be non-empty",
                field_path="primitive_family",
            )
        if not self.candidate_hash:
            raise ContractValidationError(
                code="invalid_frontier_candidate",
                message="candidate_hash must be non-empty",
                field_path="candidate_hash",
            )
        for field_name, value in (
            ("total_code_bits", self.total_code_bits),
            ("structure_code_bits", self.structure_code_bits),
            ("description_gain_bits", self.description_gain_bits),
        ):
            _require_finite(value, field_path=field_name)
        unsupported_axes = sorted(set(self.axis_values) - _SUPPORTED_AXES)
        if unsupported_axes:
            raise ContractValidationError(
                code="unsupported_frontier_axis",
                message="axis_values contain unsupported frontier axes",
                field_path="axis_values",
                details={"unsupported_axes": unsupported_axes},
            )
        for axis_name, value in self.axis_values.items():
            _require_finite(value, field_path=f"axis_values[{axis_name}]")


@dataclass(frozen=True)
class FrontierDominanceRecord:
    candidate_id: str
    primitive_family: str
    candidate_hash: str
    dominated_by_candidate_ids: tuple[str, ...]
    frontier_retained: bool
    retained_after_width: bool
    frozen_for_evaluation: bool


@dataclass(frozen=True)
class StageLocalFrontierCoverage:
    search_class: str
    requested_axes: tuple[str, ...]
    comparable_axes: tuple[str, ...]
    incomparable_axes: tuple[str, ...]
    frontier_width: int
    shortlist_limit: int
    candidate_count_considered: int
    frontier_candidate_count: int
    retained_frontier_candidate_count: int
    dominated_candidate_count: int
    omitted_by_frontier_width_count: int
    frozen_shortlist_candidate_count: int
    dominance_rule: str = _DOMINANCE_RULE


@dataclass(frozen=True)
class StageLocalFrontierResult:
    frontier_candidates: tuple[FrontierCandidateMetrics, ...]
    retained_frontier_candidates: tuple[FrontierCandidateMetrics, ...]
    frozen_shortlist_candidates: tuple[FrontierCandidateMetrics, ...]
    dominance_records: tuple[FrontierDominanceRecord, ...]
    coverage: StageLocalFrontierCoverage


def construct_stage_local_frontier(
    *,
    candidate_metrics: Sequence[FrontierCandidateMetrics],
    requested_axes: Sequence[str],
    frontier_width: int,
    shortlist_limit: int,
    search_class: str = "bounded_heuristic",
) -> StageLocalFrontierResult:
    requested = tuple(_validate_requested_axes(requested_axes))
    if frontier_width < 1:
        raise ContractValidationError(
            code="invalid_frontier_width",
            message="frontier_width must be positive",
            field_path="frontier_width",
        )
    if shortlist_limit < 1:
        raise ContractValidationError(
            code="invalid_shortlist_limit",
            message="shortlist_limit must be positive",
            field_path="shortlist_limit",
        )
    if not candidate_metrics:
        return StageLocalFrontierResult(
            frontier_candidates=(),
            retained_frontier_candidates=(),
            frozen_shortlist_candidates=(),
            dominance_records=(),
            coverage=StageLocalFrontierCoverage(
                search_class=search_class,
                requested_axes=requested,
                comparable_axes=requested,
                incomparable_axes=(),
                frontier_width=frontier_width,
                shortlist_limit=shortlist_limit,
                candidate_count_considered=0,
                frontier_candidate_count=0,
                retained_frontier_candidate_count=0,
                dominated_candidate_count=0,
                omitted_by_frontier_width_count=0,
                frozen_shortlist_candidate_count=0,
            ),
        )

    comparable_axes = tuple(
        axis
        for axis in requested
        if all(_candidate_has_axis(candidate, axis) for candidate in candidate_metrics)
    )
    incomparable_axes = tuple(
        axis for axis in requested if axis not in set(comparable_axes)
    )
    if (
        search_class in FULL_COMPARABILITY_SEARCH_CLASSES
        and incomparable_axes
        and candidate_metrics
    ):
        raise ContractValidationError(
            code="incomplete_frontier_evidence",
            message=(
                "search_class requires comparable values for every declared "
                "frontier axis before publication"
            ),
            field_path="requested_axes",
            details={
                "search_class": search_class,
                "incomparable_axes": list(incomparable_axes),
            },
        )
    if not comparable_axes:
        raise ContractValidationError(
            code="no_comparable_frontier_axes",
            message="stage-local frontier requires at least one comparable axis",
            field_path="requested_axes",
            details={"requested_axes": list(requested)},
        )

    dominated_by: dict[str, tuple[str, ...]] = {}
    frontier_candidates = []
    for candidate in candidate_metrics:
        dominators = tuple(
            competing.candidate_id
            for competing in candidate_metrics
            if competing.candidate_hash != candidate.candidate_hash
            and _dominates(competing, candidate, comparable_axes=comparable_axes)
        )
        dominated_by[candidate.candidate_hash] = dominators
        if not dominators:
            frontier_candidates.append(candidate)

    sorted_frontier = tuple(sorted(frontier_candidates, key=_freeze_sort_key))
    retained_frontier = sorted_frontier[:frontier_width]
    frozen_shortlist = retained_frontier[:shortlist_limit]
    retained_hashes = {candidate.candidate_hash for candidate in retained_frontier}
    frozen_hashes = {candidate.candidate_hash for candidate in frozen_shortlist}
    frontier_hashes = {candidate.candidate_hash for candidate in sorted_frontier}

    dominance_records = tuple(
        FrontierDominanceRecord(
            candidate_id=candidate.candidate_id,
            primitive_family=candidate.primitive_family,
            candidate_hash=candidate.candidate_hash,
            dominated_by_candidate_ids=dominated_by[candidate.candidate_hash],
            frontier_retained=candidate.candidate_hash in frontier_hashes,
            retained_after_width=candidate.candidate_hash in retained_hashes,
            frozen_for_evaluation=candidate.candidate_hash in frozen_hashes,
        )
        for candidate in sorted(candidate_metrics, key=lambda item: item.candidate_id)
    )

    coverage = StageLocalFrontierCoverage(
        search_class=search_class,
        requested_axes=requested,
        comparable_axes=comparable_axes,
        incomparable_axes=incomparable_axes,
        frontier_width=frontier_width,
        shortlist_limit=shortlist_limit,
        candidate_count_considered=len(candidate_metrics),
        frontier_candidate_count=len(sorted_frontier),
        retained_frontier_candidate_count=len(retained_frontier),
        dominated_candidate_count=len(candidate_metrics) - len(sorted_frontier),
        omitted_by_frontier_width_count=max(
            len(sorted_frontier) - len(retained_frontier),
            0,
        ),
        frozen_shortlist_candidate_count=len(frozen_shortlist),
    )
    return StageLocalFrontierResult(
        frontier_candidates=sorted_frontier,
        retained_frontier_candidates=retained_frontier,
        frozen_shortlist_candidates=frozen_shortlist,
        dominance_records=dominance_records,
        coverage=coverage,
    )


def _validate_requested_axes(requested_axes: Sequence[str]) -> tuple[str, ...]:
    if not requested_axes:
        raise ContractValidationError(
            code="empty_frontier_axis_set",
            message="requested_axes must declare at least one frontier axis",
            field_path="requested_axes",
        )
    validated: list[str] = []
    seen: set[str] = set()
    for index, axis in enumerate(requested_axes):
        if axis not in _SUPPORTED_AXES:
            raise ContractValidationError(
                code="unsupported_frontier_axis",
                message="requested frontier axis is not supported",
                field_path=f"requested_axes[{index}]",
                details={"axis": axis},
            )
        if axis in seen:
            raise ContractValidationError(
                code="duplicate_frontier_axis",
                message="requested frontier axes must be unique",
                field_path=f"requested_axes[{index}]",
                details={"axis": axis},
            )
        seen.add(axis)
        validated.append(axis)
    return tuple(validated)


def _candidate_has_axis(candidate: FrontierCandidateMetrics, axis: str) -> bool:
    value = candidate.axis_values.get(axis)
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _dominates(
    left: FrontierCandidateMetrics,
    right: FrontierCandidateMetrics,
    *,
    comparable_axes: Sequence[str],
) -> bool:
    weakly_better_all_axes = all(
        _axis_weakly_better(
            axis=axis,
            left_value=float(left.axis_values[axis]),
            right_value=float(right.axis_values[axis]),
        )
        for axis in comparable_axes
    )
    strictly_better_one_axis = any(
        _axis_strictly_better(
            axis=axis,
            left_value=float(left.axis_values[axis]),
            right_value=float(right.axis_values[axis]),
        )
        for axis in comparable_axes
    )
    return weakly_better_all_axes and strictly_better_one_axis


def _axis_weakly_better(*, axis: str, left_value: float, right_value: float) -> bool:
    if axis in _MINIMIZE_AXES:
        return left_value <= right_value
    return left_value >= right_value


def _axis_strictly_better(*, axis: str, left_value: float, right_value: float) -> bool:
    if axis in _MINIMIZE_AXES:
        return left_value < right_value
    return left_value > right_value


def _freeze_sort_key(
    candidate: FrontierCandidateMetrics,
) -> tuple[float, float, float, str]:
    return (
        float(candidate.total_code_bits),
        -float(candidate.description_gain_bits),
        float(candidate.structure_code_bits),
        candidate.candidate_id,
    )


def _require_finite(value: float, *, field_path: str) -> None:
    if not math.isfinite(float(value)):
        raise ContractValidationError(
            code="invalid_frontier_candidate",
            message=f"{field_path} must be finite",
            field_path=field_path,
        )


__all__ = [
    "FrontierCandidateMetrics",
    "FrontierDominanceRecord",
    "StageLocalFrontierCoverage",
    "StageLocalFrontierResult",
    "construct_stage_local_frontier",
]
