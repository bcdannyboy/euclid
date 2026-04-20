from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.search.frontier import (
    FrontierCandidateMetrics,
    construct_stage_local_frontier,
)


def test_construct_stage_local_frontier_tracks_dominance_and_incomparable_axes(
) -> None:
    result = construct_stage_local_frontier(
        candidate_metrics=(
            FrontierCandidateMetrics(
                candidate_id="analytic_intercept",
                primitive_family="analytic",
                candidate_hash="hash-a",
                total_code_bits=39.0,
                structure_code_bits=1.0,
                description_gain_bits=18.0,
                axis_values={
                    "structure_code_bits": 1.0,
                    "description_gain_bits": 18.0,
                },
            ),
            FrontierCandidateMetrics(
                candidate_id="analytic_lag1_affine",
                primitive_family="analytic",
                candidate_hash="hash-b",
                total_code_bits=31.0,
                structure_code_bits=2.0,
                description_gain_bits=26.0,
                axis_values={
                    "structure_code_bits": 2.0,
                    "description_gain_bits": 26.0,
                },
            ),
            FrontierCandidateMetrics(
                candidate_id="recursive_level_smoother",
                primitive_family="recursive",
                candidate_hash="hash-c",
                total_code_bits=41.0,
                structure_code_bits=1.0,
                description_gain_bits=16.0,
                axis_values={
                    "structure_code_bits": 1.0,
                    "description_gain_bits": 16.0,
                },
            ),
        ),
        requested_axes=(
            "structure_code_bits",
            "description_gain_bits",
            "inner_primary_score",
        ),
        frontier_width=4,
        shortlist_limit=1,
        search_class="equality_saturation_heuristic",
    )

    assert result.coverage.requested_axes == (
        "structure_code_bits",
        "description_gain_bits",
        "inner_primary_score",
    )
    assert result.coverage.comparable_axes == (
        "structure_code_bits",
        "description_gain_bits",
    )
    assert result.coverage.incomparable_axes == ("inner_primary_score",)
    assert [record.candidate_id for record in result.frontier_candidates] == [
        "analytic_lag1_affine",
        "analytic_intercept",
    ]
    assert [record.candidate_id for record in result.retained_frontier_candidates] == [
        "analytic_lag1_affine",
        "analytic_intercept",
    ]
    assert [record.candidate_id for record in result.frozen_shortlist_candidates] == [
        "analytic_lag1_affine",
    ]

    dominance = {
        record.candidate_id: record.dominated_by_candidate_ids
        for record in result.dominance_records
    }
    assert dominance["analytic_intercept"] == ()
    assert dominance["analytic_lag1_affine"] == ()
    assert dominance["recursive_level_smoother"] == ("analytic_intercept",)


def test_construct_stage_local_frontier_applies_lexicographic_freeze_tie_break(
) -> None:
    result = construct_stage_local_frontier(
        candidate_metrics=(
            FrontierCandidateMetrics(
                candidate_id="candidate_alpha",
                primitive_family="analytic",
                candidate_hash="hash-alpha",
                total_code_bits=20.0,
                structure_code_bits=1.0,
                description_gain_bits=9.0,
                axis_values={
                    "structure_code_bits": 1.0,
                    "description_gain_bits": 9.0,
                },
            ),
            FrontierCandidateMetrics(
                candidate_id="candidate_beta",
                primitive_family="recursive",
                candidate_hash="hash-beta",
                total_code_bits=20.0,
                structure_code_bits=1.0,
                description_gain_bits=9.0,
                axis_values={
                    "structure_code_bits": 1.0,
                    "description_gain_bits": 9.0,
                },
            ),
        ),
        requested_axes=("structure_code_bits", "description_gain_bits"),
        frontier_width=2,
        shortlist_limit=1,
    )

    assert [record.candidate_id for record in result.frontier_candidates] == [
        "candidate_alpha",
        "candidate_beta",
    ]
    assert [record.candidate_id for record in result.frozen_shortlist_candidates] == [
        "candidate_alpha",
    ]


def test_construct_stage_local_frontier_rejects_empty_comparable_axes() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        construct_stage_local_frontier(
            candidate_metrics=(
                FrontierCandidateMetrics(
                    candidate_id="candidate_alpha",
                    primitive_family="analytic",
                    candidate_hash="hash-alpha",
                    total_code_bits=20.0,
                    structure_code_bits=1.0,
                    description_gain_bits=9.0,
                    axis_values={},
                ),
            ),
            requested_axes=("inner_primary_score",),
            frontier_width=1,
            shortlist_limit=1,
            search_class="equality_saturation_heuristic",
        )

    assert exc_info.value.code == "no_comparable_frontier_axes"


def test_bounded_search_frontier_requires_full_axis_evidence() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        construct_stage_local_frontier(
            candidate_metrics=(
                FrontierCandidateMetrics(
                    candidate_id="candidate_alpha",
                    primitive_family="analytic",
                    candidate_hash="hash-alpha",
                    total_code_bits=20.0,
                    structure_code_bits=1.0,
                    description_gain_bits=9.0,
                    axis_values={
                        "structure_code_bits": 1.0,
                        "description_gain_bits": 9.0,
                    },
                ),
            ),
            requested_axes=(
                "structure_code_bits",
                "description_gain_bits",
                "inner_primary_score",
            ),
            frontier_width=1,
            shortlist_limit=1,
            search_class="bounded_heuristic",
        )

    assert exc_info.value.code == "incomplete_frontier_evidence"
