from __future__ import annotations

from euclid.adapters.algorithmic_dsl import enumerate_algorithmic_proposal_specs
from euclid.search.dsl.enumerator import (
    AlgorithmicEnumerationBounds,
    enumerate_algorithmic_programs,
)


def test_algorithmic_enumerator_respects_small_node_budget() -> None:
    programs = enumerate_algorithmic_programs(
        AlgorithmicEnumerationBounds(max_program_nodes=3)
    )

    assert tuple(program.canonical_source for program in programs) == (
        "(program (state (lit 0)) (next (lit 0)) (emit (state 0)))",
        "(program (state (lit 0)) (next (lit 1)) (emit (state 0)))",
        "(program (state (lit 0)) (next (lit 2)) (emit (state 0)))",
        "(program (state (lit 0)) (next (obs 0)) (emit (state 0)))",
        "(program (state (lit 0)) (next (state 0)) (emit (state 0)))",
    )
    assert all(program.node_count <= 3 for program in programs)


def test_algorithmic_enumerator_is_deterministic_and_preserves_aliases() -> None:
    first = enumerate_algorithmic_proposal_specs()
    second = enumerate_algorithmic_proposal_specs()

    assert tuple(spec.candidate_id for spec in first) == tuple(
        spec.candidate_id for spec in second
    )
    assert len(first) > 2
    assert len({spec.candidate_id for spec in first}) == len(first)

    by_id = {spec.candidate_id: spec for spec in first}
    assert by_id["algorithmic_last_observation"].program.canonical_source == (
        "(program (state (lit 0)) (next (obs 0)) (emit (state 0)))"
    )
    assert by_id["algorithmic_running_half_average"].program.canonical_source == (
        "(program (state (lit 0)) "
        "(next (div (add (state 0) (obs 0)) (lit 2))) "
        "(emit (state 0)))"
    )


def test_algorithmic_enumerator_supports_explicit_bounded_v1_machine_readable_limits(
) -> None:
    multi_state_programs = enumerate_algorithmic_programs(
        AlgorithmicEnumerationBounds(
            state_slot_count=2,
            max_program_nodes=6,
            allowed_observation_lags=(0, 1),
            literal_palette=(0, 1),
            emit_state_index=1,
        )
    )
    conditional_programs = enumerate_algorithmic_programs(
        AlgorithmicEnumerationBounds(
            state_slot_count=1,
            max_program_nodes=8,
            allowed_observation_lags=(0,),
            literal_palette=(-1, 0, 2),
            unary_real_ops=("abs",),
            binary_real_ops=("div",),
            boolean_literals=("false",),
            logical_ops=("not",),
            include_conditional_templates=True,
        )
    )

    multi_state_sources = {
        program.canonical_source for program in multi_state_programs
    }
    conditional_sources = {
        program.canonical_source for program in conditional_programs
    }

    assert (
        "(program (state (lit 0) (lit 0)) "
        "(next (obs 1) (state 1)) "
        "(emit (state 1)))"
    ) in multi_state_sources
    assert (
        "(program (state (lit 0)) "
        "(next (if (not false) (abs (lit -1)) (lit 0))) "
        "(emit (state 0)))"
    ) in conditional_sources
    assert (
        "(program (state (lit 0)) "
        "(next (div (obs 0) (lit 2))) "
        "(emit (state 0)))"
    ) in conditional_sources
