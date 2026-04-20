from __future__ import annotations

from euclid.search.dsl.enumerator import (
    AlgorithmicEnumerationBounds,
    enumerate_algorithmic_programs,
)
from euclid.search.dsl.parser import parse_algorithmic_program


def test_parser_accepts_v1_fragment() -> None:
    program = parse_algorithmic_program(
        (
            "(program "
            "(state (lit 0) (lit 1)) "
            "(next "
            "(if (not false) (abs (obs 1)) (lit 0)) "
            "(div (add (state 1) (obs 0)) (lit 2))) "
            "(emit (state 1)))"
        ),
        state_slot_count=2,
        max_program_nodes=14,
        allowed_observation_lags=(0, 1),
    )

    assert program.state_slot_count == 2
    assert program.allowed_observation_lags == (0, 1)
    assert program.canonical_source == (
        "(program (state (lit 0) (lit 1)) "
        "(next (if (not false) (abs (obs 1)) (lit 0)) "
        "(div (add (state 1) (obs 0)) (lit 2))) "
        "(emit (state 1)))"
    )


def test_enumerator_counts_bounded_fragment_space() -> None:
    programs = enumerate_algorithmic_programs(
        AlgorithmicEnumerationBounds(
            state_slot_count=1,
            max_program_nodes=6,
            allowed_observation_lags=(0,),
            literal_palette=(0, 1),
            unary_real_ops=("abs",),
            binary_real_ops=("div",),
            comparison_ops=("lt",),
            boolean_literals=("false",),
            logical_ops=("not",),
            include_conditional_templates=True,
        )
    )
    sources = {program.canonical_source for program in programs}

    assert len(programs) == 40
    assert (
        "(program (state (lit 0)) (next (abs (lit 0))) (emit (state 0)))"
    ) in sources
    assert (
        "(program (state (lit 0)) "
        "(next (div (obs 0) (lit 1))) "
        "(emit (state 0)))"
    ) in sources
