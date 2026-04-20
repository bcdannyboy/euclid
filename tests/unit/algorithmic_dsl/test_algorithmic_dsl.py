from __future__ import annotations

from fractions import Fraction

import pytest

from euclid.algorithmic_dsl import (
    AlgorithmicProgram,
    canonicalize_algorithmic_program,
    evaluate_algorithmic_program,
    parse_algorithmic_program,
)
from euclid.contracts.errors import ContractValidationError
from euclid.modules.algorithmic_dsl import (
    parse_algorithmic_program as module_parse_algorithmic_program,
)
from euclid.search.dsl import (
    canonicalize_algorithmic_program as runtime_canonicalize_algorithmic_program,
)
from euclid.search.dsl import (
    evaluate_algorithmic_program as runtime_evaluate_algorithmic_program,
)
from euclid.search.dsl import (
    parse_algorithmic_program as runtime_parse_algorithmic_program,
)


def test_parse_algorithmic_program_canonicalizes_equivalent_sources() -> None:
    left = """
    (program
      (state (lit 2/4))
      (next (obs 0))
      (emit (state 0)))
    """
    right = "(program (state (lit 1/2)) (next (obs 0)) (emit (state 0)))"

    left_program = parse_algorithmic_program(left)
    right_program = parse_algorithmic_program(right)

    assert left_program.canonical_source == right_program.canonical_source
    assert canonicalize_algorithmic_program(left) == right_program.canonical_source
    assert left_program.node_count == 3


def test_public_wrappers_are_thin_reexports_of_runtime_package() -> None:
    assert parse_algorithmic_program is runtime_parse_algorithmic_program
    assert evaluate_algorithmic_program is runtime_evaluate_algorithmic_program
    assert canonicalize_algorithmic_program is runtime_canonicalize_algorithmic_program
    assert module_parse_algorithmic_program is runtime_parse_algorithmic_program


def test_runtime_package_returns_algorithmic_program_ast() -> None:
    program = runtime_parse_algorithmic_program(
        "(program (state (lit 0)) (next (obs 0)) (emit (state 0)))"
    )

    assert isinstance(program, AlgorithmicProgram)
    assert program.canonical_source == (
        "(program (state (lit 0)) (next (obs 0)) (emit (state 0)))"
    )


def test_evaluate_algorithmic_program_uses_exact_rational_updates(
) -> None:
    program = parse_algorithmic_program(
        """
        (program
          (state (lit 1/2))
          (next (add (state 0) (obs 0)))
          (emit (state 0)))
        """
    )

    step = evaluate_algorithmic_program(
        program,
        state=(Fraction(1, 2),),
        observation=Fraction(3, 2),
    )

    assert step.emit_value == Fraction(1, 2)
    assert step.next_state == (Fraction(2, 1),)


def test_parse_algorithmic_program_rejects_forbidden_emit_observation_access() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        parse_algorithmic_program(
            "(program (state (lit 0)) (next (obs 0)) (emit (obs 0)))"
        )

    assert exc_info.value.code == "forbidden_construct"


def test_parse_algorithmic_program_rejects_disallowed_observation_lags() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        parse_algorithmic_program(
            "(program (state (lit 0)) (next (obs 1)) (emit (state 0)))"
        )

    assert exc_info.value.code == "bound_error"
