from __future__ import annotations

from euclid.expr.ast import BinaryOp, Feature, Literal, State
from euclid.expr.serialization import expression_to_dict
from euclid.modules.algorithmic_dsl import (
    lower_algorithmic_program_to_expression_ir,
    parse_algorithmic_program,
)


def test_legacy_algorithmic_dsl_lowers_emit_expression_to_expression_ir() -> None:
    program = parse_algorithmic_program(
        "(program (state (lit 0)) (next (add (state 0) (obs 0))) (emit (state 0)))",
        state_slot_count=1,
        allowed_observation_lags=(0,),
    )

    lowered = lower_algorithmic_program_to_expression_ir(program)

    assert expression_to_dict(lowered.emit_expression) == expression_to_dict(State("state_0"))
    assert expression_to_dict(lowered.next_state_expressions[0]) == expression_to_dict(
        BinaryOp("add", State("state_0"), Feature("obs_0"))
    )
    assert lowered.legacy_source_ref == program.canonical_source
    assert lowered.primary_runtime == "expression_ir"

