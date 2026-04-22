from __future__ import annotations

from euclid.expr.ast import BinaryOp, Feature, Literal, Parameter, expression_hash
from euclid.expr.serialization import (
    expression_canonical_json,
    expression_from_dict,
    expression_to_dict,
)


def test_serialization_round_trip_keeps_replay_stable_expression_ids() -> None:
    expression = BinaryOp(
        "mul",
        BinaryOp("add", Feature("x", unit="meter"), Literal(1.0, unit="meter")),
        Parameter("beta", domain="real"),
    )

    payload = expression_to_dict(expression)
    restored = expression_from_dict(payload)

    assert expression_to_dict(restored) == payload
    assert expression_canonical_json(restored) == expression_canonical_json(expression)
    assert expression_hash(restored) == expression_hash(expression)

