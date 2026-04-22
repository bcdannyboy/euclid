from __future__ import annotations

from euclid.expr.ast import Feature, Literal, NaryOp
from euclid.expr.serialization import expression_canonical_json
from euclid.rewrites.egglog_runner import (
    EqualitySaturationConfig,
    run_equality_saturation,
)
from euclid.rewrites.extraction import expression_cost


def test_equality_saturation_extracts_lowest_cost_expression_with_evidence() -> None:
    x = Feature("x")
    expression = NaryOp(
        "mul",
        (
            NaryOp("add", (x, Literal(0))),
            Literal(1),
        ),
    )

    result = run_equality_saturation(
        expression,
        config=EqualitySaturationConfig(max_iterations=8, node_limit=64),
    )

    assert result.status == "completed"
    assert result.best_expression == x
    assert result.extraction.original_cost == expression_cost(expression)
    assert result.extraction.best_cost == expression_cost(x)
    assert result.extraction.best_cost < result.extraction.original_cost
    assert result.equivalence_evidence["status"] == "verified"
    assert result.equivalence_evidence["saturation_backend"] == "egglog"
    assert result.equivalence_evidence["egglog_rewrite_match_count"] > 0
    assert result.equivalence_evidence["egglog_extracted_expression"]
    assert result.eclass_count >= 1
    assert result.replay_identity.startswith("sha256:")


def test_equality_saturation_is_deterministic() -> None:
    expression = NaryOp(
        "add",
        (
            Literal(0),
            Feature("x"),
            NaryOp("mul", (Feature("y"), Literal(1))),
        ),
    )

    first = run_equality_saturation(expression)
    second = run_equality_saturation(expression)

    assert first.replay_identity == second.replay_identity
    first_json = expression_canonical_json(first.best_expression)
    second_json = expression_canonical_json(second.best_expression)
    assert first_json == second_json


def test_equality_saturation_respects_resource_bounds() -> None:
    expression = NaryOp("add", (Feature("x"), Literal(0)))

    result = run_equality_saturation(
        expression,
        config=EqualitySaturationConfig(max_iterations=0, node_limit=2),
    )

    assert result.status == "partial"
    assert result.best_expression == expression
    assert result.omission_disclosure["resource_limited"] is True
    assert result.omission_disclosure["max_iterations"] == 0


def test_extractor_tie_breaks_by_canonical_expression() -> None:
    expression = NaryOp(
        "add",
        (
            NaryOp("add", (Feature("b"), Literal(0))),
            NaryOp("add", (Feature("a"), Literal(0))),
        ),
    )

    result = run_equality_saturation(expression)

    assert result.extraction.tie_break_rule == (
        "min_cost_then_min_canonical_expression_json"
    )
    assert result.extraction.best_expression_canonical_json == (
        expression_canonical_json(result.best_expression)
    )
