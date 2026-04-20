from __future__ import annotations

from dataclasses import dataclass

from euclid.contracts.errors import ContractValidationError
from euclid.search.dsl.ast import AlgorithmicProgram
from euclid.search.dsl.canonicalize import canonicalize_fraction
from euclid.search.dsl.parser import parse_algorithmic_program


@dataclass(frozen=True)
class AlgorithmicEnumerationBounds:
    state_slot_count: int = 1
    max_program_nodes: int = 8
    allowed_observation_lags: tuple[int, ...] = (0,)
    literal_palette: tuple[int | str, ...] = (0, 1, 2)
    unary_real_ops: tuple[str, ...] = ()
    binary_terminal_ops: tuple[str, ...] = ("add", "sub", "mul", "min", "max")
    binary_real_ops: tuple[str, ...] | None = None
    comparison_ops: tuple[str, ...] = ()
    boolean_literals: tuple[str, ...] = ()
    logical_ops: tuple[str, ...] = ()
    include_half_average_templates: bool = True
    include_conditional_templates: bool = False
    emit_state_index: int = 0


def enumerate_algorithmic_programs(
    bounds: AlgorithmicEnumerationBounds | None = None,
) -> tuple[AlgorithmicProgram, ...]:
    resolved_bounds = bounds or AlgorithmicEnumerationBounds()
    if resolved_bounds.state_slot_count < 1:
        raise ContractValidationError(
            code="bound_error",
            message="algorithmic enumerator requires at least one state slot",
            field_path="state_slot_count",
        )
    if not 0 <= resolved_bounds.emit_state_index < resolved_bounds.state_slot_count:
        raise ContractValidationError(
            code="bound_error",
            message="emit_state_index must point at a declared state slot",
            field_path="emit_state_index",
            details={
                "emit_state_index": resolved_bounds.emit_state_index,
                "state_slot_count": resolved_bounds.state_slot_count,
            },
        )

    next_expr_sources = _real_expression_sources(resolved_bounds)
    state_overhead = resolved_bounds.state_slot_count + 1
    next_node_budget = resolved_bounds.max_program_nodes - state_overhead
    if next_node_budget < resolved_bounds.state_slot_count:
        return ()

    next_expr_tuples = _next_state_expression_tuples(
        bounds=resolved_bounds,
        next_expr_sources=next_expr_sources,
        next_node_budget=next_node_budget,
    )
    program_sources = (
        _program_source(
            state_slot_count=resolved_bounds.state_slot_count,
            next_expr_sources=next_expr_tuple,
            emit_state_index=resolved_bounds.emit_state_index,
        )
        for next_expr_tuple in next_expr_tuples
    )
    programs_by_source = {
        program.canonical_source: program
        for program in (
            parse_algorithmic_program(
                source,
                state_slot_count=resolved_bounds.state_slot_count,
                max_program_nodes=resolved_bounds.max_program_nodes,
                allowed_observation_lags=resolved_bounds.allowed_observation_lags,
            )
            for source in program_sources
        )
    }
    return tuple(
        programs_by_source[source] for source in sorted(programs_by_source)
    )


def _real_expression_sources(
    bounds: AlgorithmicEnumerationBounds,
) -> tuple[tuple[str, int], ...]:
    terminals = _terminal_expression_sources(bounds)
    sources: dict[str, int] = dict(terminals)
    terminal_sources = tuple(terminals)
    binary_ops = tuple(bounds.binary_real_ops or bounds.binary_terminal_ops)

    for op in bounds.unary_real_ops:
        for inner_source, inner_nodes in terminal_sources:
            sources[f"({op} {inner_source})"] = 1 + inner_nodes

    for op in binary_ops:
        for left_source, left_nodes in terminal_sources:
            for right_source, right_nodes in terminal_sources:
                sources[f"({op} {left_source} {right_source})"] = (
                    1 + left_nodes + right_nodes
                )

    if bounds.include_half_average_templates:
        for left_source, left_nodes in terminal_sources:
            for right_source, right_nodes in terminal_sources:
                sources[
                    f"(div (add {left_source} {right_source}) (lit 2))"
                ] = 3 + left_nodes + right_nodes

    if bounds.include_conditional_templates:
        bool_sources = _boolean_expression_sources(bounds, terminal_sources)
        branch_sources = tuple(sorted(sources.items()))
        for predicate_source, predicate_nodes in bool_sources:
            for when_true, true_nodes in branch_sources:
                for when_false, false_nodes in branch_sources:
                    sources[
                        f"(if {predicate_source} {when_true} {when_false})"
                    ] = 1 + predicate_nodes + true_nodes + false_nodes

    return tuple(sorted(sources.items()))


def _boolean_expression_sources(
    bounds: AlgorithmicEnumerationBounds,
    terminal_sources: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    sources: dict[str, int] = {
        literal: 1 for literal in bounds.boolean_literals if literal in {"true", "false"}
    }

    for op in bounds.comparison_ops:
        for left_source, left_nodes in terminal_sources:
            for right_source, right_nodes in terminal_sources:
                sources[f"({op} {left_source} {right_source})"] = (
                    1 + left_nodes + right_nodes
                )

    boolean_atoms = tuple(sorted(sources.items()))
    if "not" in bounds.logical_ops:
        for inner_source, inner_nodes in boolean_atoms:
            sources[f"(not {inner_source})"] = 1 + inner_nodes
    if "and" in bounds.logical_ops:
        for left_source, left_nodes in boolean_atoms:
            for right_source, right_nodes in boolean_atoms:
                sources[f"(and {left_source} {right_source})"] = (
                    1 + left_nodes + right_nodes
                )
    if "or" in bounds.logical_ops:
        for left_source, left_nodes in boolean_atoms:
            for right_source, right_nodes in boolean_atoms:
                sources[f"(or {left_source} {right_source})"] = (
                    1 + left_nodes + right_nodes
                )

    return tuple(sorted(sources.items()))


def _next_state_expression_tuples(
    *,
    bounds: AlgorithmicEnumerationBounds,
    next_expr_sources: tuple[tuple[str, int], ...],
    next_node_budget: int,
) -> tuple[tuple[str, ...], ...]:
    tuples: set[tuple[str, ...]] = set()

    def walk(
        slot_index: int,
        consumed_nodes: int,
        selected_sources: tuple[str, ...],
    ) -> None:
        if slot_index == bounds.state_slot_count:
            tuples.add(selected_sources)
            return
        remaining_slots = bounds.state_slot_count - slot_index - 1
        for source, node_count in next_expr_sources:
            total_nodes = consumed_nodes + node_count
            if total_nodes + remaining_slots > next_node_budget:
                continue
            walk(
                slot_index + 1,
                total_nodes,
                (*selected_sources, source),
            )

    walk(0, 0, ())
    return tuple(sorted(tuples))


def _terminal_expression_sources(
    bounds: AlgorithmicEnumerationBounds,
) -> tuple[tuple[str, int], ...]:
    literals = tuple(
        (f"(lit {canonicalize_fraction(value)})", 1)
        for value in bounds.literal_palette
    )
    observations = tuple(
        (f"(obs {lag})", 1) for lag in sorted(set(bounds.allowed_observation_lags))
    )
    states = tuple((f"(state {index})", 1) for index in range(bounds.state_slot_count))
    return tuple(sorted((*literals, *observations, *states)))


def _program_source(
    *,
    state_slot_count: int,
    next_expr_sources: tuple[str, ...],
    emit_state_index: int,
) -> str:
    initial_state = " ".join("(lit 0)" for _ in range(state_slot_count))
    next_state = " ".join(next_expr_sources)
    return (
        f"(program (state {initial_state}) "
        f"(next {next_state}) "
        f"(emit (state {emit_state_index})))"
    )


__all__ = [
    "AlgorithmicEnumerationBounds",
    "enumerate_algorithmic_programs",
]
