from __future__ import annotations

import re
from typing import Any, Sequence

from euclid.contracts.errors import ContractValidationError
from euclid.search.dsl.ast import AlgorithmicProgram
from euclid.search.dsl.typing import parse_real_expr

_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


def parse_algorithmic_program(
    source: str,
    *,
    state_slot_count: int = 1,
    max_program_nodes: int = 8,
    allowed_observation_lags: Sequence[int] = (0,),
) -> AlgorithmicProgram:
    if state_slot_count < 0:
        raise ContractValidationError(
            code="bound_error",
            message="state_slot_count must be non-negative",
            field_path="state_slot_count",
        )
    if max_program_nodes <= 0:
        raise ContractValidationError(
            code="bound_error",
            message="max_program_nodes must be positive",
            field_path="max_program_nodes",
        )
    allowed_lags = tuple(sorted({int(lag) for lag in allowed_observation_lags}))
    if not allowed_lags or any(lag < 0 for lag in allowed_lags):
        raise ContractValidationError(
            code="bound_error",
            message="allowed_observation_lags must be non-empty non-negative integers",
            field_path="allowed_observation_lags",
        )
    tree = _parse_s_expression(source)
    if not isinstance(tree, list) or not tree or tree[0] != "program":
        raise ContractValidationError(
            code="parse_failed",
            message="program source must start with (program ...)",
            field_path="source",
        )
    sections = tree[1:]
    if len(sections) != 3:
        raise ContractValidationError(
            code="parse_failed",
            message="program requires exactly state, next, and emit sections",
            field_path="source",
        )
    section_map: dict[str, list[Any]] = {}
    for section in sections:
        if (
            not isinstance(section, list)
            or not section
            or not isinstance(section[0], str)
        ):
            raise ContractValidationError(
                code="parse_failed",
                message="program sections must be lists headed by identifiers",
                field_path="source",
            )
        section_name = section[0]
        if section_name in section_map:
            raise ContractValidationError(
                code="parse_failed",
                message=f"duplicate program section {section_name!r}",
                field_path="source",
            )
        section_map[section_name] = section[1:]
    if set(section_map) != {"state", "next", "emit"}:
        raise ContractValidationError(
            code="parse_failed",
            message="program requires state, next, and emit sections",
            field_path="source",
        )
    initial_exprs = tuple(
        parse_real_expr(
            expr,
            state_slot_count=state_slot_count,
            allowed_observation_lags=allowed_lags,
            field_path=f"state[{index}]",
        )
        for index, expr in enumerate(section_map["state"])
    )
    next_exprs = tuple(
        parse_real_expr(
            expr,
            state_slot_count=state_slot_count,
            allowed_observation_lags=allowed_lags,
            field_path=f"next[{index}]",
        )
        for index, expr in enumerate(section_map["next"])
    )
    if len(initial_exprs) != state_slot_count:
        raise ContractValidationError(
            code="bound_error",
            message="state section length must match state_slot_count",
            field_path="state",
            details={"expected": state_slot_count, "actual": len(initial_exprs)},
        )
    if len(next_exprs) != state_slot_count:
        raise ContractValidationError(
            code="bound_error",
            message="next section length must match state_slot_count",
            field_path="next",
            details={"expected": state_slot_count, "actual": len(next_exprs)},
        )
    for index, expr in enumerate(initial_exprs):
        if expr.contains_observation_access() or expr.contains_state_access():
            raise ContractValidationError(
                code="forbidden_construct",
                message="initial state expressions must be closed over literals only",
                field_path=f"state[{index}]",
            )
    emit_args = section_map["emit"]
    if len(emit_args) != 1:
        raise ContractValidationError(
            code="parse_failed",
            message="emit section must contain exactly one expression",
            field_path="emit",
        )
    emit_expr = parse_real_expr(
        emit_args[0],
        state_slot_count=state_slot_count,
        allowed_observation_lags=allowed_lags,
        field_path="emit",
    )
    if emit_expr.contains_observation_access():
        raise ContractValidationError(
            code="forbidden_construct",
            message="emit expressions may not access observations directly",
            field_path="emit",
        )
    node_count = (
        sum(expr.node_count() for expr in initial_exprs)
        + sum(expr.node_count() for expr in next_exprs)
        + emit_expr.node_count()
    )
    if node_count > max_program_nodes:
        raise ContractValidationError(
            code="bound_error",
            message="program exceeds the declared max_program_nodes",
            field_path="source",
            details={"node_count": node_count, "max_program_nodes": max_program_nodes},
        )
    canonical_source = (
        "(program "
        f"(state {' '.join(expr.canonical_source() for expr in initial_exprs)}) "
        f"(next {' '.join(expr.canonical_source() for expr in next_exprs)}) "
        f"(emit {emit_expr.canonical_source()}))"
    )
    return AlgorithmicProgram(
        initial_state_exprs=initial_exprs,
        next_state_exprs=next_exprs,
        emit_expr=emit_expr,
        canonical_source=canonical_source,
        state_slot_count=state_slot_count,
        node_count=node_count,
        allowed_observation_lags=allowed_lags,
    )


def _parse_s_expression(source: str) -> Any:
    tokens = _TOKEN_RE.findall(source)
    if not tokens:
        raise ContractValidationError(
            code="parse_failed",
            message="program source must not be empty",
            field_path="source",
        )
    index = 0

    def parse_node() -> Any:
        nonlocal index
        if index >= len(tokens):
            raise ContractValidationError(
                code="parse_failed",
                message="unexpected end of program source",
                field_path="source",
            )
        token = tokens[index]
        index += 1
        if token == "(":
            items: list[Any] = []
            while True:
                if index >= len(tokens):
                    raise ContractValidationError(
                        code="parse_failed",
                        message="unclosed parenthesis in program source",
                        field_path="source",
                    )
                if tokens[index] == ")":
                    index += 1
                    return items
                items.append(parse_node())
        if token == ")":
            raise ContractValidationError(
                code="parse_failed",
                message="unexpected closing parenthesis in program source",
                field_path="source",
            )
        return token

    tree = parse_node()
    if index != len(tokens):
        raise ContractValidationError(
            code="parse_failed",
            message="unexpected trailing tokens in program source",
            field_path="source",
        )
    return tree


__all__ = [
    "parse_algorithmic_program",
]
