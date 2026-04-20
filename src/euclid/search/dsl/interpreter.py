from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from euclid.contracts.errors import ContractValidationError
from euclid.search.dsl.ast import (
    AlgorithmicExpr,
    AlgorithmicProgram,
    AlgorithmicStepResult,
)
from euclid.search.dsl.canonicalize import _coerce_fraction


def evaluate_algorithmic_program(
    program: AlgorithmicProgram,
    *,
    state: Sequence[Fraction | int | float | str],
    observation: Sequence[Fraction | int | float | str] | Fraction | int | float | str,
) -> AlgorithmicStepResult:
    state_tuple = tuple(_coerce_fraction(value) for value in state)
    if len(state_tuple) != program.state_slot_count:
        raise ContractValidationError(
            code="bound_error",
            message="state length must match the program state_slot_count",
            field_path="state",
            details={"expected": program.state_slot_count, "actual": len(state_tuple)},
        )
    observation_history = _normalize_observation_history(
        observation=observation,
        allowed_observation_lags=program.allowed_observation_lags,
    )
    emit_value = _evaluate_real_expr(
        program.emit_expr,
        state_tuple,
        observation_history,
    )
    next_state = tuple(
        _evaluate_real_expr(expr, state_tuple, observation_history)
        for expr in program.next_state_exprs
    )
    return AlgorithmicStepResult(emit_value=emit_value, next_state=next_state)


def initialize_algorithmic_state(program: AlgorithmicProgram) -> tuple[Fraction, ...]:
    empty_state = tuple(Fraction(0, 1) for _ in range(program.state_slot_count))
    return tuple(
        _evaluate_real_expr(expr, empty_state, (Fraction(0, 1),))
        for expr in program.initial_state_exprs
    )


def _evaluate_real_expr(
    expr: AlgorithmicExpr,
    state: tuple[Fraction, ...],
    observation_history: tuple[Fraction, ...],
) -> Fraction:
    op = expr.op
    if op == "lit":
        return expr.args[0]
    if op == "state":
        return state[expr.args[0]]
    if op == "obs":
        return observation_history[expr.args[0]]
    if op == "add":
        return _evaluate_real_expr(
            expr.args[0], state, observation_history
        ) + _evaluate_real_expr(expr.args[1], state, observation_history)
    if op == "sub":
        return _evaluate_real_expr(
            expr.args[0], state, observation_history
        ) - _evaluate_real_expr(expr.args[1], state, observation_history)
    if op == "mul":
        return _evaluate_real_expr(
            expr.args[0], state, observation_history
        ) * _evaluate_real_expr(expr.args[1], state, observation_history)
    if op == "div":
        denominator = _evaluate_real_expr(expr.args[1], state, observation_history)
        if denominator == 0:
            raise ContractValidationError(
                code="division_by_zero_reachable",
                message="algorithmic evaluation reached division by zero",
                field_path="evaluation",
            )
        return (
            _evaluate_real_expr(expr.args[0], state, observation_history) / denominator
        )
    if op == "neg":
        return -_evaluate_real_expr(expr.args[0], state, observation_history)
    if op == "abs":
        return abs(_evaluate_real_expr(expr.args[0], state, observation_history))
    if op == "min":
        return min(
            _evaluate_real_expr(expr.args[0], state, observation_history),
            _evaluate_real_expr(expr.args[1], state, observation_history),
        )
    if op == "max":
        return max(
            _evaluate_real_expr(expr.args[0], state, observation_history),
            _evaluate_real_expr(expr.args[1], state, observation_history),
        )
    if op == "if":
        predicate = _evaluate_bool_expr(expr.args[0], state, observation_history)
        branch = expr.args[1] if predicate else expr.args[2]
        return _evaluate_real_expr(branch, state, observation_history)
    raise ContractValidationError(
        code="forbidden_construct",
        message=f"{op!r} is not a legal real operator",
        field_path="evaluation",
    )


def _evaluate_bool_expr(
    expr: AlgorithmicExpr,
    state: tuple[Fraction, ...],
    observation_history: tuple[Fraction, ...],
) -> bool:
    op = expr.op
    if op == "true":
        return True
    if op == "false":
        return False
    if op == "lt":
        return _evaluate_real_expr(
            expr.args[0], state, observation_history
        ) < _evaluate_real_expr(expr.args[1], state, observation_history)
    if op == "le":
        return _evaluate_real_expr(
            expr.args[0], state, observation_history
        ) <= _evaluate_real_expr(expr.args[1], state, observation_history)
    if op == "eq":
        return _evaluate_real_expr(
            expr.args[0], state, observation_history
        ) == _evaluate_real_expr(expr.args[1], state, observation_history)
    if op == "and":
        return _evaluate_bool_expr(
            expr.args[0], state, observation_history
        ) and _evaluate_bool_expr(expr.args[1], state, observation_history)
    if op == "or":
        return _evaluate_bool_expr(
            expr.args[0], state, observation_history
        ) or _evaluate_bool_expr(expr.args[1], state, observation_history)
    if op == "not":
        return not _evaluate_bool_expr(expr.args[0], state, observation_history)
    raise ContractValidationError(
        code="forbidden_construct",
        message=f"{op!r} is not a legal boolean operator",
        field_path="evaluation",
    )


def _normalize_observation_history(
    *,
    observation: Sequence[Fraction | int | float | str] | Fraction | int | float | str,
    allowed_observation_lags: tuple[int, ...],
) -> tuple[Fraction, ...]:
    if isinstance(observation, Sequence) and not isinstance(observation, (str, bytes)):
        history = tuple(_coerce_fraction(value) for value in observation)
    else:
        history = (_coerce_fraction(observation),)
    required_length = (
        (max(allowed_observation_lags) + 1) if allowed_observation_lags else 1
    )
    if len(history) < required_length:
        raise ContractValidationError(
            code="bound_error",
            message="observation history must cover the declared lag set",
            field_path="observation",
            details={
                "required_length": required_length,
                "actual_length": len(history),
                "allowed_observation_lags": list(allowed_observation_lags),
            },
        )
    return history


__all__ = [
    "evaluate_algorithmic_program",
    "initialize_algorithmic_state",
]
