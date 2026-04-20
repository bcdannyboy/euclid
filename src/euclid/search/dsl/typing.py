from __future__ import annotations

from typing import Any

from euclid.contracts.errors import ContractValidationError
from euclid.search.dsl.ast import AlgorithmicExpr
from euclid.search.dsl.canonicalize import _coerce_fraction

_ALLOWED_REAL_OPS = frozenset(
    {
        "lit",
        "obs",
        "state",
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "min",
        "max",
        "if",
    }
)
_ALLOWED_BOOL_OPS = frozenset({"true", "false", "lt", "le", "eq", "and", "or", "not"})


def parse_real_expr(
    node: Any,
    *,
    state_slot_count: int,
    allowed_observation_lags: tuple[int, ...],
    field_path: str,
) -> AlgorithmicExpr:
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ContractValidationError(
            code="parse_failed",
            message=f"{field_path} must be a legal real-expression form",
            field_path=field_path,
        )
    op = node[0]
    args = node[1:]
    if op not in _ALLOWED_REAL_OPS:
        raise ContractValidationError(
            code="forbidden_construct",
            message=f"{op!r} is not a legal real operator",
            field_path=field_path,
        )
    if op == "lit":
        if len(args) != 1 or not isinstance(args[0], str):
            raise ContractValidationError(
                code="parse_failed",
                message="lit requires exactly one rational atom",
                field_path=field_path,
            )
        return AlgorithmicExpr(
            op="lit",
            args=(_coerce_fraction(args[0]),),
            expr_type="real",
        )
    if op in {"obs", "state"}:
        if len(args) != 1 or not isinstance(args[0], str):
            raise ContractValidationError(
                code="parse_failed",
                message=f"{op} requires exactly one integer atom",
                field_path=field_path,
            )
        try:
            index = int(args[0])
        except ValueError as exc:
            raise ContractValidationError(
                code="parse_failed",
                message=f"{op} indices must be integers",
                field_path=field_path,
            ) from exc
        if index < 0:
            raise ContractValidationError(
                code="bound_error",
                message=f"{op} indices must be non-negative",
                field_path=field_path,
            )
        if op == "state" and index >= state_slot_count:
            raise ContractValidationError(
                code="bound_error",
                message="state index exceeds the declared state_slot_count",
                field_path=field_path,
                details={"index": index, "state_slot_count": state_slot_count},
            )
        if op == "obs" and index not in allowed_observation_lags:
            raise ContractValidationError(
                code="bound_error",
                message="observation lag is outside the admitted lag set",
                field_path=field_path,
                details={
                    "lag": index,
                    "allowed_observation_lags": list(allowed_observation_lags),
                },
            )
        return AlgorithmicExpr(op=op, args=(index,), expr_type="real")
    if op in {"add", "sub", "mul", "div", "min", "max"}:
        if len(args) != 2:
            raise ContractValidationError(
                code="parse_failed",
                message=f"{op} requires exactly two real arguments",
                field_path=field_path,
            )
        return AlgorithmicExpr(
            op=op,
            args=tuple(
                parse_real_expr(
                    arg,
                    state_slot_count=state_slot_count,
                    allowed_observation_lags=allowed_observation_lags,
                    field_path=f"{field_path}.{op}[{index}]",
                )
                for index, arg in enumerate(args)
            ),
            expr_type="real",
        )
    if op in {"neg", "abs"}:
        if len(args) != 1:
            raise ContractValidationError(
                code="parse_failed",
                message=f"{op} requires exactly one real argument",
                field_path=field_path,
            )
        return AlgorithmicExpr(
            op=op,
            args=(
                parse_real_expr(
                    args[0],
                    state_slot_count=state_slot_count,
                    allowed_observation_lags=allowed_observation_lags,
                    field_path=f"{field_path}.{op}[0]",
                ),
            ),
            expr_type="real",
        )
    if op == "if":
        if len(args) != 3:
            raise ContractValidationError(
                code="parse_failed",
                message="if requires predicate, then, and else branches",
                field_path=field_path,
            )
        predicate = parse_bool_expr(
            args[0],
            state_slot_count=state_slot_count,
            allowed_observation_lags=allowed_observation_lags,
            field_path=f"{field_path}.if[0]",
        )
        when_true = parse_real_expr(
            args[1],
            state_slot_count=state_slot_count,
            allowed_observation_lags=allowed_observation_lags,
            field_path=f"{field_path}.if[1]",
        )
        when_false = parse_real_expr(
            args[2],
            state_slot_count=state_slot_count,
            allowed_observation_lags=allowed_observation_lags,
            field_path=f"{field_path}.if[2]",
        )
        return AlgorithmicExpr(
            op="if",
            args=(predicate, when_true, when_false),
            expr_type="real",
        )
    raise ContractValidationError(
        code="forbidden_construct",
        message=f"{op!r} is not a legal real operator",
        field_path=field_path,
    )


def parse_bool_expr(
    node: Any,
    *,
    state_slot_count: int,
    allowed_observation_lags: tuple[int, ...],
    field_path: str,
) -> AlgorithmicExpr:
    if isinstance(node, str):
        if node in {"true", "false"}:
            return AlgorithmicExpr(op=node, args=(), expr_type="bool")
        raise ContractValidationError(
            code="parse_failed",
            message=f"{field_path} must be a legal boolean-expression form",
            field_path=field_path,
        )
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ContractValidationError(
            code="parse_failed",
            message=f"{field_path} must be a legal boolean-expression form",
            field_path=field_path,
        )
    op = node[0]
    args = node[1:]
    if op not in _ALLOWED_BOOL_OPS:
        raise ContractValidationError(
            code="forbidden_construct",
            message=f"{op!r} is not a legal boolean operator",
            field_path=field_path,
        )
    if op in {"true", "false"}:
        if args:
            raise ContractValidationError(
                code="parse_failed",
                message=f"{op} takes no arguments",
                field_path=field_path,
            )
        return AlgorithmicExpr(op=op, args=(), expr_type="bool")
    if op in {"lt", "le", "eq"}:
        if len(args) != 2:
            raise ContractValidationError(
                code="parse_failed",
                message=f"{op} requires exactly two real arguments",
                field_path=field_path,
            )
        return AlgorithmicExpr(
            op=op,
            args=tuple(
                parse_real_expr(
                    arg,
                    state_slot_count=state_slot_count,
                    allowed_observation_lags=allowed_observation_lags,
                    field_path=f"{field_path}.{op}[{index}]",
                )
                for index, arg in enumerate(args)
            ),
            expr_type="bool",
        )
    if op in {"and", "or"}:
        if len(args) != 2:
            raise ContractValidationError(
                code="parse_failed",
                message=f"{op} requires exactly two boolean arguments",
                field_path=field_path,
            )
        return AlgorithmicExpr(
            op=op,
            args=tuple(
                parse_bool_expr(
                    arg,
                    state_slot_count=state_slot_count,
                    allowed_observation_lags=allowed_observation_lags,
                    field_path=f"{field_path}.{op}[{index}]",
                )
                for index, arg in enumerate(args)
            ),
            expr_type="bool",
        )
    if len(args) != 1:
        raise ContractValidationError(
            code="parse_failed",
            message="not requires exactly one boolean argument",
            field_path=field_path,
        )
    return AlgorithmicExpr(
        op="not",
        args=(
            parse_bool_expr(
                args[0],
                state_slot_count=state_slot_count,
                allowed_observation_lags=allowed_observation_lags,
                field_path=f"{field_path}.not[0]",
            ),
        ),
        expr_type="bool",
    )


__all__ = [
    "parse_bool_expr",
    "parse_real_expr",
]
