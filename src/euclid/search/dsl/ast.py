from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_STATE_SLOT_PREFIX = "state_"


@dataclass(frozen=True)
class AlgorithmicExpr:
    op: str
    args: tuple[Any, ...]
    expr_type: str

    def node_count(self) -> int:
        count = 1
        for arg in self.args:
            if isinstance(arg, AlgorithmicExpr):
                count += arg.node_count()
        return count

    def canonical_source(self) -> str:
        from euclid.search.dsl.canonicalize import canonicalize_fraction

        if self.op in {"true", "false"}:
            return self.op
        if self.op == "lit":
            return f"(lit {canonicalize_fraction(self.args[0])})"
        if self.op in {"obs", "state"}:
            return f"({self.op} {self.args[0]})"
        inner = " ".join(
            arg.canonical_source() if isinstance(arg, AlgorithmicExpr) else str(arg)
            for arg in self.args
        )
        return f"({self.op} {inner})"

    def contains_observation_access(self) -> bool:
        if self.op == "obs":
            return True
        return any(
            arg.contains_observation_access()
            for arg in self.args
            if isinstance(arg, AlgorithmicExpr)
        )

    def contains_state_access(self) -> bool:
        if self.op == "state":
            return True
        return any(
            arg.contains_state_access()
            for arg in self.args
            if isinstance(arg, AlgorithmicExpr)
        )


@dataclass(frozen=True)
class AlgorithmicProgram:
    initial_state_exprs: tuple[AlgorithmicExpr, ...]
    next_state_exprs: tuple[AlgorithmicExpr, ...]
    emit_expr: AlgorithmicExpr
    canonical_source: str
    state_slot_count: int
    node_count: int
    allowed_observation_lags: tuple[int, ...]

    @property
    def state_slot_names(self) -> tuple[str, ...]:
        return tuple(
            f"{_STATE_SLOT_PREFIX}{index}" for index in range(self.state_slot_count)
        )


@dataclass(frozen=True)
class AlgorithmicStepResult:
    emit_value: object
    next_state: tuple[object, ...]


__all__ = [
    "AlgorithmicExpr",
    "AlgorithmicProgram",
    "AlgorithmicStepResult",
]
