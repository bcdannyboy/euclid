from __future__ import annotations

import hashlib
from dataclasses import dataclass

from euclid.search.dsl import AlgorithmicProgram
from euclid.search.dsl.enumerator import (
    AlgorithmicEnumerationBounds,
    enumerate_algorithmic_programs,
)

DEFAULT_ALGORITHMIC_ENUMERATION_BOUNDS = AlgorithmicEnumerationBounds()
_PROGRAM_ALIASES = {
    "(program (state (lit 0)) (next (obs 0)) (emit (state 0)))": (
        "algorithmic_last_observation"
    ),
    (
        "(program (state (lit 0)) "
        "(next (div (add (state 0) (obs 0)) (lit 2))) "
        "(emit (state 0)))"
    ): "algorithmic_running_half_average",
}


@dataclass(frozen=True)
class AlgorithmicProposalSpec:
    candidate_id: str
    program: AlgorithmicProgram


def enumerate_algorithmic_proposal_specs(
    bounds: AlgorithmicEnumerationBounds | None = None,
) -> tuple[AlgorithmicProposalSpec, ...]:
    return tuple(
        AlgorithmicProposalSpec(
            candidate_id=_candidate_id_for_program(program),
            program=program,
        )
        for program in enumerate_algorithmic_programs(
            bounds or DEFAULT_ALGORITHMIC_ENUMERATION_BOUNDS
        )
    )


def _candidate_id_for_program(program: AlgorithmicProgram) -> str:
    alias = _PROGRAM_ALIASES.get(program.canonical_source)
    if alias is not None:
        return alias
    digest = hashlib.sha256(program.canonical_source.encode("utf-8")).hexdigest()[:12]
    return f"algorithmic_{digest}"


__all__ = [
    "AlgorithmicProposalSpec",
    "DEFAULT_ALGORITHMIC_ENUMERATION_BOUNDS",
    "enumerate_algorithmic_proposal_specs",
]
