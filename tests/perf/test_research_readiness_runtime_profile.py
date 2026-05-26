from __future__ import annotations

import time

from tests.fixtures.research_readiness import (
    restore_seeded_inputs,
    seed_research_ready_inputs,
)

import euclid.release as release


def test_research_readiness_evaluation_stays_within_runtime_budget(
    project_root,
) -> None:
    seeded_inputs = seed_research_ready_inputs(project_root)
    try:
        start = time.perf_counter()
        result = release.certify_research_readiness(project_root=project_root)
        elapsed = time.perf_counter() - start

        assert result.status == "ready"
        assert elapsed <= 30.0
    finally:
        restore_seeded_inputs(seeded_inputs)
