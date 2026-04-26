from __future__ import annotations

import pytest
from tests.live.phase_static_live_helpers import assert_static_live_phase_gate

pytestmark = pytest.mark.live_api


def test_benchmark_static_live_gate_is_traceable() -> None:
    assert_static_live_phase_gate("P13", "tests/live/test_benchmark_live_smoke.py")
