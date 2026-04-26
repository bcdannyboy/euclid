from __future__ import annotations

import pytest
from tests.live.phase_static_live_helpers import assert_static_live_phase_gate

pytestmark = pytest.mark.live_api


def test_documented_live_commands_static_live_gate_is_traceable() -> None:
    assert_static_live_phase_gate("P16", "tests/live/test_documented_live_commands.py")
