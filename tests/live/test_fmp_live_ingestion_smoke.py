from __future__ import annotations

import math

import pytest

from euclid.fmp_smoke import build_csv_rows_from_fmp_history, fetch_fmp_eod_history
from euclid.runtime.env import FMP_API_KEY_ENV_VAR, EuclidEnv
from euclid.testing.live_api import LiveApiGate

pytestmark = pytest.mark.live_api


def test_fmp_live_ingestion_smoke(tmp_path) -> None:
    env = EuclidEnv.load()
    evidence_path = tmp_path / "fmp-live-evidence.json"

    def _check(credentials):
        history = fetch_fmp_eod_history(
            symbol="SPY",
            api_key=credentials[FMP_API_KEY_ENV_VAR],
        )
        rows = build_csv_rows_from_fmp_history(history[-8:], symbol="SPY")
        values = [float(row["observed_value"]) for row in rows]
        return {
            "schema_valid": True,
            "row_count": len(rows),
            "ordered": rows == sorted(rows, key=lambda row: row["event_time"]),
            "finite_values": all(math.isfinite(value) for value in values),
            "claim_published": False,
        }

    result = LiveApiGate(
        gate_id="P00-T06-live-fmp-ingestion",
        provider="fmp",
        endpoint_class="historical-price-eod",
        required_env=(FMP_API_KEY_ENV_VAR,),
        evidence_path=evidence_path,
        check=_check,
    ).run(env)

    if not env.live_tests_enabled:
        pytest.skip("EUCLID_LIVE_API_TESTS is not enabled")
    assert result.status == "passed"
