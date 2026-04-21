from __future__ import annotations

import math

import pytest

from euclid.fmp_smoke import build_csv_rows_from_fmp_history, fetch_fmp_eod_history
from euclid.runtime.env import FMP_API_KEY_ENV_VAR, OPENAI_API_KEY_ENV_VAR, EuclidEnv
from euclid.runtime.numerical_environment import capture_numerical_environment
from euclid.testing.live_api import LiveApiGate
from euclid.workbench.explainer import ensure_cached_workbench_explanations

pytestmark = pytest.mark.live_api


def test_live_dependency_runtime_metadata_is_sanitized_for_fmp(tmp_path) -> None:
    env = EuclidEnv.load()
    evidence_path = tmp_path / "dependency-fmp-live-evidence.json"

    def _check(credentials):
        history = fetch_fmp_eod_history(
            symbol="SPY",
            api_key=credentials[FMP_API_KEY_ENV_VAR],
        )
        rows = build_csv_rows_from_fmp_history(history[-5:], symbol="SPY")
        values = [float(row["observed_value"]) for row in rows]
        return {
            "schema_valid": True,
            "row_count": len(rows),
            "finite_values": all(math.isfinite(value) for value in values),
            "numerical_environment": capture_numerical_environment(),
            "claim_published": False,
        }

    result = LiveApiGate(
        gate_id="P01-live-dependency-fmp-runtime",
        provider="fmp",
        endpoint_class="historical-price-eod",
        required_env=(FMP_API_KEY_ENV_VAR,),
        evidence_path=evidence_path,
        check=_check,
    ).run(env)

    if not env.live_tests_enabled:
        pytest.skip("EUCLID_LIVE_API_TESTS is not enabled")
    assert result.status == "passed"
    evidence_text = evidence_path.read_text(encoding="utf-8")
    assert env.get(FMP_API_KEY_ENV_VAR) not in evidence_text


def test_live_dependency_runtime_metadata_is_sanitized_for_openai(tmp_path) -> None:
    env = EuclidEnv.load()
    evidence_path = tmp_path / "dependency-openai-live-evidence.json"
    analysis = {
        "dataset": {"symbol": "SPY", "rows": 1, "series": []},
        "operator_point": {
            "status": "abstained",
            "publication": {"headline": "No scientific claim published."},
        },
        "benchmark": {"status": "not_run"},
        "probabilistic": {},
    }

    def _check(credentials):
        updated = ensure_cached_workbench_explanations(
            analysis,
            api_key=credentials[OPENAI_API_KEY_ENV_VAR],
            analysis_path=None,
            timeout_seconds=60.0,
        )
        return {
            "schema_valid": isinstance(updated.get("llm_explanations", {}), dict),
            "numerical_environment": capture_numerical_environment(),
            "claim_published": False,
        }

    result = LiveApiGate(
        gate_id="P01-live-dependency-openai-runtime",
        provider="openai",
        endpoint_class="responses-workbench-explainer",
        required_env=(OPENAI_API_KEY_ENV_VAR,),
        evidence_path=evidence_path,
        check=_check,
    ).run(env)

    if not env.live_tests_enabled:
        pytest.skip("EUCLID_LIVE_API_TESTS is not enabled")
    assert result.status == "passed"
    evidence_text = evidence_path.read_text(encoding="utf-8")
    assert env.get(OPENAI_API_KEY_ENV_VAR) not in evidence_text
