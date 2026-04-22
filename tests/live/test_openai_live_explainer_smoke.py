from __future__ import annotations

import pytest

from euclid.runtime.env import OPENAI_API_KEY_ENV_VAR, EuclidEnv
from euclid.testing.live_api import LiveApiGate
from euclid.workbench.explainer import ensure_cached_workbench_explanations

pytestmark = pytest.mark.live_api


def test_openai_live_explainer_smoke(tmp_path) -> None:
    env = EuclidEnv.load()
    evidence_path = tmp_path / "openai-live-evidence.json"
    analysis = {
        "dataset": {
            "symbol": "SPY",
            "rows": 3,
            "target": {"label": "Close", "description": "Fixture close values"},
            "series": [
                {"event_time": "2026-01-01T00:00:00Z", "observed_value": 1.0},
                {"event_time": "2026-01-02T00:00:00Z", "observed_value": 2.0},
                {"event_time": "2026-01-03T00:00:00Z", "observed_value": 3.0},
            ],
            "stats": {"min": 1.0, "max": 3.0, "mean": 2.0},
        },
        "operator_point": {
            "status": "abstained",
            "publication": {"headline": "No scientific claim published."},
            "abstention": {"reason_codes": ["live_smoke_fixture"]},
        },
        "benchmark": {"status": "not_run", "portfolio_selection": {}},
        "probabilistic": {},
    }

    def _check(credentials):
        updated = ensure_cached_workbench_explanations(
            analysis,
            api_key=credentials[OPENAI_API_KEY_ENV_VAR],
            analysis_path=None,
            timeout_seconds=60.0,
        )
        explanations = updated.get("llm_explanations", {})
        return {
            "schema_valid": isinstance(explanations, dict),
            "status": explanations.get("status"),
            "model": explanations.get("model"),
            "claim_published": False,
        }

    result = LiveApiGate(
        gate_id="P00-T06-live-openai-explainer",
        provider="openai",
        endpoint_class="responses-workbench-explainer",
        required_env=(OPENAI_API_KEY_ENV_VAR,),
        evidence_path=evidence_path,
        check=_check,
    ).run(env)

    if not env.live_tests_enabled:
        pytest.skip("EUCLID_LIVE_API_TESTS is not enabled")
    assert result.status == "passed"
