from __future__ import annotations

import json
import socket
from urllib.error import HTTPError

from euclid.runtime.env import EuclidEnv
from euclid.testing.live_api import LiveApiGate


def test_fixture_mode_live_gate_records_rate_limit_without_secret_leak(tmp_path) -> None:
    secret = "fmp-fixture-secret"
    env_path = tmp_path / ".env"
    env_path.write_text(
        f"EUCLID_LIVE_API_TESTS=1\nFMP_API_KEY={secret}\n",
        encoding="utf-8",
    )
    env = EuclidEnv.load(env_file=env_path, environ={})

    def _rate_limited(_credentials):
        raise HTTPError(
            url=f"https://example.test/history?apikey={secret}",
            code=429,
            msg=f"rate limited key {secret}",
            hdrs=None,
            fp=None,
        )

    gate = LiveApiGate(
        gate_id="P00-T06-fixture-rate-limit",
        provider="fmp",
        endpoint_class="historical-price-eod",
        required_env=("FMP_API_KEY",),
        evidence_path=tmp_path / "evidence.json",
        check=_rate_limited,
    )

    result = gate.run(env)

    assert result.status == "failed"
    assert result.reason_codes == ("provider_http_429",)
    evidence_text = (tmp_path / "evidence.json").read_text(encoding="utf-8")
    assert secret not in evidence_text
    assert json.loads(evidence_text)["provider"] == "fmp"


def test_fixture_mode_live_gate_records_malformed_payload_as_typed_failure(
    tmp_path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EUCLID_LIVE_API_TESTS=1\nFMP_API_KEY=fmp-fixture-secret\n",
        encoding="utf-8",
    )
    env = EuclidEnv.load(env_file=env_path, environ={})

    def _malformed(_credentials):
        raise ValueError("malformed JSON from provider")

    gate = LiveApiGate(
        gate_id="P00-T06-fixture-malformed-json",
        provider="fmp",
        endpoint_class="historical-price-eod",
        required_env=("FMP_API_KEY",),
        evidence_path=tmp_path / "evidence.json",
        check=_malformed,
    )

    result = gate.run(env)

    assert result.status == "failed"
    assert result.reason_codes == ("provider_payload_invalid",)
    assert "malformed JSON" in json.loads(
        (tmp_path / "evidence.json").read_text(encoding="utf-8")
    )["message"]


def test_fixture_mode_live_gate_maps_http_401_and_500_without_secret_leak(
    tmp_path,
) -> None:
    for code, expected_reason in ((401, "provider_http_401"), (500, "provider_http_5xx")):
        secret = f"fmp-fixture-secret-{code}"
        env_path = tmp_path / f".env.{code}"
        env_path.write_text(
            f"EUCLID_LIVE_API_TESTS=1\nFMP_API_KEY={secret}\n",
            encoding="utf-8",
        )
        env = EuclidEnv.load(env_file=env_path, environ={})

        def _http_failure(_credentials, *, status_code=code):
            raise HTTPError(
                url=f"https://example.test/history?apikey={secret}",
                code=status_code,
                msg=f"provider rejected {secret}",
                hdrs=None,
                fp=None,
            )

        evidence_path = tmp_path / f"evidence-{code}.json"
        result = LiveApiGate(
            gate_id=f"P00-T06-fixture-http-{code}",
            provider="fmp",
            endpoint_class="historical-price-eod",
            required_env=("FMP_API_KEY",),
            evidence_path=evidence_path,
            check=_http_failure,
        ).run(env)

        assert result.status == "failed"
        assert result.reason_codes == (expected_reason,)
        assert secret not in evidence_path.read_text(encoding="utf-8")


def test_fixture_mode_live_gate_maps_timeout_as_typed_failure(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EUCLID_LIVE_API_TESTS=1\nOPENAI_API_KEY=openai-fixture-secret\n",
        encoding="utf-8",
    )
    env = EuclidEnv.load(env_file=env_path, environ={})

    result = LiveApiGate(
        gate_id="P00-T06-fixture-openai-timeout",
        provider="openai",
        endpoint_class="responses-workbench-explainer",
        required_env=("OPENAI_API_KEY",),
        evidence_path=tmp_path / "timeout-evidence.json",
        check=lambda _credentials: (_ for _ in ()).throw(socket.timeout("timed out")),
    ).run(env)

    assert result.status == "failed"
    assert result.reason_codes == ("provider_timeout",)


def test_fixture_mode_live_gate_records_schema_edge_case_failures(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EUCLID_LIVE_API_TESTS=1\nFMP_API_KEY=fmp-fixture-secret\n",
        encoding="utf-8",
    )
    env = EuclidEnv.load(env_file=env_path, environ={})

    def _schema_drift(_credentials):
        rows = [
            {"date": "2026-01-02", "close": 2.0},
            {"date": "2026-01-02", "close": 2.0},
            {"date": "2026-01-01", "close": 1.0},
        ]
        if not rows:
            raise ValueError("empty history")
        dates = [row["date"] for row in rows]
        if len(set(dates)) != len(dates):
            raise ValueError("duplicate dates")
        if dates != sorted(dates):
            raise ValueError("out-of-order rows")
        return {"schema_valid": True}

    result = LiveApiGate(
        gate_id="P00-T06-fixture-fmp-schema-drift",
        provider="fmp",
        endpoint_class="historical-price-eod",
        required_env=("FMP_API_KEY",),
        evidence_path=tmp_path / "schema-drift-evidence.json",
        check=_schema_drift,
    ).run(env)

    assert result.status == "failed"
    assert result.reason_codes == ("provider_payload_invalid",)


def test_fixture_mode_openai_success_handles_refusal_and_tool_free_text(
    tmp_path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EUCLID_LIVE_API_TESTS=1\nOPENAI_API_KEY=openai-fixture-secret\n",
        encoding="utf-8",
    )
    env = EuclidEnv.load(env_file=env_path, environ={})

    result = LiveApiGate(
        gate_id="P00-T06-fixture-openai-refusal-or-text",
        provider="openai",
        endpoint_class="responses-workbench-explainer",
        required_env=("OPENAI_API_KEY",),
        evidence_path=tmp_path / "openai-evidence.json",
        check=lambda _credentials: {
            "schema_valid": True,
            "typed_abstention_reason": "explanation_unavailable",
            "tool_calls_required": False,
            "claim_published": False,
        },
    ).run(env)

    assert result.status == "passed"
    evidence = json.loads((tmp_path / "openai-evidence.json").read_text("utf-8"))
    assert evidence["semantic_checks"]["typed_abstention_reason"] == (
        "explanation_unavailable"
    )
