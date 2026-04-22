from __future__ import annotations

import json
from urllib.error import HTTPError, URLError

import pytest

from euclid.runtime.env import EuclidEnv
from euclid.testing.live_api import LiveApiGate, validate_ordered_provider_rows


def test_live_api_gate_skips_when_live_tests_are_disabled(tmp_path) -> None:
    env = EuclidEnv.load(env_file=tmp_path / ".env", environ={})
    gate = LiveApiGate(
        gate_id="P00-T06-live-disabled",
        provider="fmp",
        endpoint_class="historical-price-eod",
        required_env=("FMP_API_KEY",),
        evidence_path=tmp_path / "evidence.json",
        check=lambda credentials: {"schema_valid": True},
    )

    result = gate.run(env)

    assert result.status == "skipped"
    assert result.reason_codes == ("live_api_tests_disabled",)
    assert json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))[
        "status"
    ] == "skipped"
    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["claim_evidence_status"] == "not_claim_evidence"
    assert evidence["claim_boundary"] == {
        "counts_as_scientific_claim_evidence": False,
        "reason_code": "live_api_gate_validates_access_and_payload_shape_only",
        "status": "non_claim_evidence",
    }


def test_live_api_gate_fails_closed_for_missing_strict_credentials(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EUCLID_LIVE_API_TESTS=1\nEUCLID_LIVE_API_STRICT=1\n",
        encoding="utf-8",
    )
    env = EuclidEnv.load(env_file=env_path, environ={})
    gate = LiveApiGate(
        gate_id="P00-T06-live-missing-key",
        provider="fmp",
        endpoint_class="historical-price-eod",
        required_env=("FMP_API_KEY",),
        evidence_path=tmp_path / "evidence.json",
        check=lambda credentials: {"schema_valid": True},
    )

    result = gate.run(env)

    assert result.status == "failed"
    assert result.reason_codes == ("missing_live_api_credentials",)
    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["credential_presence"] == {
        "FMP_API_KEY": {"present": False, "source": None}
    }
    assert "apikey" not in repr(evidence).lower()


def test_live_api_gate_fails_closed_for_blank_strict_credentials(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EUCLID_LIVE_API_TESTS=1\nEUCLID_LIVE_API_STRICT=1\nFMP_API_KEY=\n",
        encoding="utf-8",
    )
    env = EuclidEnv.load(env_file=env_path, environ={})
    gate = LiveApiGate(
        gate_id="P15-live-blank-key",
        provider="fmp",
        endpoint_class="historical-price-eod",
        required_env=("FMP_API_KEY",),
        evidence_path=tmp_path / "evidence.json",
        check=lambda credentials: {"schema_valid": True},
    )

    result = gate.run(env)

    assert result.status == "failed"
    assert result.reason_codes == ("missing_live_api_credentials",)
    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["credential_presence"]["FMP_API_KEY"]["present"] is False


def test_live_api_gate_writes_sanitized_success_evidence(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "EUCLID_LIVE_API_TESTS=1",
                "EUCLID_LIVE_API_STRICT=1",
                "FMP_API_KEY=fmp-secret",
            ]
        ),
        encoding="utf-8",
    )
    env = EuclidEnv.load(env_file=env_path, environ={})

    def _check(credentials):
        assert credentials == {"FMP_API_KEY": "fmp-secret"}
        return {
            "schema_valid": True,
            "request_url": "https://example.test/history?apikey=fmp-secret",
            "authorization": "Bearer fmp-secret",
        }

    gate = LiveApiGate(
        gate_id="P00-T06-live-success",
        provider="fmp",
        endpoint_class="historical-price-eod",
        required_env=("FMP_API_KEY",),
        evidence_path=tmp_path / "evidence.json",
        check=_check,
    )

    result = gate.run(env)

    assert result.status == "passed"
    evidence_text = (tmp_path / "evidence.json").read_text(encoding="utf-8")
    assert "fmp-secret" not in evidence_text
    evidence = json.loads(evidence_text)
    assert evidence["semantic_checks"]["schema_valid"] is True
    assert evidence["semantic_checks"]["request_url"].endswith("apikey=%5BREDACTED%5D")
    assert evidence["claim_evidence_status"] == "not_claim_evidence"
    assert evidence["claim_boundary"]["counts_as_scientific_claim_evidence"] is False


@pytest.mark.parametrize(
    ("exc", "expected_reason"),
    [
        (
            HTTPError(
                "https://example.test/history",
                401,
                "invalid credential fmp-secret",
                hdrs=None,
                fp=None,
            ),
            "provider_http_401",
        ),
        (
            HTTPError(
                "https://example.test/history",
                429,
                "rate limit for fmp-secret",
                hdrs=None,
                fp=None,
            ),
            "provider_http_429",
        ),
        (TimeoutError("request timed out for fmp-secret"), "provider_timeout"),
        (URLError(TimeoutError("timed out")), "provider_timeout"),
        (ValueError("empty payload from provider fmp-secret"), "provider_payload_invalid"),
    ],
)
def test_live_api_gate_fails_closed_with_sanitized_provider_evidence(
    tmp_path,
    exc: Exception,
    expected_reason: str,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "EUCLID_LIVE_API_TESTS=1",
                "EUCLID_LIVE_API_STRICT=1",
                "FMP_API_KEY=fmp-secret",
            ]
        ),
        encoding="utf-8",
    )
    env = EuclidEnv.load(env_file=env_path, environ={})
    gate = LiveApiGate(
        gate_id="P15-live-provider-failure",
        provider="fmp",
        endpoint_class="historical-price-eod",
        required_env=("FMP_API_KEY",),
        evidence_path=tmp_path / "evidence.json",
        check=lambda credentials: (_ for _ in ()).throw(exc),
    )

    result = gate.run(env)

    assert result.status == "failed"
    assert result.reason_codes == (expected_reason,)
    evidence_text = (tmp_path / "evidence.json").read_text(encoding="utf-8")
    assert "fmp-secret" not in evidence_text
    evidence = json.loads(evidence_text)
    assert evidence["claim_boundary"]["counts_as_scientific_claim_evidence"] is False
    assert evidence["credential_presence"]["FMP_API_KEY"]["present"] is True


def test_validate_ordered_provider_rows_rejects_empty_payloads() -> None:
    with pytest.raises(ValueError, match="empty payload"):
        validate_ordered_provider_rows([], timestamp_key="date")
