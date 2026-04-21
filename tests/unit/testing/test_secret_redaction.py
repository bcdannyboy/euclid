from __future__ import annotations

from euclid.testing.redaction import (
    collect_secret_values,
    redact_mapping,
    redact_text,
)


def test_redacts_secret_values_from_nested_payloads_headers_urls_and_text() -> None:
    secret = "sk-test-secret-value"
    payload = {
        "url": f"https://example.test/data?apikey={secret}&symbol=SPY",
        "headers": {
            "Authorization": f"Bearer {secret}",
            "X-Api-Key": secret,
        },
        "nested": [
            {"message": f"provider rejected {secret}"},
            {"safe": "value"},
        ],
    }

    redacted = redact_mapping(payload, secrets=[secret])
    rendered = repr(redacted)

    assert secret not in rendered
    assert "[REDACTED]" in rendered
    assert redacted["headers"]["Authorization"] == "[REDACTED]"
    assert redacted["headers"]["X-Api-Key"] == "[REDACTED]"


def test_collect_secret_values_ignores_blank_and_missing_env_values() -> None:
    secrets = collect_secret_values(
        {"FMP_API_KEY": " fmp-secret ", "OPENAI_API_KEY": "", "OTHER": "value"},
        names=["FMP_API_KEY", "OPENAI_API_KEY", "MISSING"],
    )

    assert secrets == ("fmp-secret",)


def test_redact_text_handles_exception_strings() -> None:
    secret = "abc123"

    assert redact_text(RuntimeError(f"bad key {secret}"), secrets=[secret]) == (
        "bad key [REDACTED]"
    )
