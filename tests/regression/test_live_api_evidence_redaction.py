from __future__ import annotations

import json

from euclid.runtime.env import EuclidEnv
from euclid.testing.live_api import LiveApiGate


def test_live_api_evidence_redacts_secret_like_values_in_regression_artifact(
    tmp_path,
) -> None:
    fake_fmp_key = "fmp-regression-secret"
    fake_openai_key = "sk-regression-secret"
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "EUCLID_LIVE_API_TESTS=1",
                f"FMP_API_KEY={fake_fmp_key}",
                f"OPENAI_API_KEY={fake_openai_key}",
            ]
        ),
        encoding="utf-8",
    )
    env = EuclidEnv.load(env_file=env_path, environ={})

    gate = LiveApiGate(
        gate_id="P00-T06-regression-redaction",
        provider="combined",
        endpoint_class="redaction-regression",
        required_env=("FMP_API_KEY", "OPENAI_API_KEY"),
        evidence_path=tmp_path / "evidence.json",
        check=lambda _credentials: {
            "url": f"https://example.test?apikey={fake_fmp_key}",
            "headers": {"Authorization": f"Bearer {fake_openai_key}"},
            "message": f"{fake_fmp_key} {fake_openai_key}",
        },
    )

    assert gate.run(env).status == "passed"

    evidence_text = (tmp_path / "evidence.json").read_text(encoding="utf-8")
    assert fake_fmp_key not in evidence_text
    assert fake_openai_key not in evidence_text
    payload = json.loads(evidence_text)
    assert payload["credential_presence"]["FMP_API_KEY"]["present"] is True
    assert payload["credential_presence"]["OPENAI_API_KEY"]["present"] is True
