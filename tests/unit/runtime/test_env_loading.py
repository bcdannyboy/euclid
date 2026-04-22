from __future__ import annotations

import pytest

from euclid.runtime.env import (
    EnvConfigurationError,
    EuclidEnv,
    MissingLiveApiCredentialError,
)


def test_load_env_file_overlays_process_environment_without_exposing_values(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "FMP_API_KEY=file-fmp-secret",
                "OPENAI_API_KEY=file-openai-secret",
                "EUCLID_LIVE_API_TESTS=yes",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FMP_API_KEY", "process-fmp-secret")

    loaded = EuclidEnv.load(env_file=env_path)

    assert loaded.get("FMP_API_KEY") == "process-fmp-secret"
    assert loaded.get("OPENAI_API_KEY") == "file-openai-secret"
    assert loaded.live_tests_enabled is True
    assert loaded.presence_metadata(["FMP_API_KEY", "OPENAI_API_KEY"]) == {
        "FMP_API_KEY": {"present": True, "source": "environment"},
        "OPENAI_API_KEY": {"present": True, "source": ".env"},
    }
    assert "process-fmp-secret" not in repr(loaded)
    assert "file-openai-secret" not in repr(loaded)


def test_require_live_credentials_reports_names_only(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("EUCLID_LIVE_API_STRICT=1\n", encoding="utf-8")

    loaded = EuclidEnv.load(env_file=env_path, environ={})

    with pytest.raises(MissingLiveApiCredentialError) as exc_info:
        loaded.require(["FMP_API_KEY", "OPENAI_API_KEY"])

    message = str(exc_info.value)
    assert "FMP_API_KEY" in message
    assert "OPENAI_API_KEY" in message
    assert "secret" not in message.lower()


def test_blank_env_values_are_treated_as_missing_without_leaking_values(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "EUCLID_LIVE_API_TESTS= ",
                "EUCLID_LIVE_API_STRICT=1",
                "FMP_API_KEY=   ",
                "OPENAI_API_KEY=",
            ]
        ),
        encoding="utf-8",
    )

    loaded = EuclidEnv.load(env_file=env_path, environ={})

    assert loaded.live_tests_enabled is False
    assert loaded.strict_live_api is True
    assert loaded.presence_metadata(["FMP_API_KEY", "OPENAI_API_KEY"]) == {
        "FMP_API_KEY": {"present": False, "source": None},
        "OPENAI_API_KEY": {"present": False, "source": None},
    }
    with pytest.raises(MissingLiveApiCredentialError) as exc_info:
        loaded.require(["FMP_API_KEY", "OPENAI_API_KEY"])
    assert "FMP_API_KEY" in str(exc_info.value)
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_invalid_live_flag_is_typed_configuration_error(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("EUCLID_LIVE_API_TESTS=maybe\n", encoding="utf-8")

    with pytest.raises(EnvConfigurationError, match="EUCLID_LIVE_API_TESTS"):
        _ = EuclidEnv.load(env_file=env_path, environ={}).live_tests_enabled
