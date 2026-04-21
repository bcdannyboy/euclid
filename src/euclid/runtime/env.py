from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:  # pragma: no cover - exercised when dependency is unavailable.
    from dotenv import dotenv_values
except Exception:  # pragma: no cover
    dotenv_values = None

LIVE_TESTS_ENV_VAR = "EUCLID_LIVE_API_TESTS"
LIVE_STRICT_ENV_VAR = "EUCLID_LIVE_API_STRICT"
LIVE_ARTIFACT_DIR_ENV_VAR = "EUCLID_LIVE_ARTIFACT_DIR"
LIVE_TEST_TIMEOUT_ENV_VAR = "EUCLID_LIVE_TEST_TIMEOUT_SECONDS"
FMP_API_KEY_ENV_VAR = "FMP_API_KEY"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
OPENAI_EXPLAINER_MODEL_ENV_VAR = "EUCLID_OPENAI_EXPLAINER_MODEL"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})


class EnvConfigurationError(ValueError):
    """Raised when Euclid live-test environment configuration is invalid."""


class MissingLiveApiCredentialError(EnvConfigurationError):
    """Raised when a required live API credential is missing."""

    def __init__(self, missing_names: Sequence[str]) -> None:
        self.missing_names = tuple(missing_names)
        joined = ", ".join(self.missing_names)
        super().__init__(f"Missing required live API credential variables: {joined}")


@dataclass(frozen=True)
class EuclidEnv:
    """Secret-safe access to Euclid environment and `.env` values."""

    values: Mapping[str, str]
    sources: Mapping[str, str]
    env_file: Path | None = None

    @classmethod
    def load(
        cls,
        *,
        project_root: Path | None = None,
        env_file: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "EuclidEnv":
        resolved_env_file = _resolve_env_file(
            project_root=project_root,
            env_file=env_file,
        )
        file_values = _read_env_file(resolved_env_file)
        process_values = os.environ if environ is None else environ

        values: dict[str, str] = {}
        sources: dict[str, str] = {}
        for key, raw_value in file_values.items():
            value = str(raw_value or "").strip()
            if value:
                values[key] = value
                sources[key] = ".env"
        for key, raw_value in process_values.items():
            value = str(raw_value or "").strip()
            if value:
                values[key] = value
                sources[key] = "environment"
            elif key in values and key in process_values:
                values.pop(key, None)
                sources.pop(key, None)
        return cls(
            values=values,
            sources=sources,
            env_file=resolved_env_file if resolved_env_file.is_file() else None,
        )

    def __repr__(self) -> str:
        names = ", ".join(sorted(self.values))
        return f"EuclidEnv(keys=[{names}], env_file={self.env_file!s})"

    def get(self, name: str, default: str = "") -> str:
        return self.values.get(name, default)

    def require(self, names: Sequence[str]) -> dict[str, str]:
        missing = [name for name in names if not self.get(name)]
        if missing:
            raise MissingLiveApiCredentialError(missing)
        return {name: self.get(name) for name in names}

    def presence_metadata(
        self,
        names: Sequence[str],
    ) -> dict[str, dict[str, str | bool | None]]:
        return {
            name: {
                "present": bool(self.get(name)),
                "source": self.sources.get(name),
            }
            for name in names
        }

    @property
    def live_tests_enabled(self) -> bool:
        return self.flag(LIVE_TESTS_ENV_VAR)

    @property
    def strict_live_api(self) -> bool:
        return self.flag(LIVE_STRICT_ENV_VAR)

    def flag(self, name: str) -> bool:
        raw_value = self.get(name).strip().lower()
        if raw_value in _TRUE_VALUES:
            return True
        if raw_value in _FALSE_VALUES:
            return False
        raise EnvConfigurationError(
            f"{name} must be one of 1/true/yes/on or 0/false/no/off"
        )


def _resolve_env_file(*, project_root: Path | None, env_file: Path | None) -> Path:
    if env_file is not None:
        return env_file.expanduser().resolve()
    root = project_root.expanduser().resolve() if project_root else Path.cwd()
    return root / ".env"


def _read_env_file(env_file: Path) -> dict[str, str]:
    if not env_file.is_file():
        return {}
    if dotenv_values is not None:
        return {
            str(key): str(value or "")
            for key, value in dotenv_values(env_file).items()
            if key is not None
        }
    return _read_env_file_fallback(env_file)


def _read_env_file_fallback(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


__all__ = [
    "EnvConfigurationError",
    "EuclidEnv",
    "FMP_API_KEY_ENV_VAR",
    "LIVE_ARTIFACT_DIR_ENV_VAR",
    "LIVE_STRICT_ENV_VAR",
    "LIVE_TESTS_ENV_VAR",
    "LIVE_TEST_TIMEOUT_ENV_VAR",
    "MissingLiveApiCredentialError",
    "OPENAI_API_KEY_ENV_VAR",
    "OPENAI_EXPLAINER_MODEL_ENV_VAR",
]
