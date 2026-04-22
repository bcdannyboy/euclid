#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" - <<'PY'
from euclid.runtime.env import (
    FMP_API_KEY_ENV_VAR,
    LIVE_ARTIFACT_DIR_ENV_VAR,
    OPENAI_API_KEY_ENV_VAR,
    EuclidEnv,
)

env = EuclidEnv.load()
if env.live_tests_enabled and env.strict_live_api:
    env.require([FMP_API_KEY_ENV_VAR, OPENAI_API_KEY_ENV_VAR])
artifact_dir = env.get(LIVE_ARTIFACT_DIR_ENV_VAR)
if artifact_dir:
    from pathlib import Path

    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
print("Euclid live API preflight complete: secrets redacted by policy")
PY

"${PYTHON_BIN}" -m pytest -q tests/live
