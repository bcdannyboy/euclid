#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
export PYTHONPATH=src
"${PYTHON_BIN}" -m pytest -q tests/unit tests/integration tests/golden tests/fixtures
