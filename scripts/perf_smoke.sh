#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${PYTHON_BIN}" -m pytest -q "${PROJECT_ROOT}/tests/perf/test_runtime_smoke.py"
