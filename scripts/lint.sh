#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"${PYTHON_BIN}" -m ruff check src/euclid tests/conftest.py tests/unit tests/integration tests/benchmarks tests/perf
