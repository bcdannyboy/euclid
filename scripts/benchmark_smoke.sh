#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${PROJECT_ROOT}/build/benchmark-smoke}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m euclid benchmarks run \
  --suite "${PROJECT_ROOT}/benchmarks/suites/mvp.yaml" \
  --benchmark-root "${BENCHMARK_ROOT}"
