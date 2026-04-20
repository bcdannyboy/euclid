#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(pwd)}"
SUITE_NAME="${SUITE_NAME:-current-release.yaml}"
mkdir -p "${WORKSPACE_ROOT}/build"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(mktemp -d "${WORKSPACE_ROOT}/build/benchmark-suite.XXXXXX")}"

"${PYTHON_BIN}" -m euclid benchmarks run \
  --suite "${SUITE_NAME}" \
  --no-resume \
  --benchmark-root "${BENCHMARK_ROOT}"
