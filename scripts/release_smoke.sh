#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src"

"${PYTHON_BIN}" -m euclid release repo-test-matrix --project-root "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m euclid benchmarks run --suite "current-release.yaml" --no-resume --benchmark-root "${PROJECT_ROOT}/build/certification/current_release_suite" --project-root "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m euclid benchmarks run --suite "full-vision.yaml" --no-resume --benchmark-root "${PROJECT_ROOT}/build/certification/full_vision_suite" --project-root "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m euclid run --config "${PROJECT_ROOT}/examples/full_vision_run.yaml" --output-root "${PROJECT_ROOT}/build/certification/full_vision_run" --evidence-report "${PROJECT_ROOT}/build/reports/full_vision_operator_run_evidence.json"
"${PYTHON_BIN}" -m euclid replay --run-id "full-vision-run" --output-root "${PROJECT_ROOT}/build/certification/full_vision_run" --evidence-report "${PROJECT_ROOT}/build/reports/full_vision_operator_replay_evidence.json"
"${PYTHON_BIN}" -m euclid release certify-clean-install --project-root "${PROJECT_ROOT}" --wheel-dir "${PROJECT_ROOT}/build/certification/wheels" --output-root "${PROJECT_ROOT}/build/certification/clean_install"
"${PYTHON_BIN}" -m euclid release status --project-root "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m euclid release verify-completion --project-root "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m euclid release certify-research-readiness --project-root "${PROJECT_ROOT}"
