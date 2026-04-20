#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
export PYTHONPATH=src
MANIFEST_PATH="${MANIFEST_PATH:-fixtures/runtime/prototype-demo.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-build/demo-script}"
"${PYTHON_BIN}" -m euclid demo run --manifest "${MANIFEST_PATH}" --output-root "${OUTPUT_ROOT}"
"${PYTHON_BIN}" -m euclid demo replay --output-root "${OUTPUT_ROOT}"
