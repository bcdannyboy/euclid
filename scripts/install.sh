#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install -e ".[dev]"
