#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

"${PYTHON_BIN}" -m build --sdist --wheel
