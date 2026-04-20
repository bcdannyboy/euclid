#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${DIST_DIR:-${PROJECT_ROOT}/dist}"
SMOKE_ROOT="${SMOKE_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/euclid-install-smoke.XXXXXX")}"
VENV_DIR="${VENV_DIR:-${SMOKE_ROOT}/venv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SMOKE_ROOT}/demo-output}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SMOKE_ROOT}/benchmarks}"
WHEEL_PATH="${WHEEL_PATH:-$(ls -1t "${DIST_DIR}"/euclid-*.whl | head -n 1)}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install "${WHEEL_PATH}"

WORKFLOW_INFO="$(
  python - <<'PY'
from __future__ import annotations

from pathlib import Path

import yaml

import euclid

workflow = euclid.get_release_candidate_workflow()
payload = yaml.safe_load(Path(workflow.example_manifest).read_text(encoding="utf-8"))

print(workflow.example_manifest)
print(payload["request_id"])
PY
)"
WORKFLOW_MANIFEST="$(printf '%s\n' "${WORKFLOW_INFO}" | sed -n '1p')"
WORKFLOW_RUN_ID="$(printf '%s\n' "${WORKFLOW_INFO}" | sed -n '2p')"

euclid smoke
euclid release status
euclid benchmarks run --suite "current-release.yaml" --benchmark-root "${BENCHMARK_ROOT}"
euclid run --config "${WORKFLOW_MANIFEST}" --output-root "${OUTPUT_ROOT}"
euclid replay --run-id "${WORKFLOW_RUN_ID}" --output-root "${OUTPUT_ROOT}"
euclid demo validate --output-root "${OUTPUT_ROOT}"
euclid release validate-contracts

python - <<'PY'
from __future__ import annotations

from pathlib import Path

import euclid
import yaml

workflow = euclid.get_release_candidate_workflow()
status = euclid.get_release_status()
validation = euclid.validate_release_contracts()
workflow_payload = yaml.safe_load(
    Path(workflow.example_manifest).read_text(encoding="utf-8")
)

assert status.current_version == euclid.__version__
assert validation.schema_count > 0
assert workflow.example_manifest.is_file()
assert workflow.benchmark_suite.is_file()
assert workflow.notebook_path.is_file()
assert workflow_payload["request_id"] == "current-release-run"

print("Euclid API install smoke passed")
PY
