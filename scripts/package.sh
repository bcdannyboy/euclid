#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${DIST_DIR:-${PROJECT_ROOT}/dist}"
PACKAGE_LOG_ROOT="${PACKAGE_LOG_ROOT:-${DIST_DIR}/logs}"
PACKAGE_EVIDENCE_PATH="${PACKAGE_EVIDENCE_PATH:-${DIST_DIR}/package-wheelhouse-evidence.json}"

mkdir -p "${DIST_DIR}" "${PACKAGE_LOG_ROOT}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m build --sdist --wheel --outdir "${DIST_DIR}"

PROJECT_ROOT="${PROJECT_ROOT}" \
DIST_DIR="${DIST_DIR}" \
PACKAGE_LOG_ROOT="${PACKAGE_LOG_ROOT}" \
PACKAGE_EVIDENCE_PATH="${PACKAGE_EVIDENCE_PATH}" \
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import euclid.release as release

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
dist_dir = Path(os.environ["DIST_DIR"]).resolve()
log_root = Path(os.environ["PACKAGE_LOG_ROOT"]).resolve()
evidence_path = Path(os.environ["PACKAGE_EVIDENCE_PATH"]).resolve()

runtime_distributions = release._build_runtime_dependency_wheelhouse(
    checkout_root=project_root,
    dist_dir=dist_dir,
    log_root=log_root,
)
project_wheels = sorted(path.name for path in dist_dir.glob("euclid-*.whl"))
dependency_wheels = sorted(
    path.name for path in dist_dir.glob("*.whl") if path.name not in set(project_wheels)
)
payload = {
    "report_id": "euclid_package_wheelhouse_evidence_v1",
    "producer_script": "scripts/package.sh",
    "dist_dir": str(dist_dir),
    "project_wheels": project_wheels,
    "runtime_dependency_distributions": list(runtime_distributions),
    "runtime_dependency_wheel_count": len(dependency_wheels),
    "runtime_dependency_wheels": dependency_wheels,
    "wheelhouse_digest": f"runtime_directory_digest:{release._directory_digest(dist_dir)}",
}
evidence_path.parent.mkdir(parents=True, exist_ok=True)
evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
