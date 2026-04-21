from __future__ import annotations

import tomllib

import pytest

from euclid.testing.gate_manifest import (
    GateManifestError,
    extract_plan_phase_ids,
    load_gate_manifest,
)


def test_gate_manifest_requires_every_task_and_subtask_id(tmp_path) -> None:
    manifest_path = tmp_path / "P00.yaml"
    manifest_path.write_text(
        """
phase_id: P00
covered_ids:
  - P00-T06
  - P00-T06-S01
fixture_unit:
  commands:
    - pytest -q tests/unit/runtime/test_env_loading.py
fixture_integration:
  commands:
    - pytest -q tests/integration/test_live_api_gate_fixture_mode.py
fixture_regression:
  commands:
    - pytest -q tests/regression/test_live_api_evidence_redaction.py
live_api:
  commands:
    - EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 ./scripts/live_api_smoke.sh
redaction:
  assertions:
    - no_secret_in_stdout
replay:
  assertions:
    - replay_metadata_complete
claim_scope:
  assertions:
    - no_unjustified_claim_promotion
edge_cases:
  required:
    - missing_key
""",
        encoding="utf-8",
    )

    with pytest.raises(GateManifestError, match="P00-T06-S02"):
        load_gate_manifest(
            manifest_path,
            required_ids=("P00-T06", "P00-T06-S01", "P00-T06-S02"),
        )


def test_repository_p00_gate_manifest_covers_p00_t06(project_root) -> None:
    manifest = load_gate_manifest(
        project_root / "tests/gates/P00.yaml",
        required_ids=(
            "P00-T06",
            "P00-T06-S01",
            "P00-T06-S02",
            "P00-T06-S03",
            "P00-T06-S04",
            "P00-T06-S05",
            "P00-T06-S06",
            "P00-T06-S07",
            "P00-T06-S08",
            "P00-T06-S09",
            "P00-T06-S10",
            "P00-T06-S11",
            "P00-T06-S12",
            "P00-T06-S13",
            "P00-T06-S14",
        ),
    )

    assert manifest.phase_id == "P00"
    assert manifest.live_api.commands
    assert manifest.fixture_unit.commands
    assert manifest.fixture_integration.commands
    assert manifest.fixture_regression.commands


def test_repository_phase_gate_manifests_cover_all_plan_ids(project_root) -> None:
    plan_path = project_root / "docs/plans/2026-04-21-euclid-enhancement-master-plan.md"

    for phase_id in ("P00", "P01"):
        required_ids = extract_plan_phase_ids(plan_path, phase_id=phase_id)
        manifest = load_gate_manifest(
            project_root / "tests" / "gates" / f"{phase_id}.yaml",
            required_ids=required_ids,
        )

        assert set(required_ids) <= set(manifest.covered_ids)


def test_pytest_timeout_default_is_configured(project_root) -> None:
    payload = tomllib.loads((project_root / "pyproject.toml").read_text("utf-8"))

    assert payload["tool"]["pytest"]["ini_options"]["timeout"] == 120
