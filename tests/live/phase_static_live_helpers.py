from __future__ import annotations

from pathlib import Path

from euclid.testing.gate_manifest import load_gate_manifest
from euclid.testing.live_api import NON_CLAIM_EVIDENCE_BOUNDARY

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def assert_static_live_phase_gate(phase_id: str, test_file: str) -> None:
    manifest = load_gate_manifest(
        PROJECT_ROOT / "tests" / "gates" / f"{phase_id}.yaml",
        project_root=PROJECT_ROOT,
        validate_references=True,
    )
    command_text = "\n".join(manifest.live_api.commands)
    assert test_file in command_text
    assert manifest.phase_id == phase_id
    assert manifest.covered_ids
    assert manifest.fixture_unit.commands
    assert manifest.fixture_integration.commands
    assert manifest.fixture_regression.commands
    assert NON_CLAIM_EVIDENCE_BOUNDARY["counts_as_scientific_claim_evidence"] is False
