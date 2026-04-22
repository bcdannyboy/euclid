from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from euclid.cli import app

RUNNER = CliRunner()


def _load_report(project_root: Path) -> dict[str, object]:
    report_path = project_root / "build" / "reports" / "completion-report.json"
    assert report_path.is_file(), f"missing completion report at {report_path}"
    return json.loads(report_path.read_text(encoding="utf-8"))


@pytest.mark.timeout(600)
def test_release_status_emits_closure_metadata_and_scope_evidence_bundles(
    project_root: Path,
) -> None:
    clean_install = RUNNER.invoke(
        app,
        ["release", "certify-clean-install", "--project-root", str(project_root)],
    )
    assert clean_install.exit_code == 0, clean_install.stdout

    result = RUNNER.invoke(
        app,
        ["release", "status", "--project-root", str(project_root)],
    )

    assert result.exit_code == 0, result.stdout
    payload = _load_report(project_root)

    assert payload["authority_snapshot_id"] == "euclid-authority-2026-04-15-b"
    assert payload["command_contract_id"] == "euclid-certification-commands-v1"
    assert payload["closure_map_id"] == "euclid-closure-map-2026-04-15-v1"
    assert payload["traceability_id"] == "euclid-traceability-2026-04-15-v1"
    assert payload["fixture_spec_id"] == "euclid-certification-fixtures-v1"

    bundles = payload["scope_evidence_bundles"]
    bundle_ids = {bundle["scope_id"]: bundle["evidence_bundle_id"] for bundle in bundles}
    assert {"current_release", "full_vision", "shipped_releasable"} <= set(bundle_ids)
    assert len(set(bundle_ids.values())) == len(bundle_ids)

    verdict_ids = {entry["policy_id"] for entry in payload["policy_verdicts"]}
    assert {
        "current_release_v1",
        "full_vision_v1",
        "shipped_releasable_v1",
    } <= verdict_ids


@pytest.mark.timeout(600)
def test_full_vision_only_rows_do_not_close_from_current_release_bundle(
    project_root: Path,
) -> None:
    clean_install = RUNNER.invoke(
        app,
        ["release", "certify-clean-install", "--project-root", str(project_root)],
    )
    assert clean_install.exit_code == 0, clean_install.stdout

    result = RUNNER.invoke(
        app,
        ["release", "status", "--project-root", str(project_root)],
    )
    assert result.exit_code == 0, result.stdout
    payload = _load_report(project_root)
    bundles = {
        bundle["scope_id"]: bundle["evidence_bundle_id"]
        for bundle in payload["scope_evidence_bundles"]
    }
    distribution_row = next(
        row
        for row in payload["capability_rows"]
        if row["row_id"] == "forecast_object_type:distribution"
    )

    assert bundles["full_vision"] in distribution_row["evidence_bundle_ids"]
    assert bundles["current_release"] not in distribution_row["evidence_bundle_ids"]
