from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from euclid.cli import app

RUNNER = CliRunner()


def _completion_report(project_root: Path) -> dict[str, object]:
    report_path = project_root / "build" / "reports" / "completion-report.json"
    assert report_path.is_file()
    return json.loads(report_path.read_text(encoding="utf-8"))


@pytest.mark.timeout(600)
def test_shipped_releasable_is_not_alias_of_current_release(
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

    payload = _completion_report(project_root)
    bundles = {
        bundle["scope_id"]: bundle["evidence_bundle_id"]
        for bundle in payload["scope_evidence_bundles"]
    }
    assert bundles["shipped_releasable"] != bundles["current_release"]

    verdicts = {
        entry["policy_id"]: entry
        for entry in payload["policy_verdicts"]
    }
    assert verdicts["shipped_releasable_v1"]["policy_id"] != (
        verdicts["current_release_v1"]["policy_id"]
    )
    assert verdicts["shipped_releasable_v1"]["evidence_refs"] != (
        verdicts["current_release_v1"]["evidence_refs"]
    )


@pytest.mark.timeout(600)
def test_shipped_releasable_requires_packaging_install_evidence(
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

    payload = _completion_report(project_root)
    readiness_lane = next(
        row
        for row in payload["capability_rows"]
        if row["row_id"] == "evidence_lane:readiness_and_closure"
    )
    assert "packaging_install" in readiness_lane["available_evidence_classes"]
    assert any(
        "clean-install" in ref for ref in readiness_lane["evidence_refs"]
    )
