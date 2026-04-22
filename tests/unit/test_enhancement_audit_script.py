from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_audit_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "audit_euclid_enhancement.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_euclid_enhancement",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_plan_ids_groups_tasks_and_subtasks_by_phase() -> None:
    audit = _load_audit_module()
    text = """
### P00-T01: Task
- `P00-T01-S01`: Subtask
- `P00-T01-S02`: Subtask
### P01-T03: Task
- `P01-T03-S04`: Subtask
"""

    grouped = audit.extract_plan_ids(text)

    assert grouped["P00"] == {"P00-T01", "P00-T01-S01", "P00-T01-S02"}
    assert grouped["P01"] == {"P01-T03", "P01-T03-S04"}


def test_audit_reports_missing_gate_manifest(tmp_path: Path) -> None:
    audit = _load_audit_module()
    (tmp_path / "docs/plans").mkdir(parents=True)
    (tmp_path / "docs/plans/2026-04-21-euclid-enhancement-master-plan.md").write_text(
        "### P00-T01: Task\n- `P00-T01-S01`: Subtask\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.example").write_text(
        "\n".join(
            [
                "EUCLID_LIVE_API_TESTS=",
                "EUCLID_LIVE_API_STRICT=",
                "FMP_API_KEY=",
                "OPENAI_API_KEY=",
                "EUCLID_OPENAI_EXPLAINER_MODEL=",
                "EUCLID_LIVE_TEST_TIMEOUT_SECONDS=",
                "EUCLID_LIVE_ARTIFACT_DIR=",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies=[]\n[project.optional-dependencies]\ndev=[]\n",
        encoding="utf-8",
    )

    report = audit.run_audit(tmp_path, run_fixtures=False, run_live=False)

    assert any(
        finding.check_id == "gate_manifest.exists" and finding.status == "failed"
        for finding in report.findings
    )
    assert report.failed_count > 0
