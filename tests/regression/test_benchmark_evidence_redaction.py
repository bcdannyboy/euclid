from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from euclid.release import write_suite_evidence_bundle


@dataclass(frozen=True)
class _SuiteManifest:
    suite_id: str
    source_path: Path
    task_manifest_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _SurfaceStatus:
    surface_id: str
    benchmark_status: str
    replay_status: str


@dataclass(frozen=True)
class _SuiteResult:
    suite_manifest: _SuiteManifest
    summary_path: Path
    surface_statuses: tuple[_SurfaceStatus, ...]


def test_benchmark_suite_evidence_redacts_secret_like_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    secret = "p13-redaction-secret"
    monkeypatch.setenv("FMP_API_KEY", secret)

    secret_root = tmp_path / secret
    suite_path = secret_root / "current-release.yaml"
    task_path = secret_root / "planted-analytic-demo.yaml"
    summary_path = secret_root / "benchmark-suite-summary.json"
    for path in (suite_path, task_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    evidence_path = write_suite_evidence_bundle(
        suite_result=_SuiteResult(
            suite_manifest=_SuiteManifest(
                suite_id="current_release",
                source_path=suite_path,
                task_manifest_paths=(task_path,),
            ),
            summary_path=summary_path,
            surface_statuses=(
                _SurfaceStatus(
                    surface_id="retained_core_release",
                    benchmark_status="passed",
                    replay_status="passed",
                ),
            ),
        ),
        workspace_root=tmp_path,
    )

    assert evidence_path is not None
    text = evidence_path.read_text(encoding="utf-8")
    assert secret not in text
    payload = json.loads(text)
    assert "[REDACTED]" in payload["summary_path"]
    assert "[REDACTED]" in json.dumps(payload["input_manifest_digests"])
