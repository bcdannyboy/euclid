from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from euclid.operator_runtime.replay import replay_operator
from euclid.operator_runtime.resources import resolve_example_path
from euclid.operator_runtime.run import run_operator

CURRENT_RELEASE_MANIFEST = resolve_example_path("current_release_run.yaml")
FULL_VISION_MANIFEST = resolve_example_path("full_vision_run.yaml")


def test_operator_replay_no_longer_depends_on_demo(tmp_path: Path) -> None:
    output_root = tmp_path / "operator-run"
    run_operator(manifest_path=CURRENT_RELEASE_MANIFEST, output_root=output_root)

    replay = replay_operator(output_root=output_root, run_id="current-release-run")

    assert replay.summary.replay_verification_status == "verified"
    assert replay.summary.forecast_object_type == "point"


def test_operator_replay_verifies_sealed_bundle(tmp_path: Path) -> None:
    output_root = tmp_path / "operator-run"
    run_operator(manifest_path=FULL_VISION_MANIFEST, output_root=output_root)

    replay = replay_operator(output_root=output_root, run_id="full-vision-run")

    assert replay.summary.replay_verification_status == "verified"
    assert replay.summary.forecast_object_type == "distribution"


def test_operator_replay_rejects_wrong_bundle_ref(tmp_path: Path) -> None:
    output_root = tmp_path / "operator-run"
    run_operator(manifest_path=CURRENT_RELEASE_MANIFEST, output_root=output_root)

    with pytest.raises(KeyError):
        replay_operator(
            output_root=output_root,
            run_id="current-release-run",
            bundle_ref="reproducibility_bundle_manifest@1.0.0:not-the-bundle",
        )


def test_operator_replay_reports_hash_mismatch_as_failed_evidence(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "operator-run"
    run_operator(manifest_path=CURRENT_RELEASE_MANIFEST, output_root=output_root)
    registry_path = (
        output_root / "sealed-runs" / "current-release-run" / "registry.sqlite3"
    )
    with sqlite3.connect(registry_path) as connection:
        artifact_path = Path(
            connection.execute(
                (
                    "select artifact_path from manifests "
                    "where schema_name = 'scorecard_manifest@1.1.0' limit 1"
                )
            ).fetchone()[0]
        )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["body"]["predictive_reason_codes"] = ["tampered_replay_canary"]
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    replay = replay_operator(output_root=output_root, run_id="current-release-run")

    assert replay.summary.replay_verification_status == "failed"
    assert "artifact_hash_mismatch" in replay.summary.failure_reason_codes
