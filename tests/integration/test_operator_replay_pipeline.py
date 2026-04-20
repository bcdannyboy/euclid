from __future__ import annotations

from pathlib import Path

import pytest

from euclid.operator_runtime.replay import replay_operator
from euclid.operator_runtime.run import run_operator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_RELEASE_MANIFEST = PROJECT_ROOT / "examples" / "current_release_run.yaml"
FULL_VISION_MANIFEST = PROJECT_ROOT / "examples" / "full_vision_run.yaml"


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
