from __future__ import annotations

from pathlib import Path

import yaml

import euclid
from euclid.runtime.profiling import (
    SEED_SENSITIVE_ARTIFACT_ROLES,
    SEED_SENSITIVE_SEED_SCOPES,
    capture_demo_runtime_snapshot,
    compare_runtime_determinism,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_MANIFEST = PROJECT_ROOT / "fixtures/runtime/prototype-demo.yaml"


def _write_seeded_manifest(tmp_path: Path, *, seed: str) -> Path:
    payload = yaml.safe_load(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    search_payload = dict(payload.get("search", {}))
    search_payload["seed"] = seed
    payload["search"] = search_payload
    payload["dataset_csv"] = str(
        (SAMPLE_MANIFEST.parent / str(payload["dataset_csv"])).resolve()
    )
    manifest_path = tmp_path / f"prototype-demo-seed-{seed}.yaml"
    manifest_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path


def test_same_seed_runs_produce_identical_replay_bundle_surfaces(
    tmp_path: Path,
) -> None:
    manifest_path = _write_seeded_manifest(tmp_path, seed="0")
    first_run = euclid.run_demo(
        manifest_path=manifest_path,
        output_root=tmp_path / "seed-0-first",
    )
    second_run = euclid.run_demo(
        manifest_path=manifest_path,
        output_root=tmp_path / "seed-0-second",
    )

    first_snapshot = capture_demo_runtime_snapshot(
        output_root=first_run.paths.output_root
    )
    second_snapshot = capture_demo_runtime_snapshot(
        output_root=second_run.paths.output_root
    )
    comparison = compare_runtime_determinism(first_snapshot, second_snapshot)

    assert comparison.identical
    assert comparison.changed_artifact_roles == ()
    assert comparison.changed_seed_scopes == ()
    assert comparison.unexpected_artifact_roles == ()
    assert comparison.unexpected_seed_scopes == ()


def test_different_seed_runs_only_change_declared_stochastic_surfaces(
    tmp_path: Path,
) -> None:
    zero_manifest_path = _write_seeded_manifest(tmp_path, seed="0")
    one_manifest_path = _write_seeded_manifest(tmp_path, seed="1")
    zero_run = euclid.run_demo(
        manifest_path=zero_manifest_path,
        output_root=tmp_path / "seed-0",
    )
    one_run = euclid.run_demo(
        manifest_path=one_manifest_path,
        output_root=tmp_path / "seed-1",
    )

    zero_snapshot = capture_demo_runtime_snapshot(
        output_root=zero_run.paths.output_root
    )
    one_snapshot = capture_demo_runtime_snapshot(output_root=one_run.paths.output_root)
    comparison = compare_runtime_determinism(
        zero_snapshot,
        one_snapshot,
        stochastic_artifact_roles=SEED_SENSITIVE_ARTIFACT_ROLES,
        stochastic_seed_scopes=SEED_SENSITIVE_SEED_SCOPES,
    )

    assert zero_snapshot.seed_records["search"] == "0"
    assert one_snapshot.seed_records["search"] == "1"
    assert (
        zero_snapshot.selected_family
        == one_snapshot.selected_family
        == "seasonal_naive"
    )
    assert zero_snapshot.run_result_ref == one_snapshot.run_result_ref
    assert "search_plan" in comparison.changed_artifact_roles
    assert set(comparison.changed_artifact_roles) <= set(
        SEED_SENSITIVE_ARTIFACT_ROLES
    )
    assert set(comparison.changed_seed_scopes) <= set(SEED_SENSITIVE_SEED_SCOPES)
    assert comparison.unexpected_artifact_roles == ()
    assert comparison.unexpected_seed_scopes == ()
