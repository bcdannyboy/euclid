from __future__ import annotations

from pathlib import Path

from euclid.control_plane import RuntimeWorkspace


def test_runtime_workspace_materializes_active_sealed_cache_and_temp_roots(
    tmp_path: Path,
) -> None:
    workspace = RuntimeWorkspace(tmp_path / "runtime")

    paths = workspace.paths_for_run("demo_run")
    workspace.materialize(paths)

    assert paths.active_run_root == tmp_path / "runtime" / "active-runs" / "demo_run"
    assert paths.sealed_run_root == (
        tmp_path / "runtime" / "sealed-runs" / "demo_run"
    )
    assert paths.control_plane_root.is_dir()
    assert paths.active_work_root.is_dir()
    assert paths.lock_root.is_dir()
    assert paths.artifact_root.is_dir()
    assert paths.cache_root.is_dir()
    assert paths.temp_root.is_dir()

