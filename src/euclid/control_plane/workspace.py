from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunWorkspacePaths:
    workspace_root: Path
    run_id: str
    active_runs_root: Path
    sealed_runs_root: Path
    cache_root: Path
    temp_root: Path
    active_run_root: Path
    control_plane_root: Path
    active_work_root: Path
    lock_root: Path
    run_lock_path: Path
    control_plane_store_path: Path
    sealed_run_root: Path
    artifact_root: Path
    metadata_store_path: Path
    run_summary_path: Path


class RuntimeWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root

    def paths_for_run(self, run_id: str) -> RunWorkspacePaths:
        active_runs_root = self.root / "active-runs"
        sealed_runs_root = self.root / "sealed-runs"
        cache_root = self.root / "caches"
        temp_root = self.root / "tmp"
        active_run_root = active_runs_root / run_id
        control_plane_root = active_run_root / "control-plane"
        active_work_root = active_run_root / "work"
        lock_root = active_run_root / "locks"
        sealed_run_root = sealed_runs_root / run_id
        return RunWorkspacePaths(
            workspace_root=self.root,
            run_id=run_id,
            active_runs_root=active_runs_root,
            sealed_runs_root=sealed_runs_root,
            cache_root=cache_root,
            temp_root=temp_root,
            active_run_root=active_run_root,
            control_plane_root=control_plane_root,
            active_work_root=active_work_root,
            lock_root=lock_root,
            run_lock_path=lock_root / "run.lock",
            control_plane_store_path=control_plane_root / "execution-state.sqlite3",
            sealed_run_root=sealed_run_root,
            artifact_root=sealed_run_root / "artifacts",
            metadata_store_path=sealed_run_root / "registry.sqlite3",
            run_summary_path=sealed_run_root / "run-summary.json",
        )

    def materialize(self, paths: RunWorkspacePaths) -> None:
        for directory in (
            paths.workspace_root,
            paths.active_runs_root,
            paths.sealed_runs_root,
            paths.cache_root,
            paths.temp_root,
            paths.active_run_root,
            paths.control_plane_root,
            paths.active_work_root,
            paths.lock_root,
            paths.sealed_run_root,
            paths.artifact_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)


__all__ = ["RunWorkspacePaths", "RuntimeWorkspace"]
