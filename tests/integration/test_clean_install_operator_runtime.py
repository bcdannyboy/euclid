from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import euclid.release as release

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_command(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _build_wheel(dist_dir: Path) -> Path:
    dist_dir.mkdir(parents=True, exist_ok=True)
    build_result = _run_command(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
        ],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
    )
    assert build_result.returncode == 0, build_result.stderr

    wheels = sorted(dist_dir.glob("euclid-*.whl"))
    assert wheels, "expected build to produce a wheel"
    return wheels[-1]


def _prepare_offline_wheelhouse(dist_dir: Path) -> Path:
    wheel_path = _build_wheel(dist_dir)
    log_root = dist_dir.parent / "wheelhouse-logs"
    log_root.mkdir(parents=True, exist_ok=True)
    release._build_runtime_dependency_wheelhouse(  # noqa: SLF001
        checkout_root=PROJECT_ROOT,
        dist_dir=dist_dir,
        log_root=log_root,
    )
    return wheel_path


def _venv_bin(venv_dir: Path, executable: str) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / f"{executable}.exe"
    return venv_dir / "bin" / executable


def _wheel_members(wheel_path: Path) -> set[str]:
    with zipfile.ZipFile(wheel_path) as archive:
        return set(archive.namelist())


def test_clean_install_operator_runtime_runs_outside_repo_checkout(
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "dist"
    wheel_path = _prepare_offline_wheelhouse(dist_dir)
    output_root = tmp_path / "installed-operator-run"
    benchmark_root = tmp_path / "installed-benchmarks"
    venv_dir = tmp_path / "venv"
    outside_repo = tmp_path / "outside-repo"
    outside_repo.mkdir(parents=True, exist_ok=True)

    wheel_members = _wheel_members(wheel_path)
    assert "euclid/_assets/examples/current_release_run.yaml" in wheel_members
    assert "euclid/_assets/examples/full_vision_run.yaml" in wheel_members
    assert "euclid/_assets/benchmarks/suites/current-release.yaml" in wheel_members
    assert "euclid/_assets/schemas/readiness/completion-regression-policy.yaml" in (
        wheel_members
    )
    assert "euclid/_assets/schemas/readiness/shipped-releasable-v1.yaml" in (
        wheel_members
    )
    assert "euclid/_assets/src/euclid/_assets/docs/implementation/authority-snapshot.yaml" in wheel_members
    assert (
        "euclid/_assets/src/euclid/_assets/docs/implementation/certification-command-contract.yaml"
        in wheel_members
    )
    assert "euclid/_assets/output/jupyter-notebook/current_release.ipynb" in (
        wheel_members
    )
    assert (
        "euclid/_assets/fixtures/runtime/phase06/"
        "algorithmic-benchmark-publication-golden.json"
    ) in wheel_members

    clean_env = os.environ.copy()
    clean_env.pop("EUCLID_PROJECT_ROOT", None)
    clean_env.pop("PYTHONPATH", None)

    create_venv = _run_command(
        [sys.executable, "-m", "venv", str(venv_dir)],
        cwd=outside_repo,
        env=clean_env,
    )
    assert create_venv.returncode == 0, create_venv.stderr

    python_bin = _venv_bin(venv_dir, "python")

    install_wheel = _run_command(
        [
            str(python_bin),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(dist_dir),
            str(wheel_path),
        ],
        cwd=outside_repo,
        env=clean_env,
    )
    assert install_wheel.returncode == 0, install_wheel.stderr

    workflow_manifest = _run_command(
        [
            str(python_bin),
            "-c",
            (
                "import euclid; "
                "print(euclid.get_release_candidate_workflow().example_manifest)"
            ),
        ],
        cwd=outside_repo,
        env=clean_env,
    )
    assert workflow_manifest.returncode == 0, workflow_manifest.stderr
    manifest_path = Path(workflow_manifest.stdout.strip())
    assert manifest_path.is_file()
    assert manifest_path.name == "current_release_run.yaml"

    run_result = _run_command(
        [
            str(python_bin),
            "-m",
            "euclid",
            "run",
            "--config",
            str(manifest_path),
            "--output-root",
            str(output_root),
        ],
        cwd=outside_repo,
        env=clean_env,
    )
    assert run_result.returncode == 0, run_result.stderr + "\n" + run_result.stdout
    assert "Euclid run" in run_result.stdout

    replay_result = _run_command(
        [
            str(python_bin),
            "-m",
            "euclid",
            "replay",
            "--run-id",
            "current-release-run",
            "--output-root",
            str(output_root),
        ],
        cwd=outside_repo,
        env=clean_env,
    )
    assert replay_result.returncode == 0, (
        replay_result.stderr + "\n" + replay_result.stdout
    )
    assert "Replay verification: verified" in replay_result.stdout

    benchmark_result = _run_command(
        [
            str(python_bin),
            "-m",
            "euclid",
            "benchmarks",
            "run",
            "--suite",
            "current-release.yaml",
            "--benchmark-root",
            str(benchmark_root),
        ],
        cwd=outside_repo,
        env=clean_env,
    )
    assert benchmark_result.returncode == 0, (
        benchmark_result.stderr + "\n" + benchmark_result.stdout
    )
    assert "Suite: current_release" in benchmark_result.stdout
