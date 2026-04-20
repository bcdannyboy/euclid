from __future__ import annotations

import os
import subprocess
import sys
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


def test_release_status_runs_without_repo_checkout(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    wheel_path = _prepare_offline_wheelhouse(dist_dir)
    venv_dir = tmp_path / "venv"
    outside_repo = tmp_path / "outside-repo"
    outside_repo.mkdir(parents=True, exist_ok=True)

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

    status_result = _run_command(
        [str(python_bin), "-m", "euclid", "release", "status"],
        cwd=outside_repo,
        env=clean_env,
    )
    assert status_result.returncode == 0, (
        status_result.stderr + "\n" + status_result.stdout
    )
    assert "Euclid release status" in status_result.stdout
    assert "Current release verdict:" in status_result.stdout
    assert "Full vision verdict:" in status_result.stdout
    assert "Shipped or releasable verdict:" in status_result.stdout
    assert "determinism.same_seed" in status_result.stdout
    assert str(PROJECT_ROOT) not in status_result.stdout


def test_benchmark_suite_executes_from_installed_assets(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    wheel_path = _prepare_offline_wheelhouse(dist_dir)
    venv_dir = tmp_path / "venv"
    outside_repo = tmp_path / "outside-repo"
    outside_repo.mkdir(parents=True, exist_ok=True)

    clean_env = os.environ.copy()
    clean_env.pop("EUCLID_PROJECT_ROOT", None)
    clean_env.pop("PYTHONPATH", None)

    assert _run_command(
        [sys.executable, "-m", "venv", str(venv_dir)],
        cwd=outside_repo,
        env=clean_env,
    ).returncode == 0

    python_bin = _venv_bin(venv_dir, "python")
    install = _run_command(
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
    assert install.returncode == 0, install.stderr

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
            str(tmp_path / "benchmarks"),
        ],
        cwd=outside_repo,
        env=clean_env,
    )

    assert benchmark_result.returncode == 0, (
        benchmark_result.stderr + "\n" + benchmark_result.stdout
    )
    assert "Euclid benchmark suite" in benchmark_result.stdout
    assert "Suite: current_release" in benchmark_result.stdout
