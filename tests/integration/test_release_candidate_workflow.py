from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

import euclid
from euclid.cli import app
from euclid.operator_runtime.resources import EuclidAssetError, resolve_example_path


RUNNER = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_RELEASE_EXAMPLE = PROJECT_ROOT / "examples" / "current_release_run.yaml"
FULL_VISION_EXAMPLE = PROJECT_ROOT / "examples" / "full_vision_run.yaml"


def test_current_release_example_matches_current_scope() -> None:
    workflow = euclid.get_release_candidate_workflow(project_root=PROJECT_ROOT)
    payload = yaml.safe_load(workflow.example_manifest.read_text(encoding="utf-8"))
    checkout_payload = yaml.safe_load(CURRENT_RELEASE_EXAMPLE.read_text(encoding="utf-8"))

    assert workflow.example_manifest.name == "current_release_run.yaml"
    assert "src/euclid/_assets/examples" in workflow.example_manifest.as_posix()
    assert payload == checkout_payload
    assert payload["workflow_id"] == "euclid_current_release_candidate"
    assert payload["request_id"] == "current-release-run"
    assert payload["forecast_object_type"] == "point"
    assert payload["benchmark_suite"] == "benchmarks/suites/current-release.yaml"
    assert payload["replay_run_id"] == "current-release-run"
    assert payload["notebook_path"] == "output/jupyter-notebook/current_release.ipynb"
    assert payload["search"]["family_ids"] == [
        "constant",
        "drift",
        "linear_trend",
        "seasonal_naive",
    ]


def test_full_vision_example_uses_non_point_and_extension_lane() -> None:
    payload = yaml.safe_load(FULL_VISION_EXAMPLE.read_text(encoding="utf-8"))

    assert payload["request_id"] == "full-vision-run"
    assert payload["probabilistic"]["forecast_object_type"] != "point"
    assert any(
        family_id.startswith("algorithmic_")
        for family_id in payload["search"]["family_ids"]
    )


def test_missing_packaged_current_release_example_fails_closed(tmp_path: Path) -> None:
    fake_asset_root = tmp_path / "fake-assets"
    (fake_asset_root / "schemas" / "contracts").mkdir(parents=True)
    (fake_asset_root / "schemas" / "contracts" / "schema-registry.yaml").write_text(
        "schemas: []\n",
        encoding="utf-8",
    )
    (fake_asset_root / "examples").mkdir()

    try:
        resolve_example_path(
            "current_release_run.yaml",
            project_root=fake_asset_root,
        )
    except EuclidAssetError as exc:
        assert exc.code == "euclid_asset_missing"
        assert exc.asset_path == fake_asset_root / "examples" / "current_release_run.yaml"
    else:
        raise AssertionError("missing current_release_run.yaml did not fail closed")


def test_release_candidate_workflow_matches_example_and_executes_core_steps(
    tmp_path: Path,
) -> None:
    workflow = euclid.get_release_candidate_workflow(project_root=PROJECT_ROOT)

    assert workflow.target_version == euclid.__version__
    assert workflow.benchmark_suite.name == "current-release.yaml"
    assert "src/euclid/_assets/benchmarks/suites" in workflow.benchmark_suite.as_posix()
    assert workflow.notebook_path.name == "current_release.ipynb"

    benchmark_result = RUNNER.invoke(
        app,
        [
            "benchmarks",
            "run",
            "--suite",
            str(workflow.benchmark_suite),
            "--benchmark-root",
            str(tmp_path / "benchmarks"),
        ],
    )
    assert benchmark_result.exit_code == 0
    assert "Suite: current_release" in benchmark_result.stdout

    run_result = RUNNER.invoke(
        app,
        [
            "run",
            "--config",
            str(workflow.example_manifest),
            "--output-root",
            str(tmp_path / "operator-run"),
        ],
    )
    assert run_result.exit_code == 0
    assert "Euclid run" in run_result.stdout

    replay_result = RUNNER.invoke(
        app,
        [
            "replay",
            "--run-id",
            "current-release-run",
            "--output-root",
            str(tmp_path / "operator-run"),
        ],
    )
    assert replay_result.exit_code == 0
    assert "Replay verification: verified" in replay_result.stdout
