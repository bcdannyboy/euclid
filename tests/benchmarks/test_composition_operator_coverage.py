from __future__ import annotations

import json
from pathlib import Path

import yaml

import euclid
from euclid.benchmarks import load_benchmark_task_manifest
from euclid.benchmarks.submitters import ANALYTIC_BACKEND_SUBMITTER_ID

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSITION_OPERATORS_PATH = PROJECT_ROOT / "schemas/core/composition-operators.yaml"


def test_composition_manifest_exposes_declared_operator_surface(tmp_path: Path) -> None:
    manifest_path = _write_composition_manifest(
        tmp_path=tmp_path,
        task_id="piecewise_composition_surface_demo",
        operator_ids=("piecewise", "additive_residual"),
    )

    manifest = load_benchmark_task_manifest(manifest_path)

    assert manifest.composition_operators == (
        "piecewise",
        "additive_residual",
    )


def test_composition_suite_summary_closes_rows_for_every_admitted_operator(
    tmp_path: Path,
) -> None:
    expected_operator_ids = _composition_operator_ids()
    suite_manifest = _write_composition_suite_manifest(
        tmp_path=tmp_path,
        operator_ids=expected_operator_ids,
    )

    result = euclid.profile_benchmark_suite(
        manifest_path=suite_manifest,
        benchmark_root=tmp_path / "composition-suite-output",
        project_root=PROJECT_ROOT,
        resume=False,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    coverage_rows = {
        row["composition_operator"]: row
        for row in summary["composition_operator_coverage"]
    }

    assert set(coverage_rows) == expected_operator_ids
    for operator_id in sorted(expected_operator_ids):
        row = coverage_rows[operator_id]
        assert row["proof_mode"] == "direct_benchmark_task"
        assert row["covered_task_ids"], f"{operator_id} needs direct benchmark evidence"
        assert row["replay_verified"] is True


def _composition_operator_ids() -> set[str]:
    payload = yaml.safe_load(COMPOSITION_OPERATORS_PATH.read_text(encoding="utf-8"))
    return {entry["id"] for entry in payload["entries"]}


def _write_composition_suite_manifest(
    tmp_path: Path,
    *,
    operator_ids: set[str],
) -> Path:
    task_manifest_paths = [
        str(
            _write_composition_manifest(
                tmp_path=tmp_path,
                task_id=f"{operator_id}_composition_demo",
                operator_ids=(operator_id,),
            ).resolve()
        )
        for operator_id in sorted(operator_ids)
    ]

    suite_manifest = (
        tmp_path / "benchmarks" / "suites" / "composition-operator-coverage.yaml"
    )
    suite_manifest.parent.mkdir(parents=True, exist_ok=True)
    suite_manifest.write_text(
        yaml.safe_dump(
            {
                "suite_id": "composition_operator_coverage",
                "description": (
                    "Temporary suite used to prove that benchmark summaries close "
                    "all admitted composition operators directly."
                ),
                "task_manifest_paths": task_manifest_paths,
                "required_tracks": ["predictive_generalization"],
                "surface_requirements": [
                    {
                        "surface_id": "composition_operator_semantics",
                        "task_ids": [
                            Path(path).stem.replace("-", "_")
                            for path in task_manifest_paths
                        ],
                        "replay_required": True,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return suite_manifest


def _write_composition_manifest(
    *,
    tmp_path: Path,
    task_id: str,
    operator_ids: tuple[str, ...],
) -> Path:
    manifest_path = (
        tmp_path
        / "tasks"
        / "predictive_generalization"
        / f"{task_id.replace('_', '-')}.yaml"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "task_id": task_id,
                "track_id": "predictive_generalization",
                "task_family": "single_entity_point_forecast",
                "regime_tags": ["composition_surface"],
                "dataset_ref": (
                    "fixtures/runtime/full_vision_certification/"
                    "single_entity_predictive/single-entity-forecast-series.csv"
                ),
                "snapshot_policy": {
                    "freeze_mode": "content_addressed_copy",
                    "availability_cutoff": "2026-04-12T00:00:00Z",
                },
                "generator_status": "unknown_real_world",
                "target_transform_policy": {"transform_id": "identity"},
                "quantization_policy": {"lattice": "decimal_1e-6"},
                "observation_model_policy": {"model_id": "gaussian_point"},
                "split_policy": {
                    "policy_id": "rolling_origin",
                    "initial_window": 18,
                    "step": 1,
                },
                "forecast_object_type": "point",
                "score_policy": {"metric_id": "mean_absolute_error"},
                "practical_significance_margin": 0.02,
                "budget_policy": {
                    "wall_clock_seconds": 60,
                    "candidate_limit": 16,
                },
                "baseline_registry": [{"baseline_id": "naive_last_value"}],
                "submitter_registry": [
                    {"submitter_id": ANALYTIC_BACKEND_SUBMITTER_ID}
                ],
                "seed_policy": {"seed": 11, "restarts": 0},
                "adversarial_tags": ["composition_surface"],
                "abstention_policy": {
                    "allow_abstention": True,
                    "expected_mode": "no_publishable_winner",
                },
                "forbidden_shortcuts": ["oracle_fit"],
                "replay_policy": {
                    "ledger_mode": "append_only",
                    "persist_candidate_ledgers": True,
                },
                "origin_policy": {
                    "policy_id": "rolling_origin",
                    "min_origins": 4,
                },
                "horizon_policy": {"horizons": [1]},
                "baseline_comparison_policy": {
                    "paired_test": "diebold_mariano",
                    "require_margin_win": True,
                },
                "composition_operators": list(operator_ids),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest_path
