from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import euclid
from euclid.benchmarks import load_benchmark_task_manifest
from euclid.benchmarks.submitters import ANALYTIC_BACKEND_SUBMITTER_ID

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("forecast_object_type", "score_metric", "threshold_key"),
    (
        ("point", "mean_absolute_error", None),
        (
            "distribution",
            "continuous_ranked_probability_score",
            "max_ks_distance",
        ),
        ("interval", "interval_score", "max_abs_coverage_gap"),
        ("quantile", "pinball_loss", "max_abs_hit_balance_gap"),
        ("event_probability", "brier_score", "max_reliability_gap"),
    ),
)
def test_harness_accepts_all_forecast_object_types(
    tmp_path: Path,
    forecast_object_type: str,
    score_metric: str,
    threshold_key: str | None,
) -> None:
    manifest_path = _write_probabilistic_benchmark_manifest(
        tmp_path=tmp_path,
        task_id=f"{forecast_object_type}_benchmark_demo",
        forecast_object_type=forecast_object_type,
        score_metric=score_metric,
        threshold_key=threshold_key,
    )

    manifest = load_benchmark_task_manifest(manifest_path)

    assert manifest.frozen_protocol.forecast_object_type == forecast_object_type
    assert manifest.frozen_protocol.score_policy["metric_id"] == score_metric
    if forecast_object_type == "point":
        assert manifest.frozen_protocol.calibration_policy is None
    else:
        assert manifest.frozen_protocol.calibration_policy == {
            "required": True,
            threshold_key: 0.25,
        }
    assert manifest.abstention_policy["expected_mode"] == "calibrated_or_abstain"
    assert (
        manifest.frozen_protocol.replay_policy["verification_mode"]
        == "candidate_and_score_replay"
    )


def test_probabilistic_benchmark_harness_preserves_distribution_semantics(
    tmp_path: Path,
) -> None:
    manifest_path = _write_probabilistic_benchmark_manifest(
        tmp_path=tmp_path,
        task_id="distribution_benchmark_demo",
        forecast_object_type="distribution",
        score_metric="continuous_ranked_probability_score",
        threshold_key="max_ks_distance",
    )

    result = euclid.profile_benchmark_task(
        manifest_path=manifest_path,
        benchmark_root=tmp_path / "probabilistic-benchmark-output",
        project_root=PROJECT_ROOT,
        resume=False,
    )

    task_result = json.loads(result.report_paths.task_result_path.read_text())
    track_summary = task_result["track_summary"]

    assert task_result["forecast_object_type"] == "distribution"
    assert track_summary["forecast_object_type"] == "distribution"
    assert track_summary["score_law"] == "continuous_ranked_probability_score"
    assert track_summary["calibration_required"] is True
    assert track_summary["calibration_verdict"] == "required"
    assert track_summary["abstention_mode"] == "calibrated_or_abstain"
    assert track_summary["replay_verification"] == "candidate_and_score_replay"
    assert track_summary["replay_verification_status"] == "verified"


def test_suite_summary_contains_semantic_fields(tmp_path: Path) -> None:
    suite_manifest = tmp_path / "benchmarks" / "suites" / "probabilistic-suite.yaml"
    suite_manifest.parent.mkdir(parents=True, exist_ok=True)
    distribution_manifest = _write_probabilistic_benchmark_manifest(
        tmp_path=tmp_path,
        task_id="distribution_benchmark_demo",
        forecast_object_type="distribution",
        score_metric="continuous_ranked_probability_score",
        threshold_key="max_ks_distance",
    )
    point_manifest = _write_probabilistic_benchmark_manifest(
        tmp_path=tmp_path,
        task_id="point_benchmark_demo",
        forecast_object_type="point",
        score_metric="mean_absolute_error",
        threshold_key=None,
    )
    suite_manifest.write_text(
        yaml.safe_dump(
            {
                "suite_id": "probabilistic_semantics",
                "description": "Temporary suite for benchmark semantic summary checks.",
                "task_manifest_paths": [
                    str(distribution_manifest.resolve()),
                    str(point_manifest.resolve()),
                ],
                "required_tracks": ["predictive_generalization"],
                "surface_requirements": [
                    {
                        "surface_id": "probabilistic_forecast_surface",
                        "task_ids": [
                            "distribution_benchmark_demo",
                            "point_benchmark_demo",
                        ],
                        "replay_required": True,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = euclid.profile_benchmark_suite(
        manifest_path=suite_manifest,
        benchmark_root=tmp_path / "probabilistic-suite-output",
        project_root=PROJECT_ROOT,
        resume=False,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    distribution_row = next(
        row
        for row in summary["task_results"]
        if row["task_id"] == "distribution_benchmark_demo"
    )
    surface_row = next(
        row
        for row in summary["surface_statuses"]
        if row["surface_id"] == "probabilistic_forecast_surface"
    )

    assert distribution_row["forecast_object_type"] == "distribution"
    assert distribution_row["score_law"] == "continuous_ranked_probability_score"
    assert distribution_row["calibration_verdict"] == "required"
    assert distribution_row["abstention_mode"] == "calibrated_or_abstain"
    assert distribution_row["replay_verification"] == "verified"
    assert surface_row["evidence"]["forecast_object_types"] == [
        "distribution",
        "point",
    ]
    assert surface_row["evidence"]["score_laws"] == [
        "continuous_ranked_probability_score",
        "mean_absolute_error",
    ]
    assert surface_row["evidence"]["calibration_verdicts"] == [
        "not_applicable",
        "required",
    ]
    assert surface_row["evidence"]["abstention_modes"] == [
        "calibrated_or_abstain",
    ]
    assert surface_row["evidence"]["replay_verification"] == "verified"


def _write_probabilistic_benchmark_manifest(
    *,
    tmp_path: Path,
    task_id: str,
    forecast_object_type: str,
    score_metric: str,
    threshold_key: str | None,
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
                "task_family": "probabilistic_forecast_surface",
                "regime_tags": ["seasonal", "probabilistic"],
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
                "forecast_object_type": forecast_object_type,
                "score_policy": {"metric_id": score_metric},
                "practical_significance_margin": 0.02,
                "budget_policy": {
                    "wall_clock_seconds": 60,
                    "candidate_limit": 16,
                },
                "baseline_registry": [
                    {
                        "baseline_id": "naive_last_value",
                        "manifest_path": (
                            "benchmarks/baselines/predictive_generalization/"
                            "naive-last-value.yaml"
                        ),
                    }
                ],
                "submitter_registry": [
                    {"submitter_id": ANALYTIC_BACKEND_SUBMITTER_ID}
                ],
                "seed_policy": {"seed": 11, "restarts": 0},
                "adversarial_tags": ["semantic_probabilistic_surface"],
                "abstention_policy": {
                    "allow_abstention": True,
                    "expected_mode": "calibrated_or_abstain",
                },
                "forbidden_shortcuts": ["oracle_fit"],
                "replay_policy": {
                    "ledger_mode": "append_only",
                    "persist_candidate_ledgers": True,
                    "verification_mode": "candidate_and_score_replay",
                },
                "origin_policy": {
                    "policy_id": "rolling_origin",
                    "min_origins": 4,
                },
                "horizon_policy": {"horizons": [1, 3]},
                "baseline_comparison_policy": {
                    "paired_test": "diebold_mariano",
                    "require_margin_win": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if threshold_key is not None:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        payload["calibration_policy"] = {
            "required": True,
            threshold_key: 0.25,
        }
        manifest_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
    return manifest_path
