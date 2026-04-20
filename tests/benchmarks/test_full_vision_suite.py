from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import euclid
from euclid.benchmarks import (
    load_benchmark_suite_manifest,
    load_benchmark_task_manifest,
)
from euclid.benchmarks.submitters import ANALYTIC_BACKEND_SUBMITTER_ID
from euclid.contracts.errors import ContractValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINT_TASK = PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
FULL_VISION_SUITE = PROJECT_ROOT / "benchmarks/suites/full-vision.yaml"


def test_full_vision_suite_summary_emits_semantic_fields_for_task_and_suite_levels(
    tmp_path: Path,
) -> None:
    suite_manifest = _write_full_vision_suite_manifest(tmp_path)

    result = euclid.profile_benchmark_suite(
        manifest_path=suite_manifest,
        benchmark_root=tmp_path / "full-vision-suite-output",
        project_root=PROJECT_ROOT,
        resume=False,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    task_results = {item["task_id"]: item for item in summary["task_results"]}
    semantic_summary = summary["semantic_summary"]

    assert semantic_summary["forecast_object_types"] == [
        "distribution",
        "event_probability",
        "interval",
        "point",
        "quantile",
    ]
    assert semantic_summary["score_laws"] == [
        "brier_score",
        "continuous_ranked_probability_score",
        "interval_score",
        "mean_absolute_error",
        "pinball_loss",
    ]
    assert semantic_summary["abstention_modes"] == [
        "calibrated_or_abstain",
        "structural_miss",
    ]
    assert semantic_summary["replay_obligations"] == [
        "candidate_and_score_replay",
        "ledger_only",
    ]
    assert semantic_summary["calibration_expectations"] == {
        "distribution": "required",
        "event_probability": "required",
        "interval": "required",
        "quantile": "required",
    }

    assert task_results["planted_analytic_demo"]["forecast_object_type"] == "point"
    assert task_results["planted_analytic_demo"]["calibration_verdict"] == (
        "not_applicable"
    )
    assert task_results["planted_analytic_demo"]["abstention_mode"] == "structural_miss"
    assert task_results["planted_analytic_demo"]["replay_verification"] == "verified"
    assert task_results["distribution_full_vision_demo"]["forecast_object_type"] == (
        "distribution"
    )
    assert task_results["distribution_full_vision_demo"]["score_law"] == (
        "continuous_ranked_probability_score"
    )
    assert task_results["distribution_full_vision_demo"]["calibration_required"] is True
    assert task_results["distribution_full_vision_demo"]["calibration_verdict"] == (
        "required"
    )
    assert task_results["distribution_full_vision_demo"]["replay_obligation"] == (
        "candidate_and_score_replay"
    )
    assert task_results["distribution_full_vision_demo"]["replay_verification"] == (
        "verified"
    )

    surface_rows = {row["surface_id"]: row for row in summary["surface_statuses"]}
    probabilistic_surface = surface_rows["probabilistic_forecast_surface"]["evidence"]
    assert probabilistic_surface["forecast_object_types"] == [
        "distribution",
        "event_probability",
        "interval",
        "quantile",
    ]
    assert probabilistic_surface["score_laws"] == [
        "brier_score",
        "continuous_ranked_probability_score",
        "interval_score",
        "pinball_loss",
    ]
    assert probabilistic_surface["calibration_verdicts"] == ["required"]
    assert probabilistic_surface["abstention_modes"] == ["calibrated_or_abstain"]
    assert probabilistic_surface["replay_verification"] == "verified"


def test_full_vision_suite_manifest_declares_breadth_across_surfaces_and_cases(
) -> None:
    suite_manifest = load_benchmark_suite_manifest(FULL_VISION_SUITE)
    task_manifests = tuple(
        load_benchmark_task_manifest(path)
        for path in suite_manifest.task_manifest_paths
    )

    assert suite_manifest.suite_id == "full_vision"
    assert [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in suite_manifest.task_manifest_paths
    ] == [
        "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml",
        "benchmarks/tasks/predictive_generalization/seasonal-trend-medium.yaml",
        "benchmarks/tasks/adversarial_honesty/leakage-trap-demo.yaml",
        "benchmarks/tasks/algorithmic_rediscovery/causal-last-observation-medium.yaml",
        (
            "benchmarks/tasks/predictive_generalization/"
            "search-class-exact-enumeration-medium.yaml"
        ),
        (
            "benchmarks/tasks/predictive_generalization/"
            "search-class-bounded-medium.yaml"
        ),
        (
            "benchmarks/tasks/predictive_generalization/"
            "search-class-equality-saturation-medium.yaml"
        ),
        (
            "benchmarks/tasks/predictive_generalization/"
            "search-class-stochastic-medium.yaml"
        ),
        (
            "benchmarks/tasks/predictive_generalization/"
            "composition-piecewise-medium.yaml"
        ),
        (
            "benchmarks/tasks/predictive_generalization/"
            "composition-additive-residual-medium.yaml"
        ),
        (
            "benchmarks/tasks/predictive_generalization/"
            "composition-regime-conditioned-medium.yaml"
        ),
        (
            "benchmarks/tasks/predictive_generalization/"
            "portfolio-selection-medium.yaml"
        ),
        (
            "benchmarks/tasks/predictive_generalization/"
            "probabilistic-distribution-medium-positive.yaml"
        ),
        (
            "benchmarks/tasks/predictive_generalization/"
            "probabilistic-interval-medium-robustness.yaml"
        ),
        (
            "benchmarks/tasks/predictive_generalization/"
            "probabilistic-quantile-medium-misspecification.yaml"
        ),
        (
            "benchmarks/tasks/predictive_generalization/"
            "probabilistic-event-probability-medium-abstention.yaml"
        ),
        (
            "benchmarks/tasks/multi_entity/"
            "shared-local-generalization-medium-positive.yaml"
        ),
        (
            "benchmarks/tasks/multi_entity/"
            "shared-local-generalization-medium-negative.yaml"
        ),
        "benchmarks/tasks/mechanistic/mechanistic-lane-medium-positive.yaml",
        "benchmarks/tasks/mechanistic/mechanistic-lane-medium-negative.yaml",
        "benchmarks/tasks/mechanistic/mechanistic-lane-medium-insufficient.yaml",
        "benchmarks/tasks/robustness/robustness-medium-positive.yaml",
        "benchmarks/tasks/robustness/robustness-medium-leakage.yaml",
        (
            "benchmarks/tasks/robustness/"
            "robustness-medium-sensitivity-abstention.yaml"
        ),
    ]
    assert {surface.surface_id for surface in suite_manifest.surface_requirements} == {
        "retained_core_release",
        "probabilistic_forecast_surface",
        "algorithmic_backend",
        "search_class_honesty",
        "composition_operator_semantics",
        "shared_plus_local_decomposition",
        "mechanistic_lane",
        "external_evidence_ingestion",
        "robustness_lane",
        "portfolio_orchestration",
    }
    assert {manifest.track_id for manifest in task_manifests} == {
        "rediscovery",
        "predictive_generalization",
        "adversarial_honesty",
    }
    assert {
        manifest.frozen_protocol.forecast_object_type for manifest in task_manifests
    } == {
        "point",
        "distribution",
        "interval",
        "quantile",
        "event_probability",
    }
    assert {
        tag
        for manifest in task_manifests
        for tag in (*manifest.regime_tags, *manifest.adversarial_tags)
    } >= {
        "positive_case",
        "negative_case",
        "abstention_case",
        "leakage",
        "misspecification_case",
        "robustness_sensitive",
        "external_evidence_required",
    }
    manifests_by_id = {manifest.task_id: manifest for manifest in task_manifests}
    assert (
        manifests_by_id["exact_enumeration_search_medium_demo"].search_class
        == "exact_finite_enumeration"
    )
    assert (
        manifests_by_id["bounded_search_medium_demo"].search_class
        == "bounded_heuristic"
    )
    assert (
        manifests_by_id["equality_saturation_search_medium_demo"].search_class
        == "equality_saturation_heuristic"
    )
    assert (
        manifests_by_id["stochastic_search_medium_demo"].search_class
        == "stochastic_heuristic"
    )
    assert (
        manifests_by_id["piecewise_composition_medium_demo"].composition_operators
        == ("piecewise",)
    )
    assert manifests_by_id[
        "additive_residual_composition_medium_demo"
    ].composition_operators == ("additive_residual",)
    assert manifests_by_id[
        "regime_conditioned_composition_medium_demo"
    ].composition_operators == ("regime_conditioned",)
    assert manifests_by_id[
        "shared_local_panel_medium_positive_demo"
    ].composition_operators == ("shared_plus_local_decomposition",)
    assert all(
        manifest.dataset_ref.startswith("fixtures/runtime/full_vision_certification/")
        for manifest in task_manifests
    )
    assert all(
        (PROJECT_ROOT / manifest.dataset_ref).is_file()
        for manifest in task_manifests
    )


def test_full_vision_suite_covers_every_capability_surface() -> None:
    suite_manifest = load_benchmark_suite_manifest(FULL_VISION_SUITE)
    surface_ids = {
        requirement.surface_id for requirement in suite_manifest.surface_requirements
    }

    assert {
        "retained_core_release",
        "probabilistic_forecast_surface",
        "algorithmic_backend",
        "search_class_honesty",
        "composition_operator_semantics",
        "shared_plus_local_decomposition",
        "mechanistic_lane",
        "external_evidence_ingestion",
        "robustness_lane",
        "portfolio_orchestration",
    } == surface_ids


def test_suite_has_positive_negative_and_abstention_cases() -> None:
    suite_manifest = load_benchmark_suite_manifest(FULL_VISION_SUITE)
    task_manifests = tuple(
        load_benchmark_task_manifest(path)
        for path in suite_manifest.task_manifest_paths
    )
    all_case_tags = {
        tag
        for manifest in task_manifests
        for tag in (*manifest.regime_tags, *manifest.adversarial_tags)
    }

    assert "positive_case" in all_case_tags
    assert "negative_case" in all_case_tags
    assert "abstention_case" in all_case_tags


def test_suite_manifests_reference_authority_snapshot_and_fixture_spec() -> None:
    suite_manifest = load_benchmark_suite_manifest(FULL_VISION_SUITE)

    assert suite_manifest.authority_snapshot_id == "euclid-authority-2026-04-15-b"
    assert suite_manifest.fixture_spec_id == "euclid-certification-fixtures-v1"


def test_suite_defaults_to_certification_fixtures_for_certification() -> None:
    suite_manifest = load_benchmark_suite_manifest(FULL_VISION_SUITE)
    task_manifests = tuple(
        load_benchmark_task_manifest(path)
        for path in suite_manifest.task_manifest_paths
    )

    assert all(
        manifest.dataset_ref.startswith("fixtures/runtime/full_vision_certification/")
        for manifest in task_manifests
    )


def test_profile_benchmark_suite_runs_declared_full_vision_tasks_and_writes_summary(
    tmp_path: Path,
) -> None:
    result = euclid.profile_benchmark_suite(
        manifest_path=FULL_VISION_SUITE,
        benchmark_root=tmp_path / "full-vision-suite",
        resume=False,
    )

    assert result.suite_manifest.suite_id == "full_vision"
    assert [task.task_manifest.task_id for task in result.task_results] == [
        "planted_analytic_demo",
        "seasonal_trend_medium_demo",
        "leakage_trap_demo",
        "algorithmic_last_observation_medium_demo",
        "exact_enumeration_search_medium_demo",
        "bounded_search_medium_demo",
        "equality_saturation_search_medium_demo",
        "stochastic_search_medium_demo",
        "piecewise_composition_medium_demo",
        "additive_residual_composition_medium_demo",
        "regime_conditioned_composition_medium_demo",
        "portfolio_selection_medium_demo",
        "distribution_medium_positive_demo",
        "interval_medium_robustness_demo",
        "quantile_medium_misspecification_demo",
        "event_probability_medium_abstention_demo",
        "shared_local_panel_medium_positive_demo",
        "shared_local_panel_medium_negative_demo",
        "mechanistic_lane_medium_positive_demo",
        "mechanistic_lane_medium_negative_demo",
        "mechanistic_lane_medium_insufficient_demo",
        "robustness_medium_positive_demo",
        "robustness_medium_leakage_demo",
        "robustness_medium_sensitivity_abstention_demo",
    ]
    assert {surface.surface_id for surface in result.surface_statuses} == {
        "retained_core_release",
        "probabilistic_forecast_surface",
        "algorithmic_backend",
        "search_class_honesty",
        "composition_operator_semantics",
        "shared_plus_local_decomposition",
        "mechanistic_lane",
        "external_evidence_ingestion",
        "robustness_lane",
        "portfolio_orchestration",
    }
    assert all(
        surface.benchmark_status == "passed" for surface in result.surface_statuses
    )
    assert all(surface.replay_status == "passed" for surface in result.surface_statuses)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["suite_id"] == "full_vision"
    assert summary["task_count"] == 24
    assert summary["completed_task_count"] == 24
    assert summary["semantic_summary"]["forecast_object_types"] == [
        "distribution",
        "event_probability",
        "interval",
        "point",
        "quantile",
    ]
    search_class_rows = {
        row["search_class"]: row for row in summary["search_class_coverage"]
    }
    assert set(search_class_rows) == {
        "exact_finite_enumeration",
        "bounded_heuristic",
        "equality_saturation_heuristic",
        "stochastic_heuristic",
    }
    assert search_class_rows["exact_finite_enumeration"]["covered_task_ids"] == [
        "exact_enumeration_search_medium_demo"
    ]
    assert search_class_rows["bounded_heuristic"]["covered_task_ids"] == [
        "bounded_search_medium_demo"
    ]
    assert search_class_rows["equality_saturation_heuristic"]["covered_task_ids"] == [
        "equality_saturation_search_medium_demo"
    ]
    assert search_class_rows["stochastic_heuristic"]["covered_task_ids"] == [
        "stochastic_search_medium_demo"
    ]
    composition_rows = {
        row["composition_operator"]: row
        for row in summary["composition_operator_coverage"]
    }
    assert set(composition_rows) == {
        "piecewise",
        "additive_residual",
        "regime_conditioned",
        "shared_plus_local_decomposition",
    }
    assert composition_rows["piecewise"]["covered_task_ids"] == [
        "piecewise_composition_medium_demo"
    ]
    assert composition_rows["additive_residual"]["covered_task_ids"] == [
        "additive_residual_composition_medium_demo"
    ]
    assert composition_rows["regime_conditioned"]["covered_task_ids"] == [
        "regime_conditioned_composition_medium_demo"
    ]
    assert composition_rows["shared_plus_local_decomposition"]["covered_task_ids"] == [
        "shared_local_panel_medium_positive_demo"
    ]


def test_full_vision_suite_fails_if_point_only_fallback_is_used(
    tmp_path: Path,
) -> None:
    suite_manifest = tmp_path / "benchmarks" / "suites" / "full-vision-point-only.yaml"
    suite_manifest.parent.mkdir(parents=True, exist_ok=True)
    suite_manifest.write_text(
        yaml.safe_dump(
            {
                "suite_id": "full_vision",
                "description": (
                    "Dishonest full-vision suite that falls back to "
                    "point-only tasks."
                ),
                "task_manifest_paths": [str(POINT_TASK.resolve())],
                "required_tracks": ["rediscovery"],
                "surface_requirements": [
                    {
                        "surface_id": "retained_core_release",
                        "task_ids": ["planted_analytic_demo"],
                        "replay_required": True,
                    }
                ],
                "authority_snapshot_id": "euclid-authority-2026-04-15-b",
                "fixture_spec_id": "euclid-certification-fixtures-v1",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractValidationError):
        euclid.profile_benchmark_suite(
            manifest_path=suite_manifest,
            benchmark_root=tmp_path / "point-only-full-vision-output",
            project_root=PROJECT_ROOT,
            resume=False,
        )


def _write_full_vision_suite_manifest(tmp_path: Path) -> Path:
    task_manifest_paths = [
        str(POINT_TASK.resolve()),
        str(
            _write_probabilistic_manifest(
                tmp_path=tmp_path,
                task_id="distribution_full_vision_demo",
                forecast_object_type="distribution",
                score_metric="continuous_ranked_probability_score",
                threshold_key="max_ks_distance",
            ).resolve()
        ),
        str(
            _write_probabilistic_manifest(
                tmp_path=tmp_path,
                task_id="interval_full_vision_demo",
                forecast_object_type="interval",
                score_metric="interval_score",
                threshold_key="max_abs_coverage_gap",
            ).resolve()
        ),
        str(
            _write_probabilistic_manifest(
                tmp_path=tmp_path,
                task_id="quantile_full_vision_demo",
                forecast_object_type="quantile",
                score_metric="pinball_loss",
                threshold_key="max_abs_hit_balance_gap",
            ).resolve()
        ),
        str(
            _write_probabilistic_manifest(
                tmp_path=tmp_path,
                task_id="event_probability_full_vision_demo",
                forecast_object_type="event_probability",
                score_metric="brier_score",
                threshold_key="max_reliability_gap",
            ).resolve()
        ),
    ]

    suite_manifest = tmp_path / "benchmarks" / "suites" / "full-vision-temp.yaml"
    suite_manifest.parent.mkdir(parents=True, exist_ok=True)
    suite_manifest.write_text(
        yaml.safe_dump(
            {
                "suite_id": "full_vision_temp",
                "description": (
                    "Temporary suite used to prove that benchmark summaries "
                    "retain semantic fields across point and probabilistic tasks."
                ),
                "task_manifest_paths": task_manifest_paths,
                "required_tracks": ["rediscovery", "predictive_generalization"],
                "surface_requirements": [
                    {
                        "surface_id": "retained_point_surface",
                        "task_ids": ["planted_analytic_demo"],
                        "replay_required": True,
                    },
                    {
                        "surface_id": "probabilistic_forecast_surface",
                        "task_ids": [
                            "distribution_full_vision_demo",
                            "interval_full_vision_demo",
                            "quantile_full_vision_demo",
                            "event_probability_full_vision_demo",
                        ],
                        "replay_required": True,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return suite_manifest


def _write_probabilistic_manifest(
    *,
    tmp_path: Path,
    task_id: str,
    forecast_object_type: str,
    score_metric: str,
    threshold_key: str,
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
                "calibration_policy": {
                    "required": True,
                    threshold_key: 0.25,
                },
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
    return manifest_path
