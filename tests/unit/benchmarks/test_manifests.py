from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from euclid.benchmarks.manifests import (
    AdversarialHonestyTaskManifest,
    PredictiveGeneralizationTaskManifest,
    RediscoveryTaskManifest,
    ensure_benchmark_repository_tree,
    load_benchmark_task_manifest,
    load_benchmark_task_manifests,
)
from euclid.contracts.errors import ContractValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_ensure_benchmark_repository_tree_creates_phase08_layout(
    tmp_path: Path,
) -> None:
    tree = ensure_benchmark_repository_tree(tmp_path / "benchmarks")

    assert tree.root == tmp_path / "benchmarks"
    for directory in (
        tree.tracks_dir,
        tree.tasks_dir,
        tree.manifests_dir,
        tree.baselines_dir,
        tree.results_dir,
        tree.reports_dir,
    ):
        assert directory.is_dir()

    for track_id in (
        "rediscovery",
        "predictive_generalization",
        "adversarial_honesty",
    ):
        assert tree.track_directory("tracks", track_id).is_dir()
        assert tree.track_directory("tasks", track_id).is_dir()
        assert tree.track_directory("manifests", track_id).is_dir()
        assert tree.track_directory("baselines", track_id).is_dir()
        assert tree.track_directory("results", track_id).is_dir()
        assert tree.track_directory("reports", track_id).is_dir()


@pytest.mark.parametrize(
    ("relative_path", "expected_type", "expected_track"),
    (
        (
            "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml",
            RediscoveryTaskManifest,
            "rediscovery",
        ),
        (
            "benchmarks/tasks/predictive_generalization/seasonal-trend-demo.yaml",
            PredictiveGeneralizationTaskManifest,
            "predictive_generalization",
        ),
        (
            "benchmarks/tasks/adversarial_honesty/leakage-trap-demo.yaml",
            AdversarialHonestyTaskManifest,
            "adversarial_honesty",
        ),
    ),
)
def test_load_benchmark_task_manifest_builds_track_specific_models(
    relative_path: str,
    expected_type: type[object],
    expected_track: str,
) -> None:
    manifest = load_benchmark_task_manifest(PROJECT_ROOT / relative_path)

    assert isinstance(manifest, expected_type)
    assert manifest.track_id == expected_track
    assert manifest.frozen_protocol.dataset_ref.startswith("fixtures/runtime/")
    assert manifest.frozen_protocol.forecast_object_type == "point"
    assert manifest.submitter_ids == (
        "analytic_backend",
        "recursive_spectral_backend",
        "algorithmic_search_backend",
        "portfolio_orchestrator",
    )
    assert manifest.source_path == PROJECT_ROOT / relative_path


def test_load_benchmark_task_manifests_discovers_full_vision_task_corpus() -> None:
    manifests = load_benchmark_task_manifests(PROJECT_ROOT / "benchmarks" / "tasks")

    assert len(manifests) == 30
    assert Counter(manifest.track_id for manifest in manifests) == {
        "adversarial_honesty": 1,
        "rediscovery": 3,
        "predictive_generalization": 26,
    }
    assert {manifest.task_id for manifest in manifests} == {
        "additive_residual_composition_medium_demo",
        "algorithmic_last_observation_demo",
        "algorithmic_last_observation_medium_demo",
        "bounded_search_medium_demo",
        "distribution_medium_positive_demo",
        "equality_saturation_search_medium_demo",
        "exact_enumeration_search_medium_demo",
        "event_probability_medium_abstention_demo",
        "interval_medium_robustness_demo",
        "leakage_trap_demo",
        "mechanistic_lane_medium_negative_demo",
        "mechanistic_lane_medium_positive_demo",
        "mechanistic_lane_medium_insufficient_demo",
        "mechanistic_lane_negative_demo",
        "mechanistic_lane_positive_demo",
        "piecewise_composition_medium_demo",
        "portfolio_selection_medium_demo",
        "quantile_medium_misspecification_demo",
        "regime_conditioned_composition_medium_demo",
        "seasonal_trend_demo",
        "seasonal_trend_medium_demo",
        "shared_local_panel_medium_negative_demo",
        "shared_local_panel_medium_positive_demo",
        "shared_local_panel_negative_demo",
        "shared_local_panel_positive_demo",
        "stochastic_search_medium_demo",
        "planted_analytic_demo",
        "robustness_medium_leakage_demo",
        "robustness_medium_positive_demo",
        "robustness_medium_sensitivity_abstention_demo",
    }
    assert {
        manifest.frozen_protocol.forecast_object_type for manifest in manifests
    } == {
        "point",
        "distribution",
        "interval",
        "quantile",
        "event_probability",
    }
    assert any(
        manifest.dataset_ref.startswith("fixtures/runtime/full_vision_certification/")
        for manifest in manifests
    )


def test_rediscovery_manifest_requires_equivalence_policy(tmp_path: Path) -> None:
    manifest_path = tmp_path / "missing-equivalence.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "task_id": "broken_rediscovery",
                "track_id": "rediscovery",
                "task_family": "analytic_symbolic_regression",
                "regime_tags": ["planted_generator"],
                "dataset_ref": "fixtures/runtime/prototype-series.csv",
                "snapshot_policy": {"freeze_mode": "content_addressed_copy"},
                "generator_status": "known_generator",
                "target_transform_policy": {"transform_id": "identity"},
                "quantization_policy": {"lattice": "decimal_1e-6"},
                "observation_model_policy": {"model_id": "gaussian_point"},
                "split_policy": {"policy_id": "walk_forward"},
                "forecast_object_type": "point",
                "score_policy": {"metric_id": "mae"},
                "practical_significance_margin": 0.01,
                "budget_policy": {"wall_clock_seconds": 60},
                "baseline_registry": [{"baseline_id": "naive"}],
                "submitter_registry": [{"submitter_id": "algorithmic_search_backend"}],
                "seed_policy": {"seed": 7},
                "adversarial_tags": ["equivalence_hacking"],
                "abstention_policy": {"allow_abstention": True},
                "forbidden_shortcuts": ["oracle_fit"],
                "replay_policy": {"ledger_mode": "append_only"},
                "target_structure_ref": "fixtures/generators/planted-analytic.json",
                "parameter_tolerance_policy": {"absolute_tolerance": 1e-6},
                "predictive_adequacy_floor": {"metric_id": "mae", "max_value": 0.2},
            },
            sort_keys=True,
        )
    )

    with pytest.raises(ContractValidationError) as excinfo:
        load_benchmark_task_manifest(manifest_path)

    assert excinfo.value.field_path == "equivalence_policy"


def test_probabilistic_benchmark_manifest_requires_calibration_policy(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "missing-calibration.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "task_id": "broken_probabilistic_task",
                "track_id": "predictive_generalization",
                "task_family": "single_entity_probabilistic_forecast",
                "regime_tags": ["seasonal"],
                "dataset_ref": "fixtures/runtime/prototype-series.csv",
                "snapshot_policy": {"freeze_mode": "content_addressed_copy"},
                "generator_status": "unknown_real_world",
                "target_transform_policy": {"transform_id": "identity"},
                "quantization_policy": {"lattice": "decimal_1e-6"},
                "observation_model_policy": {"model_id": "gaussian_point"},
                "split_policy": {"policy_id": "rolling_origin"},
                "forecast_object_type": "distribution",
                "score_policy": {"metric_id": "crps"},
                "practical_significance_margin": 0.01,
                "budget_policy": {"wall_clock_seconds": 60},
                "baseline_registry": [{"baseline_id": "naive"}],
                "submitter_registry": [{"submitter_id": "algorithmic_search_backend"}],
                "seed_policy": {"seed": 7},
                "adversarial_tags": ["baseline_overclaim_guard"],
                "abstention_policy": {"allow_abstention": True},
                "forbidden_shortcuts": ["oracle_fit"],
                "replay_policy": {"ledger_mode": "append_only"},
                "origin_policy": {"policy_id": "rolling_origin"},
                "horizon_policy": {"horizons": [1, 3]},
                "baseline_comparison_policy": {"paired_test": "diebold_mariano"},
            },
            sort_keys=True,
        )
    )

    with pytest.raises(ContractValidationError) as excinfo:
        load_benchmark_task_manifest(manifest_path)

    assert excinfo.value.field_path == "calibration_policy"


def test_manifest_track_id_must_match_tasks_directory(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tasks" / "rediscovery" / "mismatch.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "task_id": "mismatched_task",
                "track_id": "predictive_generalization",
                "task_family": "single_entity_point_forecast",
                "regime_tags": ["trend"],
                "dataset_ref": "fixtures/runtime/prototype-series.csv",
                "snapshot_policy": {"freeze_mode": "content_addressed_copy"},
                "generator_status": "unknown_real_world",
                "target_transform_policy": {"transform_id": "identity"},
                "quantization_policy": {"lattice": "decimal_1e-6"},
                "observation_model_policy": {"model_id": "gaussian_point"},
                "split_policy": {"policy_id": "rolling_origin"},
                "forecast_object_type": "point",
                "score_policy": {"metric_id": "mae"},
                "practical_significance_margin": 0.01,
                "budget_policy": {"wall_clock_seconds": 60},
                "baseline_registry": [{"baseline_id": "naive"}],
                "submitter_registry": [{"submitter_id": "algorithmic_search_backend"}],
                "seed_policy": {"seed": 7},
                "adversarial_tags": ["baseline_overclaim_guard"],
                "abstention_policy": {"allow_abstention": True},
                "forbidden_shortcuts": ["oracle_fit"],
                "replay_policy": {"ledger_mode": "append_only"},
                "origin_policy": {"policy_id": "rolling_origin"},
                "horizon_policy": {"horizons": [1]},
                "baseline_comparison_policy": {"paired_test": "diebold_mariano"},
            },
            sort_keys=True,
        )
    )

    with pytest.raises(ContractValidationError) as excinfo:
        load_benchmark_task_manifest(manifest_path)

    assert excinfo.value.code == "benchmark_track_path_mismatch"
    assert excinfo.value.field_path == "track_id"
