from __future__ import annotations

from pathlib import Path

import yaml

from euclid.benchmarks import load_benchmark_suite_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "src/euclid/_assets"
PHASE08_SUITE = ASSET_ROOT / "benchmarks/suites/phase08-holistic-honesty.yaml"


def test_phase08_holistic_honesty_suite_declares_wave3_assets() -> None:
    manifest = load_benchmark_suite_manifest(PHASE08_SUITE)
    payload = yaml.safe_load(PHASE08_SUITE.read_text(encoding="utf-8"))

    assert manifest.suite_id == "phase08_holistic_honesty"
    assert manifest.required_tracks == (
        "rediscovery",
        "predictive_generalization",
        "adversarial_honesty",
    )
    assert [path.relative_to(ASSET_ROOT).as_posix() for path in manifest.task_manifest_paths] == [
        "benchmarks/tasks/rediscovery/planted-linear-exact-phase08.yaml",
        "benchmarks/tasks/rediscovery/planted-affine-lag-exact-phase08.yaml",
        "benchmarks/tasks/rediscovery/planted-damped-harmonic-exact-phase08.yaml",
        "benchmarks/tasks/rediscovery/planted-additive-composition-exact-phase08.yaml",
        "benchmarks/tasks/predictive_generalization/planted-linear-noisy-phase08.yaml",
        "benchmarks/tasks/predictive_generalization/planted-affine-lag-noisy-phase08.yaml",
        "benchmarks/tasks/predictive_generalization/planted-damped-harmonic-noisy-phase08.yaml",
        "benchmarks/tasks/predictive_generalization/planted-additive-composition-noisy-phase08.yaml",
        "benchmarks/tasks/adversarial_honesty/random-walk-canary.yaml",
        "benchmarks/tasks/adversarial_honesty/near-persistence-canary.yaml",
        "benchmarks/tasks/adversarial_honesty/interpolation-bait-canary.yaml",
        "benchmarks/tasks/adversarial_honesty/row-index-leakage-canary.yaml",
        "benchmarks/tasks/adversarial_honesty/sample-wide-closure-canary.yaml",
        "benchmarks/tasks/predictive_generalization/real-series-spy-daily-return-honesty-20260418.yaml",
        "benchmarks/tasks/predictive_generalization/real-series-spy-price-close-honesty-20260416.yaml",
        "benchmarks/tasks/predictive_generalization/real-series-gld-price-close-honesty-20260418.yaml",
    ]
    assert {surface.surface_id for surface in manifest.surface_requirements} == {
        "phase08_planted_law_recovery",
        "phase08_adversarial_honesty",
        "phase08_real_series_honesty",
    }
    assert payload["release_gate_profile_path"] == (
        "schemas/readiness/benchmark-threshold-gates-v1.yaml"
    )
    assert payload["release_gate_task_buckets"] == {
        "descriptive_non_abstention": {
            "required_task_ids": [
                "phase08_planted_linear_noisy",
                "phase08_planted_affine_lag_noisy",
                "phase08_planted_damped_harmonic_noisy",
                "phase08_planted_additive_composition_noisy",
                "real_series_spy_price_close_honesty_20260416",
                "real_series_gld_price_close_honesty_20260418",
            ]
        },
        "false_holistic_rate": {
            "required_task_ids": [
                "random_walk_canary_demo",
                "near_persistence_canary_demo",
                "interpolation_bait_canary_demo",
                "row_index_leakage_canary_demo",
                "sample_wide_closure_canary_demo",
            ]
        },
        "planted_law_recovery": {
            "required_task_ids": [
                "phase08_planted_linear_exact",
                "phase08_planted_affine_lag_exact",
                "phase08_planted_damped_harmonic_exact",
                "phase08_planted_additive_composition_exact",
                "phase08_planted_linear_noisy",
                "phase08_planted_affine_lag_noisy",
                "phase08_planted_damped_harmonic_noisy",
                "phase08_planted_additive_composition_noisy",
            ]
        },
    }
