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


def test_phase08_efficacy_metrics_report_planted_law_exact_and_near_recovery() -> None:
    from euclid.benchmarks.efficacy_metrics import (
        PLANTED_LAW_RECOVERY_RATE,
        compute_efficacy_metric,
    )

    metrics = compute_efficacy_metric(
        PLANTED_LAW_RECOVERY_RATE,
        [
            {
                "task_id": "phase08_planted_linear_exact",
                "submitter_id": "analytic_backend",
                "candidate_id": "linear_exact",
                "replay_id": "replay:linear_exact",
                "row_count": 64,
                "recovery_status": "exact",
            },
            {
                "task_id": "phase08_planted_affine_lag_noisy",
                "submitter_id": "algorithmic_search_backend",
                "candidate_id": "affine_lag_near",
                "replay_id": "replay:affine_lag_near",
                "row_count": 64,
                "recovery_status": "near",
            },
            {
                "task_id": "phase08_planted_damped_harmonic_noisy",
                "submitter_id": "recursive_spectral_backend",
                "candidate_id": "wrong_family",
                "replay_id": "replay:wrong_family",
                "row_count": 64,
                "recovery_status": "miss",
            },
        ]
    ).as_dict()

    assert metrics["metric_id"] == "planted_law_recovery_rate"
    assert metrics["details"]["exact_recovery_count"] == 1
    assert metrics["details"]["near_recovery_count"] == 1
    assert metrics["numerator"] == 2
    assert metrics["denominator"] == 3
    assert metrics["observed_value"] == 2 / 3
    provenance_row = metrics["provenance"]["rows"][0]
    assert provenance_row["task_id"] == "phase08_planted_linear_exact"
    assert provenance_row["submitter_id"] == "analytic_backend"
    assert provenance_row["candidate_id"] == "linear_exact"
    assert provenance_row["replay_id"] == "replay:linear_exact"
    assert provenance_row["row_count"] == 64


def test_phase08_efficacy_metrics_report_false_holistic_claim_rate() -> None:
    from euclid.benchmarks.efficacy_metrics import (
        FALSE_HOLISTIC_RATE,
        compute_efficacy_metric,
    )

    metrics = compute_efficacy_metric(
        FALSE_HOLISTIC_RATE,
        [
            {
                "task_id": "random_walk_canary_demo",
                "expected_safe_outcome": "abstain",
                "claim_scope": "holistic_law_claim",
                "local_winner_submitter_id": "analytic_backend",
            },
            {
                "task_id": "near_persistence_canary_demo",
                "expected_safe_outcome": "abstain",
                "claim_scope": "abstention_only",
                "local_winner_submitter_id": None,
            },
        ]
    ).as_dict()

    assert metrics["metric_id"] == "false_holistic_rate"
    assert metrics["details"]["false_positive_count"] == 1
    assert metrics["denominator"] == 2
    assert metrics["observed_value"] == 0.5
    assert metrics["details"]["false_positive_task_ids"] == ["random_walk_canary_demo"]


def test_phase08_efficacy_metrics_report_probabilistic_attachment_quality() -> None:
    from euclid.benchmarks.efficacy_metrics import (
        THIN_EVIDENCE_PROBABILISTIC_ATTACHMENT_RATE,
        compute_efficacy_metric,
    )

    metrics = compute_efficacy_metric(
        THIN_EVIDENCE_PROBABILISTIC_ATTACHMENT_RATE,
        [
            {
                "task_id": "probabilistic_interval_thin_evidence",
                "submitter_id": "probabilistic_backend",
                "candidate_id": "interval_candidate",
                "replay_id": "replay:interval_candidate",
                "row_count": 20,
                "probabilistic_attachment_retained": True,
                "thin_evidence": True,
                "evidence_strength": "thin",
                "coverage": 0.55,
                "mean_interval_width": 12.5,
                "calibration_count": 20,
                "calibration_status": "failed",
            }
        ]
    ).as_dict()

    assert metrics["metric_id"] == "thin_evidence_probabilistic_attachment_rate"
    assert metrics["observed_value"] == 1.0
    assert metrics["details"]["coverage"] == 0.55
    assert metrics["details"]["width"] == 12.5
    assert metrics["details"]["calibration_count"] == 20
    assert metrics["details"]["calibration_status"] == "failed"
    assert metrics["status"] == "failed"
    assert metrics["provenance"]["rows"][0]["row_count"] == 20


def test_phase08_efficacy_metrics_report_nonstationary_detection_tolerance() -> None:
    from euclid.benchmarks.efficacy_metrics import (
        NONSTATIONARY_DETECTION_DELAY,
        NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE,
        NONSTATIONARY_DETECTION_PRECISION,
        NONSTATIONARY_DETECTION_RECALL,
        compute_nonstationary_detection_placeholders,
    )

    metrics = {
        metric.metric_id: metric.as_dict()
        for metric in compute_nonstationary_detection_placeholders(
            [
                {
                    "task_id": "phase08_nonstationary_break_demo",
                    "truth_change_points": [40, 80],
                    "detected_change_points": [39, 84],
                    "detection_tolerance_steps": 2,
                }
            ]
        )
    }

    assert set(metrics) == {
        NONSTATIONARY_DETECTION_PRECISION,
        NONSTATIONARY_DETECTION_RECALL,
        NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE,
        NONSTATIONARY_DETECTION_DELAY,
    }
    assert metrics[NONSTATIONARY_DETECTION_PRECISION]["status"] == "missing"
    assert metrics[NONSTATIONARY_DETECTION_PRECISION]["reason"] == (
        "nonstationary_lane_missing_until_phase6"
    )
    assert metrics[NONSTATIONARY_DETECTION_PRECISION]["details"][
        "detection_tolerance_steps"
    ] == 2
    assert metrics[NONSTATIONARY_DETECTION_RECALL]["details"][
        "detection_tolerance_steps"
    ] == 2
    assert metrics[NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE]["details"][
        "detection_tolerance_steps"
    ] == 2
    assert metrics[NONSTATIONARY_DETECTION_DELAY]["details"][
        "detection_tolerance_steps"
    ] == 2
