from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from euclid.benchmarks.efficacy_metrics import (
    FALSE_HOLISTIC_RATE,
    NONSTATIONARY_DETECTION_DELAY,
    NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE,
    NONSTATIONARY_DETECTION_PRECISION,
    NONSTATIONARY_DETECTION_RECALL,
    PLANTED_LAW_RECOVERY_RATE,
    THIN_EVIDENCE_PROBABILISTIC_ATTACHMENT_RATE,
    compute_efficacy_metric,
    compute_nonstationary_detection_placeholders,
    efficacy_metric_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
THRESHOLD_PROFILE = (
    PROJECT_ROOT
    / "src/euclid/_assets/schemas/readiness/benchmark-threshold-gates-v1.yaml"
)


def test_registry_exposes_threshold_yaml_metric_ids_and_nonstationary_placeholders() -> (
    None
):
    profile = yaml.safe_load(THRESHOLD_PROFILE.read_text(encoding="utf-8"))
    threshold_metric_ids = {str(gate["metric_id"]) for gate in profile["gates"]}

    registry = efficacy_metric_registry()

    assert threshold_metric_ids <= set(registry)
    assert {
        NONSTATIONARY_DETECTION_PRECISION,
        NONSTATIONARY_DETECTION_RECALL,
        NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE,
        NONSTATIONARY_DETECTION_DELAY,
    } <= set(registry)
    for metric_id, spec in registry.items():
        assert spec.metric_id == metric_id
        assert callable(spec.compute)


def test_planted_law_recovery_reports_exact_near_recovery_and_provenance() -> None:
    result = compute_efficacy_metric(
        PLANTED_LAW_RECOVERY_RATE,
        (
            _row(
                "phase08_planted_linear_exact",
                "analytic_backend",
                "linear",
                "r-1",
                recovery_status="exact",
            ),
            _row(
                "phase08_planted_affine_lag_exact",
                "analytic_backend",
                "affine",
                "r-2",
                recovery_status="near",
            ),
            _row(
                "phase08_planted_damped_harmonic_exact",
                "algorithmic_search_backend",
                "harmonic",
                "r-3",
                recovery_status="missed",
            ),
        ),
    )

    assert result.metric_id == PLANTED_LAW_RECOVERY_RATE
    assert result.observed_value == pytest.approx(2 / 3)
    assert result.numerator == 2
    assert result.denominator == 3
    assert result.details["exact_recovery_count"] == 1
    assert result.details["near_recovery_count"] == 1
    assert result.details["missed_count"] == 1
    _assert_required_provenance(result.as_dict(), row_count=3)


def test_false_holistic_claim_rate_counts_false_positives_over_adversarial_tasks() -> (
    None
):
    result = compute_efficacy_metric(
        FALSE_HOLISTIC_RATE,
        (
            _row(
                "random_walk_canary_demo",
                "portfolio_orchestrator",
                "random_walk_story",
                "r-1",
                track_id="adversarial_honesty",
                expected_safe_outcome="abstain",
                holistic_claim_published=True,
            ),
            _row(
                "near_persistence_canary_demo",
                "portfolio_orchestrator",
                None,
                "r-2",
                track_id="adversarial_honesty",
                expected_safe_outcome="abstain",
                holistic_claim_published=False,
            ),
            _row(
                "phase08_planted_linear_exact",
                "analytic_backend",
                "linear",
                "r-3",
                track_id="rediscovery",
                holistic_claim_published=True,
            ),
        ),
    )

    assert result.metric_id == FALSE_HOLISTIC_RATE
    assert result.observed_value == pytest.approx(0.5)
    assert result.numerator == 1
    assert result.denominator == 2
    assert result.details["false_positive_task_ids"] == ["random_walk_canary_demo"]
    _assert_required_provenance(result.as_dict(), row_count=2)


def test_probabilistic_attachment_quality_reports_coverage_width_calibration_and_status() -> (
    None
):
    result = compute_efficacy_metric(
        THIN_EVIDENCE_PROBABILISTIC_ATTACHMENT_RATE,
        (
            _row(
                "quantile_medium_misspecification_demo",
                "probabilistic_backend",
                "quantile_candidate",
                "r-1",
                forecast_object_type="quantile",
                coverage=0.92,
                width=1.4,
                calibration_count=40,
                probabilistic_attachment_status="retained",
            ),
            _row(
                "interval_medium_robustness_demo",
                "probabilistic_backend",
                "interval_candidate",
                "r-2",
                forecast_object_type="interval",
                coverage=0.88,
                width=1.0,
                calibration_count=60,
                probabilistic_attachment_status="retained",
            ),
        ),
        minimum_calibration_count=20,
    )

    assert result.metric_id == THIN_EVIDENCE_PROBABILISTIC_ATTACHMENT_RATE
    assert result.observed_value == 0.0
    assert result.numerator == 0
    assert result.denominator == 2
    assert result.details["coverage"] == pytest.approx(0.9)
    assert result.details["width"] == pytest.approx(1.2)
    assert result.details["calibration_count"] == 100
    assert result.details["status"] == "passed"
    _assert_required_provenance(result.as_dict(), row_count=2)


def test_probabilistic_attachment_quality_fails_closed_for_thin_retained_attachment() -> (
    None
):
    result = compute_efficacy_metric(
        THIN_EVIDENCE_PROBABILISTIC_ATTACHMENT_RATE,
        (
            _row(
                "event_probability_medium_abstention_demo",
                "probabilistic_backend",
                "event_probability_candidate",
                "r-1",
                forecast_object_type="event_probability",
                coverage=None,
                width=0.7,
                calibration_count=0,
                probabilistic_attachment_status="retained",
            ),
        ),
        minimum_calibration_count=20,
    )

    assert result.observed_value == 1.0
    assert result.numerator == 1
    assert result.denominator == 1
    assert result.status == "failed"
    assert result.reason == "thin_probabilistic_attachment"
    assert result.details["status"] == "failed"
    assert result.details["thin_attachment_task_ids"] == [
        "event_probability_medium_abstention_demo"
    ]
    _assert_required_provenance(result.as_dict(), row_count=1)


def test_nonstationary_detection_placeholders_fail_closed_until_phase6_lane_exists() -> (
    None
):
    results = compute_nonstationary_detection_placeholders(
        (
            _row(
                "nonstationary_piecewise_demo",
                "change_point_backend",
                "piecewise_candidate",
                "r-1",
                track_id="predictive_generalization",
            ),
        )
    )

    assert [result.metric_id for result in results] == [
        NONSTATIONARY_DETECTION_PRECISION,
        NONSTATIONARY_DETECTION_RECALL,
        NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE,
        NONSTATIONARY_DETECTION_DELAY,
    ]
    for result in results:
        assert result.observed_value is None
        assert result.status == "missing"
        assert result.reason == "nonstationary_lane_missing_until_phase6"
        assert result.details["lane_status"] == "missing"
        _assert_required_provenance(result.as_dict(), row_count=1)


def _row(
    task_id: str,
    submitter_id: str,
    candidate_id: str | None,
    replay_id: str,
    **extra,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "submitter_id": submitter_id,
        "candidate_id": candidate_id,
        "replay_id": replay_id,
        **extra,
    }


def _assert_required_provenance(
    metric: dict[str, object],
    *,
    row_count: int,
) -> None:
    provenance = metric["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["row_count"] == row_count
    rows = provenance["rows"]
    assert isinstance(rows, list)
    assert len(rows) == row_count
    for row in rows:
        assert set(row) == {
            "task_id",
            "submitter_id",
            "candidate_id",
            "replay_id",
            "row_count",
            "row_index",
        }
