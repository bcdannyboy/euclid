from __future__ import annotations

from pathlib import Path

import pytest

import euclid
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.modules.evaluation_governance import (
    ComparisonKey,
    build_comparison_universe,
    build_predictive_gate_policy,
    resolve_confirmatory_promotion_allowed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISTRIBUTION_MANIFEST = (
    PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-distribution-demo.yaml"
)
INTERVAL_MANIFEST = (
    PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-interval-demo.yaml"
)
CALIBRATION_FAILURE_MANIFEST = (
    PROJECT_ROOT
    / (
        "fixtures/runtime/phase06/"
        "probabilistic-distribution-calibration-failure-demo.yaml"
    )
)


def test_probabilistic_fixture_blocks_promotion_and_keeps_prediction_artifact(
    tmp_path: Path,
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=CALIBRATION_FAILURE_MANIFEST,
        output_root=tmp_path / "probabilistic-calibration-failure-output",
    )

    prediction_artifact = euclid.resolve_demo_artifact(
        output_root=result.paths.output_root,
        ref=result.prediction.prediction_artifact_ref,
    )
    calibration_result = euclid.resolve_demo_artifact(
        output_root=result.paths.output_root,
        ref=result.calibration.calibration_result_ref,
    )
    predictive_gate_policy = build_predictive_gate_policy(
        allowed_forecast_object_types=(
            "distribution",
            "interval",
            "quantile",
            "event_probability",
        )
    ).to_manifest(catalog)

    assert result.request.request_id == (
        "phase06-probabilistic-distribution-calibration-failure-demo"
    )
    assert result.prediction.forecast_object_type == "distribution"
    assert result.calibration.forecast_object_type == "distribution"
    assert result.calibration.status == "failed"
    assert result.calibration.passed is False
    assert (
        result.calibration.prediction_artifact_ref
        == result.prediction.prediction_artifact_ref
    )
    assert prediction_artifact.manifest.body["forecast_object_type"] == "distribution"
    assert (
        len(prediction_artifact.manifest.body["rows"])
        == result.prediction.row_count
    )
    assert resolve_confirmatory_promotion_allowed(
        candidate_beats_baseline=True,
        predictive_gate_policy_manifest=predictive_gate_policy,
        calibration_result_manifest=calibration_result.manifest,
    ) is False


def test_probabilistic_prediction_artifacts_reject_cross_object_comparison(
    tmp_path: Path,
) -> None:
    distribution_result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=DISTRIBUTION_MANIFEST,
        output_root=tmp_path / "distribution-output",
    )
    interval_result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=INTERVAL_MANIFEST,
        output_root=tmp_path / "interval-output",
    )

    distribution_prediction = euclid.resolve_demo_artifact(
        output_root=distribution_result.paths.output_root,
        ref=distribution_result.prediction.prediction_artifact_ref,
    )
    interval_prediction = euclid.resolve_demo_artifact(
        output_root=interval_result.paths.output_root,
        ref=interval_result.prediction.prediction_artifact_ref,
    )

    with pytest.raises(ContractValidationError) as exc_info:
        build_comparison_universe(
            selected_candidate_id=str(
                distribution_prediction.manifest.body["candidate_id"]
            ),
            baseline_id=str(interval_prediction.manifest.body["candidate_id"]),
            candidate_primary_score=distribution_result.prediction.aggregated_primary_score,
            baseline_primary_score=interval_result.prediction.aggregated_primary_score,
            candidate_comparison_key=_comparison_key_from_prediction_artifact(
                distribution_prediction.manifest.body
            ),
            baseline_comparison_key=_comparison_key_from_prediction_artifact(
                interval_prediction.manifest.body
            ),
        )

    assert exc_info.value.code == "comparison_key_mismatch"


def _comparison_key_from_prediction_artifact(body: dict[str, object]) -> ComparisonKey:
    comparison_key = body["comparison_key"]
    assert isinstance(comparison_key, dict)
    score_policy_ref = body["score_policy_ref"]
    assert isinstance(score_policy_ref, dict)
    horizon_set = comparison_key["horizon_set"]
    assert isinstance(horizon_set, list)
    return ComparisonKey(
        forecast_object_type=str(comparison_key["forecast_object_type"]),
        score_policy_ref=TypedRef(
            schema_name=str(score_policy_ref["schema_name"]),
            object_id=str(score_policy_ref["object_id"]),
        ),
        horizon_set=tuple(int(horizon) for horizon in horizon_set),
        scored_origin_set_id=str(comparison_key["scored_origin_set_id"]),
    )
