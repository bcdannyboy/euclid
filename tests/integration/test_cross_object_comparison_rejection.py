from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import euclid
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.modules.evaluation_governance import (
    ComparisonKey,
    build_comparison_universe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINT_MANIFEST = PROJECT_ROOT / "fixtures/runtime/prototype-demo.yaml"
PROBABILISTIC_MANIFESTS = {
    "distribution": (
        PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-distribution-demo.yaml"
    ),
    "interval": (
        PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-interval-demo.yaml"
    ),
    "quantile": (
        PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-quantile-demo.yaml"
    ),
    "event_probability": (
        PROJECT_ROOT
        / "fixtures/runtime/phase06/probabilistic-event-probability-demo.yaml"
    ),
}


@dataclass(frozen=True)
class ComparisonCase:
    candidate_id: str
    aggregated_primary_score: float
    comparison_key: ComparisonKey


def _comparison_key(
    comparison_key_payload: dict[str, object],
    score_policy_payload: dict[str, object],
) -> ComparisonKey:
    horizon_set = comparison_key_payload["horizon_set"]
    assert isinstance(horizon_set, list)

    return ComparisonKey(
        forecast_object_type=str(comparison_key_payload["forecast_object_type"]),
        score_policy_ref=TypedRef(
            schema_name=str(score_policy_payload["schema_name"]),
            object_id=str(score_policy_payload["object_id"]),
        ),
        horizon_set=tuple(int(horizon) for horizon in horizon_set),
        scored_origin_set_id=str(comparison_key_payload["scored_origin_set_id"]),
    )


def _run_comparison_case(
    *,
    forecast_object_type: str,
    output_root: Path,
) -> ComparisonCase:
    if forecast_object_type == "point":
        point_result = euclid.run_demo_point_evaluation(
            manifest_path=POINT_MANIFEST,
            output_root=output_root,
        )
        prediction_artifact = euclid.resolve_demo_artifact(
            output_root=output_root,
            ref=point_result.prediction.prediction_artifact_ref,
        )
        comparison_key_payload = point_result.comparison.candidate_comparison_key
        assert isinstance(comparison_key_payload, dict)
        score_policy_payload = prediction_artifact.manifest.body["score_policy_ref"]
        assert isinstance(score_policy_payload, dict)
        return ComparisonCase(
            candidate_id=point_result.prediction.candidate_id,
            aggregated_primary_score=point_result.prediction.aggregated_primary_score,
            comparison_key=_comparison_key(
                comparison_key_payload=comparison_key_payload,
                score_policy_payload=score_policy_payload,
            ),
        )

    probabilistic_result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=PROBABILISTIC_MANIFESTS[forecast_object_type],
        output_root=output_root,
    )
    prediction_artifact = euclid.resolve_demo_artifact(
        output_root=output_root,
        ref=probabilistic_result.prediction.prediction_artifact_ref,
    )
    comparison_key_payload = prediction_artifact.manifest.body["comparison_key"]
    assert isinstance(comparison_key_payload, dict)
    score_policy_payload = prediction_artifact.manifest.body["score_policy_ref"]
    assert isinstance(score_policy_payload, dict)
    return ComparisonCase(
        candidate_id=str(prediction_artifact.manifest.body["candidate_id"]),
        aggregated_primary_score=probabilistic_result.prediction.aggregated_primary_score,
        comparison_key=_comparison_key(
            comparison_key_payload=comparison_key_payload,
            score_policy_payload=score_policy_payload,
        ),
    )


@pytest.mark.parametrize(
    ("candidate_type", "baseline_type"),
    (
        ("point", "distribution"),
        ("distribution", "interval"),
        ("interval", "quantile"),
        ("quantile", "event_probability"),
    ),
)
def test_cross_object_comparison_requires_explicit_reduction_contract(
    tmp_path: Path,
    candidate_type: str,
    baseline_type: str,
) -> None:
    candidate_case = _run_comparison_case(
        forecast_object_type=candidate_type,
        output_root=tmp_path / f"candidate-{candidate_type}",
    )
    baseline_case = _run_comparison_case(
        forecast_object_type=baseline_type,
        output_root=tmp_path / f"baseline-{baseline_type}",
    )

    with pytest.raises(ContractValidationError) as exc_info:
        build_comparison_universe(
            selected_candidate_id=candidate_case.candidate_id,
            baseline_id=baseline_case.candidate_id,
            candidate_primary_score=candidate_case.aggregated_primary_score,
            baseline_primary_score=baseline_case.aggregated_primary_score,
            candidate_comparison_key=candidate_case.comparison_key,
            baseline_comparison_key=baseline_case.comparison_key,
        )

    assert exc_info.value.code == "comparison_key_mismatch"
    assert "explicit reduction contract" in exc_info.value.message
