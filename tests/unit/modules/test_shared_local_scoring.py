from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import PredictionArtifactManifest, PredictionRow
from euclid.modules.evaluation_governance import (
    ComparisonKey,
    build_comparison_universe,
)
from euclid.modules.scoring import score_point_prediction_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_score_point_prediction_artifact_aggregates_per_entity_then_declared_weights(  # noqa: E501
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _score_policy_manifest(
        catalog=catalog,
        horizon_weights=((1, "0.5"), (2, "0.5")),
        entity_aggregation_mode="per_entity_primary_score_then_declared_entity_weights",
    )
    artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="shared-local-candidate",
        score_policy=score_policy,
        rows=(
            ("entity-a", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 1, 10.0, 11.0),
            ("entity-a", "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", 2, 20.0, 23.0),
            ("entity-b", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 1, 30.0, 32.0),
            ("entity-b", "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", 2, 40.0, 44.0),
        ),
        horizon_weights=((1, "0.5"), (2, "0.5")),
        entity_weights=(("entity-a", "0.25"), ("entity-b", "0.75")),
        scored_origin_set_id="shared-local-origin-panel",
    )

    result = score_point_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["comparison_status"] == "comparable"
    assert result.body["failure_reason_code"] is None
    assert result.body["per_horizon"] == [
        {"horizon": 1, "valid_origin_count": 2, "mean_point_loss": 1.75},
        {"horizon": 2, "valid_origin_count": 2, "mean_point_loss": 3.75},
    ]
    assert result.body["aggregated_primary_score"] == pytest.approx(2.75)


def test_build_comparison_universe_rejects_mismatched_entity_panels() -> None:
    score_policy_ref = TypedRef(
        schema_name="point_score_policy_manifest@1.0.0",
        object_id="shared-local-point-policy",
    )
    candidate_key = ComparisonKey(
        forecast_object_type="point",
        score_policy_ref=score_policy_ref,
        horizon_set=(1,),
        scored_origin_set_id="shared-panel",
        entity_panel=("entity-a", "entity-b"),
        entity_weights=(
            {"entity": "entity-a", "weight": "0.5"},
            {"entity": "entity-b", "weight": "0.5"},
        ),
    )
    baseline_key = replace(
        candidate_key,
        entity_panel=("entity-a", "entity-c"),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        build_comparison_universe(
            selected_candidate_id="shared-local-candidate",
            baseline_id="constant_baseline",
            candidate_primary_score=1.0,
            baseline_primary_score=2.0,
            candidate_comparison_key=candidate_key,
            baseline_comparison_key=baseline_key,
        )

    assert exc_info.value.code == "comparison_key_mismatch"
    assert exc_info.value.details["entity_panel"] == {
        "candidate": ["entity-a", "entity-b"],
        "baseline": ["entity-a", "entity-c"],
    }


def test_build_comparison_universe_allows_missing_baseline_composition_signature() -> (
    None
):
    score_policy_ref = TypedRef(
        schema_name="point_score_policy_manifest@1.0.0",
        object_id="shared-local-point-policy",
    )
    candidate_key = ComparisonKey(
        forecast_object_type="point",
        score_policy_ref=score_policy_ref,
        horizon_set=(1,),
        scored_origin_set_id="shared-panel",
        entity_panel=("entity-a", "entity-b"),
        entity_weights=(
            {"entity": "entity-a", "weight": "0.5"},
            {"entity": "entity-b", "weight": "0.5"},
        ),
        composition_signature="shared-local:panel-v1",
    )
    baseline_key = replace(candidate_key, composition_signature=None)

    universe = build_comparison_universe(
        selected_candidate_id="shared-local-candidate",
        baseline_id="constant_baseline",
        candidate_primary_score=1.0,
        baseline_primary_score=2.0,
        candidate_comparison_key=candidate_key,
        baseline_comparison_key=baseline_key,
    )

    assert universe.comparison_class_status == "comparable"


def _score_policy_manifest(
    *,
    catalog,
    horizon_weights: tuple[tuple[int, str], ...],
    entity_aggregation_mode: str,
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": "shared_local_point_policy",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": "point",
            "point_loss_id": "absolute_error",
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [
                {"horizon": horizon, "weight": weight}
                for horizon, weight in horizon_weights
            ],
            "entity_aggregation_mode": entity_aggregation_mode,
            "secondary_diagnostic_ids": [],
            "forbidden_primary_metric_ids": [],
            "lower_is_better": True,
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )


def _prediction_artifact(
    *,
    catalog,
    candidate_id: str,
    score_policy: ManifestEnvelope,
    rows: tuple[tuple[str, str, str, int, float, float], ...],
    horizon_weights: tuple[tuple[int, str], ...],
    entity_weights: tuple[tuple[str, str], ...],
    scored_origin_set_id: str,
) -> ManifestEnvelope:
    scored_origin_panel = [
        {
            "scored_origin_id": f"{candidate_id}_origin_{index}",
            "entity": entity,
            "origin_time": origin_time,
            "available_at": available_at,
            "horizon": horizon,
        }
        for index, (entity, origin_time, available_at, horizon, _, _) in enumerate(rows)
    ]
    return PredictionArtifactManifest(
        prediction_artifact_id=f"{candidate_id}_prediction",
        candidate_id=candidate_id,
        stage_id="confirmatory_holdout",
        fit_window_id="fit_window",
        test_window_id="confirmatory_segment",
        model_freeze_status="global_finalist_frozen",
        refit_rule_applied="pre_holdout_development_refit",
        score_policy_ref=score_policy.ref,
        rows=tuple(
            PredictionRow(
                entity=entity,
                origin_time=origin_time,
                available_at=available_at,
                horizon=horizon,
                point_forecast=point_forecast,
                realized_observation=realized_observation,
            )
            for (
                entity,
                origin_time,
                available_at,
                horizon,
                point_forecast,
                realized_observation,
            ) in rows
        ),
        score_law_id=str(score_policy.body["point_loss_id"]),
        horizon_weights=tuple(
            {
                "horizon": horizon,
                "weight": weight,
            }
            for horizon, weight in horizon_weights
        ),
        entity_panel=tuple(entity for entity, _ in entity_weights),
        entity_weights=tuple(
            {"entity": entity, "weight": weight} for entity, weight in entity_weights
        ),
        scored_origin_panel=tuple(scored_origin_panel),
        scored_origin_set_id=scored_origin_set_id,
        comparison_key={
            "forecast_object_type": "point",
            "horizon_set": [horizon for horizon, _ in horizon_weights],
            "score_law_id": score_policy.body["point_loss_id"],
            "scored_origin_set_id": scored_origin_set_id,
            "entity_panel": [entity for entity, _ in entity_weights],
            "entity_weights": [
                {"entity": entity, "weight": weight}
                for entity, weight in entity_weights
            ],
        },
        missing_scored_origins=(),
        timeguard_checks=tuple(
            {
                "scored_origin_id": scored_origin["scored_origin_id"],
                "expected_available_at": scored_origin["available_at"],
                "observed_available_at": scored_origin["available_at"],
                "status": "passed",
            }
            for scored_origin in scored_origin_panel
        ),
    ).to_manifest(catalog)
