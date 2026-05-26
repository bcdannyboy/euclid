from __future__ import annotations

from pathlib import Path

from euclid.contracts.loader import load_contract_catalog
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import PredictionArtifactManifest, PredictionRow
from euclid.modules.scoring import evaluate_point_comparators

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_point_comparator_record_emits_paired_loss_differential_stream() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _point_score_policy_manifest(
        catalog=catalog,
        point_loss_id="absolute_error",
        horizon_weights=((1, "0.25"), (2, "0.75")),
    )
    candidate = _prediction_artifact(
        catalog=catalog,
        candidate_id="candidate",
        score_policy=score_policy,
        scored_origin_set_id="confirmatory-row-set",
        rows=_panel_rows(offset=0.0),
        horizon_weights=((1, "0.25"), (2, "0.75")),
    )
    baseline = _prediction_artifact(
        catalog=catalog,
        candidate_id="baseline",
        score_policy=score_policy,
        scored_origin_set_id="confirmatory-row-set",
        rows=_panel_rows(offset=4.0),
        horizon_weights=((1, "0.25"), (2, "0.75")),
    )

    result = evaluate_point_comparators(
        catalog=catalog,
        score_policy_manifest=score_policy,
        candidate_prediction_artifact=candidate,
        baseline_registry_manifest=_baseline_registry_manifest(
            catalog=catalog,
            score_policy=score_policy,
            declarations=(("baseline", "baseline", "constant"),),
            primary_baseline_id="baseline",
        ),
        comparator_prediction_artifacts={"baseline": baseline},
        practical_significance_margin=0.1,
    )

    record = result.comparison_universe.body["paired_comparison_records"][0]
    stream = record["paired_predictive_test_result"]["paired_loss_differential_stream"]
    assert stream["schema_name"] == "paired_loss_differential_stream@1.0.0"
    assert stream["loss_id"] == "absolute_error"
    assert stream["candidate_id"] == "candidate"
    assert stream["baseline_id"] == "baseline"
    assert stream["candidate_row_set_id"] == "confirmatory-row-set"
    assert stream["baseline_row_set_id"] == "confirmatory-row-set"
    assert stream["origin_ids"] == [
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
    ]
    assert stream["entity_ids"] == ["entity-a", "entity-b"]
    assert stream["raw_pair_count"] == 4
    assert stream["effective_sample_size"] == 4
    assert stream["block_count"] == 4
    assert stream["horizon_geometry"] == {
        "mode": "per_origin_complete_horizon_simplex_weighted_sum",
        "horizons": [1, 2],
        "horizon_weights": [
            {"horizon": 1, "weight": 0.25},
            {"horizon": 2, "weight": 0.75},
        ],
    }
    assert stream["pairs"] == [
        {
            "origin_id": "2026-01-01T00:00:00Z",
            "entity_id": "entity-a",
            "horizon_losses": [
                {
                    "horizon": 1,
                    "candidate_loss": 0.0,
                    "baseline_loss": 4.0,
                    "loss_differential": 4.0,
                },
                {
                    "horizon": 2,
                    "candidate_loss": 0.0,
                    "baseline_loss": 4.0,
                    "loss_differential": 4.0,
                },
            ],
            "candidate_primary_loss": 0.0,
            "baseline_primary_loss": 4.0,
            "loss_differential": 4.0,
        },
        {
            "origin_id": "2026-01-01T00:00:00Z",
            "entity_id": "entity-b",
            "horizon_losses": [
                {
                    "horizon": 1,
                    "candidate_loss": 0.0,
                    "baseline_loss": 4.0,
                    "loss_differential": 4.0,
                },
                {
                    "horizon": 2,
                    "candidate_loss": 0.0,
                    "baseline_loss": 4.0,
                    "loss_differential": 4.0,
                },
            ],
            "candidate_primary_loss": 0.0,
            "baseline_primary_loss": 4.0,
            "loss_differential": 4.0,
        },
        {
            "origin_id": "2026-01-02T00:00:00Z",
            "entity_id": "entity-a",
            "horizon_losses": [
                {
                    "horizon": 1,
                    "candidate_loss": 0.0,
                    "baseline_loss": 4.0,
                    "loss_differential": 4.0,
                },
                {
                    "horizon": 2,
                    "candidate_loss": 0.0,
                    "baseline_loss": 4.0,
                    "loss_differential": 4.0,
                },
            ],
            "candidate_primary_loss": 0.0,
            "baseline_primary_loss": 4.0,
            "loss_differential": 4.0,
        },
        {
            "origin_id": "2026-01-02T00:00:00Z",
            "entity_id": "entity-b",
            "horizon_losses": [
                {
                    "horizon": 1,
                    "candidate_loss": 0.0,
                    "baseline_loss": 4.0,
                    "loss_differential": 4.0,
                },
                {
                    "horizon": 2,
                    "candidate_loss": 0.0,
                    "baseline_loss": 4.0,
                    "loss_differential": 4.0,
                },
            ],
            "candidate_primary_loss": 0.0,
            "baseline_primary_loss": 4.0,
            "loss_differential": 4.0,
        },
    ]


def test_separately_averaged_losses_are_not_predictive_promotion_eligible() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _point_score_policy_manifest(
        catalog=catalog,
        point_loss_id="absolute_error",
        horizon_weights=((1, "1.0"),),
    )
    candidate = _prediction_artifact(
        catalog=catalog,
        candidate_id="candidate",
        score_policy=score_policy,
        scored_origin_set_id="confirmatory-row-set",
        rows=_single_horizon_rows(offset=0.0),
        horizon_weights=((1, "1.0"),),
    )
    paired_baseline = _prediction_artifact(
        catalog=catalog,
        candidate_id="paired_baseline",
        score_policy=score_policy,
        scored_origin_set_id="confirmatory-row-set",
        rows=_single_horizon_rows(offset=3.0),
        horizon_weights=((1, "1.0"),),
    )
    separately_averaged_baseline = _prediction_artifact(
        catalog=catalog,
        candidate_id="separately_averaged_baseline",
        score_policy=score_policy,
        scored_origin_set_id="confirmatory-row-set",
        rows=_single_horizon_rows(
            offset=3.0,
            origin_times=(
                "2026-02-01T00:00:00Z",
                "2026-02-02T00:00:00Z",
            ),
        ),
        horizon_weights=((1, "1.0"),),
    )

    result = evaluate_point_comparators(
        catalog=catalog,
        score_policy_manifest=score_policy,
        candidate_prediction_artifact=candidate,
        baseline_registry_manifest=_baseline_registry_manifest(
            catalog=catalog,
            score_policy=score_policy,
            declarations=(
                ("paired_baseline", "baseline", "constant"),
                ("separately_averaged_baseline", "baseline", "constant"),
            ),
            primary_baseline_id="paired_baseline",
        ),
        comparator_prediction_artifacts={
            "paired_baseline": paired_baseline,
            "separately_averaged_baseline": separately_averaged_baseline,
        },
        practical_significance_margin=0.1,
    )

    mismatched_record = result.comparison_universe.body["paired_comparison_records"][1]
    assert mismatched_record["comparison_status"] == "not_comparable"
    assert mismatched_record["failure_reason_code"] == (
        "paired_loss_stream_identity_mismatch"
    )
    assert mismatched_record["predictive_promotion_eligible"] is False
    assert "paired_predictive_test_result" not in mismatched_record


def test_one_element_paired_stream_abstains_with_insufficient_paired_count() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _point_score_policy_manifest(
        catalog=catalog,
        point_loss_id="absolute_error",
        horizon_weights=((1, "1.0"),),
    )
    candidate = _prediction_artifact(
        catalog=catalog,
        candidate_id="candidate",
        score_policy=score_policy,
        scored_origin_set_id="single-origin-row-set",
        rows=(
            (
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                1,
                "entity-a",
                10.0,
                10.0,
            ),
        ),
        horizon_weights=((1, "1.0"),),
    )
    baseline = _prediction_artifact(
        catalog=catalog,
        candidate_id="baseline",
        score_policy=score_policy,
        scored_origin_set_id="single-origin-row-set",
        rows=(
            (
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                1,
                "entity-a",
                14.0,
                10.0,
            ),
        ),
        horizon_weights=((1, "1.0"),),
    )

    result = evaluate_point_comparators(
        catalog=catalog,
        score_policy_manifest=score_policy,
        candidate_prediction_artifact=candidate,
        baseline_registry_manifest=_baseline_registry_manifest(
            catalog=catalog,
            score_policy=score_policy,
            declarations=(("baseline", "baseline", "constant"),),
            primary_baseline_id="baseline",
        ),
        comparator_prediction_artifacts={"baseline": baseline},
        practical_significance_margin=0.1,
    )

    record = result.comparison_universe.body["paired_comparison_records"][0]
    predictive_test = record["paired_predictive_test_result"]
    assert predictive_test["paired_loss_differential_stream"]["raw_pair_count"] == 1
    assert predictive_test["status"] == "abstained"
    assert predictive_test["promotion_allowed"] is False
    assert predictive_test["reason_codes"] == ["insufficient_paired_count"]


def _point_score_policy_manifest(
    *,
    catalog,
    point_loss_id: str,
    horizon_weights: tuple[tuple[int, str], ...],
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": f"policy_{point_loss_id}",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": "point",
            "point_loss_id": point_loss_id,
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [
                {"horizon": horizon, "weight": weight}
                for horizon, weight in horizon_weights
            ],
            "entity_aggregation_mode": (
                "per_entity_primary_score_then_declared_entity_weights"
            ),
            "secondary_diagnostic_ids": [],
            "forbidden_primary_metric_ids": [
                "root_mean_squared_error",
                "mean_absolute_percentage_error",
            ],
            "lower_is_better": True,
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )


def _baseline_registry_manifest(
    *,
    catalog,
    score_policy: ManifestEnvelope,
    declarations: tuple[tuple[str, str, str], ...],
    primary_baseline_id: str,
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="baseline_registry_manifest@1.1.0",
        module_id="evaluation_governance",
        body={
            "baseline_registry_id": "paired_stream_worker_registry_v1",
            "owner_prompt_id": "prompt.predictive-validation-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "primary_baseline_id": primary_baseline_id,
            "baseline_ids": [baseline_id for baseline_id, _, _ in declarations],
            "baseline_declarations": [
                {
                    "baseline_id": baseline_id,
                    "comparator_declaration_id": f"{baseline_id}_declaration",
                    "comparator_kind": comparator_kind,
                    "family_id": family_id,
                    "forecast_object_type": "point",
                    "freeze_rule": "frozen_before_confirmatory_access",
                }
                for baseline_id, comparator_kind, family_id in declarations
            ],
            "compatible_point_score_policy_ref": score_policy.ref.as_dict(),
        },
        catalog=catalog,
    )


def _prediction_artifact(
    *,
    catalog,
    candidate_id: str,
    score_policy: ManifestEnvelope,
    rows: tuple[tuple[str, str, int, str, float, float], ...],
    horizon_weights: tuple[tuple[int, str], ...],
    scored_origin_set_id: str,
) -> ManifestEnvelope:
    scored_origin_panel = [
        {
            "scored_origin_id": (
                f"{scored_origin_set_id}__{entity}__{origin_time}__h{horizon}"
            ),
            "origin_time": origin_time,
            "available_at": available_at,
            "horizon": horizon,
            "entity": entity,
        }
        for origin_time, available_at, horizon, entity, _, _ in rows
    ]
    entity_panel = tuple(dict.fromkeys(entity for *_, entity, _forecast, _realized in rows))
    entity_weight = "1.0" if len(entity_panel) == 1 else str(1 / len(entity_panel))
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
                origin_time=origin_time,
                available_at=available_at,
                horizon=horizon,
                entity=entity,
                point_forecast=point_forecast,
                realized_observation=realized_observation,
            )
            for (
                origin_time,
                available_at,
                horizon,
                entity,
                point_forecast,
                realized_observation,
            ) in rows
        ),
        score_law_id=str(score_policy.body["point_loss_id"]),
        horizon_weights=tuple(
            {"horizon": horizon, "weight": weight}
            for horizon, weight in horizon_weights
        ),
        entity_panel=entity_panel,
        entity_weights=tuple(
            {"entity": entity, "weight": entity_weight} for entity in entity_panel
        ),
        scored_origin_panel=tuple(scored_origin_panel),
        scored_origin_set_id=scored_origin_set_id,
        comparison_key={
            "forecast_object_type": "point",
            "horizon_set": [horizon for horizon, _ in horizon_weights],
            "score_law_id": score_policy.body["point_loss_id"],
            "scored_origin_set_id": scored_origin_set_id,
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


def _panel_rows(
    *,
    offset: float,
) -> tuple[tuple[str, str, int, str, float, float], ...]:
    rows: list[tuple[str, str, int, str, float, float]] = []
    for origin_index, origin_time in enumerate(
        ("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
        start=1,
    ):
        for entity in ("entity-a", "entity-b"):
            for horizon in (1, 2):
                realized = float((origin_index * 10) + horizon)
                rows.append(
                    (
                        origin_time,
                        f"2026-01-0{origin_index + horizon}T00:00:00Z",
                        horizon,
                        entity,
                        realized + offset,
                        realized,
                    )
                )
    return tuple(rows)


def _single_horizon_rows(
    *,
    offset: float,
    origin_times: tuple[str, str] = (
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
    ),
) -> tuple[tuple[str, str, int, str, float, float], ...]:
    rows: list[tuple[str, str, int, str, float, float]] = []
    for origin_index, origin_time in enumerate(origin_times, start=1):
        for entity in ("entity-a", "entity-b"):
            realized = float(origin_index * 10)
            rows.append(
                (
                    origin_time,
                    f"2026-03-0{origin_index}T00:00:00Z",
                    1,
                    entity,
                    realized + offset,
                    realized,
                )
            )
    return tuple(rows)
