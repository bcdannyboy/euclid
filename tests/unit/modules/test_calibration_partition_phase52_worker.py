from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from euclid.contracts.loader import load_contract_catalog
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import PredictionArtifactManifest
from euclid.modules import calibration as calibration_module
from euclid.modules.calibration import (
    build_calibration_contract,
    evaluate_prediction_calibration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class _IntervalCalibrationRow:
    origin_time: str
    available_at: str
    horizon: int
    nominal_coverage: float
    lower_bound: float
    upper_bound: float
    realized_observation: float
    entity: str | None = None
    regime: str | None = None

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "origin_time": self.origin_time,
            "available_at": self.available_at,
            "horizon": self.horizon,
            "nominal_coverage": self.nominal_coverage,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "realized_observation": self.realized_observation,
        }
        if self.entity is not None:
            body["entity"] = self.entity
        if self.regime is not None:
            body["regime"] = self.regime
        return body


def test_calibration_partition_has_stable_horizon_entity_regime_identity() -> None:
    assert hasattr(calibration_module, "CalibrationPartition")

    partition = calibration_module.CalibrationPartition(
        horizon=3,
        entity_id="plant_a",
        regime_id="stress",
    )

    assert partition.as_dict() == {
        "partition_id": "horizon=3|entity=plant_a|regime=stress",
        "horizon": 3,
        "entity_id": "plant_a",
        "regime_id": "stress",
    }


def test_horizon_partitions_prevent_implicit_pooling_undercoverage() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    rows = _coverage_rows(
        horizon=1,
        hit_count=8,
        miss_count=2,
        origin_start=1,
    ) + _coverage_rows(
        horizon=3,
        hit_count=0,
        miss_count=2,
        origin_start=20,
    )
    artifact = _prediction_artifact(catalog=catalog, candidate_id="horizon", rows=rows)
    contract = _calibration_contract(
        catalog=catalog,
        thresholds={
            "max_abs_coverage_gap": 0.2,
            "minimum_sample_count": 2,
        },
    )

    result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=contract,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["status"] == "failed"
    assert result.body["failure_reason_code"] == "calibration_failed"
    diagnostic = result.body["diagnostics"][0]
    assert diagnostic["diagnostic_id"] == "nominal_coverage"
    assert diagnostic["status"] == "failed"
    assert [
        item["partition"]["horizon"] for item in diagnostic["partition_diagnostics"]
    ] == [1, 3]
    horizon_three = diagnostic["partition_diagnostics"][1]
    assert horizon_three["partition"]["partition_id"] == "horizon=3"
    assert horizon_three["level_diagnostics"][0]["empirical_coverage"] == 0.0
    assert horizon_three["status"] == "failed"


def test_entity_and_regime_partition_undercoverage_blocks_calibration() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    rows = _coverage_rows(
        horizon=1,
        hit_count=8,
        miss_count=0,
        origin_start=1,
        entity="plant_a",
        regime="calm",
    ) + _coverage_rows(
        horizon=1,
        hit_count=0,
        miss_count=2,
        origin_start=20,
        entity="plant_b",
        regime="stress",
    )
    artifact = _prediction_artifact(
        catalog=catalog, candidate_id="entity_regime", rows=rows
    )
    contract = _calibration_contract(
        catalog=catalog,
        thresholds={
            "max_abs_coverage_gap": 0.05,
            "minimum_sample_count": 2,
        },
        partition_by_entity=True,
        partition_by_regime=True,
    )

    result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=contract,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["status"] == "failed"
    diagnostic = result.body["diagnostics"][0]
    assert [
        item["partition"]["partition_id"]
        for item in diagnostic["partition_diagnostics"]
    ] == [
        "horizon=1|entity=plant_a|regime=calm",
        "horizon=1|entity=plant_b|regime=stress",
    ]
    stress_partition = diagnostic["partition_diagnostics"][1]
    assert stress_partition["partition"]["regime_id"] == "stress"
    assert stress_partition["status"] == "failed"
    assert stress_partition["level_diagnostics"][0]["empirical_coverage"] == 0.0


def test_minimum_calibration_count_is_enforced_per_partition() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    rows = _coverage_rows(horizon=1, hit_count=2, miss_count=0, origin_start=1)
    rows += _coverage_rows(horizon=3, hit_count=2, miss_count=0, origin_start=20)
    artifact = _prediction_artifact(
        catalog=catalog, candidate_id="partition_min", rows=rows
    )
    contract = _calibration_contract(
        catalog=catalog,
        thresholds={
            "max_abs_coverage_gap": 1.0,
            "minimum_sample_count": 3,
        },
    )

    result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=contract,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["status"] == "failed"
    assert (
        result.body["failure_reason_code"] == "insufficient_calibration_partition_count"
    )
    diagnostic = result.body["diagnostics"][0]
    assert diagnostic["diagnostic_id"] == "calibration_partition_sample_count"
    assert diagnostic["minimum_sample_count"] == 3
    assert [
        item["partition"]["partition_id"]
        for item in diagnostic["partition_diagnostics"]
    ] == ["horizon=1", "horizon=3"]
    assert [item["sample_size"] for item in diagnostic["partition_diagnostics"]] == [
        2,
        2,
    ]


def test_valid_explicit_horizon_pooling_uses_declared_pooled_partition() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    rows = _coverage_rows(
        horizon=1,
        hit_count=8,
        miss_count=2,
        origin_start=1,
    ) + _coverage_rows(
        horizon=3,
        hit_count=0,
        miss_count=2,
        origin_start=20,
    )
    artifact = _prediction_artifact(
        catalog=catalog, candidate_id="pooled_horizon", rows=rows
    )
    contract = _calibration_contract(
        catalog=catalog,
        thresholds={
            "max_abs_coverage_gap": 0.2,
            "minimum_sample_count": 2,
        },
        horizon_pooling_policy="explicit_horizon_pooling_v1",
    )

    result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=contract,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["status"] == "passed"
    diagnostic = result.body["diagnostics"][0]
    assert diagnostic["status"] == "passed"
    assert len(diagnostic["partition_diagnostics"]) == 1
    partition = diagnostic["partition_diagnostics"][0]["partition"]
    assert partition["partition_id"] == "horizon=pooled"
    assert partition["horizon"] == "pooled"
    assert partition["pooled_horizons"] == [1, 3]


def test_empirical_coverage_lower_bound_diagnostic_blocks_weak_evidence() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    rows = _coverage_rows(horizon=1, hit_count=8, miss_count=2, origin_start=1)
    artifact = _prediction_artifact(
        catalog=catalog,
        candidate_id="coverage_lower_bound",
        rows=rows,
    )
    contract = _calibration_contract(
        catalog=catalog,
        thresholds={
            "max_abs_coverage_gap": 0.05,
            "minimum_empirical_coverage_lower_bound": 0.7,
        },
    )

    result = evaluate_prediction_calibration(
        catalog=catalog,
        calibration_contract_manifest=contract,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["status"] == "failed"
    assert result.body["failure_reason_code"] == "calibration_failed"
    level = result.body["diagnostics"][0]["level_diagnostics"][0]
    assert level["sample_size"] == 10
    assert level["empirical_coverage"] == 0.8
    assert level["empirical_coverage_lower_bound_method"] == "wilson_score"
    assert level["empirical_coverage_lower_bound"] < 0.7
    assert level["empirical_coverage_lower_bound_status"] == "failed"
    assert level["status"] == "failed"


def _coverage_rows(
    *,
    horizon: int,
    hit_count: int,
    miss_count: int,
    origin_start: int,
    entity: str | None = None,
    regime: str | None = None,
) -> tuple[_IntervalCalibrationRow, ...]:
    rows = []
    for offset in range(hit_count):
        origin_day = origin_start + offset
        rows.append(
            _interval_row(
                origin_day=origin_day,
                horizon=horizon,
                realized_observation=10.0,
                entity=entity,
                regime=regime,
            )
        )
    for offset in range(miss_count):
        origin_day = origin_start + hit_count + offset
        rows.append(
            _interval_row(
                origin_day=origin_day,
                horizon=horizon,
                realized_observation=12.0,
                entity=entity,
                regime=regime,
            )
        )
    return tuple(rows)


def _interval_row(
    *,
    origin_day: int,
    horizon: int,
    realized_observation: float,
    entity: str | None = None,
    regime: str | None = None,
) -> _IntervalCalibrationRow:
    return _IntervalCalibrationRow(
        origin_time=f"2026-01-{origin_day:02d}T00:00:00Z",
        available_at=f"2026-01-{origin_day + horizon:02d}T00:00:00Z",
        horizon=horizon,
        nominal_coverage=0.8,
        lower_bound=9.0,
        upper_bound=11.0,
        realized_observation=realized_observation,
        entity=entity,
        regime=regime,
    )


def _calibration_contract(
    *,
    catalog,
    thresholds: Mapping[str, float],
    partition_by_entity: bool = False,
    partition_by_regime: bool = False,
    horizon_pooling_policy: str | None = None,
) -> ManifestEnvelope:
    base = build_calibration_contract(
        catalog=catalog,
        forecast_object_type="interval",
        thresholds=thresholds,
    )
    body = dict(base.body)
    if partition_by_entity:
        body["partition_by_entity"] = True
    if partition_by_regime:
        body["partition_by_regime"] = True
    if horizon_pooling_policy is not None:
        body["horizon_pooling_policy"] = horizon_pooling_policy
    return ManifestEnvelope.build(
        schema_name=base.schema_name,
        module_id=base.module_id,
        body=body,
        catalog=catalog,
    )


def _score_policy_manifest(catalog) -> ManifestEnvelope:
    body = {
        "score_policy_id": "interval_score_policy_v1",
        "owner_prompt_id": "prompt.scoring-calibration-v1",
        "scope_id": "euclid_v1_binding_scope@1.0.0",
        "forecast_object_type": "interval",
        "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
        "horizon_weights": [{"horizon": 1, "weight": "1.0"}],
        "entity_aggregation_mode": "single_entity_only_no_cross_entity_aggregation",
        "comparison_class_rule": "identical_score_policy_required",
        "primary_score": "interval_score",
        "secondary_diagnostic_ids": [],
        "forbidden_primary_metric_ids": [],
        "lower_is_better": True,
    }
    return ManifestEnvelope.build(
        schema_name="interval_score_policy_manifest@1.0.0",
        module_id="scoring",
        body=body,
        catalog=catalog,
    )


def _prediction_artifact(
    *,
    catalog,
    candidate_id: str,
    rows: tuple[_IntervalCalibrationRow, ...],
) -> ManifestEnvelope:
    score_policy = _score_policy_manifest(catalog)
    scored_origin_panel = [
        {
            "scored_origin_id": f"{candidate_id}_origin_{index}",
            "origin_time": row.origin_time,
            "available_at": row.available_at,
            "horizon": row.horizon,
        }
        for index, row in enumerate(rows)
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
        rows=rows,
        forecast_object_type="interval",
        score_law_id="interval_score",
        horizon_weights=tuple(
            {"horizon": horizon, "weight": "1.0"}
            for horizon in sorted({row.horizon for row in rows})
        ),
        scored_origin_panel=tuple(scored_origin_panel),
        scored_origin_set_id=f"{candidate_id}_panel",
        comparison_key={
            "forecast_object_type": "interval",
            "horizon_set": sorted({row.horizon for row in rows}),
            "score_law_id": "interval_score",
            "scored_origin_set_id": f"{candidate_id}_panel",
        },
        entity_panel=tuple(sorted({row.entity for row in rows if row.entity})),
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
