from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from euclid.modules.residual_history import (
    ForecastResidualRecord,
    ResidualHistorySummary,
    residual_history_digest,
    summarize_residual_history,
    validate_residual_history,
)


def _record(
    *,
    target_index: int = 4,
    horizon: int = 1,
    entity: str = "demo-series",
    split_role: str = "development",
    replay_identity: str | None = None,
) -> ForecastResidualRecord:
    return ForecastResidualRecord(
        candidate_id="candidate.mean",
        fit_window_id="outer_fold_0",
        entity=entity,
        origin_index=target_index - horizon,
        origin_time=f"2026-01-0{target_index - horizon + 1}T00:00:00Z",
        origin_available_at=f"2026-01-0{target_index - horizon + 1}T00:00:00Z",
        target_index=target_index,
        target_event_time=f"2026-01-0{target_index + 1}T00:00:00Z",
        target_available_at=f"2026-01-0{target_index + 1}T00:00:00Z",
        horizon=horizon,
        point_forecast=12.0,
        realized_observation=15.0,
        residual=3.0,
        split_role=split_role,
        residual_basis="observation_minus_point_forecast",
        time_safety_status="passed",
        replay_identity=(
            replay_identity or f"candidate.mean:outer_fold_0:{entity}:{target_index}"
        ),
    )


def test_forecast_residual_record_is_immutable_and_serializes_contract_fields() -> None:
    record = _record()

    with pytest.raises(FrozenInstanceError):
        record.residual = 0.0  # type: ignore[misc]

    payload = record.as_dict()

    assert payload["candidate_id"] == "candidate.mean"
    assert payload["fit_window_id"] == "outer_fold_0"
    assert payload["entity"] == "demo-series"
    assert payload["origin_index"] == 3
    assert payload["origin_available_at"] == "2026-01-04T00:00:00Z"
    assert payload["target_index"] == 4
    assert payload["target_available_at"] == "2026-01-05T00:00:00Z"
    assert payload["horizon"] == 1
    assert payload["point_forecast"] == 12.0
    assert payload["realized_observation"] == 15.0
    assert payload["realized_value"] == 15.0
    assert payload["residual"] == 3.0
    assert payload["split_role"] == "development"
    assert payload["time_safety_status"] == "passed"
    assert payload["replay_identity"].startswith("candidate.mean:outer_fold_0")


def test_residual_history_summary_computes_digest_and_residual_moments() -> None:
    records = (
        _record(target_index=3, replay_identity="r0"),
        _record(target_index=4, replay_identity="r1"),
        _record(target_index=5, horizon=2, entity="other", replay_identity="r2"),
    )

    summary = summarize_residual_history(records)

    assert isinstance(summary, ResidualHistorySummary)
    assert summary.candidate_id == "candidate.mean"
    assert summary.fit_window_id == "outer_fold_0"
    assert summary.residual_count == 3
    assert summary.horizon_set == (1, 2)
    assert summary.entity_count == 2
    assert summary.split_roles == ("development",)
    assert summary.residual_mean == 3.0
    assert summary.residual_rmse == 3.0
    assert summary.residual_history_digest == residual_history_digest(records)


def test_residual_history_digest_is_deterministic_across_row_order() -> None:
    records = (
        _record(target_index=3, replay_identity="r0"),
        _record(target_index=4, replay_identity="r1"),
    )

    assert residual_history_digest(records) == residual_history_digest(
        tuple(reversed(records))
    )


def test_residual_history_validation_requires_geometry_availability_and_identity() -> (
    None
):
    result = validate_residual_history(
        (
            {
                "candidate_id": "candidate.mean",
                "fit_window_id": "outer_fold_0",
                "entity": "demo-series",
                "origin_index": 2,
                "origin_time": "2026-01-03T00:00:00Z",
                "target_index": 3,
                "target_event_time": "2026-01-04T00:00:00Z",
                "horizon": 1,
                "point_forecast": 12.0,
                "realized_value": 15.0,
                "residual": 3.0,
                "residual_basis": "observation_minus_point_forecast",
                "time_safety_status": "passed",
            },
        )
    )

    assert result.status == "failed"
    assert {issue.code for issue in result.issues} >= {
        "missing_origin_or_target_availability",
        "missing_split_role_metadata",
        "missing_replay_identity",
    }
    assert any("origin_available_at" in issue.message for issue in result.issues)


def test_confirmatory_residuals_are_not_production_evidence() -> None:
    result = validate_residual_history((_record(split_role="confirmatory"),))

    assert result.status == "failed"
    assert result.reason_codes == (
        "confirmatory_residual_history_not_production_evidence",
    )
