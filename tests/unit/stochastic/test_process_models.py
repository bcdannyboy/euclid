from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.manifests.runtime_models import StochasticModelManifest
from euclid.modules.residual_history import (
    ForecastResidualRecord,
    residual_history_digest,
)
from euclid.stochastic.process_models import fit_residual_stochastic_model


def _residual_history_ref() -> TypedRef:
    return TypedRef("residual_history_manifest@1.0.0", "residual_history_fixture")


def _record(
    *,
    residual: float,
    horizon: int = 1,
    target_index: int = 3,
    replay_identity: str = "row",
) -> ForecastResidualRecord:
    return ForecastResidualRecord(
        candidate_id="candidate",
        fit_window_id="outer_fold_0",
        entity="demo-series",
        origin_index=target_index - horizon,
        origin_time=f"2026-01-{target_index - horizon + 1:02d}T00:00:00Z",
        origin_available_at=f"2026-01-{target_index - horizon + 1:02d}T00:00:00Z",
        target_index=target_index,
        target_event_time=f"2026-01-{target_index + 1:02d}T00:00:00Z",
        target_available_at=f"2026-01-{target_index + 1:02d}T00:00:00Z",
        horizon=horizon,
        point_forecast=10.0,
        realized_observation=10.0 + residual,
        residual=residual,
        split_role="development",
        residual_basis="observation_minus_point_forecast",
        time_safety_status="passed",
        replay_identity=f"{replay_identity}:h{horizon}:t{target_index}",
    )


def _residual_history() -> tuple[ForecastResidualRecord, ...]:
    return (
        _record(residual=-1.0, target_index=2, replay_identity="r0"),
        _record(residual=0.0, target_index=3, replay_identity="r1"),
        _record(residual=1.0, target_index=4, replay_identity="r2"),
    )


def test_production_mode_rejects_bare_synthetic_residual_lists() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        fit_residual_stochastic_model(
            candidate_id="candidate",
            residuals=(-1.0, 0.0, 1.0),
            point_path={1: 10.0},
            family_id="gaussian",
            residual_history_ref=_residual_history_ref(),
            evidence_status="production",
        )

    assert exc_info.value.code == "synthetic_residual_source_not_production"


def test_validated_residual_histories_are_accepted_for_production() -> None:
    model = fit_residual_stochastic_model(
        candidate_id="candidate",
        residual_history=_residual_history(),
        point_path={1: 10.0, 2: 11.0},
        family_id="student_t",
        horizon_scale_law="sqrt_horizon",
        residual_history_ref=_residual_history_ref(),
        student_t_degrees_of_freedom=7.0,
    )

    support = model.support_path()
    manifest = model.as_manifest()

    assert manifest["schema_name"] == "stochastic_model_manifest@1.0.0"
    assert manifest["production_evidence"] is True
    assert manifest["heuristic_gaussian_support"] is False
    assert manifest["evidence_status"] == "production"
    assert manifest["residual_history_ref"] == _residual_history_ref().as_dict()
    assert manifest["residual_count"] == 3
    assert manifest["min_count_policy"]["minimum_residual_count"] == 2
    assert manifest["residual_history_digest"] == residual_history_digest(
        _residual_history()
    )
    assert manifest["residual_source_kind"] == "validated_residual_history"
    assert manifest["horizon_coverage"] == [1]
    assert manifest["observation_family"] == "student_t"
    assert manifest["residual_family"] == "student_t"
    assert manifest["support_kind"] == "all_real"
    assert manifest["fitted_parameters"]["df"] == 7.0
    assert support[2].scale > support[1].scale
    assert model.replay_identity.startswith("stochastic-model:")


def test_production_residual_process_fit_requires_residual_history_evidence() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        fit_residual_stochastic_model(
            candidate_id="candidate",
            residual_history=_residual_history(),
            point_path={1: 10.0},
            family_id="gaussian",
            evidence_status="production",
        )

    assert exc_info.value.code == "missing_residual_history_evidence"


def test_heuristic_gaussian_support_is_compatibility_only() -> None:
    model = fit_residual_stochastic_model(
        candidate_id="candidate",
        residuals=(-1.0, 1.0),
        point_path={1: 10.0},
        family_id="gaussian",
    )

    manifest = model.as_manifest()

    assert manifest["production_evidence"] is False
    assert manifest["evidence_status"] == "compatibility"
    assert manifest["heuristic_gaussian_support"] is True
    assert manifest["residual_history_ref"] is None

    with pytest.raises(ContractValidationError) as exc_info:
        StochasticModelManifest(
            stochastic_model_id="bad_production_heuristic",
            candidate_id="candidate",
            residual_history_ref=_residual_history_ref(),
            observation_family="gaussian",
            residual_family="gaussian",
            support_kind="all_real",
            horizon_scale_law="sqrt_horizon",
            fitted_parameters={"location": 0.0, "scale": 1.0},
            residual_count=2,
            min_count_policy={"minimum_residual_count": 2},
            evidence_status="production",
            heuristic_gaussian_support=True,
            replay_identity="stochastic-model:bad",
        ).body()

    assert exc_info.value.code == "heuristic_gaussian_support_not_production"


def test_minimum_residual_count_and_coverage_policies_are_enforced() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        fit_residual_stochastic_model(
            candidate_id="candidate",
            residual_history=(_record(residual=0.0),),
            point_path={1: 10.0},
            family_id="gaussian",
            min_residual_count=2,
            residual_history_ref=_residual_history_ref(),
        )

    assert exc_info.value.code == "insufficient_stochastic_training_support"

    with pytest.raises(ContractValidationError) as coverage_exc:
        fit_residual_stochastic_model(
            candidate_id="candidate",
            residual_history=_residual_history(),
            point_path={1: 10.0, 3: 12.0},
            family_id="gaussian",
            required_horizon_set=(1, 3),
            residual_history_ref=_residual_history_ref(),
        )

    assert coverage_exc.value.code == "insufficient_residual_horizon_coverage"


def test_gaussian_parameter_map_uses_validated_residual_history() -> None:
    model = fit_residual_stochastic_model(
        candidate_id="candidate",
        residual_history=_residual_history(),
        point_path={1: 10.0},
        family_id="gaussian",
        residual_history_ref=_residual_history_ref(),
    )

    assert model.residual_parameter_summary == {
        "location": 0.0,
        "scale": pytest.approx((2.0 / 3.0) ** 0.5),
    }
    assert model.as_manifest()["residual_family"] == "gaussian"


def test_student_t_parameter_map_records_degrees_of_freedom() -> None:
    model = fit_residual_stochastic_model(
        candidate_id="candidate",
        residual_history=_residual_history(),
        point_path={1: 10.0},
        family_id="student_t",
        residual_history_ref=_residual_history_ref(),
        student_t_degrees_of_freedom=9.0,
    )

    assert model.residual_parameter_summary["location"] == 0.0
    assert model.residual_parameter_summary["scale"] == pytest.approx(
        (2.0 / 3.0) ** 0.5
    )
    assert model.residual_parameter_summary["df"] == 9.0


def test_laplace_parameter_map_uses_median_and_mean_absolute_deviation() -> None:
    history = (
        _record(residual=-2.0, target_index=2, replay_identity="r0"),
        _record(residual=0.0, target_index=3, replay_identity="r1"),
        _record(residual=4.0, target_index=4, replay_identity="r2"),
    )

    model = fit_residual_stochastic_model(
        candidate_id="candidate",
        residual_history=history,
        point_path={1: 10.0},
        family_id="laplace",
        residual_history_ref=_residual_history_ref(),
    )

    assert model.residual_parameter_summary == {
        "location": 0.0,
        "scale": 2.0,
    }
    assert model.support_path()[1].distribution_family == "laplace_location_scale"


def test_unsupported_residual_family_or_support_combination_fails_clearly() -> None:
    with pytest.raises(ContractValidationError) as family_exc:
        fit_residual_stochastic_model(
            candidate_id="candidate",
            residual_history=_residual_history(),
            point_path={1: 10.0},
            family_id="poisson",
            residual_history_ref=_residual_history_ref(),
        )

    assert family_exc.value.code == "unsupported_stochastic_process_family"

    with pytest.raises(ContractValidationError) as support_exc:
        fit_residual_stochastic_model(
            candidate_id="candidate",
            residual_history=_residual_history(),
            point_path={1: 10.0},
            family_id="gaussian",
            residual_history_ref=_residual_history_ref(),
            support_kind="positive_real",
        )

    assert support_exc.value.code == "unsupported_residual_support_kind"


def test_compatibility_residual_lists_still_reject_nonfinite_values() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        fit_residual_stochastic_model(
            candidate_id="candidate",
            residuals=(0.0, float("nan")),
            point_path={1: 10.0},
            family_id="gaussian",
        )

    assert exc_info.value.code == "nonfinite_stochastic_training_value"
