from __future__ import annotations

from euclid.workbench.service import (
    build_equation_summary,
    build_target_rows_from_history,
)


def test_build_target_rows_from_history_supports_price_and_return_targets() -> None:
    history = [
        {"date": "2024-01-02", "close": 100.0, "open": 99.0, "volume": 1000},
        {"date": "2024-01-03", "close": 102.0, "open": 100.0, "volume": 1200},
        {"date": "2024-01-04", "close": 101.0, "open": 103.0, "volume": 900},
    ]

    price_rows = build_target_rows_from_history(
        history,
        symbol="SPY",
        target_id="price_close",
    )
    return_rows = build_target_rows_from_history(
        history,
        symbol="SPY",
        target_id="daily_return",
    )
    log_rows = build_target_rows_from_history(
        history,
        symbol="SPY",
        target_id="log_return",
    )
    event_rows = build_target_rows_from_history(
        history,
        symbol="SPY",
        target_id="next_day_up",
    )

    assert [row["observed_value"] for row in price_rows] == [100.0, 102.0, 101.0]
    assert [row["event_time"] for row in return_rows] == [
        "2024-01-03T00:00:00Z",
        "2024-01-04T00:00:00Z",
    ]
    assert [round(float(row["observed_value"]), 6) for row in return_rows] == [
        0.02,
        -0.009804,
    ]
    assert [round(float(row["observed_value"]), 6) for row in log_rows] == [
        0.019803,
        -0.009852,
    ]
    assert [row["observed_value"] for row in event_rows] == [1.0, 0.0]
    assert all(row["series_id"] == "SPY" for row in event_rows)
    assert event_rows[0]["close"] == 102.0
    assert event_rows[0]["previous_close"] == 100.0


def test_build_equation_summary_renders_supported_families_honestly() -> None:
    dataset_rows = [
        {"event_time": "2024-01-02T00:00:00Z", "observed_value": 10.0},
        {"event_time": "2024-01-03T00:00:00Z", "observed_value": 12.0},
        {"event_time": "2024-01-04T00:00:00Z", "observed_value": 14.0},
    ]

    drift = build_equation_summary(
        candidate_id="operator_drift_candidate_v1",
        family_id="drift",
        parameter_summary={"intercept": 10.0, "slope": 2.0},
        structure_signature="drift:intercept=10.0,slope=2.0",
        dataset_rows=dataset_rows,
    )
    lag_affine = build_equation_summary(
        candidate_id="analytic_lag1_affine",
        family_id="analytic",
        parameter_summary={"intercept": 1.0, "lag_coefficient": 0.5},
        structure_signature="analytic:intercept=1.0,lag_coefficient=0.5",
        dataset_rows=dataset_rows,
    )

    assert drift["label"] == "y(t) = 10 + 2*t"
    assert [point["fitted_value"] for point in drift["curve"]] == [10.0, 12.0, 14.0]
    assert drift["delta_form_label"] is None

    assert lag_affine["label"] == "y(t) = 1 + 0.5*y(t-1)"
    assert lag_affine["curve"][0]["fitted_value"] == 10.0
    assert lag_affine["curve"][1]["fitted_value"] == 6.0
    assert lag_affine["curve"][2]["fitted_value"] == 7.0
    assert lag_affine["delta_form_label"] == "Δy(t) = 1 - 0.5*y(t-1)"


def test_build_equation_summary_supports_descriptive_recursive_and_algorithmic_fits() -> None:
    dataset_rows = [
        {"event_time": "2024-01-02T00:00:00Z", "observed_value": 10.0},
        {"event_time": "2024-01-03T00:00:00Z", "observed_value": 14.0},
        {"event_time": "2024-01-04T00:00:00Z", "observed_value": 18.0},
    ]

    recursive = build_equation_summary(
        candidate_id="recursive_level_smoother",
        family_id="recursive",
        parameter_summary={},
        structure_signature="recursive:level_smoother",
        dataset_rows=dataset_rows,
        literals={"alpha": 0.5},
        state={"level": 10.0, "step_count": 0},
    )
    algorithmic = build_equation_summary(
        candidate_id="algorithmic_running_half_average",
        family_id="algorithmic",
        parameter_summary={},
        structure_signature="algorithmic:running_half_average",
        dataset_rows=dataset_rows,
        literals={
            "algorithmic_program": (
                "(program (state (lit 0)) "
                "(next (div (add (state 0) (obs 0)) (lit 2))) "
                "(emit (state 0)))"
            )
        },
        state={"state_0": 0.0},
    )

    assert recursive["label"] == "level(t) = 0.5*x(t-1) + 0.5*level(t-1)"
    assert [point["fitted_value"] for point in recursive["curve"]] == [10.0, 10.0, 12.0]
    assert recursive["delta_form_label"] is None

    assert algorithmic["label"] == "y(t) = 0.5*y(t-1) + 0.5*x(t-1)"
    assert [round(point["fitted_value"], 4) for point in algorithmic["curve"]] == [
        0.0,
        5.0,
        9.5,
    ]
    assert algorithmic["delta_form_label"] is None
