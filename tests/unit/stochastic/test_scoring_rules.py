from __future__ import annotations

import math

import pytest
from scipy import stats

from euclid.contracts.errors import ContractValidationError
from euclid.stochastic.observation_models import get_observation_model
from euclid.stochastic.scoring_rules import (
    brier_score,
    continuous_ranked_probability_score,
    interval_score,
    log_score,
    pinball_loss,
)


def test_log_score_rejects_nonfinite_likelihood() -> None:
    model = get_observation_model("gaussian").bind(
        {"location": 0.0, "scale": float("nan")}
    )

    with pytest.raises(ContractValidationError) as exc_info:
        log_score(model, 0.0)

    assert exc_info.value.code == "nonfinite_likelihood"


def test_proper_scoring_rules_are_finite_for_valid_forecasts() -> None:
    model = get_observation_model("gaussian").bind({"location": 0.0, "scale": 1.0})

    assert math.isfinite(log_score(model, 0.25))
    assert math.isfinite(continuous_ranked_probability_score(model, 0.25))
    assert (
        interval_score(
            nominal_coverage=0.8,
            lower_bound=-1.0,
            upper_bound=1.0,
            observed=0.0,
        )
        == 2.0
    )
    assert pinball_loss(level=0.9, quantile=1.0, observed=1.5) == pytest.approx(0.45)
    assert brier_score(probability=0.8, realized_event=True) == pytest.approx(0.04)


def test_gaussian_crps_matches_scipy_normal_reference_formula() -> None:
    model = get_observation_model("gaussian").bind({"location": 0.0, "scale": 2.0})
    z = 0.25 / 2.0
    expected = 2.0 * (
        z * (2.0 * stats.norm.cdf(z) - 1.0)
        + (2.0 * stats.norm.pdf(z))
        - (1.0 / math.sqrt(math.pi))
    )

    assert continuous_ranked_probability_score(model, 0.25) == pytest.approx(
        expected
    )


def test_interval_quantile_and_brier_inputs_fail_closed() -> None:
    with pytest.raises(ContractValidationError):
        interval_score(
            nominal_coverage=1.0,
            lower_bound=0.0,
            upper_bound=1.0,
            observed=0.5,
        )
    with pytest.raises(ContractValidationError):
        pinball_loss(level=0.0, quantile=1.0, observed=1.5)
    with pytest.raises(ContractValidationError):
        brier_score(probability=1.2, realized_event=True)
