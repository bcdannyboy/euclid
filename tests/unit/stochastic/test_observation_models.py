from __future__ import annotations

import math

import pytest
from scipy import stats

from euclid.contracts.errors import ContractValidationError
from euclid.stochastic.observation_models import (
    MixtureObservationModel,
    ObservationModelSpec,
    get_observation_model,
)


@pytest.mark.parametrize(
    ("family", "parameters", "value"),
    (
        ("gaussian", {"location": 0.0, "scale": 1.0}, 0.0),
        ("student_t", {"location": 0.0, "scale": 1.0, "df": 5.0}, 0.5),
        ("laplace", {"location": 0.0, "scale": 1.0}, 0.25),
        ("poisson", {"rate": 3.0}, 2.0),
        ("negative_binomial", {"mean": 3.0, "dispersion": 2.0}, 1.0),
        ("bernoulli", {"probability": 0.7}, 1.0),
        ("beta", {"alpha": 2.0, "beta": 3.0}, 0.4),
        ("lognormal", {"log_location": 0.0, "log_scale": 0.25}, 1.0),
    ),
)
def test_observation_families_validate_parameters_and_score_finite_likelihood(
    family: str,
    parameters: dict[str, float],
    value: float,
) -> None:
    model = get_observation_model(family).bind(parameters)

    assert model.family_id == family
    assert model.support_contains(value) is True
    assert math.isfinite(model.log_likelihood(value))


@pytest.mark.parametrize(
    ("family", "parameters", "value"),
    (
        ("gaussian", {"location": 0.0, "scale": 0.0}, 0.0),
        ("student_t", {"location": 0.0, "scale": 1.0, "df": 1.0}, 0.0),
        ("poisson", {"rate": -1.0}, 0.0),
        ("bernoulli", {"probability": 1.5}, 1.0),
        ("beta", {"alpha": 1.0, "beta": 1.0}, 1.2),
    ),
)
def test_invalid_stochastic_parameters_or_support_fail_closed(
    family: str,
    parameters: dict[str, float],
    value: float,
) -> None:
    with pytest.raises(ContractValidationError):
        get_observation_model(family).bind(parameters).log_likelihood(value)


def test_mixture_model_requires_explicit_weight_simplex() -> None:
    gaussian = ObservationModelSpec("gaussian").bind({"location": 0.0, "scale": 1.0})

    with pytest.raises(ContractValidationError) as exc_info:
        MixtureObservationModel(
            components=(gaussian, gaussian),
            weights=(0.75, 0.75),
        ).log_likelihood(0.0)

    assert exc_info.value.code == "invalid_mixture_weight_simplex"


@pytest.mark.parametrize(
    ("family", "parameters", "value", "expected"),
    (
        (
            "student_t",
            {"location": 0.0, "scale": 2.0, "df": 5.0},
            0.75,
            stats.t(df=5.0, loc=0.0, scale=2.0).cdf(0.75),
        ),
        (
            "poisson",
            {"rate": 3.0},
            4.0,
            stats.poisson(mu=3.0).cdf(4.0),
        ),
        (
            "lognormal",
            {"log_location": 0.25, "log_scale": 0.5},
            1.2,
            stats.lognorm(s=0.5, scale=math.exp(0.25)).cdf(1.2),
        ),
    ),
)
def test_observation_model_cdf_uses_scipy_distribution_semantics(
    family: str,
    parameters: dict[str, float],
    value: float,
    expected: float,
) -> None:
    model = get_observation_model(family).bind(parameters)

    assert model.distribution_backend == "scipy.stats"
    assert model.cdf(value) == pytest.approx(expected)
