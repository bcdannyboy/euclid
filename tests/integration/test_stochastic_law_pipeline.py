from __future__ import annotations

import pytest

from euclid.stochastic.observation_models import get_observation_model


def test_stochastic_law_pipeline_uses_explicit_scipy_observation_model() -> None:
    model = get_observation_model("student_t").bind(
        {"location": 0.0, "scale": 1.0, "df": 5.0}
    )

    assert model.distribution_backend == "scipy.stats"
    assert model.cdf(0.0) == pytest.approx(0.5)
    assert model.survival(0.0) == pytest.approx(0.5)
