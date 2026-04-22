from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.stochastic.event_definitions import EventDefinition
from euclid.stochastic.observation_models import get_observation_model


def test_event_probability_comes_from_declared_event_definition() -> None:
    event = EventDefinition.from_manifest(
        {
            "event_id": "above_10_usd",
            "variable": "target",
            "operator": "greater_than",
            "threshold": 10.0,
            "threshold_source": "declared_literal",
            "units": "USD",
            "scope": "declared_validation_scope",
            "calibration_required": True,
        }
    )
    model = get_observation_model("gaussian").bind({"location": 11.0, "scale": 1.0})

    assert event.evaluate(11.5) is True
    assert event.evaluate(9.5) is False
    assert event.probability(model) > 0.5
    assert event.as_manifest()["threshold_source"] == "declared_literal"


def test_event_definition_rejects_hardcoded_or_undeclared_thresholds() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        EventDefinition.from_manifest(
            {
                "event_id": "origin_target_shortcut",
                "operator": "greater_than_or_equal",
                "threshold": 10.0,
                "threshold_source": "origin_target",
            }
        )

    assert exc_info.value.code == "undeclared_event_threshold_source"
