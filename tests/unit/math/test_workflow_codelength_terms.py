from __future__ import annotations

from euclid.math.codelength import description_components
from euclid.math.quantization import FixedStepMidTreadQuantizer
from euclid.operator_runtime.workflow import _description_components as operator_terms
from euclid.prototype.workflow import _description_components as prototype_terms


def test_prototype_and_operator_share_codelength_terms_for_same_policy() -> None:
    quantizer = FixedStepMidTreadQuantizer.from_string("0.5")
    kwargs = {
        "family_id": "linear_trend",
        "parameters": {"intercept": 1.0, "slope": 0.5},
        "fitted_values": (1.0, 1.5, 2.0),
        "actual_values": (1.0, 2.0, 2.5),
        "reference_bits": 32.0,
        "quantizer": quantizer,
    }

    assert prototype_terms(**kwargs) == operator_terms(**kwargs)
    assert prototype_terms(**kwargs) == description_components(**kwargs)
