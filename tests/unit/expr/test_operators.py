from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.expr.operators import OPERATOR_REGISTRY, get_operator


def test_operator_registry_covers_required_p02_operator_metadata() -> None:
    required = {
        "add",
        "sub",
        "mul",
        "div",
        "protected_div",
        "neg",
        "abs",
        "min",
        "max",
        "pow",
        "pow2",
        "sqrt",
        "exp",
        "log",
        "protected_log",
        "sin",
        "cos",
        "tan",
        "tanh",
        "sigmoid",
        "logit",
        "floor",
        "ceil",
        "clip",
        "where",
        "lag",
        "finite_difference",
        "rolling_mean",
        "rolling_sum",
        "cumulative_sum",
        "convolution",
        "seasonal_phase",
        "derivative_estimate",
        "integral_estimate",
        "location_parameter",
        "scale_parameter",
        "rate_parameter",
        "probability_parameter",
        "dispersion_parameter",
    }

    assert required <= set(OPERATOR_REGISTRY)
    for name in required:
        metadata = get_operator(name)
        assert metadata.name == name
        assert metadata.output_domain
        assert metadata.unit_rule
        assert metadata.differentiability
        assert metadata.safe_evaluation
        assert metadata.sympy_name is not None


def test_operator_lookup_fails_closed_for_unknown_operator() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        get_operator("external_publish_claim")

    assert exc_info.value.code == "unsupported_expression_operator"

