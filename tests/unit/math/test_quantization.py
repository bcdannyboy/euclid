from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.math.quantization import (
    FixedStepMidTreadQuantizer,
    QuantizationPolicy,
    ResolvedQuantization,
    resolve_quantizer,
)


def test_fixed_step_policy_preserves_legacy_mid_tread_quantization() -> None:
    resolved = resolve_quantizer(
        QuantizationPolicy(
            quantization_mode="fixed_step_mid_tread",
            quantization_step="0.5",
        ),
        observed_values=(0.0, 1.0, -1.0),
    )

    assert isinstance(resolved, ResolvedQuantization)
    assert resolved.quantization_mode == "fixed_step_mid_tread"
    assert resolved.quantizer.step_string == "0.5"
    assert resolved.quantizer.quantize_sequence((0.0, 0.24, 0.25, -0.25)) == (
        0,
        0,
        1,
        -1,
    )
    assert FixedStepMidTreadQuantizer.from_string("0.5").step_string == "0.5"


def test_measurement_resolution_policy_uses_declared_resolution_as_step() -> None:
    resolved = resolve_quantizer(
        QuantizationPolicy(
            quantization_mode="measurement_resolution_mid_tread",
            measurement_resolution="0.25",
        ),
        observed_values=(1.0, 1.5, 2.0),
    )

    assert resolved.quantization_mode == "measurement_resolution_mid_tread"
    assert resolved.quantizer.step_string == "0.25"
    assert resolved.quantizer.quantize_index(1.125) == 5
    assert resolved.quantizer.representative(5) == pytest.approx(1.25)


def test_scale_adaptive_policy_is_equivariant_under_positive_rescaling() -> None:
    base = resolve_quantizer(
        QuantizationPolicy(
            quantization_mode="scale_adaptive_mid_tread",
            scale_fraction="0.1",
            zero_scale_fallback_step="0.25",
        ),
        observed_values=(-10.0, 0.0, 10.0),
    )
    scaled = resolve_quantizer(
        QuantizationPolicy(
            quantization_mode="scale_adaptive_mid_tread",
            scale_fraction="0.1",
            zero_scale_fallback_step="0.25",
        ),
        observed_values=(-30.0, 0.0, 30.0),
    )

    assert base.quantizer.step_string == "1"
    assert scaled.quantizer.step_string == "3"
    assert base.quantizer.quantize_sequence((-10.0, 0.0, 10.0)) == (
        scaled.quantizer.quantize_sequence((-30.0, 0.0, 30.0))
    )


def test_scale_adaptive_policy_uses_zero_scale_fallback() -> None:
    resolved = resolve_quantizer(
        QuantizationPolicy(
            quantization_mode="scale_adaptive_mid_tread",
            scale_fraction="0.1",
            zero_scale_fallback_step="0.25",
        ),
        observed_values=(0.0, 0.0, 0.0),
    )

    assert resolved.quantizer.step_string == "0.25"
    assert resolved.scale_reference == 0.0


@pytest.mark.parametrize(
    "policy",
    (
        QuantizationPolicy(quantization_mode="fixed_step_mid_tread"),
        QuantizationPolicy(
            quantization_mode="measurement_resolution_mid_tread",
            measurement_resolution="0",
        ),
        QuantizationPolicy(
            quantization_mode="scale_adaptive_mid_tread",
            scale_fraction="-0.1",
            zero_scale_fallback_step="0.25",
        ),
        QuantizationPolicy(
            quantization_mode="scale_adaptive_mid_tread",
            scale_fraction="0.1",
            zero_scale_fallback_step="0",
        ),
        QuantizationPolicy(quantization_mode="not_a_quantizer"),
    ),
)
def test_invalid_quantization_policy_fails_closed(policy: QuantizationPolicy) -> None:
    with pytest.raises(ContractValidationError):
        resolve_quantizer(policy, observed_values=(1.0, 2.0))
