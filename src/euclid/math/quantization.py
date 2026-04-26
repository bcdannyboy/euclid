from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Iterable, Sequence

from euclid.contracts.errors import ContractValidationError


def canonical_decimal_string(value: Decimal) -> str:
    normalized = value.normalize()
    rendered = format(normalized, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


@dataclass(frozen=True)
class FixedStepMidTreadQuantizer:
    step: Decimal

    @classmethod
    def from_string(cls, value: str) -> "FixedStepMidTreadQuantizer":
        try:
            step = Decimal(value)
        except InvalidOperation as exc:
            raise ContractValidationError(
                code="invalid_quantization_step",
                message="quantization_step must be a finite decimal string",
                field_path="quantization_step",
            ) from exc
        if step <= 0:
            raise ContractValidationError(
                code="invalid_quantization_step",
                message="quantization_step must be strictly positive",
                field_path="quantization_step",
            )
        return cls(step=step)

    @property
    def step_string(self) -> str:
        return canonical_decimal_string(self.step)

    def quantize_index(self, value: float) -> int:
        scaled = Decimal(str(value)) / self.step
        return int(scaled.to_integral_value(rounding=ROUND_HALF_UP))

    def representative(self, index: int) -> float:
        return float(self.step * Decimal(index))

    def quantize_value(self, value: float) -> float:
        return self.representative(self.quantize_index(value))

    def quantize_sequence(self, values: Iterable[float]) -> tuple[int, ...]:
        return tuple(self.quantize_index(value) for value in values)


@dataclass(frozen=True)
class QuantizationPolicy:
    quantization_mode: str = "fixed_step_mid_tread"
    quantization_step: str | None = None
    measurement_resolution: str | None = None
    scale_fraction: str | None = None
    scale_statistic: str = "max_abs"
    zero_scale_fallback_step: str = "0.5"


@dataclass(frozen=True)
class ResolvedQuantization:
    policy: QuantizationPolicy
    quantizer: FixedStepMidTreadQuantizer
    quantization_mode: str
    scale_reference: float | None = None

    @property
    def step_string(self) -> str:
        return self.quantizer.step_string

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "quantization_mode": self.quantization_mode,
            "quantization_step": self.step_string,
        }
        if self.scale_reference is not None:
            payload["scale_reference"] = self.scale_reference
        return payload


def resolve_quantizer(
    policy: QuantizationPolicy | None = None,
    *,
    observed_values: Sequence[float] = (),
) -> ResolvedQuantization:
    active_policy = policy or QuantizationPolicy(
        quantization_mode="fixed_step_mid_tread",
        quantization_step="0.5",
    )
    mode = active_policy.quantization_mode
    if mode == "fixed_step_mid_tread":
        if active_policy.quantization_step is None:
            _raise_invalid_policy(
                field_path="quantization_policy.quantization_step",
                message="fixed-step quantization requires quantization_step",
            )
        quantizer = FixedStepMidTreadQuantizer.from_string(
            active_policy.quantization_step
        )
        return ResolvedQuantization(
            policy=active_policy,
            quantizer=quantizer,
            quantization_mode=mode,
        )
    if mode == "measurement_resolution_mid_tread":
        if active_policy.measurement_resolution is None:
            _raise_invalid_policy(
                field_path="quantization_policy.measurement_resolution",
                message=(
                    "measurement-resolution quantization requires "
                    "measurement_resolution"
                ),
            )
        quantizer = FixedStepMidTreadQuantizer.from_string(
            active_policy.measurement_resolution
        )
        return ResolvedQuantization(
            policy=active_policy,
            quantizer=quantizer,
            quantization_mode=mode,
        )
    if mode == "scale_adaptive_mid_tread":
        scale_fraction = _positive_decimal(
            active_policy.scale_fraction,
            field_path="quantization_policy.scale_fraction",
        )
        fallback_step = _positive_decimal(
            active_policy.zero_scale_fallback_step,
            field_path="quantization_policy.zero_scale_fallback_step",
        )
        scale_reference = _scale_reference(
            observed_values=observed_values,
            scale_statistic=active_policy.scale_statistic,
        )
        step = (
            fallback_step
            if scale_reference == Decimal("0")
            else scale_reference * scale_fraction
        )
        quantizer = FixedStepMidTreadQuantizer(step=step)
        return ResolvedQuantization(
            policy=active_policy,
            quantizer=quantizer,
            quantization_mode=mode,
            scale_reference=float(scale_reference),
        )
    _raise_invalid_policy(
        field_path="quantization_policy.quantization_mode",
        message="unsupported quantization mode",
    )


def _positive_decimal(value: str | None, *, field_path: str) -> Decimal:
    if value is None:
        _raise_invalid_policy(
            field_path=field_path,
            message=f"{field_path} is required",
        )
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise ContractValidationError(
            code="invalid_quantization_policy",
            message=f"{field_path} must be a finite decimal string",
            field_path=field_path,
        ) from exc
    if not parsed.is_finite() or parsed <= 0:
        _raise_invalid_policy(
            field_path=field_path,
            message=f"{field_path} must be strictly positive",
        )
    return parsed


def _scale_reference(
    *,
    observed_values: Sequence[float],
    scale_statistic: str,
) -> Decimal:
    values = tuple(Decimal(str(float(value))) for value in observed_values)
    if not values:
        return Decimal("0")
    if scale_statistic == "max_abs":
        return max(abs(value) for value in values)
    if scale_statistic == "range":
        return max(values) - min(values)
    _raise_invalid_policy(
        field_path="quantization_policy.scale_statistic",
        message="unsupported scale statistic",
    )


def _raise_invalid_policy(*, field_path: str, message: str) -> None:
    raise ContractValidationError(
        code="invalid_quantization_policy",
        message=message,
        field_path=field_path,
    )
