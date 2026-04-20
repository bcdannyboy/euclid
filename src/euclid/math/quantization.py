from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Iterable

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
