from __future__ import annotations

from fractions import Fraction
from typing import Any

from euclid.contracts.errors import ContractValidationError


def _coerce_fraction(value: Fraction | int | float | str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ContractValidationError(
            code="type_error",
            message="boolean values are not legal rationals in the algorithmic DSL",
            field_path="value",
        )
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not value.is_integer():
            return Fraction(str(value))
        return Fraction(int(value), 1)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            raise ContractValidationError(
                code="parse_failed",
                message="rational literals must not be empty",
                field_path="value",
            )
        try:
            return Fraction(token)
        except ValueError as exc:
            raise ContractValidationError(
                code="parse_failed",
                message=f"{token!r} is not a legal exact-rational literal",
                field_path="value",
            ) from exc
    raise ContractValidationError(
        code="type_error",
        message="unsupported algorithmic scalar value",
        field_path="value",
        details={"type": type(value).__name__},
    )


def canonicalize_fraction(value: Fraction | int | float | str) -> str:
    fraction = _coerce_fraction(value)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def canonicalize_algorithmic_program(source: str, **kwargs: Any) -> str:
    from euclid.search.dsl.parser import parse_algorithmic_program

    return parse_algorithmic_program(source, **kwargs).canonical_source


__all__ = [
    "canonicalize_algorithmic_program",
    "canonicalize_fraction",
]
