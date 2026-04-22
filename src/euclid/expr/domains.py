from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DOMAIN_TYPES: tuple[str, ...] = (
    "real",
    "positive_real",
    "nonnegative_real",
    "integer",
    "probability",
    "bounded_interval",
    "boolean",
)


@dataclass(frozen=True)
class DomainConstraint:
    symbol: str
    domain: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "domain": self.domain,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "lower_inclusive": self.lower_inclusive,
            "upper_inclusive": self.upper_inclusive,
        }


__all__ = ["DOMAIN_TYPES", "DomainConstraint"]

