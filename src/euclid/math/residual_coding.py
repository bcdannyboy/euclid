from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from euclid.contracts.errors import ContractValidationError

_DEFAULT_ESCAPE_POLICY = "explicit_escape_on_first_seen_v1"
_ESCAPE_POLICY_ALIASES = {
    _DEFAULT_ESCAPE_POLICY,
    "explicit_escape_then_symbol_identity",
}
_DEFAULT_INNOVATION_CODE_FAMILY = "signed_integer_elias_delta_v1"


@dataclass(frozen=True, init=False)
class ResidualAlphabetPolicy:
    alphabet_mode: str
    symbols: tuple[int, ...] = ()
    escape_policy: str | None
    innovation_code_family: str

    def __init__(
        self,
        *,
        alphabet_mode: str | None = None,
        symbols: Sequence[int] = (),
        alphabet: Sequence[int] | None = None,
        escape_policy: str | None = _DEFAULT_ESCAPE_POLICY,
        innovation_code_family: str = _DEFAULT_INNOVATION_CODE_FAMILY,
    ) -> None:
        if alphabet is not None:
            symbols = alphabet
        normalized_symbols = tuple(int(symbol) for symbol in symbols)
        if alphabet_mode is None:
            alphabet_mode = "fixed_finite" if normalized_symbols else "open_prequential"
        if escape_policy == "none":
            escape_policy = None
        object.__setattr__(self, "alphabet_mode", alphabet_mode)
        object.__setattr__(self, "symbols", normalized_symbols)
        object.__setattr__(self, "escape_policy", escape_policy)
        object.__setattr__(self, "innovation_code_family", innovation_code_family)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.alphabet_mode not in {"open_prequential", "fixed_finite"}:
            raise ContractValidationError(
                code="unsupported_residual_alphabet_mode",
                message="residual alphabet mode is not supported",
                field_path="alphabet_mode",
                details={"alphabet_mode": self.alphabet_mode},
            )
        if len(set(self.symbols)) != len(self.symbols):
            raise ContractValidationError(
                code="duplicate_residual_alphabet_symbol",
                message="fixed residual alphabet symbols must be unique",
                field_path="symbols",
            )
        if self.alphabet_mode == "fixed_finite" and not self.symbols:
            raise ContractValidationError(
                code="empty_fixed_residual_alphabet",
                message="fixed finite residual alphabet must not be empty",
                field_path="symbols",
            )
        if self.escape_policy not in ({None} | _ESCAPE_POLICY_ALIASES):
            raise ContractValidationError(
                code="unsupported_residual_escape_policy",
                message="residual escape policy is not supported",
                field_path="escape_policy",
                details={"escape_policy": self.escape_policy},
            )
        if self.innovation_code_family != _DEFAULT_INNOVATION_CODE_FAMILY:
            raise ContractValidationError(
                code="unsupported_residual_innovation_code_family",
                message="residual innovation code family is not supported",
                field_path="innovation_code_family",
                details={"innovation_code_family": self.innovation_code_family},
            )

    @classmethod
    def open_prequential(cls) -> "ResidualAlphabetPolicy":
        return cls(alphabet_mode="open_prequential")

    @classmethod
    def fixed_finite(
        cls,
        symbols: Sequence[int],
        *,
        escape_policy: str | None = None,
    ) -> "ResidualAlphabetPolicy":
        return cls(
            alphabet_mode="fixed_finite",
            symbols=tuple(symbols),
            escape_policy=escape_policy,
        )

    @property
    def alphabet(self) -> tuple[int, ...]:
        return self.symbols


@dataclass(frozen=True)
class ResidualCodeEvent:
    row_index: int
    residual_index: int
    event_kind: str
    event_bits: float
    prefix_count: int
    symbol_count_before: int
    alphabet_size_before: int
    future_count_used: int
    alphabet_mode: str
    escape_policy: str | None
    innovation_code_family: str
    symbol_identity_bits: float

    def as_diagnostic_row(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "residual_index": self.residual_index,
            "event_type": self.event_type,
            "event_kind": self.event_kind,
            "symbol": self.symbol,
            "prefix_count": self.prefix_count,
            "symbol_count_before": self.symbol_count_before,
            "alphabet_size_before": self.alphabet_size_before,
            "probability": round(2 ** (-self.event_bits), 12),
            "incremental_bits": round(self.event_bits, 12),
            "future_count_used": self.future_count_used,
            "alphabet_mode": self.alphabet_mode,
            "escape_policy": self.escape_policy,
            "innovation_code_family": self.innovation_code_family,
            "symbol_identity_bits": round(self.symbol_identity_bits, 12),
        }

    @property
    def event_type(self) -> str:
        if self.event_kind == "SYMBOL_IDENTITY":
            return "symbol_identity"
        if self.event_kind == "SYMBOL":
            return "symbol"
        return self.event_kind

    @property
    def symbol(self) -> int | str:
        if self.event_kind == "ESC":
            return "ESC"
        return self.residual_index

    @property
    def incremental_bits(self) -> float:
        return self.event_bits


@dataclass(frozen=True)
class ResidualCodeResult:
    total_bits: float
    sequence_length_bits: float
    events: tuple[ResidualCodeEvent, ...]

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(event.as_diagnostic_row() for event in self.events)


def natural_integer_code_length(value: int) -> int:
    if value < 0:
        raise ContractValidationError(
            code="invalid_codelength_value",
            message="natural integer code expects a non-negative integer",
            field_path="value",
        )
    level_1 = math.floor(math.log2(value + 1))
    level_2 = math.floor(math.log2(level_1 + 1))
    return level_1 + (2 * level_2) + 1


def signed_integer_code_length(value: int) -> int:
    zigzag_index = (2 * value) if value >= 0 else (-2 * value) - 1
    return natural_integer_code_length(zigzag_index)


def prequential_escape_residual_bin_code_v1(
    residual_indices: Sequence[int],
    *,
    alphabet_policy: ResidualAlphabetPolicy | None = None,
) -> ResidualCodeResult:
    policy = alphabet_policy or ResidualAlphabetPolicy.open_prequential()
    counts: dict[int, int] = {}
    events: list[ResidualCodeEvent] = []

    for prefix_count, raw_residual_index in enumerate(residual_indices):
        residual_index = int(raw_residual_index)
        _validate_residual_against_policy(residual_index, policy=policy)
        symbol_count_before = counts.get(residual_index, 0)
        alphabet_size_before = len(counts)
        if symbol_count_before == 0:
            if _is_declared_fixed_symbol(residual_index, policy=policy):
                symbol_identity_bits = float(
                    _symbol_identity_bits(residual_index, policy=policy)
                )
                events.append(
                    _event(
                        row_index=len(events),
                        residual_index=residual_index,
                        event_kind="SYMBOL",
                        event_bits=symbol_identity_bits,
                        prefix_count=prefix_count,
                        symbol_count_before=symbol_count_before,
                        alphabet_size_before=alphabet_size_before,
                        policy=policy,
                        symbol_identity_bits=symbol_identity_bits,
                    )
                )
            elif policy.escape_policy is None:
                raise ContractValidationError(
                    code="residual_escape_policy_required",
                    message="new residual symbol requires an escape policy",
                    field_path="alphabet_policy.escape_policy",
                    details={"residual_index": residual_index},
                )
            else:
                events.append(
                    _event(
                        row_index=len(events),
                        residual_index=residual_index,
                        event_kind="ESC",
                        event_bits=_escape_bits(prefix_count, alphabet_size_before),
                        prefix_count=prefix_count,
                        symbol_count_before=symbol_count_before,
                        alphabet_size_before=alphabet_size_before,
                        policy=policy,
                        symbol_identity_bits=0.0,
                    )
                )
                symbol_identity_bits = float(
                    _symbol_identity_bits(residual_index, policy=policy)
                )
                events.append(
                    _event(
                        row_index=len(events),
                        residual_index=residual_index,
                        event_kind="SYMBOL_IDENTITY",
                        event_bits=symbol_identity_bits,
                        prefix_count=prefix_count,
                        symbol_count_before=symbol_count_before,
                        alphabet_size_before=alphabet_size_before,
                        policy=policy,
                        symbol_identity_bits=symbol_identity_bits,
                    )
                )
        else:
            events.append(
                _event(
                    row_index=len(events),
                    residual_index=residual_index,
                    event_kind="SYMBOL",
                    event_bits=_seen_symbol_bits(
                        prefix_count,
                        alphabet_size_before,
                        symbol_count_before,
                    ),
                    prefix_count=prefix_count,
                    symbol_count_before=symbol_count_before,
                    alphabet_size_before=alphabet_size_before,
                    policy=policy,
                    symbol_identity_bits=0.0,
                )
            )
        counts[residual_index] = symbol_count_before + 1

    sequence_length_bits = float(natural_integer_code_length(len(residual_indices)))
    event_bits = sum(event.event_bits for event in events)
    return ResidualCodeResult(
        total_bits=round(sequence_length_bits + event_bits, 12),
        sequence_length_bits=sequence_length_bits,
        events=tuple(events),
    )


def _validate_residual_against_policy(
    residual_index: int,
    *,
    policy: ResidualAlphabetPolicy,
) -> None:
    if (
        policy.alphabet_mode == "fixed_finite"
        and residual_index not in policy.symbols
        and policy.escape_policy is None
    ):
        raise ContractValidationError(
            code="residual_symbol_outside_fixed_alphabet",
            message="residual is outside the fixed finite alphabet",
            field_path="residual_indices",
            details={
                "residual_index": residual_index,
                "alphabet": list(policy.symbols),
            },
        )


def _event(
    *,
    row_index: int,
    residual_index: int,
    event_kind: str,
    event_bits: float,
    prefix_count: int,
    symbol_count_before: int,
    alphabet_size_before: int,
    policy: ResidualAlphabetPolicy,
    symbol_identity_bits: float,
) -> ResidualCodeEvent:
    return ResidualCodeEvent(
        row_index=prefix_count,
        residual_index=residual_index,
        event_kind=event_kind,
        event_bits=round(event_bits, 12),
        prefix_count=prefix_count,
        symbol_count_before=symbol_count_before,
        alphabet_size_before=alphabet_size_before,
        future_count_used=0,
        alphabet_mode=policy.alphabet_mode,
        escape_policy=policy.escape_policy,
        innovation_code_family=policy.innovation_code_family,
        symbol_identity_bits=round(symbol_identity_bits, 12),
    )


def _escape_bits(prefix_count: int, alphabet_size_before: int) -> float:
    denominator = prefix_count + alphabet_size_before + 1
    return -math.log2(1.0 / denominator)


def _seen_symbol_bits(
    prefix_count: int,
    alphabet_size_before: int,
    symbol_count_before: int,
) -> float:
    denominator = prefix_count + alphabet_size_before + 1
    probability = symbol_count_before / denominator
    return -math.log2(probability)


def _is_declared_fixed_symbol(
    residual_index: int,
    *,
    policy: ResidualAlphabetPolicy,
) -> bool:
    return policy.alphabet_mode == "fixed_finite" and residual_index in policy.symbols


def _symbol_identity_bits(
    residual_index: int,
    *,
    policy: ResidualAlphabetPolicy,
) -> int:
    if policy.alphabet_mode == "fixed_finite" and residual_index in policy.symbols:
        return math.ceil(math.log2(len(policy.symbols)))
    return signed_integer_code_length(residual_index)


__all__ = [
    "ResidualAlphabetPolicy",
    "ResidualCodeEvent",
    "ResidualCodeResult",
    "prequential_escape_residual_bin_code_v1",
]
