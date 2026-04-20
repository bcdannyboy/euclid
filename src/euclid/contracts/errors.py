from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class ContractValidationError(ValueError):
    code: str
    message: str
    field_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.details = _normalize_detail_mapping(self.details)

    def __str__(self) -> str:
        location = f" [{self.field_path}]" if self.field_path else ""
        return f"{self.code}{location}: {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field_path": self.field_path,
            "details": _normalize_detail_mapping(self.details),
        }


def _normalize_detail_mapping(details: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalize_detail_value(details[key]) for key in sorted(details)}


def _normalize_detail_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_detail_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_detail_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_detail_value(item) for item in value]
    if isinstance(value, set):
        return [_normalize_detail_value(item) for item in sorted(value)]
    return value
