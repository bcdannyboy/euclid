from __future__ import annotations

import math
from datetime import datetime, timezone

from euclid.contracts.errors import ContractValidationError


def canonical_timestamp(value: str | datetime) -> str:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = f"{candidate[:-1]}+00:00"
        try:
            timestamp = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ContractValidationError(
                code="invalid_timestamp",
                message="timestamps must be ISO-8601 values",
                field_path="timestamp",
            ) from exc
    else:
        raise ContractValidationError(
            code="invalid_timestamp",
            message="timestamps must be strings or datetimes",
            field_path="timestamp",
        )
    if timestamp.tzinfo is None:
        raise ContractValidationError(
            code="invalid_timestamp",
            message="timestamps must include a timezone offset",
            field_path="timestamp",
        )
    utc_timestamp = timestamp.astimezone(timezone.utc)
    return utc_timestamp.isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_finite_float(value: str | float, *, field_path: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ContractValidationError(
            code="nonfinite_numeric_value",
            message="observed values must be finite",
            field_path=field_path,
        )
    return numeric
