from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


def stable_float(value: float) -> float:
    return float(round(float(value), 12))


def finite_tuple(values: Sequence[float]) -> tuple[float, ...] | None:
    result: list[float] = []
    for value in values:
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        result.append(numeric)
    return tuple(result)


def unique_codes(codes: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for code in codes:
        text = str(code)
        if text:
            seen.setdefault(text, None)
    return tuple(seen)


def replay_identity(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        normalize_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def normalize_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): normalize_payload(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple | list):
        return [normalize_payload(item) for item in value]
    if isinstance(value, float):
        if math.isfinite(value):
            return stable_float(value)
        return str(value)
    return value
