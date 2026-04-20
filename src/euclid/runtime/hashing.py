from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from euclid.contracts.errors import ContractValidationError


def normalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(
                code="unstable_serialization",
                message="non-finite floats are forbidden in canonical serialization",
            )
        return value
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(
                    code="unstable_serialization",
                    message="all canonical JSON object keys must be strings",
                )
            normalized[key] = normalize_json_value(nested_value)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    raise ContractValidationError(
        code="unstable_serialization",
        message=f"{type(value).__name__} is not supported by canonical serialization",
    )


def canonicalize_json(value: Any) -> str:
    normalized = normalize_json_value(value)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_digest(value: Any) -> str:
    canonical = canonicalize_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"
