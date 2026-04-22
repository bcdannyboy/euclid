from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.runtime.hashing import normalize_json_value, sha256_digest

_CACHE_KEY_VERSION = 1
_ALLOWED_CATEGORIES = frozenset(
    {
        "feature_matrix",
        "expression_evaluation",
        "subtree_evaluation",
        "fitted_constants",
        "simplification",
    }
)


@dataclass(frozen=True)
class CacheStats:
    hit_count: int
    miss_count: int
    entry_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "entry_count": self.entry_count,
        }


@dataclass(frozen=True)
class CacheRecord:
    category: str
    cache_key: str
    payload_digest: str
    value: Any

    def replay_entry(self) -> dict[str, str]:
        return {
            "category": self.category,
            "cache_key": self.cache_key,
            "payload_digest": self.payload_digest,
        }


def cache_key_for(category: str, payload: Any) -> str:
    resolved_category = _validate_category(category)
    digest = sha256_digest(
        {
            "cache_key_version": _CACHE_KEY_VERSION,
            "category": resolved_category,
            "payload": normalize_json_value(payload),
        }
    )
    return f"cache:{resolved_category}:{digest}"


class EvaluationCache:
    def __init__(self) -> None:
        self._records: dict[str, CacheRecord] = {}
        self._hit_count = 0
        self._miss_count = 0

    def get_or_compute(
        self,
        *,
        category: str,
        payload: Any,
        compute: Callable[[], Any],
    ) -> Any:
        key = cache_key_for(category, payload)
        record = self._records.get(key)
        if record is not None:
            self._hit_count += 1
            return record.value
        value = compute()
        self._miss_count += 1
        self._records[key] = CacheRecord(
            category=_validate_category(category),
            cache_key=key,
            payload_digest=sha256_digest(normalize_json_value(payload)),
            value=value,
        )
        return value

    def stats(self) -> CacheStats:
        return CacheStats(
            hit_count=self._hit_count,
            miss_count=self._miss_count,
            entry_count=len(self._records),
        )

    def replay_diagnostics(self) -> dict[str, Any]:
        return {
            "cache_status": "captured",
            "cache_key_version": _CACHE_KEY_VERSION,
            "stats": self.stats().as_dict(),
            "entries": [
                record.replay_entry()
                for record in sorted(
                    self._records.values(),
                    key=lambda item: (item.category, item.cache_key),
                )
            ],
        }


def cache_replay_diagnostics(
    records: Mapping[str, CacheRecord],
    *,
    hit_count: int = 0,
    miss_count: int = 0,
) -> dict[str, Any]:
    cache = EvaluationCache()
    cache._records.update(records)
    cache._hit_count = hit_count
    cache._miss_count = miss_count
    return cache.replay_diagnostics()


def _validate_category(category: str) -> str:
    if not isinstance(category, str) or not category.strip():
        raise ContractValidationError(
            code="invalid_cache_category",
            message="cache category must be a non-empty string",
            field_path="category",
        )
    resolved = category.strip()
    if resolved not in _ALLOWED_CATEGORIES:
        raise ContractValidationError(
            code="invalid_cache_category",
            message=f"unsupported evaluation cache category {resolved!r}",
            field_path="category",
            details={"allowed_categories": sorted(_ALLOWED_CATEGORIES)},
        )
    return resolved


__all__ = [
    "CacheRecord",
    "CacheStats",
    "EvaluationCache",
    "cache_key_for",
    "cache_replay_diagnostics",
]
