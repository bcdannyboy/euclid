from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTION_TOKEN = "[REDACTED]"
_SECRET_HEADER_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "apikey",
        "openai-api-key",
    }
)
_SECRET_QUERY_NAMES = frozenset(
    {
        "apikey",
        "api_key",
        "key",
        "token",
        "access_token",
        "authorization",
    }
)


def collect_secret_values(
    env_values: Mapping[str, str],
    *,
    names: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        value
        for name in names
        if (value := str(env_values.get(name, "")).strip())
    )


def redact_text(value: object, *, secrets: Sequence[str]) -> str:
    text = str(value)
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            text = text.replace(secret, REDACTION_TOKEN)
    return _redact_secret_bearing_url(text)


def redact_mapping(value: Any, *, secrets: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in _SECRET_HEADER_NAMES:
                redacted[key_text] = REDACTION_TOKEN if child else child
            else:
                redacted[key_text] = redact_mapping(child, secrets=secrets)
        return redacted
    if isinstance(value, list):
        return [redact_mapping(item, secrets=secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_mapping(item, secrets=secrets) for item in value)
    if isinstance(value, str):
        return redact_text(value, secrets=secrets)
    return value


def _redact_secret_bearing_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme or not parsed.netloc or not parsed.query:
        return value
    query_pairs = []
    changed = False
    for key, child in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in _SECRET_QUERY_NAMES:
            query_pairs.append((key, REDACTION_TOKEN))
            changed = True
        else:
            query_pairs.append((key, child))
    if not changed:
        return value
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query_pairs),
            parsed.fragment,
        )
    )


__all__ = [
    "REDACTION_TOKEN",
    "collect_secret_values",
    "redact_mapping",
    "redact_text",
]
