from __future__ import annotations

from typing import Any

from euclid.runtime.hashing import sha256_digest


def make_content_addressed_id(schema_name: str, payload: Any) -> str:
    digest = sha256_digest(payload).removeprefix("sha256:")
    stem = schema_name.split("@", 1)[0]
    return f"{stem}_{digest[:20]}"
