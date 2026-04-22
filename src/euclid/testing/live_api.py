from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError

from euclid.runtime.env import EuclidEnv
from euclid.testing.redaction import collect_secret_values, redact_mapping, redact_text

LiveApiCheck = Callable[[Mapping[str, str]], Mapping[str, Any]]

NON_CLAIM_EVIDENCE_BOUNDARY = {
    "counts_as_scientific_claim_evidence": False,
    "reason_code": "live_api_gate_validates_access_and_payload_shape_only",
    "status": "non_claim_evidence",
}


@dataclass(frozen=True)
class LiveApiGateResult:
    gate_id: str
    status: str
    reason_codes: tuple[str, ...]
    evidence_path: Path


@dataclass(frozen=True)
class LiveApiGate:
    gate_id: str
    provider: str
    endpoint_class: str
    required_env: tuple[str, ...]
    evidence_path: Path
    check: LiveApiCheck

    def run(self, env: EuclidEnv) -> LiveApiGateResult:
        started = time.perf_counter()
        secrets = collect_secret_values(env.values, names=self.required_env)
        credential_presence = env.presence_metadata(self.required_env)
        semantic_checks: Mapping[str, Any] = {}
        message = ""

        if not env.live_tests_enabled:
            result = LiveApiGateResult(
                gate_id=self.gate_id,
                status="skipped",
                reason_codes=("live_api_tests_disabled",),
                evidence_path=self.evidence_path,
            )
            self._write_evidence(
                result=result,
                credential_presence=credential_presence,
                semantic_checks=semantic_checks,
                message=message,
                elapsed_ms=_elapsed_ms(started),
                secrets=secrets,
            )
            return result

        missing = tuple(name for name in self.required_env if not env.get(name))
        if missing:
            status = "failed" if env.strict_live_api else "skipped"
            result = LiveApiGateResult(
                gate_id=self.gate_id,
                status=status,
                reason_codes=("missing_live_api_credentials",),
                evidence_path=self.evidence_path,
            )
            self._write_evidence(
                result=result,
                credential_presence=credential_presence,
                semantic_checks=semantic_checks,
                message=f"Missing required live API credential variables: {', '.join(missing)}",
                elapsed_ms=_elapsed_ms(started),
                secrets=secrets,
            )
            return result

        try:
            semantic_checks = self.check(env.require(self.required_env))
            result = LiveApiGateResult(
                gate_id=self.gate_id,
                status="passed",
                reason_codes=(),
                evidence_path=self.evidence_path,
            )
        except Exception as exc:  # pragma: no cover - exact branches unit tested.
            result = LiveApiGateResult(
                gate_id=self.gate_id,
                status="failed",
                reason_codes=(_reason_code_for_exception(exc),),
                evidence_path=self.evidence_path,
            )
            message = redact_text(exc, secrets=secrets)

        self._write_evidence(
            result=result,
            credential_presence=credential_presence,
            semantic_checks=semantic_checks,
            message=message,
            elapsed_ms=_elapsed_ms(started),
            secrets=secrets,
        )
        return result

    def _write_evidence(
        self,
        *,
        result: LiveApiGateResult,
        credential_presence: Mapping[str, Mapping[str, Any]],
        semantic_checks: Mapping[str, Any],
        message: str,
        elapsed_ms: int,
        secrets: Sequence[str],
    ) -> None:
        payload = {
            "evidence_kind": "live_api_gate",
            "gate_id": self.gate_id,
            "provider": self.provider,
            "endpoint_class": self.endpoint_class,
            "status": result.status,
            "reason_codes": list(result.reason_codes),
            "claim_evidence_status": "not_claim_evidence",
            "claim_boundary": dict(NON_CLAIM_EVIDENCE_BOUNDARY),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "latency_ms": elapsed_ms,
            "credential_presence": credential_presence,
            "semantic_checks": redact_mapping(semantic_checks, secrets=secrets),
        }
        if message:
            payload["message"] = redact_text(message, secrets=secrets)
        sanitized = redact_mapping(payload, secrets=secrets)
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_path.write_text(
            json.dumps(sanitized, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def _reason_code_for_exception(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        if exc.code == 429:
            return "provider_http_429"
        if exc.code in {401, 403}:
            return f"provider_http_{exc.code}"
        if 500 <= exc.code < 600:
            return "provider_http_5xx"
        return f"provider_http_{exc.code}"
    if isinstance(exc, TimeoutError | socket.timeout):
        return "provider_timeout"
    if isinstance(exc, URLError) and "timed out" in str(exc.reason).lower():
        return "provider_timeout"
    if isinstance(exc, ValueError):
        return "provider_payload_invalid"
    return "provider_error"


def validate_ordered_provider_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    timestamp_key: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty payload from provider")
    timestamps: list[str] = []
    for index, row in enumerate(rows):
        timestamp = str(row.get(timestamp_key, "")).strip()
        if not timestamp:
            raise ValueError(f"missing timestamp at row {index}")
        timestamps.append(timestamp)
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("duplicate timestamp in provider payload")
    if timestamps != sorted(timestamps):
        raise ValueError("out-of-order timestamp in provider payload")
    return {
        "row_count": len(rows),
        "timestamps_ordered": True,
        "timestamps_unique": True,
    }


__all__ = [
    "LiveApiGate",
    "LiveApiGateResult",
    "NON_CLAIM_EVIDENCE_BOUNDARY",
    "validate_ordered_provider_rows",
]
