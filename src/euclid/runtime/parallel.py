from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from euclid.contracts.errors import ContractValidationError
from euclid.runtime.hashing import sha256_digest


@dataclass(frozen=True)
class ParallelExecutionPolicy:
    max_workers: int = 1
    timeout_seconds: float = 60.0
    backend: str = "joblib"
    aggregation_rule: str = "sort_by_item_id"
    ray_enabled: bool = False

    def __post_init__(self) -> None:
        if self.max_workers <= 0:
            raise ContractValidationError(
                code="invalid_parallel_policy",
                message="max_workers must be positive",
                field_path="max_workers",
            )
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise ContractValidationError(
                code="invalid_parallel_policy",
                message="timeout_seconds must be positive and finite",
                field_path="timeout_seconds",
            )
        if self.aggregation_rule != "sort_by_item_id":
            raise ContractValidationError(
                code="invalid_parallel_policy",
                message="only sort_by_item_id aggregation is replay-safe",
                field_path="aggregation_rule",
            )

    def as_dict(self, *, adapter: str | None = None) -> dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "timeout_seconds": self.timeout_seconds,
            "backend": self.backend,
            "adapter": adapter or self.backend,
            "aggregation_rule": self.aggregation_rule,
            "ray_enabled": self.ray_enabled,
        }


@dataclass(frozen=True)
class ParallelWorkItem:
    item_id: str
    payload: Any

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id.strip():
            raise ContractValidationError(
                code="invalid_parallel_item",
                message="item_id must be a non-empty string",
                field_path="item_id",
            )
        object.__setattr__(self, "item_id", self.item_id.strip())

    def replay_payload(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "payload": self.payload}


@dataclass(frozen=True)
class ParallelWorkerOutput:
    item_id: str
    value: Any

    def as_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "value": self.value}


@dataclass(frozen=True)
class ParallelDiagnostic:
    item_id: str
    reason_code: str
    message: str
    recoverable: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "reason_code": self.reason_code,
            "message": self.message,
            "recoverable": self.recoverable,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ParallelRunResult:
    status: str
    outputs: tuple[ParallelWorkerOutput, ...]
    diagnostics: tuple[ParallelDiagnostic, ...]
    policy: Mapping[str, Any]
    elapsed_seconds: float
    replay_identity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "outputs": [output.as_dict() for output in self.outputs],
            "diagnostics": [diagnostic.as_dict() for diagnostic in self.diagnostics],
            "policy": dict(self.policy),
            "elapsed_seconds": self.elapsed_seconds,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class _WorkerEnvelope:
    item_id: str
    value: Any = None
    diagnostic: ParallelDiagnostic | None = None


def run_replay_safe_parallel(
    items: Sequence[ParallelWorkItem],
    *,
    worker: Callable[[ParallelWorkItem], Any],
    policy: ParallelExecutionPolicy | None = None,
) -> ParallelRunResult:
    resolved_policy = policy or ParallelExecutionPolicy()
    ordered_items = tuple(sorted(items, key=lambda item: item.item_id))
    started = time.perf_counter()
    adapter = _adapter_label(resolved_policy)
    envelopes = (
        _run_joblib(ordered_items, worker, resolved_policy)
        if adapter == "joblib-threading"
        else _run_threadpool(ordered_items, worker, resolved_policy)
    )
    elapsed = time.perf_counter() - started
    outputs = tuple(
        sorted(
            (
                ParallelWorkerOutput(item_id=envelope.item_id, value=envelope.value)
                for envelope in envelopes
                if envelope.diagnostic is None
            ),
            key=lambda output: output.item_id,
        )
    )
    diagnostics = tuple(
        sorted(
            (
                envelope.diagnostic
                for envelope in envelopes
                if envelope.diagnostic is not None
            ),
            key=lambda diagnostic: (diagnostic.item_id, diagnostic.reason_code),
        )
    )
    if any(diagnostic.reason_code == "worker_timeout" for diagnostic in diagnostics):
        status = "timeout"
    elif diagnostics:
        status = "degraded"
    else:
        status = "completed"
    replay_payload = {
        "status": status,
        "outputs": [output.as_dict() for output in outputs],
        "diagnostics": [diagnostic.as_dict() for diagnostic in diagnostics],
        "aggregation_rule": resolved_policy.aggregation_rule,
        "input_item_ids": [item.item_id for item in ordered_items],
    }
    return ParallelRunResult(
        status=status,
        outputs=outputs,
        diagnostics=diagnostics,
        policy=resolved_policy.as_dict(adapter=adapter),
        elapsed_seconds=elapsed,
        replay_identity=sha256_digest(replay_payload),
    )


def _run_joblib(
    items: tuple[ParallelWorkItem, ...],
    worker: Callable[[ParallelWorkItem], Any],
    policy: ParallelExecutionPolicy,
) -> tuple[_WorkerEnvelope, ...]:
    try:
        from joblib import Parallel, delayed
    except Exception:
        return _run_threadpool(items, worker, policy)

    return tuple(
        Parallel(n_jobs=policy.max_workers, prefer="threads")(
            delayed(_execute_item)(item, worker) for item in items
        )
    )


def _run_threadpool(
    items: tuple[ParallelWorkItem, ...],
    worker: Callable[[ParallelWorkItem], Any],
    policy: ParallelExecutionPolicy,
) -> tuple[_WorkerEnvelope, ...]:
    if not items:
        return ()
    envelopes: list[_WorkerEnvelope] = []
    with ThreadPoolExecutor(max_workers=min(policy.max_workers, len(items))) as executor:
        futures: dict[str, Future[_WorkerEnvelope]] = {
            item.item_id: executor.submit(_execute_item, item, worker) for item in items
        }
        for item in items:
            future = futures[item.item_id]
            try:
                envelopes.append(future.result(timeout=policy.timeout_seconds))
            except TimeoutError:
                future.cancel()
                envelopes.append(
                    _WorkerEnvelope(
                        item_id=item.item_id,
                        diagnostic=ParallelDiagnostic(
                            item_id=item.item_id,
                            reason_code="worker_timeout",
                            message="parallel worker exceeded timeout",
                            recoverable=True,
                            details={"timeout_seconds": policy.timeout_seconds},
                        ),
                    )
                )
    return tuple(envelopes)


def _execute_item(
    item: ParallelWorkItem,
    worker: Callable[[ParallelWorkItem], Any],
) -> _WorkerEnvelope:
    try:
        return _WorkerEnvelope(item_id=item.item_id, value=worker(item))
    except Exception as exc:
        return _WorkerEnvelope(
            item_id=item.item_id,
            diagnostic=ParallelDiagnostic(
                item_id=item.item_id,
                reason_code="worker_exception",
                message=type(exc).__name__,
                recoverable=True,
                details={"exception_type": type(exc).__name__},
            ),
        )


def _adapter_label(policy: ParallelExecutionPolicy) -> str:
    if policy.ray_enabled:
        return "ray-requested-local-fallback"
    if policy.backend == "joblib" and policy.max_workers > 1:
        try:
            import joblib  # noqa: F401

            return "joblib-threading"
        except Exception:
            return "threadpool"
    return "threadpool"


__all__ = [
    "ParallelDiagnostic",
    "ParallelExecutionPolicy",
    "ParallelRunResult",
    "ParallelWorkItem",
    "ParallelWorkerOutput",
    "run_replay_safe_parallel",
]
