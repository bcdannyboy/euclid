from __future__ import annotations

import json
import threading
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:  # pragma: no cover - platform dependent best-effort memory sampling
    import resource
except ImportError:  # pragma: no cover
    resource = None

_ARTIFACT_VERSION = "1.0.0"
_SUBMITTER_ORDER = {
    "analytic_backend": 0,
    "recursive_spectral_backend": 1,
    "algorithmic_search_backend": 2,
    "portfolio_orchestrator": 3,
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


def _json_sort_key(value: Mapping[str, Any]) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def _submitter_sort_key(submitter_id: str) -> tuple[int, str]:
    return (_SUBMITTER_ORDER.get(submitter_id, len(_SUBMITTER_ORDER)), submitter_id)


def _rss_bytes() -> int | None:
    if resource is None:  # pragma: no cover
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if rss <= 0:
        return 0
    return rss if rss > 1024 * 1024 else rss * 1024


@dataclass(frozen=True)
class TelemetrySpanRecord:
    name: str
    category: str
    status: str
    wall_time_seconds: float
    cpu_time_seconds: float
    start_memory_bytes: int
    end_memory_bytes: int
    peak_memory_bytes: int
    rss_max_bytes: int | None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    error_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "wall_time_seconds": self.wall_time_seconds,
            "cpu_time_seconds": self.cpu_time_seconds,
            "start_memory_bytes": self.start_memory_bytes,
            "end_memory_bytes": self.end_memory_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "attributes": _json_ready(self.attributes),
        }
        if self.rss_max_bytes is not None:
            payload["rss_max_bytes"] = self.rss_max_bytes
        if self.error_type is not None:
            payload["error_type"] = self.error_type
        return payload


@dataclass(frozen=True)
class TelemetryMeasurement:
    name: str
    category: str
    value: float | int
    unit: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "value": self.value,
            "unit": self.unit,
            "attributes": _json_ready(self.attributes),
        }


@dataclass(frozen=True)
class SeedTelemetryRecord:
    scope: str
    value: str

    def as_dict(self) -> dict[str, str]:
        return {"scope": self.scope, "value": self.value}


@dataclass(frozen=True)
class RestartTelemetryRecord:
    submitter_id: str
    declared_restarts: int
    used_restarts: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "submitter_id": self.submitter_id,
            "declared_restarts": self.declared_restarts,
            "used_restarts": self.used_restarts,
        }


@dataclass(frozen=True)
class BudgetTelemetryRecord:
    submitter_id: str
    declared_candidate_limit: int
    declared_wall_clock_seconds: int
    attempted_candidate_count: int
    accepted_candidate_count: int
    rejected_candidate_count: int
    omitted_candidate_count: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "submitter_id": self.submitter_id,
            "declared_candidate_limit": self.declared_candidate_limit,
            "declared_wall_clock_seconds": self.declared_wall_clock_seconds,
            "attempted_candidate_count": self.attempted_candidate_count,
            "accepted_candidate_count": self.accepted_candidate_count,
            "rejected_candidate_count": self.rejected_candidate_count,
            "omitted_candidate_count": self.omitted_candidate_count,
        }


@dataclass(frozen=True)
class ArtifactStoreTelemetry:
    read_operation_count: int
    write_operation_count: int
    read_bytes: int
    write_bytes: int
    cache_hit_count: int
    cache_miss_count: int
    read_wall_time_seconds: float
    write_wall_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "read_operation_count": self.read_operation_count,
            "write_operation_count": self.write_operation_count,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "read_wall_time_seconds": self.read_wall_time_seconds,
            "write_wall_time_seconds": self.write_wall_time_seconds,
            "read_throughput_bytes_per_second": (
                0.0
                if self.read_wall_time_seconds <= 0.0
                else self.read_bytes / self.read_wall_time_seconds
            ),
            "write_throughput_bytes_per_second": (
                0.0
                if self.write_wall_time_seconds <= 0.0
                else self.write_bytes / self.write_wall_time_seconds
            ),
        }


@dataclass(frozen=True)
class PerformanceTelemetryArtifact:
    profile_kind: str
    subject_id: str
    status: str
    wall_time_seconds: float
    cpu_time_seconds: float
    current_memory_bytes: int
    peak_memory_bytes: int
    rss_max_bytes: int | None
    spans: tuple[TelemetrySpanRecord, ...]
    measurements: tuple[TelemetryMeasurement, ...]
    seed_records: tuple[SeedTelemetryRecord, ...]
    restart_records: tuple[RestartTelemetryRecord, ...]
    budget_records: tuple[BudgetTelemetryRecord, ...]
    artifact_store: ArtifactStoreTelemetry
    artifact_type: str = "performance_telemetry"
    artifact_version: str = _ARTIFACT_VERSION
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_type": self.artifact_type,
            "artifact_version": self.artifact_version,
            "profile_kind": self.profile_kind,
            "subject_id": self.subject_id,
            "status": self.status,
            "wall_time_seconds": self.wall_time_seconds,
            "cpu_time_seconds": self.cpu_time_seconds,
            "current_memory_bytes": self.current_memory_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "spans": [span.as_dict() for span in self.spans],
            "measurements": [
                measurement.as_dict() for measurement in self.measurements
            ],
            "seed_records": [record.as_dict() for record in self.seed_records],
            "restart_records": [record.as_dict() for record in self.restart_records],
            "budget_records": [record.as_dict() for record in self.budget_records],
            "artifact_store": self.artifact_store.as_dict(),
            "attributes": _json_ready(self.attributes),
        }
        if self.rss_max_bytes is not None:
            payload["rss_max_bytes"] = self.rss_max_bytes
        return payload


@dataclass(frozen=True)
class PerformanceBudget:
    budget_id: str
    max_wall_time_seconds: float
    max_cpu_time_seconds: float | None = None
    max_peak_memory_bytes: int | None = None


@dataclass(frozen=True)
class PerformanceBudgetEvaluation:
    budget_id: str
    subject_id: str
    passed: bool
    failure_reasons: tuple[str, ...]
    observed_wall_time_seconds: float
    observed_cpu_time_seconds: float
    observed_peak_memory_bytes: int


@dataclass(frozen=True)
class PerformanceSuiteEntry:
    telemetry_path: Path
    profile_kind: str
    subject_id: str
    status: str
    wall_time_seconds: float
    cpu_time_seconds: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class PerformanceSuite:
    suite_id: str
    profiles: tuple[PerformanceSuiteEntry, ...]
    total_wall_time_seconds: float
    max_wall_time_seconds: float
    max_peak_memory_bytes: int

    @property
    def profile_count(self) -> int:
        return len(self.profiles)


@dataclass(frozen=True)
class SuitePerformanceBudget:
    budget_id: str
    max_total_wall_time_seconds: float
    max_profile_wall_time_seconds: float
    max_peak_memory_bytes: int | None = None


@dataclass(frozen=True)
class SuitePerformanceBudgetEvaluation:
    budget_id: str
    suite_id: str
    passed: bool
    failure_reasons: tuple[str, ...]
    profile_count: int
    total_wall_time_seconds: float
    max_wall_time_seconds: float
    max_peak_memory_bytes: int


@dataclass(frozen=True)
class CandidateThroughputBudget:
    budget_id: str
    min_candidates_per_second: float


@dataclass(frozen=True)
class CandidateThroughputBudgetEvaluation:
    budget_id: str
    passed: bool
    failure_reasons: tuple[str, ...]
    candidate_count: int
    elapsed_seconds: float
    observed_candidates_per_second: float
    min_candidates_per_second: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget_id": self.budget_id,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "candidate_count": self.candidate_count,
            "elapsed_seconds": self.elapsed_seconds,
            "observed_candidates_per_second": self.observed_candidates_per_second,
            "thresholds": {
                "min_candidates_per_second": self.min_candidates_per_second,
            },
        }


@dataclass(frozen=True)
class EngineRuntimeBudget:
    budget_id: str
    timeout_seconds: float
    candidate_limit: int
    max_iterations: int | None = None
    node_limit: int | None = None
    max_memory_bytes: int | None = None

    def resource_limits(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timeout_seconds": self.timeout_seconds,
            "candidate_limit": self.candidate_limit,
        }
        if self.max_iterations is not None:
            payload["max_iterations"] = self.max_iterations
        if self.node_limit is not None:
            payload["node_limit"] = self.node_limit
        if self.max_memory_bytes is not None:
            payload["max_memory_bytes"] = self.max_memory_bytes
        return payload


@dataclass(frozen=True)
class EngineRuntimeBudgetReport:
    engine_id: str
    budget_id: str
    status: str
    passed: bool
    failure_reasons: tuple[str, ...]
    elapsed_seconds: float
    candidate_count: int
    reason_codes: tuple[str, ...]
    resource_limits: Mapping[str, Any]
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "budget_id": self.budget_id,
            "status": self.status,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "elapsed_seconds": self.elapsed_seconds,
            "candidate_count": self.candidate_count,
            "reason_codes": list(self.reason_codes),
            "resource_limits": _json_ready(self.resource_limits),
            "details": _json_ready(self.details),
        }


@dataclass(frozen=True)
class RuntimeDegradationDecision:
    status: str
    reason_codes: tuple[str, ...]
    claim_publication_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "claim_publication_allowed": self.claim_publication_allowed,
        }


class TelemetryRecorder:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        self._started_tracing = False
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_tracing = True
        self._spans: list[TelemetrySpanRecord] = []
        self._measurements: list[TelemetryMeasurement] = []
        self._seed_records: list[SeedTelemetryRecord] = []
        self._restart_records: list[RestartTelemetryRecord] = []
        self._budget_records: list[BudgetTelemetryRecord] = []
        self._artifact_read_operation_count = 0
        self._artifact_write_operation_count = 0
        self._artifact_read_bytes = 0
        self._artifact_write_bytes = 0
        self._artifact_cache_hit_count = 0
        self._artifact_cache_miss_count = 0
        self._artifact_read_wall_time_seconds = 0.0
        self._artifact_write_wall_time_seconds = 0.0

    @contextmanager
    def span(
        self,
        name: str,
        *,
        category: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[None]:
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        start_current, start_peak = tracemalloc.get_traced_memory()
        status = "completed"
        error_type: str | None = None
        try:
            yield
        except Exception as exc:
            status = "failed"
            error_type = exc.__class__.__name__
            raise
        finally:
            end_current, end_peak = tracemalloc.get_traced_memory()
            with self._lock:
                self._spans.append(
                    TelemetrySpanRecord(
                        name=name,
                        category=category,
                        status=status,
                        wall_time_seconds=time.perf_counter() - start_wall,
                        cpu_time_seconds=time.process_time() - start_cpu,
                        start_memory_bytes=start_current,
                        end_memory_bytes=end_current,
                        peak_memory_bytes=max(start_peak, end_peak),
                        rss_max_bytes=_rss_bytes(),
                        attributes=dict(attributes or {}),
                        error_type=error_type,
                    )
                )

    def record_measurement(
        self,
        *,
        name: str,
        category: str,
        value: float | int,
        unit: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._measurements.append(
                TelemetryMeasurement(
                    name=name,
                    category=category,
                    value=value,
                    unit=unit,
                    attributes=dict(attributes or {}),
                )
            )

    def record_seed(self, *, scope: str, value: str) -> None:
        record = SeedTelemetryRecord(scope=scope, value=value)
        with self._lock:
            if record not in self._seed_records:
                self._seed_records.append(record)

    def record_restart(
        self,
        *,
        submitter_id: str,
        declared_restarts: int,
        used_restarts: int,
    ) -> None:
        with self._lock:
            self._restart_records.append(
                RestartTelemetryRecord(
                    submitter_id=submitter_id,
                    declared_restarts=declared_restarts,
                    used_restarts=used_restarts,
                )
            )

    def record_budget(
        self,
        *,
        submitter_id: str,
        declared_candidate_limit: int,
        declared_wall_clock_seconds: int,
        attempted_candidate_count: int,
        accepted_candidate_count: int,
        rejected_candidate_count: int,
        omitted_candidate_count: int,
    ) -> None:
        with self._lock:
            self._budget_records.append(
                BudgetTelemetryRecord(
                    submitter_id=submitter_id,
                    declared_candidate_limit=declared_candidate_limit,
                    declared_wall_clock_seconds=declared_wall_clock_seconds,
                    attempted_candidate_count=attempted_candidate_count,
                    accepted_candidate_count=accepted_candidate_count,
                    rejected_candidate_count=rejected_candidate_count,
                    omitted_candidate_count=omitted_candidate_count,
                )
            )

    def record_artifact_io(
        self,
        *,
        operation: str,
        size_bytes: int,
        elapsed_seconds: float,
        cache_hit: bool = False,
    ) -> None:
        with self._lock:
            if operation == "read":
                self._artifact_read_operation_count += 1
                self._artifact_read_bytes += max(0, size_bytes)
                self._artifact_read_wall_time_seconds += max(0.0, elapsed_seconds)
                return
            if operation != "write":
                raise ValueError(f"unsupported artifact store operation: {operation}")
            self._artifact_write_operation_count += 1
            self._artifact_write_bytes += max(0, size_bytes)
            self._artifact_write_wall_time_seconds += max(0.0, elapsed_seconds)
            if cache_hit:
                self._artifact_cache_hit_count += 1
            else:
                self._artifact_cache_miss_count += 1

    def build_artifact(
        self,
        *,
        profile_kind: str,
        subject_id: str,
        status: str = "completed",
        attributes: Mapping[str, Any] | None = None,
    ) -> PerformanceTelemetryArtifact:
        current_memory_bytes, peak_memory_bytes = tracemalloc.get_traced_memory()
        with self._lock:
            return PerformanceTelemetryArtifact(
                profile_kind=profile_kind,
                subject_id=subject_id,
                status=status,
                wall_time_seconds=time.perf_counter() - self._start_wall,
                cpu_time_seconds=time.process_time() - self._start_cpu,
                current_memory_bytes=current_memory_bytes,
                peak_memory_bytes=peak_memory_bytes,
                rss_max_bytes=_rss_bytes(),
                spans=tuple(
                    sorted(
                        self._spans,
                        key=lambda record: (
                            record.category,
                            record.name,
                            _json_sort_key(record.attributes),
                        ),
                    )
                ),
                measurements=tuple(
                    sorted(
                        self._measurements,
                        key=lambda record: (
                            record.category,
                            record.name,
                            _json_sort_key(record.attributes),
                        ),
                    )
                ),
                seed_records=tuple(
                    sorted(
                        self._seed_records,
                        key=lambda record: (record.scope, record.value),
                    )
                ),
                restart_records=tuple(
                    sorted(
                        self._restart_records,
                        key=lambda record: _submitter_sort_key(record.submitter_id),
                    )
                ),
                budget_records=tuple(
                    sorted(
                        self._budget_records,
                        key=lambda record: _submitter_sort_key(record.submitter_id),
                    )
                ),
                artifact_store=ArtifactStoreTelemetry(
                    read_operation_count=self._artifact_read_operation_count,
                    write_operation_count=self._artifact_write_operation_count,
                    read_bytes=self._artifact_read_bytes,
                    write_bytes=self._artifact_write_bytes,
                    cache_hit_count=self._artifact_cache_hit_count,
                    cache_miss_count=self._artifact_cache_miss_count,
                    read_wall_time_seconds=self._artifact_read_wall_time_seconds,
                    write_wall_time_seconds=self._artifact_write_wall_time_seconds,
                ),
                attributes=dict(attributes or {}),
            )


def write_performance_telemetry(
    path: Path | str,
    artifact: PerformanceTelemetryArtifact,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(artifact.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved


def load_performance_telemetry(path: Path | str) -> PerformanceTelemetryArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("performance telemetry payload must deserialize to a mapping")

    spans = tuple(
        TelemetrySpanRecord(
            name=str(record["name"]),
            category=str(record["category"]),
            status=str(record["status"]),
            wall_time_seconds=float(record["wall_time_seconds"]),
            cpu_time_seconds=float(record["cpu_time_seconds"]),
            start_memory_bytes=int(record["start_memory_bytes"]),
            end_memory_bytes=int(record["end_memory_bytes"]),
            peak_memory_bytes=int(record["peak_memory_bytes"]),
            rss_max_bytes=(
                None
                if record.get("rss_max_bytes") is None
                else int(record["rss_max_bytes"])
            ),
            attributes=_mapping_copy(record.get("attributes")),
            error_type=_string_or_none(record.get("error_type")),
        )
        for record in _sequence(payload.get("spans"))
    )
    measurements = tuple(
        TelemetryMeasurement(
            name=str(record["name"]),
            category=str(record["category"]),
            value=record["value"],
            unit=str(record["unit"]),
            attributes=_mapping_copy(record.get("attributes")),
        )
        for record in _sequence(payload.get("measurements"))
    )
    seed_records = tuple(
        SeedTelemetryRecord(
            scope=str(record["scope"]),
            value=str(record["value"]),
        )
        for record in _sequence(payload.get("seed_records"))
    )
    restart_records = tuple(
        RestartTelemetryRecord(
            submitter_id=str(record["submitter_id"]),
            declared_restarts=int(record["declared_restarts"]),
            used_restarts=int(record["used_restarts"]),
        )
        for record in _sequence(payload.get("restart_records"))
    )
    budget_records = tuple(
        BudgetTelemetryRecord(
            submitter_id=str(record["submitter_id"]),
            declared_candidate_limit=int(record["declared_candidate_limit"]),
            declared_wall_clock_seconds=int(record["declared_wall_clock_seconds"]),
            attempted_candidate_count=int(record["attempted_candidate_count"]),
            accepted_candidate_count=int(record["accepted_candidate_count"]),
            rejected_candidate_count=int(record["rejected_candidate_count"]),
            omitted_candidate_count=int(record["omitted_candidate_count"]),
        )
        for record in _sequence(payload.get("budget_records"))
    )
    artifact_store_payload = payload.get("artifact_store")
    if not isinstance(artifact_store_payload, Mapping):
        raise ValueError("performance telemetry artifact_store must be a mapping")

    return PerformanceTelemetryArtifact(
        artifact_type=str(payload.get("artifact_type", "performance_telemetry")),
        artifact_version=str(payload.get("artifact_version", _ARTIFACT_VERSION)),
        profile_kind=str(payload["profile_kind"]),
        subject_id=str(payload["subject_id"]),
        status=str(payload["status"]),
        wall_time_seconds=float(payload["wall_time_seconds"]),
        cpu_time_seconds=float(payload["cpu_time_seconds"]),
        current_memory_bytes=int(payload["current_memory_bytes"]),
        peak_memory_bytes=int(payload["peak_memory_bytes"]),
        rss_max_bytes=(
            None
            if payload.get("rss_max_bytes") is None
            else int(payload["rss_max_bytes"])
        ),
        spans=spans,
        measurements=measurements,
        seed_records=seed_records,
        restart_records=restart_records,
        budget_records=budget_records,
        artifact_store=ArtifactStoreTelemetry(
            read_operation_count=int(artifact_store_payload["read_operation_count"]),
            write_operation_count=int(artifact_store_payload["write_operation_count"]),
            read_bytes=int(artifact_store_payload["read_bytes"]),
            write_bytes=int(artifact_store_payload["write_bytes"]),
            cache_hit_count=int(artifact_store_payload["cache_hit_count"]),
            cache_miss_count=int(artifact_store_payload["cache_miss_count"]),
            read_wall_time_seconds=float(
                artifact_store_payload["read_wall_time_seconds"]
            ),
            write_wall_time_seconds=float(
                artifact_store_payload["write_wall_time_seconds"]
            ),
        ),
        attributes=_mapping_copy(payload.get("attributes")),
    )


def evaluate_performance_budget(
    artifact: PerformanceTelemetryArtifact,
    budget: PerformanceBudget,
) -> PerformanceBudgetEvaluation:
    failure_reasons: list[str] = []
    if artifact.wall_time_seconds > budget.max_wall_time_seconds:
        failure_reasons.append(
            "wall_time_seconds"
            f"={artifact.wall_time_seconds:.6f} exceeds "
            f"{budget.max_wall_time_seconds:.6f}"
        )
    if (
        budget.max_cpu_time_seconds is not None
        and artifact.cpu_time_seconds > budget.max_cpu_time_seconds
    ):
        failure_reasons.append(
            "cpu_time_seconds"
            f"={artifact.cpu_time_seconds:.6f} exceeds "
            f"{budget.max_cpu_time_seconds:.6f}"
        )
    if (
        budget.max_peak_memory_bytes is not None
        and artifact.peak_memory_bytes > budget.max_peak_memory_bytes
    ):
        failure_reasons.append(
            "peak_memory_bytes"
            f"={artifact.peak_memory_bytes} exceeds {budget.max_peak_memory_bytes}"
        )
    return PerformanceBudgetEvaluation(
        budget_id=budget.budget_id,
        subject_id=artifact.subject_id,
        passed=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        observed_wall_time_seconds=artifact.wall_time_seconds,
        observed_cpu_time_seconds=artifact.cpu_time_seconds,
        observed_peak_memory_bytes=artifact.peak_memory_bytes,
    )


def collect_performance_suite(
    *,
    suite_id: str,
    telemetry_paths: Sequence[Path | str],
) -> PerformanceSuite:
    profiles = tuple(
        PerformanceSuiteEntry(
            telemetry_path=Path(path).resolve(),
            profile_kind=artifact.profile_kind,
            subject_id=artifact.subject_id,
            status=artifact.status,
            wall_time_seconds=artifact.wall_time_seconds,
            cpu_time_seconds=artifact.cpu_time_seconds,
            peak_memory_bytes=artifact.peak_memory_bytes,
        )
        for path in telemetry_paths
        for artifact in (load_performance_telemetry(path),)
    )
    return PerformanceSuite(
        suite_id=suite_id,
        profiles=profiles,
        total_wall_time_seconds=sum(profile.wall_time_seconds for profile in profiles),
        max_wall_time_seconds=max(
            (profile.wall_time_seconds for profile in profiles),
            default=0.0,
        ),
        max_peak_memory_bytes=max(
            (profile.peak_memory_bytes for profile in profiles),
            default=0,
        ),
    )


def evaluate_suite_performance_budget(
    suite: PerformanceSuite,
    budget: SuitePerformanceBudget,
) -> SuitePerformanceBudgetEvaluation:
    failure_reasons: list[str] = []
    if suite.total_wall_time_seconds > budget.max_total_wall_time_seconds:
        failure_reasons.append(
            "total_wall_time_seconds"
            f"={suite.total_wall_time_seconds:.6f} exceeds "
            f"{budget.max_total_wall_time_seconds:.6f}"
        )
    if suite.max_wall_time_seconds > budget.max_profile_wall_time_seconds:
        failure_reasons.append(
            "max_wall_time_seconds"
            f"={suite.max_wall_time_seconds:.6f} exceeds "
            f"{budget.max_profile_wall_time_seconds:.6f}"
        )
    if (
        budget.max_peak_memory_bytes is not None
        and suite.max_peak_memory_bytes > budget.max_peak_memory_bytes
    ):
        failure_reasons.append(
            "max_peak_memory_bytes"
            f"={suite.max_peak_memory_bytes} exceeds {budget.max_peak_memory_bytes}"
        )
    return SuitePerformanceBudgetEvaluation(
        budget_id=budget.budget_id,
        suite_id=suite.suite_id,
        passed=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        profile_count=suite.profile_count,
        total_wall_time_seconds=suite.total_wall_time_seconds,
        max_wall_time_seconds=suite.max_wall_time_seconds,
        max_peak_memory_bytes=suite.max_peak_memory_bytes,
    )


def evaluate_candidate_throughput_budget(
    *,
    candidate_count: int,
    elapsed_seconds: float,
    budget: CandidateThroughputBudget,
) -> CandidateThroughputBudgetEvaluation:
    resolved_candidate_count = max(0, int(candidate_count))
    resolved_elapsed_seconds = max(0.0, float(elapsed_seconds))
    observed_rate = (
        0.0
        if resolved_elapsed_seconds <= 0.0
        else resolved_candidate_count / resolved_elapsed_seconds
    )
    failure_reasons: list[str] = []
    if observed_rate < budget.min_candidates_per_second:
        failure_reasons.append(
            "candidates_per_second"
            f"={observed_rate:.6f} below "
            f"{budget.min_candidates_per_second:.6f}"
        )
    return CandidateThroughputBudgetEvaluation(
        budget_id=budget.budget_id,
        passed=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        candidate_count=resolved_candidate_count,
        elapsed_seconds=resolved_elapsed_seconds,
        observed_candidates_per_second=observed_rate,
        min_candidates_per_second=float(budget.min_candidates_per_second),
    )


def build_engine_runtime_budget_report(
    *,
    engine_id: str,
    elapsed_seconds: float,
    candidate_count: int,
    status: str,
    budget: EngineRuntimeBudget,
    reason_codes: Sequence[str] = (),
    details: Mapping[str, Any] | None = None,
) -> EngineRuntimeBudgetReport:
    resolved_elapsed_seconds = max(0.0, float(elapsed_seconds))
    resolved_candidate_count = max(0, int(candidate_count))
    failure_reasons: list[str] = []
    if resolved_elapsed_seconds > budget.timeout_seconds:
        failure_reasons.append(
            "elapsed_seconds"
            f"={resolved_elapsed_seconds:.6f} exceeds "
            f"{budget.timeout_seconds:.6f}"
        )
    if resolved_candidate_count > budget.candidate_limit:
        failure_reasons.append(
            f"candidate_count={resolved_candidate_count} exceeds "
            f"{budget.candidate_limit}"
        )
    if status == "timeout" and not any(
        reason.startswith("elapsed_seconds=") for reason in failure_reasons
    ):
        failure_reasons.append("engine reported timeout")
    if status == "failed":
        failure_reasons.append("engine reported failed")
    return EngineRuntimeBudgetReport(
        engine_id=engine_id,
        budget_id=budget.budget_id,
        status=status,
        passed=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        elapsed_seconds=resolved_elapsed_seconds,
        candidate_count=resolved_candidate_count,
        reason_codes=tuple(str(code) for code in reason_codes),
        resource_limits=budget.resource_limits(),
        details=dict(details or {}),
    )


def degradation_decision_from_budget_report(
    report: EngineRuntimeBudgetReport,
) -> RuntimeDegradationDecision:
    if not report.passed:
        return RuntimeDegradationDecision(
            status="abstained",
            reason_codes=(
                report.reason_codes
                if report.reason_codes
                else tuple(report.failure_reasons)
            ),
            claim_publication_allowed=False,
        )
    if report.status == "partial" or report.reason_codes:
        return RuntimeDegradationDecision(
            status="degraded",
            reason_codes=report.reason_codes or ("partial_result",),
            claim_publication_allowed=False,
        )
    return RuntimeDegradationDecision(
        status="completed",
        reason_codes=(),
        claim_publication_allowed=False,
    )


def benchmark_budget_report(
    *,
    task_id: str,
    candidate_limit: int,
    wall_clock_seconds: int,
    parallel_workers: int,
    submitter_count: int,
) -> dict[str, Any]:
    return {
        "budget_id": f"benchmark_task_budget:{task_id}",
        "candidate_limit": int(candidate_limit),
        "wall_clock_seconds": int(wall_clock_seconds),
        "parallel_workers": max(1, int(parallel_workers)),
        "submitter_count": int(submitter_count),
        "status": "reported",
    }


def _mapping_copy(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("expected a mapping value")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError("expected a sequence of mappings")
    records = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("expected sequence items to be mappings")
        records.append(item)
    return tuple(records)


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "PerformanceBudget",
    "PerformanceBudgetEvaluation",
    "PerformanceTelemetryArtifact",
    "PerformanceSuite",
    "PerformanceSuiteEntry",
    "SuitePerformanceBudget",
    "SuitePerformanceBudgetEvaluation",
    "TelemetryRecorder",
    "collect_performance_suite",
    "evaluate_performance_budget",
    "evaluate_suite_performance_budget",
    "load_performance_telemetry",
    "write_performance_telemetry",
]
