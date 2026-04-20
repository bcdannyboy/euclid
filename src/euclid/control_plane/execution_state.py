from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from euclid.contracts.refs import TypedRef


def _canonical_json(value: Mapping[str, Any] | None) -> str:
    payload = dict(value or {})
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _load_json(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("execution-state payloads must deserialize to a mapping")
    return payload


@dataclass(frozen=True)
class StageEventRecord:
    run_id: str
    sequence_id: int
    state_id: str
    stage: str
    module_id: str
    status: str
    details: dict[str, Any]


@dataclass(frozen=True)
class FreezeMarkerRecord:
    run_id: str
    marker_id: str
    state_id: str
    manifest_ref: TypedRef | None
    details: dict[str, Any]


@dataclass(frozen=True)
class BudgetCounterRecord:
    run_id: str
    counter_name: str
    consumed: int
    limit: int | None
    unit: str


@dataclass(frozen=True)
class SeedRegistryRecord:
    run_id: str
    seed_scope: str
    seed_value: str
    recorded_by: str | None


@dataclass(frozen=True)
class WorkerMetadataRecord:
    run_id: str
    worker_id: str
    module_id: str
    status: str
    details: dict[str, Any]


@dataclass(frozen=True)
class StepStateRecord:
    run_id: str
    step_id: str
    module_id: str
    status: str
    cursor: str | None
    details: dict[str, Any]


@dataclass(frozen=True)
class RunExecutionSnapshot:
    run_id: str
    stage_events: tuple[StageEventRecord, ...]
    freeze_markers: tuple[FreezeMarkerRecord, ...]
    budget_counters: tuple[BudgetCounterRecord, ...]
    seed_records: tuple[SeedRegistryRecord, ...]
    worker_metadata: tuple[WorkerMetadataRecord, ...]
    step_states: tuple[StepStateRecord, ...]


class SQLiteExecutionStateStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path)
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def append_stage_event(
        self,
        *,
        run_id: str,
        state_id: str,
        stage: str,
        module_id: str,
        status: str,
        details: Mapping[str, Any] | None = None,
    ) -> StageEventRecord:
        cursor = self._connection.execute(
            """
            INSERT INTO stage_events (
                run_id,
                state_id,
                stage,
                module_id,
                status,
                details_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                state_id,
                stage,
                module_id,
                status,
                _canonical_json(details),
            ),
        )
        self._connection.commit()
        return StageEventRecord(
            run_id=run_id,
            sequence_id=int(cursor.lastrowid),
            state_id=state_id,
            stage=stage,
            module_id=module_id,
            status=status,
            details=dict(details or {}),
        )

    def upsert_freeze_marker(
        self,
        *,
        run_id: str,
        marker_id: str,
        state_id: str,
        manifest_ref: TypedRef | None,
        details: Mapping[str, Any] | None = None,
    ) -> FreezeMarkerRecord:
        self._connection.execute(
            """
            INSERT INTO freeze_markers (
                run_id,
                marker_id,
                state_id,
                manifest_schema_name,
                manifest_object_id,
                details_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, marker_id) DO UPDATE SET
                state_id = excluded.state_id,
                manifest_schema_name = excluded.manifest_schema_name,
                manifest_object_id = excluded.manifest_object_id,
                details_json = excluded.details_json
            """,
            (
                run_id,
                marker_id,
                state_id,
                manifest_ref.schema_name if manifest_ref is not None else None,
                manifest_ref.object_id if manifest_ref is not None else None,
                _canonical_json(details),
            ),
        )
        self._connection.commit()
        return FreezeMarkerRecord(
            run_id=run_id,
            marker_id=marker_id,
            state_id=state_id,
            manifest_ref=manifest_ref,
            details=dict(details or {}),
        )

    def set_budget_counter(
        self,
        *,
        run_id: str,
        counter_name: str,
        consumed: int,
        limit: int | None,
        unit: str,
    ) -> BudgetCounterRecord:
        self._connection.execute(
            """
            INSERT INTO budget_counters (
                run_id,
                counter_name,
                consumed,
                limit_value,
                unit
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id, counter_name) DO UPDATE SET
                consumed = excluded.consumed,
                limit_value = excluded.limit_value,
                unit = excluded.unit
            """,
            (run_id, counter_name, consumed, limit, unit),
        )
        self._connection.commit()
        return BudgetCounterRecord(
            run_id=run_id,
            counter_name=counter_name,
            consumed=consumed,
            limit=limit,
            unit=unit,
        )

    def record_seed(
        self,
        *,
        run_id: str,
        seed_scope: str,
        seed_value: str,
        recorded_by: str | None,
    ) -> SeedRegistryRecord:
        self._connection.execute(
            """
            INSERT INTO seed_records (
                run_id,
                seed_scope,
                seed_value,
                recorded_by
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id, seed_scope) DO UPDATE SET
                seed_value = excluded.seed_value,
                recorded_by = excluded.recorded_by
            """,
            (run_id, seed_scope, seed_value, recorded_by),
        )
        self._connection.commit()
        return SeedRegistryRecord(
            run_id=run_id,
            seed_scope=seed_scope,
            seed_value=seed_value,
            recorded_by=recorded_by,
        )

    def upsert_worker_metadata(
        self,
        *,
        run_id: str,
        worker_id: str,
        module_id: str,
        status: str,
        details: Mapping[str, Any] | None = None,
    ) -> WorkerMetadataRecord:
        self._connection.execute(
            """
            INSERT INTO worker_metadata (
                run_id,
                worker_id,
                module_id,
                status,
                details_json
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id, worker_id) DO UPDATE SET
                module_id = excluded.module_id,
                status = excluded.status,
                details_json = excluded.details_json
            """,
            (
                run_id,
                worker_id,
                module_id,
                status,
                _canonical_json(details),
            ),
        )
        self._connection.commit()
        return WorkerMetadataRecord(
            run_id=run_id,
            worker_id=worker_id,
            module_id=module_id,
            status=status,
            details=dict(details or {}),
        )

    def save_step_state(
        self,
        *,
        run_id: str,
        step_id: str,
        module_id: str,
        status: str,
        cursor: str | None,
        details: Mapping[str, Any] | None = None,
    ) -> StepStateRecord:
        self._connection.execute(
            """
            INSERT INTO step_states (
                run_id,
                step_id,
                module_id,
                status,
                cursor,
                details_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, step_id) DO UPDATE SET
                module_id = excluded.module_id,
                status = excluded.status,
                cursor = excluded.cursor,
                details_json = excluded.details_json
            """,
            (
                run_id,
                step_id,
                module_id,
                status,
                cursor,
                _canonical_json(details),
            ),
        )
        self._connection.commit()
        return StepStateRecord(
            run_id=run_id,
            step_id=step_id,
            module_id=module_id,
            status=status,
            cursor=cursor,
            details=dict(details or {}),
        )

    def load_run_snapshot(self, run_id: str) -> RunExecutionSnapshot:
        return RunExecutionSnapshot(
            run_id=run_id,
            stage_events=self._load_stage_events(run_id),
            freeze_markers=self._load_freeze_markers(run_id),
            budget_counters=self._load_budget_counters(run_id),
            seed_records=self._load_seed_records(run_id),
            worker_metadata=self._load_worker_metadata(run_id),
            step_states=self._load_step_states(run_id),
        )

    def _ensure_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stage_events (
                sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                state_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                module_id TEXT NOT NULL,
                status TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS freeze_markers (
                run_id TEXT NOT NULL,
                marker_id TEXT NOT NULL,
                state_id TEXT NOT NULL,
                manifest_schema_name TEXT,
                manifest_object_id TEXT,
                details_json TEXT NOT NULL,
                PRIMARY KEY (run_id, marker_id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS budget_counters (
                run_id TEXT NOT NULL,
                counter_name TEXT NOT NULL,
                consumed INTEGER NOT NULL,
                limit_value INTEGER,
                unit TEXT NOT NULL,
                PRIMARY KEY (run_id, counter_name)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS seed_records (
                run_id TEXT NOT NULL,
                seed_scope TEXT NOT NULL,
                seed_value TEXT NOT NULL,
                recorded_by TEXT,
                PRIMARY KEY (run_id, seed_scope)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS worker_metadata (
                run_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                module_id TEXT NOT NULL,
                status TEXT NOT NULL,
                details_json TEXT NOT NULL,
                PRIMARY KEY (run_id, worker_id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS step_states (
                run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                module_id TEXT NOT NULL,
                status TEXT NOT NULL,
                cursor TEXT,
                details_json TEXT NOT NULL,
                PRIMARY KEY (run_id, step_id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS stage_events_run_idx
            ON stage_events(run_id, sequence_id)
            """
        )
        self._connection.commit()

    def _load_stage_events(self, run_id: str) -> tuple[StageEventRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                run_id,
                sequence_id,
                state_id,
                stage,
                module_id,
                status,
                details_json
            FROM stage_events
            WHERE run_id = ?
            ORDER BY sequence_id
            """,
            (run_id,),
        ).fetchall()
        return tuple(
            StageEventRecord(
                run_id=row["run_id"],
                sequence_id=row["sequence_id"],
                state_id=row["state_id"],
                stage=row["stage"],
                module_id=row["module_id"],
                status=row["status"],
                details=_load_json(row["details_json"]),
            )
            for row in rows
        )

    def _load_freeze_markers(self, run_id: str) -> tuple[FreezeMarkerRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                run_id,
                marker_id,
                state_id,
                manifest_schema_name,
                manifest_object_id,
                details_json
            FROM freeze_markers
            WHERE run_id = ?
            ORDER BY marker_id
            """,
            (run_id,),
        ).fetchall()
        return tuple(
            FreezeMarkerRecord(
                run_id=row["run_id"],
                marker_id=row["marker_id"],
                state_id=row["state_id"],
                manifest_ref=(
                    TypedRef(
                        schema_name=row["manifest_schema_name"],
                        object_id=row["manifest_object_id"],
                    )
                    if row["manifest_schema_name"] is not None
                    and row["manifest_object_id"] is not None
                    else None
                ),
                details=_load_json(row["details_json"]),
            )
            for row in rows
        )

    def _load_budget_counters(self, run_id: str) -> tuple[BudgetCounterRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                run_id,
                counter_name,
                consumed,
                limit_value,
                unit
            FROM budget_counters
            WHERE run_id = ?
            ORDER BY counter_name
            """,
            (run_id,),
        ).fetchall()
        return tuple(
            BudgetCounterRecord(
                run_id=row["run_id"],
                counter_name=row["counter_name"],
                consumed=row["consumed"],
                limit=row["limit_value"],
                unit=row["unit"],
            )
            for row in rows
        )

    def _load_seed_records(self, run_id: str) -> tuple[SeedRegistryRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                run_id,
                seed_scope,
                seed_value,
                recorded_by
            FROM seed_records
            WHERE run_id = ?
            ORDER BY seed_scope
            """,
            (run_id,),
        ).fetchall()
        return tuple(
            SeedRegistryRecord(
                run_id=row["run_id"],
                seed_scope=row["seed_scope"],
                seed_value=row["seed_value"],
                recorded_by=row["recorded_by"],
            )
            for row in rows
        )

    def _load_worker_metadata(self, run_id: str) -> tuple[WorkerMetadataRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                run_id,
                worker_id,
                module_id,
                status,
                details_json
            FROM worker_metadata
            WHERE run_id = ?
            ORDER BY worker_id
            """,
            (run_id,),
        ).fetchall()
        return tuple(
            WorkerMetadataRecord(
                run_id=row["run_id"],
                worker_id=row["worker_id"],
                module_id=row["module_id"],
                status=row["status"],
                details=_load_json(row["details_json"]),
            )
            for row in rows
        )

    def _load_step_states(self, run_id: str) -> tuple[StepStateRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                run_id,
                step_id,
                module_id,
                status,
                cursor,
                details_json
            FROM step_states
            WHERE run_id = ?
            ORDER BY step_id
            """,
            (run_id,),
        ).fetchall()
        return tuple(
            StepStateRecord(
                run_id=row["run_id"],
                step_id=row["step_id"],
                module_id=row["module_id"],
                status=row["status"],
                cursor=row["cursor"],
                details=_load_json(row["details_json"]),
            )
            for row in rows
        )


__all__ = [
    "BudgetCounterRecord",
    "FreezeMarkerRecord",
    "RunExecutionSnapshot",
    "SeedRegistryRecord",
    "SQLiteExecutionStateStore",
    "StageEventRecord",
    "StepStateRecord",
    "WorkerMetadataRecord",
]
