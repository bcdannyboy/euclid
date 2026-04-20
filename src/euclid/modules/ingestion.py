from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.manifests.base import ManifestEnvelope
from euclid.modules._shared import canonical_timestamp, ensure_finite_float
from euclid.runtime.hashing import normalize_json_value, sha256_digest

try:
    import pandas as pd
except ImportError:  # pragma: no cover - install path should provide pandas.
    pd = None


_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_FLOAT_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
_CANONICAL_FIELD_ALIASES = {
    "entity": ("entity", "series_id"),
    "event_time": ("event_time",),
    "availability_time": ("availability_time", "available_at"),
    "target": ("target", "observed_value"),
    "source_id": ("source_id",),
}
_SIDE_INFORMATION_EXCLUDED_FIELDS = frozenset(
    alias for aliases in _CANONICAL_FIELD_ALIASES.values() for alias in aliases
)


@dataclass(frozen=True)
class ObservationRecord:
    entity: str
    event_time: str
    availability_time: str
    target: float | None
    side_information: Mapping[str, Any]
    payload_hash: str
    source_id: str | None = None

    @property
    def series_id(self) -> str:
        return self.entity

    @property
    def available_at(self) -> str:
        return self.availability_time

    @property
    def observed_value(self) -> float:
        if self.target is None:
            raise ContractValidationError(
                code="missing_target_not_coded",
                message="observations with missing targets stay out of coding",
                field_path="target",
            )
        return self.target

    @property
    def revision_id(self) -> int:
        revision = self.side_information.get("revision_id", 0)
        return int(revision)

    def as_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "entity": self.entity,
            "event_time": self.event_time,
            "availability_time": self.availability_time,
            "target": self.target,
            "side_information": dict(self.side_information),
            "payload_hash": self.payload_hash,
        }
        if self.source_id is not None:
            body["source_id"] = self.source_id
        return body

    def to_manifest(self, catalog: ContractCatalog) -> ManifestEnvelope:
        return ManifestEnvelope.build(
            schema_name="observation_record@1.0.0",
            module_id="ingestion",
            body=self.as_dict(),
            catalog=catalog,
        )


@dataclass(frozen=True)
class AdmittedOrderedNumericData:
    entity_panel: tuple[str, ...]
    observations: tuple[ObservationRecord, ...]
    order_relation: str = "entity_then_event_time_then_availability_time"

    @property
    def entity(self) -> str:
        if len(self.entity_panel) != 1:
            raise ContractValidationError(
                code="entity_panel_required",
                message=(
                    "multi-entity datasets must use entity_panel rather than the "
                    "singleton entity alias"
                ),
                field_path="entity_panel",
                details={"entity_panel": list(self.entity_panel)},
            )
        return self.entity_panel[0]

    @property
    def admitted_side_information_fields(self) -> tuple[str, ...]:
        names = {
            key
            for observation in self.observations
            for key in observation.side_information
        }
        return tuple(sorted(names))

    @property
    def coded_observations(self) -> tuple[ObservationRecord, ...]:
        return tuple(
            observation
            for observation in self.observations
            if observation.target is not None
        )

    @property
    def coded_targets(self) -> tuple[float, ...]:
        return tuple(
            observation.target
            for observation in self.observations
            if observation.target is not None
        )


def ingest_csv_dataset(path: Path) -> AdmittedOrderedNumericData:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return _build_admitted_dataset(rows, source_label=str(path))


def ingest_parquet_dataset(path: Path) -> AdmittedOrderedNumericData:
    frame = _require_pandas().read_parquet(path)
    return ingest_dataframe_dataset(frame, source_label=str(path))


def ingest_dataframe_dataset(
    frame: Any,
    *,
    source_label: str = "<dataframe>",
) -> AdmittedOrderedNumericData:
    if not hasattr(frame, "to_dict"):
        raise ContractValidationError(
            code="invalid_dataframe_input",
            message="dataframe ingestion requires a tabular object with to_dict",
            field_path="frame",
        )
    rows = frame.to_dict(orient="records")
    return _build_admitted_dataset(rows, source_label=source_label)


def ingest_csv_observations(path: Path) -> tuple[ObservationRecord, ...]:
    return ingest_csv_dataset(path).coded_observations


def _build_admitted_dataset(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_label: str,
) -> AdmittedOrderedNumericData:
    normalized: list[tuple[tuple[str, str, str, str, int], ObservationRecord]] = []
    entities: set[str] = set()

    for row_index, row in enumerate(rows):
        source_id = _normalize_optional_string(
            _field_value(row, "source_id"),
            field_path=f"{source_label}[{row_index}].source_id",
        )
        entity = _normalize_required_string(
            _field_value(row, "entity"),
            field_path=f"{source_label}[{row_index}].entity",
            aliases=_CANONICAL_FIELD_ALIASES["entity"],
        )
        event_time = canonical_timestamp(
            _field_value(row, "event_time"),
        )
        availability_time = canonical_timestamp(
            _field_value(row, "availability_time"),
        )
        target = _normalize_target_value(
            _field_value(row, "target"),
            field_path=f"{source_label}[{row_index}].target",
        )
        side_information = _extract_side_information(row)
        payload = {
            "entity": entity,
            "event_time": event_time,
            "availability_time": availability_time,
            "target": target,
            "side_information": side_information,
        }
        if source_id is not None:
            payload["source_id"] = source_id
        payload_hash = sha256_digest(payload)
        observation = ObservationRecord(
            entity=entity,
            event_time=event_time,
            availability_time=availability_time,
            target=target,
            side_information=side_information,
            payload_hash=payload_hash,
            source_id=source_id,
        )
        normalized.append(
            (
                (
                    entity,
                    event_time,
                    availability_time,
                    payload_hash,
                    row_index,
                ),
                observation,
            )
        )
        entities.add(entity)

    if not normalized:
        raise ContractValidationError(
            code="empty_dataset",
            message="at least one observation is required",
            field_path=source_label,
        )
    observations = tuple(
        observation for _, observation in sorted(normalized, key=lambda item: item[0])
    )
    return AdmittedOrderedNumericData(
        entity_panel=tuple(sorted(entities)),
        observations=observations,
    )


def _field_value(row: Mapping[str, Any], field_name: str) -> Any:
    for alias in _CANONICAL_FIELD_ALIASES[field_name]:
        if alias in row:
            return row[alias]
    return None


def _normalize_required_string(
    value: Any,
    *,
    field_path: str,
    aliases: tuple[str, ...],
) -> str:
    candidate = _normalize_optional_string(value, field_path=field_path)
    if candidate is None:
        raise ContractValidationError(
            code="missing_required_field",
            message=f"{aliases[0]} is required",
            field_path=field_path,
            details={"accepted_aliases": aliases},
        )
    return candidate


def _normalize_optional_string(value: Any, *, field_path: str) -> str | None:
    if _is_missing_value(value):
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    return candidate


def _normalize_target_value(value: Any, *, field_path: str) -> float | None:
    if _is_missing_value(value):
        return None
    return ensure_finite_float(value, field_path=field_path)


def _extract_side_information(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in sorted(row):
        if key in _SIDE_INFORMATION_EXCLUDED_FIELDS:
            continue
        value = _normalize_side_information_value(row[key], field_path=key)
        if value is None:
            continue
        normalized[key] = value
    return normalized


def _normalize_side_information_value(value: Any, *, field_path: str) -> Any:
    if _is_missing_value(value):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, datetime):
        return canonical_timestamp(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        lowered = candidate.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if _INTEGER_PATTERN.match(candidate):
            return int(candidate)
        if _FLOAT_PATTERN.match(candidate):
            return ensure_finite_float(candidate, field_path=field_path)
        return candidate
    if isinstance(value, float):
        return ensure_finite_float(value, field_path=field_path)
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, Mapping):
        nested = {
            str(key): _normalize_side_information_value(
                nested_value,
                field_path=f"{field_path}.{key}",
            )
            for key, nested_value in value.items()
        }
        return normalize_json_value(
            {key: nested[key] for key in sorted(nested) if nested[key] is not None}
        )
    if isinstance(value, (list, tuple)):
        return normalize_json_value(
            [
                normalized
                for item in value
                if (
                    normalized := _normalize_side_information_value(
                        item,
                        field_path=field_path,
                    )
                )
                is not None
            ]
        )
    try:
        return normalize_json_value(value)
    except ContractValidationError as exc:
        raise ContractValidationError(
            code="invalid_side_information",
            message="side-information values must be canonically serializable",
            field_path=field_path,
        ) from exc


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if pd is not None:
        try:
            missing = pd.isna(value)
        except TypeError:
            missing = False
        if isinstance(missing, (bool, int)):
            return bool(missing)
    return isinstance(value, float) and math.isnan(value)


def _require_pandas():
    if pd is None:
        raise ContractValidationError(
            code="missing_dataframe_support",
            message="pandas is required for dataframe and parquet ingestion",
            field_path="pandas",
        )
    return pd
