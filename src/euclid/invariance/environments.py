from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EnvironmentSlice:
    environment_id: str
    row_indices: tuple[int, ...]
    entity_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "environment_id": self.environment_id,
            "row_indices": list(self.row_indices),
            "entity_ids": list(self.entity_ids),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class EnvironmentConstructionResult:
    status: str
    policy: Mapping[str, Any]
    slices: tuple[EnvironmentSlice, ...]
    replay_identity: str
    reason_codes: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "invariance_environment_construction@1.0.0",
            "status": self.status,
            "policy": dict(self.policy),
            "slices": [slice.as_dict() for slice in self.slices],
            "reason_codes": list(self.reason_codes),
            "replay_identity": self.replay_identity,
        }


def construct_environments(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    label_field: str = "environment",
    entity_field: str = "entity",
    time_field: str = "t",
    target_field: str = "target",
    min_environment_count: int = 2,
    min_rows_per_environment: int = 1,
    era_count: int = 2,
    intervention_windows: Sequence[Mapping[str, Any]] = (),
) -> EnvironmentConstructionResult:
    row_tuple = tuple(rows)
    policy_payload: dict[str, Any] = {
        "policy": policy,
        "label_field": label_field,
        "entity_field": entity_field,
        "time_field": time_field,
        "target_field": target_field,
        "min_environment_count": min_environment_count,
        "min_rows_per_environment": min_rows_per_environment,
    }
    if era_count:
        policy_payload["era_count"] = int(era_count)
    if intervention_windows:
        policy_payload["intervention_windows"] = [
            _stable_mapping(window) for window in intervention_windows
        ]

    if not row_tuple:
        return _result(
            status="empty_input",
            policy=policy_payload,
            slices=(),
            reason_codes=("empty_input",),
        )

    if policy == "explicit_label":
        slices = _explicit_label_slices(row_tuple, label_field, entity_field)
    elif policy == "entity":
        slices = _entity_slices(row_tuple, entity_field)
    elif policy == "rolling_era":
        slices = _rolling_era_slices(row_tuple, time_field, era_count, entity_field)
    elif policy == "volatility_regime":
        slices = _volatility_regime_slices(
            row_tuple,
            time_field=time_field,
            target_field=target_field,
            entity_field=entity_field,
        )
    elif policy == "intervention_window":
        slices = _intervention_window_slices(
            row_tuple,
            time_field=time_field,
            entity_field=entity_field,
            intervention_windows=intervention_windows,
        )
    else:
        return _result(
            status="invalid_policy",
            policy=policy_payload,
            slices=(),
            reason_codes=("invalid_environment_policy",),
        )

    slices = tuple(
        slice
        for slice in slices
        if len(slice.row_indices) >= int(min_rows_per_environment)
    )
    if len(slices) < int(min_environment_count):
        return _result(
            status="insufficient_environments",
            policy=policy_payload,
            slices=(),
            reason_codes=("insufficient_environments",),
        )
    return _result(status="constructed", policy=policy_payload, slices=slices)


def _explicit_label_slices(
    rows: Sequence[Mapping[str, Any]],
    label_field: str,
    entity_field: str,
) -> tuple[EnvironmentSlice, ...]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        label = str(row.get(label_field, "")).strip()
        if not label:
            label = "missing_label"
        groups.setdefault(label, []).append(index)
    return tuple(
        EnvironmentSlice(
            environment_id=label,
            row_indices=tuple(indices),
            entity_ids=_entity_ids(rows, indices, entity_field),
            metadata={"construction": "explicit_label"},
        )
        for label, indices in groups.items()
    )


def _entity_slices(
    rows: Sequence[Mapping[str, Any]],
    entity_field: str,
) -> tuple[EnvironmentSlice, ...]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        entity = str(row.get(entity_field, "")).strip()
        if not entity:
            entity = "missing_entity"
        groups.setdefault(entity, []).append(index)
    return tuple(
        EnvironmentSlice(
            environment_id=f"entity:{entity}",
            row_indices=tuple(indices),
            entity_ids=(entity,),
            metadata={"construction": "entity"},
        )
        for entity, indices in sorted(groups.items())
    )


def _rolling_era_slices(
    rows: Sequence[Mapping[str, Any]],
    time_field: str,
    era_count: int,
    entity_field: str,
) -> tuple[EnvironmentSlice, ...]:
    ordered = sorted(
        range(len(rows)),
        key=lambda index: _time_sort_key(rows[index].get(time_field), index),
    )
    count = max(1, int(era_count))
    chunk_size = max(1, math.ceil(len(ordered) / count))
    slices: list[EnvironmentSlice] = []
    for era_index in range(count):
        chunk = tuple(sorted(ordered[era_index * chunk_size : (era_index + 1) * chunk_size]))
        if not chunk:
            continue
        slices.append(
            EnvironmentSlice(
                environment_id=f"era:{era_index + 1}",
                row_indices=chunk,
                entity_ids=_entity_ids(rows, chunk, entity_field),
                metadata={"construction": "rolling_era", "era_index": era_index + 1},
            )
        )
    return tuple(slices)


def _volatility_regime_slices(
    rows: Sequence[Mapping[str, Any]],
    *,
    time_field: str,
    target_field: str,
    entity_field: str,
) -> tuple[EnvironmentSlice, ...]:
    ordered = sorted(
        range(len(rows)),
        key=lambda index: _time_sort_key(rows[index].get(time_field), index),
    )
    if len(ordered) < 2:
        return ()
    deltas: dict[int, float] = {ordered[0]: 0.0}
    previous = _finite_float(rows[ordered[0]].get(target_field))
    if previous is None:
        return ()
    for index in ordered[1:]:
        current = _finite_float(rows[index].get(target_field))
        if current is None:
            return ()
        deltas[index] = abs(current - previous)
        previous = current
    median_delta = sorted(deltas.values())[len(deltas) // 2]
    low = tuple(sorted(index for index, delta in deltas.items() if delta <= median_delta))
    high = tuple(sorted(index for index, delta in deltas.items() if delta > median_delta))
    return tuple(
        slice
        for slice in (
            EnvironmentSlice(
                environment_id="volatility:low",
                row_indices=low,
                entity_ids=_entity_ids(rows, low, entity_field),
                metadata={"construction": "volatility_regime"},
            ),
            EnvironmentSlice(
                environment_id="volatility:high",
                row_indices=high,
                entity_ids=_entity_ids(rows, high, entity_field),
                metadata={"construction": "volatility_regime"},
            ),
        )
        if slice.row_indices
    )


def _intervention_window_slices(
    rows: Sequence[Mapping[str, Any]],
    *,
    time_field: str,
    entity_field: str,
    intervention_windows: Sequence[Mapping[str, Any]],
) -> tuple[EnvironmentSlice, ...]:
    windows = tuple(_normalize_window(window) for window in intervention_windows)
    intervention_groups: dict[str, list[int]] = {label: [] for label, _, _ in windows}
    outside: list[int] = []
    for index, row in enumerate(rows):
        time_value = _finite_float(row.get(time_field))
        matched = False
        if time_value is not None:
            for label, start, end in windows:
                if start <= time_value <= end:
                    intervention_groups[label].append(index)
                    matched = True
        if not matched:
            outside.append(index)
    slices: list[EnvironmentSlice] = []
    if outside:
        slices.append(
            EnvironmentSlice(
                environment_id="outside_intervention",
                row_indices=tuple(outside),
                entity_ids=_entity_ids(rows, outside, entity_field),
                metadata={"construction": "intervention_window"},
            )
        )
    for label, start, end in windows:
        indices = tuple(intervention_groups[label])
        if not indices:
            continue
        slices.append(
            EnvironmentSlice(
                environment_id=f"intervention:{label}",
                row_indices=indices,
                entity_ids=_entity_ids(rows, indices, entity_field),
                metadata={
                    "construction": "intervention_window",
                    "window": {"start": _stable_number(start), "end": _stable_number(end)},
                },
            )
        )
    return tuple(slices)


def _normalize_window(window: Mapping[str, Any]) -> tuple[str, float, float]:
    label = str(window.get("label", "window")).strip() or "window"
    start = _finite_float(window.get("start"))
    end = _finite_float(window.get("end"))
    if start is None or end is None:
        return label, math.inf, -math.inf
    return label, min(start, end), max(start, end)


def _entity_ids(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    entity_field: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(rows[index].get(entity_field, "")).strip()
                for index in indices
                if str(rows[index].get(entity_field, "")).strip()
            }
        )
    )


def _time_sort_key(value: Any, fallback: int) -> tuple[int, Any]:
    numeric = _finite_float(value)
    if numeric is not None:
        return (0, numeric)
    return (1, str(value) if value is not None else fallback)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _stable_number(value: float) -> int | float:
    if float(value).is_integer():
        return int(value)
    return float(round(value, 12))


def _stable_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): payload[key] for key in sorted(payload)}


def _result(
    *,
    status: str,
    policy: Mapping[str, Any],
    slices: tuple[EnvironmentSlice, ...],
    reason_codes: tuple[str, ...] = (),
) -> EnvironmentConstructionResult:
    identity_payload = {
        "policy": dict(policy),
        "slices": [slice.as_dict() for slice in slices],
        "reason_codes": list(reason_codes),
        "status": status,
    }
    digest = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return EnvironmentConstructionResult(
        status=status,
        policy=dict(policy),
        slices=slices,
        reason_codes=reason_codes,
        replay_identity=f"invariance-environments:{digest}",
    )


__all__ = [
    "EnvironmentConstructionResult",
    "EnvironmentSlice",
    "construct_environments",
]
