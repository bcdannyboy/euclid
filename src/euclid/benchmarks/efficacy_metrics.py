from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

DESCRIPTIVE_NON_ABSTENTION_RATE = "descriptive_non_abstention_rate"
FALSE_HOLISTIC_RATE = "false_holistic_rate"
PLANTED_LAW_RECOVERY_RATE = "planted_law_recovery_rate"
THIN_EVIDENCE_PROBABILISTIC_ATTACHMENT_RATE = (
    "thin_evidence_probabilistic_attachment_rate"
)

NONSTATIONARY_DETECTION_PRECISION = "nonstationary_detection_precision"
NONSTATIONARY_DETECTION_RECALL = "nonstationary_detection_recall"
NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE = (
    "nonstationary_detection_hausdorff_distance"
)
NONSTATIONARY_DETECTION_DELAY = "nonstationary_detection_delay"
NONSTATIONARY_DETECTION_TOLERANCE = "nonstationary_detection_tolerance"

_NONSTATIONARY_DETECTION_METRIC_IDS = (
    NONSTATIONARY_DETECTION_PRECISION,
    NONSTATIONARY_DETECTION_RECALL,
    NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE,
    NONSTATIONARY_DETECTION_DELAY,
)
_PROBABILISTIC_FORECAST_OBJECT_TYPES = {
    "distribution",
    "interval",
    "quantile",
    "event_probability",
}
_EXACT_RECOVERY_STATUSES = {
    "equivalent_generator",
    "exact",
    "exact_generator",
    "exact_recovery",
    "identical_generator",
    "structurally_equivalent",
}
_NEAR_RECOVERY_STATUSES = {
    "near",
    "near_equivalent_generator",
    "near_recovery",
    "parameter_near_match",
}
_RETAINED_PROBABILISTIC_ATTACHMENT_STATUSES = {
    "attached",
    "passed",
    "promoted",
    "published",
    "retained",
}


MetricCompute = Callable[..., "EfficacyMetricResult"]


@dataclass(frozen=True)
class EfficacyMetricProvenance:
    row_count: int
    rows: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "rows": [_json_ready(row) for row in self.rows],
        }


@dataclass(frozen=True)
class EfficacyMetricResult:
    metric_id: str
    status: str
    observed_value: float | None
    unit: str
    numerator: float | int | None
    denominator: int | None
    provenance: EfficacyMetricProvenance
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    legacy_fields: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "metric_id": self.metric_id,
            "status": self.status,
            "observed_value": self.observed_value,
            "unit": self.unit,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "provenance": self.provenance.as_dict(),
            "details": _json_ready(dict(self.details)),
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload

    def legacy_dict(self) -> dict[str, Any]:
        if self.legacy_fields:
            return _json_ready(dict(self.legacy_fields))
        payload = {
            "metric_id": self.metric_id,
            "observed_value": self.observed_value,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }
        payload.update(dict(self.details))
        return _json_ready(payload)

    def __getitem__(self, key: str) -> Any:
        return self.legacy_dict()[key]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self.legacy_dict() == dict(other)
        return NotImplemented


@dataclass(frozen=True)
class EfficacyMetricSpec:
    metric_id: str
    description: str
    unit: str
    compute: MetricCompute


def efficacy_metric_registry() -> dict[str, EfficacyMetricSpec]:
    return dict(_REGISTRY)


def compute_efficacy_metric(
    metric_id: str,
    records: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> EfficacyMetricResult:
    spec = _REGISTRY.get(metric_id)
    if spec is None:
        raise ValueError(f"unknown efficacy metric id: {metric_id}")
    return spec.compute(records, **kwargs)


def compute_registered_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    metric_ids: Sequence[str] | None = None,
    **kwargs: Any,
) -> tuple[EfficacyMetricResult, ...]:
    selected_metric_ids = tuple(metric_ids or _REGISTRY)
    return tuple(
        compute_efficacy_metric(metric_id, records, **kwargs)
        for metric_id in selected_metric_ids
    )


def compute_planted_law_recovery_rate(
    records: Sequence[Mapping[str, Any]],
) -> EfficacyMetricResult:
    rows = _materialize_records(records)
    exact_count = 0
    near_count = 0
    missed_task_ids: list[str] = []

    for row in rows:
        recovery_class = _recovery_class(row)
        if recovery_class == "exact":
            exact_count += 1
        elif recovery_class == "near":
            near_count += 1
        else:
            task_id = _string_or_none(_lookup(row, "task_id"))
            if task_id is not None:
                missed_task_ids.append(task_id)

    recovered_count = exact_count + near_count
    denominator = len(rows)
    observed_value = _rate(recovered_count, denominator)
    status = "measured" if denominator else "missing"
    reason = None if denominator else "no_planted_law_rows"
    return EfficacyMetricResult(
        metric_id=PLANTED_LAW_RECOVERY_RATE,
        status=status,
        observed_value=observed_value,
        unit="fraction",
        numerator=recovered_count,
        denominator=denominator,
        reason=reason,
        provenance=_provenance(rows),
        details={
            "exact_recovery_count": exact_count,
            "near_recovery_count": near_count,
            "missed_count": denominator - recovered_count,
            "missed_task_ids": missed_task_ids,
        },
        legacy_fields={
            "metric_id": PLANTED_LAW_RECOVERY_RATE,
            "exact_recovery_numerator": exact_count,
            "near_recovery_numerator": near_count,
            "denominator": denominator,
            "observed_value": observed_value,
            "provenance": _legacy_provenance(rows),
        },
    )


def compute_planted_law_recovery_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return compute_planted_law_recovery_rate(records).legacy_dict()


def compute_false_holistic_claim_rate(
    records: Sequence[Mapping[str, Any]],
) -> EfficacyMetricResult:
    adversarial_rows = tuple(
        row for row in _materialize_records(records) if _is_adversarial_row(row)
    )
    false_positive_task_ids: list[str] = []
    for row in adversarial_rows:
        if _is_false_holistic_claim(row):
            task_id = _string_or_none(_lookup(row, "task_id"))
            if task_id is not None:
                false_positive_task_ids.append(task_id)

    numerator = len(false_positive_task_ids)
    denominator = len(adversarial_rows)
    observed_value = _rate(numerator, denominator)
    status = "measured" if denominator else "missing"
    reason = None if denominator else "no_adversarial_honesty_rows"
    return EfficacyMetricResult(
        metric_id=FALSE_HOLISTIC_RATE,
        status=status,
        observed_value=observed_value,
        unit="fraction",
        numerator=numerator,
        denominator=denominator,
        reason=reason,
        provenance=_provenance(adversarial_rows),
        details={
            "false_positive_count": numerator,
            "false_positive_task_ids": false_positive_task_ids,
        },
        legacy_fields={
            "metric_id": FALSE_HOLISTIC_RATE,
            "false_positive_count": numerator,
            "denominator": denominator,
            "observed_value": observed_value,
            "false_positive_task_ids": false_positive_task_ids,
        },
    )


def compute_descriptive_non_abstention_rate(
    records: Sequence[Mapping[str, Any]],
) -> EfficacyMetricResult:
    rows = _materialize_records(records)
    numerator = sum(1 for row in rows if _is_non_abstained(row))
    denominator = len(rows)
    observed_value = _rate(numerator, denominator)
    status = "measured" if denominator else "missing"
    reason = None if denominator else "no_descriptive_rows"
    return EfficacyMetricResult(
        metric_id=DESCRIPTIVE_NON_ABSTENTION_RATE,
        status=status,
        observed_value=observed_value,
        unit="fraction",
        numerator=numerator,
        denominator=denominator,
        reason=reason,
        provenance=_provenance(rows),
        details={"non_abstention_count": numerator},
    )


def compute_probabilistic_attachment_quality(
    records: Sequence[Mapping[str, Any]],
    *,
    minimum_calibration_count: int = 1,
) -> EfficacyMetricResult:
    rows = tuple(
        row for row in _materialize_records(records) if _is_probabilistic_row(row)
    )
    coverage_values: list[float] = []
    width_values: list[float] = []
    calibration_counts: list[int] = []
    calibration_statuses: list[str] = []
    thin_attachment_task_ids: list[str] = []

    for row in rows:
        coverage = _numeric_value(
            _lookup(row, "coverage", "coverage_observed", "interval_coverage")
        )
        width = _numeric_value(
            _lookup(row, "width", "mean_width", "interval_width", "mean_interval_width")
        )
        calibration_count = _integer_value(
            _lookup(
                row, "calibration_count", "calibration_sample_count", "calibration_n"
            )
        )
        calibration_status = _normalized_string(_lookup(row, "calibration_status"))
        if coverage is not None:
            coverage_values.append(coverage)
        if width is not None:
            width_values.append(width)
        if calibration_count is not None:
            calibration_counts.append(calibration_count)
        if calibration_status:
            calibration_statuses.append(calibration_status)
        if _is_thin_probabilistic_attachment(
            row,
            coverage=coverage,
            width=width,
            calibration_count=calibration_count,
            minimum_calibration_count=minimum_calibration_count,
        ):
            task_id = _string_or_none(_lookup(row, "task_id"))
            if task_id is not None:
                thin_attachment_task_ids.append(task_id)

    numerator = len(thin_attachment_task_ids)
    denominator = len(rows)
    observed_value = _rate(numerator, denominator)
    if not denominator:
        status = "missing"
        reason = "no_probabilistic_attachment_rows"
    elif numerator:
        status = "failed"
        reason = "thin_probabilistic_attachment"
    else:
        status = "passed"
        reason = None
    return EfficacyMetricResult(
        metric_id=THIN_EVIDENCE_PROBABILISTIC_ATTACHMENT_RATE,
        status=status,
        observed_value=observed_value,
        unit="fraction",
        numerator=numerator,
        denominator=denominator,
        reason=reason,
        provenance=_provenance(rows),
        details={
            "coverage": _mean_or_none(coverage_values),
            "width": _mean_or_none(width_values),
            "calibration_count": sum(calibration_counts),
            "calibration_status": _rollup_status(calibration_statuses),
            "minimum_calibration_count": minimum_calibration_count,
            "status": "passed" if denominator and not numerator else status,
            "thin_attachment_count": numerator,
            "thin_attachment_task_ids": thin_attachment_task_ids,
        },
        legacy_fields={
            "metric_id": THIN_EVIDENCE_PROBABILISTIC_ATTACHMENT_RATE,
            "observed_value": observed_value,
            "coverage": _mean_or_none(coverage_values),
            "mean_interval_width": _mean_or_none(width_values),
            "calibration_count": sum(calibration_counts),
            "calibration_status": _rollup_status(calibration_statuses),
            "status": "passed" if denominator and not numerator else status,
            "provenance": _legacy_provenance(rows),
        },
    )


def compute_nonstationary_detection_placeholders(
    records: Sequence[Mapping[str, Any]],
) -> tuple[EfficacyMetricResult, ...]:
    rows = _materialize_records(records)
    if _has_change_point_lane_evidence(rows):
        return compute_change_point_detection_metrics(rows)
    return tuple(
        _nonstationary_detection_placeholder(metric_id, rows)
        for metric_id in _NONSTATIONARY_DETECTION_METRIC_IDS
    )


def _compute_nonstationary_detection_metric(
    metric_id: str,
    records: Sequence[Mapping[str, Any]],
) -> EfficacyMetricResult:
    rows = _materialize_records(records)
    if _has_change_point_lane_evidence(rows):
        return _compute_change_point_detection_metric(metric_id, rows)
    for result in compute_nonstationary_detection_placeholders(rows):
        if result.metric_id == metric_id:
            return result
    raise ValueError(f"unknown nonstationary detection metric id: {metric_id}")


def compute_change_point_detection_metrics(
    records: Sequence[Mapping[str, Any]],
) -> tuple[EfficacyMetricResult, ...]:
    rows = _materialize_records(records)
    return tuple(
        _compute_change_point_detection_metric(metric_id, rows)
        for metric_id in _NONSTATIONARY_DETECTION_METRIC_IDS
    )


def compute_nonstationary_detection_tolerance_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = _materialize_records(records)
    precision_numerator = 0
    precision_denominator = 0
    recall_numerator = 0
    recall_denominator = 0
    detection_tolerance_steps = None

    for row in rows:
        truth_change_points = tuple(
            item
            for item in _lookup(row, "truth_change_points", "true_change_points") or ()
            if isinstance(item, int) and not isinstance(item, bool)
        )
        detected_change_points = tuple(
            item
            for item in _lookup(row, "detected_change_points") or ()
            if isinstance(item, int) and not isinstance(item, bool)
        )
        tolerance = _integer_value(_lookup(row, "detection_tolerance_steps"))
        if tolerance is None:
            tolerance = 0
        detection_tolerance_steps = tolerance
        matched_truth = _matched_truth_indices(
            truth_change_points=truth_change_points,
            detected_change_points=detected_change_points,
            tolerance=tolerance,
        )
        precision_numerator += len(matched_truth)
        precision_denominator += len(detected_change_points)
        recall_numerator += len(matched_truth)
        recall_denominator += len(truth_change_points)

    return {
        "metric_id": NONSTATIONARY_DETECTION_TOLERANCE,
        "status": "missing_lane",
        "reason": "nonstationary_lane_missing_until_phase6",
        "detection_tolerance_steps": detection_tolerance_steps,
        "precision_numerator": precision_numerator,
        "precision_denominator": precision_denominator,
        "recall_numerator": recall_numerator,
        "recall_denominator": recall_denominator,
        "provenance": _legacy_provenance(rows),
    }


def _compute_change_point_detection_metric(
    metric_id: str,
    records: Sequence[Mapping[str, Any]],
) -> EfficacyMetricResult:
    summary = _change_point_detection_summary(records)
    lane_rows = summary["lane_rows"]
    if not lane_rows:
        return _nonstationary_detection_placeholder(metric_id, records)

    if metric_id == NONSTATIONARY_DETECTION_PRECISION:
        numerator = summary["matched_count"]
        denominator = summary["detected_count"]
        observed_value = _rate(numerator, denominator)
        reason = None if denominator else "no_detected_change_points"
    elif metric_id == NONSTATIONARY_DETECTION_RECALL:
        numerator = summary["matched_count"]
        denominator = summary["truth_count"]
        observed_value = _rate(numerator, denominator)
        reason = None if denominator else "no_truth_change_points"
    elif metric_id == NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE:
        observed_value = (
            max(summary["hausdorff_distances"])
            if summary["hausdorff_distances"]
            else None
        )
        numerator = observed_value
        denominator = len(summary["hausdorff_distances"])
        reason = None if denominator else "hausdorff_distance_not_defined"
    elif metric_id == NONSTATIONARY_DETECTION_DELAY:
        numerator = summary["total_absolute_delay"]
        denominator = summary["matched_count"]
        observed_value = _rate(numerator, denominator)
        reason = None if denominator else "no_matched_change_points_within_tolerance"
    else:
        raise ValueError(f"unknown nonstationary detection metric id: {metric_id}")

    return EfficacyMetricResult(
        metric_id=metric_id,
        status="measured" if observed_value is not None else "missing",
        observed_value=observed_value,
        unit=_unit_for_nonstationary_metric(metric_id),
        numerator=numerator,
        denominator=denominator,
        reason=reason,
        provenance=_provenance(lane_rows),
        details={
            "tolerance": summary["tolerance"],
            "matched_count": summary["matched_count"],
            "truth_count": summary["truth_count"],
            "detected_count": summary["detected_count"],
            "lane_row_count": len(lane_rows),
        },
    )


def _nonstationary_detection_placeholder(
    metric_id: str,
    records: Sequence[Mapping[str, Any]],
) -> EfficacyMetricResult:
    return EfficacyMetricResult(
        metric_id=metric_id,
        status="missing",
        observed_value=None,
        unit=_unit_for_nonstationary_metric(metric_id),
        numerator=None,
        denominator=len(records),
        reason="nonstationary_lane_missing_until_phase6",
        provenance=_provenance(records),
        details={
            "lane_status": "missing",
            "phase_dependency": "phase6",
            "tolerance_status": "not_evaluated",
            "detection_tolerance_steps": _first_integer(
                records,
                "detection_tolerance_steps",
            ),
        },
    )


def _provenance(records: Sequence[Mapping[str, Any]]) -> EfficacyMetricProvenance:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        rows.append(
            {
                "task_id": _string_or_none(_lookup(record, "task_id")),
                "submitter_id": _string_or_none(
                    _lookup(record, "submitter_id", "local_winner_submitter_id")
                ),
                "candidate_id": _string_or_none(
                    _lookup(
                        record,
                        "candidate_id",
                        "selected_candidate_id",
                        "local_winner_candidate_id",
                    )
                ),
                "replay_id": _string_or_none(
                    _lookup(
                        record,
                        "replay_id",
                        "replay_ref",
                        "replay_contract.replay_id",
                        "replay_contract.search_plan_id",
                    )
                ),
                "row_count": _lookup(record, "row_count"),
                "row_index": index,
            }
        )
    return EfficacyMetricProvenance(row_count=len(rows), rows=tuple(rows))


def _legacy_provenance(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    provenance: list[dict[str, Any]] = []
    for record in records:
        provenance.append(
            {
                "task_id": _string_or_none(_lookup(record, "task_id")),
                "submitter_id": _string_or_none(
                    _lookup(record, "submitter_id", "local_winner_submitter_id")
                ),
                "candidate_id": _string_or_none(
                    _lookup(
                        record,
                        "candidate_id",
                        "selected_candidate_id",
                        "local_winner_candidate_id",
                    )
                ),
                "replay_id": _string_or_none(
                    _lookup(
                        record,
                        "replay_id",
                        "replay_ref",
                        "replay_contract.replay_id",
                        "replay_contract.search_plan_id",
                    )
                ),
                "row_count": _lookup(record, "row_count"),
            }
        )
    return provenance


def _materialize_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(records)


def _lookup(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        current: Any = record
        found = True
        for part in key.split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                found = False
                break
        if found:
            return current
    return None


def _recovery_class(row: Mapping[str, Any]) -> str:
    if _bool_value(_lookup(row, "exact_recovery")) is True:
        return "exact"
    if _bool_value(_lookup(row, "near_recovery")) is True:
        return "near"
    status = _normalized_string(
        _lookup(
            row,
            "recovery_status",
            "observed_recovery_class",
            "structural_recovery_status",
            "structural_recovery_class",
        )
    )
    if status in _EXACT_RECOVERY_STATUSES:
        return "exact"
    if status in _NEAR_RECOVERY_STATUSES:
        return "near"
    return "missed"


def _is_adversarial_row(row: Mapping[str, Any]) -> bool:
    if _normalized_string(_lookup(row, "track_id")) == "adversarial_honesty":
        return True
    if _normalized_string(_lookup(row, "expected_safe_outcome")) == "abstain":
        return True
    expectations = _lookup(row, "false_claim_expectations")
    if isinstance(expectations, str):
        return expectations == "no_false_publication"
    if isinstance(expectations, Sequence) and not isinstance(
        expectations, (str, bytes)
    ):
        return "no_false_publication" in expectations
    return False


def _is_false_holistic_claim(row: Mapping[str, Any]) -> bool:
    explicit = _bool_value(
        _lookup(row, "false_holistic_claim", "holistic_claim_false_positive")
    )
    if explicit is not None:
        return explicit
    holistic_claim = _bool_value(
        _lookup(row, "holistic_claim_published", "emitted_holistic_claim")
    )
    if holistic_claim is True:
        return True
    claim_scope = _normalized_string(_lookup(row, "claim_scope", "claim_ceiling"))
    if claim_scope in {
        "scientific_truth",
        "claim_evidence",
        "publication_claim",
        "holistic_law",
        "holistic_law_claim",
    }:
        return True
    return (
        _normalized_string(_lookup(row, "expected_safe_outcome")) == "abstain"
        and _lookup(
            row,
            "candidate_id",
            "selected_candidate_id",
            "local_winner_candidate_id",
            "local_winner_submitter_id",
        )
        is not None
    )


def _is_non_abstained(row: Mapping[str, Any]) -> bool:
    explicit = _bool_value(_lookup(row, "non_abstained"))
    if explicit is not None:
        return explicit
    status = _normalized_string(_lookup(row, "status", "task_status"))
    if status in {"abstained", "safe_abstention"}:
        return False
    if _lookup(row, "abstention_reason") is not None:
        return False
    if (
        _lookup(
            row, "candidate_id", "selected_candidate_id", "local_winner_candidate_id"
        )
        is not None
    ):
        return True
    return status in {"completed", "passed", "selected", "winner"}


def _is_probabilistic_row(row: Mapping[str, Any]) -> bool:
    forecast_object_type = _normalized_string(_lookup(row, "forecast_object_type"))
    if forecast_object_type in _PROBABILISTIC_FORECAST_OBJECT_TYPES:
        return True
    return any(
        key in row
        for key in (
            "coverage",
            "coverage_observed",
            "interval_coverage",
            "probabilistic_attachment_status",
        )
    )


def _is_thin_probabilistic_attachment(
    row: Mapping[str, Any],
    *,
    coverage: float | None,
    width: float | None,
    calibration_count: int | None,
    minimum_calibration_count: int,
) -> bool:
    explicit = _bool_value(_lookup(row, "thin_evidence", "thin_attachment"))
    if explicit is not None:
        return explicit
    if _normalized_string(_lookup(row, "evidence_strength")) == "thin":
        return True
    if _normalized_string(_lookup(row, "calibration_status")) == "failed":
        return True
    attachment_status = _normalized_string(
        _lookup(row, "probabilistic_attachment_status", "attachment_status")
    )
    retained = attachment_status in _RETAINED_PROBABILISTIC_ATTACHMENT_STATUSES
    if (
        not retained
        and _bool_value(
            _lookup(row, "probabilistic_attachment_retained", "attachment_retained")
        )
        is True
    ):
        retained = True
    if not retained:
        return False
    return (
        coverage is None
        or width is None
        or calibration_count is None
        or calibration_count < minimum_calibration_count
    )


def _rate(numerator: int | float, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _first_integer(records: Sequence[Mapping[str, Any]], key: str) -> int | None:
    for record in records:
        value = _integer_value(_lookup(record, key))
        if value is not None:
            return value
    return None


def _integer_sequence(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    integers: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        integers.append(item)
    return tuple(integers)


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _normalized_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _unit_for_nonstationary_metric(metric_id: str) -> str:
    if metric_id in {NONSTATIONARY_DETECTION_PRECISION, NONSTATIONARY_DETECTION_RECALL}:
        return "fraction"
    if metric_id == NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE:
        return "samples"
    if metric_id == NONSTATIONARY_DETECTION_DELAY:
        return "samples"
    return "unknown"


def _has_change_point_lane_evidence(records: Sequence[Mapping[str, Any]]) -> bool:
    return any(_change_point_artifact(row) is not None for row in records)


def _change_point_detection_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lane_rows: list[Mapping[str, Any]] = []
    tolerances: list[int] = []
    matched_count = 0
    truth_count = 0
    detected_count = 0
    total_absolute_delay = 0.0
    hausdorff_distances: list[float] = []

    for row in records:
        artifact = _change_point_artifact(row)
        if artifact is None:
            continue
        lane_rows.append(row)
        truth = _integer_sequence(
            _lookup(row, "truth_change_points", "true_change_points")
        )
        detected = _integer_sequence(
            _lookup(
                row,
                "detected_change_points",
                "change_point_artifact.breakpoints",
                "change_point_artifact.detected_change_points",
            )
        )
        tolerance = _integer_value(
            _lookup(
                row,
                "detection_tolerance_steps",
                "change_point_artifact.tolerance",
                "tolerance",
            )
        )
        if tolerance is None:
            tolerance = 0
        tolerances.append(tolerance)
        matches = _matched_change_point_pairs(
            truth_change_points=truth,
            detected_change_points=detected,
            tolerance=tolerance,
        )
        matched_count += len(matches)
        truth_count += len(truth)
        detected_count += len(detected)
        total_absolute_delay += sum(
            abs(detected - truth) for truth, detected in matches
        )
        hausdorff = _change_point_hausdorff_distance(truth, detected)
        if hausdorff is not None:
            hausdorff_distances.append(hausdorff)

    return {
        "lane_rows": tuple(lane_rows),
        "tolerance": _summary_tolerance(tolerances),
        "matched_count": matched_count,
        "truth_count": truth_count,
        "detected_count": detected_count,
        "total_absolute_delay": total_absolute_delay,
        "hausdorff_distances": tuple(hausdorff_distances),
    }


def _change_point_artifact(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    artifact = _lookup(row, "change_point_artifact", "changepoint_artifact")
    if hasattr(artifact, "as_manifest"):
        artifact = artifact.as_manifest()
    if not isinstance(artifact, Mapping):
        return None
    if _normalized_string(artifact.get("status")) != "passed":
        return None
    return artifact


def _summary_tolerance(tolerances: Sequence[int]) -> int | list[int] | None:
    if not tolerances:
        return None
    unique = tuple(dict.fromkeys(tolerances))
    if len(unique) == 1:
        return unique[0]
    return list(unique)


def _change_point_hausdorff_distance(
    truth_change_points: Sequence[int],
    detected_change_points: Sequence[int],
) -> float | None:
    if not truth_change_points and not detected_change_points:
        return 0.0
    if not truth_change_points or not detected_change_points:
        return None
    truth_to_detected = _directed_change_point_distance(
        truth_change_points, detected_change_points
    )
    detected_to_truth = _directed_change_point_distance(
        detected_change_points, truth_change_points
    )
    return float(max(truth_to_detected, detected_to_truth))


def _directed_change_point_distance(
    source: Sequence[int],
    target: Sequence[int],
) -> int:
    return max(min(abs(item - other) for other in target) for item in source)


def _matched_change_point_pairs(
    *,
    truth_change_points: Sequence[int],
    detected_change_points: Sequence[int],
    tolerance: int,
) -> tuple[tuple[int, int], ...]:
    unmatched_truth = set(range(len(truth_change_points)))
    matches: list[tuple[int, int]] = []
    for detected in detected_change_points:
        best_index = None
        best_distance = None
        for index in sorted(unmatched_truth):
            distance = abs(detected - truth_change_points[index])
            if distance <= tolerance and (
                best_distance is None or distance < best_distance
            ):
                best_index = index
                best_distance = distance
        if best_index is not None:
            unmatched_truth.remove(best_index)
            matches.append((truth_change_points[best_index], detected))
    return tuple(matches)


def _matched_truth_indices(
    *,
    truth_change_points: Sequence[int],
    detected_change_points: Sequence[int],
    tolerance: int,
) -> set[int]:
    unmatched_truth = set(range(len(truth_change_points)))
    matched_truth: set[int] = set()
    for detected in detected_change_points:
        best_index = None
        best_distance = None
        for index in sorted(unmatched_truth):
            distance = abs(detected - truth_change_points[index])
            if distance <= tolerance and (
                best_distance is None or distance < best_distance
            ):
                best_index = index
                best_distance = distance
        if best_index is not None:
            unmatched_truth.remove(best_index)
            matched_truth.add(best_index)
    return matched_truth


def _rollup_status(statuses: Sequence[str]) -> str | None:
    if not statuses:
        return None
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "missing" for status in statuses):
        return "missing"
    if all(status == "passed" for status in statuses):
        return "passed"
    return statuses[0]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


_REGISTRY: dict[str, EfficacyMetricSpec] = {
    DESCRIPTIVE_NON_ABSTENTION_RATE: EfficacyMetricSpec(
        metric_id=DESCRIPTIVE_NON_ABSTENTION_RATE,
        description="Fraction of descriptive benchmark rows with a selected candidate.",
        unit="fraction",
        compute=compute_descriptive_non_abstention_rate,
    ),
    FALSE_HOLISTIC_RATE: EfficacyMetricSpec(
        metric_id=FALSE_HOLISTIC_RATE,
        description="False holistic claim rate over adversarial honesty rows.",
        unit="fraction",
        compute=compute_false_holistic_claim_rate,
    ),
    PLANTED_LAW_RECOVERY_RATE: EfficacyMetricSpec(
        metric_id=PLANTED_LAW_RECOVERY_RATE,
        description="Exact or near planted-law recovery rate.",
        unit="fraction",
        compute=compute_planted_law_recovery_rate,
    ),
    THIN_EVIDENCE_PROBABILISTIC_ATTACHMENT_RATE: EfficacyMetricSpec(
        metric_id=THIN_EVIDENCE_PROBABILISTIC_ATTACHMENT_RATE,
        description="Rate of retained probabilistic attachments with thin evidence.",
        unit="fraction",
        compute=compute_probabilistic_attachment_quality,
    ),
    NONSTATIONARY_DETECTION_PRECISION: EfficacyMetricSpec(
        metric_id=NONSTATIONARY_DETECTION_PRECISION,
        description="Placeholder precision for nonstationary detection lanes.",
        unit="fraction",
        compute=lambda records, **_: _compute_nonstationary_detection_metric(
            NONSTATIONARY_DETECTION_PRECISION, records
        ),
    ),
    NONSTATIONARY_DETECTION_RECALL: EfficacyMetricSpec(
        metric_id=NONSTATIONARY_DETECTION_RECALL,
        description="Placeholder recall for nonstationary detection lanes.",
        unit="fraction",
        compute=lambda records, **_: _compute_nonstationary_detection_metric(
            NONSTATIONARY_DETECTION_RECALL, records
        ),
    ),
    NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE: EfficacyMetricSpec(
        metric_id=NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE,
        description="Placeholder Hausdorff distance for nonstationary detection lanes.",
        unit="samples",
        compute=lambda records, **_: _compute_nonstationary_detection_metric(
            NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE, records
        ),
    ),
    NONSTATIONARY_DETECTION_DELAY: EfficacyMetricSpec(
        metric_id=NONSTATIONARY_DETECTION_DELAY,
        description="Placeholder detection delay for nonstationary detection lanes.",
        unit="samples",
        compute=lambda records, **_: _compute_nonstationary_detection_metric(
            NONSTATIONARY_DETECTION_DELAY, records
        ),
    ),
}


__all__ = [
    "DESCRIPTIVE_NON_ABSTENTION_RATE",
    "FALSE_HOLISTIC_RATE",
    "NONSTATIONARY_DETECTION_DELAY",
    "NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE",
    "NONSTATIONARY_DETECTION_PRECISION",
    "NONSTATIONARY_DETECTION_RECALL",
    "NONSTATIONARY_DETECTION_TOLERANCE",
    "PLANTED_LAW_RECOVERY_RATE",
    "THIN_EVIDENCE_PROBABILISTIC_ATTACHMENT_RATE",
    "EfficacyMetricProvenance",
    "EfficacyMetricResult",
    "EfficacyMetricSpec",
    "compute_change_point_detection_metrics",
    "compute_descriptive_non_abstention_rate",
    "compute_efficacy_metric",
    "compute_false_holistic_claim_rate",
    "compute_nonstationary_detection_placeholders",
    "compute_nonstationary_detection_tolerance_metrics",
    "compute_planted_law_recovery_metrics",
    "compute_planted_law_recovery_rate",
    "compute_probabilistic_attachment_quality",
    "compute_registered_metrics",
    "efficacy_metric_registry",
]
