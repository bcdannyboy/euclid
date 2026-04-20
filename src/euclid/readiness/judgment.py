from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

_ALLOWED_GATE_STATUSES = {"passed", "failed", "missing", "not_applicable"}
_VERDICT_PRIORITY = {
    "ready": 0,
    "review_required": 1,
    "blocked": 2,
}


@dataclass(frozen=True)
class ReadinessGateResult:
    gate_id: str
    status: str
    required: bool = True
    summary: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _ALLOWED_GATE_STATUSES:
            raise ValueError(f"unsupported readiness gate status: {self.status!r}")


@dataclass(frozen=True)
class ReadinessJudgment:
    judgment_id: str
    final_verdict: str
    catalog_scope: str
    verdict_summary: str
    reason_codes: tuple[str, ...]
    required_gate_count: int
    passed_gate_count: int
    failed_gate_count: int
    missing_gate_count: int
    gate_results: tuple[ReadinessGateResult, ...]


def judge_readiness(
    *,
    judgment_id: str,
    gate_results: Sequence[ReadinessGateResult],
    required_gate_ids: Sequence[str] | None = None,
) -> ReadinessJudgment:
    gate_by_id = {gate.gate_id: gate for gate in gate_results}
    augmented: list[ReadinessGateResult] = list(gate_results)
    for gate_id in tuple(required_gate_ids or ()):
        if gate_id in gate_by_id:
            continue
        gate = ReadinessGateResult(
            gate_id=gate_id,
            status="missing",
            required=True,
            summary=f"{gate_id} evidence has not been captured yet.",
            evidence={},
        )
        gate_by_id[gate_id] = gate
        augmented.append(gate)

    ordered_gates = tuple(
        sorted(
            augmented,
            key=lambda gate: (not gate.required, gate.gate_id),
        )
    )
    required_gates = tuple(gate for gate in ordered_gates if gate.required)
    failed_required = tuple(gate for gate in required_gates if gate.status == "failed")
    missing_required = tuple(
        gate for gate in required_gates if gate.status == "missing"
    )
    review_only_gates = tuple(
        gate
        for gate in ordered_gates
        if not gate.required and gate.status in {"failed", "missing"}
    )

    if failed_required or missing_required:
        final_verdict = "blocked"
    elif review_only_gates:
        final_verdict = "review_required"
    else:
        final_verdict = "ready"

    reason_codes = tuple(
        f"{gate.gate_id}_{gate.status}"
        for gate in ordered_gates
        if gate.status in {"failed", "missing"}
        and (gate.required or final_verdict == "review_required")
    )
    if final_verdict == "ready":
        verdict_summary = "All required readiness gates passed."
    elif final_verdict == "review_required":
        verdict_summary = (
            "Required readiness gates passed, but review-only gates still need "
            "attention."
        )
    else:
        verdict_summary = "Required readiness gates are not satisfied."

    return ReadinessJudgment(
        judgment_id=judgment_id,
        final_verdict=final_verdict,
        catalog_scope="public" if final_verdict == "ready" else "internal",
        verdict_summary=verdict_summary,
        reason_codes=reason_codes,
        required_gate_count=len(required_gates),
        passed_gate_count=sum(1 for gate in ordered_gates if gate.status == "passed"),
        failed_gate_count=sum(1 for gate in ordered_gates if gate.status == "failed"),
        missing_gate_count=sum(1 for gate in ordered_gates if gate.status == "missing"),
        gate_results=ordered_gates,
    )


def merge_readiness_judgments(
    *,
    judgment_id: str,
    judgments: Sequence[ReadinessJudgment],
) -> ReadinessJudgment:
    if not judgments:
        raise ValueError("merge_readiness_judgments requires at least one judgment")
    merged_gates: list[ReadinessGateResult] = []
    for judgment in judgments:
        merged_gates.extend(judgment.gate_results)

    merged = judge_readiness(judgment_id=judgment_id, gate_results=merged_gates)
    strongest = max(judgments, key=lambda item: _VERDICT_PRIORITY[item.final_verdict])
    if (
        _VERDICT_PRIORITY[strongest.final_verdict]
        > _VERDICT_PRIORITY[merged.final_verdict]
    ):
        return ReadinessJudgment(
            judgment_id=merged.judgment_id,
            final_verdict=strongest.final_verdict,
            catalog_scope=strongest.catalog_scope,
            verdict_summary=strongest.verdict_summary,
            reason_codes=strongest.reason_codes,
            required_gate_count=merged.required_gate_count,
            passed_gate_count=merged.passed_gate_count,
            failed_gate_count=merged.failed_gate_count,
            missing_gate_count=merged.missing_gate_count,
            gate_results=merged.gate_results,
        )
    return merged


def gate_results_by_id(
    gate_results: Sequence[ReadinessGateResult],
) -> dict[str, ReadinessGateResult]:
    return {gate.gate_id: gate for gate in gate_results}


def judge_benchmark_suite_readiness(
    *,
    judgment_id: str,
    suite_result,
    supplemental_gate_results: Sequence[ReadinessGateResult] = (),
) -> ReadinessJudgment:
    summary_payload = _load_suite_summary_payload(suite_result.summary_path)
    missing_task_semantic_fields = _missing_task_semantic_fields(
        suite_result=suite_result,
        summary_payload=summary_payload,
    )
    suite_gate = ReadinessGateResult(
        gate_id=f"suite.{suite_result.suite_manifest.suite_id}",
        status=(
            "passed"
            if suite_result.summary_path.is_file()
            and len(suite_result.task_results)
            == len(suite_result.suite_manifest.task_manifest_paths)
            and not missing_task_semantic_fields
            else "failed"
        ),
        required=True,
        summary=(
            "Benchmark suite summary exists, every declared task executed, and "
            "semantic task evidence is captured."
        ),
        evidence={
            "suite_id": suite_result.suite_manifest.suite_id,
            "summary_path": str(suite_result.summary_path),
            "task_count": len(suite_result.task_results),
            "missing_task_semantic_fields": missing_task_semantic_fields,
        },
    )
    track_gates = tuple(
        ReadinessGateResult(
            gate_id=f"track.{track_id}",
            status=_track_gate_status(suite_result=suite_result, track_id=track_id),
            required=True,
            summary=f"Certified suite captured required track {track_id}.",
            evidence={
                "track_id": track_id,
                "task_ids": [
                    result.task_manifest.task_id
                    for result in suite_result.task_results
                    if result.task_manifest.track_id == track_id
                ],
            },
        )
        for track_id in suite_result.suite_manifest.required_tracks
    )
    surface_gates = tuple(
        _surface_gate(summary_payload=summary_payload, surface=surface)
        for surface in suite_result.surface_statuses
    )
    return judge_readiness(
        judgment_id=judgment_id,
        gate_results=(
            suite_gate,
            *track_gates,
            *surface_gates,
            *tuple(supplemental_gate_results),
        ),
    )


def _track_gate_status(*, suite_result, track_id: str) -> str:
    matching_results = tuple(
        result
        for result in suite_result.task_results
        if result.task_manifest.track_id == track_id
    )
    if not matching_results:
        return "failed"
    if all(
        result.report_paths.task_result_path.is_file() for result in matching_results
    ):
        return "passed"
    return "failed"


def _surface_gate(
    *,
    summary_payload: Mapping[str, Any] | None,
    surface,
) -> ReadinessGateResult:
    summary_surface = _summary_surface_row(
        summary_payload=summary_payload,
        surface_id=surface.surface_id,
    )
    summary_evidence = (
        dict(summary_surface.get("evidence", {}))
        if isinstance(summary_surface, Mapping)
        and isinstance(summary_surface.get("evidence"), Mapping)
        else {}
    )
    missing_surface_diagnostics = _missing_surface_diagnostics(summary_evidence)
    return ReadinessGateResult(
        gate_id=f"surface.{surface.surface_id}",
        status=(
            "passed"
            if surface.benchmark_status == "passed"
            and surface.replay_status == "passed"
            and not missing_surface_diagnostics
            else "failed"
        ),
        required=True,
        summary=(
            f"Certified suite captured benchmark, replay, and semantic evidence "
            f"for {surface.surface_id}."
        ),
        evidence={
            **summary_evidence,
            "missing_surface_diagnostics": missing_surface_diagnostics,
        },
    )


def _load_suite_summary_payload(summary_path: Path) -> Mapping[str, Any] | None:
    if not summary_path.is_file():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _missing_task_semantic_fields(
    *,
    suite_result,
    summary_payload: Mapping[str, Any] | None,
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    summary_rows = {
        str(row["task_id"]): row
        for row in summary_payload.get("task_results", ())
        if isinstance(row, Mapping) and isinstance(row.get("task_id"), str)
    } if isinstance(summary_payload, Mapping) else {}
    required_fields = (
        "forecast_object_type",
        "score_law",
        "calibration_verdict",
        "abstention_mode",
        "replay_verification",
    )
    for result in suite_result.task_results:
        row = summary_rows.get(result.task_manifest.task_id, {})
        if not isinstance(row, Mapping):
            row = {}
        missing_fields = [
            field_name
            for field_name in required_fields
            if not _has_semantic_value(row.get(field_name))
        ]
        if missing_fields:
            missing[result.task_manifest.task_id] = missing_fields
    return missing


def _summary_surface_row(
    *,
    summary_payload: Mapping[str, Any] | None,
    surface_id: str,
) -> Mapping[str, Any] | None:
    if not isinstance(summary_payload, Mapping):
        return None
    for row in summary_payload.get("surface_statuses", ()):
        if isinstance(row, Mapping) and row.get("surface_id") == surface_id:
            return row
    return None


def _missing_surface_diagnostics(summary_evidence: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for field_name in (
        "forecast_object_types",
        "score_laws",
        "calibration_verdicts",
        "abstention_modes",
        "replay_verification",
    ):
        if not _has_semantic_value(summary_evidence.get(field_name)):
            missing.append(field_name)
    return missing


def _has_semantic_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) > 0
    return True


__all__ = [
    "ReadinessGateResult",
    "ReadinessJudgment",
    "gate_results_by_id",
    "judge_benchmark_suite_readiness",
    "judge_readiness",
    "merge_readiness_judgments",
]
