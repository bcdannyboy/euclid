from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from euclid.benchmarks import load_benchmark_task_manifest
from euclid.readiness import ReadinessGateResult, judge_benchmark_suite_readiness

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "src/euclid/_assets"
CURRENT_RELEASE_SUITE = ASSET_ROOT / "benchmarks/suites/current-release.yaml"
PROFILE_PATH = ASSET_ROOT / "schemas/readiness/benchmark-threshold-gates-v1.yaml"


@pytest.fixture(scope="module")
def current_release_suite_result(tmp_path_factory: pytest.TempPathFactory):
    workspace = tmp_path_factory.mktemp("phase08-release-gate-current-release")
    suite_payload = _load_yaml(CURRENT_RELEASE_SUITE)
    task_manifests = tuple(
        load_benchmark_task_manifest(ASSET_ROOT / relative_path)
        for relative_path in suite_payload["task_manifest_paths"]
    )
    task_results = []
    task_summary_rows = []
    for task_manifest in task_manifests:
        task_result_path = workspace / f"{task_manifest.task_id}.json"
        task_result_path.write_text("{}", encoding="utf-8")
        task_results.append(
            SimpleNamespace(
                task_manifest=SimpleNamespace(
                    task_id=task_manifest.task_id,
                    track_id=task_manifest.track_id,
                ),
                report_paths=SimpleNamespace(task_result_path=task_result_path),
            )
        )
        task_summary_rows.append(
            {
                "task_id": task_manifest.task_id,
                "track_id": task_manifest.track_id,
                "forecast_object_type": "point",
                "score_law": "mean_absolute_error",
                "calibration_verdict": "not_applicable",
                "abstention_mode": "structural_miss",
                "replay_verification": "verified",
            }
        )

    surface_statuses = tuple(
        SimpleNamespace(
            surface_id=surface_requirement["surface_id"],
            benchmark_status="passed",
            replay_status="passed",
        )
        for surface_requirement in suite_payload["surface_requirements"]
    )
    summary_path = workspace / "benchmark-suite-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "suite_id": suite_payload["suite_id"],
                "task_results": task_summary_rows,
                "surface_statuses": [
                    {
                        "surface_id": surface_requirement["surface_id"],
                        "evidence": {
                            "forecast_object_types": ["point"],
                            "score_laws": ["mean_absolute_error"],
                            "calibration_verdicts": ["not_applicable"],
                            "abstention_modes": ["structural_miss"],
                            "replay_verification": "verified",
                        },
                    }
                    for surface_requirement in suite_payload["surface_requirements"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return SimpleNamespace(
        suite_manifest=SimpleNamespace(
            suite_id=suite_payload["suite_id"],
            required_tracks=tuple(suite_payload["required_tracks"]),
            task_manifest_paths=tuple(
                ASSET_ROOT / relative_path
                for relative_path in suite_payload["task_manifest_paths"]
            ),
        ),
        task_results=tuple(task_results),
        surface_statuses=surface_statuses,
        summary_path=summary_path,
    )


def test_phase08_threshold_failure_surfaces_as_release_blocker(
    current_release_suite_result,
) -> None:
    profile = _load_yaml(PROFILE_PATH)
    suite_payload = _load_yaml(CURRENT_RELEASE_SUITE)
    judgment = judge_benchmark_suite_readiness(
        judgment_id="phase08_current_release_threshold_failure",
        suite_result=current_release_suite_result,
        supplemental_gate_results=_build_threshold_gate_results(
            profile=profile,
            suite_payload=suite_payload,
            suite_id="current_release",
            observed_metrics={
                "descriptive_non_abstention_rate": 1.0,
                "false_holistic_rate": 0.25,
                "planted_law_recovery_rate": 0.9,
            },
        ),
    )

    gate_by_id = {gate.gate_id: gate for gate in judgment.gate_results}
    false_holistic_gate = gate_by_id["benchmark.false_holistic_rate"]

    assert judgment.final_verdict == "blocked"
    assert "benchmark.false_holistic_rate_failed" in judgment.reason_codes
    assert false_holistic_gate.status == "failed"
    assert false_holistic_gate.evidence["report_as_release_blocker"] is True
    assert false_holistic_gate.evidence["blocker_reason_code"] == (
        "false_holistic_rate_above_threshold"
    )
    assert false_holistic_gate.evidence["required_task_ids"] == ["leakage_trap_demo"]


def test_phase08_threshold_boundaries_preserve_ready_judgment(
    current_release_suite_result,
) -> None:
    profile = _load_yaml(PROFILE_PATH)
    suite_payload = _load_yaml(CURRENT_RELEASE_SUITE)
    judgment = judge_benchmark_suite_readiness(
        judgment_id="phase08_current_release_threshold_boundary",
        suite_result=current_release_suite_result,
        supplemental_gate_results=_build_threshold_gate_results(
            profile=profile,
            suite_payload=suite_payload,
            suite_id="current_release",
            observed_metrics={
                "descriptive_non_abstention_rate": 1.0,
                "false_holistic_rate": 0.0,
                "planted_law_recovery_rate": 0.75,
            },
        ),
    )

    assert judgment.final_verdict == "ready"
    assert not any(
        code.startswith("benchmark.")
        and code.endswith(("_failed", "_missing"))
        for code in judgment.reason_codes
    )


def _build_threshold_gate_results(
    *,
    profile: dict[str, Any],
    suite_payload: dict[str, Any],
    suite_id: str,
    observed_metrics: dict[str, float],
) -> tuple[ReadinessGateResult, ...]:
    gate_by_id = {entry["gate_id"]: entry for entry in profile["gates"]}
    bucket_by_id = suite_payload["release_gate_task_buckets"]
    gate_results: list[ReadinessGateResult] = []
    for gate_id in profile["suites"][suite_id]["required_gate_ids"]:
        gate = gate_by_id[gate_id]
        metric_id = str(gate["metric_id"])
        observed_value = observed_metrics.get(metric_id)
        if observed_value is None:
            status = "missing"
        else:
            status = (
                "passed"
                if _compare(
                    float(observed_value),
                    comparator=str(gate["comparator"]),
                    threshold=float(gate["threshold"]),
                )
                else "failed"
            )
        task_bucket = bucket_by_id[str(gate["task_bucket_id"])]
        gate_results.append(
            ReadinessGateResult(
                gate_id=gate_id,
                status=status,
                required=True,
                summary=str(gate["summary"]),
                evidence={
                    "metric_id": metric_id,
                    "observed_value": observed_value,
                    "comparator": str(gate["comparator"]),
                    "threshold": float(gate["threshold"]),
                    "unit": str(gate["unit"]),
                    "task_bucket_id": str(gate["task_bucket_id"]),
                    "required_task_ids": list(task_bucket["required_task_ids"]),
                    "report_as_release_blocker": bool(
                        gate["report_as_release_blocker"]
                    ),
                    "blocker_reason_code": str(gate["blocker_reason_code"]),
                },
            )
        )
    return tuple(gate_results)


def _compare(observed_value: float, *, comparator: str, threshold: float) -> bool:
    if comparator == ">=":
        return observed_value >= threshold
    if comparator == "<=":
        return observed_value <= threshold
    raise AssertionError(f"unsupported comparator: {comparator}")


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
