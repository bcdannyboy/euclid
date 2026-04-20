from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from euclid.benchmarks import load_benchmark_task_manifest
from euclid.readiness import ReadinessGateResult, judge_readiness

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "src/euclid/_assets"
CURRENT_RELEASE_SUITE = ASSET_ROOT / "benchmarks/suites/current-release.yaml"
FULL_VISION_SUITE = ASSET_ROOT / "benchmarks/suites/full-vision.yaml"
PHASE08_HOLISTIC_HONESTY_SUITE = (
    ASSET_ROOT / "benchmarks/suites/phase08-holistic-honesty.yaml"
)
PROFILE_PATH = ASSET_ROOT / "schemas/readiness/benchmark-threshold-gates-v1.yaml"


def test_wave3_release_threshold_profile_declares_expected_blocking_metrics() -> None:
    profile = _load_yaml(PROFILE_PATH)
    gates = {entry["gate_id"]: entry for entry in profile["gates"]}

    assert profile["profile_id"] == "benchmark_threshold_gates_v1"
    assert gates == {
        "benchmark.descriptive_non_abstention": {
            "gate_id": "benchmark.descriptive_non_abstention",
            "metric_id": "descriptive_non_abstention_rate",
            "comparator": ">=",
            "threshold": 1.0,
            "unit": "fraction",
            "task_bucket_id": "descriptive_non_abstention",
            "report_as_release_blocker": True,
            "blocker_reason_code": "descriptive_non_abstention_below_threshold",
            "summary": (
                "Descriptive search must emit the best available approximation "
                "whenever fallback search is available."
            ),
        },
        "benchmark.false_holistic_rate": {
            "gate_id": "benchmark.false_holistic_rate",
            "metric_id": "false_holistic_rate",
            "comparator": "<=",
            "threshold": 0.0,
            "unit": "fraction",
            "task_bucket_id": "false_holistic_rate",
            "report_as_release_blocker": True,
            "blocker_reason_code": "false_holistic_rate_above_threshold",
            "summary": (
                "Adversarial and leakage canaries must not produce any false "
                "holistic claim."
            ),
        },
        "benchmark.planted_law_recovery": {
            "gate_id": "benchmark.planted_law_recovery",
            "metric_id": "planted_law_recovery_rate",
            "comparator": ">=",
            "threshold": 0.75,
            "unit": "fraction",
            "task_bucket_id": "planted_law_recovery",
            "report_as_release_blocker": True,
            "blocker_reason_code": "planted_law_recovery_below_threshold",
            "summary": (
                "Planted-law suites must recover the intended compact structure "
                "at or above the declared recovery floor."
            ),
        },
        "benchmark.thin_evidence_probabilistic_attachment": {
            "gate_id": "benchmark.thin_evidence_probabilistic_attachment",
            "metric_id": "thin_evidence_probabilistic_attachment_rate",
            "comparator": "<=",
            "threshold": 0.0,
            "unit": "fraction",
            "task_bucket_id": "thin_evidence_probabilistic_attachment",
            "report_as_release_blocker": True,
            "blocker_reason_code": (
                "thin_evidence_probabilistic_attachment_above_threshold"
            ),
            "summary": (
                "Thin-evidence cases must not retain probabilistic attachment "
                "as a stronger claim."
            ),
        },
    }
    assert profile["suites"] == {
        "current_release": {
            "required_gate_ids": [
                "benchmark.descriptive_non_abstention",
                "benchmark.false_holistic_rate",
                "benchmark.planted_law_recovery",
            ]
        },
        "full_vision": {
            "required_gate_ids": [
                "benchmark.descriptive_non_abstention",
                "benchmark.false_holistic_rate",
                "benchmark.planted_law_recovery",
                "benchmark.thin_evidence_probabilistic_attachment",
            ]
        },
        "phase08_holistic_honesty": {
            "required_gate_ids": [
                "benchmark.descriptive_non_abstention",
                "benchmark.false_holistic_rate",
                "benchmark.planted_law_recovery",
            ]
        },
    }


def test_wave3_suites_declare_release_gate_buckets_with_declared_task_ids() -> None:
    profile = _load_yaml(PROFILE_PATH)
    for suite_path, expected_gate_ids in (
        (
            CURRENT_RELEASE_SUITE,
            tuple(profile["suites"]["current_release"]["required_gate_ids"]),
        ),
        (
            FULL_VISION_SUITE,
            tuple(profile["suites"]["full_vision"]["required_gate_ids"]),
        ),
        (
            PHASE08_HOLISTIC_HONESTY_SUITE,
            tuple(profile["suites"]["phase08_holistic_honesty"]["required_gate_ids"]),
        ),
    ):
        suite_payload = _load_yaml(suite_path)
        suite_id = str(suite_payload["suite_id"])
        declared_task_ids = {
            load_benchmark_task_manifest(ASSET_ROOT / relative_path).task_id
            for relative_path in suite_payload["task_manifest_paths"]
        }
        gate_buckets = suite_payload["release_gate_task_buckets"]
        gate_by_id = {entry["gate_id"]: entry for entry in profile["gates"]}

        assert (
            suite_payload["release_gate_profile_path"]
            == "schemas/readiness/benchmark-threshold-gates-v1.yaml"
        )
        for gate_id in expected_gate_ids:
            bucket_id = gate_by_id[gate_id]["task_bucket_id"]
            assert bucket_id in gate_buckets, f"{suite_id} missing {bucket_id}"
            required_task_ids = set(gate_buckets[bucket_id]["required_task_ids"])
            assert required_task_ids
            assert required_task_ids <= declared_task_ids


def test_wave3_release_threshold_judgment_blocks_on_failed_or_missing_metrics() -> None:
    profile = _load_yaml(PROFILE_PATH)
    suite_payload = _load_yaml(FULL_VISION_SUITE)

    judgment = judge_readiness(
        judgment_id="wave3_thresholds",
        gate_results=_build_threshold_gate_results(
            profile=profile,
            suite_payload=suite_payload,
            suite_id="full_vision",
            observed_metrics={
                "descriptive_non_abstention_rate": 1.0,
                "false_holistic_rate": 0.2,
                "planted_law_recovery_rate": 0.8,
            },
        ),
        required_gate_ids=profile["suites"]["full_vision"]["required_gate_ids"],
    )

    gate_by_id = {gate.gate_id: gate for gate in judgment.gate_results}
    false_holistic_gate = gate_by_id["benchmark.false_holistic_rate"]
    thin_evidence_gate = gate_by_id["benchmark.thin_evidence_probabilistic_attachment"]

    assert judgment.final_verdict == "blocked"
    assert "benchmark.false_holistic_rate_failed" in judgment.reason_codes
    assert (
        "benchmark.thin_evidence_probabilistic_attachment_missing"
        in judgment.reason_codes
    )
    assert false_holistic_gate.evidence["observed_value"] == 0.2
    assert false_holistic_gate.evidence["blocker_reason_code"] == (
        "false_holistic_rate_above_threshold"
    )
    assert thin_evidence_gate.evidence["task_bucket_id"] == (
        "thin_evidence_probabilistic_attachment"
    )
    assert thin_evidence_gate.evidence["required_task_ids"] == [
        "interval_medium_robustness_demo",
        "quantile_medium_misspecification_demo",
        "event_probability_medium_abstention_demo",
    ]


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
