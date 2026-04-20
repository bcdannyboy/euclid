from __future__ import annotations

from euclid.readiness import ReadinessGateResult, judge_readiness


def _gate(
    gate_id: str,
    *,
    status: str,
    required: bool = True,
    summary: str | None = None,
) -> ReadinessGateResult:
    return ReadinessGateResult(
        gate_id=gate_id,
        status=status,
        required=required,
        summary=summary or f"{gate_id} => {status}",
        evidence={"gate_id": gate_id},
    )


def test_judge_readiness_returns_public_ready_when_all_required_gates_pass() -> None:
    judgment = judge_readiness(
        judgment_id="release_v1_readiness",
        gate_results=(
            _gate("contracts.catalog", status="passed"),
            _gate("benchmarks.rediscovery", status="passed"),
            _gate("benchmarks.predictive_generalization", status="passed"),
            _gate("benchmarks.adversarial_honesty", status="passed"),
            _gate("notebook.smoke", status="passed"),
            _gate("determinism.same_seed", status="passed"),
            _gate("performance.runtime_smoke", status="passed"),
        ),
    )

    assert judgment.final_verdict == "ready"
    assert judgment.catalog_scope == "public"
    assert judgment.reason_codes == ()
    assert judgment.required_gate_count == 7
    assert judgment.passed_gate_count == 7
    assert judgment.failed_gate_count == 0
    assert judgment.missing_gate_count == 0


def test_judge_readiness_blocks_when_required_benchmark_gate_fails() -> None:
    judgment = judge_readiness(
        judgment_id="release_v1_readiness",
        gate_results=(
            _gate("contracts.catalog", status="passed"),
            _gate("benchmarks.rediscovery", status="passed"),
            _gate("benchmarks.predictive_generalization", status="failed"),
            _gate("benchmarks.adversarial_honesty", status="passed"),
            _gate("notebook.smoke", status="passed"),
        ),
    )

    assert judgment.final_verdict == "blocked"
    assert judgment.catalog_scope == "internal"
    assert judgment.reason_codes == ("benchmarks.predictive_generalization_failed",)
    assert judgment.failed_gate_count == 1
