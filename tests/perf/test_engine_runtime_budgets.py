from __future__ import annotations

from euclid.expr.ast import Feature, Literal, NaryOp
from euclid.performance import (
    EngineRuntimeBudget,
    build_engine_runtime_budget_report,
    degradation_decision_from_budget_report,
)
from euclid.rewrites.egglog_runner import (
    EqualitySaturationConfig,
    run_equality_saturation,
)


def test_engine_budget_report_marks_timeout_as_abstention_boundary() -> None:
    report = build_engine_runtime_budget_report(
        engine_id="pysr-engine-v1",
        elapsed_seconds=1.5,
        candidate_count=0,
        status="timeout",
        budget=EngineRuntimeBudget(
            budget_id="pysr_tiny_budget",
            timeout_seconds=1.0,
            candidate_limit=4,
            max_iterations=2,
        ),
        reason_codes=("engine_timeout",),
    )

    decision = degradation_decision_from_budget_report(report)

    assert not report.passed
    assert report.as_dict()["resource_limits"]["timeout_seconds"] == 1.0
    assert decision.status == "abstained"
    assert decision.reason_codes == ("engine_timeout",)
    assert decision.claim_publication_allowed is False


def test_egraph_resource_exhaustion_is_reported_as_graceful_degradation() -> None:
    saturation = run_equality_saturation(
        NaryOp("add", (Feature("x"), Literal(0))),
        config=EqualitySaturationConfig(max_iterations=0, node_limit=2),
    )

    report = build_engine_runtime_budget_report(
        engine_id="egraph-engine-v1",
        elapsed_seconds=0.01,
        candidate_count=0,
        status=saturation.status,
        budget=EngineRuntimeBudget(
            budget_id="egraph_saturation_tiny_budget",
            timeout_seconds=1.0,
            candidate_limit=8,
            max_iterations=0,
            node_limit=2,
        ),
        reason_codes=("resource_limited",),
        details={"rewrite_trace": saturation.as_dict()},
    )

    decision = degradation_decision_from_budget_report(report)

    assert saturation.status == "partial"
    assert report.passed
    assert report.as_dict()["details"]["rewrite_trace"]["omission_disclosure"][
        "resource_limited"
    ] is True
    assert decision.status == "degraded"
    assert decision.reason_codes == ("resource_limited",)
