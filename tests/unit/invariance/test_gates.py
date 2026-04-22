from __future__ import annotations

from euclid.invariance.gates import evaluate_invariance, evaluate_transport
from euclid.invariance.environments import construct_environments


def test_invariance_gate_requires_multiple_environments() -> None:
    environments = construct_environments(
        ({"environment": "only", "target": 1.0},),
        policy="explicit_label",
        label_field="environment",
        min_environment_count=1,
    )

    evaluation = evaluate_invariance(
        environments=environments,
        residuals_by_environment={"only": (0.1,)},
        parameters_by_environment={"only": {"a": 1.0}},
        supports_by_environment={"only": {"x"}},
        min_environment_count=2,
    )

    assert evaluation.status == "failed"
    assert "insufficient_environments" in evaluation.reason_codes
    assert evaluation.claim_lane_allowed is False


def test_parameter_drift_blocks_invariant_predictive_law() -> None:
    environments = construct_environments(
        (
            {"environment": "a", "target": 1.0},
            {"environment": "b", "target": 1.0},
        ),
        policy="explicit_label",
        label_field="environment",
    )

    evaluation = evaluate_invariance(
        environments=environments,
        residuals_by_environment={"a": (0.01,), "b": (0.02,)},
        parameters_by_environment={"a": {"slope": 1.0}, "b": {"slope": 3.0}},
        supports_by_environment={"a": {"x"}, "b": {"x"}},
        parameter_drift_threshold=0.2,
    )

    assert evaluation.status == "failed"
    assert "parameter_drift_failed" in evaluation.reason_codes
    assert evaluation.metrics["max_parameter_drift"] == 2.0


def test_passed_invariance_evaluation_has_manifest_replay_evidence() -> None:
    environments = construct_environments(
        (
            {"environment": "a", "target": 1.0},
            {"environment": "b", "target": 1.1},
        ),
        policy="explicit_label",
        label_field="environment",
    )

    evaluation = evaluate_invariance(
        environments=environments,
        residuals_by_environment={"a": (0.01, 0.02), "b": (0.02, 0.01)},
        parameters_by_environment={
            "a": {"slope": 1.0, "intercept": 0.1},
            "b": {"slope": 1.01, "intercept": 0.09},
        },
        supports_by_environment={"a": {"x"}, "b": {"x"}},
        holdout_losses_by_environment={"a": {"train": 0.1, "holdout": 0.11}},
    )

    manifest = evaluation.as_manifest()

    assert evaluation.status == "passed"
    assert evaluation.claim_lane_allowed is True
    assert manifest["status"] == "passed"
    assert manifest["environment_construction_ref"] == environments.replay_identity
    assert manifest["replay_identity"] == evaluation.replay_identity


def test_transport_gate_requires_target_holdout_and_fails_closed() -> None:
    failed = evaluate_transport(
        source_environment_ids=("source-a",),
        target_environment_ids=(),
        holdout_scores={},
    )
    degraded = evaluate_transport(
        source_environment_ids=("source-a",),
        target_environment_ids=("target-b",),
        holdout_scores={"target-b": {"source_loss": 0.1, "target_loss": 1.0}},
        max_transport_degradation=0.2,
    )

    assert failed.status == "failed"
    assert "missing_target_environment" in failed.reason_codes
    assert degraded.status == "failed"
    assert "transport_holdout_failed" in degraded.reason_codes
