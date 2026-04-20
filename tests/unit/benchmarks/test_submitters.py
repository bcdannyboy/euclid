from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from euclid.benchmarks import load_benchmark_task_manifest
from euclid.benchmarks.submitters import (
    ALGORITHMIC_SEARCH_SUBMITTER_ID,
    ANALYTIC_BACKEND_SUBMITTER_ID,
    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
    RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
    REQUIRED_BENCHMARK_SUBMITTER_IDS,
    BenchmarkHarnessContext,
    run_benchmark_submitter,
    run_benchmark_submitters,
)
from euclid.contracts.refs import TypedRef
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.performance import TelemetryRecorder
from euclid.search.backends import DescriptiveSearchProposal

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_phase08_submitter_registry_exposes_all_required_submitters() -> None:
    assert REQUIRED_BENCHMARK_SUBMITTER_IDS == (
        ANALYTIC_BACKEND_SUBMITTER_ID,
        RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
        ALGORITHMIC_SEARCH_SUBMITTER_ID,
        PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
    )


def test_submitters_share_one_protocol_and_preserve_ledgers_and_budget_records(
) -> None:
    context = _build_context()

    results = {
        result.submitter_id: result for result in run_benchmark_submitters(context)
    }

    assert tuple(results) == REQUIRED_BENCHMARK_SUBMITTER_IDS
    assert {
        submitter_id: result.protocol_contract
        for submitter_id, result in results.items()
        if submitter_id != PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
    } == {
        ANALYTIC_BACKEND_SUBMITTER_ID: context.protocol_contract,
        RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID: context.protocol_contract,
        ALGORITHMIC_SEARCH_SUBMITTER_ID: context.protocol_contract,
    }

    analytic_backend = results[ANALYTIC_BACKEND_SUBMITTER_ID]
    assert analytic_backend.status == "selected"
    assert analytic_backend.selected_candidate_id == "analytic_lag1_affine"
    assert [
        entry.candidate_id
        for entry in analytic_backend.candidate_ledger
        if entry.ledger_status == "accepted"
    ] == ["analytic_intercept", "analytic_lag1_affine"]
    assert analytic_backend.budget_consumption == {
        "declared_candidate_limit": 128,
        "declared_wall_clock_seconds": 300,
        "canonical_program_count": 2,
        "attempted_candidate_count": 2,
        "accepted_candidate_count": 2,
        "rejected_candidate_count": 0,
        "omitted_candidate_count": 0,
    }

    recursive_spectral_backend = results[RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID]
    assert recursive_spectral_backend.status == "selected"
    assert (
        recursive_spectral_backend.selected_candidate_id
        == "recursive_level_smoother"
    )
    assert {
        entry.primitive_family for entry in recursive_spectral_backend.candidate_ledger
    } == {
        "recursive",
        "spectral",
    }
    assert {
        entry.candidate_id: entry.ledger_status
        for entry in recursive_spectral_backend.candidate_ledger
    } == {
        "recursive_level_smoother": "accepted",
        "recursive_running_mean": "accepted",
        "spectral_harmonic_1": "rejected",
        "spectral_harmonic_2": "rejected",
    }

    algorithmic_search_backend = results[ALGORITHMIC_SEARCH_SUBMITTER_ID]
    assert algorithmic_search_backend.status == "selected"
    assert (
        algorithmic_search_backend.selected_candidate_id
        == "algorithmic_last_observation"
    )
    assert {
        entry.candidate_id: entry.ledger_status
        for entry in algorithmic_search_backend.candidate_ledger
    } == {
        "algorithmic_last_observation": "accepted",
        "algorithmic_running_half_average": "rejected",
    }


def test_portfolio_submitter_preserves_child_results_and_selection_trace() -> None:
    context = _build_context()

    result = run_benchmark_submitter(
        context=context,
        submitter_id=PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
    )

    assert result.status == "selected"
    assert result.selected_candidate_id == "analytic_lag1_affine"
    assert tuple(child.submitter_id for child in result.child_results) == (
        ANALYTIC_BACKEND_SUBMITTER_ID,
        RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
        ALGORITHMIC_SEARCH_SUBMITTER_ID,
    )
    assert {item["submitter_id"] for item in result.compared_finalists} == {
        ANALYTIC_BACKEND_SUBMITTER_ID,
        RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
        ALGORITHMIC_SEARCH_SUBMITTER_ID,
    }
    assert [step["step"] for step in result.decision_trace] == [
        "collect_submitter_finalists",
        "rank_submitter_finalists",
        "select_portfolio_winner",
    ]


def test_submitter_abstention_keeps_rejected_ledgers_visible() -> None:
    context = _build_context(minimum_description_gain_bits=100.0)

    result = run_benchmark_submitter(
        context=context,
        submitter_id=ALGORITHMIC_SEARCH_SUBMITTER_ID,
    )

    assert result.status == "abstained"
    assert result.selected_candidate_id is None
    assert result.abstention_reason == "no_admissible_candidate"
    assert {
        entry.candidate_id: entry.ledger_status for entry in result.candidate_ledger
    } == {
        "algorithmic_last_observation": "rejected",
        "algorithmic_running_half_average": "rejected",
    }


def test_submitter_keeps_default_fallback_candidates_when_custom_specs_exist() -> None:
    context = replace(
        _build_context(),
        proposal_specs=(
            DescriptiveSearchProposal(
                candidate_id="shared_local_panel_false_generalization",
                primitive_family="analytic",
                form_class="closed_form_expression",
                feature_dependencies=("missing_panel_feature",),
                parameter_values={"shared_intercept": 0.0},
            ),
        ),
    )

    result = run_benchmark_submitter(
        context=context,
        submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
    )

    assert result.status == "selected"
    assert result.selected_candidate_id == "analytic_lag1_affine"
    assert {
        entry.candidate_id: entry.ledger_status for entry in result.candidate_ledger
    } == {
        "shared_local_panel_false_generalization": "rejected",
        "analytic_intercept": "accepted",
        "analytic_lag1_affine": "accepted",
    }


def test_submitter_runner_reuses_single_submitter_results_for_portfolio() -> None:
    context = _build_context()
    telemetry = TelemetryRecorder()

    results = run_benchmark_submitters(
        context,
        telemetry=telemetry,
        parallel_workers=3,
    )
    artifact = telemetry.build_artifact(
        profile_kind="benchmark_submitters",
        subject_id=context.task_manifest.task_id,
    )

    assert tuple(result.submitter_id for result in results) == (
        ANALYTIC_BACKEND_SUBMITTER_ID,
        RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
        ALGORITHMIC_SEARCH_SUBMITTER_ID,
        PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
    )
    assert [record.submitter_id for record in artifact.budget_records] == [
        ANALYTIC_BACKEND_SUBMITTER_ID,
        RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
        ALGORITHMIC_SEARCH_SUBMITTER_ID,
    ]
    assert any(
        measurement.name == "parallel_worker_count"
        and measurement.category == "benchmark_runtime"
        and measurement.value == 3
        for measurement in artifact.measurements
    )


def test_parallel_submitter_execution_is_deterministic_and_sorts_telemetry() -> None:
    context = _build_context()
    serial_telemetry = TelemetryRecorder()
    parallel_telemetry = TelemetryRecorder()

    serial_results = run_benchmark_submitters(
        context,
        telemetry=serial_telemetry,
        parallel_workers=1,
    )
    parallel_results = run_benchmark_submitters(
        context,
        telemetry=parallel_telemetry,
        parallel_workers=3,
    )

    assert _result_summary(serial_results) == _result_summary(parallel_results)

    artifact = parallel_telemetry.build_artifact(
        profile_kind="benchmark_submitters",
        subject_id=context.task_manifest.task_id,
    )
    assert [record.scope for record in artifact.seed_records] == [
        f"search:{ALGORITHMIC_SEARCH_SUBMITTER_ID}",
        f"search:{ANALYTIC_BACKEND_SUBMITTER_ID}",
        f"search:{RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID}",
    ]
    assert [record.submitter_id for record in artifact.budget_records] == [
        ANALYTIC_BACKEND_SUBMITTER_ID,
        RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
        ALGORITHMIC_SEARCH_SUBMITTER_ID,
    ]
    assert [record.submitter_id for record in artifact.restart_records] == [
        ANALYTIC_BACKEND_SUBMITTER_ID,
        RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
        ALGORITHMIC_SEARCH_SUBMITTER_ID,
    ]


def test_adversarial_portfolio_abstains_when_task_requires_safe_outcome() -> None:
    context = _build_context(
        manifest_relative_path=(
            "benchmarks/tasks/adversarial_honesty/leakage-trap-demo.yaml"
        )
    )

    result = run_benchmark_submitter(
        context=context,
        submitter_id=PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
    )

    assert result.status == "abstained"
    assert result.selected_candidate_id is None
    assert result.abstention_reason == "no_admissible_candidate"
    assert tuple(child.submitter_id for child in result.child_results) == (
        ANALYTIC_BACKEND_SUBMITTER_ID,
        RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
        ALGORITHMIC_SEARCH_SUBMITTER_ID,
    )
    assert {item["submitter_id"] for item in result.compared_finalists} == {
        child.submitter_id
        for child in result.child_results
        if child.status == "selected"
    }
    assert [step["step"] for step in result.decision_trace] == [
        "collect_submitter_finalists",
        "rank_submitter_finalists",
        "select_portfolio_winner",
        "honesty_safe_outcome_gate",
        "select_portfolio_winner",
    ]
    assert result.replay_contract["replay_policy"] == dict(
        context.task_manifest.frozen_protocol.replay_policy
    )
    assert result.replay_contract["selected_submitter_id"] is None
    assert result.replay_contract["selected_provenance_id"] is None
    assert result.replay_contract["selected_candidate_id"] is None
    assert result.replay_contract["selected_candidate_hash"] is None
    assert result.replay_contract["compared_finalists"] == list(
        result.compared_finalists
    )
    assert result.replay_contract["decision_trace"] == list(result.decision_trace)


def _build_context(
    *,
    manifest_relative_path: str = (
        "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
    ),
    minimum_description_gain_bits: float | None = None,
) -> BenchmarkHarnessContext:
    manifest = load_benchmark_task_manifest(
        PROJECT_ROOT / manifest_relative_path
    )
    snapshot = FrozenDatasetSnapshot(
        series_id="benchmark-submitter-series",
        cutoff_available_at="2026-01-06T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=(
            SnapshotRow(
                event_time="2026-01-01T00:00:00Z",
                available_at="2026-01-01T00:00:00Z",
                observed_value=10.0,
                revision_id=0,
                payload_hash="sha256:a",
            ),
            SnapshotRow(
                event_time="2026-01-02T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                observed_value=12.0,
                revision_id=0,
                payload_hash="sha256:b",
            ),
            SnapshotRow(
                event_time="2026-01-03T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                observed_value=13.0,
                revision_id=0,
                payload_hash="sha256:c",
            ),
            SnapshotRow(
                event_time="2026-01-04T00:00:00Z",
                available_at="2026-01-04T00:00:00Z",
                observed_value=15.0,
                revision_id=0,
                payload_hash="sha256:d",
            ),
            SnapshotRow(
                event_time="2026-01-05T00:00:00Z",
                available_at="2026-01-05T00:00:00Z",
                observed_value=16.0,
                revision_id=0,
                payload_hash="sha256:e",
            ),
            SnapshotRow(
                event_time="2026-01-06T00:00:00Z",
                available_at="2026-01-06T00:00:00Z",
                observed_value=18.0,
                revision_id=0,
                payload_hash="sha256:f",
            ),
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    canonicalization_policy = build_canonicalization_policy()
    template_plan = build_search_plan(
        evaluation_plan=evaluation_plan,
        canonicalization_policy_ref=TypedRef(
            "canonicalization_policy_manifest@1.0.0",
            canonicalization_policy.canonicalization_policy_id,
        ),
        codelength_policy_ref=TypedRef(
            "codelength_policy_manifest@1.1.0",
            "mdl_policy_default",
        ),
        reference_description_policy_ref=TypedRef(
            "reference_description_policy_manifest@1.1.0",
            "reference_description_default",
        ),
        observation_model_ref=TypedRef(
            "observation_model_manifest@1.1.0",
            "observation_model_default",
        ),
        candidate_family_ids=("analytic_intercept",),
        search_class="exact_finite_enumeration",
        proposal_limit=1,
        seasonal_period=4,
        minimum_description_gain_bits=minimum_description_gain_bits,
    )
    return BenchmarkHarnessContext(
        task_manifest=manifest,
        snapshot=snapshot,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        canonicalization_policy_ref=template_plan.canonicalization_policy_ref,
        codelength_policy_ref=template_plan.codelength_policy_ref,
        reference_description_policy_ref=(
            template_plan.reference_description_policy_ref
        ),
        observation_model_ref=template_plan.observation_model_ref,
        search_class="exact_finite_enumeration",
        seasonal_period=4,
        minimum_description_gain_bits=minimum_description_gain_bits,
    )


def _result_summary(results: tuple) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            result.submitter_id,
            result.status,
            result.selected_candidate_id,
            tuple(
                (
                    entry.candidate_id,
                    entry.primitive_family,
                    entry.ledger_status,
                    entry.reason_codes,
                )
                for entry in result.candidate_ledger
            ),
            tuple(
                child.submitter_id for child in result.child_results
            ),
            tuple(
                (
                    item.get("submitter_id"),
                    item.get("candidate_id"),
                )
                for item in result.compared_finalists
            ),
            tuple(step.get("step") for step in result.decision_trace),
        )
        for result in results
    )
