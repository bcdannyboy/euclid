from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from euclid.benchmarks import load_benchmark_task_manifest
from euclid.benchmarks.reporting import write_benchmark_task_report_artifacts
from euclid.benchmarks.reporting import evaluate_benchmark_semantic_assertions
from euclid.benchmarks.reporting import _portfolio_selection_explanation
from euclid.benchmarks.submitters import (
    ALGORITHMIC_SEARCH_SUBMITTER_ID,
    ANALYTIC_BACKEND_SUBMITTER_ID,
    BenchmarkHarnessContext,
    BenchmarkSubmitterResult,
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_write_benchmark_task_report_artifacts_emits_typed_json_and_markdown(
    tmp_path: Path,
) -> None:
    context = _build_context()
    submitter_results = run_benchmark_submitters(context)

    written = write_benchmark_task_report_artifacts(
        benchmark_root=tmp_path / "benchmarks",
        task_manifest=context.task_manifest,
        submitter_results=submitter_results,
        task_status="completed",
        track_summary={
            "structural_recovery_class": "equivalent_generator",
            "predictive_adequacy_status": "passed",
            "mdl_tie_break_status": "not_needed",
        },
        created_on=date(2026, 4, 14),
    )

    assert (
        written.task_result_path
        == tmp_path
        / "benchmarks"
        / "results"
        / "rediscovery"
        / "planted_analytic_demo"
        / "benchmark-task-result.json"
    )
    assert (
        written.report_path
        == tmp_path
        / "benchmarks"
        / "reports"
        / "rediscovery"
        / "planted_analytic_demo.md"
    )

    task_result = json.loads(written.task_result_path.read_text(encoding="utf-8"))
    assert task_result["artifact_type"] == "benchmark_task_result"
    assert task_result["status"] == "completed"
    assert task_result["local_winner_submitter_id"] == ANALYTIC_BACKEND_SUBMITTER_ID
    assert {
        "artifact_type": "portfolio_selection_record",
        "relative_path": (
            "results/rediscovery/planted_analytic_demo/"
            "portfolio-selection-record.json"
        ),
    }.items() <= task_result["portfolio_selection_record_ref"].items()
    assert task_result["portfolio_selection_record_ref"]["sha256"].startswith(
        "sha256:"
    )
    assert len(task_result["submitter_result_refs"]) == 4
    assert len(task_result["replay_ref_refs"]) == 4
    assert all(
        ref["sha256"].startswith("sha256:")
        for ref in task_result["submitter_result_refs"]
    )
    assert all(
        ref["sha256"].startswith("sha256:") for ref in task_result["replay_ref_refs"]
    )

    submitter_result = json.loads(
        written.submitter_result_paths[ANALYTIC_BACKEND_SUBMITTER_ID].read_text(
            encoding="utf-8"
        )
    )
    assert submitter_result["artifact_type"] == "submitter_result"
    assert submitter_result["selected_candidate_id"] == "analytic_lag1_affine"
    assert submitter_result["candidate_ledger"][0]["candidate_id"] == (
        "analytic_intercept"
    )
    assert {
        "artifact_type": "benchmark_replay_ref",
        "relative_path": (
            "results/rediscovery/planted_analytic_demo/"
            f"replay-{ANALYTIC_BACKEND_SUBMITTER_ID}.json"
        ),
    }.items() <= submitter_result["replay_ref"].items()
    assert submitter_result["replay_ref"]["sha256"].startswith("sha256:")

    replay_ref = json.loads(
        written.replay_ref_paths[ANALYTIC_BACKEND_SUBMITTER_ID].read_text(
            encoding="utf-8"
        )
    )
    assert replay_ref["artifact_type"] == "benchmark_replay_ref"
    assert replay_ref["submitter_id"] == ANALYTIC_BACKEND_SUBMITTER_ID
    assert replay_ref["replay_contract"]["candidate_id"] == "analytic_lag1_affine"

    portfolio_record = json.loads(
        written.portfolio_selection_record_path.read_text(encoding="utf-8")
    )
    assert portfolio_record["artifact_type"] == "portfolio_selection_record"
    assert (
        portfolio_record["selected_submitter_id"] == ANALYTIC_BACKEND_SUBMITTER_ID
    )
    assert portfolio_record["selected_candidate_id"] == "analytic_lag1_affine"
    assert len(portfolio_record["child_submitter_result_refs"]) == 3
    assert portfolio_record["selection_explanation"]["winner"]["submitter_id"] == (
        ANALYTIC_BACKEND_SUBMITTER_ID
    )
    assert portfolio_record["selection_explanation"]["winner"][
        "candidate_id"
    ] == "analytic_lag1_affine"
    assert (
        portfolio_record["selection_explanation"]["winner"]["total_code_bits"]
        <= portfolio_record["selection_explanation"]["runner_up"]["total_code_bits"]
    )
    assert portfolio_record["selection_explanation"]["decisive_axis"] in {
        "total_code_bits",
        "description_gain_bits",
        "structure_code_bits",
        "canonical_byte_length",
        "candidate_id",
    }

    report = written.report_path.read_text(encoding="utf-8")
    assert report.startswith("---\n")
    assert "type: report" in report
    assert "title: 'Track A Rediscovery Benchmark Report: Planted Analytic Demo'" in (
        report
    )
    assert "created: '2026-04-14'" in report
    assert "[[Benchmarking-System]]" in report
    assert "[[Benchmark-Task-Specification]]" in report
    assert "[[Benchmarking Principles]]" in report
    assert "[[Rediscovery Semantics]]" in report
    assert "This report covers only Track A: Rediscovery." in report
    assert "No single vanity score is reported here." in report
    assert "## Semantic Assertions" in report
    assert "- Overall status:" in report
    assert "- `metric_thresholds`: `failed`" in report
    assert "- `search_scope`: `passed`" in report
    assert "Winner beat runner-up on" in report


def test_portfolio_explanation_honors_metric_threshold_override() -> None:
    portfolio_result = BenchmarkSubmitterResult(
        submitter_id="portfolio_orchestrator",
        submitter_class="portfolio",
        task_id="portfolio-demo",
        track_id="predictive_generalization",
        status="selected",
        protocol_contract={},
        budget_consumption={},
        selected_candidate_id="threshold_candidate",
        selected_candidate_hash="sha256:threshold",
        replay_contract={
            "selection_rule": "min_total_code_bits",
            "selected_submitter_id": "threshold_backend",
            "selected_candidate_id": "threshold_candidate",
            "selected_candidate_hash": "sha256:threshold",
        },
        compared_finalists=(
            {
                "submitter_id": "threshold_backend",
                "candidate_id": "threshold_candidate",
                "total_code_bits": 499.0,
            },
            {
                "submitter_id": "mdl_backend",
                "candidate_id": "mdl_candidate",
                "total_code_bits": 492.0,
            },
        ),
        decision_trace=(
            {
                "step": "benchmark_metric_threshold_gate",
                "reason_code": "selected_first_threshold_passing_finalist",
            },
        ),
    )

    explanation = _portfolio_selection_explanation(portfolio_result)

    assert explanation is not None
    assert explanation["decisive_axis"] == "metric_threshold_gate"
    assert (
        explanation["decision_reason_code"]
        == "selected_first_threshold_passing_finalist"
    )


@pytest.mark.parametrize(
    ("relative_path", "expected_track_heading", "expected_related_link"),
    (
        (
            "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml",
            "Track A: Rediscovery",
            "[[Rediscovery Semantics]]",
        ),
        (
            "benchmarks/tasks/predictive_generalization/seasonal-trend-demo.yaml",
            "Track B: Predictive Generalization",
            "[[Forecast Evaluation]]",
        ),
        (
            "benchmarks/tasks/adversarial_honesty/leakage-trap-demo.yaml",
            "Track C: Adversarial And Honesty",
            "[[Nulls And Stability]]",
        ),
    ),
)
def test_report_artifacts_keep_each_track_separate(
    tmp_path: Path,
    relative_path: str,
    expected_track_heading: str,
    expected_related_link: str,
) -> None:
    manifest = load_benchmark_task_manifest(PROJECT_ROOT / relative_path)

    written = write_benchmark_task_report_artifacts(
        benchmark_root=tmp_path / "benchmarks",
        task_manifest=manifest,
        submitter_results=(_stub_submitter_result(manifest),),
        task_status="abstained",
        track_summary={"track_doctrine": "split_track_only"},
        created_on=date(2026, 4, 14),
    )

    report = written.report_path.read_text(encoding="utf-8")
    task_result = json.loads(written.task_result_path.read_text(encoding="utf-8"))

    assert expected_track_heading in report
    assert expected_related_link in report
    assert "No single vanity score is reported here." in report
    assert task_result["track_id"] == manifest.track_id
    assert (
        written.report_path.parent
        == tmp_path / "benchmarks" / "reports" / manifest.track_id
    )


def test_missing_non_abstention_metric_threshold_fails_with_submitter_source() -> None:
    manifest = replace(
        load_benchmark_task_manifest(
            PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
        ),
        metric_thresholds={
            "phase3_required_metric": {
                "metric_id": "phase3_recovery_rate",
                "comparator": ">=",
                "threshold": 0.75,
            }
        },
    )
    result = _stub_submitter_result(
        manifest,
        submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        status="selected",
        selected_candidate_id="analytic_lag1_affine",
        selected_candidate_metrics={"total_code_bits": 12.0},
    )

    assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=(result,),
        local_winner_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        local_winner_candidate_id="analytic_lag1_affine",
    )

    row = assertions["metric_thresholds"]["assertions"][0]
    assert assertions["metric_thresholds"]["status"] == "failed"
    assert {
        "threshold_id": "phase3_required_metric",
        "metric_id": "phase3_recovery_rate",
        "comparator": ">=",
        "threshold": 0.75,
        "observed_value": None,
        "status": "failed",
        "reason": "missing_observed_metric",
        "reason_code": "missing_observed_metric",
        "source_submitter_id": ANALYTIC_BACKEND_SUBMITTER_ID,
        "source_candidate_id": "analytic_lag1_affine",
    }.items() <= row.items()


def test_safe_abstention_only_passes_missing_metrics_without_a_winner() -> None:
    manifest = replace(
        load_benchmark_task_manifest(
            PROJECT_ROOT / "benchmarks/tasks/adversarial_honesty/leakage-trap-demo.yaml"
        ),
        metric_thresholds={
            "phase3_safe_abstention_metric": {
                "metric_id": "false_holistic_rate",
                "comparator": "<=",
                "threshold": 0.0,
            }
        },
    )

    safe_assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=(
            _stub_submitter_result(
                manifest,
                submitter_id=ALGORITHMIC_SEARCH_SUBMITTER_ID,
                status="abstained",
                abstention_reason="no_publishable_candidate_after_falsification",
                safe_abstention_evidence={
                    "status": "verified",
                    "evidence_type": "falsification_gate",
                    "reason_code": "no_publishable_candidate_after_falsification",
                    "support": [{"candidate_id": "unsafe_candidate"}],
                },
            ),
        ),
        local_winner_submitter_id=None,
        local_winner_candidate_id=None,
    )
    unsafe_assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=(
            _stub_submitter_result(
                manifest,
                submitter_id=ALGORITHMIC_SEARCH_SUBMITTER_ID,
                status="selected",
                selected_candidate_id="unsafe_candidate",
                selected_candidate_metrics={"total_code_bits": 1.0},
            ),
        ),
        local_winner_submitter_id=ALGORITHMIC_SEARCH_SUBMITTER_ID,
        local_winner_candidate_id="unsafe_candidate",
    )

    assert safe_assertions["metric_thresholds"]["assertions"][0]["reason"] == (
        "not_applicable_safe_abstention"
    )
    assert safe_assertions["metric_thresholds"]["status"] == "passed"
    assert unsafe_assertions["metric_thresholds"]["status"] == "failed"
    assert unsafe_assertions["metric_thresholds"]["assertions"][0]["reason"] == (
        "missing_observed_metric"
    )


def test_calibrated_or_abstain_requires_calibration_evidence_for_selected_winner(
) -> None:
    manifest = load_benchmark_task_manifest(
        PROJECT_ROOT
        / "benchmarks/tasks/predictive_generalization/probabilistic-event-probability-medium-abstention.yaml"
    )
    passing = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=(
            _stub_submitter_result(
                manifest,
                submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
                status="selected",
                selected_candidate_id="analytic_lag1_affine",
                selected_candidate_metrics={
                    "practical_significance_margin": 0.03,
                    "max_reliability_gap": 0.10,
                },
            ),
        ),
        local_winner_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        local_winner_candidate_id="analytic_lag1_affine",
    )
    failing = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=(
            _stub_submitter_result(
                manifest,
                submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
                status="selected",
                selected_candidate_id="analytic_lag1_affine",
                selected_candidate_metrics={
                    "practical_significance_margin": 0.03,
                    "max_reliability_gap": 0.40,
                },
            ),
        ),
        local_winner_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        local_winner_candidate_id="analytic_lag1_affine",
    )

    assert passing["abstention_policy"]["status"] == "passed"
    assert (
        passing["abstention_policy"]["reason_code"]
        == "calibrated_candidate_selected"
    )
    assert failing["abstention_policy"]["status"] == "failed"
    assert (
        failing["abstention_policy"]["reason_code"]
        == "selected_candidate_without_required_calibration_evidence"
    )


def test_search_scope_assertion_discloses_declared_candidate_boundary() -> None:
    manifest = load_benchmark_task_manifest(
        PROJECT_ROOT
        / "benchmarks/tasks/predictive_generalization/search-class-bounded-medium.yaml"
    )

    assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=manifest,
        submitter_results=(
            _stub_submitter_result(
                manifest,
                submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
                status="selected",
                selected_candidate_id="algorithmic_lag_plus_two",
                selected_candidate_metrics={"practical_significance_margin": 0.05},
            ),
        ),
        local_winner_submitter_id=ANALYTIC_BACKEND_SUBMITTER_ID,
        local_winner_candidate_id="algorithmic_lag_plus_two",
    )

    search_scope = assertions["search_scope"]
    assert search_scope["status"] == "passed"
    assert search_scope["independent_symbolic_rediscovery_claim"] is False
    assert search_scope["declared_candidate_count"] >= 1


def _stub_submitter_result(
    manifest,
    *,
    submitter_id: str = ALGORITHMIC_SEARCH_SUBMITTER_ID,
    status: str = "abstained",
    selected_candidate_id: str | None = None,
    selected_candidate_metrics: dict[str, float] | None = None,
    abstention_reason: str | None = "no_admissible_candidate",
    safe_abstention_evidence: dict[str, object] | None = None,
) -> BenchmarkSubmitterResult:
    return BenchmarkSubmitterResult(
        submitter_id=submitter_id,
        submitter_class="bounded_grammar",
        task_id=manifest.task_id,
        track_id=manifest.track_id,
        status=status,
        protocol_contract={
            "task_id": manifest.task_id,
            "track_id": manifest.track_id,
            "dataset_ref": manifest.dataset_ref,
            "snapshot_policy": dict(manifest.frozen_protocol.snapshot_policy),
            "target_transform_policy": dict(
                manifest.frozen_protocol.target_transform_policy
            ),
            "quantization_policy": dict(manifest.frozen_protocol.quantization_policy),
            "observation_model_policy": dict(
                manifest.frozen_protocol.observation_model_policy
            ),
            "forecast_object_type": manifest.frozen_protocol.forecast_object_type,
            "budget_policy": dict(manifest.frozen_protocol.budget_policy),
            "seed_policy": dict(manifest.frozen_protocol.seed_policy),
            "replay_policy": dict(manifest.frozen_protocol.replay_policy),
        },
        budget_consumption={
            "declared_candidate_limit": 1,
            "declared_wall_clock_seconds": 1,
            "canonical_program_count": 0,
            "attempted_candidate_count": 0,
            "accepted_candidate_count": 0,
            "rejected_candidate_count": 0,
            "omitted_candidate_count": 0,
        },
        replay_contract={
            "search_plan_id": f"{manifest.task_id}__{submitter_id}__search_plan",
            "replay_policy": dict(manifest.frozen_protocol.replay_policy),
        },
        selected_candidate_id=selected_candidate_id,
        selected_candidate_metrics=selected_candidate_metrics,
        abstention_reason=abstention_reason,
        safe_abstention_evidence=safe_abstention_evidence or {},
    )


def _build_context(
    *,
    minimum_description_gain_bits: float | None = None,
) -> BenchmarkHarnessContext:
    manifest = load_benchmark_task_manifest(
        PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
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
