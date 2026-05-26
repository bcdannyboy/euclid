from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from euclid.benchmarks import profile_benchmark_task
from euclid.benchmarks import runtime as benchmark_runtime
from euclid.benchmarks.manifests import BenchmarkSuiteSurfaceRequirement
from euclid.benchmarks.submitters import BenchmarkSubmitterResult
from euclid.control_plane import SQLiteExecutionStateStore

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _submitter_result(
    *,
    submitter_id: str,
    candidate_id: str | None = None,
    candidate_hash: str | None = None,
    replay_contract: dict[str, object] | None = None,
    selected_candidate: Any | None = None,
    selected_candidate_metrics: Mapping[str, Any] | None = None,
) -> BenchmarkSubmitterResult:
    selected = selected_candidate or (
        SimpleNamespace(
            structural_layer=SimpleNamespace(cir_family_id="analytic"),
        )
        if candidate_id is not None
        else None
    )
    return BenchmarkSubmitterResult(
        submitter_id=submitter_id,
        submitter_class="decomposition",
        task_id="portfolio-demo",
        track_id="predictive_generalization",
        status="selected" if candidate_id is not None else "abstained",
        protocol_contract={},
        budget_consumption={},
        selected_candidate=selected,
        selected_candidate_id=candidate_id,
        selected_candidate_hash=candidate_hash,
        selected_candidate_metrics=selected_candidate_metrics or {},
        replay_contract=replay_contract or {},
    )


class _FeatureViewStub(SimpleNamespace):
    def require_stage_reuse(self, stage: str) -> "_FeatureViewStub":
        del stage
        return self


def _point_metric_context() -> SimpleNamespace:
    feature_view = _FeatureViewStub(
        series_id="demo",
        entity_panel=("demo",),
        rows=(
            {"entity": "demo", "event_time": "t0", "target": 8.0},
            {"entity": "demo", "event_time": "t1", "target": 11.0},
            {"entity": "demo", "event_time": "t2", "target": 14.0},
            {"entity": "demo", "event_time": "t3", "target": 20.0},
        ),
    )
    segment = SimpleNamespace(segment_id="dev", horizon_set=(1,))
    scored_origin = SimpleNamespace(
        segment_id="dev",
        horizon=1,
        entity="demo",
        origin_index=2,
        target_index=3,
    )
    return SimpleNamespace(
        feature_view=feature_view,
        evaluation_plan=SimpleNamespace(
            development_segments=(segment,),
            confirmatory_segment=segment,
            scored_origin_panel=(scored_origin,),
        ),
        build_search_plan=lambda **kwargs: SimpleNamespace(**kwargs),
    )


def _point_metric_manifest(
    *,
    baseline_id: str,
    policy: Mapping[str, Any],
) -> SimpleNamespace:
    return SimpleNamespace(
        task_id=f"{baseline_id}_point_metric_demo",
        track_id="predictive_generalization",
        source_path=(
            PROJECT_ROOT / "benchmarks/tasks/predictive_generalization/demo.yaml"
        ),
        frozen_protocol=SimpleNamespace(forecast_object_type="point"),
        score_law="mean_absolute_error",
        baseline_registry=(
            SimpleNamespace(
                entry_id=baseline_id,
                payload={"baseline_id": baseline_id, "policy": dict(policy)},
            ),
        ),
        metric_thresholds={
            "practical_significance_margin": {
                "metric_id": "practical_significance_margin",
                "comparator": ">=",
                "threshold": 0.01,
            },
        },
    )


def test_portfolio_metric_selection_reverifies_replay_contract() -> None:
    old_finalist = {
        "submitter_id": "old_backend",
        "candidate_id": "old_candidate",
        "candidate_hash": "old_hash",
    }
    selected_finalist = {
        "submitter_id": "selected_backend",
        "candidate_id": "selected_candidate",
        "candidate_hash": "selected_hash",
    }
    portfolio_result = _submitter_result(
        submitter_id="portfolio_orchestrator",
        candidate_id="old_candidate",
        candidate_hash="old_hash",
        replay_contract={
            "replay_verification_status": "failed",
            "failure_reason_codes": ["stale_preselection_contract"],
            "selected_candidate_id": "old_candidate",
            "selected_candidate_hash": "old_hash",
            "compared_finalists": [old_finalist, selected_finalist],
            "decision_trace": [],
        },
    )
    selected_child = _submitter_result(
        submitter_id="selected_backend",
        candidate_id="selected_candidate",
        candidate_hash="selected_hash",
    )

    updated = benchmark_runtime._portfolio_result_with_metric_selection(
        task_manifest=SimpleNamespace(metric_thresholds={}),
        portfolio_result=portfolio_result,
        child_results=(selected_child,),
        selected_child=selected_child,
        previous_child=None,
    )

    assert updated.replay_contract["replay_verification_status"] == "verified"
    assert updated.replay_contract["failure_reason_codes"] == []


def test_point_practical_margin_uses_declared_baseline_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _point_metric_context()

    def _fit_candidate_window(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(fitted_candidate=kwargs["candidate"])

    def _forecast_path(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            predictions={1: float(kwargs["candidate"].forecast_value)}
        )

    monkeypatch.setattr(
        benchmark_runtime,
        "fit_candidate_window",
        _fit_candidate_window,
    )
    monkeypatch.setattr(benchmark_runtime, "build_forecast_path", _forecast_path)

    seasonal_result = benchmark_runtime._submitter_result_with_threshold_metrics(
        task_manifest=_point_metric_manifest(
            baseline_id="seasonal_naive",
            policy={"rule": "last_seasonal_cycle", "seasonal_period": 3},
        ),
        submitter_result=_submitter_result(
            submitter_id="analytic_backend",
            candidate_id="candidate_beating_declared_seasonal_baseline",
            selected_candidate=SimpleNamespace(forecast_value=19.0),
        ),
        context=context,
        catalog=object(),
    )

    assert seasonal_result.selected_candidate_metrics[
        "practical_significance_margin"
    ] == pytest.approx(11.0)

    self_baseline_manifest = _point_metric_manifest(
        baseline_id="naive_last_value",
        policy={"rule": "last_observation_carried_forward"},
    )
    self_baseline_result = benchmark_runtime._submitter_result_with_threshold_metrics(
        task_manifest=self_baseline_manifest,
        submitter_result=_submitter_result(
            submitter_id="algorithmic_search_backend",
            candidate_id="algorithmic_last_observation",
            selected_candidate=SimpleNamespace(forecast_value=14.0),
        ),
        context=context,
        catalog=object(),
    )

    assert self_baseline_result.selected_candidate_metrics[
        "practical_significance_margin"
    ] == pytest.approx(0.0)
    assert not benchmark_runtime._result_satisfies_metric_thresholds(
        task_manifest=self_baseline_manifest,
        submitter_result=self_baseline_result,
    )


def test_threshold_metric_enrichment_skips_measurement_when_metrics_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_manifest = _point_metric_manifest(
        baseline_id="naive_last_value",
        policy={"rule": "last_observation_carried_forward"},
    )
    submitter_result = _submitter_result(
        submitter_id="analytic_backend",
        candidate_id="already_measured",
        selected_candidate_metrics={
            "practical_significance_margin": 0.25,
        },
    )

    def _unexpected_measurement(**kwargs: Any) -> dict[str, float]:
        raise AssertionError("threshold measurement should not run")

    monkeypatch.setattr(
        benchmark_runtime,
        "_measured_threshold_metrics",
        _unexpected_measurement,
    )

    enriched = benchmark_runtime._submitter_result_with_threshold_metrics(
        task_manifest=task_manifest,
        submitter_result=submitter_result,
        context=object(),
        catalog=object(),
    )

    assert enriched is submitter_result


def test_submitter_cache_signature_binds_feature_view_context(tmp_path: Path) -> None:
    manifest_path = tmp_path / "task.yaml"
    manifest_path.write_text("task_id: cache_signature_demo\n", encoding="utf-8")
    task_manifest = SimpleNamespace(
        task_id="cache_signature_demo",
        track_id="predictive_generalization",
        source_path=manifest_path,
    )
    snapshot = SimpleNamespace(
        series_id="demo",
        cutoff_available_at="2026-04-14T00:00:00Z",
        revision_policy="latest_visible",
        row_count=2,
        entity_panel=("demo",),
        lineage_payload_hashes=("row-a", "row-b"),
        materialization_hashes=SimpleNamespace(
            raw_observation_hash="raw-a",
            coded_target_hash="coded-a",
            lineage_payload_hash="lineage-a",
        ),
    )
    evaluation_plan = SimpleNamespace(
        as_dict=lambda: {"split_policy": "rolling_origin", "horizon": 1}
    )
    base_context = SimpleNamespace(
        task_manifest=task_manifest,
        protocol_contract={"task_id": "cache_signature_demo"},
        search_class="bounded_heuristic",
        seasonal_period=2,
        proposal_specs=(),
        project_root=PROJECT_ROOT,
        snapshot=snapshot,
        feature_view=SimpleNamespace(
            series_id="demo",
            feature_names=("lag_1",),
            entity_panel=("demo",),
            rows=({"lag_1": 1.0}, {"lag_1": 2.0}),
            materialization_report=None,
        ),
        evaluation_plan=evaluation_plan,
    )
    changed_context = SimpleNamespace(
        **{
            **base_context.__dict__,
            "feature_view": SimpleNamespace(
                series_id="demo",
                feature_names=("lag_1",),
                entity_panel=("demo",),
                rows=({"lag_1": 1.0}, {"lag_1": 3.0}),
                materialization_report=None,
            ),
        }
    )

    assert benchmark_runtime._submitter_cache_signature(
        context=base_context,
        submitter_id="analytic_backend",
    ) != benchmark_runtime._submitter_cache_signature(
        context=changed_context,
        submitter_id="analytic_backend",
    )


def test_runtime_cache_signature_covers_submitter_semantic_dependencies() -> None:
    signatures = benchmark_runtime._runtime_source_signatures(project_root=PROJECT_ROOT)

    assert {
        "adapters/portfolio.py",
        "algorithmic_dsl.py",
        "cir/models.py",
        "math/quantization.py",
        "math/reference_descriptions.py",
        "reducers/models.py",
        "search/frontier.py",
        "search/policies.py",
    } <= set(signatures)
    assert all(signatures[path]["sha256"] for path in signatures)


def test_profile_benchmark_task_emits_telemetry_and_report_artifacts(
    tmp_path: Path,
) -> None:
    result = profile_benchmark_task(
        manifest_path=(
            PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
        ),
        benchmark_root=tmp_path / "benchmarks",
    )

    assert result.report_paths.task_result_path.is_file()
    assert result.telemetry_path.is_file()

    payload = json.loads(result.telemetry_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "performance_telemetry"
    assert payload["profile_kind"] == "benchmark_task"
    assert payload["subject_id"] == "planted_analytic_demo"
    assert payload["artifact_store"]["write_operation_count"] > 0
    assert payload["artifact_store"]["write_throughput_bytes_per_second"] >= 0.0
    assert any(
        record["submitter_id"] == "analytic_backend"
        and record["declared_candidate_limit"] == 128
        for record in payload["budget_records"]
    )
    assert payload["attributes"]["benchmark_budget_report"] == {
        "budget_id": "benchmark_task_budget:planted_analytic_demo",
        "candidate_limit": 128,
        "wall_clock_seconds": 300,
        "parallel_workers": 1,
        "submitter_count": 4,
        "status": "reported",
    }
    assert any(
        record["submitter_id"] == "analytic_backend"
        and record["declared_restarts"] == 3
        for record in payload["restart_records"]
    )
    assert {span["category"] for span in payload["spans"]} >= {
        "benchmark_intake",
        "search",
        "portfolio_selection",
        "reporting",
    }

    task_result = json.loads(
        result.report_paths.task_result_path.read_text(encoding="utf-8")
    )
    semantic_summary = task_result["semantic_summary"]
    assert semantic_summary["run_support_object_ids"] == [
        "observation_model:gaussian_point",
        "quantization:decimal_1e-6",
        "target_transform:identity",
    ]
    assert semantic_summary["claim_lane_ids"] == ["forecast_object:point"]
    assert semantic_summary["replay_ids"] == ["replay:ledger_only"]
    assert semantic_summary["engine_ids"] == [
        "algorithmic_search_backend",
        "analytic_backend",
        "portfolio_orchestrator",
        "recursive_spectral_backend",
    ]
    assert semantic_summary["score_policy_ids"] == ["score:mean_absolute_error"]
    assert semantic_summary["threshold_ids"] == [
        "practical_significance_margin",
        "predictive_adequacy_floor:mean_absolute_error",
    ]


def test_surface_status_fails_when_task_result_file_lacks_semantic_summary(
    tmp_path: Path,
) -> None:
    result = profile_benchmark_task(
        manifest_path=(
            PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
        ),
        benchmark_root=tmp_path / "benchmarks",
    )
    result.report_paths.task_result_path.write_text(
        json.dumps(
            {
                "artifact_type": "benchmark_task_result",
                "status": "completed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    status = benchmark_runtime._surface_status(
        requirement=BenchmarkSuiteSurfaceRequirement(
            surface_id="retained_core_release",
            task_ids=("planted_analytic_demo",),
            replay_required=True,
        ),
        task_results=(result,),
    )

    assert status.benchmark_status == "failed"
    assert status.evidence["semantic_status"] == "missing"


def test_profile_benchmark_task_resume_reuses_cached_context_and_submitters(
    tmp_path: Path,
) -> None:
    benchmark_root = tmp_path / "benchmarks"

    first = profile_benchmark_task(
        manifest_path=(
            PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
        ),
        benchmark_root=benchmark_root,
        parallel_workers=3,
    )
    second = profile_benchmark_task(
        manifest_path=(
            PROJECT_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
        ),
        benchmark_root=benchmark_root,
        parallel_workers=3,
    )

    assert [
        result.selected_candidate_id for result in first.submitter_results
    ] == [result.selected_candidate_id for result in second.submitter_results]

    payload = json.loads(second.telemetry_path.read_text(encoding="utf-8"))
    assert any(
        measurement["name"] == "resume_checkpoint_hits"
        and measurement["category"] == "checkpoint_resume"
        and measurement["value"] >= 4
        for measurement in payload["measurements"]
    )
    assert any(
        measurement["name"] == "parallel_worker_count"
        and measurement["category"] == "benchmark_runtime"
        and measurement["value"] == 3
        for measurement in payload["measurements"]
    )

    control_plane_path = (
        benchmark_root
        / "results"
        / "rediscovery"
        / "planted_analytic_demo"
        / "_profile_runtime"
        / "active-runs"
        / "planted_analytic_demo"
        / "control-plane"
        / "execution-state.sqlite3"
    )
    snapshot = SQLiteExecutionStateStore(control_plane_path).load_run_snapshot(
        "planted_analytic_demo"
    )
    assert {
        state.step_id: state.status for state in snapshot.step_states
    } == {
        "benchmark.runtime.context": "completed",
        "benchmark.submitter.analytic_backend": "completed",
        "benchmark.submitter.recursive_spectral_backend": "completed",
        "benchmark.submitter.algorithmic_search_backend": "completed",
        "benchmark.submitter.portfolio_orchestrator": "completed",
        "benchmark.runtime.reporting": "completed",
    }


def test_pickle_cache_requires_signature_sidecar_before_unpickling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "cached-result.pkl"
    cache_path.write_bytes(b"not a trusted pickle")

    def _fail_if_unpickled(raw: bytes) -> object:
        raise AssertionError("pickle cache was loaded before signature sidecar")

    monkeypatch.setattr(benchmark_runtime.pickle, "loads", _fail_if_unpickled)

    assert (
        benchmark_runtime._load_pickle_cache(
            cache_path,
            expected_signature="sha256:test",
        )
        is None
    )


def test_cache_signature_changes_when_manifest_content_changes(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    dataset_path = tmp_path / "dataset.csv"
    manifest_path.write_text(
        (
            PROJECT_ROOT
            / "benchmarks/tasks/predictive_generalization/"
            "search-class-bounded-medium.yaml"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    dataset_path.write_text("entity_id,target\nfixture,1.0\n", encoding="utf-8")
    manifest = benchmark_runtime.load_benchmark_task_manifest(manifest_path)
    first_signature = benchmark_runtime._cache_signature(
        task_manifest=manifest,
        dataset_path=dataset_path,
        suffix="context",
    )

    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            "candidate_limit: 2",
            "candidate_limit: 3",
        ),
        encoding="utf-8",
    )
    changed_manifest = benchmark_runtime.load_benchmark_task_manifest(manifest_path)

    assert (
        benchmark_runtime._cache_signature(
            task_manifest=changed_manifest,
            dataset_path=dataset_path,
            suffix="context",
        )
        != first_signature
    )


def test_runtime_cache_signature_covers_transitive_semantic_sources() -> None:
    signatures = benchmark_runtime._runtime_source_signatures()

    assert "modules/replay.py" in signatures
    assert "cir/normalize.py" in signatures
    assert "search/descriptive_coding.py" in signatures
    assert "reducers/composition.py" in signatures


def test_probabilistic_baseline_row_uses_independent_scale() -> None:
    baseline_row = benchmark_runtime._probabilistic_prediction_row_with_location(
        row={
            "distribution_parameters": {
                "family": "normal",
                "location": 100.0,
                "scale": 9.0,
            },
            "lower_bound": 91.0,
            "upper_bound": 109.0,
            "quantiles": [
                {"level": 0.1, "value": 91.0},
                {"level": 0.9, "value": 109.0},
            ],
        },
        location=50.0,
        scale=2.0,
    )

    assert baseline_row["distribution_parameters"]["scale"] == 2.0
    assert baseline_row["lower_bound"] == 48.0
    assert baseline_row["upper_bound"] == 52.0
    assert baseline_row["quantiles"] == [
        {"level": 0.1, "value": 48.0},
        {"level": 0.9, "value": 52.0},
    ]
