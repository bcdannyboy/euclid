from __future__ import annotations

from pathlib import Path

import yaml

import euclid.operator_runtime.intake_planning as intake_planning
from euclid.benchmarks import profile_benchmark_task
from euclid.operator_runtime.run import run_operator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINT_MANIFEST = PROJECT_ROOT / "examples/current_release_run.yaml"
BENCHMARK_MANIFEST = (
    PROJECT_ROOT
    / "benchmarks/tasks/predictive_generalization/"
    "search-class-equality-saturation-medium.yaml"
)


def test_operator_run_uses_equality_saturation_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = yaml.safe_load(POINT_MANIFEST.read_text(encoding="utf-8"))
    payload["dataset_csv"] = str(POINT_MANIFEST.parent / payload["dataset_csv"])
    payload["search"]["class"] = "equality_saturation_heuristic"
    payload["search"]["proposal_limit"] = 4
    manifest_path = tmp_path / "equality-saturation-operator.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    seen_search_classes: list[str] = []
    original = intake_planning.build_search_plan

    def _capture_search_plan(*args, **kwargs):
        search_plan = original(*args, **kwargs)
        seen_search_classes.append(search_plan.search_class)
        return search_plan

    monkeypatch.setattr(intake_planning, "build_search_plan", _capture_search_plan)

    result = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "operator-run",
    )

    assert seen_search_classes == ["equality_saturation_heuristic"]
    assert result.summary.selected_family in {
        "constant",
        "drift",
        "linear_trend",
        "seasonal_naive",
    }
    assert result.summary.selected_candidate_id


def test_benchmark_submitter_result_is_not_just_analytic_alias(tmp_path: Path) -> None:
    result = profile_benchmark_task(
        manifest_path=BENCHMARK_MANIFEST,
        benchmark_root=tmp_path / "benchmark-output",
        project_root=PROJECT_ROOT,
        resume=False,
    )
    submitter_result = result.submitter_results[0]
    search_evidence = (
        submitter_result.selected_candidate.evidence_layer.transient_diagnostics[
            "search_evidence"
        ]
    )

    assert submitter_result.submitter_id == "algorithmic_search_backend"
    assert (
        submitter_result.selected_candidate.structural_layer.cir_family_id
        == "algorithmic"
    )
    assert search_evidence["search_class"] == "equality_saturation_heuristic"
    assert search_evidence["rewrite_space_candidate_ids"]
    assert all(
        candidate_id.startswith("algorithmic_")
        or candidate_id == "algorithmic_last_observation"
        or candidate_id == "algorithmic_running_half_average"
        for candidate_id in search_evidence["rewrite_space_candidate_ids"]
    )
