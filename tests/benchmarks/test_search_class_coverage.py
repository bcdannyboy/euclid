from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

import euclid
from euclid.benchmarks import load_benchmark_task_manifest
from euclid.benchmarks.submitters import ALGORITHMIC_SEARCH_SUBMITTER_ID

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEARCH_CLASSES_PATH = PROJECT_ROOT / "schemas/contracts/search-classes.yaml"
EQUALITY_MANIFEST = (
    PROJECT_ROOT
    / "benchmarks/tasks/predictive_generalization/"
    "search-class-equality-saturation-medium.yaml"
)
STOCHASTIC_MANIFEST = (
    PROJECT_ROOT
    / "benchmarks/tasks/predictive_generalization/search-class-stochastic-medium.yaml"
)


def test_search_class_manifest_exposes_declared_honesty_contract(
    tmp_path: Path,
) -> None:
    contracts = _search_class_contracts()
    manifest_path = _write_search_class_manifest(
        tmp_path=tmp_path,
        task_id="equality_saturation_surface_demo",
        search_class="equality_saturation_heuristic",
        search_contract=contracts["equality_saturation_heuristic"],
    )

    manifest = load_benchmark_task_manifest(manifest_path)

    assert manifest.search_class == "equality_saturation_heuristic"
    assert manifest.search_class_honesty["coverage_statement"] == (
        "incomplete_search_disclosed"
    )
    assert manifest.search_class_honesty["exactness_ceiling"] == (
        "no_global_exactness_claim"
    )
    assert manifest.search_class_honesty["requires_disclosure"] == [
        "rewrite_system",
        "extractor_cost",
        "stop_rule",
    ]


def test_search_class_suite_summary_closes_rows_for_every_admitted_class(
    tmp_path: Path,
) -> None:
    contracts = _search_class_contracts()
    suite_manifest = _write_search_class_suite_manifest(tmp_path, contracts=contracts)

    result = euclid.profile_benchmark_suite(
        manifest_path=suite_manifest,
        benchmark_root=tmp_path / "search-class-suite-output",
        project_root=PROJECT_ROOT,
        resume=False,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    coverage_rows = {
        row["search_class"]: row for row in summary["search_class_coverage"]
    }

    assert set(coverage_rows) == set(contracts)
    for search_class, contract in contracts.items():
        row = coverage_rows[search_class]
        assert row["coverage_statement"] == contract["coverage_statement"]
        assert row["exactness_ceiling"] == contract["exactness_ceiling"]
        assert row["requires_disclosure"] == contract["requires_disclosure"]
        assert row["proof_mode"] == "direct_benchmark_task"
        assert row[
            "covered_task_ids"
        ], f"{search_class} needs direct benchmark evidence"
        assert row["replay_verified"] is True


def test_equality_saturation_tasks_use_equality_saturation_backend(
    tmp_path: Path,
) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=EQUALITY_MANIFEST,
        benchmark_root=tmp_path / "equality-saturation-benchmark",
        project_root=PROJECT_ROOT,
        resume=False,
    )

    submitter_result = result.submitter_results[0]
    selected_candidate = submitter_result.selected_candidate
    assert selected_candidate is not None
    assert (
        selected_candidate.evidence_layer.backend_origin_record.search_class
        == "equality_saturation_heuristic"
    )
    assert (
        selected_candidate.evidence_layer.backend_origin_record.backend_family
        == "algorithmic"
    )


def test_stochastic_tasks_use_stochastic_backend(tmp_path: Path) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=STOCHASTIC_MANIFEST,
        benchmark_root=tmp_path / "stochastic-benchmark",
        project_root=PROJECT_ROOT,
        resume=False,
    )

    submitter_result = result.submitter_results[0]
    selected_candidate = submitter_result.selected_candidate
    assert selected_candidate is not None
    assert (
        selected_candidate.evidence_layer.backend_origin_record.search_class
        == "stochastic_heuristic"
    )
    assert (
        selected_candidate.evidence_layer.backend_origin_record.backend_family
        == "algorithmic"
    )


def _search_class_contracts() -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(SEARCH_CLASSES_PATH.read_text(encoding="utf-8"))
    return {entry["search_class"]: entry for entry in payload["contracts"]}


def _write_search_class_suite_manifest(
    tmp_path: Path,
    *,
    contracts: dict[str, dict[str, Any]],
) -> Path:
    task_manifest_paths = []
    for index, search_class in enumerate(sorted(contracts)):
        task_manifest_paths.append(
            str(
                _write_search_class_manifest(
                    tmp_path=tmp_path,
                    task_id=f"{search_class}_search_demo_{index}",
                    search_class=search_class,
                    search_contract=contracts[search_class],
                ).resolve()
            )
        )

    suite_manifest = tmp_path / "benchmarks" / "suites" / "search-class-coverage.yaml"
    suite_manifest.parent.mkdir(parents=True, exist_ok=True)
    suite_manifest.write_text(
        yaml.safe_dump(
            {
                "suite_id": "search_class_coverage",
                "description": (
                    "Temporary suite used to prove that benchmark summaries close "
                    "all admitted search classes directly."
                ),
                "task_manifest_paths": task_manifest_paths,
                "required_tracks": ["predictive_generalization"],
                "surface_requirements": [
                    {
                        "surface_id": "search_class_honesty",
                        "task_ids": [
                            Path(path).stem.replace("-", "_")
                            for path in task_manifest_paths
                        ],
                        "replay_required": True,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return suite_manifest


def _write_search_class_manifest(
    *,
    tmp_path: Path,
    task_id: str,
    search_class: str,
    search_contract: dict[str, Any],
) -> Path:
    manifest_path = (
        tmp_path
        / "tasks"
        / "predictive_generalization"
        / f"{task_id.replace('_', '-')}.yaml"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "task_id": task_id,
                "track_id": "predictive_generalization",
                "task_family": "single_entity_point_forecast",
                "regime_tags": ["search_class_proof"],
                "dataset_ref": (
                    "fixtures/runtime/full_vision_certification/"
                    "algorithmic_rediscovery/algorithmic-search-series.csv"
                ),
                "snapshot_policy": {
                    "freeze_mode": "content_addressed_copy",
                    "availability_cutoff": "2026-04-12T00:00:00Z",
                },
                "generator_status": "unknown_real_world",
                "target_transform_policy": {"transform_id": "identity"},
                "quantization_policy": {"lattice": "decimal_1e-6"},
                "observation_model_policy": {"model_id": "gaussian_point"},
                "split_policy": {
                    "policy_id": "rolling_origin",
                    "initial_window": 18,
                    "step": 1,
                },
                "forecast_object_type": "point",
                "score_policy": {"metric_id": "mean_absolute_error"},
                "practical_significance_margin": 0.02,
                "budget_policy": {
                    "wall_clock_seconds": 60,
                    "candidate_limit": 16,
                },
                "baseline_registry": [{"baseline_id": "naive_last_value"}],
                "submitter_registry": [
                    {"submitter_id": ALGORITHMIC_SEARCH_SUBMITTER_ID}
                ],
                "seed_policy": {
                    "seed": 11,
                    "restarts": 3 if search_class == "stochastic_heuristic" else 0,
                },
                "adversarial_tags": ["search_class_surface"],
                "abstention_policy": {
                    "allow_abstention": True,
                    "expected_mode": "no_publishable_winner",
                },
                "forbidden_shortcuts": ["oracle_fit"],
                "replay_policy": {
                    "ledger_mode": "append_only",
                    "persist_candidate_ledgers": True,
                },
                "origin_policy": {
                    "policy_id": "rolling_origin",
                    "min_origins": 4,
                },
                "horizon_policy": {"horizons": [1]},
                "baseline_comparison_policy": {
                    "paired_test": "diebold_mariano",
                    "require_margin_win": True,
                },
                "search_class": search_class,
                "search_class_honesty": {
                    "coverage_statement": search_contract["coverage_statement"],
                    "exactness_ceiling": search_contract["exactness_ceiling"],
                    "requires_disclosure": search_contract["requires_disclosure"],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest_path
