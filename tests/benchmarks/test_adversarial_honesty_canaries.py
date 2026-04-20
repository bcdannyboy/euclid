from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from euclid.benchmarks import load_benchmark_task_manifest, profile_benchmark_task

ASSET_ROOT = Path(__file__).resolve().parents[2] / "src" / "euclid" / "_assets"
TASK_ROOT = ASSET_ROOT / "benchmarks" / "tasks" / "adversarial_honesty"
FIXTURE_ROOT = ASSET_ROOT / "fixtures" / "runtime" / "phase08" / "adversarial_honesty"
FIXTURE_SET = FIXTURE_ROOT / "fixture-set.yaml"

CANARY_CASES = (
    (
        "random-walk-canary.yaml",
        "random_walk_null",
        "fixtures/runtime/phase08/adversarial_honesty/random-walk-canary-series.csv",
        {"false_holistic_canary", "random_walk", "negative_case"},
    ),
    (
        "near-persistence-canary.yaml",
        "near_persistence_bait",
        (
            "fixtures/runtime/phase08/adversarial_honesty/"
            "near-persistence-canary-series.csv"
        ),
        {"false_holistic_canary", "near_persistence", "negative_case"},
    ),
    (
        "interpolation-bait-canary.yaml",
        "interpolation_bait",
        (
            "fixtures/runtime/phase08/adversarial_honesty/"
            "interpolation-bait-canary-series.csv"
        ),
        {"false_holistic_canary", "interpolation_bait", "negative_case"},
    ),
    (
        "row-index-leakage-canary.yaml",
        "row_index_leakage",
        (
            "fixtures/runtime/phase08/adversarial_honesty/"
            "row-index-leakage-canary-series.csv"
        ),
        {"false_holistic_canary", "row_index_leakage", "leakage"},
    ),
    (
        "sample-wide-closure-canary.yaml",
        "sample_wide_closure_bait",
        (
            "fixtures/runtime/phase08/adversarial_honesty/"
            "sample-wide-closure-canary-series.csv"
        ),
        {"false_holistic_canary", "sample_wide_closure", "negative_case"},
    ),
)


@pytest.mark.parametrize(
    ("manifest_name", "trap_class", "dataset_ref", "expected_tags"),
    CANARY_CASES,
)
def test_phase08_adversarial_canary_manifests_declare_blocking_abstention_contract(
    manifest_name: str,
    trap_class: str,
    dataset_ref: str,
    expected_tags: set[str],
) -> None:
    manifest = load_benchmark_task_manifest(TASK_ROOT / manifest_name)

    assert manifest.track_id == "adversarial_honesty"
    assert manifest.task_family == "false_holistic_canary"
    assert manifest.trap_class == trap_class
    assert manifest.expected_safe_outcome == "abstain"
    assert manifest.failure_severity == "blocking"
    assert manifest.fixture_spec_id == "euclid-certification-fixtures-v1"
    assert manifest.fixture_family_id == "phase08_adversarial_honesty"
    assert manifest.dataset_ref == dataset_ref
    assert set(manifest.adversarial_tags) >= expected_tags
    assert set(manifest.submitter_ids) == {
        "analytic_backend",
        "recursive_spectral_backend",
        "algorithmic_search_backend",
        "portfolio_orchestrator",
    }
    assert manifest.abstention_policy["expected_mode"] == "abstain_on_trap"
    assert "exact_sample_closure" in manifest.forbidden_shortcuts
    assert "posthoc_symbolic_synthesis" in manifest.forbidden_shortcuts


def test_phase08_adversarial_fixture_set_covers_all_false_holistic_canaries() -> None:
    payload = yaml.safe_load(FIXTURE_SET.read_text(encoding="utf-8"))

    assert payload["fixture_family_id"] == "phase08_adversarial_honesty"
    assert payload["fixture_spec_id"] == "euclid-certification-fixtures-v1"
    assert payload["series_count"] == len(CANARY_CASES)
    assert payload["entity_count"] == 1
    assert set(payload["diversity_axes"]) >= {
        "false_holistic_prevention",
        "leakage",
        "interpolation_bait",
    }
    assert set(payload["dataset_refs"]) == {dataset_ref for _, _, dataset_ref, _ in CANARY_CASES}


@pytest.mark.parametrize(
    ("manifest_name", "_trap_class", "dataset_ref", "_tags"),
    CANARY_CASES,
)
def test_phase08_adversarial_canary_fixtures_have_enough_rows_for_benchmark_splits(
    manifest_name: str,
    _trap_class: str,
    dataset_ref: str,
    _tags: set[str],
) -> None:
    manifest = load_benchmark_task_manifest(TASK_ROOT / manifest_name)
    fixture_path = ASSET_ROOT / dataset_ref

    with fixture_path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    assert len(rows) >= 12
    assert {row["series_id"] for row in rows} == {manifest.task_id}
    assert rows[0]["event_time"] < rows[-1]["event_time"]
    assert all(row["available_at"] for row in rows)


@pytest.mark.parametrize(("manifest_name", "_trap_class", "_dataset_ref", "_tags"), CANARY_CASES)
def test_phase08_adversarial_canaries_do_not_emit_local_winners(
    tmp_path: Path,
    manifest_name: str,
    _trap_class: str,
    _dataset_ref: str,
    _tags: set[str],
) -> None:
    try:
        result = profile_benchmark_task(
            manifest_path=TASK_ROOT / manifest_name,
            benchmark_root=tmp_path / "benchmarks",
            parallel_workers=2,
            resume=False,
        )
    except TypeError as exc:
        pytest.fail(
            f"{manifest_name} triggered a benchmark finalist normalization crash "
            f"instead of a safe abstention: {exc}"
        )

    task_result = json.loads(result.report_paths.task_result_path.read_text(encoding="utf-8"))
    submitter_by_id = {
        submitter_result.submitter_id: submitter_result
        for submitter_result in result.submitter_results
    }
    portfolio = submitter_by_id["portfolio_orchestrator"]

    assert task_result["track_id"] == "adversarial_honesty"
    assert task_result.get("local_winner_submitter_id") is None, (
        f"{manifest_name} emitted false top-line winner "
        f"{task_result.get('local_winner_submitter_id')!r}"
    )
    assert task_result.get("local_winner_candidate_id") is None, (
        f"{manifest_name} emitted false top-line candidate "
        f"{task_result.get('local_winner_candidate_id')!r}"
    )
    assert portfolio.status == "abstained", (
        f"{manifest_name} let the portfolio remain {portfolio.status!r}"
    )
    assert portfolio.selected_candidate_id is None, (
        f"{manifest_name} selected portfolio candidate "
        f"{portfolio.selected_candidate_id!r}"
    )
    assert portfolio.replay_contract["selected_submitter_id"] is None, (
        f"{manifest_name} retained selected_submitter_id="
        f"{portfolio.replay_contract['selected_submitter_id']!r}"
    )
    assert portfolio.replay_contract["selected_candidate_id"] is None, (
        f"{manifest_name} retained selected_candidate_id="
        f"{portfolio.replay_contract['selected_candidate_id']!r}"
    )
    assert all(
        submitter_by_id[submitter_id].status == "abstained"
        for submitter_id in (
            "analytic_backend",
            "recursive_spectral_backend",
            "algorithmic_search_backend",
        )
    ), f"{manifest_name} left at least one benchmark submitter selected"
