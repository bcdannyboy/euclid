from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SPEC_PATH = REPO_ROOT / "src/euclid/_assets/docs/implementation/certification-fixture-spec.yaml"
FIXTURE_COVERAGE_PATH = REPO_ROOT / "fixtures/canonical/fixture-coverage.yaml"
FIXTURE_ROOT = REPO_ROOT / "fixtures/runtime/full_vision_certification"
PACKAGED_FIXTURE_ROOT = (
    REPO_ROOT / "src/euclid/_assets/fixtures/runtime/full_vision_certification"
)
FULL_VISION_SUITE_PATH = REPO_ROOT / "benchmarks/suites/full-vision.yaml"
PLANTED_ANALYTIC_TASK_PATH = (
    REPO_ROOT / "benchmarks/tasks/rediscovery/planted-analytic-demo.yaml"
)
ALGORITHMIC_MEDIUM_TASK_PATH = (
    REPO_ROOT
    / "benchmarks/tasks/algorithmic_rediscovery/causal-last-observation-medium.yaml"
)
SEARCH_CLASS_MEDIUM_TASK_PATHS = (
    REPO_ROOT
    / "benchmarks/tasks/predictive_generalization/search-class-exact-enumeration-medium.yaml",
    REPO_ROOT
    / "benchmarks/tasks/predictive_generalization/search-class-bounded-medium.yaml",
    REPO_ROOT
    / "benchmarks/tasks/predictive_generalization/search-class-equality-saturation-medium.yaml",
    REPO_ROOT
    / "benchmarks/tasks/predictive_generalization/search-class-stochastic-medium.yaml",
)
FULL_VISION_POSITIVE_POINT_TASK_PATHS = (
    REPO_ROOT
    / "benchmarks/tasks/multi_entity/shared-local-generalization-medium-positive.yaml",
    REPO_ROOT / "benchmarks/tasks/mechanistic/mechanistic-lane-medium-positive.yaml",
    REPO_ROOT / "benchmarks/tasks/robustness/robustness-medium-positive.yaml",
)
ALLOWED_FIXTURE_SOURCE_KINDS = {
    "synthetic",
    "sanitized_live",
    "public_benchmark",
    "hand_authored_adversarial",
}
PHASE6_REQUIRED_FIXTURE_IDS = {
    "residual_history_backed_probabilistic_publication",
    "heuristic_gaussian_downgrade",
    "student_t_calibrated_distribution",
    "non_contiguous_horizon_panel",
    "additive_residual_multi_horizon_fit",
    "mdl_comparability_failure",
    "conformal_recalibration_no_leak_failure",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _csv_observed_values(path: Path) -> list[float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [float(row["observed_value"]) for row in csv.DictReader(handle)]


def _file_index(base_dir: Path) -> dict[str, Path]:
    return {
        path.relative_to(base_dir).as_posix(): path
        for path in base_dir.rglob("*")
        if path.is_file()
    }


def test_planted_analytic_fixture_matches_generator_and_clears_last_observation_margin(
) -> None:
    manifest = _load_yaml(PLANTED_ANALYTIC_TASK_PATH)
    generator_ref = str(manifest["target_structure_ref"])
    generator = _load_json(REPO_ROOT / generator_ref)
    values = _csv_observed_values(REPO_ROOT / manifest["dataset_ref"])
    parameters = generator["parameters"]

    intercept = float(parameters["intercept"])
    lag_coefficient = float(parameters["lag_coefficient"])
    assert abs(values[0] - float(parameters["initial_value"])) <= 1e-6
    for previous, observed in zip(values[:-1], values[1:], strict=True):
        expected = intercept + (lag_coefficient * previous)
        assert abs(observed - expected) <= 5e-6

    initial_window = int(manifest["split_policy"]["initial_window"])
    scored_indices = range(initial_window, len(values) - 2)
    last_observation_mae = fmean(
        abs(values[index] - values[index - 1]) for index in scored_indices
    )
    generator_mae = fmean(
        abs(values[index] - (intercept + (lag_coefficient * values[index - 1])))
        for index in scored_indices
    )
    assert (
        last_observation_mae - generator_mae
        >= float(manifest["practical_significance_margin"])
    )


def test_algorithmic_medium_fixture_targets_nontrivial_program_with_margin(
) -> None:
    manifest = _load_yaml(ALGORITHMIC_MEDIUM_TASK_PATH)
    target_ref = str(manifest["target_structure_ref"])
    assert target_ref == (
        "fixtures/runtime/full_vision_certification/"
        "algorithmic_rediscovery/algorithmic-search-generator.json"
    )
    generator = _load_json(REPO_ROOT / target_ref)
    values = _csv_observed_values(REPO_ROOT / manifest["dataset_ref"])

    assert generator["family"] == "algorithmic_running_half_average"
    assert generator["family"] != "algorithmic_last_observation"
    parameters = generator["parameters"]
    expected_values = [
        float(parameters["alternating_low"])
        if index % 2 == 0
        else float(parameters["alternating_high"])
        for index in range(int(parameters["row_count"]))
    ]
    assert values == expected_values

    initial_window = int(manifest["split_policy"]["initial_window"])
    scored_indices = range(initial_window, len(values) - 2)
    last_observation_mae = fmean(
        abs(values[index] - values[index - 1]) for index in scored_indices
    )
    target_mae = fmean(
        abs(values[index] - _running_half_average_forecast(values[:index]))
        for index in scored_indices
    )
    assert target_mae <= float(manifest["predictive_adequacy_floor"]["max_value"])
    assert (
        last_observation_mae - target_mae
        >= float(manifest["practical_significance_margin"])
    )


def test_search_class_medium_fixtures_reward_declared_lag_plus_two_program(
) -> None:
    for manifest_path in SEARCH_CLASS_MEDIUM_TASK_PATHS:
        manifest = _load_yaml(manifest_path)
        values = _csv_observed_values(REPO_ROOT / manifest["dataset_ref"])

        initial_window = int(manifest["split_policy"]["initial_window"])
        scored_indices = range(initial_window, len(values) - 1)
        last_observation_mae = fmean(
            abs(values[index + 1] - values[index]) for index in scored_indices
        )
        lag_plus_two_mae = fmean(
            abs(values[index + 1] - (values[index] + 2.0))
            for index in scored_indices
        )

        assert lag_plus_two_mae <= 1e-6, manifest_path
        assert (
            last_observation_mae - lag_plus_two_mae
            >= float(manifest["practical_significance_margin"])
        ), manifest_path


def test_positive_point_full_vision_tasks_declare_measured_comparator_baseline(
) -> None:
    for manifest_path in FULL_VISION_POSITIVE_POINT_TASK_PATHS:
        manifest = _load_yaml(manifest_path)
        baseline_entries = manifest["baseline_registry"]
        baseline_ids = {entry["baseline_id"] for entry in baseline_entries}

        assert "naive_last_value" in baseline_ids, manifest_path
        assert any(
            entry["baseline_id"] == "naive_last_value"
            and entry["manifest_path"]
            == "benchmarks/baselines/predictive_generalization/naive-last-value.yaml"
            for entry in baseline_entries
        ), manifest_path


def _running_half_average_forecast(training_values: list[float]) -> float:
    state = 0.0
    for observed in training_values:
        state = 0.5 * (state + observed)
    return state


def test_full_vision_runtime_fixtures_match_packaged_mirror_byte_for_byte() -> None:
    root_files = _file_index(FIXTURE_ROOT)
    packaged_files = _file_index(PACKAGED_FIXTURE_ROOT)
    failures: list[str] = []

    for relative_path in sorted(set(root_files) - set(packaged_files)):
        failures.append(f"packaged fixture missing: {relative_path}")
    for relative_path in sorted(set(packaged_files) - set(root_files)):
        failures.append(f"root fixture missing: {relative_path}")
    for relative_path in sorted(set(root_files) & set(packaged_files)):
        if root_files[relative_path].read_bytes() != packaged_files[
            relative_path
        ].read_bytes():
            failures.append(f"fixture mirror content differs: {relative_path}")

    assert not failures, "\n".join(failures)


def test_certification_fixture_sets_exist_for_all_major_surfaces() -> None:
    fixture_spec = _load_yaml(FIXTURE_SPEC_PATH)

    for family in fixture_spec["families"]:
        family_dir = FIXTURE_ROOT / family["family_id"]
        manifest_path = family_dir / "fixture-set.yaml"
        assert family_dir.is_dir()
        assert manifest_path.is_file()

        payload = _load_yaml(manifest_path)
        assert payload["fixture_family_id"] == family["family_id"]
        assert payload["fixture_spec_id"] == fixture_spec["fixture_spec_id"]
        assert len(payload["dataset_refs"]) >= family["minimum_dataset_count"]
        assert payload["series_count"] >= family["minimum_series_count"]
        assert payload["entity_count"] >= family["minimum_entity_count"]
        assert (
            len(payload["evidence_bundle_refs"])
            >= family["minimum_evidence_bundle_count"]
        )
        assert set(family["required_diversity_axes"]) <= set(payload["diversity_axes"])


def test_certification_fixture_sets_disclose_provenance_without_golden_evidence() -> None:
    fixture_spec = _load_yaml(FIXTURE_SPEC_PATH)

    for family in fixture_spec["families"]:
        payload = _load_yaml(FIXTURE_ROOT / family["family_id"] / "fixture-set.yaml")
        provenance = payload.get("fixture_provenance")
        assert isinstance(provenance, dict), family["family_id"]
        assert provenance.get("source_kind") in ALLOWED_FIXTURE_SOURCE_KINDS
        assert provenance.get("source_ref")
        assert (REPO_ROOT / provenance["source_ref"]).is_file()
        assert provenance.get("license")
        assert provenance.get("edge_cases")
        assert provenance.get("regression_reason")
        shortcut_disclosure = provenance.get("shortcut_disclosure")
        assert isinstance(shortcut_disclosure, dict), family["family_id"]
        assert shortcut_disclosure.get("fixture_only") is True
        assert "golden_acceptance" in set(
            shortcut_disclosure.get("forbidden_uses", [])
        )

        for ref in payload["dataset_refs"] + payload["evidence_bundle_refs"]:
            assert (REPO_ROOT / ref).is_file(), ref
        for ref in payload["evidence_bundle_refs"]:
            assert "golden" not in Path(ref).name, ref


def test_certification_tasks_reference_certification_fixtures() -> None:
    suite = _load_yaml(FULL_VISION_SUITE_PATH)

    for relative_path in suite["task_manifest_paths"]:
        task_manifest = _load_yaml(REPO_ROOT / relative_path)
        assert task_manifest["fixture_spec_id"] == "euclid-certification-fixtures-v1"
        assert task_manifest["dataset_ref"].startswith(
            "fixtures/runtime/full_vision_certification/"
        )
        assert task_manifest["generator_status"] != "unknown_real_world"
        assert task_manifest["fixture_family_id"]
        assert task_manifest["forbidden_shortcuts"]


def test_phase6_readiness_fixture_coverage_is_materialized() -> None:
    coverage = _load_yaml(FIXTURE_COVERAGE_PATH)

    entries = {
        entry["fixture_id"]: entry
        for entry in coverage["phase6_readiness_fixtures"]
    }
    assert PHASE6_REQUIRED_FIXTURE_IDS <= set(entries)

    for fixture_id in PHASE6_REQUIRED_FIXTURE_IDS:
        entry = entries[fixture_id]
        fixture_path = REPO_ROOT / entry["path"]
        payload = _load_yaml(fixture_path)

        assert payload["version"] == 1
        assert payload["kind"] == "phase6_readiness_fixture"
        assert payload["fixture_id"] == fixture_id
        assert set(entry["coverage_assertions"]) <= set(payload["coverage_assertions"])
        assert payload["expected_runtime_disposition"] in {
            "production_supported",
            "legacy_compatibility_downgrade",
            "validator_rejected",
        }
