from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml

from euclid.benchmarks import load_benchmark_task_manifest
from euclid.benchmarks.manifests import (
    PredictiveGeneralizationTaskManifest,
    RediscoveryTaskManifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "src" / "euclid" / "_assets"
TASK_ROOT = ASSET_ROOT / "benchmarks" / "tasks"
FIXTURE_ROOT = ASSET_ROOT / "fixtures" / "runtime" / "phase08" / "planted_law_suite"
FIXTURE_SET_PATH = FIXTURE_ROOT / "fixture-set.yaml"

REDISCOVERY_CASES = {
    "linear": TASK_ROOT / "rediscovery" / "planted-linear-exact-phase08.yaml",
    "affine_lag": TASK_ROOT / "rediscovery" / "planted-affine-lag-exact-phase08.yaml",
    "damped_harmonic": (
        TASK_ROOT / "rediscovery" / "planted-damped-harmonic-exact-phase08.yaml"
    ),
    "additive_composition": (
        TASK_ROOT
        / "rediscovery"
        / "planted-additive-composition-exact-phase08.yaml"
    ),
}
PREDICTIVE_CASES = {
    "linear": (
        TASK_ROOT / "predictive_generalization" / "planted-linear-noisy-phase08.yaml"
    ),
    "affine_lag": (
        TASK_ROOT
        / "predictive_generalization"
        / "planted-affine-lag-noisy-phase08.yaml"
    ),
    "damped_harmonic": (
        TASK_ROOT
        / "predictive_generalization"
        / "planted-damped-harmonic-noisy-phase08.yaml"
    ),
    "additive_composition": (
        TASK_ROOT
        / "predictive_generalization"
        / "planted-additive-composition-noisy-phase08.yaml"
    ),
}


def test_phase08_rediscovery_manifests_bind_exact_recovery_contracts() -> None:
    for family, manifest_path in REDISCOVERY_CASES.items():
        manifest = load_benchmark_task_manifest(manifest_path)
        raw_manifest = _load_yaml(manifest_path)
        contract = raw_manifest["phase08_planted_contract"]
        generator_spec = _load_generator_spec(
            PROJECT_ROOT / contract["generator_ref"],
        )

        assert isinstance(manifest, RediscoveryTaskManifest)
        assert manifest.generator_status == "known_generator"
        assert contract["family"] == family
        assert contract["variant"] == "exact"
        assert manifest.target_structure_ref == contract["generator_ref"]
        assert {"phase08_planted_law", "exact_case"} <= set(manifest.regime_tags)
        assert manifest.predictive_adequacy_floor["max_value"] <= 1.0e-6
        assert (
            manifest.equivalence_policy["equivalence_class"]
            == generator_spec["equivalence_class"]
        )
        assert (PROJECT_ROOT / manifest.dataset_ref).is_file()


def test_phase08_predictive_manifests_bind_noisy_recovery_contracts() -> None:
    for family, manifest_path in PREDICTIVE_CASES.items():
        manifest = load_benchmark_task_manifest(manifest_path)
        raw_manifest = _load_yaml(manifest_path)
        contract = raw_manifest["phase08_planted_contract"]
        generator_spec = _load_generator_spec(
            PROJECT_ROOT / contract["generator_ref"],
        )

        assert isinstance(manifest, PredictiveGeneralizationTaskManifest)
        assert manifest.generator_status == "near_equivalent_generator"
        assert contract["family"] == family
        assert contract["variant"] == "noisy"
        assert {"phase08_planted_law", "noisy_variant"} <= set(manifest.regime_tags)
        assert manifest.origin_policy["min_origins"] >= 4
        assert manifest.horizon_policy["horizons"] == [1]
        if family == "additive_composition":
            assert manifest.composition_operators == ("additive_residual",)
        else:
            assert manifest.composition_operators == ()
        assert (
            contract["expected_recovery"]["max_holdout_mae"]
            == generator_spec["variants"]["noisy"]["max_holdout_mae"]
        )
        assert (PROJECT_ROOT / manifest.dataset_ref).is_file()


def test_phase08_planted_series_follow_declared_family_thresholds() -> None:
    manifest_paths = tuple(REDISCOVERY_CASES.values()) + tuple(PREDICTIVE_CASES.values())

    for manifest_path in manifest_paths:
        raw_manifest = _load_yaml(manifest_path)
        contract = raw_manifest["phase08_planted_contract"]
        generator_spec = _load_generator_spec(PROJECT_ROOT / contract["generator_ref"])
        series = _load_series(PROJECT_ROOT / raw_manifest["dataset_ref"])
        expected = _evaluate_generator(generator_spec, len(series))
        deltas = [abs(observed - predicted) for observed, predicted in zip(series, expected)]

        if contract["variant"] == "exact":
            assert max(deltas) <= generator_spec["variants"]["exact"]["max_pointwise_abs_error"]
            continue

        noisy_spec = generator_spec["variants"]["noisy"]
        tolerance = 1.0e-4
        assert min(deltas) >= noisy_spec["min_pointwise_abs_error"] - tolerance
        assert max(deltas) <= noisy_spec["max_pointwise_abs_error"] + tolerance
        assert sum(deltas) / len(deltas) <= noisy_spec["max_holdout_mae"] + tolerance
        assert noisy_spec["forbid_exact_equivalence"] is True


def test_phase08_fixture_bundle_covers_all_planted_series_and_generator_specs() -> None:
    payload = _load_yaml(FIXTURE_SET_PATH)

    expected_dataset_refs = {
        "src/euclid/_assets/fixtures/runtime/phase08/planted_law_suite/"
        "linear-exact-series.csv",
        "src/euclid/_assets/fixtures/runtime/phase08/planted_law_suite/"
        "linear-noisy-series.csv",
        "src/euclid/_assets/fixtures/runtime/phase08/planted_law_suite/"
        "affine-lag-exact-series.csv",
        "src/euclid/_assets/fixtures/runtime/phase08/planted_law_suite/"
        "affine-lag-noisy-series.csv",
        "src/euclid/_assets/fixtures/runtime/phase08/planted_law_suite/"
        "damped-harmonic-exact-series.csv",
        "src/euclid/_assets/fixtures/runtime/phase08/planted_law_suite/"
        "damped-harmonic-noisy-series.csv",
        "src/euclid/_assets/fixtures/runtime/phase08/planted_law_suite/"
        "additive-composition-exact-series.csv",
        "src/euclid/_assets/fixtures/runtime/phase08/planted_law_suite/"
        "additive-composition-noisy-series.csv",
    }
    expected_generator_refs = {
        "src/euclid/_assets/fixtures/runtime/phase08/planted_law_suite/generators/"
        "linear.json",
        "src/euclid/_assets/fixtures/runtime/phase08/planted_law_suite/generators/"
        "affine-lag.json",
        "src/euclid/_assets/fixtures/runtime/phase08/planted_law_suite/generators/"
        "damped-harmonic.json",
        "src/euclid/_assets/fixtures/runtime/phase08/planted_law_suite/generators/"
        "additive-composition.json",
    }

    assert payload["fixture_family_id"] == "phase08_planted_law_suite"
    assert set(payload["dataset_refs"]) == expected_dataset_refs
    assert expected_generator_refs <= set(payload["evidence_bundle_refs"])
    assert payload["series_count"] == 8
    assert payload["diversity_axes"] == [
        "law_family",
        "generator_variant",
        "benchmark_track",
    ]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_generator_spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_series(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return [float(row["observed_value"]) for row in rows]


def _evaluate_generator(spec: dict[str, Any], length: int) -> list[float]:
    family = spec["family"]
    parameters = spec["parameters"]
    if family == "linear":
        return [
            parameters["intercept"] + parameters["slope"] * index
            for index in range(length)
        ]
    if family == "affine_lag":
        values = [parameters["initial_value"]]
        for _ in range(1, length):
            values.append(
                parameters["intercept"]
                + parameters["lag_coefficient"] * values[-1]
            )
        return values
    if family == "damped_harmonic":
        return [
            parameters["offset"]
            + parameters["amplitude"]
            * math.exp(-parameters["decay_rate"] * index)
            * math.cos(
                parameters["angular_frequency"] * index
                + parameters["phase_shift"]
            )
            for index in range(length)
        ]
    if family == "additive_composition":
        return [
            parameters["intercept"]
            + parameters["slope"] * index
            + parameters["seasonal_amplitude"]
            * math.sin(
                parameters["seasonal_frequency"] * index
                + parameters["phase_shift"]
            )
            for index in range(length)
        ]
    raise AssertionError(f"unsupported family: {family}")
