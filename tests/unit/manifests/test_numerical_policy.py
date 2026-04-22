from __future__ import annotations

from pathlib import Path

import yaml

from euclid.manifests.numerical import (
    DEFAULT_NUMERICAL_POLICY,
    derive_deterministic_seed,
    load_numerical_policy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_default_numerical_policy_defines_tolerances_optimizer_and_downgrades() -> None:
    policy = DEFAULT_NUMERICAL_POLICY

    assert policy.policy_id == "euclid_numerical_policy_v1"
    assert policy.absolute_tolerance > 0
    assert policy.relative_tolerance > 0
    assert policy.optimizer_max_iterations >= 1000
    assert policy.failure_thresholds["max_condition_number"] > 1
    assert "optimizer_nonconvergence" in policy.allowed_instability_downgrades


def test_deterministic_seed_derivation_is_stable_and_scoped() -> None:
    first = derive_deterministic_seed("P01-T03", "search")
    second = derive_deterministic_seed("P01-T03", "search")
    other = derive_deterministic_seed("P01-T03", "fit")

    assert first == second
    assert first != other
    assert 0 <= first < 2**32


def test_numerical_policy_schema_and_manifest_round_trip() -> None:
    schema_path = PROJECT_ROOT / "schemas/contracts/numerical-policy.yaml"
    runtime_schema_path = PROJECT_ROOT / "schemas/contracts/numerical-runtime.yaml"

    assert yaml.safe_load(schema_path.read_text(encoding="utf-8"))["kind"] == (
        "numerical_policy_contract"
    )
    assert yaml.safe_load(runtime_schema_path.read_text(encoding="utf-8"))["kind"] == (
        "numerical_runtime_contract"
    )
    loaded = load_numerical_policy(schema_path)

    assert loaded.policy_id == DEFAULT_NUMERICAL_POLICY.policy_id
    assert loaded.optimizer_max_iterations == (
        DEFAULT_NUMERICAL_POLICY.optimizer_max_iterations
    )
