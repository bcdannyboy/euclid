from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
SYSTEM_PATH = REPO_ROOT / "docs/reference/system.md"
MODELING_PIPELINE_PATH = REPO_ROOT / "docs/reference/modeling-pipeline.md"
CONTRACTS_MANIFESTS_PATH = REPO_ROOT / "docs/reference/contracts-manifests.md"
SCORING_CONTRACT_PATH = REPO_ROOT / "schemas/contracts/scoring.yaml"
CALIBRATION_CONTRACT_PATH = REPO_ROOT / "schemas/contracts/calibration.yaml"
GOVERNANCE_CONTRACT_PATH = REPO_ROOT / "schemas/contracts/evaluation-governance.yaml"

AUTHORITY_DOCS = {
    "README.md": {
        "required_strings": {
            "Publication is gated.",
            "calibration, robustness, and mechanistic evidence artifacts",
        },
    },
    "docs/reference/system.md": {
        "required_strings": {
            "Score, calibrate, and enrich those artifacts with robustness or mechanistic evidence.",
            "replayable experiments",
            "publication and release plane",
        },
    },
    "docs/reference/modeling-pipeline.md": {
        "required_strings": {
            "distribution",
            "interval",
            "quantile",
            "event_probability",
            "Non-point outputs require forecast-object-specific evaluation and, for predictive promotion, successful calibration.",
            "Cross-object comparisons are invalid by design.",
        },
    },
    "docs/reference/contracts-manifests.md": {
        "required_strings": {
            "point score and calibration results",
            "scorecards, claims, and abstentions",
            "run results and publication records",
        },
    },
}

BANNED_POINT_ONLY_PHRASES = {
    "public forecast object type is `point`",
    "retained public forecast object type is `point`",
    "the public forecast object type is `point` only",
    "retained public object type is `point`",
}


def _load_yaml(path: Path) -> dict:
    assert path.is_file(), f"missing required file: {path.relative_to(REPO_ROOT).as_posix()}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_probabilistic_reference_docs_admit_the_full_forecast_object_set_without_point_only_regressions() -> None:
    for relative_path, expectations in AUTHORITY_DOCS.items():
        body = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        lowered = body.lower()

        for required_string in expectations["required_strings"]:
            assert (
                required_string.lower() in lowered
            ), f"{relative_path} missing required content: {required_string}"

        for phrase in BANNED_POINT_ONLY_PHRASES:
            assert phrase not in lowered, (
                f"{relative_path} regressed to point-only probabilistic scope"
            )


def test_probabilistic_reference_docs_match_scoring_calibration_and_gate_contracts() -> None:
    scoring = _load_yaml(SCORING_CONTRACT_PATH)
    calibration = _load_yaml(CALIBRATION_CONTRACT_PATH)
    governance = _load_yaml(GOVERNANCE_CONTRACT_PATH)

    modeling_doc = MODELING_PIPELINE_PATH.read_text(encoding="utf-8")
    system_doc = SYSTEM_PATH.read_text(encoding="utf-8")
    contracts_doc = CONTRACTS_MANIFESTS_PATH.read_text(encoding="utf-8")
    readme_doc = README_PATH.read_text(encoding="utf-8")

    non_point_types = {
        "distribution",
        "interval",
        "quantile",
        "event_probability",
    }

    assert scoring["global_rules"]["cross_object_comparison"] == (
        "forbidden_without_explicit_reduction_contract"
    )
    assert "Cross-object comparisons are invalid by design." in modeling_doc

    assert calibration["global_rules"]["probabilistic_objects_require_type_matched_calibration"] is True
    assert calibration["global_rules"]["calibration_failure_effect"] == (
        "blocks_probabilistic_predictive_publication"
    )
    assert "successful calibration" in modeling_doc
    assert "Publication is gated." in readme_doc

    gate_entries = {
        entry["policy_id"]: entry for entry in governance["predictive_gate_policies"]
    }
    assert gate_entries["probabilistic_predictive_gate"]["requires_calibration_pass"] is True
    assert set(gate_entries["probabilistic_predictive_gate"]["allowed_forecast_object_types"]) == non_point_types
    assert "calibration" in system_doc.lower()

    for contract in scoring["contracts"]:
        forecast_object_type = contract["forecast_object_type"]
        if forecast_object_type != "point":
            assert forecast_object_type in modeling_doc
        assert contract["allowed_primary_scores"], (
            f"{forecast_object_type} must declare at least one primary score"
        )

    for contract in calibration["contracts"]:
        forecast_object_type = contract["forecast_object_type"]
        if forecast_object_type in non_point_types:
            assert contract["calibration_mode"] == "required"
            assert forecast_object_type in modeling_doc
            assert contract["gate_effect"] == "required_for_probabilistic_publication"

    assert "point score and calibration results" in contracts_doc
