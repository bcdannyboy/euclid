from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
SYSTEM_PATH = REPO_ROOT / "docs/reference/system.md"
MODELING_PIPELINE_PATH = REPO_ROOT / "docs/reference/modeling-pipeline.md"
CONTRACTS_MANIFESTS_PATH = REPO_ROOT / "docs/reference/contracts-manifests.md"
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"
ENUM_REGISTRY_PATH = REPO_ROOT / "schemas/contracts/enum-registry.yaml"

LIFECYCLE_SCHEMAS = {
    "schemas/contracts/run-lifecycle.yaml": {
        "kind": "run_lifecycle",
        "required_states": {
            "run_declared",
            "observations_ingested",
            "dataset_snapshot_frozen",
            "search_contract_frozen",
            "shared_plus_local_aggregated",
            "mechanistic_evidence_materialized",
            "claims_resolved",
            "candidate_publication_completed",
            "abstention_publication_completed",
        },
        "required_transitions": {
            "ingestion_complete",
            "search_contract_freeze_complete",
            "shared_plus_local_aggregation_complete",
            "mechanistic_evidence_complete",
            "candidate_publication_complete",
            "abstention_publication_complete",
        },
    },
    "schemas/contracts/candidate-state-machine.yaml": {
        "kind": "candidate_state_machine",
        "required_states": {
            "proposed",
            "fit",
            "frontier_retained",
            "rejected_pre_freeze",
            "shared_plus_local_aggregated",
            "frozen",
            "evaluated_point",
            "evaluated_probabilistic",
            "robustness_checked",
            "mechanistic_evaluated",
            "claim_emitted",
            "abstained",
            "replay_verified",
            "replay_failed",
            "published",
        },
        "required_transitions": {
            "candidate_fit_succeeds",
            "candidate_rejected_before_freeze",
            "shared_local_state_bound",
            "candidate_freeze_complete",
            "probabilistic_evaluation_complete",
            "mechanistic_evaluation_complete",
            "claim_emission_complete",
            "abstention_emission_complete",
            "replay_verification_failed",
            "catalog_publication_complete",
        },
    },
    "schemas/contracts/publication-lifecycle.yaml": {
        "kind": "publication_lifecycle",
        "required_states": {
            "publication_requested",
            "meaning_bound",
            "replay_bundle_assembled",
            "replay_verified",
            "replay_failed",
            "run_result_assembled",
            "candidate_publication_selected",
            "abstention_only_publication_selected",
            "publication_record_written",
            "catalog_projected",
            "publication_completed",
            "publication_blocked",
        },
        "required_transitions": {
            "meaning_objects_bound",
            "replay_bundle_complete",
            "replay_verification_succeeds",
            "replay_verification_fails",
            "candidate_publication_mode_selected",
            "abstention_publication_mode_selected",
            "publication_record_persisted",
            "catalog_projection_complete",
            "publication_blocked_on_replay_failure",
        },
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _parse_front_matter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    assert match, f"{path.relative_to(REPO_ROOT).as_posix()} must start with YAML front matter"
    return yaml.safe_load(match.group(1)), match.group(2)


def test_reference_docs_describe_runtime_lifecycle_without_deleted_architecture_docs() -> None:
    system_front_matter, system_body = _parse_front_matter(SYSTEM_PATH)
    modeling_front_matter, modeling_body = _parse_front_matter(MODELING_PIPELINE_PATH)
    contracts_front_matter, contracts_body = _parse_front_matter(CONTRACTS_MANIFESTS_PATH)
    readme_body = README_PATH.read_text(encoding="utf-8")

    assert system_front_matter["title"] == "System"
    assert "## End-to-end lifecycle" in system_body
    for required_string in {
        "Load a run request and resolve the packaged assets and formal specs it needs.",
        "Build and verify a reproducibility bundle.",
        "Persist a run result, publication record, and optional readiness or benchmark evidence.",
    }:
        assert required_string in system_body

    assert modeling_front_matter["title"] == "Modeling Pipeline"
    for required_string in {
        "## Stage map",
        "### 1. Ingestion",
        "### 9. Claims, replay, and publication",
        "`modules/replay.py` builds reproducibility bundles and verifies replay.",
        "`modules/catalog_publishing.py` assembles run results, publication records, and local catalog entries, subject to replay and readiness constraints.",
    }:
        assert required_string in modeling_body

    assert contracts_front_matter["title"] == "Contracts And Manifests"
    for required_string in {
        "schemas/contracts`: module, schema, ref, lifecycle, and domain rules",
        "reproducibility bundles",
        "run results and publication records",
        "## Registry and lineage",
    }:
        assert required_string in contracts_body

    for required_string in {
        "abstentions",
        "Publication is gated.",
    }:
        assert required_string in readme_body


def test_lifecycle_schema_artifacts_cover_run_candidate_and_publication_paths() -> None:
    for relative_path, expectations in LIFECYCLE_SCHEMAS.items():
        path = REPO_ROOT / relative_path
        assert path.is_file(), f"missing lifecycle schema artifact: {relative_path}"

        payload = _load_yaml(path)
        assert payload["version"] == 1
        assert payload["kind"] == expectations["kind"]

        owners = payload["owners"]
        assert isinstance(owners, list) and owners, f"{relative_path} must declare owners"
        owner_ids = {owner["id"] for owner in owners}

        states = payload["states"]
        transitions = payload["transitions"]
        state_ids = {state["state_id"] for state in states}
        transition_ids = {transition["transition_id"] for transition in transitions}

        assert expectations["required_states"] <= state_ids, (
            f"{relative_path} is missing required states"
        )
        assert expectations["required_transitions"] <= transition_ids, (
            f"{relative_path} is missing required transitions"
        )

        for state in states:
            assert state["owner_ref"] in owner_ids, (
                f"unknown owner_ref in {relative_path}: {state['state_id']}"
            )

        for transition in transitions:
            assert transition["owner_ref"] in owner_ids, (
                f"unknown owner_ref in {relative_path}: {transition['transition_id']}"
            )
            assert transition["from"] in state_ids, (
                f"unknown from state in {relative_path}: {transition['transition_id']}"
            )
            assert transition["to"] in state_ids, (
                f"unknown to state in {relative_path}: {transition['transition_id']}"
            )


def test_lifecycle_artifacts_are_linked_from_live_reference_spine_and_enum_registry() -> None:
    readme_doc = README_PATH.read_text(encoding="utf-8")
    source_map = _load_yaml(SOURCE_MAP_PATH)
    enum_registry = _load_yaml(ENUM_REGISTRY_PATH)

    for required_path in (
        "docs/reference/system.md",
        "docs/reference/modeling-pipeline.md",
        "docs/reference/contracts-manifests.md",
    ):
        assert required_path in readme_doc

    assert source_map["reference_workspace"]["docs_root"] == "docs/reference"

    enum_names = {entry["enum_name"] for entry in enum_registry["enums"]}
    assert {
        "run_lifecycle_states",
        "candidate_lifecycle_states",
        "publication_lifecycle_states",
    } <= enum_names
