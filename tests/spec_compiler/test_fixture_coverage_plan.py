from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_COVERAGE_PATH = REPO_ROOT / "fixtures/canonical/fixture-coverage.yaml"
REQUIRED_SOURCE_PATHS = {
    "docs/reference/examples/fixture-walkthroughs.md",
    "docs/reference/modeling-pipeline.md",
    "docs/reference/search-core.md",
    "docs/reference/contracts-manifests.md",
}
REQUIRED_SCENARIO_IDS = {
    "descriptive-publication",
    "predictive-publication-point",
    "probabilistic-publication-distribution",
    "probabilistic-publication-interval",
    "probabilistic-publication-quantile",
    "probabilistic-publication-event-probability",
    "mechanistic-publication",
    "shared-plus-local-publication",
    "algorithmic-discovery-publication",
    "abstention-only-publication",
    "inadmissible-candidate",
    "failed-predictive-promotion",
    "probabilistic-gate-failure",
    "mechanistic-evidence-insufficiency",
    "shared-plus-local-contract-failure",
    "illegal-time-access",
    "illegal-ref-shape",
    "unresolved-scope-violation",
}
def _load_yaml(path: Path) -> dict:
    assert path.is_file(), f"missing fixture coverage plan: {path.relative_to(REPO_ROOT).as_posix()}"
    return yaml.safe_load(path.read_text())


def test_fixture_coverage_plan_exists_and_is_rooted_in_canonical_sources() -> None:
    payload = _load_yaml(FIXTURE_COVERAGE_PATH)

    assert payload["version"] == 1
    assert payload["kind"] == "canonical_fixture_coverage_matrix"

    cited_sources = {
        source["path"]
        for sources in payload["source_inventory"].values()
        for source in sources
    }
    assert REQUIRED_SOURCE_PATHS <= cited_sources


def test_fixture_coverage_plan_enumerates_required_scenarios_with_explicit_contract_coverage() -> None:
    payload = _load_yaml(FIXTURE_COVERAGE_PATH)
    scenarios = payload["scenarios"]

    scenario_ids = {scenario["scenario_id"] for scenario in scenarios}
    assert REQUIRED_SCENARIO_IDS <= scenario_ids

    for scenario in scenarios:
        assert scenario["planned_fixture_bundle"].startswith("fixtures/canonical/")
        assert scenario["required_modules"], f"{scenario['scenario_id']} must list module coverage"
        assert scenario["required_schema_families"], f"{scenario['scenario_id']} must list schema families"
        assert scenario["required_evidence_classes"] is not None, (
            f"{scenario['scenario_id']} must declare evidence coverage explicitly"
        )
        assert scenario["validator_disposition"] in {"must_accept", "must_reject"}
