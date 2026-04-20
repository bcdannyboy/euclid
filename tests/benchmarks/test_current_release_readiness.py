from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import euclid
import yaml
from euclid.readiness import judge_benchmark_suite_readiness

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_RELEASE_SUITE = PROJECT_ROOT / "benchmarks/suites/current-release.yaml"


def test_current_release_suite_is_truthfully_narrow(
    tmp_path: Path,
) -> None:
    suite_result = euclid.profile_benchmark_suite(
        manifest_path=CURRENT_RELEASE_SUITE,
        benchmark_root=tmp_path / "current-release-suite",
        resume=False,
    )

    judgment = judge_benchmark_suite_readiness(
        judgment_id="current_release_readiness",
        suite_result=suite_result,
    )

    assert judgment.final_verdict == "ready"
    assert judgment.catalog_scope == "public"
    assert {gate.gate_id for gate in judgment.gate_results} >= {
        "suite.current_release",
        "track.rediscovery",
        "track.predictive_generalization",
        "track.adversarial_honesty",
        "surface.retained_core_release",
        "surface.algorithmic_backend",
        "surface.shared_plus_local_decomposition",
        "surface.mechanistic_lane",
    }


def test_current_release_suite_is_proper_subset_of_full_vision() -> None:
    current_release = yaml.safe_load(CURRENT_RELEASE_SUITE.read_text(encoding="utf-8"))
    full_vision = yaml.safe_load(
        (
            PROJECT_ROOT / "benchmarks/suites/full-vision.yaml"
        ).read_text(encoding="utf-8")
    )

    current_surfaces = {
        requirement["surface_id"]
        for requirement in current_release["surface_requirements"]
    }
    full_surfaces = {
        requirement["surface_id"]
        for requirement in full_vision["surface_requirements"]
    }

    assert current_surfaces < full_surfaces
    assert "portfolio_orchestration" not in current_surfaces
    assert current_release["fixture_spec_id"] == "euclid-certification-fixtures-v1"


def test_current_release_readiness_blocks_when_a_surface_lacks_replay_evidence(
    tmp_path: Path,
) -> None:
    suite_result = euclid.profile_benchmark_suite(
        manifest_path=CURRENT_RELEASE_SUITE,
        benchmark_root=tmp_path / "current-release-suite",
        resume=False,
    )
    broken_surface = replace(
        next(
            surface
            for surface in suite_result.surface_statuses
            if surface.surface_id == "mechanistic_lane"
        ),
        replay_status="failed",
    )
    judgment = judge_benchmark_suite_readiness(
        judgment_id="current_release_readiness_missing_replay",
        suite_result=replace(
            suite_result,
            surface_statuses=tuple(
                broken_surface if surface.surface_id == "mechanistic_lane" else surface
                for surface in suite_result.surface_statuses
            ),
        ),
    )

    assert judgment.final_verdict == "blocked"
    assert "surface.mechanistic_lane_failed" in judgment.reason_codes


def test_current_release_readiness_blocks_when_suite_summary_omits_semantic_evidence(
    tmp_path: Path,
) -> None:
    suite_result = euclid.profile_benchmark_suite(
        manifest_path=CURRENT_RELEASE_SUITE,
        benchmark_root=tmp_path / "current-release-suite",
        resume=False,
    )
    summary = json.loads(suite_result.summary_path.read_text(encoding="utf-8"))
    summary["task_results"][0].pop("replay_verification", None)
    summary["surface_statuses"][0]["evidence"].pop("score_laws", None)
    suite_result.summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    judgment = judge_benchmark_suite_readiness(
        judgment_id="current_release_readiness_missing_semantics",
        suite_result=suite_result,
    )

    assert judgment.final_verdict == "blocked"
    suite_gate = next(
        gate
        for gate in judgment.gate_results
        if gate.gate_id == "suite.current_release"
    )
    retained_surface_gate = next(
        gate
        for gate in judgment.gate_results
        if gate.gate_id == "surface.retained_core_release"
    )
    assert suite_gate.status == "failed"
    assert suite_gate.evidence["missing_task_semantic_fields"] == {
        "planted_analytic_demo": ["replay_verification"]
    }
    assert retained_surface_gate.status == "failed"
    assert retained_surface_gate.evidence["missing_surface_diagnostics"] == [
        "score_laws"
    ]
