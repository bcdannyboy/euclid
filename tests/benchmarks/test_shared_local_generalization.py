from __future__ import annotations

from pathlib import Path

import yaml

import euclid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSITIVE_MANIFEST = (
    PROJECT_ROOT
    / "benchmarks/tasks/multi_entity/shared-local-generalization-positive.yaml"
)
NEGATIVE_MANIFEST = (
    PROJECT_ROOT
    / "benchmarks/tasks/multi_entity/shared-local-generalization-negative.yaml"
)


def test_positive_case_requires_shared_local_semantic_evidence(tmp_path: Path) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=POSITIVE_MANIFEST,
        benchmark_root=tmp_path / "shared-local-positive-benchmark",
        resume=False,
    )

    assert result.task_manifest.task_family == "shared_local_panel_generalization"
    assert tuple(run.submitter_id for run in result.submitter_results) == (
        "analytic_backend",
    )

    submitter = result.submitter_results[0]
    assert submitter.status == "selected"
    assert submitter.selected_candidate_id == "shared_local_panel_joint_optimizer"
    assert submitter.replay_contract["candidate_id"] == submitter.selected_candidate_id
    assert submitter.budget_consumption["canonical_program_count"] == 1
    assert submitter.budget_consumption["accepted_candidate_count"] == 1
    assert submitter.candidate_ledger[0].ledger_status == "accepted"
    assert result.report_paths.task_result_path.exists()
    assert result.report_paths.submitter_result_paths["analytic_backend"].exists()
    assert result.report_paths.replay_ref_paths["analytic_backend"].exists()


def test_negative_case_rejects_false_generalization(tmp_path: Path) -> None:
    result = euclid.profile_benchmark_task(
        manifest_path=NEGATIVE_MANIFEST,
        benchmark_root=tmp_path / "shared-local-negative-benchmark",
        resume=False,
    )

    submitter = result.submitter_results[0]
    assert submitter.status == "abstained"
    assert submitter.selected_candidate_id is None
    assert submitter.abstention_reason == "no_admissible_candidate"
    assert submitter.budget_consumption["canonical_program_count"] == 1
    assert submitter.budget_consumption["accepted_candidate_count"] == 0
    assert submitter.candidate_ledger[0].ledger_status == "rejected"
    assert submitter.candidate_ledger[0].reason_codes == ("bounds_invalid",)


def test_abstention_case_emits_expected_mode(tmp_path: Path) -> None:
    manifest_path = _write_abstention_manifest(tmp_path)

    result = euclid.profile_benchmark_task(
        manifest_path=manifest_path,
        benchmark_root=tmp_path / "shared-local-abstention-benchmark",
        resume=False,
    )

    submitter = result.submitter_results[0]
    assert result.task_manifest.abstention_mode == "structural_miss"
    assert submitter.status == "abstained"
    assert submitter.selected_candidate_id is None
    assert submitter.abstention_reason == "no_admissible_candidate"
    assert submitter.budget_consumption["canonical_program_count"] == 1
    assert submitter.candidate_ledger[0].ledger_status == "rejected"


def _write_abstention_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "shared-local-generalization-abstention.yaml"
    payload = yaml.safe_load(NEGATIVE_MANIFEST.read_text(encoding="utf-8"))
    payload["task_id"] = "shared_local_panel_abstention_demo"
    payload["adversarial_tags"] = ["declared_entity_panel", "abstention_case"]
    manifest_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path
