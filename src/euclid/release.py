from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import yaml

from euclid._version import (
    PACKAGE_VERSION,
    RELEASE_CERTIFICATION_TEST_TARGETS,
    RELEASE_TARGET_VERSION,
    RELEASE_WORKFLOW_ID,
)
from euclid.benchmarks import profile_benchmark_suite, profile_benchmark_task
from euclid.contracts import load_contract_catalog
from euclid.operator_runtime._compat_runtime import profile_operator_run
from euclid.operator_runtime.resources import (
    resolve_asset_root,
    resolve_checkout_root,
    resolve_example_path,
    resolve_notebook_path,
)
from euclid.operator_runtime.resources import (
    resolve_workspace_root as resolve_runtime_workspace_root,
)
from euclid.operator_runtime.run import run_operator
from euclid.performance import (
    PerformanceBudget,
    SuitePerformanceBudget,
    collect_performance_suite,
    evaluate_performance_budget,
    evaluate_suite_performance_budget,
    load_performance_telemetry,
)
from euclid.readiness import (
    ReadinessGateResult,
    ReadinessJudgment,
    gate_results_by_id,
    judge_benchmark_suite_readiness,
    judge_readiness,
    merge_readiness_judgments,
)
from euclid.runtime.profiling import (
    capture_operator_runtime_snapshot,
    compare_runtime_determinism,
)

_EXPECTED_NOTEBOOK_CASE_IDS = (
    "distribution",
    "event_probability",
    "interval",
    "quantile",
)
_PACKAGED_READINESS_POLICY_DIRECTORY = "schemas/readiness"
_PACKAGED_READINESS_SCHEMA_PATH = "schemas/readiness/euclid-readiness.yaml"
_PACKAGED_COMPLETION_REPORT_SCHEMA_PATH = (
    "schemas/readiness/completion-report.schema.yaml"
)
_COMPLETION_REGRESSION_POLICY_PATH = (
    "schemas/readiness/completion-regression-policy.yaml"
)
_AUTHORITY_SNAPSHOT_PATH = "docs/implementation/authority-snapshot.yaml"
_CLOSURE_MAP_PATH = "docs/implementation/euclid-closure-map.yaml"
_TRACEABILITY_PATH = "docs/implementation/subtask-test-traceability.yaml"
_CERTIFICATION_FIXTURE_SPEC_PATH = (
    "docs/implementation/certification-fixture-spec.yaml"
)
_CERTIFICATION_EVIDENCE_CONTRACT_PATH = (
    "docs/implementation/certification-evidence-contract.yaml"
)
_CERTIFICATION_COMMAND_CONTRACT_PATH = (
    "docs/implementation/certification-command-contract.yaml"
)
_REPO_TEST_MATRIX_REPORT_PATH = "build/reports/repo_test_matrix.json"
_VERIFY_COMPLETION_REPORT_PATH = "build/reports/verify-completion.json"
_RESEARCH_READINESS_REPORT_PATH = "build/reports/research-readiness.json"
_CURRENT_RELEASE_SUITE_EVIDENCE_PATH = "build/reports/current_release_suite_evidence.json"
_FULL_VISION_SUITE_EVIDENCE_PATH = "build/reports/full_vision_suite_evidence.json"
_FULL_VISION_OPERATOR_RUN_EVIDENCE_PATH = (
    "build/reports/full_vision_operator_run_evidence.json"
)
_FULL_VISION_OPERATOR_REPLAY_EVIDENCE_PATH = (
    "build/reports/full_vision_operator_replay_evidence.json"
)
_SUMMARY_TOKEN_PATTERN = re.compile(r"(?P<count>\\d+) (?P<label>[a-zA-Z_]+)")
_REQUIREMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+")
_SKIP_CLEAN_INSTALL_CERTIFICATION_ENV = "EUCLID_SKIP_CLEAN_INSTALL_CERTIFICATION"
_CLEAN_INSTALL_REQUIRED_SURFACE_IDS = (
    "release_status",
    "operator_run",
    "operator_replay",
    "benchmark_execution",
    "determinism_same_seed",
    "performance_runtime_smoke",
    "packaged_notebook_smoke",
)
_SUBMITTER_REDUCER_FAMILY_IDS = {
    "analytic_backend": ("analytic",),
    "recursive_spectral_backend": ("recursive", "spectral"),
    "algorithmic_search_backend": ("algorithmic",),
    "portfolio_orchestrator": ("analytic", "recursive", "spectral", "algorithmic"),
}
_SUITE_GENERIC_LIFECYCLE_ARTIFACT_IDS = frozenset(
    {
        "observation_record",
        "dataset_snapshot",
        "dataset_lineage_snapshot",
        "causal_availability_window",
        "time_safety_audit",
        "feature_view",
        "feature_materialization_report",
        "evaluation_plan",
        "split_manifest",
        "search_plan",
        "canonicalization_policy",
        "search_budget",
        "comparison_universe",
        "predictive_gate_policy",
        "candidate_spec",
        "reducer_artifact",
        "frontier",
        "rejected_diagnostics",
        "freeze_event",
        "frozen_shortlist",
        "evaluation_event_log",
        "prediction_artifact",
        "scorecard",
        "lifecycle_gate_decision",
    }
)
_SUITE_SURFACE_LIFECYCLE_ARTIFACT_IDS: dict[str, frozenset[str]] = {
    "external_evidence_ingestion": frozenset(
        {"external_evidence_bundle", "source_digest"}
    ),
    "mechanistic_lane": frozenset(
        {"mechanistic_evidence_dossier", "evidence_independence_attestation"}
    ),
    "robustness_lane": frozenset({"robustness_report", "abstention"}),
    "shared_plus_local_decomposition": frozenset(
        {
            "shared_plus_local_decomposition_policy",
            "shared_plus_local_aggregation_table",
        }
    ),
}
_OPERATOR_REGISTRY_LIFECYCLE_ARTIFACT_IDS: dict[str, frozenset[str]] = {
    "calibration_result_manifest@1.0.0": frozenset({"calibration_result"}),
    "candidate_spec@1.0.0": frozenset({"candidate_spec"}),
    "canonicalization_policy_manifest@1.0.0": frozenset({"canonicalization_policy"}),
    "claim_card_manifest@1.1.0": frozenset({"claim_card", "claim_card_or_abstention"}),
    "comparison_universe_manifest@1.0.0": frozenset({"comparison_universe"}),
    "dataset_snapshot_manifest@1.0.0": frozenset({"dataset_snapshot"}),
    "evaluation_event_log_manifest@1.0.0": frozenset({"evaluation_event_log"}),
    "evaluation_plan_manifest@1.1.0": frozenset({"evaluation_plan"}),
    "feature_view_manifest@1.0.0": frozenset({"feature_view"}),
    "freeze_event_manifest@1.0.0": frozenset({"freeze_event"}),
    "frontier_manifest@1.0.0": frozenset({"frontier"}),
    "frozen_shortlist_manifest@1.0.0": frozenset({"frozen_shortlist"}),
    "observation_record@1.0.0": frozenset({"observation_record"}),
    "prediction_artifact_manifest@1.1.0": frozenset({"prediction_artifact"}),
    "predictive_gate_policy_manifest@1.1.0": frozenset({"predictive_gate_policy"}),
    "probabilistic_score_result_manifest@1.0.0": frozenset({"point_score_result"}),
    "publication_record_manifest@1.1.0": frozenset(
        {"publication_record", "catalog_entry"}
    ),
    "readiness_judgment_manifest@1.0.0": frozenset({"lifecycle_gate_decision"}),
    "reducer_artifact_manifest@1.0.0": frozenset({"reducer_artifact"}),
    "rejected_diagnostics_manifest@1.0.0": frozenset({"rejected_diagnostics"}),
    "reproducibility_bundle_manifest@1.0.0": frozenset({"reproducibility_bundle"}),
    "robustness_report_manifest@1.1.0": frozenset({"robustness_report"}),
    "run_result_manifest@1.1.0": frozenset({"run_result"}),
    "scorecard_manifest@1.1.0": frozenset({"scorecard"}),
    "search_ledger_manifest@1.0.0": frozenset({"search_budget"}),
    "search_plan_manifest@1.0.0": frozenset({"search_plan"}),
    "time_safety_audit_manifest@1.0.0": frozenset({"time_safety_audit"}),
}
_PACKAGING_LIFECYCLE_ARTIFACT_SURFACES = {
    "catalog_entry": "operator_run",
    "publication_record": "operator_run",
    "run_result": "operator_run",
}


@dataclass(frozen=True)
class ReleaseStatus:
    project_root: Path
    current_version: str
    target_version: str
    target_ready: bool
    blocked_reason: str
    catalog_scope: str
    policy_judgments: Mapping[str, ReadinessJudgment]
    shipped_releasable_judgment: ReadinessJudgment
    readiness_judgment: ReadinessJudgment


@dataclass(frozen=True)
class ContractCatalogValidationResult:
    project_root: Path
    schema_count: int
    module_count: int
    enum_count: int
    contract_document_count: int


@dataclass(frozen=True)
class BenchmarkSmokeCaseResult:
    track_id: str
    task_id: str
    task_result_path: Path
    report_path: Path
    telemetry_path: Path
    task_status: str
    local_winner_submitter_id: str | None
    within_budget: bool
    declared_wall_clock_seconds: float
    observed_wall_time_seconds: float


@dataclass(frozen=True)
class BenchmarkSmokeResult:
    project_root: Path
    benchmark_root: Path
    cases: tuple[BenchmarkSmokeCaseResult, ...]


@dataclass(frozen=True)
class NotebookSmokeResult:
    project_root: Path
    notebook_path: Path
    output_root: Path
    summary_path: Path
    probabilistic_case_ids: tuple[str, ...]
    publication_mode: str
    catalog_entries: int


@dataclass(frozen=True)
class ReleaseDeterminismSmokeResult:
    project_root: Path
    output_root: Path
    summary_path: Path
    identical: bool
    changed_artifact_roles: tuple[str, ...]
    changed_seed_scopes: tuple[str, ...]
    changed_manifest_refs: tuple[str, ...]


@dataclass(frozen=True)
class ReleasePerformanceSmokeResult:
    project_root: Path
    output_root: Path
    summary_path: Path
    operator_budget_passed: bool
    benchmark_budget_passed: bool
    suite_budget_passed: bool
    total_wall_time_seconds: float
    max_wall_time_seconds: float


@dataclass(frozen=True)
class ReleaseCandidateWorkflow:
    workflow_id: str
    package_version: str
    target_version: str
    project_root: Path
    example_manifest: Path
    benchmark_suite: Path
    notebook_path: Path
    required_test_targets: tuple[str, ...]
    certification_commands: tuple[str, ...]


@dataclass(frozen=True)
class CleanInstallCertificationSurfaceResult:
    surface_id: str
    status: str
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class CleanInstallCertificationResult:
    project_root: Path
    workspace_root: Path
    report_path: Path
    surface_completion: float
    surfaces: tuple[CleanInstallCertificationSurfaceResult, ...]


@dataclass(frozen=True)
class CompletionRegressionCheckResult:
    project_root: Path
    report_path: Path
    policy_path: Path
    passed: bool
    failure_messages: tuple[str, ...]


@dataclass(frozen=True)
class RepositoryTestMatrixResult:
    project_root: Path
    report_path: Path
    command: tuple[str, ...]
    passed: bool
    exit_code: int
    passed_count: int
    failed_count: int
    skipped_count: int
    xfailed_count: int
    xpassed_count: int
    summary_line: str


@dataclass(frozen=True)
class ResearchReadinessCertificationResult:
    project_root: Path
    report_path: Path
    status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class _CompletionLedgerRow:
    row_id: str
    status: str
    required_closing_classes: tuple[str, ...]
    available_evidence_classes: tuple[str, ...]
    non_closing_evidence_classes: tuple[str, ...]
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    evidence_bundle_ids: tuple[str, ...]
    proof_status: str | None


def _now_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _directory_digest(root: Path) -> str:
    entries: list[tuple[str, str]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        entries.append((str(path.relative_to(root)), _sha256_file(path)))
    return _sha256_payload(entries)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _declared_build_toolchain(checkout_root: Path) -> tuple[str, tuple[str, ...]]:
    pyproject_path = checkout_root / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    build_system = dict(payload.get("build-system", {}))
    backend = str(build_system.get("build-backend", "setuptools.build_meta"))
    requirements = tuple(
        _ordered_unique(
            [
                "python3.11",
                "build",
                *[str(requirement) for requirement in build_system.get("requires", ())],
            ]
        )
    )
    return backend, requirements


def _declared_runtime_requirements(checkout_root: Path) -> tuple[str, ...]:
    pyproject_path = checkout_root / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project_table = dict(payload.get("project", {}))
    return tuple(str(requirement) for requirement in project_table.get("dependencies", ()))


def _requirement_distribution_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME_PATTERN.match(requirement.strip())
    if match is None:
        raise ValueError(f"unable to parse requirement name from {requirement!r}")
    return match.group(0)


def _runtime_dependency_distribution_names(checkout_root: Path) -> tuple[str, ...]:
    from packaging.requirements import Requirement

    pending = list(_declared_runtime_requirements(checkout_root))
    discovered: dict[str, str] = {}

    while pending:
        raw_requirement = pending.pop()
        parsed_requirement = Requirement(raw_requirement)
        if parsed_requirement.marker is not None and not parsed_requirement.marker.evaluate():
            continue
        distribution_name = parsed_requirement.name
        normalized_name = distribution_name.lower().replace("-", "_")
        if normalized_name in discovered:
            continue
        distribution = importlib_metadata.distribution(distribution_name)
        discovered[normalized_name] = str(distribution.metadata["Name"])
        for nested_requirement in distribution.requires or ():
            pending.append(str(nested_requirement))

    return tuple(sorted(discovered.values(), key=str.lower))


def _load_packaged_implementation_contract(relative_path: str) -> dict[str, Any]:
    return _load_packaged_yaml(relative_path)


def _load_packaged_authority_snapshot() -> dict[str, Any]:
    return _load_packaged_implementation_contract(_AUTHORITY_SNAPSHOT_PATH)


def _load_packaged_closure_map() -> dict[str, Any]:
    return _load_packaged_implementation_contract(_CLOSURE_MAP_PATH)


def _load_packaged_traceability() -> dict[str, Any]:
    return _load_packaged_implementation_contract(_TRACEABILITY_PATH)


def _load_packaged_fixture_spec() -> dict[str, Any]:
    return _load_packaged_implementation_contract(_CERTIFICATION_FIXTURE_SPEC_PATH)


def _load_packaged_evidence_contract() -> dict[str, Any]:
    return _load_packaged_implementation_contract(_CERTIFICATION_EVIDENCE_CONTRACT_PATH)


def _load_packaged_command_contract() -> dict[str, Any]:
    return _load_packaged_implementation_contract(_CERTIFICATION_COMMAND_CONTRACT_PATH)


def _report_path(workspace_root: Path, relative_path: str) -> Path:
    return workspace_root / relative_path


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_report_if_present(path: Path) -> dict[str, Any] | None:
    payload = _load_json_if_present(path)
    if payload is None:
        return None
    return {**payload, "__report_path__": str(path)}


def _suite_task_result_payloads(
    suite_summary: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    payloads: list[Mapping[str, Any]] = []
    for row in (suite_summary or {}).get("task_results", ()):
        if not isinstance(row, Mapping):
            continue
        task_result_path = row.get("task_result_path")
        if not isinstance(task_result_path, str) or not task_result_path:
            continue
        payload = _load_json_if_present(Path(task_result_path))
        if payload is not None:
            payloads.append(payload)
    return tuple(payloads)


def _suite_reducer_family_ids(suite_summary: Mapping[str, Any] | None) -> set[str]:
    reducer_family_ids = _semantic_summary_ids(suite_summary, "reducer_family_ids")
    if reducer_family_ids:
        return reducer_family_ids

    resolved: set[str] = set()
    for payload in _suite_task_result_payloads(suite_summary):
        submitter_ids = []
        local_winner_submitter_id = payload.get("local_winner_submitter_id")
        if isinstance(local_winner_submitter_id, str) and local_winner_submitter_id:
            submitter_ids.append(local_winner_submitter_id)
        for ref in payload.get("submitter_result_refs", ()):
            if not isinstance(ref, Mapping):
                continue
            relative_path = ref.get("relative_path")
            if not isinstance(relative_path, str) or not relative_path:
                continue
            filename = Path(relative_path).name
            if filename.startswith("submitter-") and filename.endswith(".json"):
                submitter_ids.append(
                    filename.removeprefix("submitter-").removesuffix(".json")
                )
        for submitter_id in submitter_ids:
            resolved.update(_SUBMITTER_REDUCER_FAMILY_IDS.get(str(submitter_id), ()))
    return resolved


def _suite_lifecycle_artifact_ids(
    suite_summary: Mapping[str, Any] | None,
) -> set[str]:
    lifecycle_artifact_ids = _semantic_summary_ids(suite_summary, "lifecycle_artifact_ids")
    if lifecycle_artifact_ids:
        return lifecycle_artifact_ids

    task_rows = tuple(
        row
        for row in (suite_summary or {}).get("task_results", ())
        if isinstance(row, Mapping)
    )
    surface_rows = tuple(
        row
        for row in (suite_summary or {}).get("surface_statuses", ())
        if isinstance(row, Mapping) and row.get("benchmark_status") == "passed"
    )
    if not task_rows and not surface_rows:
        return set()

    resolved = set(_SUITE_GENERIC_LIFECYCLE_ARTIFACT_IDS)
    if task_rows:
        resolved.add("point_score_result")
        resolved.add("claim_card_or_abstention")
    if any(bool(row.get("calibration_required")) for row in task_rows):
        resolved.add("calibration_result")
    if any(
        isinstance(row.get("abstention_mode"), str) and row.get("abstention_mode")
        for row in task_rows
    ):
        resolved.add("abstention")
    if any(row.get("replay_verification") == "verified" for row in task_rows):
        resolved.update({"reproducibility_bundle", "replay_verification_report"})
    for row in surface_rows:
        surface_id = row.get("surface_id")
        if not isinstance(surface_id, str):
            continue
        resolved.update(_SUITE_SURFACE_LIFECYCLE_ARTIFACT_IDS.get(surface_id, ()))
    return resolved


def _operator_registry_artifact_paths(
    registry_path: Path,
) -> tuple[tuple[str, Path], ...]:
    if not registry_path.is_file():
        return ()
    with sqlite3.connect(registry_path) as connection:
        rows = connection.execute(
            "select schema_name, artifact_path from manifests"
        ).fetchall()
    return tuple(
        (str(schema_name), Path(artifact_path))
        for schema_name, artifact_path in rows
        if isinstance(schema_name, str) and isinstance(artifact_path, str)
    )


def _load_operator_runtime_semantics(
    *,
    operator_run_report: Mapping[str, Any] | None,
    operator_replay_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    semantics: dict[str, Any] = {
        "run_support_object_ids": set(),
        "admissibility_rule_ids": set(),
        "lifecycle_artifact_ids": set(),
        "operator_runtime_surface_ids": set(),
        "replay_surface_ids": set(),
        "evidence_refs": set(),
    }
    if operator_run_report is not None:
        semantics["operator_runtime_surface_ids"].add("operator_run")
        report_path = operator_run_report.get("__report_path__")
        if isinstance(report_path, str) and report_path:
            semantics["evidence_refs"].add(f"artifact:{report_path}")
        run_summary_path = operator_run_report.get("run_summary_path")
        if isinstance(run_summary_path, str) and run_summary_path:
            run_summary_file = Path(run_summary_path)
            semantics["evidence_refs"].add(f"artifact:{run_summary_file}")
            if run_summary_file.is_file():
                run_summary = _load_json_if_present(run_summary_file) or {}
                if operator_run_report.get("manifest_path"):
                    semantics["lifecycle_artifact_ids"].add("run_manifest")
                    semantics["evidence_refs"].add(
                        f"artifact:{operator_run_report['manifest_path']}"
                    )
                if run_summary.get("claim_card_ref") is not None:
                    semantics["lifecycle_artifact_ids"].update(
                        {"claim_card", "claim_card_or_abstention"}
                    )
                if run_summary.get("abstention_ref") is not None:
                    semantics["lifecycle_artifact_ids"].update(
                        {"abstention", "claim_card_or_abstention"}
                    )
                registry_path = run_summary_file.parent / "registry.sqlite3"
                for schema_name, artifact_path in _operator_registry_artifact_paths(
                    registry_path
                ):
                    semantics["lifecycle_artifact_ids"].update(
                        _OPERATOR_REGISTRY_LIFECYCLE_ARTIFACT_IDS.get(schema_name, ())
                    )
                    semantics["evidence_refs"].add(f"artifact:{artifact_path}")
                run_result_ref = run_summary.get("run_result_ref")
                if isinstance(run_result_ref, Mapping):
                    run_result_object_id = run_result_ref.get("object_id")
                    validation_scope_ref = None
                    if isinstance(run_result_object_id, str):
                        with sqlite3.connect(registry_path) as connection:
                            row = connection.execute(
                                (
                                    "select artifact_path from manifests "
                                    "where schema_name = ? and object_id = ?"
                                ),
                                ("run_result_manifest@1.1.0", run_result_object_id),
                            ).fetchone()
                        if row is not None and isinstance(row[0], str):
                            run_result_payload = _load_json_if_present(Path(row[0])) or {}
                            validation_scope_ref = (
                                run_result_payload.get("body", {}).get(
                                    "primary_validation_scope_ref"
                                )
                            )
                    if validation_scope_ref is None:
                        validation_scope_ref = run_summary.get("validation_scope_ref")
                    for ref, key in (
                        (run_summary.get("scope_ledger_ref"), "scope_ledger"),
                        (validation_scope_ref, "validation_scope"),
                    ):
                        if not isinstance(ref, Mapping):
                            continue
                        object_id = ref.get("object_id")
                        schema_name = ref.get("schema_name")
                        if not isinstance(object_id, str) or not isinstance(
                            schema_name, str
                        ):
                            continue
                        with sqlite3.connect(registry_path) as connection:
                            row = connection.execute(
                                (
                                    "select artifact_path from manifests "
                                    "where schema_name = ? and object_id = ?"
                                ),
                                (schema_name, object_id),
                            ).fetchone()
                        if row is None or not isinstance(row[0], str):
                            continue
                        payload = _load_json_if_present(Path(row[0])) or {}
                        body = payload.get("body", {})
                        if key == "scope_ledger":
                            semantics["run_support_object_ids"].update(
                                str(value)
                                for value in body.get("run_support_object_ids", ())
                                if value
                            )
                            semantics["admissibility_rule_ids"].update(
                                str(value)
                                for value in body.get("admissibility_rule_ids", ())
                                if value
                            )
                        elif key == "validation_scope":
                            semantics["run_support_object_ids"].update(
                                str(value)
                                for value in body.get("run_support_object_ids", ())
                                if value
                            )
                            semantics["admissibility_rule_ids"].update(
                                str(value)
                                for value in body.get("admissibility_rule_ids", ())
                                if value
                            )
                        semantics["evidence_refs"].add(f"artifact:{row[0]}")
    if operator_replay_report is not None:
        semantics["operator_runtime_surface_ids"].add("operator_replay")
        if operator_replay_report.get("replay_verification_status") == "verified":
            semantics["replay_surface_ids"].add("operator_native_replay")
            semantics["lifecycle_artifact_ids"].add("replay_verification_report")
        report_path = operator_replay_report.get("__report_path__")
        if isinstance(report_path, str) and report_path:
            semantics["evidence_refs"].add(f"artifact:{report_path}")
        run_summary_path = operator_replay_report.get("run_summary_path")
        if isinstance(run_summary_path, str) and run_summary_path:
            semantics["evidence_refs"].add(f"artifact:{run_summary_path}")
    return semantics


def _bundle_by_scope_id() -> dict[str, Mapping[str, Any]]:
    return {
        str(bundle["scope_id"]): bundle
        for bundle in _load_packaged_evidence_contract().get("bundles", ())
    }


def _bundle_by_producer_command() -> dict[str, Mapping[str, Any]]:
    return {
        str(bundle["producer_command_id"]): bundle
        for bundle in _load_packaged_evidence_contract().get("bundles", ())
    }


def get_release_status(
    *,
    project_root: Path | str | None = None,
    benchmark_root: Path | str | None = None,
    notebook_output_root: Path | str | None = None,
    supplemental_gate_results: tuple[ReadinessGateResult, ...] = (),
) -> ReleaseStatus:
    asset_root = _resolve_asset_root(project_root)
    workspace_root = _resolve_workspace_root(project_root)
    workflow = get_release_candidate_workflow(project_root=project_root)
    build_root = workspace_root / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    resolved_benchmark_root = (
        Path(benchmark_root)
        if benchmark_root is not None
        else Path(
            tempfile.mkdtemp(
                prefix="release-status-benchmarks-",
                dir=build_root,
            )
        )
    )
    current_benchmark_root = resolved_benchmark_root / "current_release"
    full_vision_benchmark_root = resolved_benchmark_root / "full_vision"
    resolved_notebook_output_root = (
        Path(notebook_output_root)
        if notebook_output_root is not None
        else Path(
            tempfile.mkdtemp(
                prefix="release-status-notebook-",
                dir=build_root,
            )
        )
    )
    contract_validation = validate_release_contracts(project_root=project_root)
    current_release_suite = profile_benchmark_suite(
        manifest_path=workflow.benchmark_suite,
        benchmark_root=current_benchmark_root,
        project_root=asset_root,
    )
    full_vision_suite = profile_benchmark_suite(
        manifest_path=asset_root / "benchmarks" / "suites" / "full-vision.yaml",
        benchmark_root=full_vision_benchmark_root,
        project_root=asset_root,
    )
    write_suite_evidence_bundle(
        suite_result=current_release_suite,
        workspace_root=workspace_root,
    )
    write_suite_evidence_bundle(
        suite_result=full_vision_suite,
        workspace_root=workspace_root,
    )
    notebook_smoke = execute_release_notebook_smoke(
        project_root=project_root,
        output_root=resolved_notebook_output_root,
    )
    clean_install_report = _load_clean_install_report(workspace_root=workspace_root)
    repo_test_matrix_report = _load_json_report_if_present(
        _report_path(workspace_root, _REPO_TEST_MATRIX_REPORT_PATH)
    )
    operator_run_report = _load_json_report_if_present(
        _report_path(workspace_root, _FULL_VISION_OPERATOR_RUN_EVIDENCE_PATH)
    )
    operator_replay_report = _load_json_report_if_present(
        _report_path(workspace_root, _FULL_VISION_OPERATOR_REPLAY_EVIDENCE_PATH)
    )
    operator_runtime_semantics = _load_operator_runtime_semantics(
        operator_run_report=operator_run_report,
        operator_replay_report=operator_replay_report,
    )
    packaged_policies = load_packaged_release_policies(project_root=project_root)
    release_gate_results = tuple(
        _build_release_readiness_gates(
            contract_validation=contract_validation,
            notebook_smoke=notebook_smoke,
            clean_install_report=clean_install_report,
        )
    ) + tuple(supplemental_gate_results)
    current_release_suite_judgment = judge_benchmark_suite_readiness(
        judgment_id="euclid_current_release_status_v1",
        suite_result=current_release_suite,
        supplemental_gate_results=release_gate_results,
    )
    full_vision_suite_judgment = judge_benchmark_suite_readiness(
        judgment_id="euclid_full_vision_status_v1",
        suite_result=full_vision_suite,
        supplemental_gate_results=release_gate_results,
    )
    current_release_judgment = merge_readiness_judgments(
        judgment_id="current_release_v1",
        judgments=(
            _judge_current_release_gate_readiness(
                suite_readiness_judgment=current_release_suite_judgment
            ),
            _judge_packaged_policy_rows(
                policy=packaged_policies["current_release_v1"],
                suite_result=current_release_suite,
                suite_readiness_judgment=current_release_suite_judgment,
                notebook_smoke=notebook_smoke,
                clean_install_report=clean_install_report,
                repo_test_matrix_report=repo_test_matrix_report,
                operator_run_report=operator_run_report,
                operator_replay_report=operator_replay_report,
                operator_runtime_semantics=operator_runtime_semantics,
            ),
        ),
    )
    full_vision_judgment = merge_readiness_judgments(
        judgment_id="full_vision_v1",
        judgments=(
            _judge_full_vision_gate_readiness(
                suite_readiness_judgment=full_vision_suite_judgment
            ),
            _judge_packaged_policy_rows(
                policy=packaged_policies["full_vision_v1"],
                suite_result=full_vision_suite,
                suite_readiness_judgment=full_vision_suite_judgment,
                notebook_smoke=notebook_smoke,
                clean_install_report=clean_install_report,
                repo_test_matrix_report=repo_test_matrix_report,
                operator_run_report=operator_run_report,
                operator_replay_report=operator_replay_report,
                operator_runtime_semantics=operator_runtime_semantics,
            ),
        ),
    )
    shipped_releasable_judgment = merge_readiness_judgments(
        judgment_id="shipped_releasable_v1",
        judgments=(
            current_release_judgment,
            _judge_packaged_policy_rows(
                policy=packaged_policies["shipped_releasable_v1"],
                suite_result=current_release_suite,
                suite_readiness_judgment=current_release_suite_judgment,
                notebook_smoke=notebook_smoke,
                clean_install_report=clean_install_report,
                repo_test_matrix_report=repo_test_matrix_report,
                operator_run_report=operator_run_report,
                operator_replay_report=operator_replay_report,
                operator_runtime_semantics=operator_runtime_semantics,
            ),
        ),
    )
    policy_judgments = {
        "current_release_v1": current_release_judgment,
        "full_vision_v1": full_vision_judgment,
        "shipped_releasable_v1": shipped_releasable_judgment,
    }
    _write_completion_report(
        workspace_root=workspace_root,
        payload=_build_completion_report(
            policies=packaged_policies,
            policy_judgments=policy_judgments,
            current_release_suite=current_release_suite,
            full_vision_suite=full_vision_suite,
            current_release_suite_judgment=current_release_suite_judgment,
            full_vision_suite_judgment=full_vision_suite_judgment,
            notebook_smoke=notebook_smoke,
            clean_install_report=clean_install_report,
            workspace_root=workspace_root,
        ),
    )
    return ReleaseStatus(
        project_root=workspace_root,
        current_version=PACKAGE_VERSION,
        target_version=RELEASE_TARGET_VERSION,
        target_ready=shipped_releasable_judgment.final_verdict == "ready",
        blocked_reason=(
            ""
            if shipped_releasable_judgment.final_verdict == "ready"
            else "; ".join(shipped_releasable_judgment.reason_codes)
        ),
        catalog_scope=shipped_releasable_judgment.catalog_scope,
        policy_judgments=policy_judgments,
        shipped_releasable_judgment=shipped_releasable_judgment,
        readiness_judgment=shipped_releasable_judgment,
    )


def validate_release_contracts(
    *,
    project_root: Path | str | None = None,
) -> ContractCatalogValidationResult:
    asset_root = _resolve_asset_root(project_root)
    workspace_root = _resolve_workspace_root(project_root)
    catalog = load_contract_catalog(asset_root)
    return ContractCatalogValidationResult(
        project_root=workspace_root,
        schema_count=len(catalog.schemas),
        module_count=len(catalog.modules),
        enum_count=len(catalog.enums),
        contract_document_count=len(catalog.contract_documents_by_name),
    )


def run_release_benchmark_smoke(
    *,
    project_root: Path | str | None = None,
    benchmark_root: Path | str | None = None,
    parallel_workers: int = 1,
    resume: bool = True,
) -> BenchmarkSmokeResult:
    asset_root = _resolve_asset_root(project_root)
    workspace_root = _resolve_workspace_root(project_root)
    resolved_benchmark_root = (
        Path(benchmark_root)
        if benchmark_root is not None
        else workspace_root / "build" / "release-benchmark-smoke"
    )
    cases = []
    for manifest_path in _benchmark_smoke_manifest_paths(asset_root):
        profiled = profile_benchmark_task(
            manifest_path=manifest_path,
            benchmark_root=resolved_benchmark_root,
            project_root=asset_root,
            parallel_workers=parallel_workers,
            resume=resume,
        )
        cases.append(
            BenchmarkSmokeCaseResult(
                track_id=profiled.task_manifest.track_id,
                task_id=profiled.task_manifest.task_id,
                task_result_path=profiled.report_paths.task_result_path,
                report_path=profiled.report_paths.report_path,
                telemetry_path=profiled.telemetry_path,
                task_status=_load_json(profiled.report_paths.task_result_path)[
                    "status"
                ],
                local_winner_submitter_id=_load_json(
                    profiled.report_paths.task_result_path
                ).get("local_winner_submitter_id"),
                within_budget=(
                    _load_json(profiled.telemetry_path)["wall_time_seconds"]
                    <= float(
                        profiled.task_manifest.frozen_protocol.budget_policy[
                            "wall_clock_seconds"
                        ]
                    )
                ),
                declared_wall_clock_seconds=float(
                    profiled.task_manifest.frozen_protocol.budget_policy[
                        "wall_clock_seconds"
                    ]
                ),
                observed_wall_time_seconds=float(
                    _load_json(profiled.telemetry_path)["wall_time_seconds"]
                ),
            )
        )
    return BenchmarkSmokeResult(
        project_root=workspace_root,
        benchmark_root=resolved_benchmark_root,
        cases=tuple(cases),
    )


def execute_release_notebook_smoke(
    *,
    project_root: Path | str | None = None,
    output_root: Path | str | None = None,
    notebook_path: Path | str | None = None,
) -> NotebookSmokeResult:
    asset_root = _resolve_asset_root(project_root)
    workspace_root = _resolve_workspace_root(project_root)
    resolved_notebook_path = (
        Path(notebook_path)
        if notebook_path is not None
        else resolve_notebook_path(project_root=asset_root)
    )
    resolved_output_root = (
        Path(output_root)
        if output_root is not None
        else workspace_root / "build" / "release-notebook-smoke"
    )
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    notebook_json = json.loads(resolved_notebook_path.read_text(encoding="utf-8"))
    cells = notebook_json["cells"]
    cells[_notebook_setup_cell_index(cells)]["source"] = _release_notebook_setup_cell(
        project_root=asset_root,
        output_root=resolved_output_root,
    )

    namespace: dict[str, object] = {"__name__": "__main__"}
    for cell in cells:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)

    probabilistic_cases = namespace["probabilistic_cases"]
    published_catalog = namespace["published_catalog"]
    published_entry = namespace["published_entry"]
    summary_payload = {
        "notebook_path": str(resolved_notebook_path),
        "output_root": str(resolved_output_root),
        "probabilistic_case_ids": sorted(probabilistic_cases),
        "catalog_entries": published_catalog.entry_count,
        "publication_mode": published_entry.publication_mode,
    }
    summary_path = resolved_output_root / "notebook-smoke-summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return NotebookSmokeResult(
        project_root=workspace_root,
        notebook_path=resolved_notebook_path,
        output_root=resolved_output_root,
        summary_path=summary_path,
        probabilistic_case_ids=tuple(summary_payload["probabilistic_case_ids"]),
        publication_mode=published_entry.publication_mode,
        catalog_entries=published_catalog.entry_count,
    )


def run_release_determinism_smoke(
    *,
    project_root: Path | str | None = None,
    output_root: Path | str | None = None,
) -> ReleaseDeterminismSmokeResult:
    asset_root = _resolve_asset_root(project_root)
    workspace_root = _resolve_workspace_root(project_root)
    resolved_output_root = (
        Path(output_root)
        if output_root is not None
        else workspace_root / "build" / "release-determinism-smoke"
    )
    if resolved_output_root.exists():
        shutil.rmtree(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = resolve_example_path(
        "current_release_run.yaml",
        project_root=asset_root,
    )
    first_run = run_operator(
        manifest_path=manifest_path,
        output_root=resolved_output_root / "first-run",
    )
    second_run = run_operator(
        manifest_path=manifest_path,
        output_root=resolved_output_root / "second-run",
    )
    comparison = compare_runtime_determinism(
        capture_operator_runtime_snapshot(output_root=first_run.paths.output_root),
        capture_operator_runtime_snapshot(output_root=second_run.paths.output_root),
    )
    summary_payload = {
        "output_root": str(resolved_output_root),
        "identical": comparison.identical,
        "changed_artifact_roles": list(comparison.changed_artifact_roles),
        "changed_seed_scopes": list(comparison.changed_seed_scopes),
        "changed_manifest_refs": list(comparison.changed_manifest_refs),
    }
    summary_path = resolved_output_root / "determinism-smoke-summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ReleaseDeterminismSmokeResult(
        project_root=workspace_root,
        output_root=resolved_output_root,
        summary_path=summary_path,
        identical=comparison.identical,
        changed_artifact_roles=comparison.changed_artifact_roles,
        changed_seed_scopes=comparison.changed_seed_scopes,
        changed_manifest_refs=comparison.changed_manifest_refs,
    )


def run_release_performance_smoke(
    *,
    project_root: Path | str | None = None,
    output_root: Path | str | None = None,
) -> ReleasePerformanceSmokeResult:
    asset_root = _resolve_asset_root(project_root)
    workspace_root = _resolve_workspace_root(project_root)
    resolved_output_root = (
        Path(output_root)
        if output_root is not None
        else workspace_root / "build" / "release-performance-smoke"
    )
    if resolved_output_root.exists():
        shutil.rmtree(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = resolve_example_path(
        "current_release_run.yaml",
        project_root=asset_root,
    )
    operator_run = profile_operator_run(
        manifest_path=manifest_path,
        output_root=resolved_output_root / "operator-output",
    )
    benchmark = profile_benchmark_task(
        manifest_path=_benchmark_smoke_manifest_paths(asset_root)[0],
        benchmark_root=resolved_output_root / "benchmark-output",
    )
    operator_telemetry = load_performance_telemetry(operator_run.telemetry_path)
    benchmark_telemetry = load_performance_telemetry(benchmark.telemetry_path)
    suite = collect_performance_suite(
        suite_id="release_runtime_smoke",
        telemetry_paths=(operator_run.telemetry_path, benchmark.telemetry_path),
    )
    operator_budget = evaluate_performance_budget(
        operator_telemetry,
        PerformanceBudget(
            budget_id="release_operator_run_smoke",
            max_wall_time_seconds=5.0,
            max_peak_memory_bytes=256 * 1024 * 1024,
        ),
    )
    benchmark_budget = evaluate_performance_budget(
        benchmark_telemetry,
        PerformanceBudget(
            budget_id="release_benchmark_task_smoke",
            max_wall_time_seconds=10.0,
            max_peak_memory_bytes=256 * 1024 * 1024,
        ),
    )
    suite_budget = evaluate_suite_performance_budget(
        suite,
        SuitePerformanceBudget(
            budget_id="release_runtime_suite",
            max_total_wall_time_seconds=12.0,
            max_profile_wall_time_seconds=10.0,
        ),
    )
    summary_payload = {
        "output_root": str(resolved_output_root),
        "operator_budget_passed": operator_budget.passed,
        "benchmark_budget_passed": benchmark_budget.passed,
        "suite_budget_passed": suite_budget.passed,
        "total_wall_time_seconds": suite.total_wall_time_seconds,
        "max_wall_time_seconds": suite.max_wall_time_seconds,
    }
    summary_path = resolved_output_root / "performance-smoke-summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ReleasePerformanceSmokeResult(
        project_root=workspace_root,
        output_root=resolved_output_root,
        summary_path=summary_path,
        operator_budget_passed=operator_budget.passed,
        benchmark_budget_passed=benchmark_budget.passed,
        suite_budget_passed=suite_budget.passed,
        total_wall_time_seconds=suite.total_wall_time_seconds,
        max_wall_time_seconds=suite.max_wall_time_seconds,
    )


def _supports_clean_install_certification(root: Path) -> bool:
    return (root / "pyproject.toml").is_file()


def _clean_install_report_path(workspace_root: Path) -> Path:
    return workspace_root / "build" / "reports" / "clean-install-certification.json"


def _clean_install_work_root(workspace_root: Path) -> Path:
    return workspace_root / "build" / "clean-install-certification"


def _clean_install_report_placeholder() -> dict[str, Any]:
    return {
        "report_path": "",
        "surface_completion": 0.0,
        "surfaces": [
            {
                "surface_id": surface_id,
                "status": "missing",
                "reason_codes": [f"clean_install.{surface_id}_missing"],
                "evidence_refs": [],
            }
            for surface_id in _CLEAN_INSTALL_REQUIRED_SURFACE_IDS
        ],
    }


def _load_clean_install_report(
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    report_path = _clean_install_report_path(workspace_root)
    if not report_path.is_file():
        return _clean_install_report_placeholder()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    surfaces_by_id = {
        str(surface["surface_id"]): dict(surface)
        for surface in payload.get("surfaces", ())
    }
    surfaces = []
    for surface_id in _CLEAN_INSTALL_REQUIRED_SURFACE_IDS:
        surface = surfaces_by_id.get(surface_id)
        if surface is None:
            surfaces.append(
                {
                    "surface_id": surface_id,
                    "status": "missing",
                    "reason_codes": [f"clean_install.{surface_id}_missing"],
                    "evidence_refs": [],
                }
            )
            continue
        surfaces.append(surface)
    passed_surfaces = sum(
        1 for surface in surfaces if surface.get("status") == "passed"
    )
    normalized_payload = {
        **payload,
        "report_path": str(report_path),
        "surface_completion": round(
            passed_surfaces / max(len(_CLEAN_INSTALL_REQUIRED_SURFACE_IDS), 1), 6
        ),
        "surfaces": surfaces,
    }
    return normalized_payload


def _surface_log_root(work_root: Path) -> Path:
    return work_root / "logs"


def _write_surface_logs(
    *,
    stdout_path: Path,
    stderr_path: Path,
    result: subprocess.CompletedProcess[str],
) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")


def _run_command_with_logs(
    *,
    args: list[str],
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    _write_surface_logs(
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        result=result,
    )
    return result


def _build_runtime_dependency_wheelhouse(
    *,
    checkout_root: Path,
    dist_dir: Path,
    log_root: Path,
) -> tuple[str, ...]:
    runtime_distributions = _runtime_dependency_distribution_names(checkout_root)
    staging_root = dist_dir / "_wheel_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    try:
        for distribution_name in runtime_distributions:
            distribution = importlib_metadata.distribution(distribution_name)
            files = distribution.files
            if not files:
                raise ValueError(
                    f"installed distribution for {distribution_name!r} does not expose files"
                )
            package_root = Path(distribution.locate_file(""))
            normalized_name = distribution.metadata["Name"].replace("-", "_")
            unpacked_root = staging_root / f"{normalized_name}-{distribution.version}"
            if unpacked_root.exists():
                shutil.rmtree(unpacked_root)
            unpacked_root.mkdir(parents=True, exist_ok=True)
            for file_entry in files:
                source_path = package_root / file_entry
                if not source_path.exists() or source_path.is_dir():
                    continue
                destination_path = unpacked_root / file_entry
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)
            wheel_result = _run_command_with_logs(
                args=[
                    sys.executable,
                    "-m",
                    "wheel",
                    "pack",
                    str(unpacked_root),
                    "--dest-dir",
                    str(dist_dir),
                ],
                cwd=checkout_root,
                env=os.environ.copy(),
                stdout_path=log_root / f"wheel-pack-{normalized_name}.stdout.log",
                stderr_path=log_root / f"wheel-pack-{normalized_name}.stderr.log",
            )
            if wheel_result.returncode != 0:
                raise RuntimeError(wheel_result.stderr)
            shutil.rmtree(unpacked_root)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return runtime_distributions


def _venv_python_bin(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _installed_workflow_metadata(
    *,
    python_bin: Path,
    cwd: Path,
    env: dict[str, str],
    log_root: Path,
) -> tuple[str, str]:
    metadata_result = _run_command_with_logs(
        args=[
            str(python_bin),
            "-c",
            (
                "import yaml, euclid; "
                "from pathlib import Path; "
                "workflow = euclid.get_release_candidate_workflow(); "
                "payload = yaml.safe_load(Path(workflow.example_manifest)"
                ".read_text(encoding='utf-8')); "
                "print(workflow.example_manifest); "
                "print(payload['request_id'])"
            ),
        ],
        cwd=cwd,
        env=env,
        stdout_path=log_root / "workflow-metadata.stdout.log",
        stderr_path=log_root / "workflow-metadata.stderr.log",
    )
    if metadata_result.returncode != 0:
        raise RuntimeError(metadata_result.stderr)
    lines = [
        line.strip() for line in metadata_result.stdout.splitlines() if line.strip()
    ]
    if len(lines) < 2:
        raise ValueError(
            "installed workflow metadata did not emit manifest path and run id"
        )
    return lines[0], lines[1]


def _build_clean_install_surface_result(
    *,
    surface_id: str,
    result: subprocess.CompletedProcess[str],
    stdout_path: Path,
    stderr_path: Path,
    extra_refs: tuple[str, ...] = (),
) -> CleanInstallCertificationSurfaceResult:
    status = "passed" if result.returncode == 0 else "failed"
    reason_codes = () if status == "passed" else (f"clean_install.{surface_id}_failed",)
    return CleanInstallCertificationSurfaceResult(
        surface_id=surface_id,
        status=status,
        reason_codes=reason_codes,
        evidence_refs=tuple(
            _ordered_unique(
                [
                    f"artifact:{stdout_path}",
                    f"artifact:{stderr_path}",
                    *extra_refs,
                ]
            )
        ),
    )


def run_clean_install_certification(
    *,
    project_root: Path | str | None = None,
    output_root: Path | str | None = None,
    wheel_dir: Path | str | None = None,
) -> CleanInstallCertificationResult:
    checkout_root = _resolve_checkout_root(project_root)
    workspace_root = _resolve_workspace_root(project_root)
    if not _supports_clean_install_certification(checkout_root):
        raise ValueError(
            "clean-install certification requires a checkout root with pyproject.toml"
        )
    canonical_report_path = _clean_install_report_path(workspace_root)
    work_root = (
        Path(output_root)
        if output_root is not None
        else _clean_install_work_root(workspace_root)
    ).resolve()
    build_backend, build_toolchain_requirements = _declared_build_toolchain(
        checkout_root
    )
    declared_runtime_requirements = _declared_runtime_requirements(checkout_root)
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    report_path = (work_root / "clean-install-report.json").resolve()
    dist_dir = (
        Path(wheel_dir) if wheel_dir is not None else work_root / "dist"
    ).resolve()
    venv_dir = (work_root / "venv").resolve()
    outside_repo = (work_root / "outside-repo").resolve()
    artifact_root = (work_root / "artifacts").resolve()
    log_root = _surface_log_root(work_root).resolve()
    outside_repo.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    build_result = _run_command_with_logs(
        args=[
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
        ],
        cwd=checkout_root,
        env=os.environ.copy(),
        stdout_path=log_root / "build-wheel.stdout.log",
        stderr_path=log_root / "build-wheel.stderr.log",
    )
    if build_result.returncode != 0:
        raise RuntimeError(build_result.stderr)
    wheels = sorted(dist_dir.glob("euclid-*.whl"))
    if not wheels:
        raise FileNotFoundError("expected clean-install certification to build a wheel")
    wheel_path = wheels[-1]
    wheel_digest = _sha256_file(wheel_path)
    runtime_distributions = _build_runtime_dependency_wheelhouse(
        checkout_root=checkout_root,
        dist_dir=dist_dir,
        log_root=log_root,
    )

    create_venv = _run_command_with_logs(
        args=[sys.executable, "-m", "venv", str(venv_dir)],
        cwd=outside_repo,
        env=os.environ.copy(),
        stdout_path=log_root / "create-venv.stdout.log",
        stderr_path=log_root / "create-venv.stderr.log",
    )
    if create_venv.returncode != 0:
        raise RuntimeError(create_venv.stderr)

    python_bin = _venv_python_bin(venv_dir)
    clean_env = os.environ.copy()
    clean_env.pop("EUCLID_PROJECT_ROOT", None)
    clean_env.pop("PYTHONPATH", None)
    clean_env[_SKIP_CLEAN_INSTALL_CERTIFICATION_ENV] = "1"

    install_wheel = _run_command_with_logs(
        args=[
            str(python_bin),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(dist_dir),
            str(wheel_path),
        ],
        cwd=outside_repo,
        env=clean_env,
        stdout_path=log_root / "install-wheel.stdout.log",
        stderr_path=log_root / "install-wheel.stderr.log",
    )
    if install_wheel.returncode != 0:
        raise RuntimeError(install_wheel.stderr)

    workflow_manifest, workflow_run_id = _installed_workflow_metadata(
        python_bin=python_bin,
        cwd=outside_repo,
        env=clean_env,
        log_root=log_root,
    )

    surface_specs = (
        (
            "release_status",
            [
                str(python_bin),
                "-m",
                "euclid",
                "release",
                "status",
            ],
            (),
        ),
        (
            "operator_run",
            [
                str(python_bin),
                "-m",
                "euclid",
                "run",
                "--config",
                workflow_manifest,
                "--output-root",
                str(artifact_root / "operator-run"),
            ],
            (f"artifact:{artifact_root / 'operator-run'}",),
        ),
        (
            "operator_replay",
            [
                str(python_bin),
                "-m",
                "euclid",
                "replay",
                "--run-id",
                workflow_run_id,
                "--output-root",
                str(artifact_root / "operator-run"),
            ],
            (f"artifact:{artifact_root / 'operator-run'}",),
        ),
        (
            "determinism_same_seed",
            [
                str(python_bin),
                "-m",
                "euclid",
                "release",
                "determinism-smoke",
                "--output-root",
                str(artifact_root / "determinism"),
            ],
            (
                "artifact:"
                f"{artifact_root / 'determinism' / 'determinism-smoke-summary.json'}",
            ),
        ),
        (
            "performance_runtime_smoke",
            [
                str(python_bin),
                "-m",
                "euclid",
                "release",
                "performance-smoke",
                "--output-root",
                str(artifact_root / "performance"),
            ],
            (
                "artifact:"
                f"{artifact_root / 'performance' / 'performance-smoke-summary.json'}",
            ),
        ),
        (
            "packaged_notebook_smoke",
            [
                str(python_bin),
                "-m",
                "euclid",
                "release",
                "notebook-smoke",
                "--output-root",
                str(artifact_root / "notebook"),
            ],
            (
                "artifact:"
                f"{artifact_root / 'notebook' / 'notebook-smoke-summary.json'}",
            ),
        ),
    )

    surface_results = []
    for surface_id, args, extra_refs in surface_specs:
        stdout_path = log_root / f"{surface_id}.stdout.log"
        stderr_path = log_root / f"{surface_id}.stderr.log"
        result = _run_command_with_logs(
            args=args,
            cwd=outside_repo,
            env=clean_env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        surface_results.append(
            _build_clean_install_surface_result(
                surface_id=surface_id,
                result=result,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                extra_refs=extra_refs,
            )
        )

    benchmark_current_stdout = log_root / "benchmark_execution.current_release.stdout.log"
    benchmark_current_stderr = log_root / "benchmark_execution.current_release.stderr.log"
    benchmark_current_result = _run_command_with_logs(
        args=[
            str(python_bin),
            "-m",
            "euclid",
            "benchmarks",
            "run",
            "--suite",
            "current-release.yaml",
            "--benchmark-root",
            str(artifact_root / "benchmarks" / "current_release"),
        ],
        cwd=outside_repo,
        env=clean_env,
        stdout_path=benchmark_current_stdout,
        stderr_path=benchmark_current_stderr,
    )
    benchmark_full_stdout = log_root / "benchmark_execution.full_vision.stdout.log"
    benchmark_full_stderr = log_root / "benchmark_execution.full_vision.stderr.log"
    benchmark_full_result = _run_command_with_logs(
        args=[
            str(python_bin),
            "-m",
            "euclid",
            "benchmarks",
            "run",
            "--suite",
            "full-vision.yaml",
            "--benchmark-root",
            str(artifact_root / "benchmarks" / "full_vision"),
        ],
        cwd=outside_repo,
        env=clean_env,
        stdout_path=benchmark_full_stdout,
        stderr_path=benchmark_full_stderr,
    )
    benchmark_status = (
        "passed"
        if benchmark_current_result.returncode == 0 and benchmark_full_result.returncode == 0
        else "failed"
    )
    surface_results.append(
        CleanInstallCertificationSurfaceResult(
            surface_id="benchmark_execution",
            status=benchmark_status,
            reason_codes=(
                ()
                if benchmark_status == "passed"
                else ("clean_install.benchmark_execution_failed",)
            ),
            evidence_refs=tuple(
                _ordered_unique(
                    [
                        f"artifact:{benchmark_current_stdout}",
                        f"artifact:{benchmark_current_stderr}",
                        f"artifact:{benchmark_full_stdout}",
                        f"artifact:{benchmark_full_stderr}",
                        f"artifact:{artifact_root / 'benchmarks' / 'current_release'}",
                        f"artifact:{artifact_root / 'benchmarks' / 'full_vision'}",
                    ]
                )
            ),
        )
    )

    shipped_bundle = _bundle_by_scope_id()["shipped_releasable"]
    payload = {
        "report_id": "euclid_clean_install_certification_v1",
        "evidence_bundle_id": shipped_bundle["evidence_bundle_id"],
        "scope_id": shipped_bundle["scope_id"],
        "authority_snapshot_id": shipped_bundle["authority_snapshot_id"],
        "command_contract_id": shipped_bundle["command_contract_id"],
        "closure_map_id": shipped_bundle["closure_map_id"],
        "traceability_id": shipped_bundle["traceability_id"],
        "fixture_spec_id": shipped_bundle["fixture_spec_id"],
        "producer_command_id": shipped_bundle["producer_command_id"],
        "generated_at_utc": _now_utc_timestamp(),
        "input_manifest_digests": [
            {
                str(dist_dir): f"runtime_directory_digest:{_directory_digest(dist_dir)}",
            }
        ],
        "source_tree_digest_or_wheel_digest": f"wheel_digest:{wheel_digest}",
        "dirty_state_or_build_toolchain": (
            "build_frontend:build;"
            f"build_backend:{build_backend};"
            f"build_requires:{','.join(build_toolchain_requirements)}"
        ),
        "build_backend": build_backend,
        "build_toolchain_requirements": list(build_toolchain_requirements),
        "canonical_report_path": str(canonical_report_path),
        "wheel_path": str(wheel_path),
        "wheel_digest": wheel_digest,
        "output_root": str(work_root),
        "runtime_dependency_requirements": list(declared_runtime_requirements),
        "runtime_dependency_distributions": list(runtime_distributions),
        "runtime_dependency_wheelhouse": str(dist_dir),
        "runtime_dependency_wheel_count": len(
            [path for path in dist_dir.glob("*.whl") if path != wheel_path]
        ),
        "surface_completion": round(
            sum(1 for surface in surface_results if surface.status == "passed")
            / max(len(surface_results), 1),
            6,
        ),
        "surfaces": [
            {
                "surface_id": surface.surface_id,
                "status": surface.status,
                "reason_codes": list(surface.reason_codes),
                "evidence_refs": list(surface.evidence_refs),
            }
            for surface in surface_results
        ],
    }
    _write_json(report_path, payload)
    _write_json(canonical_report_path, payload)
    return CleanInstallCertificationResult(
        project_root=checkout_root,
        workspace_root=workspace_root,
        report_path=report_path,
        surface_completion=float(payload["surface_completion"]),
        surfaces=tuple(surface_results),
    )


def _suite_scope_and_command(suite_id: str) -> tuple[str, str, str] | None:
    if suite_id == "current_release":
        return ("current_release", "current_release_suite", _CURRENT_RELEASE_SUITE_EVIDENCE_PATH)
    if suite_id == "full_vision":
        return ("full_vision", "full_vision_suite", _FULL_VISION_SUITE_EVIDENCE_PATH)
    return None


def write_suite_evidence_bundle(
    *,
    suite_result,
    workspace_root: Path,
) -> Path | None:
    suite_metadata = _suite_scope_and_command(suite_result.suite_manifest.suite_id)
    if suite_metadata is None:
        return None
    scope_id, producer_command_id, report_relative_path = suite_metadata
    bundle_template = _bundle_by_scope_id()[scope_id]
    manifest_digests = [
        {
            str(suite_result.suite_manifest.source_path): (
                f"runtime_sha256:{_sha256_file(suite_result.suite_manifest.source_path)}"
            )
        }
    ]
    manifest_digests.extend(
        {
            str(task_path): f"runtime_sha256:{_sha256_file(task_path)}"
        }
        for task_path in suite_result.suite_manifest.task_manifest_paths
    )
    payload = {
        "report_id": f"{suite_result.suite_manifest.suite_id}_suite_evidence_v1",
        "evidence_bundle_id": bundle_template["evidence_bundle_id"],
        "scope_id": scope_id,
        "authority_snapshot_id": bundle_template["authority_snapshot_id"],
        "command_contract_id": bundle_template["command_contract_id"],
        "closure_map_id": bundle_template["closure_map_id"],
        "traceability_id": bundle_template["traceability_id"],
        "fixture_spec_id": bundle_template["fixture_spec_id"],
        "producer_command_id": producer_command_id,
        "generated_at_utc": _now_utc_timestamp(),
        "input_manifest_digests": manifest_digests,
        "source_tree_digest_or_wheel_digest": (
            f"repo_checkout_digest:{_sha256_payload(manifest_digests)}"
        ),
        "dirty_state_or_build_toolchain": "repo_checkout_dirty_state:unavailable",
        "suite_id": suite_result.suite_manifest.suite_id,
        "summary_path": str(suite_result.summary_path),
        "surface_statuses": [
            {
                "surface_id": surface.surface_id,
                "benchmark_status": surface.benchmark_status,
                "replay_status": surface.replay_status,
            }
            for surface in suite_result.surface_statuses
        ],
    }
    return _write_json(_report_path(workspace_root, report_relative_path), payload)


def write_operator_run_evidence_report(
    *,
    result,
    report_path: Path,
    scope_id: str,
) -> Path:
    payload = {
        "report_id": "operator_run_evidence_v1",
        "command_id": "full_vision_operator_run",
        "scope_id": scope_id,
        "authority_snapshot_id": _load_packaged_authority_snapshot()[
            "authority_snapshot_id"
        ],
        "command_contract_id": _load_packaged_command_contract()["command_contract_id"],
        "run_id": result.request.request_id,
        "manifest_path": str(result.request.manifest_path),
        "run_summary_path": str(result.paths.run_summary_path),
        "output_root": str(result.paths.output_root),
        "run_result_ref": result.summary.run_result_ref.as_dict(),
        "bundle_ref": result.summary.bundle_ref.as_dict(),
        "generated_at_utc": _now_utc_timestamp(),
    }
    return _write_json(report_path, payload)


def write_operator_replay_evidence_report(
    *,
    run_id: str,
    result,
    report_path: Path,
    scope_id: str,
) -> Path:
    payload = {
        "report_id": "operator_replay_evidence_v1",
        "command_id": "full_vision_operator_replay",
        "scope_id": scope_id,
        "authority_snapshot_id": _load_packaged_authority_snapshot()[
            "authority_snapshot_id"
        ],
        "command_contract_id": _load_packaged_command_contract()["command_contract_id"],
        "run_id": run_id,
        "run_summary_path": str(result.paths.run_summary_path),
        "output_root": str(result.paths.output_root),
        "replay_verification_status": result.summary.replay_verification_status,
        "run_result_ref": result.summary.run_result_ref.as_dict(),
        "bundle_ref": result.summary.bundle_ref.as_dict(),
        "generated_at_utc": _now_utc_timestamp(),
    }
    return _write_json(report_path, payload)


def _parse_pytest_summary_counts(stdout: str) -> tuple[dict[str, int], str]:
    summary_line = ""
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("=") and stripped.endswith("="):
            summary_line = stripped.strip("= ").strip()
            break
        if " passed" in stripped or " failed" in stripped:
            summary_line = stripped
            break
    counts: dict[str, int] = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    for match in _SUMMARY_TOKEN_PATTERN.finditer(summary_line):
        counts[match.group("label")] = int(match.group("count"))
    return counts, summary_line


def run_repo_test_matrix(
    *,
    project_root: Path | str | None = None,
    report_path: Path | str | None = None,
) -> RepositoryTestMatrixResult:
    checkout_root = _resolve_checkout_root(project_root)
    workspace_root = _resolve_workspace_root(project_root)
    resolved_report_path = (
        Path(report_path)
        if report_path is not None
        else _report_path(workspace_root, _REPO_TEST_MATRIX_REPORT_PATH)
    )
    command = (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *RELEASE_CERTIFICATION_TEST_TARGETS,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(checkout_root / "src")
    result = subprocess.run(
        command,
        cwd=checkout_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    counts, summary_line = _parse_pytest_summary_counts(result.stdout)
    passed = (
        result.returncode == 0
        and counts.get("failed", 0) == 0
        and counts.get("skipped", 0) == 0
        and counts.get("xfailed", 0) == 0
        and counts.get("xpassed", 0) == 0
    )
    payload = {
        "report_id": "repo_test_matrix_v1",
        "authority_snapshot_id": _load_packaged_authority_snapshot()[
            "authority_snapshot_id"
        ],
        "command_contract_id": _load_packaged_command_contract()["command_contract_id"],
        "command": " ".join(command),
        "generated_at_utc": _now_utc_timestamp(),
        "passed": passed,
        "exit_code": result.returncode,
        "summary_line": summary_line,
        "counts": counts,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    _write_json(resolved_report_path, payload)
    return RepositoryTestMatrixResult(
        project_root=checkout_root,
        report_path=resolved_report_path,
        command=command,
        passed=passed,
        exit_code=result.returncode,
        passed_count=counts.get("passed", 0),
        failed_count=counts.get("failed", 0),
        skipped_count=counts.get("skipped", 0),
        xfailed_count=counts.get("xfailed", 0),
        xpassed_count=counts.get("xpassed", 0),
        summary_line=summary_line,
    )


def certify_research_readiness(
    *,
    project_root: Path | str | None = None,
) -> ResearchReadinessCertificationResult:
    workspace_root = _resolve_workspace_root(project_root)
    completion_report = _load_json(
        _report_path(workspace_root, "build/reports/completion-report.json")
    )
    repo_test_matrix = _load_json_if_present(
        _report_path(workspace_root, _REPO_TEST_MATRIX_REPORT_PATH)
    )
    current_suite = _load_json_if_present(
        _report_path(workspace_root, _CURRENT_RELEASE_SUITE_EVIDENCE_PATH)
    )
    full_suite = _load_json_if_present(
        _report_path(workspace_root, _FULL_VISION_SUITE_EVIDENCE_PATH)
    )
    full_run = _load_json_if_present(
        _report_path(workspace_root, _FULL_VISION_OPERATOR_RUN_EVIDENCE_PATH)
    )
    full_replay = _load_json_if_present(
        _report_path(workspace_root, _FULL_VISION_OPERATOR_REPLAY_EVIDENCE_PATH)
    )
    clean_install = _load_clean_install_report(workspace_root=workspace_root)

    reason_codes: list[str] = []
    if repo_test_matrix is None or repo_test_matrix.get("passed") is not True:
        reason_codes.append("repo_test_matrix_missing_or_failed")
    if full_run is None:
        reason_codes.append("full_vision_operator_run_evidence_missing")
    if full_replay is None or full_replay.get("replay_verification_status") != "verified":
        reason_codes.append("full_vision_operator_replay_missing_or_failed")
    if any(surface.get("status") != "passed" for surface in clean_install.get("surfaces", ())):
        reason_codes.append("clean_install_certification_incomplete")

    verdicts = {
        str(entry["policy_id"]): str(entry["verdict"])
        for entry in completion_report.get("policy_verdicts", ())
    }
    for policy_id in ("current_release_v1", "full_vision_v1", "shipped_releasable_v1"):
        if verdicts.get(policy_id) != "ready":
            reason_codes.append(f"{policy_id}_not_ready")

    incomplete_rows = [
        row
        for row in completion_report.get("capability_rows", ())
        if row.get("status") != "complete"
    ]
    if incomplete_rows:
        reason_codes.append("completion_report_has_incomplete_rows")

    required_surface_ids = [
        "retained_core_release",
        "probabilistic_forecast_surface",
        "algorithmic_backend",
        "composition_operator_semantics",
        "shared_plus_local_decomposition",
        "mechanistic_lane",
        "external_evidence_ingestion",
        "robustness_lane",
        "portfolio_orchestration",
    ]
    if full_suite is None:
        reason_codes.append("full_vision_suite_evidence_missing")
    else:
        surfaced = {
            str(entry["surface_id"])
            for entry in full_suite.get("surface_statuses", ())
        }
        if set(required_surface_ids) - surfaced:
            reason_codes.append("full_vision_surface_coverage_incomplete")

    if current_suite is None:
        reason_codes.append("current_release_suite_evidence_missing")

    status = "ready" if not reason_codes else "blocked"
    payload = {
        "report_id": "research_readiness_certification_v1",
        "authority_snapshot_id": _load_packaged_authority_snapshot()[
            "authority_snapshot_id"
        ],
        "command_contract_id": _load_packaged_command_contract()["command_contract_id"],
        "generated_at_utc": _now_utc_timestamp(),
        "status": status,
        "reason_codes": reason_codes,
        "required_surface_ids": required_surface_ids,
        "policy_verdicts": verdicts,
    }
    report_path = _write_json(
        _report_path(workspace_root, _RESEARCH_READINESS_REPORT_PATH),
        payload,
    )
    return ResearchReadinessCertificationResult(
        project_root=workspace_root,
        report_path=report_path,
        status=status,
        reason_codes=tuple(reason_codes),
    )


def load_packaged_release_policies(
    *,
    project_root: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    traversable = resources.files("euclid._assets")
    for part in Path(_PACKAGED_READINESS_POLICY_DIRECTORY).parts:
        traversable = traversable.joinpath(part)
    for child in traversable.iterdir():
        if not child.name.endswith(".yaml"):
            continue
        payload = yaml.safe_load(child.read_text(encoding="utf-8"))
        if payload.get("kind") != "packaged_readiness_policy":
            continue
        policies[str(payload["policy_id"])] = payload
    del project_root
    return policies


def _resolve_asset_root(project_root: Path | str | None) -> Path:
    return resolve_asset_root(project_root)


def _resolve_workspace_root(project_root: Path | str | None) -> Path:
    return resolve_runtime_workspace_root(project_root)


def _resolve_checkout_root(project_root: Path | str | None) -> Path:
    return resolve_checkout_root(project_root)


def _load_packaged_yaml(relative_path: str) -> dict[str, Any]:
    traversable = resources.files("euclid._assets")
    for part in Path(relative_path).parts:
        traversable = traversable.joinpath(part)
    return yaml.safe_load(traversable.read_text(encoding="utf-8"))


def _load_packaged_readiness_schema() -> dict[str, Any]:
    return _load_packaged_yaml(_PACKAGED_READINESS_SCHEMA_PATH)


def _load_packaged_completion_report_schema() -> dict[str, Any]:
    return _load_packaged_yaml(_PACKAGED_COMPLETION_REPORT_SCHEMA_PATH)


def _load_completion_regression_policy(
    *,
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    policy_path = _resolve_asset_root(project_root) / _COMPLETION_REGRESSION_POLICY_PATH
    return yaml.safe_load(policy_path.read_text(encoding="utf-8"))


def _load_suite_summary_payload(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_scope_evidence_bundles(
    *,
    workspace_root: Path,
) -> dict[str, dict[str, Any]]:
    bundle_paths = {
        "current_release": _report_path(workspace_root, _CURRENT_RELEASE_SUITE_EVIDENCE_PATH),
        "full_vision": _report_path(workspace_root, _FULL_VISION_SUITE_EVIDENCE_PATH),
        "shipped_releasable": _clean_install_report_path(workspace_root),
    }
    bundles: dict[str, dict[str, Any]] = {}
    for scope_id, path in bundle_paths.items():
        payload = _load_json_if_present(path)
        if payload is not None:
            bundles[scope_id] = payload
    return bundles


def _row_evidence_bundle_ids(
    *,
    row_id: str,
    closure_row: Mapping[str, Any] | None,
    bundles_by_scope: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    if closure_row is None:
        return ()
    closing_scope_ids = tuple(
        str(scope_id) for scope_id in closure_row.get("closing_evidence_scope_ids", ())
    )
    bundle_ids = tuple(
        str(bundles_by_scope[scope_id]["evidence_bundle_id"])
        for scope_id in closing_scope_ids
        if scope_id in bundles_by_scope
    )
    if len(bundle_ids) > 1:
        if closure_row.get("shared_provenance_mode") != "declared_shared":
            raise ValueError(
                f"{row_id} references multiple scope bundles without declared shared provenance"
            )
        if not closure_row.get("shared_provenance_id"):
            raise ValueError(
                f"{row_id} declares shared provenance but omits shared_provenance_id"
            )
    return bundle_ids


def _build_completion_report(
    *,
    policies: Mapping[str, Mapping[str, Any]],
    policy_judgments: Mapping[str, ReadinessJudgment],
    current_release_suite,
    full_vision_suite,
    current_release_suite_judgment: ReadinessJudgment,
    full_vision_suite_judgment: ReadinessJudgment,
    notebook_smoke: NotebookSmokeResult,
    clean_install_report: Mapping[str, Any],
    workspace_root: Path,
) -> dict[str, Any]:
    matrix_payload = _load_packaged_yaml(str(policies["full_vision_v1"]["matrix_path"]))
    current_row_ids = {
        str(row_id) for row_id in policies["current_release_v1"].get("required_row_ids", ())
    }
    closure_rows_by_id = {
        str(row["capability_row_id"]): row
        for row in _load_packaged_closure_map().get("rows", ())
    }
    repo_test_matrix_report = _load_json_report_if_present(
        _report_path(workspace_root, _REPO_TEST_MATRIX_REPORT_PATH)
    )
    operator_run_report = _load_json_report_if_present(
        _report_path(workspace_root, _FULL_VISION_OPERATOR_RUN_EVIDENCE_PATH)
    )
    operator_replay_report = _load_json_report_if_present(
        _report_path(workspace_root, _FULL_VISION_OPERATOR_REPLAY_EVIDENCE_PATH)
    )
    operator_runtime_semantics = _load_operator_runtime_semantics(
        operator_run_report=operator_run_report,
        operator_replay_report=operator_replay_report,
    )
    scope_evidence_bundles = _load_scope_evidence_bundles(workspace_root=workspace_root)
    completion_rows = tuple(
        _evaluate_completion_row(
            row=row,
            suite_result=(
                current_release_suite
                if str(row["row_id"]) in current_row_ids
                else full_vision_suite
            ),
            suite_readiness_judgment=(
                current_release_suite_judgment
                if str(row["row_id"]) in current_row_ids
                else full_vision_suite_judgment
            ),
            notebook_smoke=notebook_smoke,
            clean_install_report=clean_install_report,
            closure_row=closure_rows_by_id.get(str(row["row_id"])),
            bundles_by_scope=scope_evidence_bundles,
            repo_test_matrix_report=repo_test_matrix_report,
            operator_run_report=operator_run_report,
            operator_replay_report=operator_replay_report,
            operator_runtime_semantics=operator_runtime_semantics,
        )
        for row in matrix_payload.get("rows", ())
    )
    rows_by_id = {row.row_id: row for row in completion_rows}
    completion_values = _build_completion_values(
        rows_by_id=rows_by_id,
        current_policy=policies["current_release_v1"],
        full_policy=policies["full_vision_v1"],
        shipped_policy=policies["shipped_releasable_v1"],
        clean_install_report=clean_install_report,
    )
    authority_snapshot = _load_packaged_authority_snapshot()
    command_contract = _load_packaged_command_contract()
    closure_map = _load_packaged_closure_map()
    traceability = _load_packaged_traceability()
    fixture_spec = _load_packaged_fixture_spec()
    payload = {
        "report_id": "euclid_completion_report_v1",
        "authority_snapshot_id": authority_snapshot["authority_snapshot_id"],
        "command_contract_id": command_contract["command_contract_id"],
        "closure_map_id": closure_map["closure_map_id"],
        "traceability_id": traceability["traceability_id"],
        "fixture_spec_id": fixture_spec["fixture_spec_id"],
        "generated_at": _now_utc_timestamp(),
        "policy_verdicts": [
            _build_policy_verdict_payload(
                policy=policies[policy_id],
                judgment=policy_judgments[policy_id],
            )
            for policy_id in (
                "current_release_v1",
                "full_vision_v1",
                "shipped_releasable_v1",
            )
        ],
        "scope_evidence_bundles": [
            scope_evidence_bundles[scope_id]
            for scope_id in ("current_release", "full_vision", "shipped_releasable")
            if scope_id in scope_evidence_bundles
        ],
        "completion_values": completion_values,
        "clean_install_certification": {
            "surface_completion": float(clean_install_report["surface_completion"]),
            "surfaces": [
                {
                    "surface_id": str(surface["surface_id"]),
                    "status": str(surface["status"]),
                    "reason_codes": list(surface["reason_codes"]),
                    "evidence_refs": list(surface["evidence_refs"]),
                }
                for surface in clean_install_report["surfaces"]
            ],
        },
        "capability_rows": [
            {
                "row_id": row.row_id,
                "status": row.status,
                "reason_codes": list(row.reason_codes),
                "evidence_refs": list(row.evidence_refs),
                "required_evidence_classes": list(row.required_closing_classes),
                "available_evidence_classes": list(row.available_evidence_classes),
                "non_closing_evidence_classes": list(row.non_closing_evidence_classes),
                "evidence_bundle_ids": list(row.evidence_bundle_ids),
            }
            for row in completion_rows
        ],
        "residual_risk_codes": _build_residual_risk_codes(
            completion_rows=completion_rows,
            policy_judgments=policy_judgments,
        ),
        "unresolved_blockers": [
            {
                "capability_row_id": row.row_id,
                "proof_status": row.proof_status,
                "reason_codes": list(row.reason_codes),
                "evidence_refs": list(row.evidence_refs),
            }
            for row in completion_rows
            if row.proof_status is not None
        ],
        "confidence": _build_completion_report_confidence(
            completion_rows=completion_rows,
            completion_values=completion_values,
        ),
    }
    _validate_completion_report_payload(payload)
    return payload


def _evaluate_completion_row(
    *,
    row: Mapping[str, Any],
    suite_result,
    suite_readiness_judgment: ReadinessJudgment,
    notebook_smoke: NotebookSmokeResult,
    clean_install_report: Mapping[str, Any],
    closure_row: Mapping[str, Any] | None,
    bundles_by_scope: Mapping[str, Mapping[str, Any]],
    repo_test_matrix_report: Mapping[str, Any] | None = None,
    operator_run_report: Mapping[str, Any] | None = None,
    operator_replay_report: Mapping[str, Any] | None = None,
    operator_runtime_semantics: Mapping[str, Any] | None = None,
) -> _CompletionLedgerRow:
    row_id = str(row["row_id"])
    evidence = _row_evidence_payload(
        capability_type=str(row["capability_type"]),
        capability_id=str(row["capability_id"]),
        suite_summary=_load_suite_summary_payload(suite_result.summary_path),
        suite_readiness_judgment=suite_readiness_judgment,
        notebook_smoke=notebook_smoke,
        clean_install_report=clean_install_report,
        repo_test_matrix_report=repo_test_matrix_report,
        operator_run_report=operator_run_report,
        operator_replay_report=operator_replay_report,
        operator_runtime_semantics=operator_runtime_semantics,
    )
    required_classes = tuple(
        sorted(str(value) for value in row.get("minimum_closing_evidence_classes", ()))
    )
    available_classes = tuple(
        sorted(str(value) for value in evidence["available_classes"])
    )
    non_closing_classes = tuple(
        sorted(str(value) for value in evidence["non_closing_classes"])
    )
    explicit_status = str(evidence.get("status", "missing"))
    if explicit_status == "failed":
        status = "blocked"
        proof_status = "failed_proof"
    elif set(required_classes) <= set(available_classes):
        status = "complete"
        proof_status = None
    elif available_classes or non_closing_classes:
        status = "partial"
        proof_status = "missing_proof"
    else:
        status = "blocked"
        proof_status = "missing_proof"
    reason_codes = (
        ()
        if status == "complete"
        else _row_reason_codes(
            row_id=row_id,
            required_classes=required_classes,
            available_classes=available_classes,
            non_closing_classes=non_closing_classes,
            explicit_status=explicit_status,
        )
    )
    return _CompletionLedgerRow(
        row_id=row_id,
        status=status,
        required_closing_classes=required_classes,
        available_evidence_classes=available_classes,
        non_closing_evidence_classes=non_closing_classes,
        reason_codes=reason_codes,
        evidence_refs=tuple(
            _row_evidence_refs(
                row=row,
            evidence=evidence,
            suite_result=suite_result,
            notebook_smoke=notebook_smoke,
            clean_install_report=clean_install_report,
            repo_test_matrix_report=repo_test_matrix_report,
            operator_run_report=operator_run_report,
            operator_replay_report=operator_replay_report,
        )
        ),
        evidence_bundle_ids=_row_evidence_bundle_ids(
            row_id=row_id,
            closure_row=closure_row,
            bundles_by_scope=bundles_by_scope,
        ),
        proof_status=proof_status,
    )


def _build_completion_values(
    *,
    rows_by_id: Mapping[str, _CompletionLedgerRow],
    current_policy: Mapping[str, Any],
    full_policy: Mapping[str, Any],
    shipped_policy: Mapping[str, Any],
    clean_install_report: Mapping[str, Any],
) -> dict[str, float]:
    current_row_ids = tuple(
        str(row_id) for row_id in current_policy.get("required_row_ids", ())
    )
    full_row_ids = tuple(
        str(row_id) for row_id in full_policy.get("required_row_ids", ())
    )
    shipped_row_ids = tuple(
        str(row_id) for row_id in shipped_policy.get("required_row_ids", ())
    )
    shipped_surface_completion = _clean_install_surface_completion(clean_install_report)
    shipped_completed = (
        sum(
            1
            for row_id in shipped_row_ids
            if row_id in rows_by_id and rows_by_id[row_id].status == "complete"
        )
        + shipped_surface_completion[0]
    )
    shipped_total = len(shipped_row_ids) + shipped_surface_completion[1]
    return {
        "full_vision_completion": _completion_ratio(
            row_ids=full_row_ids,
            rows_by_id=rows_by_id,
        ),
        "current_gate_completion": _completion_ratio(
            row_ids=current_row_ids,
            rows_by_id=rows_by_id,
        ),
        "shipped_releasable_completion": round(
            shipped_completed / max(shipped_total, 1), 6
        ),
    }


def _clean_install_surface_completion(
    clean_install_report: Mapping[str, Any],
) -> tuple[int, int]:
    surfaces = tuple(clean_install_report.get("surfaces", ()))
    passed = sum(1 for surface in surfaces if surface.get("status") == "passed")
    return passed, len(surfaces)


def _completion_ratio(
    *,
    row_ids: tuple[str, ...],
    rows_by_id: Mapping[str, _CompletionLedgerRow],
) -> float:
    if not row_ids:
        return 0.0
    completed = sum(
        1
        for row_id in row_ids
        if row_id in rows_by_id and rows_by_id[row_id].status == "complete"
    )
    return round(completed / len(row_ids), 6)


def _build_policy_verdict_payload(
    *,
    policy: Mapping[str, Any],
    judgment: ReadinessJudgment,
) -> dict[str, Any]:
    return {
        "policy_id": str(policy["policy_id"]),
        "verdict": judgment.final_verdict,
        "reason_codes": list(judgment.reason_codes),
        "evidence_refs": _policy_evidence_refs(
            policy=policy,
            judgment=judgment,
        ),
    }


def _policy_evidence_refs(
    *,
    policy: Mapping[str, Any],
    judgment: ReadinessJudgment,
) -> list[str]:
    refs = [f"policy:{policy['policy_id']}"]
    refs.extend(
        f"doc:{authority_ref}"
        for authority_ref in policy.get("scope_authority_refs", ())
    )
    failing_gate_refs = [
        f"gate:{gate.gate_id}"
        for gate in judgment.gate_results
        if gate.status in {"failed", "missing"}
    ]
    if failing_gate_refs:
        refs.extend(failing_gate_refs)
    elif judgment.gate_results:
        refs.append(f"gate:{judgment.gate_results[0].gate_id}")
    return _ordered_unique(refs)


def _build_residual_risk_codes(
    *,
    completion_rows: tuple[_CompletionLedgerRow, ...],
    policy_judgments: Mapping[str, ReadinessJudgment],
) -> list[str]:
    codes = []
    for judgment in policy_judgments.values():
        codes.extend(judgment.reason_codes)
    for row in completion_rows:
        if row.status != "complete":
            codes.extend(row.reason_codes)
    return sorted(set(codes))


def _build_completion_report_confidence(
    *,
    completion_rows: tuple[_CompletionLedgerRow, ...],
    completion_values: Mapping[str, float],
) -> dict[str, Any]:
    total_required_slots = sum(
        len(row.required_closing_classes) for row in completion_rows
    )
    satisfied_required_slots = sum(
        len(set(row.required_closing_classes) & set(row.available_evidence_classes))
        for row in completion_rows
    )
    average_completion = sum(completion_values.values()) / max(
        len(completion_values), 1
    )
    evidence_slot_coverage = satisfied_required_slots / max(total_required_slots, 1)
    score = round((average_completion + evidence_slot_coverage) / 2.0, 6)
    reason_codes = []
    if (
        completion_values["full_vision_completion"]
        < completion_values["current_gate_completion"]
    ):
        reason_codes.append("current_release_scope_narrower_than_full_vision")
    if any(row.proof_status == "failed_proof" for row in completion_rows):
        reason_codes.append("failed_proof_present")
    if any(row.proof_status == "missing_proof" for row in completion_rows):
        reason_codes.append("missing_proof_present")
    return {
        "score": score,
        "reason_codes": reason_codes,
    }


def _validate_completion_report_payload(payload: Mapping[str, Any]) -> None:
    schema = _load_packaged_completion_report_schema()
    required_fields = set(schema["required_fields"])
    policy_verdict_required_fields = set(schema["policy_verdict_required_fields"])
    scope_evidence_bundle_required_fields = set(
        schema["scope_evidence_bundle_required_fields"]
    )
    capability_row_required_fields = set(schema["capability_row_required_fields"])
    if not required_fields <= set(payload):
        missing_fields = sorted(required_fields - set(payload))
        raise ValueError(
            "completion report is missing required fields: " + ", ".join(missing_fields)
        )
    for verdict in payload["policy_verdicts"]:
        if not policy_verdict_required_fields <= set(verdict):
            missing_fields = sorted(policy_verdict_required_fields - set(verdict))
            raise ValueError(
                "completion report policy verdict is missing required fields: "
                + ", ".join(missing_fields)
            )
    for bundle in payload["scope_evidence_bundles"]:
        if not scope_evidence_bundle_required_fields <= set(bundle):
            missing_fields = sorted(scope_evidence_bundle_required_fields - set(bundle))
            raise ValueError(
                "completion report scope evidence bundle is missing required fields: "
                + ", ".join(missing_fields)
            )
    completion_field_ids = {
        str(entry["id"]) for entry in schema["completion_value_fields"]
    }
    if completion_field_ids != set(payload["completion_values"]):
        raise ValueError("completion report emitted unexpected completion values")
    clean_install = payload["clean_install_certification"]
    if not isinstance(clean_install["surface_completion"], (int, float)):
        raise ValueError(
            "clean-install certification must expose a numeric surface completion"
        )
    surface_status_values = set(schema["clean_install_surface_status_values"])
    for surface in clean_install["surfaces"]:
        if surface["status"] not in surface_status_values:
            raise ValueError(
                f"unsupported clean-install surface status: {surface['status']!r}"
            )
    capability_status_values = set(schema["capability_status_values"])
    for row in payload["capability_rows"]:
        if not capability_row_required_fields <= set(row):
            missing_fields = sorted(capability_row_required_fields - set(row))
            raise ValueError(
                "completion report capability row is missing required fields: "
                + ", ".join(missing_fields)
            )
        if row["status"] not in capability_status_values:
            raise ValueError(f"unsupported capability row status: {row['status']!r}")
    proof_status_values = set(schema["proof_status_values"])
    for blocker in payload["unresolved_blockers"]:
        if blocker["proof_status"] not in proof_status_values:
            raise ValueError(f"unsupported proof status: {blocker['proof_status']!r}")


def _write_completion_report(
    *,
    workspace_root: Path,
    payload: Mapping[str, Any],
) -> Path:
    report_path = workspace_root / "build" / "reports" / "completion-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def _row_reason_codes(
    *,
    row_id: str,
    required_classes: tuple[str, ...],
    available_classes: tuple[str, ...],
    non_closing_classes: tuple[str, ...],
    explicit_status: str,
) -> tuple[str, ...]:
    base_code = row_id.replace(":", ".")
    codes = []
    if explicit_status == "failed":
        codes.append(f"{base_code}_failed_proof")
    codes.extend(
        f"{base_code}_missing_{required_class}"
        for required_class in required_classes
        if required_class not in available_classes
    )
    codes.extend(
        f"{base_code}_non_closing_{evidence_class}"
        for evidence_class in non_closing_classes
    )
    if not codes:
        codes.append(f"{base_code}_insufficient_proof")
    return tuple(_ordered_unique(codes))


def _row_evidence_refs(
    *,
    row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    suite_result,
    notebook_smoke: NotebookSmokeResult,
    clean_install_report: Mapping[str, Any],
    repo_test_matrix_report: Mapping[str, Any] | None = None,
    operator_run_report: Mapping[str, Any] | None = None,
    operator_replay_report: Mapping[str, Any] | None = None,
) -> list[str]:
    refs = [f"matrix:{row['row_id']}"]
    refs.extend(
        f"doc:{authority_ref}" for authority_ref in row.get("governing_doc_refs", ())
    )
    if suite_result.summary_path.is_file():
        refs.append(f"artifact:{suite_result.summary_path}")
    details = evidence.get("details", {})
    if isinstance(details, Mapping):
        refs.extend(
            f"benchmark_task:{task_id}"
            for task_id in details.get("task_ids", ())
            if task_id
        )
        refs.extend(
            f"benchmark_task:{task_id}"
            for task_id in details.get("covered_task_ids", ())
            if task_id
        )
        if details.get("track_id"):
            refs.append(f"benchmark_track:{details['track_id']}")
        if details.get("surface_id"):
            refs.append(f"benchmark_surface:{details['surface_id']}")
        if details.get("forecast_object_type"):
            refs.append(f"forecast_object_type:{details['forecast_object_type']}")
        if details.get("composition_operator"):
            refs.append(f"composition_operator:{details['composition_operator']}")
        if details.get("search_class"):
            refs.append(f"search_class:{details['search_class']}")
        if details.get("lane_id"):
            refs.append(f"evidence_lane:{details['lane_id']}")
        if details.get("reducer_family_id"):
            refs.append(f"reducer_family:{details['reducer_family_id']}")
        if details.get("run_support_object_id"):
            refs.append(f"run_support_object:{details['run_support_object_id']}")
        if details.get("admissibility_rule_id"):
            refs.append(f"admissibility_rule:{details['admissibility_rule_id']}")
        if details.get("lifecycle_artifact_id"):
            refs.append(f"lifecycle_artifact:{details['lifecycle_artifact_id']}")
        if details.get("operator_runtime_surface_id"):
            refs.append(
                f"operator_runtime_surface:{details['operator_runtime_surface_id']}"
            )
        if details.get("clean_install_surface_id"):
            refs.append(f"clean_install_surface:{details['clean_install_surface_id']}")
        if details.get("replay_surface_id"):
            refs.append(f"replay_surface:{details['replay_surface_id']}")
        refs.extend(
            str(ref)
            for ref in details.get("evidence_refs", ())
            if isinstance(ref, str) and ref
        )
    if (
        "notebook_smoke" in evidence.get("non_closing_classes", ())
        and notebook_smoke.summary_path.is_file()
    ):
        refs.append(f"artifact:{notebook_smoke.summary_path}")
    if str(row["row_id"]) == "evidence_lane:readiness_and_closure":
        refs.extend(
            ref
            for surface in clean_install_report.get("surfaces", ())
            if isinstance(surface, Mapping)
            for ref in surface.get("evidence_refs", ())
            if "clean-install-certification.json" in ref or "clean-install" in ref
        )
        report_path = clean_install_report.get("report_path")
        if isinstance(report_path, str) and report_path:
            refs.append(f"artifact:{report_path}")
    if (
        repo_test_matrix_report is not None
        and isinstance(repo_test_matrix_report.get("__report_path__"), str)
    ):
        refs.append(f"artifact:{repo_test_matrix_report['__report_path__']}")
    for report in (operator_run_report, operator_replay_report):
        if report is None:
            continue
        report_path = report.get("__report_path__")
        if isinstance(report_path, str) and report_path:
            refs.append(f"artifact:{report_path}")
    return _ordered_unique(refs)


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _semantic_summary_mapping(
    suite_summary: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    semantic_summary = (suite_summary or {}).get("semantic_summary", {})
    if isinstance(semantic_summary, Mapping):
        return semantic_summary
    return {}


def _semantic_summary_ids(
    suite_summary: Mapping[str, Any] | None,
    key: str,
) -> set[str]:
    values = _semantic_summary_mapping(suite_summary).get(key, ())
    if isinstance(values, Mapping):
        return {str(value) for value in values.keys()}
    return {
        str(value)
        for value in values
        if value is not None and str(value) != ""
    }


def _clean_install_surfaces_by_id(
    clean_install_report: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(surface["surface_id"]): surface
        for surface in clean_install_report.get("surfaces", ())
        if isinstance(surface, Mapping) and isinstance(surface.get("surface_id"), str)
    }


def _mark_clean_install_packaging_evidence(
    *,
    evidence: dict[str, Any],
    clean_install_surface: Mapping[str, Any] | None,
) -> None:
    if clean_install_surface is None:
        return
    if clean_install_surface.get("status") == "failed":
        evidence["status"] = "failed"
    if clean_install_surface.get("status") == "passed":
        evidence["available_classes"].add("packaging_install")


def _judge_current_release_gate_readiness(
    *,
    suite_readiness_judgment: ReadinessJudgment,
) -> ReadinessJudgment:
    schema = _load_packaged_readiness_schema()
    required_gate_ids = tuple(
        str(item["gate_id"])
        for item in schema["retained_core_release"]["required_gates"]
    )
    gate_results = tuple(
        _resolve_release_gate_result(
            gate_id=gate_id,
            suite_readiness_judgment=suite_readiness_judgment,
        )
        for gate_id in required_gate_ids
    )
    return judge_readiness(
        judgment_id="current_release_v1_gates",
        gate_results=gate_results,
        required_gate_ids=required_gate_ids,
    )


def _judge_full_vision_gate_readiness(
    *,
    suite_readiness_judgment: ReadinessJudgment,
) -> ReadinessJudgment:
    schema = _load_packaged_readiness_schema()
    required_gate_ids = tuple(
        str(item["gate_id"]) for item in schema["full_vision"]["required_gates"]
    )
    gate_results = tuple(
        _resolve_release_gate_result(
            gate_id=gate_id,
            suite_readiness_judgment=suite_readiness_judgment,
        )
        for gate_id in required_gate_ids
    )
    return judge_readiness(
        judgment_id="full_vision_v1_gates",
        gate_results=gate_results,
        required_gate_ids=required_gate_ids,
    )


def _resolve_release_gate_result(
    *,
    gate_id: str,
    suite_readiness_judgment: ReadinessJudgment,
) -> ReadinessGateResult:
    gate_by_id = gate_results_by_id(suite_readiness_judgment.gate_results)
    source_gate_id = gate_id
    if gate_id == "benchmarks.rediscovery":
        source_gate_id = "track.rediscovery"
    elif gate_id == "benchmarks.predictive_generalization":
        source_gate_id = "track.predictive_generalization"
    elif gate_id == "benchmarks.adversarial_honesty":
        source_gate_id = "track.adversarial_honesty"
    source_gate = gate_by_id.get(source_gate_id)
    if source_gate is None:
        return ReadinessGateResult(
            gate_id=gate_id,
            status="missing",
            required=True,
            summary=f"{gate_id} evidence has not been captured yet.",
            evidence={},
        )
    return ReadinessGateResult(
        gate_id=gate_id,
        status=source_gate.status,
        required=True,
        summary=source_gate.summary,
        evidence=dict(source_gate.evidence),
    )


def _judge_packaged_policy_rows(
    *,
    policy: Mapping[str, Any],
    suite_result,
    suite_readiness_judgment: ReadinessJudgment,
    notebook_smoke: NotebookSmokeResult,
    clean_install_report: Mapping[str, Any],
    repo_test_matrix_report: Mapping[str, Any] | None = None,
    operator_run_report: Mapping[str, Any] | None = None,
    operator_replay_report: Mapping[str, Any] | None = None,
    operator_runtime_semantics: Mapping[str, Any] | None = None,
) -> ReadinessJudgment:
    matrix_payload = _load_packaged_yaml(str(policy["matrix_path"]))
    rows_by_id = {str(row["row_id"]): row for row in matrix_payload.get("rows", ())}
    gate_results = tuple(
        _evaluate_policy_row(
            row=rows_by_id[row_id],
            suite_result=suite_result,
            suite_readiness_judgment=suite_readiness_judgment,
            notebook_smoke=notebook_smoke,
            clean_install_report=clean_install_report,
            repo_test_matrix_report=repo_test_matrix_report,
            operator_run_report=operator_run_report,
            operator_replay_report=operator_replay_report,
            operator_runtime_semantics=operator_runtime_semantics,
        )
        for row_id in tuple(policy.get("required_row_ids", ()))
        if row_id in rows_by_id
    )
    required_gate_ids = tuple(
        _policy_gate_id(str(row_id))
        for row_id in tuple(policy.get("required_row_ids", ()))
    )
    return judge_readiness(
        judgment_id=str(policy["policy_id"]),
        gate_results=gate_results,
        required_gate_ids=required_gate_ids,
    )


def _evaluate_policy_row(
    *,
    row: Mapping[str, Any],
    suite_result,
    suite_readiness_judgment: ReadinessJudgment,
    notebook_smoke: NotebookSmokeResult,
    clean_install_report: Mapping[str, Any],
    repo_test_matrix_report: Mapping[str, Any] | None = None,
    operator_run_report: Mapping[str, Any] | None = None,
    operator_replay_report: Mapping[str, Any] | None = None,
    operator_runtime_semantics: Mapping[str, Any] | None = None,
) -> ReadinessGateResult:
    row_id = str(row["row_id"])
    capability_type = str(row["capability_type"])
    capability_id = str(row["capability_id"])
    required_classes = tuple(
        str(value) for value in row.get("minimum_closing_evidence_classes", ())
    )
    suite_summary = _load_suite_summary_payload(suite_result.summary_path)
    evidence = _row_evidence_payload(
        capability_type=capability_type,
        capability_id=capability_id,
        suite_summary=suite_summary,
        suite_readiness_judgment=suite_readiness_judgment,
        notebook_smoke=notebook_smoke,
        clean_install_report=clean_install_report,
        repo_test_matrix_report=repo_test_matrix_report,
        operator_run_report=operator_run_report,
        operator_replay_report=operator_replay_report,
        operator_runtime_semantics=operator_runtime_semantics,
    )
    available_classes = tuple(evidence["available_classes"])
    non_closing_classes = tuple(evidence["non_closing_classes"])
    explicit_status = evidence["status"]
    if explicit_status == "failed":
        status = "failed"
    elif set(required_classes) <= set(available_classes):
        status = "passed"
    else:
        status = "missing"
    return ReadinessGateResult(
        gate_id=_policy_gate_id(row_id),
        status=status,
        required=True,
        summary=(
            f"{row_id} requires {', '.join(required_classes) or 'no closing proof'}; "
            f"available proof classes: {', '.join(available_classes) or 'none'}."
        ),
        evidence={
            "row_id": row_id,
            "required_closing_classes": list(required_classes),
            "available_classes": list(available_classes),
            "non_closing_classes": list(non_closing_classes),
            "details": dict(evidence["details"]),
        },
    )


def _row_evidence_payload(
    *,
    capability_type: str,
    capability_id: str,
    suite_summary: Mapping[str, Any] | None,
    suite_readiness_judgment: ReadinessJudgment,
    notebook_smoke: NotebookSmokeResult,
    clean_install_report: Mapping[str, Any],
    repo_test_matrix_report: Mapping[str, Any] | None = None,
    operator_run_report: Mapping[str, Any] | None = None,
    operator_replay_report: Mapping[str, Any] | None = None,
    operator_runtime_semantics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gate_by_id = gate_results_by_id(suite_readiness_judgment.gate_results)
    task_rows = tuple(
        row
        for row in (suite_summary or {}).get("task_results", ())
        if isinstance(row, Mapping)
    )
    surface_rows = {
        str(row["surface_id"]): row
        for row in (suite_summary or {}).get("surface_statuses", ())
        if isinstance(row, Mapping) and isinstance(row.get("surface_id"), str)
    }
    search_rows = {
        str(row["search_class"]): row
        for row in (suite_summary or {}).get("search_class_coverage", ())
        if isinstance(row, Mapping) and isinstance(row.get("search_class"), str)
    }
    composition_rows = {
        str(row["composition_operator"]): row
        for row in (suite_summary or {}).get("composition_operator_coverage", ())
        if isinstance(row, Mapping) and isinstance(row.get("composition_operator"), str)
    }
    matching_task_rows = tuple(
        row
        for row in task_rows
        if row.get("forecast_object_type") == capability_id
        or row.get("track_id") == capability_id
    )
    operator_runtime_semantics = dict(operator_runtime_semantics or {})
    suite_reducer_family_ids = _suite_reducer_family_ids(suite_summary)
    suite_lifecycle_artifact_ids = _suite_lifecycle_artifact_ids(suite_summary)
    run_support_object_ids = _semantic_summary_ids(
        suite_summary, "run_support_object_ids"
    ) | {
        str(value)
        for value in operator_runtime_semantics.get("run_support_object_ids", ())
        if value
    }
    admissibility_rule_ids = _semantic_summary_ids(
        suite_summary, "admissibility_rule_ids"
    ) | {
        str(value)
        for value in operator_runtime_semantics.get("admissibility_rule_ids", ())
        if value
    }
    lifecycle_artifact_ids = suite_lifecycle_artifact_ids | {
        str(value)
        for value in operator_runtime_semantics.get("lifecycle_artifact_ids", ())
        if value
    }
    operator_runtime_surface_ids = _semantic_summary_ids(
        suite_summary, "operator_runtime_surface_ids"
    ) | {
        str(value)
        for value in operator_runtime_semantics.get("operator_runtime_surface_ids", ())
        if value
    }
    replay_surface_ids = _semantic_summary_ids(
        suite_summary, "replay_surface_ids"
    ) | {
        str(value)
        for value in operator_runtime_semantics.get("replay_surface_ids", ())
        if value
    }
    clean_install_surfaces = _clean_install_surfaces_by_id(clean_install_report)
    repo_matrix_passed = repo_test_matrix_report is not None and (
        repo_test_matrix_report.get("passed") is True
    )
    suite_task_replay_verified = bool(task_rows) and all(
        row.get("replay_verification") == "verified" for row in task_rows
    )
    task_replay_verified = bool(matching_task_rows) and all(
        row.get("replay_verification") == "verified" for row in matching_task_rows
    )
    evidence: dict[str, Any] = {
        "status": "missing",
        "available_classes": set(),
        "non_closing_classes": set(),
        "details": {},
    }
    if capability_type == "benchmark_track":
        gate = gate_by_id.get(f"track.{capability_id}")
        if gate is not None and gate.status == "failed":
            evidence["status"] = "failed"
        if gate is not None and gate.status == "passed":
            evidence["available_classes"].add("benchmark_semantic")
        if matching_task_rows and all(
            row.get("replay_verification") == "verified" for row in matching_task_rows
        ):
            evidence["available_classes"].add("replay")
        evidence["details"] = {
            "track_id": capability_id,
            "task_ids": [str(row.get("task_id", "")) for row in matching_task_rows],
        }
        return evidence
    if capability_type == "benchmark_surface":
        gate = gate_by_id.get(f"surface.{capability_id}")
        surface_row = surface_rows.get(capability_id, {})
        if gate is not None and gate.status == "failed":
            evidence["status"] = "failed"
        if surface_row.get("benchmark_status") == "passed":
            evidence["available_classes"].add("benchmark_semantic")
        if surface_row.get("replay_status") == "passed":
            evidence["available_classes"].add("replay")
        evidence["details"] = dict(surface_row)
        return evidence
    if capability_type == "search_class":
        search_row = search_rows.get(capability_id, {})
        if search_row:
            evidence["available_classes"].add("semantic_runtime")
            evidence["available_classes"].add("benchmark_semantic")
            if bool(search_row.get("replay_verified")):
                evidence["available_classes"].add("replay")
        evidence["details"] = dict(search_row)
        return evidence
    if capability_type == "composition_operator":
        composition_row = composition_rows.get(capability_id, {})
        if composition_row:
            evidence["available_classes"].add("semantic_runtime")
            evidence["available_classes"].add("benchmark_semantic")
            if bool(composition_row.get("replay_verified")):
                evidence["available_classes"].add("replay")
        evidence["details"] = dict(composition_row)
        return evidence
    if capability_type == "reducer_family":
        if capability_id in suite_reducer_family_ids:
            evidence["available_classes"].add("semantic_runtime")
            evidence["available_classes"].add("benchmark_semantic")
            if replay_surface_ids or any(
                row.get("replay_verification") == "verified" for row in task_rows
            ):
                evidence["available_classes"].add("replay")
        evidence["details"] = {
            "reducer_family_id": capability_id,
            "task_ids": [str(row.get("task_id", "")) for row in task_rows],
        }
        return evidence
    if capability_type == "forecast_object_type":
        if matching_task_rows:
            evidence["available_classes"].add("semantic_runtime")
            evidence["available_classes"].add("benchmark_semantic")
            if task_replay_verified:
                evidence["available_classes"].add("replay")
        if capability_id in notebook_smoke.probabilistic_case_ids:
            evidence["non_closing_classes"].add("notebook_smoke")
        evidence["details"] = {
            "forecast_object_type": capability_id,
            "task_ids": [str(row.get("task_id", "")) for row in matching_task_rows],
        }
        return evidence
    if capability_type == "evidence_lane":
        if repo_matrix_passed:
            evidence["available_classes"].add("governance_spec")
        if capability_id == "predictive_generalization":
            gate = gate_by_id.get("track.predictive_generalization")
            if gate is not None and gate.status == "passed":
                evidence["available_classes"].add("benchmark_semantic")
        elif capability_id == "robustness":
            gate = gate_by_id.get("track.adversarial_honesty")
            if gate is not None and gate.status == "passed":
                evidence["available_classes"].add("benchmark_semantic")
            if repo_matrix_passed:
                evidence["available_classes"].add("golden_regression")
        elif capability_id == "descriptive_compression":
            if repo_matrix_passed:
                evidence["available_classes"].add("golden_regression")
        elif capability_id == "replay_verification":
            if (
                replay_surface_ids
                or all(
                row.get("replay_status") == "passed" for row in surface_rows.values()
                )
            ):
                evidence["available_classes"].add("replay")
        elif capability_id == "readiness_and_closure":
            clean_install_surfaces = tuple(
                surface
                for surface in clean_install_report.get("surfaces", ())
                if isinstance(surface, Mapping)
            )
            if clean_install_surfaces and all(
                surface.get("status") == "passed" for surface in clean_install_surfaces
            ):
                evidence["available_classes"].add("packaging_install")
            if replay_surface_ids:
                evidence["available_classes"].add("replay")
            if notebook_smoke.summary_path.is_file():
                evidence["non_closing_classes"].add("notebook_smoke")
        evidence["details"] = {
            "lane_id": capability_id,
            "clean_install_surface_ids": [
                str(surface.get("surface_id", ""))
                for surface in clean_install_report.get("surfaces", ())
                if isinstance(surface, Mapping)
            ],
            "evidence_refs": list(
                operator_runtime_semantics.get("evidence_refs", ())
            ),
        }
        return evidence
    if capability_type == "run_support_object":
        if capability_id in run_support_object_ids:
            evidence["available_classes"].add("semantic_runtime")
            if replay_surface_ids:
                evidence["available_classes"].add("replay")
        evidence["details"] = {
            "run_support_object_id": capability_id,
            "evidence_refs": list(operator_runtime_semantics.get("evidence_refs", ())),
        }
        return evidence
    if capability_type == "admissibility_rule":
        if capability_id in admissibility_rule_ids:
            evidence["available_classes"].add("semantic_runtime")
            if replay_surface_ids:
                evidence["available_classes"].add("replay")
        evidence["details"] = {
            "admissibility_rule_id": capability_id,
            "evidence_refs": list(operator_runtime_semantics.get("evidence_refs", ())),
        }
        return evidence
    if capability_type == "lifecycle_artifact":
        if capability_id in suite_lifecycle_artifact_ids:
            evidence["available_classes"].add("benchmark_semantic")
        if capability_id in lifecycle_artifact_ids:
            evidence["available_classes"].add("semantic_runtime")
            if (
                (capability_id in suite_lifecycle_artifact_ids and suite_task_replay_verified)
                or replay_surface_ids
            ):
                evidence["available_classes"].add("replay")
        packaging_surface_id = _PACKAGING_LIFECYCLE_ARTIFACT_SURFACES.get(capability_id)
        clean_install_surface = (
            clean_install_surfaces.get(packaging_surface_id)
            if packaging_surface_id is not None
            else None
        )
        _mark_clean_install_packaging_evidence(
            evidence=evidence,
            clean_install_surface=clean_install_surface,
        )
        evidence["details"] = {
            "lifecycle_artifact_id": capability_id,
            "clean_install_surface_id": packaging_surface_id or "",
            "evidence_refs": list(
                operator_runtime_semantics.get("evidence_refs", ())
            )
            + list(
                clean_install_surface.get("evidence_refs", ())
                if clean_install_surface is not None
                else ()
            ),
        }
        return evidence
    if capability_type == "operator_runtime_surface":
        clean_install_surface = clean_install_surfaces.get(capability_id)
        if capability_id in operator_runtime_surface_ids:
            evidence["available_classes"].add("semantic_runtime")
        if replay_surface_ids:
            evidence["available_classes"].add("replay")
        _mark_clean_install_packaging_evidence(
            evidence=evidence,
            clean_install_surface=clean_install_surface,
        )
        evidence["details"] = {
            "operator_runtime_surface_id": capability_id,
            "clean_install_surface_id": capability_id if clean_install_surface else "",
            "evidence_refs": list(
                operator_runtime_semantics.get("evidence_refs", ())
            )
            + list(
                clean_install_surface.get("evidence_refs", ())
                if clean_install_surface is not None
                else ()
            ),
        }
        return evidence
    if capability_type == "clean_install_surface":
        clean_install_surface = clean_install_surfaces.get(capability_id)
        _mark_clean_install_packaging_evidence(
            evidence=evidence,
            clean_install_surface=clean_install_surface,
        )
        evidence["details"] = {
            "clean_install_surface_id": capability_id,
            "reason_codes": list(
                clean_install_surface.get("reason_codes", ())
                if clean_install_surface is not None
                else ()
            ),
            "evidence_refs": list(
                clean_install_surface.get("evidence_refs", ())
                if clean_install_surface is not None
                else ()
            ),
        }
        return evidence
    if capability_type == "replay_surface":
        if capability_id in replay_surface_ids:
            evidence["available_classes"].add("replay")
        evidence["details"] = {
            "replay_surface_id": capability_id,
            "evidence_refs": list(operator_runtime_semantics.get("evidence_refs", ())),
        }
        return evidence
    evidence["details"] = {"capability_id": capability_id}
    return evidence


def _policy_gate_id(row_id: str) -> str:
    return row_id.replace(":", ".")


def _build_release_readiness_gates(
    *,
    contract_validation: ContractCatalogValidationResult,
    notebook_smoke: NotebookSmokeResult,
    clean_install_report: Mapping[str, Any],
) -> tuple[ReadinessGateResult, ...]:
    gates = [
        ReadinessGateResult(
            gate_id="contracts.catalog",
            status=(
                "passed"
                if min(
                    contract_validation.schema_count,
                    contract_validation.module_count,
                    contract_validation.enum_count,
                    contract_validation.contract_document_count,
                )
                > 0
                else "failed"
            ),
            required=True,
            summary=(
                "Contract catalog loads and exposes schemas, modules, enums, "
                "and docs."
            ),
            evidence={
                "schema_count": contract_validation.schema_count,
                "module_count": contract_validation.module_count,
                "enum_count": contract_validation.enum_count,
                "contract_document_count": contract_validation.contract_document_count,
            },
        )
    ]
    notebook_cases = tuple(sorted(notebook_smoke.probabilistic_case_ids))
    gates.append(
        ReadinessGateResult(
            gate_id="notebook.smoke",
            status=(
                "passed"
                if notebook_smoke.summary_path.is_file()
                and notebook_cases == _EXPECTED_NOTEBOOK_CASE_IDS
                and notebook_smoke.catalog_entries >= 1
                else "failed"
            ),
            required=True,
            summary=(
                "Notebook smoke executed the certified package APIs and "
                "published a catalog summary."
            ),
            evidence={
                "summary_path": str(notebook_smoke.summary_path),
                "probabilistic_case_ids": list(notebook_cases),
                "catalog_entries": notebook_smoke.catalog_entries,
                "publication_mode": notebook_smoke.publication_mode,
            },
        )
    )
    clean_install_surfaces = {
        str(surface["surface_id"]): surface
        for surface in clean_install_report.get("surfaces", ())
        if isinstance(surface, Mapping) and isinstance(surface.get("surface_id"), str)
    }
    for gate_id, surface_id, summary in (
        (
            "determinism.same_seed",
            "determinism_same_seed",
            "Installed-wheel determinism smoke preserved same-seed replay surfaces.",
        ),
        (
            "performance.runtime_smoke",
            "performance_runtime_smoke",
            "Installed-wheel performance smoke stayed within the release budget.",
        ),
    ):
        surface = clean_install_surfaces.get(surface_id, {})
        gates.append(
            ReadinessGateResult(
                gate_id=gate_id,
                status=("passed" if surface.get("status") == "passed" else "missing"),
                required=True,
                summary=summary,
                evidence={
                    "surface_id": surface_id,
                    "reason_codes": list(surface.get("reason_codes", ())),
                    "evidence_refs": list(surface.get("evidence_refs", ())),
                },
            )
        )
    return tuple(gates)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verdict_rank(verdict: str) -> int:
    return {"missing": 0, "blocked": 1, "review_required": 2, "ready": 3}.get(
        verdict, -1
    )


def verify_completion_report(
    *,
    project_root: Path | str | None = None,
    report_path: Path | str | None = None,
    policy_path: Path | str | None = None,
) -> CompletionRegressionCheckResult:
    workspace_root = _resolve_workspace_root(project_root)
    asset_root = _resolve_asset_root(project_root)
    resolved_report_path = (
        Path(report_path)
        if report_path is not None
        else workspace_root / "build" / "reports" / "completion-report.json"
    )
    resolved_policy_path = (
        Path(policy_path)
        if policy_path is not None
        else asset_root / _COMPLETION_REGRESSION_POLICY_PATH
    )
    payload = _load_json(resolved_report_path)
    policy = yaml.safe_load(resolved_policy_path.read_text(encoding="utf-8"))
    policy_state = str(policy.get("policy_state", ""))
    active_policy_state = dict(policy.get("policy_states", {}).get(policy_state, {}))
    minimum_completion_values = active_policy_state.get(
        "minimum_completion_values",
        policy.get("minimum_completion_values", {}),
    )
    minimum_policy_verdicts = active_policy_state.get(
        "minimum_policy_verdicts",
        policy.get("minimum_policy_verdicts", {}),
    )

    failures: list[str] = []
    for completion_id, minimum in minimum_completion_values.items():
        actual = float(payload["completion_values"][completion_id])
        if actual < float(minimum):
            failures.append(
                f"{completion_id} regressed below {minimum}: observed {actual:.6f}"
            )

    verdicts_by_id = {
        str(entry["policy_id"]): str(entry["verdict"])
        for entry in payload["policy_verdicts"]
    }
    for policy_id, minimum_verdict in minimum_policy_verdicts.items():
        actual_verdict = verdicts_by_id.get(str(policy_id), "missing")
        if _verdict_rank(actual_verdict) < _verdict_rank(str(minimum_verdict)):
            failures.append(
                f"{policy_id} fell below {minimum_verdict}: observed {actual_verdict}"
            )
    blocked_allowed_until = active_policy_state.get("blocked_allowed_until_utc")
    if isinstance(blocked_allowed_until, str) and blocked_allowed_until:
        deadline = datetime.fromisoformat(
            blocked_allowed_until.replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        if datetime.now(timezone.utc) > deadline:
            for policy_id, actual_verdict in verdicts_by_id.items():
                if actual_verdict == "blocked":
                    failures.append(
                        f"{policy_id} remains blocked past transition window ending "
                        f"{blocked_allowed_until}"
                    )

    clean_install_surfaces = {
        str(surface["surface_id"]): surface
        for surface in payload["clean_install_certification"]["surfaces"]
    }
    for surface_id in policy["required_clean_install_surface_ids"]:
        surface = clean_install_surfaces.get(str(surface_id))
        if surface is None or surface["status"] != "passed":
            failures.append(f"clean-install surface {surface_id} is not passed")

    rows_by_id = {str(row["row_id"]): row for row in payload["capability_rows"]}
    for requirement in policy.get("required_row_evidence_classes", ()):
        row_id = str(requirement["row_id"])
        row = rows_by_id.get(row_id)
        if row is None:
            failures.append(f"required capability row {row_id} is missing")
            continue
        available_classes = set(row.get("available_evidence_classes", ()))
        for required_class in requirement.get("required_evidence_classes", ()):
            if required_class not in available_classes:
                failures.append(
                    f"{row_id} lost required evidence class {required_class}"
                )

    if policy_state == "full_closure" and payload.get("unresolved_blockers"):
        failures.append("completion report still contains unresolved blockers")
    bundles = payload.get("scope_evidence_bundles", ())
    bundle_ids = [
        str(bundle.get("evidence_bundle_id"))
        for bundle in bundles
        if isinstance(bundle, Mapping)
    ]
    if len(bundle_ids) != len(set(bundle_ids)):
        failures.append("scope evidence bundles are not unique")

    _write_json(
        _report_path(workspace_root, _VERIFY_COMPLETION_REPORT_PATH),
        {
            "report_id": "verify_completion_v1",
            "authority_snapshot_id": _load_packaged_authority_snapshot()[
                "authority_snapshot_id"
            ],
            "command_contract_id": _load_packaged_command_contract()[
                "command_contract_id"
            ],
            "generated_at_utc": _now_utc_timestamp(),
            "passed": not failures,
            "failure_messages": failures,
            "report_path": str(resolved_report_path),
            "policy_path": str(resolved_policy_path),
        },
    )

    return CompletionRegressionCheckResult(
        project_root=workspace_root,
        report_path=resolved_report_path,
        policy_path=resolved_policy_path,
        passed=not failures,
        failure_messages=tuple(failures),
    )


def get_release_candidate_workflow(
    *,
    project_root: Path | str | None = None,
) -> ReleaseCandidateWorkflow:
    asset_root = _resolve_asset_root(project_root)
    workspace_root = _resolve_workspace_root(project_root)
    return ReleaseCandidateWorkflow(
        workflow_id=RELEASE_WORKFLOW_ID,
        package_version=PACKAGE_VERSION,
        target_version=RELEASE_TARGET_VERSION,
        project_root=workspace_root,
        example_manifest=resolve_example_path(
            "current_release_run.yaml",
            project_root=asset_root,
        ),
        benchmark_suite=asset_root / "benchmarks" / "suites" / "current-release.yaml",
        notebook_path=resolve_notebook_path(project_root=asset_root),
        required_test_targets=RELEASE_CERTIFICATION_TEST_TARGETS,
        certification_commands=tuple(
            str(command["command"])
            for command in _load_packaged_command_contract().get("commands", ())
        ),
    )


def _benchmark_smoke_manifest_paths(project_root: Path) -> tuple[Path, ...]:
    return (
        project_root
        / "benchmarks"
        / "tasks"
        / "rediscovery"
        / "planted-analytic-demo.yaml",
        project_root
        / "benchmarks"
        / "tasks"
        / "predictive_generalization"
        / "seasonal-trend-demo.yaml",
        project_root
        / "benchmarks"
        / "tasks"
        / "adversarial_honesty"
        / "leakage-trap-demo.yaml",
    )


def _notebook_setup_cell_index(cells: list[dict[str, object]]) -> int:
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            return index
    raise ValueError("notebook must contain at least one code cell")


def _release_notebook_setup_cell(*, project_root: Path, output_root: Path) -> str:
    return (
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n\n"
        f"PROJECT_ROOT = Path({str(project_root)!r})\n"
        "from euclid import run_operator\n\n"
        "MANIFEST_DIR = PROJECT_ROOT / 'fixtures/runtime/phase06'\n"
        "CURRENT_RELEASE_MANIFEST = (\n"
        "    PROJECT_ROOT / 'examples/current_release_run.yaml'\n"
        ")\n"
        "MANIFESTS = {\n"
        "    'distribution': MANIFEST_DIR / 'probabilistic-distribution-demo.yaml',\n"
        "    'interval': MANIFEST_DIR / 'probabilistic-interval-demo.yaml',\n"
        "    'quantile': MANIFEST_DIR / 'probabilistic-quantile-demo.yaml',\n"
        "    'event_probability':\n"
        "        MANIFEST_DIR / 'probabilistic-event-probability-demo.yaml',\n"
        "}\n"
        f"OUTPUT_ROOT = Path({str(output_root)!r})\n"
        "{\n"
        "    'manifest_dir': str(MANIFEST_DIR),\n"
        "    'current_release_manifest': str(CURRENT_RELEASE_MANIFEST),\n"
        "    'output_root': str(OUTPUT_ROOT),\n"
        "}\n"
    )


__all__ = [
    "BenchmarkSmokeCaseResult",
    "BenchmarkSmokeResult",
    "CleanInstallCertificationResult",
    "CleanInstallCertificationSurfaceResult",
    "CompletionRegressionCheckResult",
    "ContractCatalogValidationResult",
    "NotebookSmokeResult",
    "ReleaseDeterminismSmokeResult",
    "ReleasePerformanceSmokeResult",
    "ReleaseCandidateWorkflow",
    "ReleaseStatus",
    "execute_release_notebook_smoke",
    "get_release_candidate_workflow",
    "get_release_status",
    "load_packaged_release_policies",
    "run_clean_install_certification",
    "run_release_benchmark_smoke",
    "run_release_determinism_smoke",
    "run_release_performance_smoke",
    "validate_release_contracts",
    "verify_completion_report",
]
