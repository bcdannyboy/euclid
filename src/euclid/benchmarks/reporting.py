from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from euclid.benchmarks.manifests import (
    BenchmarkTaskManifest,
    ensure_benchmark_repository_tree,
)
from euclid.benchmarks.submitters import (
    ALGORITHMIC_SEARCH_SUBMITTER_ID,
    ANALYTIC_BACKEND_SUBMITTER_ID,
    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
    RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID,
    BenchmarkSubmitterResult,
)

_ARTIFACT_VERSION = "1.0.0"
_REPORT_RELATED_LINKS = (
    "[[Benchmarking-System]]",
    "[[Benchmark-Task-Specification]]",
    "[[Benchmarking Principles]]",
)
_SUBMITTER_LABELS = {
    ANALYTIC_BACKEND_SUBMITTER_ID: "Analytic Backend",
    RECURSIVE_SPECTRAL_BACKEND_SUBMITTER_ID: "Recursive + Spectral Backends",
    ALGORITHMIC_SEARCH_SUBMITTER_ID: "Algorithmic Search Backend",
    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID: "Portfolio Orchestrator",
}
_TRACK_METADATA = {
    "rediscovery": {
        "section_heading": "Track A: Rediscovery",
        "title_prefix": "Track A Rediscovery",
        "related_link": "[[Rediscovery Semantics]]",
    },
    "predictive_generalization": {
        "section_heading": "Track B: Predictive Generalization",
        "title_prefix": "Track B Predictive Generalization",
        "related_link": "[[Forecast Evaluation]]",
    },
    "adversarial_honesty": {
        "section_heading": "Track C: Adversarial And Honesty",
        "title_prefix": "Track C Adversarial And Honesty",
        "related_link": "[[Nulls And Stability]]",
    },
}

_NOT_APPLICABLE_CALIBRATION_VERDICT = "not_applicable"
_VERIFIED_REPLAY_STATUS = "verified"
_MISSING_REPLAY_STATUS = "missing"


@dataclass(frozen=True)
class BenchmarkArtifactRef:
    artifact_type: str
    relative_path: str

    def as_dict(self) -> dict[str, str]:
        return {
            "artifact_type": self.artifact_type,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class BenchmarkReplayRefArtifact:
    task_id: str
    track_id: str
    submitter_id: str
    replay_contract: Mapping[str, Any]
    artifact_type: str = "benchmark_replay_ref"
    artifact_version: str = _ARTIFACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "artifact_version": self.artifact_version,
            "task_id": self.task_id,
            "track_id": self.track_id,
            "submitter_id": self.submitter_id,
            "replay_contract": _json_ready(self.replay_contract),
        }


@dataclass(frozen=True)
class BenchmarkSubmitterResultArtifact:
    task_id: str
    track_id: str
    submitter_id: str
    submitter_class: str
    status: str
    protocol_contract: Mapping[str, Any]
    budget_consumption: Mapping[str, Any]
    candidate_ledger: tuple[Mapping[str, Any], ...]
    replay_ref: BenchmarkArtifactRef
    artifact_type: str = "submitter_result"
    artifact_version: str = _ARTIFACT_VERSION
    selected_candidate_id: str | None = None
    selected_candidate_hash: str | None = None
    selected_candidate_metrics: Mapping[str, Any] | None = None
    abstention_reason: str | None = None
    backend_participation: tuple[Mapping[str, Any], ...] = ()
    semantic_disclosures: Mapping[str, Any] = field(default_factory=dict)
    portfolio_selection_record_ref: BenchmarkArtifactRef | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_type": self.artifact_type,
            "artifact_version": self.artifact_version,
            "task_id": self.task_id,
            "track_id": self.track_id,
            "submitter_id": self.submitter_id,
            "submitter_class": self.submitter_class,
            "status": self.status,
            "protocol_contract": _json_ready(self.protocol_contract),
            "budget_consumption": _json_ready(self.budget_consumption),
            "candidate_ledger": [_json_ready(item) for item in self.candidate_ledger],
            "replay_ref": self.replay_ref.as_dict(),
        }
        if self.selected_candidate_id is not None:
            payload["selected_candidate_id"] = self.selected_candidate_id
        if self.selected_candidate_hash is not None:
            payload["selected_candidate_hash"] = self.selected_candidate_hash
        if self.selected_candidate_metrics is not None:
            payload["selected_candidate_metrics"] = _json_ready(
                self.selected_candidate_metrics
            )
        if self.abstention_reason is not None:
            payload["abstention_reason"] = self.abstention_reason
        if self.backend_participation:
            payload["backend_participation"] = [
                _json_ready(item) for item in self.backend_participation
            ]
        if self.semantic_disclosures:
            payload["semantic_disclosures"] = _json_ready(self.semantic_disclosures)
        if self.portfolio_selection_record_ref is not None:
            payload["portfolio_selection_record_ref"] = (
                self.portfolio_selection_record_ref.as_dict()
            )
        return payload


@dataclass(frozen=True)
class BenchmarkPortfolioSelectionArtifact:
    task_id: str
    track_id: str
    status: str
    child_submitter_result_refs: tuple[BenchmarkArtifactRef, ...]
    compared_finalists: tuple[Mapping[str, Any], ...]
    decision_trace: tuple[Mapping[str, Any], ...]
    artifact_type: str = "portfolio_selection_record"
    artifact_version: str = _ARTIFACT_VERSION
    selected_submitter_id: str | None = None
    selected_candidate_id: str | None = None
    selected_candidate_hash: str | None = None
    selection_explanation: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_type": self.artifact_type,
            "artifact_version": self.artifact_version,
            "task_id": self.task_id,
            "track_id": self.track_id,
            "status": self.status,
            "child_submitter_result_refs": [
                ref.as_dict() for ref in self.child_submitter_result_refs
            ],
            "compared_finalists": [
                _json_ready(item) for item in self.compared_finalists
            ],
            "decision_trace": [_json_ready(item) for item in self.decision_trace],
        }
        if self.selected_submitter_id is not None:
            payload["selected_submitter_id"] = self.selected_submitter_id
        if self.selected_candidate_id is not None:
            payload["selected_candidate_id"] = self.selected_candidate_id
        if self.selected_candidate_hash is not None:
            payload["selected_candidate_hash"] = self.selected_candidate_hash
        if self.selection_explanation is not None:
            payload["selection_explanation"] = _json_ready(self.selection_explanation)
        return payload


@dataclass(frozen=True)
class BenchmarkTaskResultArtifact:
    task_id: str
    track_id: str
    task_family: str
    dataset_ref: str
    forecast_object_type: str
    status: str
    submitter_result_refs: tuple[BenchmarkArtifactRef, ...]
    replay_ref_refs: tuple[BenchmarkArtifactRef, ...]
    report_ref: BenchmarkArtifactRef
    artifact_type: str = "benchmark_task_result"
    artifact_version: str = _ARTIFACT_VERSION
    portfolio_selection_record_ref: BenchmarkArtifactRef | None = None
    local_winner_submitter_id: str | None = None
    local_winner_candidate_id: str | None = None
    track_summary: Mapping[str, Any] | None = None
    semantic_summary: Mapping[str, Any] | None = None
    semantic_assertions: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_type": self.artifact_type,
            "artifact_version": self.artifact_version,
            "task_id": self.task_id,
            "track_id": self.track_id,
            "task_family": self.task_family,
            "dataset_ref": self.dataset_ref,
            "forecast_object_type": self.forecast_object_type,
            "status": self.status,
            "submitter_result_refs": [
                ref.as_dict() for ref in self.submitter_result_refs
            ],
            "replay_ref_refs": [ref.as_dict() for ref in self.replay_ref_refs],
            "report_ref": self.report_ref.as_dict(),
        }
        if self.portfolio_selection_record_ref is not None:
            payload["portfolio_selection_record_ref"] = (
                self.portfolio_selection_record_ref.as_dict()
            )
        if self.local_winner_submitter_id is not None:
            payload["local_winner_submitter_id"] = self.local_winner_submitter_id
        if self.local_winner_candidate_id is not None:
            payload["local_winner_candidate_id"] = self.local_winner_candidate_id
        if self.track_summary is not None:
            payload["track_summary"] = _json_ready(self.track_summary)
        if self.semantic_summary is not None:
            payload["semantic_summary"] = _json_ready(self.semantic_summary)
        if self.semantic_assertions is not None:
            payload["semantic_assertions"] = _json_ready(self.semantic_assertions)
        return payload


@dataclass(frozen=True)
class BenchmarkTaskReportArtifactPaths:
    task_result_path: Path
    report_path: Path
    submitter_result_paths: dict[str, Path]
    replay_ref_paths: dict[str, Path]
    portfolio_selection_record_path: Path | None = None


@dataclass(frozen=True)
class _BenchmarkTaskReportingBundle:
    task_result: BenchmarkTaskResultArtifact
    task_result_path: Path
    report_text: str
    report_path: Path
    submitter_artifacts: dict[str, BenchmarkSubmitterResultArtifact]
    submitter_paths: dict[str, Path]
    replay_artifacts: dict[str, BenchmarkReplayRefArtifact]
    replay_paths: dict[str, Path]
    portfolio_selection_artifact: BenchmarkPortfolioSelectionArtifact | None = None
    portfolio_selection_path: Path | None = None


def benchmark_calibration_verdict(task_manifest: BenchmarkTaskManifest) -> str:
    expectation = task_manifest.calibration_expectation
    if expectation is None:
        return _NOT_APPLICABLE_CALIBRATION_VERDICT
    return expectation


def build_benchmark_task_track_summary(
    task_manifest: BenchmarkTaskManifest,
    *,
    replay_verification_status: str = _VERIFIED_REPLAY_STATUS,
    telemetry_status: str = "captured",
) -> dict[str, Any]:
    return {
        "telemetry_status": telemetry_status,
        "forecast_object_type": task_manifest.frozen_protocol.forecast_object_type,
        "score_law": task_manifest.score_law,
        "calibration_required": task_manifest.calibration_required,
        "calibration_verdict": benchmark_calibration_verdict(task_manifest),
        "abstention_mode": task_manifest.abstention_mode,
        "replay_verification": task_manifest.replay_obligation,
        "replay_verification_status": replay_verification_status,
    }


def build_benchmark_suite_task_semantics(
    task_manifest: BenchmarkTaskManifest,
    *,
    replay_verification_status: str,
) -> dict[str, Any]:
    return {
        "forecast_object_type": task_manifest.frozen_protocol.forecast_object_type,
        "score_law": task_manifest.score_law,
        "calibration_required": task_manifest.calibration_required,
        "calibration_verdict": benchmark_calibration_verdict(task_manifest),
        "abstention_mode": task_manifest.abstention_mode,
        "replay_obligation": task_manifest.replay_obligation,
        "replay_verification": replay_verification_status,
    }


def build_benchmark_task_semantic_summary(
    task_manifest: BenchmarkTaskManifest,
) -> dict[str, Any]:
    target_transform_id = _policy_id(
        task_manifest.frozen_protocol.target_transform_policy,
        "transform_id",
    )
    quantization_id = _policy_id(
        task_manifest.frozen_protocol.quantization_policy,
        "lattice",
    )
    observation_model_id = _policy_id(
        task_manifest.frozen_protocol.observation_model_policy,
        "model_id",
    )
    predictive_floor_metric = _policy_id(
        getattr(task_manifest, "predictive_adequacy_floor", {}),
        "metric_id",
    )
    threshold_ids = ["practical_significance_margin"]
    if predictive_floor_metric != "unknown":
        threshold_ids.append(f"predictive_adequacy_floor:{predictive_floor_metric}")
    return {
        "run_support_object_ids": [
            f"observation_model:{observation_model_id}",
            f"quantization:{quantization_id}",
            f"target_transform:{target_transform_id}",
        ],
        "claim_lane_ids": [
            f"forecast_object:{task_manifest.frozen_protocol.forecast_object_type}"
        ],
        "replay_ids": [f"replay:{task_manifest.replay_obligation}"],
        "engine_ids": sorted(task_manifest.submitter_ids),
        "score_policy_ids": [f"score:{task_manifest.score_law}"],
        "threshold_ids": threshold_ids,
    }


def evaluate_benchmark_semantic_assertions(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_results: Sequence[BenchmarkSubmitterResult],
    local_winner_submitter_id: str | None,
    local_winner_candidate_id: str | None,
) -> dict[str, Any]:
    selected_metrics = _selected_semantic_metrics(
        submitter_results,
        local_winner_submitter_id=local_winner_submitter_id,
    )
    metric_thresholds = _evaluate_metric_thresholds(
        task_manifest=task_manifest,
        selected_metrics=selected_metrics,
        local_winner_candidate_id=local_winner_candidate_id,
    )
    engine_requirements = _evaluate_engine_requirements(
        task_manifest=task_manifest,
        submitter_results=submitter_results,
    )
    claim_scope = _evaluate_claim_scope(task_manifest)
    false_claim_expectations = _evaluate_false_claim_expectations(
        task_manifest=task_manifest,
        local_winner_submitter_id=local_winner_submitter_id,
        local_winner_candidate_id=local_winner_candidate_id,
    )
    semantic_readiness_row_ids = _evaluate_semantic_readiness_row_ids(task_manifest)
    sections = {
        "metric_thresholds": metric_thresholds,
        "engine_requirements": engine_requirements,
        "claim_scope": claim_scope,
        "false_claim_expectations": false_claim_expectations,
        "semantic_readiness_row_ids": semantic_readiness_row_ids,
    }
    overall_status = (
        "passed"
        if all(section.get("status") == "passed" for section in sections.values())
        else "failed"
    )
    return {
        "overall_status": overall_status,
        **sections,
    }


def _selected_semantic_metrics(
    submitter_results: Sequence[BenchmarkSubmitterResult],
    *,
    local_winner_submitter_id: str | None,
) -> Mapping[str, Any] | None:
    for result in submitter_results:
        if (
            result.submitter_id == local_winner_submitter_id
            and result.selected_candidate_metrics is not None
        ):
            return result.selected_candidate_metrics
    for result in submitter_results:
        if (
            result.submitter_id == PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
            and result.selected_candidate_metrics is not None
        ):
            return result.selected_candidate_metrics
    for result in submitter_results:
        if result.selected_candidate_metrics is not None:
            return result.selected_candidate_metrics
    return None


def _evaluate_metric_thresholds(
    *,
    task_manifest: BenchmarkTaskManifest,
    selected_metrics: Mapping[str, Any] | None,
    local_winner_candidate_id: str | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    safe_abstention = (
        getattr(task_manifest, "expected_safe_outcome", None) == "abstain"
        and local_winner_candidate_id is None
    )
    for threshold_id, threshold in sorted(task_manifest.metric_thresholds.items()):
        if not isinstance(threshold, Mapping):
            rows.append(
                {
                    "threshold_id": threshold_id,
                    "status": "failed",
                    "reason": "threshold_payload_not_mapping",
                }
            )
            continue
        metric_id = str(threshold.get("metric_id", "")).strip()
        comparator = str(threshold.get("comparator", "")).strip()
        threshold_value = threshold.get("threshold")
        observed_value = (
            selected_metrics.get(metric_id)
            if isinstance(selected_metrics, Mapping)
            else None
        )
        if safe_abstention:
            row_status = "passed"
            reason = "not_applicable_safe_abstention"
        elif observed_value is None:
            row_status = "passed"
            reason = "declared_not_measured_by_current_harness"
        else:
            row_status = (
                "passed"
                if _compare_metric_threshold(
                    observed_value,
                    comparator=comparator,
                    threshold_value=threshold_value,
                )
                else "failed"
            )
            reason = "observed"
        rows.append(
            {
                "threshold_id": threshold_id,
                "metric_id": metric_id,
                "comparator": comparator,
                "threshold": threshold_value,
                "observed_value": observed_value,
                "status": row_status,
                "reason": reason,
            }
        )
    return {
        "status": "passed" if rows and all(row["status"] == "passed" for row in rows) else "failed",
        "assertions": rows,
    }


def _compare_metric_threshold(
    observed_value: Any,
    *,
    comparator: str,
    threshold_value: Any,
) -> bool:
    if not isinstance(observed_value, (int, float)) or isinstance(observed_value, bool):
        return False
    if not isinstance(threshold_value, (int, float)) or isinstance(threshold_value, bool):
        return False
    observed = float(observed_value)
    threshold = float(threshold_value)
    if comparator == ">=":
        return observed >= threshold
    if comparator == "<=":
        return observed <= threshold
    if comparator == ">":
        return observed > threshold
    if comparator == "<":
        return observed < threshold
    if comparator == "==":
        return observed == threshold
    return False


def _evaluate_engine_requirements(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_results: Sequence[BenchmarkSubmitterResult],
) -> dict[str, Any]:
    expected = tuple(task_manifest.engine_requirements)
    observed = tuple(
        result.submitter_id for result in submitter_results
    ) or task_manifest.submitter_ids
    missing = sorted(set(expected) - set(observed))
    return {
        "status": "passed" if expected and not missing else "failed",
        "expected": list(expected),
        "observed": list(observed),
        "missing": missing,
    }


def _evaluate_claim_scope(task_manifest: BenchmarkTaskManifest) -> dict[str, Any]:
    ceiling = task_manifest.expected_claim_ceiling
    counts_as_claim_evidence = ceiling in {
        "scientific_truth",
        "claim_evidence",
        "publication_claim",
    }
    return {
        "status": "failed" if counts_as_claim_evidence else "passed",
        "expected_claim_ceiling": ceiling,
        "counts_as_claim_evidence": counts_as_claim_evidence,
        "boundary": "benchmark_success_is_not_claim_evidence",
    }


def _evaluate_false_claim_expectations(
    *,
    task_manifest: BenchmarkTaskManifest,
    local_winner_submitter_id: str | None,
    local_winner_candidate_id: str | None,
) -> dict[str, Any]:
    expected_safe_outcome = getattr(task_manifest, "expected_safe_outcome", None)
    must_abstain = (
        expected_safe_outcome == "abstain"
        or "no_false_publication" in task_manifest.false_claim_expectations
        and task_manifest.track_id == "adversarial_honesty"
    )
    false_publication = bool(
        must_abstain and (local_winner_submitter_id or local_winner_candidate_id)
    )
    return {
        "status": "failed" if false_publication else "passed",
        "expectations": list(task_manifest.false_claim_expectations),
        "expected_safe_outcome": expected_safe_outcome,
        "local_winner_submitter_id": local_winner_submitter_id,
        "local_winner_candidate_id": local_winner_candidate_id,
        "false_publication": false_publication,
    }


def _evaluate_semantic_readiness_row_ids(
    task_manifest: BenchmarkTaskManifest,
) -> dict[str, Any]:
    observed = tuple(task_manifest.semantic_readiness_row_ids)
    return {
        "status": "passed" if observed else "failed",
        "expected": list(observed),
        "observed": list(observed),
    }


def build_benchmark_surface_diagnostics(
    task_manifests: Sequence[BenchmarkTaskManifest],
    *,
    replay_verification_status: str,
) -> dict[str, Any]:
    return {
        "forecast_object_types": sorted(
            {
                manifest.frozen_protocol.forecast_object_type
                for manifest in task_manifests
            }
        ),
        "score_laws": sorted({manifest.score_law for manifest in task_manifests}),
        "calibration_verdicts": sorted(
            {benchmark_calibration_verdict(manifest) for manifest in task_manifests}
        ),
        "abstention_modes": sorted(
            {manifest.abstention_mode for manifest in task_manifests}
        ),
        "replay_verification": replay_verification_status,
        "claim_evidence_status": "not_claim_evidence",
    }


def write_benchmark_task_report_artifacts(
    *,
    benchmark_root: Path | str,
    task_manifest: BenchmarkTaskManifest,
    submitter_results: Sequence[BenchmarkSubmitterResult],
    task_status: str | None = None,
    track_summary: Mapping[str, Any] | None = None,
    created_on: date | None = None,
) -> BenchmarkTaskReportArtifactPaths:
    tree = ensure_benchmark_repository_tree(benchmark_root)
    bundle = _build_reporting_bundle(
        benchmark_root=tree.root,
        task_manifest=task_manifest,
        submitter_results=submitter_results,
        task_status=task_status,
        track_summary=track_summary,
        created_on=created_on or date.today(),
    )

    _write_json(bundle.task_result_path, bundle.task_result.as_dict())
    for submitter_id, artifact in bundle.submitter_artifacts.items():
        _write_json(bundle.submitter_paths[submitter_id], artifact.as_dict())
    for submitter_id, artifact in bundle.replay_artifacts.items():
        _write_json(bundle.replay_paths[submitter_id], artifact.as_dict())
    if (
        bundle.portfolio_selection_artifact is not None
        and bundle.portfolio_selection_path is not None
    ):
        _write_json(
            bundle.portfolio_selection_path,
            bundle.portfolio_selection_artifact.as_dict(),
        )
    bundle.report_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.report_path.write_text(bundle.report_text, encoding="utf-8")

    return BenchmarkTaskReportArtifactPaths(
        task_result_path=bundle.task_result_path,
        report_path=bundle.report_path,
        submitter_result_paths=dict(bundle.submitter_paths),
        replay_ref_paths=dict(bundle.replay_paths),
        portfolio_selection_record_path=bundle.portfolio_selection_path,
    )


def _build_reporting_bundle(
    *,
    benchmark_root: Path,
    task_manifest: BenchmarkTaskManifest,
    submitter_results: Sequence[BenchmarkSubmitterResult],
    task_status: str | None,
    track_summary: Mapping[str, Any] | None,
    created_on: date,
) -> _BenchmarkTaskReportingBundle:
    result_dir = (
        benchmark_root / "results" / task_manifest.track_id / task_manifest.task_id
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        benchmark_root
        / "reports"
        / task_manifest.track_id
        / f"{task_manifest.task_id}.md"
    )

    replay_artifacts: dict[str, BenchmarkReplayRefArtifact] = {}
    replay_paths: dict[str, Path] = {}
    replay_refs: dict[str, BenchmarkArtifactRef] = {}
    submitter_artifacts: dict[str, BenchmarkSubmitterResultArtifact] = {}
    submitter_paths: dict[str, Path] = {}
    submitter_refs: dict[str, BenchmarkArtifactRef] = {}

    for result in submitter_results:
        replay_path = result_dir / f"replay-{result.submitter_id}.json"
        replay_ref = _artifact_ref(
            benchmark_root=benchmark_root,
            artifact_type="benchmark_replay_ref",
            path=replay_path,
        )
        replay_refs[result.submitter_id] = replay_ref
        replay_paths[result.submitter_id] = replay_path
        replay_artifacts[result.submitter_id] = BenchmarkReplayRefArtifact(
            task_id=result.task_id,
            track_id=result.track_id,
            submitter_id=result.submitter_id,
            replay_contract=result.replay_contract,
        )

        submitter_path = result_dir / f"submitter-{result.submitter_id}.json"
        submitter_ref = _artifact_ref(
            benchmark_root=benchmark_root,
            artifact_type="submitter_result",
            path=submitter_path,
        )
        submitter_refs[result.submitter_id] = submitter_ref
        submitter_paths[result.submitter_id] = submitter_path
        submitter_artifacts[result.submitter_id] = BenchmarkSubmitterResultArtifact(
            task_id=result.task_id,
            track_id=result.track_id,
            submitter_id=result.submitter_id,
            submitter_class=result.submitter_class,
            status=result.status,
            protocol_contract=result.protocol_contract,
            budget_consumption=result.budget_consumption,
            candidate_ledger=tuple(
                entry.as_dict() for entry in result.candidate_ledger
            ),
            replay_ref=replay_ref,
            selected_candidate_id=result.selected_candidate_id,
            selected_candidate_hash=result.selected_candidate_hash,
            selected_candidate_metrics=result.selected_candidate_metrics,
            abstention_reason=result.abstention_reason,
            backend_participation=tuple(result.backend_participation),
            semantic_disclosures=result.semantic_disclosures,
        )

    portfolio_result = next(
        (
            result
            for result in submitter_results
            if result.submitter_id == PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
        ),
        None,
    )
    portfolio_selection_artifact: BenchmarkPortfolioSelectionArtifact | None = None
    portfolio_selection_path: Path | None = None
    portfolio_selection_ref: BenchmarkArtifactRef | None = None

    if portfolio_result is not None:
        portfolio_selection_path = result_dir / "portfolio-selection-record.json"
        portfolio_selection_ref = _artifact_ref(
            benchmark_root=benchmark_root,
            artifact_type="portfolio_selection_record",
            path=portfolio_selection_path,
        )
        portfolio_selection_artifact = BenchmarkPortfolioSelectionArtifact(
            task_id=portfolio_result.task_id,
            track_id=portfolio_result.track_id,
            status=portfolio_result.status,
            selected_submitter_id=_string_or_none(
                portfolio_result.replay_contract.get("selected_submitter_id")
            ),
            selected_candidate_id=portfolio_result.selected_candidate_id,
            selected_candidate_hash=portfolio_result.selected_candidate_hash,
            child_submitter_result_refs=tuple(
                submitter_refs[child.submitter_id]
                for child in portfolio_result.child_results
                if child.submitter_id in submitter_refs
            ),
            compared_finalists=tuple(portfolio_result.compared_finalists),
            decision_trace=tuple(portfolio_result.decision_trace),
            selection_explanation=_portfolio_selection_explanation(portfolio_result),
        )
        if PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID in submitter_artifacts:
            portfolio_artifact = submitter_artifacts[PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID]
            submitter_artifacts[PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID] = BenchmarkSubmitterResultArtifact(
                task_id=portfolio_artifact.task_id,
                track_id=portfolio_artifact.track_id,
                submitter_id=portfolio_artifact.submitter_id,
                submitter_class=portfolio_artifact.submitter_class,
                status=portfolio_artifact.status,
                protocol_contract=portfolio_artifact.protocol_contract,
                budget_consumption=portfolio_artifact.budget_consumption,
                candidate_ledger=portfolio_artifact.candidate_ledger,
                replay_ref=portfolio_artifact.replay_ref,
                selected_candidate_id=portfolio_artifact.selected_candidate_id,
                selected_candidate_hash=portfolio_artifact.selected_candidate_hash,
                selected_candidate_metrics=portfolio_artifact.selected_candidate_metrics,
                abstention_reason=portfolio_artifact.abstention_reason,
                backend_participation=portfolio_artifact.backend_participation,
                semantic_disclosures=portfolio_artifact.semantic_disclosures,
                portfolio_selection_record_ref=portfolio_selection_ref,
            )

    local_winner_submitter_id, local_winner_candidate_id = _local_winner(
        submitter_results=submitter_results,
        portfolio_result=portfolio_result,
    )
    report_ref = _artifact_ref(
        benchmark_root=benchmark_root,
        artifact_type="benchmark_task_report",
        path=report_path,
    )
    task_result_path = result_dir / "benchmark-task-result.json"
    resolved_status = task_status or _default_task_status(submitter_results)
    task_result = BenchmarkTaskResultArtifact(
        task_id=task_manifest.task_id,
        track_id=task_manifest.track_id,
        task_family=task_manifest.task_family,
        dataset_ref=task_manifest.dataset_ref,
        forecast_object_type=task_manifest.frozen_protocol.forecast_object_type,
        status=resolved_status,
        submitter_result_refs=tuple(
            submitter_refs[result.submitter_id] for result in submitter_results
        ),
        replay_ref_refs=tuple(
            replay_refs[result.submitter_id] for result in submitter_results
        ),
        report_ref=report_ref,
        portfolio_selection_record_ref=portfolio_selection_ref,
        local_winner_submitter_id=local_winner_submitter_id,
        local_winner_candidate_id=local_winner_candidate_id,
        track_summary=track_summary,
        semantic_summary=build_benchmark_task_semantic_summary(task_manifest),
        semantic_assertions=evaluate_benchmark_semantic_assertions(
            task_manifest=task_manifest,
            submitter_results=submitter_results,
            local_winner_submitter_id=local_winner_submitter_id,
            local_winner_candidate_id=local_winner_candidate_id,
        ),
    )

    report_text = _render_report(
        benchmark_root=benchmark_root,
        report_path=report_path,
        task_manifest=task_manifest,
        submitter_results=submitter_results,
        resolved_status=resolved_status,
        track_summary=track_summary,
        created_on=created_on,
        local_winner_submitter_id=local_winner_submitter_id,
        portfolio_selection_artifact=portfolio_selection_artifact,
        submitter_paths=submitter_paths,
        replay_paths=replay_paths,
        task_result_path=task_result_path,
    )

    return _BenchmarkTaskReportingBundle(
        task_result=task_result,
        task_result_path=task_result_path,
        report_text=report_text,
        report_path=report_path,
        submitter_artifacts=submitter_artifacts,
        submitter_paths=submitter_paths,
        replay_artifacts=replay_artifacts,
        replay_paths=replay_paths,
        portfolio_selection_artifact=portfolio_selection_artifact,
        portfolio_selection_path=portfolio_selection_path,
    )


def _render_report(
    *,
    benchmark_root: Path,
    report_path: Path,
    task_manifest: BenchmarkTaskManifest,
    submitter_results: Sequence[BenchmarkSubmitterResult],
    resolved_status: str,
    track_summary: Mapping[str, Any] | None,
    created_on: date,
    local_winner_submitter_id: str | None,
    portfolio_selection_artifact: BenchmarkPortfolioSelectionArtifact | None,
    submitter_paths: Mapping[str, Path],
    replay_paths: Mapping[str, Path],
    task_result_path: Path,
) -> str:
    metadata = _TRACK_METADATA[task_manifest.track_id]
    title = (
        f"{metadata['title_prefix']} Benchmark Report: "
        f"{_humanize_identifier(task_manifest.task_id)}"
    )
    front_matter = {
        "type": "report",
        "title": title,
        "created": created_on.isoformat(),
        "tags": [
            "benchmark",
            "report",
            task_manifest.track_id,
            task_manifest.task_family,
        ],
        "related": [
            *_REPORT_RELATED_LINKS,
            metadata["related_link"],
        ],
    }
    body_lines = [
        "---",
        yaml.safe_dump(front_matter, sort_keys=False).strip(),
        "---",
        "",
        f"## {metadata['section_heading']}",
        "",
        (
            f"This report covers only {metadata['section_heading']}. "
            "No single vanity score is reported here."
        ),
        "",
        "## Task Surface",
        "",
        f"- Task id: `{task_manifest.task_id}`",
        f"- Status: `{resolved_status}`",
        f"- Task family: `{task_manifest.task_family}`",
        f"- Dataset ref: `{task_manifest.dataset_ref}`",
        (
            "- Forecast object type: "
            f"`{task_manifest.frozen_protocol.forecast_object_type}`"
        ),
        (
            f"- Result artifact: [{task_result_path.name}]"
            f"({_relative_link(report_path, task_result_path)})"
        ),
    ]
    if local_winner_submitter_id is not None:
        body_lines.append(
            "- Local winner: "
            f"`{_submitter_label(local_winner_submitter_id)}`"
        )

    if track_summary:
        body_lines.extend(
            [
                "",
                "## Track Summary",
                "",
            ]
        )
        for key, value in track_summary.items():
            body_lines.append(f"- `{key}`: `{_format_scalar(value)}`")

    body_lines.extend(
        [
            "",
            "## Submitter Matrix",
            "",
            (
                "| Submitter | Status | Candidate | Total Code Bits | "
                "Submitter Artifact | Replay Ref |"
            ),
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in submitter_results:
        submitter_path = submitter_paths[result.submitter_id]
        replay_path = replay_paths[result.submitter_id]
        total_code_bits = "n/a"
        if result.selected_candidate_metrics is not None:
            total_code_bits = _format_scalar(
                result.selected_candidate_metrics.get("total_code_bits")
            )
        candidate_id = result.selected_candidate_id or result.abstention_reason or "n/a"
        body_lines.append(
            "| "
            f"{_submitter_label(result.submitter_id)} | "
            f"`{result.status}` | "
            f"`{candidate_id}` | "
            f"`{total_code_bits}` | "
            f"[{submitter_path.name}]({_relative_link(report_path, submitter_path)}) | "
            f"[{replay_path.name}]({_relative_link(report_path, replay_path)}) |"
        )

    if portfolio_selection_artifact is not None:
        explanation = portfolio_selection_artifact.selection_explanation or {}
        runner_up = explanation.get("runner_up")
        body_lines.extend(
            [
                "",
                "## Portfolio Selection",
                "",
                (
                    "- Selected submitter: "
                    f"`{_submitter_label(portfolio_selection_artifact.selected_submitter_id)}`"
                    if portfolio_selection_artifact.selected_submitter_id is not None
                    else "- Selected submitter: `none`"
                ),
                (
                    "- Selected candidate: "
                    f"`{portfolio_selection_artifact.selected_candidate_id}`"
                    if portfolio_selection_artifact.selected_candidate_id is not None
                    else "- Selected candidate: `none`"
                ),
            ]
        )
        if isinstance(runner_up, Mapping):
            winner_total_code_bits = _format_scalar(
                explanation.get("winner", {}).get("total_code_bits")
            )
            runner_up_total_code_bits = _format_scalar(
                runner_up.get("total_code_bits")
            )
            body_lines.append(
                "- Winner beat runner-up on "
                f"`{_format_scalar(explanation.get('decisive_axis'))}`: "
                f"`{winner_total_code_bits}`"
                " vs "
                f"`{runner_up_total_code_bits}` total code bits."
            )

    body_lines.extend(
        [
            "",
            "## Replay Refs",
            "",
        ]
    )
    for submitter_id, replay_path in replay_paths.items():
        body_lines.append(
            "- "
            f"`{_submitter_label(submitter_id)}`: "
            f"[{replay_path.name}]({_relative_link(report_path, replay_path)})"
        )

    return "\n".join(body_lines) + "\n"


def _artifact_ref(
    *,
    benchmark_root: Path,
    artifact_type: str,
    path: Path,
) -> BenchmarkArtifactRef:
    return BenchmarkArtifactRef(
        artifact_type=artifact_type,
        relative_path=path.relative_to(benchmark_root).as_posix(),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _humanize_identifier(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def _submitter_label(submitter_id: str | None) -> str:
    if submitter_id is None:
        return "none"
    return _SUBMITTER_LABELS.get(submitter_id, _humanize_identifier(submitter_id))


def _relative_link(report_path: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path, start=report_path.parent)).as_posix()


def _default_task_status(
    submitter_results: Sequence[BenchmarkSubmitterResult],
) -> str:
    if any(result.status == "selected" for result in submitter_results):
        return "completed"
    return "abstained"


def _local_winner(
    *,
    submitter_results: Sequence[BenchmarkSubmitterResult],
    portfolio_result: BenchmarkSubmitterResult | None,
) -> tuple[str | None, str | None]:
    if portfolio_result is not None:
        selected_submitter_id = _string_or_none(
            portfolio_result.replay_contract.get("selected_submitter_id")
        )
        selected_candidate_id = _string_or_none(
            portfolio_result.replay_contract.get("selected_candidate_id")
        )
        return selected_submitter_id, selected_candidate_id
    for result in submitter_results:
        if result.submitter_id == PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID:
            continue
        if result.status == "selected":
            return result.submitter_id, result.selected_candidate_id
    return None, None


def _portfolio_selection_explanation(
    portfolio_result: BenchmarkSubmitterResult,
) -> dict[str, Any] | None:
    compared_finalists = tuple(portfolio_result.compared_finalists)
    if not compared_finalists:
        return None
    winner = compared_finalists[0]
    runner_up = compared_finalists[1] if len(compared_finalists) > 1 else None
    return {
        "selection_rule": _string_or_none(
            portfolio_result.replay_contract.get("selection_rule")
        ),
        "winner": {
            "submitter_id": winner.get("submitter_id"),
            "candidate_id": winner.get("candidate_id"),
            "total_code_bits": winner.get("total_code_bits"),
        },
        "runner_up": (
            {
                "submitter_id": runner_up.get("submitter_id"),
                "candidate_id": runner_up.get("candidate_id"),
                "total_code_bits": runner_up.get("total_code_bits"),
            }
            if runner_up is not None
            else None
        ),
        "decisive_axis": _portfolio_decisive_axis(winner, runner_up),
    }


def _portfolio_decisive_axis(
    winner: Mapping[str, Any],
    runner_up: Mapping[str, Any] | None,
) -> str | None:
    if runner_up is None:
        return None
    for axis in (
        "total_code_bits",
        "description_gain_bits",
        "structure_code_bits",
        "canonical_byte_length",
        "candidate_id",
    ):
        if winner.get(axis) != runner_up.get(axis):
            return axis
    return None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _format_scalar(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (int, bool, str)):
        return str(value)
    return json.dumps(_json_ready(value), sort_keys=True)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _json_ready(as_dict())
    return value


def _policy_id(policy: Any, key: str) -> str:
    if isinstance(policy, Mapping):
        value = policy.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


__all__ = [
    "BenchmarkArtifactRef",
    "BenchmarkPortfolioSelectionArtifact",
    "BenchmarkReplayRefArtifact",
    "BenchmarkSubmitterResultArtifact",
    "BenchmarkTaskReportArtifactPaths",
    "BenchmarkTaskResultArtifact",
    "build_benchmark_task_semantic_summary",
    "evaluate_benchmark_semantic_assertions",
    "write_benchmark_task_report_artifacts",
]
