from __future__ import annotations

import json
import math
import os
import hashlib
from dataclasses import dataclass, field, replace
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
    sha256: str | None = None

    def as_dict(self) -> dict[str, str]:
        payload = {
            "artifact_type": self.artifact_type,
            "relative_path": self.relative_path,
        }
        if self.sha256 is not None:
            payload["sha256"] = self.sha256
        return payload


@dataclass(frozen=True)
class BenchmarkReplayRefArtifact:
    task_id: str
    track_id: str
    submitter_id: str
    replay_contract: Mapping[str, Any]
    artifact_type: str = "benchmark_replay_ref"
    artifact_version: str = _ARTIFACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        replay_verification_status = _replay_verification_status_from_contract(
            self.replay_contract
        )
        return {
            "artifact_type": self.artifact_type,
            "artifact_version": self.artifact_version,
            "task_id": self.task_id,
            "track_id": self.track_id,
            "submitter_id": self.submitter_id,
            "replay_verification_status": replay_verification_status,
            "failure_reason_codes": list(
                _replay_failure_reason_codes_from_contract(self.replay_contract)
            ),
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
    replay_contract: Mapping[str, Any] = field(default_factory=dict)
    artifact_type: str = "submitter_result"
    artifact_version: str = _ARTIFACT_VERSION
    selected_candidate_id: str | None = None
    selected_candidate_hash: str | None = None
    selected_candidate_metrics: Mapping[str, Any] | None = None
    abstention_reason: str | None = None
    safe_abstention_evidence: Mapping[str, Any] = field(default_factory=dict)
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
            "replay_contract": _json_ready(self.replay_contract),
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
        if self.safe_abstention_evidence:
            payload["safe_abstention_evidence"] = _json_ready(
                self.safe_abstention_evidence
            )
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
    selected_metrics, selected_metric_source_submitter_id = (
        _selected_semantic_metric_source(
            submitter_results,
            local_winner_submitter_id=local_winner_submitter_id,
        )
    )
    metric_thresholds = _evaluate_metric_thresholds(
        task_manifest=task_manifest,
        submitter_results=submitter_results,
        selected_metrics=selected_metrics,
        local_winner_submitter_id=local_winner_submitter_id,
        local_winner_candidate_id=local_winner_candidate_id,
        source_submitter_id=selected_metric_source_submitter_id,
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
    rediscovery_target = _evaluate_rediscovery_target_assertion(
        task_manifest=task_manifest,
        submitter_results=submitter_results,
        local_winner_submitter_id=local_winner_submitter_id,
        local_winner_candidate_id=local_winner_candidate_id,
    )
    abstention_policy = _evaluate_abstention_policy_assertion(
        task_manifest=task_manifest,
        submitter_results=submitter_results,
        selected_metrics=selected_metrics,
        local_winner_submitter_id=local_winner_submitter_id,
        local_winner_candidate_id=local_winner_candidate_id,
    )
    search_scope = _evaluate_search_scope_assertion(
        task_manifest=task_manifest,
        local_winner_candidate_id=local_winner_candidate_id,
    )
    composition_operator_semantics = _evaluate_composition_operator_assertion(
        task_manifest=task_manifest,
        submitter_results=submitter_results,
        selected_metrics=selected_metrics,
        local_winner_submitter_id=local_winner_submitter_id,
        local_winner_candidate_id=local_winner_candidate_id,
    )
    sections = {
        "metric_thresholds": metric_thresholds,
        "engine_requirements": engine_requirements,
        "claim_scope": claim_scope,
        "false_claim_expectations": false_claim_expectations,
        "semantic_readiness_row_ids": semantic_readiness_row_ids,
        "abstention_policy": abstention_policy,
        "search_scope": search_scope,
        "composition_operator_semantics": composition_operator_semantics,
    }
    if rediscovery_target is not None:
        sections["rediscovery_target"] = rediscovery_target
    overall_status = (
        "passed"
        if all(section.get("status") == "passed" for section in sections.values())
        else "failed"
    )
    return {
        "overall_status": overall_status,
        "sections": _json_ready(sections),
        **sections,
    }


def _selected_semantic_metrics(
    submitter_results: Sequence[BenchmarkSubmitterResult],
    *,
    local_winner_submitter_id: str | None,
) -> Mapping[str, Any] | None:
    selected_metrics, _source_submitter_id = _selected_semantic_metric_source(
        submitter_results,
        local_winner_submitter_id=local_winner_submitter_id,
    )
    return selected_metrics


def _selected_semantic_metric_source(
    submitter_results: Sequence[BenchmarkSubmitterResult],
    *,
    local_winner_submitter_id: str | None,
) -> tuple[Mapping[str, Any] | None, str | None]:
    for result in submitter_results:
        if (
            result.submitter_id == local_winner_submitter_id
            and result.selected_candidate_metrics is not None
        ):
            return result.selected_candidate_metrics, result.submitter_id
    for result in submitter_results:
        if (
            result.submitter_id == PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
            and result.selected_candidate_metrics is not None
        ):
            return result.selected_candidate_metrics, result.submitter_id
    for result in submitter_results:
        if result.selected_candidate_metrics is not None:
            return result.selected_candidate_metrics, result.submitter_id
    return None, None


def _evaluate_metric_thresholds(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_results: Sequence[BenchmarkSubmitterResult],
    selected_metrics: Mapping[str, Any] | None,
    local_winner_submitter_id: str | None = None,
    local_winner_candidate_id: str | None = None,
    source_submitter_id: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    expected_safe_abstention = (
        getattr(task_manifest, "expected_safe_outcome", None) == "abstain"
    )
    safe_abstention = _safe_abstention_has_evidence(
        submitter_results=submitter_results,
        expected_safe_abstention=expected_safe_abstention,
        local_winner_submitter_id=local_winner_submitter_id,
        local_winner_candidate_id=local_winner_candidate_id,
    )
    for threshold_id, threshold in sorted(task_manifest.metric_thresholds.items()):
        if not isinstance(threshold, Mapping):
            rows.append(
                {
                    "threshold_id": threshold_id,
                    "source_submitter_id": source_submitter_id,
                    "source_candidate_id": local_winner_candidate_id,
                    "status": "failed",
                    "reason": "threshold_payload_not_mapping",
                    "reason_code": "threshold_payload_not_mapping",
                }
            )
            continue
        metric_id = str(threshold.get("metric_id", "")).strip()
        comparator = str(threshold.get("comparator", "")).strip()
        threshold_value = threshold.get("threshold")
        measurement_required_declared = "measurement_required" in threshold
        measurement_required = threshold.get("measurement_required", True)
        if not isinstance(measurement_required, bool):
            rows.append(
                {
                    "threshold_id": threshold_id,
                    "metric_id": metric_id,
                    "comparator": comparator,
                    "threshold": threshold_value,
                    "measurement_required": measurement_required,
                    "source_submitter_id": source_submitter_id,
                    "source_candidate_id": local_winner_candidate_id,
                    "observed_value": None,
                    "status": "failed",
                    "reason": "measurement_required_not_boolean",
                    "reason_code": "measurement_required_not_boolean",
                }
            )
            continue
        observed_value = (
            selected_metrics.get(metric_id)
            if isinstance(selected_metrics, Mapping)
            else None
        )
        if safe_abstention:
            row_status = "passed"
            reason = "not_applicable_safe_abstention"
        elif (
            expected_safe_abstention
            and observed_value is None
            and local_winner_submitter_id is None
            and local_winner_candidate_id is None
        ):
            row_status = "failed"
            reason = "safe_abstention_evidence_missing"
        elif observed_value is None and measurement_required:
            row_status = "failed"
            reason = "missing_observed_metric"
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
        row = {
            "threshold_id": threshold_id,
            "metric_id": metric_id,
            "comparator": comparator,
            "threshold": threshold_value,
            "observed_value": observed_value,
            "status": row_status,
            "reason": reason,
            "reason_code": reason,
            "source_submitter_id": source_submitter_id,
            "source_candidate_id": local_winner_candidate_id,
        }
        if measurement_required_declared:
            row["measurement_required"] = measurement_required
        rows.append(row)
    return {
        "status": (
            "passed"
            if rows and all(row["status"] == "passed" for row in rows)
            else "failed"
        ),
        "assertions": rows,
    }


def _safe_abstention_has_evidence(
    *,
    submitter_results: Sequence[BenchmarkSubmitterResult],
    expected_safe_abstention: bool,
    local_winner_submitter_id: str | None,
    local_winner_candidate_id: str | None,
) -> bool:
    if (
        not expected_safe_abstention
        or local_winner_submitter_id is not None
        or local_winner_candidate_id is not None
        or not submitter_results
    ):
        return False
    return all(
        result.status == "abstained"
        and _safe_abstention_evidence_verified(result.safe_abstention_evidence)
        for result in submitter_results
    )


def _safe_abstention_evidence_verified(evidence: Mapping[str, Any]) -> bool:
    if evidence.get("status") != "verified" or not evidence.get("reason_code"):
        return False
    if evidence.get("evidence_type") not in _SAFE_ABSTENTION_EVIDENCE_TYPES:
        return False
    return any(
        key in evidence and bool(evidence.get(key))
        for key in ("support", "child_submitter_ids", "compared_finalists")
    )


_SAFE_ABSTENTION_EVIDENCE_TYPES = {
    "falsification_gate",
    "child_falsification_gate",
    "portfolio_trap_gate",
}


def _evaluate_abstention_policy_assertion(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_results: Sequence[BenchmarkSubmitterResult],
    selected_metrics: Mapping[str, Any] | None,
    local_winner_submitter_id: str | None,
    local_winner_candidate_id: str | None,
) -> dict[str, Any]:
    expected_mode = str(
        getattr(task_manifest, "abstention_mode", None) or ""
    ).strip()
    expected_safe_abstention = (
        getattr(task_manifest, "expected_safe_outcome", None) == "abstain"
    )
    if expected_safe_abstention:
        has_evidence = _safe_abstention_has_evidence(
            submitter_results=submitter_results,
            expected_safe_abstention=True,
            local_winner_submitter_id=local_winner_submitter_id,
            local_winner_candidate_id=local_winner_candidate_id,
        )
        return {
            "status": "passed" if has_evidence else "failed",
            "expected_safe_outcome": "abstain",
            "abstention_mode": expected_mode or None,
            "reason_code": (
                "safe_abstention_falsification_evidence_verified"
                if has_evidence
                else "safe_abstention_evidence_missing"
            ),
            "evidence": [
                _json_ready(result.safe_abstention_evidence)
                for result in submitter_results
                if result.safe_abstention_evidence
            ],
        }
    if expected_mode == "calibrated_or_abstain":
        if local_winner_submitter_id is None and local_winner_candidate_id is None:
            has_evidence = _safe_abstention_has_evidence(
                submitter_results=submitter_results,
                expected_safe_abstention=True,
                local_winner_submitter_id=local_winner_submitter_id,
                local_winner_candidate_id=local_winner_candidate_id,
            )
            return {
                "status": "passed" if has_evidence else "failed",
                "abstention_mode": expected_mode,
                "reason_code": (
                    "abstained_with_falsification_evidence"
                    if has_evidence
                    else "calibrated_or_abstain_missing_calibration_and_abstention_evidence"
                ),
            }
        calibration_rows = _calibration_metric_threshold_rows(
            task_manifest=task_manifest,
            selected_metrics=selected_metrics,
        )
        calibrated = bool(calibration_rows) and all(
            row["status"] == "passed" for row in calibration_rows
        )
        return {
            "status": "passed" if calibrated else "failed",
            "abstention_mode": expected_mode,
            "reason_code": (
                "calibrated_candidate_selected"
                if calibrated
                else "selected_candidate_without_required_calibration_evidence"
            ),
            "selected_submitter_id": local_winner_submitter_id,
            "selected_candidate_id": local_winner_candidate_id,
            "calibration_assertions": calibration_rows,
        }
    return {
        "status": "passed",
        "abstention_mode": expected_mode or None,
        "reason_code": "no_special_abstention_policy",
    }


def _calibration_metric_threshold_rows(
    *,
    task_manifest: BenchmarkTaskManifest,
    selected_metrics: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold_id, threshold in sorted(task_manifest.metric_thresholds.items()):
        if not str(threshold_id).startswith("calibration:"):
            continue
        if not isinstance(threshold, Mapping):
            rows.append(
                {
                    "threshold_id": threshold_id,
                    "status": "failed",
                    "reason_code": "threshold_payload_not_mapping",
                }
            )
            continue
        metric_id = str(threshold.get("metric_id", "")).strip()
        observed_value = (
            selected_metrics.get(metric_id)
            if isinstance(selected_metrics, Mapping)
            else None
        )
        status = (
            "passed"
            if observed_value is not None
            and _compare_metric_threshold(
                observed_value,
                comparator=str(threshold.get("comparator", "")).strip(),
                threshold_value=threshold.get("threshold"),
            )
            else "failed"
        )
        rows.append(
            {
                "threshold_id": threshold_id,
                "metric_id": metric_id,
                "observed_value": observed_value,
                "status": status,
                "reason_code": (
                    "observed" if status == "passed" else "calibration_metric_failed"
                ),
            }
        )
    return rows


def _evaluate_search_scope_assertion(
    *,
    task_manifest: BenchmarkTaskManifest,
    local_winner_candidate_id: str | None,
) -> dict[str, Any]:
    honesty = task_manifest.search_class_honesty
    if not honesty:
        return {"status": "passed", "reason_code": "no_search_class_honesty_surface"}
    declared_programs = tuple(
        item
        for item in honesty.get("declared_candidate_programs", ())
        if isinstance(item, Mapping)
    )
    declared_ids = {
        str(item["candidate_id"])
        for item in declared_programs
        if isinstance(item.get("candidate_id"), str)
    }
    required_fields = (
        "coverage_statement",
        "exactness_ceiling",
        "requires_disclosure",
    )
    missing_fields = [
        field_name
        for field_name in required_fields
        if not honesty.get(field_name)
    ]
    selected_declared = (
        local_winner_candidate_id in declared_ids
        if local_winner_candidate_id is not None
        else False
    )
    status = "passed" if not missing_fields else "failed"
    return {
        "status": status,
        "reason_code": (
            "declared_candidate_scope_disclosed"
            if status == "passed"
            else "search_scope_disclosure_missing"
        ),
        "search_class": task_manifest.search_class,
        "declared_candidate_count": len(declared_ids),
        "selected_candidate_id": local_winner_candidate_id,
        "selected_candidate_in_declared_space": selected_declared,
        "independent_symbolic_rediscovery_claim": False,
        "claim_boundary": "benchmark demonstrates declared search-scope behavior, not independent law discovery outside the disclosed candidate space",
        "missing_fields": missing_fields,
    }


def _evaluate_composition_operator_assertion(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_results: Sequence[BenchmarkSubmitterResult],
    selected_metrics: Mapping[str, Any] | None,
    local_winner_submitter_id: str | None,
    local_winner_candidate_id: str | None,
) -> dict[str, Any]:
    operator_ids = tuple(task_manifest.composition_operators)
    if not operator_ids:
        return {
            "status": "passed",
            "reason_code": "no_composition_operator_surface",
        }
    selected_result = _selected_submitter_result(
        submitter_results,
        submitter_id=local_winner_submitter_id,
        candidate_id=local_winner_candidate_id,
    )
    selected_candidate = (
        selected_result.selected_candidate if selected_result is not None else None
    )
    candidate_structure = (
        selected_candidate.structural_layer.as_dict()
        if selected_candidate is not None
        else {}
    )
    composition_graph = candidate_structure.get("composition_graph")
    if not isinstance(composition_graph, Mapping):
        composition_graph = {}
    assertions: list[dict[str, Any]] = []
    for operator_id in operator_ids:
        if operator_id == "additive_residual":
            assertions.append(
                _additive_residual_behavior_assertion(
                    composition_graph=composition_graph,
                    selected_metrics=selected_metrics,
                    task_manifest=task_manifest,
                )
            )
            continue
        assertions.append(
            {
                "operator_id": operator_id,
                "status": "passed",
                "reason_code": "operator_checked_by_existing_benchmark_surface",
            }
        )
    return {
        "status": (
            "passed"
            if assertions and all(row["status"] == "passed" for row in assertions)
            else "failed"
        ),
        "assertions": assertions,
    }


def _additive_residual_behavior_assertion(
    *,
    composition_graph: Mapping[str, Any],
    selected_metrics: Mapping[str, Any] | None,
    task_manifest: BenchmarkTaskManifest,
) -> dict[str, Any]:
    operator_matches = composition_graph.get("operator_id") == "additive_residual"
    base_reducer = composition_graph.get("base_reducer")
    residual_reducer = composition_graph.get("residual_reducer")
    graph_has_distinct_components = (
        isinstance(base_reducer, str)
        and isinstance(residual_reducer, str)
        and bool(base_reducer)
        and bool(residual_reducer)
        and base_reducer != residual_reducer
    )
    margin_threshold = task_manifest.metric_thresholds.get(
        "practical_significance_margin"
    )
    margin_passed = False
    observed_margin = None
    if isinstance(margin_threshold, Mapping) and isinstance(selected_metrics, Mapping):
        metric_id = str(margin_threshold.get("metric_id", "")).strip()
        observed_margin = selected_metrics.get(metric_id)
        margin_passed = _compare_metric_threshold(
            observed_margin,
            comparator=str(margin_threshold.get("comparator", "")).strip(),
            threshold_value=margin_threshold.get("threshold"),
        )
    status = (
        "passed" if operator_matches and graph_has_distinct_components and margin_passed else "failed"
    )
    return {
        "operator_id": "additive_residual",
        "status": status,
        "reason_code": (
            "composition_operator_behavioral_margin_verified"
            if status == "passed"
            else "composition_operator_behavioral_evidence_missing"
        ),
        "operator_matches": operator_matches,
        "graph_has_distinct_components": graph_has_distinct_components,
        "base_reducer": base_reducer,
        "residual_reducer": residual_reducer,
        "observed_practical_significance_margin": observed_margin,
        "claim_boundary": "additive residual evidence requires both a distinct composition graph and a passed replayed benchmark margin",
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


def _evaluate_rediscovery_target_assertion(
    *,
    task_manifest: BenchmarkTaskManifest,
    submitter_results: Sequence[BenchmarkSubmitterResult],
    local_winner_submitter_id: str | None,
    local_winner_candidate_id: str | None,
) -> dict[str, Any] | None:
    target_ref = getattr(task_manifest, "target_structure_ref", None)
    if not isinstance(target_ref, str) or not target_ref.strip():
        return None
    target = _load_rediscovery_target_spec(
        target_ref.strip(),
        source_path=task_manifest.source_path,
    )
    target_family = _target_family(target, target_ref.strip())
    selected_result = _selected_submitter_result(
        submitter_results,
        submitter_id=local_winner_submitter_id,
        candidate_id=local_winner_candidate_id,
    )
    selected_candidate = (
        selected_result.selected_candidate if selected_result is not None else None
    )
    equivalent = _selected_candidate_matches_rediscovery_target(
        target=target,
        target_family=target_family,
        selected_candidate_id=local_winner_candidate_id,
        selected_candidate=selected_candidate,
    )
    return {
        "status": "passed" if equivalent else "failed",
        "reason_code": (
            "selected_candidate_structurally_equivalent"
            if equivalent
            else "selected_candidate_not_structurally_equivalent"
        ),
        "target_structure_ref": target_ref.strip(),
        "target_family": target_family,
        "selected_submitter_id": local_winner_submitter_id,
        "selected_candidate_id": local_winner_candidate_id,
        "equivalence_policy": dict(getattr(task_manifest, "equivalence_policy", {})),
    }


def _selected_submitter_result(
    submitter_results: Sequence[BenchmarkSubmitterResult],
    *,
    submitter_id: str | None,
    candidate_id: str | None,
) -> BenchmarkSubmitterResult | None:
    for result in submitter_results:
        if result.submitter_id == submitter_id and result.selected_candidate_id == candidate_id:
            return result
    for result in submitter_results:
        if result.selected_candidate_id == candidate_id:
            return result
    return None


def _load_rediscovery_target_spec(
    target_ref: str,
    *,
    source_path: Path,
) -> Mapping[str, Any]:
    target_path_token = target_ref.split("#", 1)[0]
    if not target_path_token:
        return {}
    target_path = Path(target_path_token)
    candidate_paths = [target_path] if target_path.is_absolute() else []
    if not target_path.is_absolute():
        project_root = _project_root_for_manifest_source(source_path)
        candidate_paths.extend(
            [
                project_root / target_path,
                project_root / "src" / "euclid" / "_assets" / target_path,
            ]
        )
    for candidate_path in candidate_paths:
        if not candidate_path.is_file():
            continue
        try:
            text = candidate_path.read_text(encoding="utf-8")
            if candidate_path.suffix.lower() == ".json":
                payload = json.loads(text)
            else:
                payload = yaml.safe_load(text)
        except (OSError, json.JSONDecodeError, yaml.YAMLError):
            continue
        return payload if isinstance(payload, Mapping) else {}
    return {}


def _project_root_for_manifest_source(source_path: Path) -> Path:
    parts = source_path.parts
    if "benchmarks" in parts:
        benchmark_index = parts.index("benchmarks")
        return Path(*parts[:benchmark_index]) if benchmark_index else Path(".")
    return source_path.parent


def _target_family(target: Mapping[str, Any], target_ref: str) -> str | None:
    raw_family = target.get("family") or target.get("equivalence_class")
    if isinstance(raw_family, str) and raw_family.strip():
        return raw_family.strip()
    if "#" in target_ref:
        fragment = target_ref.rsplit("#", 1)[1].strip()
        return fragment or None
    return None


def _selected_candidate_matches_rediscovery_target(
    *,
    target: Mapping[str, Any],
    target_family: str | None,
    selected_candidate_id: str | None,
    selected_candidate: Any,
) -> bool:
    if not selected_candidate_id:
        return False
    target_id = str(target.get("candidate_id") or target.get("law_id") or "").strip()
    if target_id and selected_candidate_id == target_id:
        return True
    if target_family == selected_candidate_id:
        return True
    if target_family == "algorithmic_last_observation":
        return selected_candidate_id == "algorithmic_last_observation"
    if target_family == "affine_lag":
        return _candidate_is_analytic_affine_lag(
            selected_candidate_id=selected_candidate_id,
            selected_candidate=selected_candidate,
        )
    return False


def _candidate_is_analytic_affine_lag(
    *,
    selected_candidate_id: str,
    selected_candidate: Any,
) -> bool:
    if selected_candidate_id == "analytic_lag1_affine":
        return True
    if selected_candidate is None:
        return False
    structural_layer = getattr(selected_candidate, "structural_layer", None)
    if getattr(structural_layer, "cir_family_id", None) != "analytic":
        return False
    parameter_block = getattr(structural_layer, "parameter_block", None)
    parameters = getattr(parameter_block, "parameters", ())
    parameter_names = {str(getattr(parameter, "name", "")) for parameter in parameters}
    return "lag_coefficient" in parameter_names


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

    for submitter_id, artifact in bundle.replay_artifacts.items():
        _write_json(bundle.replay_paths[submitter_id], artifact.as_dict())

    replay_refs_with_digest = {
        submitter_id: _artifact_ref(
            benchmark_root=tree.root,
            artifact_type="benchmark_replay_ref",
            path=path,
            sha256=_sha256_file(path),
        )
        for submitter_id, path in bundle.replay_paths.items()
    }
    submitter_artifacts = {
        submitter_id: replace(
            artifact,
            replay_ref=replay_refs_with_digest.get(
                submitter_id,
                artifact.replay_ref,
            ),
        )
        for submitter_id, artifact in bundle.submitter_artifacts.items()
    }
    for submitter_id, artifact in submitter_artifacts.items():
        _write_json(bundle.submitter_paths[submitter_id], artifact.as_dict())

    if (
        bundle.portfolio_selection_artifact is not None
        and bundle.portfolio_selection_path is not None
    ):
        _write_json(
            bundle.portfolio_selection_path,
            bundle.portfolio_selection_artifact.as_dict(),
        )

    portfolio_selection_ref = None
    if bundle.portfolio_selection_path is not None:
        portfolio_selection_ref = _artifact_ref(
            benchmark_root=tree.root,
            artifact_type="portfolio_selection_record",
            path=bundle.portfolio_selection_path,
            sha256=_sha256_file(bundle.portfolio_selection_path),
        )
        portfolio_artifact = submitter_artifacts.get(
            PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
        )
        if portfolio_artifact is not None:
            submitter_artifacts[PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID] = replace(
                portfolio_artifact,
                portfolio_selection_record_ref=portfolio_selection_ref,
            )
            _write_json(
                bundle.submitter_paths[PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID],
                submitter_artifacts[
                    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
                ].as_dict(),
            )

    submitter_refs_with_digest = {
        submitter_id: _artifact_ref(
            benchmark_root=tree.root,
            artifact_type="submitter_result",
            path=path,
            sha256=_sha256_file(path),
        )
        for submitter_id, path in bundle.submitter_paths.items()
    }
    task_result = replace(
        bundle.task_result,
        submitter_result_refs=tuple(
            submitter_refs_with_digest.get(ref_submitter_id, ref)
            for ref_submitter_id, ref in zip(
                (result.submitter_id for result in submitter_results),
                bundle.task_result.submitter_result_refs,
            )
        ),
        replay_ref_refs=tuple(
            replay_refs_with_digest.get(result.submitter_id, ref)
            for result, ref in zip(
                submitter_results,
                bundle.task_result.replay_ref_refs,
            )
        ),
        portfolio_selection_record_ref=portfolio_selection_ref
        or bundle.task_result.portfolio_selection_record_ref,
    )
    _write_json(bundle.task_result_path, task_result.as_dict())
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
            replay_contract=result.replay_contract,
            selected_candidate_id=result.selected_candidate_id,
            selected_candidate_hash=result.selected_candidate_hash,
            selected_candidate_metrics=result.selected_candidate_metrics,
            abstention_reason=result.abstention_reason,
            safe_abstention_evidence=result.safe_abstention_evidence,
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
                replay_contract=portfolio_artifact.replay_contract,
                selected_candidate_id=portfolio_artifact.selected_candidate_id,
                selected_candidate_hash=portfolio_artifact.selected_candidate_hash,
                selected_candidate_metrics=portfolio_artifact.selected_candidate_metrics,
                abstention_reason=portfolio_artifact.abstention_reason,
                safe_abstention_evidence=portfolio_artifact.safe_abstention_evidence,
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
        local_winner_candidate_id=local_winner_candidate_id,
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
    local_winner_candidate_id: str | None,
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

    semantic_assertions = evaluate_benchmark_semantic_assertions(
        task_manifest=task_manifest,
        submitter_results=submitter_results,
        local_winner_submitter_id=local_winner_submitter_id,
        local_winner_candidate_id=local_winner_candidate_id,
    )
    body_lines.extend(
        [
            "",
            "## Semantic Assertions",
            "",
            f"- Overall status: `{semantic_assertions['overall_status']}`",
        ]
    )
    for section_name, section in semantic_assertions.get("sections", {}).items():
        if not isinstance(section, Mapping):
            continue
        body_lines.append(
            f"- `{section_name}`: `{_format_scalar(section.get('status'))}`"
        )

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
            if explanation.get("decisive_axis") == "metric_threshold_gate":
                body_lines.append(
                    "- Selection reason: metric threshold gate "
                    f"`{_format_scalar(explanation.get('decision_reason_code'))}`."
                )
            else:
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
    sha256: str | None = None,
) -> BenchmarkArtifactRef:
    return BenchmarkArtifactRef(
        artifact_type=artifact_type,
        relative_path=path.relative_to(benchmark_root).as_posix(),
        sha256=sha256,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


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
    threshold_gate_step = _portfolio_threshold_gate_step(portfolio_result)
    decisive_axis = _portfolio_decisive_axis(winner, runner_up)
    if (
        threshold_gate_step is not None
        and runner_up is not None
        and _numeric_value(runner_up.get("total_code_bits"))
        < _numeric_value(winner.get("total_code_bits"))
    ):
        decisive_axis = "metric_threshold_gate"
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
        "decisive_axis": decisive_axis,
        "decision_reason_code": (
            threshold_gate_step.get("reason_code")
            if threshold_gate_step is not None
            else None
        ),
    }


def _portfolio_threshold_gate_step(
    portfolio_result: BenchmarkSubmitterResult,
) -> Mapping[str, Any] | None:
    for step in portfolio_result.decision_trace:
        if isinstance(step, Mapping) and step.get("step") == "benchmark_metric_threshold_gate":
            return step
    return None


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


def _numeric_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _replay_verification_status_from_contract(
    replay_contract: Mapping[str, Any],
) -> str:
    status = _string_or_none(replay_contract.get("replay_verification_status"))
    if status is not None:
        return status
    return "unverified"


def _replay_failure_reason_codes_from_contract(
    replay_contract: Mapping[str, Any],
) -> tuple[str, ...]:
    reason_codes = replay_contract.get("failure_reason_codes")
    if isinstance(reason_codes, str) and reason_codes.strip():
        return (reason_codes.strip(),)
    if isinstance(reason_codes, Sequence) and not isinstance(
        reason_codes,
        (str, bytes, bytearray),
    ):
        return tuple(str(code) for code in reason_codes if str(code).strip())
    if _replay_verification_status_from_contract(replay_contract) != _VERIFIED_REPLAY_STATUS:
        return ("unverified_replay_artifact",)
    return ()


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
