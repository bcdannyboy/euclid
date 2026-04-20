from __future__ import annotations

from pathlib import Path

import euclid.release as release
from euclid.readiness import ReadinessGateResult, judge_readiness

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_RELEASE_NOTEBOOK = (
    PROJECT_ROOT / "output/jupyter-notebook/current_release.ipynb"
)
NOTEBOOK_SMOKE_SUMMARY = (
    PROJECT_ROOT / "build/notebook-smoke/notebook-smoke-summary.json"
)
MISSING_SUITE_RESULT = type(
    "SuiteResult",
    (),
    {"summary_path": PROJECT_ROOT / "missing-summary.json"},
)()


def _suite_judgment():
    return judge_readiness(
        judgment_id="completion_report_generation",
        gate_results=(
            ReadinessGateResult(
                gate_id="track.predictive_generalization",
                status="passed",
                summary="predictive track passed",
            ),
            ReadinessGateResult(
                gate_id="surface.retained_core_release",
                status="passed",
                summary="retained-core surface passed",
            ),
        ),
    )


def test_row_evidence_payload_supports_all_capability_types() -> None:
    suite_summary = {
        "task_results": [
            {
                "track_id": "predictive_generalization",
                "task_id": "distribution_case",
                "forecast_object_type": "distribution",
                "replay_verification": "verified",
            }
        ],
        "surface_statuses": [
            {
                "surface_id": "retained_core_release",
                "benchmark_status": "passed",
                "replay_status": "passed",
                "evidence": {},
            }
        ],
        "search_class_coverage": [],
        "composition_operator_coverage": [],
        "semantic_summary": {
            "reducer_family_ids": ["analytic"],
            "run_support_object_ids": ["target_transform_object"],
            "admissibility_rule_ids": ["family_membership"],
            "lifecycle_artifact_ids": ["run_result"],
            "operator_runtime_surface_ids": ["operator_run"],
            "replay_surface_ids": ["operator_native_replay"],
        },
    }
    clean_install_report = {
        "surface_completion": 1.0,
        "surfaces": [
            {
                "surface_id": "release_status",
                "status": "passed",
                "reason_codes": [],
                "evidence_refs": ["artifact:clean-install-certification.json"],
            }
        ],
    }

    expectations = {
        ("reducer_family", "analytic"): {"semantic_runtime"},
        ("run_support_object", "target_transform_object"): {"semantic_runtime"},
        ("admissibility_rule", "family_membership"): {"semantic_runtime"},
        ("lifecycle_artifact", "run_result"): {"semantic_runtime"},
        ("operator_runtime_surface", "operator_run"): {"semantic_runtime"},
        ("clean_install_surface", "release_status"): {"packaging_install"},
        ("replay_surface", "operator_native_replay"): {"replay"},
    }

    for (capability_type, capability_id), expected_classes in expectations.items():
        evidence = release._row_evidence_payload(
            capability_type=capability_type,
            capability_id=capability_id,
            suite_summary=suite_summary,
            suite_readiness_judgment=_suite_judgment(),
            notebook_smoke=release.NotebookSmokeResult(
                project_root=PROJECT_ROOT,
                notebook_path=CURRENT_RELEASE_NOTEBOOK,
                output_root=PROJECT_ROOT / "build/notebook-smoke",
                summary_path=NOTEBOOK_SMOKE_SUMMARY,
                probabilistic_case_ids=(),
                publication_mode="abstention_only_publication",
                catalog_entries=0,
            ),
            clean_install_report=clean_install_report,
        )
        assert expected_classes <= set(evidence["available_classes"])


def test_completion_report_marks_missing_vs_failed_proof_correctly() -> None:
    missing_row = release._evaluate_completion_row(
        row={
            "row_id": "operator_runtime_surface:operator_replay",
            "capability_type": "operator_runtime_surface",
            "capability_id": "operator_replay",
            "minimum_closing_evidence_classes": ["semantic_runtime", "replay"],
            "governing_doc_refs": ["EUCLID.md"],
        },
        suite_result=MISSING_SUITE_RESULT,
        suite_readiness_judgment=_suite_judgment(),
        notebook_smoke=release.NotebookSmokeResult(
            project_root=PROJECT_ROOT,
            notebook_path=CURRENT_RELEASE_NOTEBOOK,
            output_root=PROJECT_ROOT / "build/notebook-smoke",
            summary_path=NOTEBOOK_SMOKE_SUMMARY,
            probabilistic_case_ids=(),
            publication_mode="abstention_only_publication",
            catalog_entries=0,
        ),
        clean_install_report={"surface_completion": 0.0, "surfaces": []},
        closure_row=None,
        bundles_by_scope={},
    )
    assert missing_row.proof_status == "missing_proof"
    assert missing_row.status == "blocked"

    failed_row = release._evaluate_completion_row(
        row={
            "row_id": "clean_install_surface:release_status",
            "capability_type": "clean_install_surface",
            "capability_id": "release_status",
            "minimum_closing_evidence_classes": ["packaging_install"],
            "governing_doc_refs": ["EUCLID.md"],
        },
        suite_result=MISSING_SUITE_RESULT,
        suite_readiness_judgment=_suite_judgment(),
        notebook_smoke=release.NotebookSmokeResult(
            project_root=PROJECT_ROOT,
            notebook_path=CURRENT_RELEASE_NOTEBOOK,
            output_root=PROJECT_ROOT / "build/notebook-smoke",
            summary_path=NOTEBOOK_SMOKE_SUMMARY,
            probabilistic_case_ids=(),
            publication_mode="abstention_only_publication",
            catalog_entries=0,
        ),
        clean_install_report={
            "surface_completion": 0.0,
            "surfaces": [
                {
                    "surface_id": "release_status",
                    "status": "failed",
                    "reason_codes": ["clean_install.release_status_failed"],
                    "evidence_refs": ["artifact:clean-install-certification.json"],
                }
            ],
        },
        closure_row=None,
        bundles_by_scope={},
    )
    assert failed_row.proof_status == "failed_proof"
    assert failed_row.status == "blocked"


def test_completion_report_confidence_reflects_real_signal_coverage() -> None:
    dense_signal_rows = (
        release._CompletionLedgerRow(
            row_id="forecast_object_type:point",
            status="complete",
            required_closing_classes=("semantic_runtime", "replay"),
            available_evidence_classes=("semantic_runtime", "replay"),
            non_closing_evidence_classes=(),
            reason_codes=(),
            evidence_refs=("artifact:a",),
            evidence_bundle_ids=(),
            proof_status=None,
        ),
        release._CompletionLedgerRow(
            row_id="forecast_object_type:distribution",
            status="partial",
            required_closing_classes=("semantic_runtime", "replay"),
            available_evidence_classes=("semantic_runtime",),
            non_closing_evidence_classes=(),
            reason_codes=("distribution_missing_replay",),
            evidence_refs=("artifact:b",),
            evidence_bundle_ids=(),
            proof_status="missing_proof",
        ),
    )
    sparse_signal_rows = (
        release._CompletionLedgerRow(
            row_id="forecast_object_type:point",
            status="complete",
            required_closing_classes=("semantic_runtime", "replay"),
            available_evidence_classes=("semantic_runtime",),
            non_closing_evidence_classes=(),
            reason_codes=(),
            evidence_refs=("artifact:a",),
            evidence_bundle_ids=(),
            proof_status=None,
        ),
        release._CompletionLedgerRow(
            row_id="forecast_object_type:distribution",
            status="partial",
            required_closing_classes=("semantic_runtime", "replay"),
            available_evidence_classes=(),
            non_closing_evidence_classes=("notebook_smoke",),
            reason_codes=("distribution_missing_runtime",),
            evidence_refs=("artifact:b",),
            evidence_bundle_ids=(),
            proof_status="missing_proof",
        ),
    )
    completion_values = {
        "full_vision_completion": 0.5,
        "current_gate_completion": 0.5,
        "shipped_releasable_completion": 0.5,
    }

    dense_confidence = release._build_completion_report_confidence(
        completion_rows=dense_signal_rows,
        completion_values=completion_values,
    )
    sparse_confidence = release._build_completion_report_confidence(
        completion_rows=sparse_signal_rows,
        completion_values=completion_values,
    )

    assert dense_confidence["score"] > sparse_confidence["score"]
