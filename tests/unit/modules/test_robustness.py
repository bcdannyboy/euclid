from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.modules.robustness import (
    PerturbationRunRecord,
    build_null_result_manifest,
    build_perturbation_family_result_manifest,
    build_robustness_report,
    build_sensitivity_analysis_manifest,
    evaluate_aggregate_metric_results,
    evaluate_null_comparison,
    evaluate_perturbation_family,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _ref(schema_name: str, object_id: str) -> TypedRef:
    return TypedRef(schema_name=schema_name, object_id=object_id)


def test_evaluate_null_comparison_uses_plus_one_upper_tail_rule() -> None:
    result = evaluate_null_comparison(
        observed_statistic=1.25,
        surrogate_statistics=(0.1, 0.4, 0.6, 1.4),
        max_p_value=0.25,
    )

    assert result.status == "null_not_rejected"
    assert result.failure_reason_code is None
    assert result.monte_carlo_p_value == pytest.approx(0.4)
    assert result.resample_count == 4


def test_aggregate_perturbation_metrics_uses_minimum_across_families() -> None:
    metric_refs = {
        "canonical_form_exact_match_rate": _ref(
            "retention_metric_manifest@1.0.0", "match"
        ),
        "positive_description_gain_rate": _ref(
            "retention_metric_manifest@1.0.0", "gain"
        ),
        "outer_baseline_win_rate": _ref(
            "retention_metric_manifest@1.0.0", "baseline"
        ),
    }
    truncation = evaluate_perturbation_family(
        family_id="recent_history_truncation",
        metric_refs_by_id=metric_refs,
        runs=(
            PerturbationRunRecord(
                perturbation_id="history-50",
                canonical_form_matches=True,
                description_gain_bits=1.1,
                outer_candidate_score=0.4,
                outer_baseline_score=0.9,
            ),
            PerturbationRunRecord(
                perturbation_id="history-75",
                canonical_form_matches=False,
                description_gain_bits=-0.2,
                outer_candidate_score=1.0,
                outer_baseline_score=0.7,
            ),
        ),
    )
    coarsening = evaluate_perturbation_family(
        family_id="quantization_coarsening",
        metric_refs_by_id=metric_refs,
        runs=(
            PerturbationRunRecord(
                perturbation_id="quantize-2x",
                canonical_form_matches=True,
                description_gain_bits=0.8,
                outer_candidate_score=0.5,
                outer_baseline_score=0.8,
            ),
            PerturbationRunRecord(
                perturbation_id="quantize-4x",
                canonical_form_matches=True,
                description_gain_bits=0.4,
                outer_candidate_score=0.6,
                outer_baseline_score=0.9,
            ),
        ),
    )

    aggregates, stability_status = evaluate_aggregate_metric_results(
        family_results=(truncation, coarsening),
        required_metric_refs=tuple(metric_refs.values()),
        metric_thresholds={"match": 0.75, "gain": 0.5, "baseline": 0.5},
    )

    aggregate_values = {item.metric_ref.object_id: item.value for item in aggregates}

    assert aggregate_values == pytest.approx(
        {"match": 0.5, "gain": 0.5, "baseline": 0.5}
    )
    assert stability_status == "failed"


def test_build_robustness_report_fails_when_canary_survives() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    metric_refs = {
        "canonical_form_exact_match_rate": _ref(
            "retention_metric_manifest@1.0.0", "match"
        ),
        "positive_description_gain_rate": _ref(
            "retention_metric_manifest@1.0.0", "gain"
        ),
        "outer_baseline_win_rate": _ref(
            "retention_metric_manifest@1.0.0", "baseline"
        ),
    }
    family_result = evaluate_perturbation_family(
        family_id="recent_history_truncation",
        metric_refs_by_id=metric_refs,
        runs=(
            PerturbationRunRecord(
                perturbation_id="history-50",
                canonical_form_matches=True,
                description_gain_bits=0.5,
                outer_candidate_score=0.3,
                outer_baseline_score=0.8,
            ),
        ),
    )
    aggregate_results, stability_status = evaluate_aggregate_metric_results(
        family_results=(family_result,),
        required_metric_refs=tuple(metric_refs.values()),
        metric_thresholds={"match": 0.5, "gain": 0.5, "baseline": 0.5},
    )
    null_result = evaluate_null_comparison(
        observed_statistic=1.0,
        surrogate_statistics=(0.1, 0.2, 0.3, 0.4),
        max_p_value=0.25,
    )
    null_result_manifest = build_null_result_manifest(
        catalog,
        null_result_id="candidate_null_result",
        null_protocol_ref=_ref("null_protocol_manifest@1.1.0", "null_protocol"),
        candidate_id="candidate",
        evaluation=null_result,
    )
    canary_refs = (
        _ref("leakage_canary_result_manifest@1.0.0", "future"),
        _ref("leakage_canary_result_manifest@1.0.0", "late"),
        _ref("leakage_canary_result_manifest@1.0.0", "holdout"),
        _ref("leakage_canary_result_manifest@1.0.0", "revision"),
    )
    canary_results = (
        {
            "canary_type": "future_target_level_feature",
            "pass": True,
        },
        {
            "canary_type": "late_available_target_copy",
            "pass": True,
        },
        {
            "canary_type": "holdout_membership_feature",
            "pass": False,
            "failure_reason_codes": ["unexpected_canary_survival"],
        },
        {
            "canary_type": "post_cutoff_revision_level_feature",
            "pass": True,
        },
    )
    perturbation_family_manifest = build_perturbation_family_result_manifest(
        catalog,
        perturbation_family_result_id="candidate_recent_history_truncation",
        perturbation_protocol_ref=_ref(
            "perturbation_protocol_manifest@1.1.0", "perturbation_protocol"
        ),
        candidate_id="candidate",
        evaluation=family_result,
    )
    sensitivity_analysis_manifest = build_sensitivity_analysis_manifest(
        catalog,
        sensitivity_analysis_id="candidate_history_50_sensitivity",
        perturbation_family_result_ref=perturbation_family_manifest.ref,
        candidate_id="candidate",
        analysis={
            "analysis_id": "history-50",
            "family_id": "recent_history_truncation",
        },
    )

    report = build_robustness_report(
        candidate_id="candidate",
        null_protocol_ref=_ref("null_protocol_manifest@1.1.0", "null_protocol"),
        null_result=null_result,
        null_result_ref=null_result_manifest.ref,
        perturbation_protocol_ref=_ref(
            "perturbation_protocol_manifest@1.1.0", "perturbation_protocol"
        ),
        perturbation_family_results=(family_result,),
        perturbation_family_result_refs=(perturbation_family_manifest.ref,),
        aggregate_metric_results=aggregate_results,
        stability_status=stability_status,
        leakage_canary_result_refs=canary_refs,
        leakage_canary_results=canary_results,
        candidate_context={"frozen_candidate_set_id": "shortlist"},
        sensitivity_analysis_refs=(sensitivity_analysis_manifest.ref,),
    )

    manifest = report.to_manifest(catalog)

    assert report.final_robustness_status == "failed"
    assert manifest.body["candidate_id"] == "candidate"
    assert manifest.body["null_result_ref"] == null_result_manifest.ref.as_dict()
    assert manifest.body["perturbation_family_result_refs"] == [
        perturbation_family_manifest.ref.as_dict()
    ]
    assert manifest.body["sensitivity_analysis_refs"] == [
        sensitivity_analysis_manifest.ref.as_dict()
    ]
    assert "null_result" not in manifest.body
    assert "perturbation_family_results" not in manifest.body
    assert manifest.body["required_canary_type_coverage"] == [
        "future_target_level_feature",
        "late_available_target_copy",
        "holdout_membership_feature",
        "post_cutoff_revision_level_feature",
    ]
    assert "sensitivity_analyses" not in manifest.body
