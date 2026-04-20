from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.manifests.runtime_models import (
    NullResultManifest,
    PerturbationFamilyResultManifest,
    RobustnessReportManifest,
    SensitivityAnalysisManifest,
)

_OWNER_PROMPT_ID = "prompt.nulls-stability-leakage-v1"
_SCOPE_ID = "euclid_v1_binding_scope@1.0.0"
_REQUIRED_CANARY_TYPE_COVERAGE = (
    "future_target_level_feature",
    "late_available_target_copy",
    "holdout_membership_feature",
    "post_cutoff_revision_level_feature",
)
_NULL_ABSTENTION_CODES = {
    "insufficient_series_length",
    "nonfinite_observation",
    "codelength_comparability_failed",
    "nonfinite_statistic",
    "surrogate_generation_failed",
    "candidate_refit_failed_on_surrogate",
}
_NULL_FAILURE_CODES = {
    "malformed_null_protocol",
    "out_of_scope_null_family",
    "unsupported_null_statistic",
}
_METRIC_APPLICABILITY = {
    "canonical_form_exact_match_rate": "always",
    "positive_description_gain_rate": "requires_description_gain",
    "outer_baseline_win_rate": "requires_outer_predictive_evaluation",
}
_CANARY_DEFAULTS = {
    "future_target_level_feature": {
        "expected_block_stage": "feature_spec_validation",
        "expected_reason_code": "future_target_feature_detected",
        "horizon": 1,
    },
    "late_available_target_copy": {
        "expected_block_stage": "time_safety_audit",
        "expected_reason_code": "feature_available_after_origin",
        "horizon": 0,
    },
    "holdout_membership_feature": {
        "expected_block_stage": "evaluation_plan_binding",
        "expected_reason_code": "holdout_membership_feature_detected",
        "horizon": 0,
    },
    "post_cutoff_revision_level_feature": {
        "expected_block_stage": "time_safety_audit",
        "expected_reason_code": "revision_after_cutoff_detected",
        "horizon": 0,
    },
}
_CANARY_STAGE_ORDER = {
    "feature_spec_validation": 0,
    "time_safety_audit": 1,
    "evaluation_plan_binding": 2,
    "not_blocked": 99,
}


@dataclass(frozen=True)
class NullComparisonEvaluation:
    status: str
    failure_reason_code: str | None
    observed_statistic: float | None
    surrogate_statistics: tuple[float, ...]
    monte_carlo_p_value: float | None
    max_p_value: float | None
    resample_count: int
    statistic_id: str = "description_gain_bits"
    statistic_orientation: str = "larger_is_more_structure"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure_reason_code": self.failure_reason_code,
            "observed_statistic": self.observed_statistic,
            "surrogate_statistics": list(self.surrogate_statistics),
            "monte_carlo_p_value": self.monte_carlo_p_value,
            "max_p_value": self.max_p_value,
            "resample_count": self.resample_count,
            "statistic_id": self.statistic_id,
            "statistic_orientation": self.statistic_orientation,
        }


@dataclass(frozen=True)
class PerturbationRunRecord:
    perturbation_id: str
    canonical_form_matches: bool
    description_gain_bits: float | None
    outer_candidate_score: float | None = None
    outer_baseline_score: float | None = None
    failure_reason_code: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "perturbation_id": self.perturbation_id,
            "canonical_form_matches": self.canonical_form_matches,
            "description_gain_bits": self.description_gain_bits,
            "outer_candidate_score": self.outer_candidate_score,
            "outer_baseline_score": self.outer_baseline_score,
            "failure_reason_code": self.failure_reason_code,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PerturbationMetricEvaluation:
    metric_ref: TypedRef
    metric_id: str
    applicability_status: str
    value: float | None
    failure_reason_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_ref": self.metric_ref.as_dict(),
            "metric_id": self.metric_id,
            "applicability_status": self.applicability_status,
            "value": self.value,
            "failure_reason_code": self.failure_reason_code,
        }


@dataclass(frozen=True)
class PerturbationFamilyEvaluation:
    family_id: str
    status: str
    valid_run_count: int
    invalid_run_count: int
    metric_results: tuple[PerturbationMetricEvaluation, ...]
    perturbation_runs: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "status": self.status,
            "valid_run_count": self.valid_run_count,
            "invalid_run_count": self.invalid_run_count,
            "metric_results": [item.as_dict() for item in self.metric_results],
            "perturbation_runs": [dict(item) for item in self.perturbation_runs],
        }


@dataclass(frozen=True)
class AggregateMetricEvaluation:
    metric_ref: TypedRef
    metric_id: str
    status: str
    value: float | None
    failure_reason_code: str | None = None
    contributing_family_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_ref": self.metric_ref.as_dict(),
            "metric_id": self.metric_id,
            "status": self.status,
            "value": self.value,
            "failure_reason_code": self.failure_reason_code,
            "contributing_family_ids": list(self.contributing_family_ids),
        }


def build_surrogate_generator_manifest(
    catalog: ContractCatalog,
    *,
    surrogate_generator_id: str,
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="surrogate_generator_manifest@1.0.0",
        module_id="robustness",
        body={
            "surrogate_generator_id": surrogate_generator_id,
            "owner_prompt_id": _OWNER_PROMPT_ID,
            "scope_id": _SCOPE_ID,
            "generator_id": "without_replacement_random_permutation",
            "input_object_type": "single_entity_ordered_numeric_sequence",
            "preserved_structure_ids": [
                "sequence_length",
                "exact_empirical_value_multiset",
            ],
            "destroyed_structure_ids": [
                "temporal_order",
                "serial_dependence",
            ],
            "minimum_series_length": 3,
            "requires_finite_values_only": True,
            "seed_type": "u64",
        },
        catalog=catalog,
    )


def build_null_protocol_manifest(
    catalog: ContractCatalog,
    *,
    protocol_id: str,
    surrogate_generator_ref: TypedRef,
    resample_count: int,
    max_p_value: str,
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="null_protocol_manifest@1.1.0",
        module_id="robustness",
        body={
            "protocol_id": protocol_id,
            "owner_prompt_id": _OWNER_PROMPT_ID,
            "scope_id": _SCOPE_ID,
            "null_family_id": "iid_permutation_null",
            "surrogate_generator_ref": surrogate_generator_ref.as_dict(),
            "statistic_id": "description_gain_bits",
            "statistic_orientation": "larger_is_more_structure",
            "resample_count": resample_count,
            "monte_carlo_p_value_rule": "plus_one_upper_tail",
            "max_p_value": max_p_value,
            "fit_policy": "refit_frozen_candidate_on_every_surrogate",
            "search_adjustment_status": (
                "not_search_adjusted_candidate_relative_only"
            ),
            "interpretation_scope": "descriptive_robustness_only",
            "codelength_comparability_required": True,
            "abstention_on_generator_failure": True,
        },
        catalog=catalog,
    )


def build_retention_metric_manifest(
    catalog: ContractCatalog,
    *,
    retention_metric_id: str,
    metric_id: str,
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="retention_metric_manifest@1.0.0",
        module_id="robustness",
        body={
            "retention_metric_id": retention_metric_id,
            "owner_prompt_id": _OWNER_PROMPT_ID,
            "scope_id": _SCOPE_ID,
            "metric_id": metric_id,
            "value_range": "[0,1]",
            "aggregation_denominator": "valid_perturbation_run_count",
            "applicability_rule": _METRIC_APPLICABILITY[metric_id],
            "higher_is_better": True,
        },
        catalog=catalog,
    )


def build_perturbation_protocol_manifest(
    catalog: ContractCatalog,
    *,
    protocol_id: str,
    base_codelength_policy_ref: TypedRef,
    baseline_registry_ref: TypedRef,
    frozen_baseline_id: str,
    point_score_policy_ref: TypedRef,
    required_metric_refs: Sequence[TypedRef],
    metric_thresholds: Mapping[str, float],
) -> ManifestEnvelope:
    threshold_entries: list[dict[str, Any]] = []
    for metric_ref in required_metric_refs:
        threshold_entries.append(
            {
                "metric_ref": metric_ref.as_dict(),
                "min_value": _format_rate(metric_thresholds[metric_ref.object_id]),
            }
        )
    return ManifestEnvelope.build(
        schema_name="perturbation_protocol_manifest@1.1.0",
        module_id="robustness",
        body={
            "protocol_id": protocol_id,
            "owner_prompt_id": _OWNER_PROMPT_ID,
            "scope_id": _SCOPE_ID,
            "base_codelength_policy_ref": base_codelength_policy_ref.as_dict(),
            "baseline_registry_ref": baseline_registry_ref.as_dict(),
            "frozen_baseline_id": frozen_baseline_id,
            "point_score_policy_ref": point_score_policy_ref.as_dict(),
            "perturbations": [
                {
                    "perturbation_id": "recent_history_truncation_grid",
                    "family_id": "recent_history_truncation",
                    "recent_history_fraction_grid": ["0.5", "0.75"],
                    "quantization_multiplier_grid": [],
                    "minimum_valid_post_perturbation_length": 3,
                },
                {
                    "perturbation_id": "quantization_coarsening_grid",
                    "family_id": "quantization_coarsening",
                    "recent_history_fraction_grid": [],
                    "quantization_multiplier_grid": [2, 4],
                    "minimum_valid_post_perturbation_length": 3,
                },
            ],
            "required_metric_refs": [ref.as_dict() for ref in required_metric_refs],
            "metric_thresholds": threshold_entries,
            "family_aggregation_mode": "minimum_across_applicable_families",
            "overall_aggregation_rule": "all_required_thresholds_must_pass",
            "predictive_evaluation_scope": "outer_folds_only",
            "holdout_reuse_forbidden": True,
            "codelength_policy_mutation_forbidden": True,
            "abstention_on_no_valid_runs": True,
        },
        catalog=catalog,
    )


def build_leakage_canary_manifest(
    catalog: ContractCatalog,
    *,
    canary_id: str,
    canary_type: str,
) -> ManifestEnvelope:
    default = _CANARY_DEFAULTS[canary_type]
    return ManifestEnvelope.build(
        schema_name="leakage_canary_manifest@1.0.0",
        module_id="robustness",
        body={
            "canary_id": canary_id,
            "owner_prompt_id": _OWNER_PROMPT_ID,
            "scope_id": _SCOPE_ID,
            "canary_type": canary_type,
            "horizon": default["horizon"],
            "expected_block_stage": default["expected_block_stage"],
            "expected_reason_code": default["expected_reason_code"],
        },
        catalog=catalog,
    )


def build_leakage_canary_result_manifest(
    catalog: ContractCatalog,
    *,
    canary_result_id: str,
    canary_ref: TypedRef,
    canary_type: str,
    observed_terminal_state: str,
    observed_block_stage: str,
    observed_reason_code: str | None,
    stage_evidence_ref: Mapping[str, Any] | None = None,
    downstream_evidence_refs: Sequence[Mapping[str, Any]] = (),
) -> ManifestEnvelope:
    expected = _CANARY_DEFAULTS[canary_type]
    blocked_state = observed_terminal_state in {
        "blocked_at_expected_stage",
        "blocked_earlier_than_expected",
    }
    failure_reason_codes: list[str] = []
    if blocked_state:
        expected_rank = _CANARY_STAGE_ORDER[expected["expected_block_stage"]]
        observed_rank = _CANARY_STAGE_ORDER[observed_block_stage]
        if observed_terminal_state == "blocked_at_expected_stage":
            if observed_block_stage != expected["expected_block_stage"]:
                failure_reason_codes.append("wrong_block_stage")
        elif observed_rank >= expected_rank:
            failure_reason_codes.append("wrong_block_stage")
    elif observed_terminal_state.startswith("survived_to_"):
        failure_reason_codes.append("unexpected_canary_survival")
    if observed_reason_code != expected["expected_reason_code"]:
        failure_reason_codes.append("wrong_block_reason_code")
    if blocked_state and stage_evidence_ref is None:
        failure_reason_codes.append("missing_stage_evidence")
    if downstream_evidence_refs:
        failure_reason_codes.append("downstream_artifact_created")

    pass_value = not failure_reason_codes
    body: dict[str, Any] = {
        "canary_result_id": canary_result_id,
        "owner_prompt_id": _OWNER_PROMPT_ID,
        "scope_id": _SCOPE_ID,
        "canary_ref": canary_ref.as_dict(),
        "observed_terminal_state": observed_terminal_state,
        "observed_block_stage": observed_block_stage,
        "observed_reason_code": observed_reason_code,
        "downstream_artifacts_created": bool(downstream_evidence_refs),
        "downstream_evidence_refs": [
            _flatten_evidence_ref(dict(item)) for item in downstream_evidence_refs
        ],
        "pass": pass_value,
        "failure_reason_codes": failure_reason_codes,
        "canary_type": canary_type,
    }
    if stage_evidence_ref is not None:
        body["stage_evidence_ref"] = _flatten_evidence_ref(dict(stage_evidence_ref))
    return ManifestEnvelope.build(
        schema_name="leakage_canary_result_manifest@1.0.0",
        module_id="robustness",
        body=body,
        catalog=catalog,
    )


def build_null_result_manifest(
    catalog: ContractCatalog,
    *,
    null_result_id: str,
    null_protocol_ref: TypedRef | None,
    candidate_id: str | None,
    evaluation: NullComparisonEvaluation,
) -> NullResultManifest:
    return NullResultManifest(
        object_id=null_result_id,
        null_result_id=null_result_id,
        null_protocol_ref=null_protocol_ref,
        candidate_id=candidate_id,
        status=evaluation.status,
        failure_reason_code=evaluation.failure_reason_code,
        observed_statistic=evaluation.observed_statistic,
        surrogate_statistics=evaluation.surrogate_statistics,
        monte_carlo_p_value=evaluation.monte_carlo_p_value,
        max_p_value=evaluation.max_p_value,
        resample_count=evaluation.resample_count,
        statistic_id=evaluation.statistic_id,
        statistic_orientation=evaluation.statistic_orientation,
    )


def build_perturbation_family_result_manifest(
    catalog: ContractCatalog,
    *,
    perturbation_family_result_id: str,
    perturbation_protocol_ref: TypedRef,
    candidate_id: str | None,
    evaluation: PerturbationFamilyEvaluation,
) -> PerturbationFamilyResultManifest:
    _ = catalog
    return PerturbationFamilyResultManifest(
        object_id=perturbation_family_result_id,
        perturbation_family_result_id=perturbation_family_result_id,
        perturbation_protocol_ref=perturbation_protocol_ref,
        candidate_id=candidate_id,
        family_id=evaluation.family_id,
        status=evaluation.status,
        valid_run_count=evaluation.valid_run_count,
        invalid_run_count=evaluation.invalid_run_count,
        metric_results=tuple(item.as_dict() for item in evaluation.metric_results),
        perturbation_runs=evaluation.perturbation_runs,
    )


def build_sensitivity_analysis_manifest(
    catalog: ContractCatalog,
    *,
    sensitivity_analysis_id: str,
    perturbation_family_result_ref: TypedRef,
    candidate_id: str | None,
    analysis: Mapping[str, Any],
) -> SensitivityAnalysisManifest:
    _ = catalog
    return SensitivityAnalysisManifest(
        object_id=sensitivity_analysis_id,
        sensitivity_analysis_id=sensitivity_analysis_id,
        perturbation_family_result_ref=perturbation_family_result_ref,
        candidate_id=candidate_id,
        family_id=str(analysis.get("family_id", "")),
        perturbation_id=(
            str(analysis["perturbation_id"])
            if analysis.get("perturbation_id") is not None
            else None
        ),
        canonical_form_matches=(
            bool(analysis["canonical_form_matches"])
            if analysis.get("canonical_form_matches") is not None
            else None
        ),
        description_gain_bits=(
            float(analysis["description_gain_bits"])
            if analysis.get("description_gain_bits") is not None
            else None
        ),
        outer_candidate_score=(
            float(analysis["outer_candidate_score"])
            if analysis.get("outer_candidate_score") is not None
            else None
        ),
        outer_baseline_score=(
            float(analysis["outer_baseline_score"])
            if analysis.get("outer_baseline_score") is not None
            else None
        ),
        failure_reason_code=(
            str(analysis["failure_reason_code"])
            if analysis.get("failure_reason_code") is not None
            else None
        ),
        metadata=dict(analysis.get("metadata", {})),
    )


def evaluate_null_comparison(
    *,
    observed_statistic: float | None,
    surrogate_statistics: Sequence[float],
    max_p_value: float,
    failure_reason_code: str | None = None,
    requested: bool = True,
) -> NullComparisonEvaluation:
    if not requested:
        return NullComparisonEvaluation(
            status="not_requested",
            failure_reason_code=None,
            observed_statistic=None,
            surrogate_statistics=(),
            monte_carlo_p_value=None,
            max_p_value=None,
            resample_count=0,
        )
    if failure_reason_code in _NULL_ABSTENTION_CODES:
        return NullComparisonEvaluation(
            status="abstained",
            failure_reason_code=failure_reason_code,
            observed_statistic=observed_statistic,
            surrogate_statistics=tuple(float(item) for item in surrogate_statistics),
            monte_carlo_p_value=None,
            max_p_value=max_p_value,
            resample_count=len(tuple(surrogate_statistics)),
        )
    if failure_reason_code in _NULL_FAILURE_CODES:
        return NullComparisonEvaluation(
            status="failed",
            failure_reason_code=failure_reason_code,
            observed_statistic=observed_statistic,
            surrogate_statistics=tuple(float(item) for item in surrogate_statistics),
            monte_carlo_p_value=None,
            max_p_value=max_p_value,
            resample_count=len(tuple(surrogate_statistics)),
        )
    observed = float(observed_statistic) if observed_statistic is not None else 0.0
    surrogate_tuple = tuple(float(item) for item in surrogate_statistics)
    exceed_count = sum(1 for item in surrogate_tuple if item >= observed)
    monte_carlo_p_value = (1 + exceed_count) / (len(surrogate_tuple) + 1)
    status = (
        "rejected"
        if monte_carlo_p_value <= float(max_p_value)
        else "null_not_rejected"
    )
    return NullComparisonEvaluation(
        status=status,
        failure_reason_code=None,
        observed_statistic=observed,
        surrogate_statistics=surrogate_tuple,
        monte_carlo_p_value=monte_carlo_p_value,
        max_p_value=float(max_p_value),
        resample_count=len(surrogate_tuple),
    )


def evaluate_perturbation_family(
    *,
    family_id: str,
    metric_refs_by_id: Mapping[str, TypedRef],
    runs: Sequence[PerturbationRunRecord],
) -> PerturbationFamilyEvaluation:
    valid_runs = tuple(run for run in runs if run.failure_reason_code is None)
    invalid_runs = tuple(run for run in runs if run.failure_reason_code is not None)

    metric_results = (
        _metric_result_canonical_form(metric_refs_by_id, valid_runs),
        _metric_result_description_gain(metric_refs_by_id, valid_runs),
        _metric_result_outer_baseline(metric_refs_by_id, valid_runs),
    )
    if any(
        item.failure_reason_code == "nonfinite_metric_value" for item in metric_results
    ):
        status = "failed"
    elif not valid_runs or any(
        item.applicability_status == "abstained" for item in metric_results
    ):
        status = "abstained"
    else:
        status = "completed"
    return PerturbationFamilyEvaluation(
        family_id=family_id,
        status=status,
        valid_run_count=len(valid_runs),
        invalid_run_count=len(invalid_runs),
        metric_results=metric_results,
        perturbation_runs=tuple(run.as_dict() for run in runs),
    )


def evaluate_aggregate_metric_results(
    *,
    family_results: Sequence[PerturbationFamilyEvaluation],
    required_metric_refs: Sequence[TypedRef],
    metric_thresholds: Mapping[str, float],
) -> tuple[tuple[AggregateMetricEvaluation, ...], str]:
    aggregates: list[AggregateMetricEvaluation] = []
    for metric_ref in required_metric_refs:
        values: list[tuple[float, str]] = []
        metric_id = metric_ref.object_id
        for family_result in family_results:
            for metric_result in family_result.metric_results:
                if metric_result.metric_ref != metric_ref:
                    continue
                metric_id = metric_result.metric_id
                if (
                    metric_result.applicability_status == "applicable"
                    and metric_result.value is not None
                ):
                    values.append((metric_result.value, family_result.family_id))
        if values:
            aggregates.append(
                AggregateMetricEvaluation(
                    metric_ref=metric_ref,
                    metric_id=metric_id,
                    status="computed",
                    value=min(item[0] for item in values),
                    failure_reason_code=None,
                    contributing_family_ids=tuple(
                        family_id
                        for _, family_id in sorted(values, key=lambda item: item[1])
                    ),
                )
            )
            continue
        aggregates.append(
            AggregateMetricEvaluation(
                metric_ref=metric_ref,
                metric_id=metric_id,
                status="abstained",
                value=None,
                failure_reason_code="required_metric_has_no_applicable_family",
            )
        )

    stability_status = "passed"
    if any(item.status == "failed" for item in family_results):
        stability_status = "failed"
    elif any(
        item.status == "computed"
        and item.value is not None
        and item.value < metric_thresholds[item.metric_ref.object_id]
        for item in aggregates
    ):
        stability_status = "failed"
    elif any(item.status == "abstained" for item in family_results) or any(
        item.status == "abstained" for item in aggregates
    ):
        stability_status = "abstained"

    return tuple(aggregates), stability_status


def build_robustness_report(
    *,
    candidate_id: str,
    null_protocol_ref: TypedRef | None,
    null_result: NullComparisonEvaluation,
    perturbation_protocol_ref: TypedRef,
    perturbation_family_results: Sequence[PerturbationFamilyEvaluation],
    aggregate_metric_results: Sequence[AggregateMetricEvaluation],
    stability_status: str,
    leakage_canary_result_refs: Sequence[TypedRef],
    leakage_canary_results: Sequence[Mapping[str, Any]],
    candidate_context: Mapping[str, Any] | None = None,
    null_result_ref: TypedRef | None = None,
    perturbation_family_result_refs: Sequence[TypedRef] = (),
    sensitivity_analysis_refs: Sequence[TypedRef] = (),
    sensitivity_analyses: Sequence[Mapping[str, Any]] = (),
    report_id: str = "prototype_robustness_report_v1",
    parent_refs: Sequence[TypedRef] = (),
) -> RobustnessReportManifest:
    canary_types = [str(item["canary_type"]) for item in leakage_canary_results]
    canary_coverage_complete = tuple(sorted(canary_types)) == tuple(
        sorted(_REQUIRED_CANARY_TYPE_COVERAGE)
    )
    canaries_pass = all(bool(item.get("pass")) for item in leakage_canary_results)
    final_status = _resolve_final_robustness_status(
        null_status=null_result.status,
        stability_status=stability_status,
        canary_coverage_complete=canary_coverage_complete,
        canaries_pass=canaries_pass,
    )
    return RobustnessReportManifest(
        robustness_report_id=report_id,
        null_protocol_ref=null_protocol_ref,
        null_result_ref=null_result_ref,
        perturbation_protocol_ref=perturbation_protocol_ref,
        perturbation_family_result_refs=tuple(perturbation_family_result_refs),
        leakage_canary_result_refs=tuple(leakage_canary_result_refs),
        status=final_status,
        candidate_id=candidate_id,
        null_result=(
            None if null_result_ref is not None else null_result.as_dict()
        ),
        perturbation_family_results=(
            ()
            if perturbation_family_result_refs
            else tuple(item.as_dict() for item in perturbation_family_results)
        ),
        aggregate_metric_results=tuple(
            item.as_dict() for item in aggregate_metric_results
        ),
        stability_status=stability_status,
        required_canary_type_coverage=_REQUIRED_CANARY_TYPE_COVERAGE,
        final_robustness_status=final_status,
        candidate_context=dict(candidate_context or {}),
        sensitivity_analysis_refs=tuple(sensitivity_analysis_refs),
        sensitivity_analyses=(
            ()
            if sensitivity_analysis_refs
            else tuple(dict(item) for item in sensitivity_analyses)
        ),
        parent_refs=tuple(parent_refs),
    )


def _metric_result_canonical_form(
    metric_refs_by_id: Mapping[str, TypedRef],
    runs: Sequence[PerturbationRunRecord],
) -> PerturbationMetricEvaluation:
    metric_ref = metric_refs_by_id["canonical_form_exact_match_rate"]
    if not runs:
        return PerturbationMetricEvaluation(
            metric_ref=metric_ref,
            metric_id="canonical_form_exact_match_rate",
            applicability_status="abstained",
            value=None,
            failure_reason_code="no_valid_perturbation_runs",
        )
    value = sum(1 for run in runs if run.canonical_form_matches) / len(runs)
    return PerturbationMetricEvaluation(
        metric_ref=metric_ref,
        metric_id="canonical_form_exact_match_rate",
        applicability_status="applicable",
        value=_normalize_rate(value),
        failure_reason_code=None,
    )


def _metric_result_description_gain(
    metric_refs_by_id: Mapping[str, TypedRef],
    runs: Sequence[PerturbationRunRecord],
) -> PerturbationMetricEvaluation:
    metric_ref = metric_refs_by_id["positive_description_gain_rate"]
    if not runs:
        return PerturbationMetricEvaluation(
            metric_ref=metric_ref,
            metric_id="positive_description_gain_rate",
            applicability_status="abstained",
            value=None,
            failure_reason_code="no_valid_perturbation_runs",
        )
    if any(run.description_gain_bits is None for run in runs):
        return PerturbationMetricEvaluation(
            metric_ref=metric_ref,
            metric_id="positive_description_gain_rate",
            applicability_status="abstained",
            value=None,
            failure_reason_code="metric_prerequisite_missing",
        )
    positive_count = sum(
        1
        for run in runs
        if run.description_gain_bits is not None and run.description_gain_bits > 0
    )
    value = positive_count / len(runs)
    return PerturbationMetricEvaluation(
        metric_ref=metric_ref,
        metric_id="positive_description_gain_rate",
        applicability_status="applicable",
        value=_normalize_rate(value),
        failure_reason_code=None,
    )


def _metric_result_outer_baseline(
    metric_refs_by_id: Mapping[str, TypedRef],
    runs: Sequence[PerturbationRunRecord],
) -> PerturbationMetricEvaluation:
    metric_ref = metric_refs_by_id["outer_baseline_win_rate"]
    if not runs:
        return PerturbationMetricEvaluation(
            metric_ref=metric_ref,
            metric_id="outer_baseline_win_rate",
            applicability_status="abstained",
            value=None,
            failure_reason_code="no_valid_perturbation_runs",
        )
    if any(
        run.outer_candidate_score is None or run.outer_baseline_score is None
        for run in runs
    ):
        return PerturbationMetricEvaluation(
            metric_ref=metric_ref,
            metric_id="outer_baseline_win_rate",
            applicability_status="abstained",
            value=None,
            failure_reason_code="metric_prerequisite_missing",
        )
    win_count = sum(
        1
        for run in runs
        if run.outer_candidate_score is not None
        and run.outer_baseline_score is not None
        and run.outer_candidate_score < run.outer_baseline_score
    )
    value = win_count / len(runs)
    return PerturbationMetricEvaluation(
        metric_ref=metric_ref,
        metric_id="outer_baseline_win_rate",
        applicability_status="applicable",
        value=_normalize_rate(value),
        failure_reason_code=None,
    )


def _resolve_final_robustness_status(
    *,
    null_status: str,
    stability_status: str,
    canary_coverage_complete: bool,
    canaries_pass: bool,
) -> str:
    if (
        null_status in {"rejected", "null_not_rejected", "not_requested"}
        and stability_status == "passed"
        and canary_coverage_complete
        and canaries_pass
    ):
        return "passed"
    if (
        null_status != "failed"
        and stability_status != "failed"
        and canary_coverage_complete
        and canaries_pass
        and (null_status == "abstained" or stability_status == "abstained")
    ):
        return "abstained"
    return "failed"


def _normalize_rate(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("retained robustness rates must be finite")
    return max(0.0, min(1.0, float(value)))


def _format_rate(value: float) -> str:
    normalized = _normalize_rate(value)
    return f"{normalized:g}"


def _flatten_evidence_ref(payload: Mapping[str, Any]) -> dict[str, Any]:
    flattened = dict(payload)
    typed_ref = flattened.pop("typed_ref", None)
    if isinstance(typed_ref, Mapping):
        flattened["schema_name"] = typed_ref["schema_name"]
        flattened["object_id"] = typed_ref["object_id"]
    return flattened
