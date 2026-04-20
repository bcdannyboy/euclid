from __future__ import annotations

from pathlib import Path

from euclid.operator_runtime._compat_runtime import (
    load_operator_request,
    run_operator_point,
    run_operator_probabilistic,
)
from euclid.operator_runtime.models import (
    DEFAULT_ADMISSIBILITY_RULE_IDS,
    DEFAULT_RUN_SUPPORT_OBJECT_IDS,
    OperatorPaths,
    OperatorRequest,
    OperatorRunResult,
    OperatorRunSummary,
    derive_extension_lane_ids,
)


def _operator_request_from_compat(request) -> OperatorRequest:
    return OperatorRequest(
        request_id=request.request_id,
        manifest_path=request.manifest_path,
        dataset_csv=request.dataset_csv,
        cutoff_available_at=request.cutoff_available_at,
        quantization_step=request.quantization_step,
        minimum_description_gain_bits=request.minimum_description_gain_bits,
        min_train_size=request.min_train_size,
        horizon=request.horizon,
        search_family_ids=request.search_family_ids,
        search_class=request.search_class,
        search_seed=request.search_seed,
        proposal_limit=request.proposal_limit,
        seasonal_period=request.seasonal_period,
        forecast_object_type=request.forecast_object_type,
        primary_score_id=request.primary_score_id,
        calibration_thresholds=request.calibration_thresholds,
        external_evidence_payload=request.external_evidence_payload,
        mechanistic_evidence_payload=request.mechanistic_evidence_payload,
        robustness_payload=request.robustness_payload,
        declared_entity_panel=request.declared_entity_panel,
        run_support_object_ids=DEFAULT_RUN_SUPPORT_OBJECT_IDS,
        admissibility_rule_ids=DEFAULT_ADMISSIBILITY_RULE_IDS,
        extension_lane_ids=derive_extension_lane_ids(
            search_family_ids=request.search_family_ids,
            forecast_object_type=request.forecast_object_type,
            external_evidence_enabled=request.external_evidence_payload is not None,
            mechanistic_evidence_enabled=(
                request.mechanistic_evidence_payload is not None
            ),
            robustness_override_enabled=request.robustness_payload is not None,
        ),
    )


def _operator_paths_from_compat(paths) -> OperatorPaths:
    return OperatorPaths(
        output_root=paths.output_root,
        active_run_root=paths.active_run_root,
        sealed_run_root=paths.sealed_run_root,
        artifact_root=paths.artifact_root,
        metadata_store_path=paths.metadata_store_path,
        control_plane_store_path=paths.control_plane_store_path,
        run_summary_path=paths.run_summary_path,
        cache_root=paths.cache_root,
        temp_root=paths.temp_root,
        run_lock_path=paths.run_lock_path,
    )


def _point_summary(result, request: OperatorRequest) -> OperatorRunSummary:
    workflow = result.workflow_result
    return OperatorRunSummary(
        selected_candidate_id=str(workflow.selected_candidate.manifest.body["candidate_id"]),
        selected_family=str(result.summary.selected_family),
        forecast_object_type=request.forecast_object_type,
        result_mode=str(result.summary.result_mode),
        bundle_ref=result.summary.bundle_ref,
        run_result_ref=result.summary.run_result_ref,
        scope_ledger_ref=workflow.scope_ledger.manifest.ref,
        selected_candidate_ref=result.summary.selected_candidate_ref,
        publication_record_ref=workflow.publication_record.manifest.ref,
        comparison_universe_ref=workflow.comparison_universe.manifest.ref,
        scorecard_ref=workflow.scorecard.manifest.ref,
        claim_card_ref=(
            workflow.claim_card.manifest.ref if workflow.claim_card is not None else None
        ),
        abstention_ref=(
            workflow.abstention.manifest.ref if workflow.abstention is not None else None
        ),
        prediction_artifact_ref=workflow.prediction_artifact.manifest.ref,
        primary_score_result_ref=workflow.point_score_result.manifest.ref,
        primary_calibration_result_ref=workflow.calibration_result.manifest.ref,
        confirmatory_primary_score=float(result.summary.confirmatory_primary_score),
        run_support_object_ids=request.run_support_object_ids,
        admissibility_rule_ids=request.admissibility_rule_ids,
        extension_lane_ids=request.extension_lane_ids,
    )


def _probabilistic_summary(result, request: OperatorRequest) -> OperatorRunSummary:
    summary = result.summary
    return OperatorRunSummary(
        selected_candidate_id=str(summary.selected_candidate_id),
        selected_family=str(summary.selected_family),
        forecast_object_type=str(summary.forecast_object_type),
        result_mode=str(summary.result_mode),
        bundle_ref=summary.bundle_ref,
        run_result_ref=summary.run_result_ref,
        scope_ledger_ref=summary.scope_ledger_ref,
        selected_candidate_ref=summary.selected_candidate_ref,
        publication_record_ref=summary.publication_record_ref,
        comparison_universe_ref=summary.comparison_universe_ref,
        scorecard_ref=summary.scorecard_ref,
        claim_card_ref=summary.claim_card_ref,
        abstention_ref=summary.abstention_ref,
        prediction_artifact_ref=summary.prediction_artifact_ref,
        primary_score_result_ref=summary.score_result_ref,
        primary_calibration_result_ref=summary.calibration_result_ref,
        confirmatory_primary_score=float(summary.aggregated_primary_score),
        calibration_status=summary.calibration_status,
        run_support_object_ids=request.run_support_object_ids,
        admissibility_rule_ids=request.admissibility_rule_ids,
        extension_lane_ids=request.extension_lane_ids,
    )


def _format_typed_ref(ref) -> str:
    return f"{ref.schema_name}:{ref.object_id}"


def run_operator(
    *,
    manifest_path: Path,
    output_root: Path | None = None,
) -> OperatorRunResult:
    compat_request = load_operator_request(manifest_path)
    request = _operator_request_from_compat(compat_request)
    if request.forecast_object_type == "point":
        compat_result = run_operator_point(
            manifest_path=manifest_path,
            output_root=output_root,
        )
        return OperatorRunResult(
            request=request,
            paths=_operator_paths_from_compat(compat_result.paths),
            summary=_point_summary(compat_result, request),
        )
    compat_result = run_operator_probabilistic(
        manifest_path=manifest_path,
        output_root=output_root,
    )
    return OperatorRunResult(
        request=request,
        paths=_operator_paths_from_compat(compat_result.paths),
        summary=_probabilistic_summary(compat_result, request),
    )


def format_operator_run_summary(result: OperatorRunResult) -> str:
    lines = [
        "Euclid run",
        f"Request id: {result.request.request_id}",
        f"Config: {result.request.manifest_path}",
        f"Output root: {result.paths.output_root}",
        f"Run result ref: {_format_typed_ref(result.summary.run_result_ref)}",
        f"Bundle ref: {_format_typed_ref(result.summary.bundle_ref)}",
        f"Scope ledger ref: {_format_typed_ref(result.summary.scope_ledger_ref)}",
        f"Forecast object type: {result.summary.forecast_object_type}",
        f"Selected family: {result.summary.selected_family}",
        f"Extension lanes: {', '.join(result.summary.extension_lane_ids) or 'retained_only'}",
        f"Saved summary: {result.paths.run_summary_path}",
    ]
    return "\n".join(lines)


__all__ = ["OperatorRunResult", "format_operator_run_summary", "run_operator"]
