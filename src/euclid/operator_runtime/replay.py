from __future__ import annotations

from pathlib import Path

from euclid.operator_runtime._compat_runtime import replay_operator_bundle
from euclid.operator_runtime.models import (
    OperatorPaths,
    OperatorReplayResult,
    OperatorReplaySummary,
)
from euclid.operator_runtime.resources import default_run_output_root


def _format_typed_ref(ref) -> str:
    return f"{ref.schema_name}:{ref.object_id}"


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


def replay_operator(
    *,
    output_root: Path | None = None,
    run_id: str | None = None,
    bundle_ref: str | None = None,
) -> OperatorReplayResult:
    resolved_output_root = output_root
    if resolved_output_root is None:
        resolved_output_root = default_run_output_root(run_id or "operator-run")
    compat_result = replay_operator_bundle(
        output_root=resolved_output_root,
        run_id=run_id,
        bundle_ref=bundle_ref,
    )
    return OperatorReplayResult(
        paths=_operator_paths_from_compat(compat_result.paths),
        summary=OperatorReplaySummary(
            bundle_ref=compat_result.summary.bundle_ref,
            run_result_ref=compat_result.summary.run_result_ref,
            selected_candidate_ref=compat_result.summary.selected_candidate_ref,
            selected_family=compat_result.summary.selected_family,
            result_mode=compat_result.summary.result_mode,
            forecast_object_type=compat_result.summary.forecast_object_type,
            replay_verification_status=compat_result.summary.replay_verification_status,
            confirmatory_primary_score=float(
                compat_result.summary.confirmatory_primary_score
            ),
            failure_reason_codes=compat_result.summary.failure_reason_codes,
        ),
    )


def format_operator_replay_summary(
    *,
    run_id: str,
    result: OperatorReplayResult,
) -> str:
    return "\n".join(
        [
            "Euclid replay",
            f"Run id: {run_id}",
            f"Output root: {result.paths.output_root}",
            f"Run result ref: {_format_typed_ref(result.summary.run_result_ref)}",
            f"Bundle ref: {_format_typed_ref(result.summary.bundle_ref)}",
            f"Forecast object type: {result.summary.forecast_object_type}",
            f"Selected family: {result.summary.selected_family}",
            f"Replay verification: {result.summary.replay_verification_status}",
        ]
    )


__all__ = ["format_operator_replay_summary", "replay_operator"]
