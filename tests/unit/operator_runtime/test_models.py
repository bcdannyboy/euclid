from __future__ import annotations

from pathlib import Path

from euclid.contracts.refs import TypedRef
from euclid.operator_runtime.models import (
    ADMITTED_FORECAST_OBJECT_TYPES,
    OperatorPaths,
    OperatorReplayResult,
    OperatorReplaySummary,
    OperatorRequest,
    OperatorRunResult,
    OperatorRunSummary,
)


def test_operator_request_roundtrip() -> None:
    request = OperatorRequest(
        request_id="current-release-run",
        manifest_path=Path("/tmp/current_release_run.yaml"),
        dataset_csv=Path("/tmp/minimal_dataset.csv"),
        cutoff_available_at=None,
        quantization_step="0.5",
        minimum_description_gain_bits=0.0,
        search_family_ids=("constant", "drift"),
        search_class="bounded_heuristic",
        search_seed="0",
        forecast_object_type="point",
        external_evidence_payload={
            "bundle_id": "external_bundle",
            "domain_id": "macro",
            "acquisition_window": {
                "start": "2026-02-01T00:00:00Z",
                "end": "2026-02-04T00:00:00Z",
            },
            "raw_sources": (),
        },
        declared_entity_panel=("entity-a", "entity-b"),
    )

    restored = OperatorRequest.from_dict(request.as_dict())

    assert restored == request
    assert restored.extension_lane_ids == ("external_evidence",)


def test_operator_result_roundtrip() -> None:
    result = OperatorRunResult(
        request=OperatorRequest(
            request_id="full-vision-run",
            manifest_path=Path("/tmp/full_vision_run.yaml"),
            dataset_csv=Path("/tmp/minimal_dataset.csv"),
            cutoff_available_at=None,
            quantization_step="0.5",
            minimum_description_gain_bits=0.0,
            search_family_ids=("algorithmic_last_observation",),
            search_class="bounded_heuristic",
            search_seed="0",
            forecast_object_type="distribution",
        ),
        paths=OperatorPaths(
            output_root=Path("/tmp/operator"),
            active_run_root=Path("/tmp/operator/active-runs/full-vision-run"),
            sealed_run_root=Path("/tmp/operator/sealed-runs/full-vision-run"),
            artifact_root=Path("/tmp/operator/sealed-runs/full-vision-run/artifacts"),
            metadata_store_path=Path("/tmp/operator/sealed-runs/full-vision-run/registry.sqlite3"),
            control_plane_store_path=Path("/tmp/operator/active-runs/full-vision-run/control-plane/execution-state.sqlite3"),
            run_summary_path=Path("/tmp/operator/sealed-runs/full-vision-run/run-summary.json"),
            cache_root=Path("/tmp/operator/caches"),
            temp_root=Path("/tmp/operator/tmp"),
            run_lock_path=Path("/tmp/operator/active-runs/full-vision-run/locks/run.lock"),
        ),
        summary=OperatorRunSummary(
            selected_candidate_id="candidate-1",
            selected_family="algorithmic_last_observation",
            forecast_object_type="distribution",
            result_mode="candidate_publication",
            bundle_ref=TypedRef("reproducibility_bundle_manifest@1.0.0", "bundle-1"),
            run_result_ref=TypedRef("run_result_manifest@1.1.0", "run-1"),
            scope_ledger_ref=TypedRef("scope_ledger_manifest@1.0.0", "scope-1"),
            selected_candidate_ref=TypedRef(
                "reducer_artifact_manifest@1.0.0",
                "candidate-1",
            ),
            publication_record_ref=TypedRef(
                "publication_record_manifest@1.0.0",
                "pub-1",
            ),
            comparison_universe_ref=TypedRef(
                "comparison_universe_manifest@1.0.0",
                "comp-1",
            ),
            scorecard_ref=TypedRef("scorecard_manifest@1.1.0", "scorecard-1"),
            claim_card_ref=TypedRef("claim_card_manifest@1.1.0", "claim-1"),
            abstention_ref=None,
            prediction_artifact_ref=TypedRef(
                "prediction_artifact_manifest@1.0.0",
                "prediction-1",
            ),
            primary_score_result_ref=TypedRef(
                "probabilistic_score_result_manifest@1.0.0",
                "score-1",
            ),
            primary_calibration_result_ref=TypedRef(
                "calibration_result_manifest@1.0.0",
                "calibration-1",
            ),
            confirmatory_primary_score=0.25,
            calibration_status="passed",
            extension_lane_ids=("algorithmic_last_observation", "distribution"),
        ),
    )

    restored = OperatorRunResult.from_dict(result.as_dict())

    assert restored == result

    replay = OperatorReplayResult(
        paths=result.paths,
        summary=OperatorReplaySummary(
            bundle_ref=result.summary.bundle_ref,
            run_result_ref=result.summary.run_result_ref,
            selected_candidate_ref=result.summary.selected_candidate_ref,
            selected_family=result.summary.selected_family,
            result_mode=result.summary.result_mode,
            forecast_object_type=result.summary.forecast_object_type,
            replay_verification_status="verified",
            confirmatory_primary_score=result.summary.confirmatory_primary_score,
        ),
    )
    assert OperatorReplayResult.from_dict(replay.as_dict()) == replay


def test_operator_models_support_all_forecast_object_types() -> None:
    for forecast_object_type in ADMITTED_FORECAST_OBJECT_TYPES:
        request = OperatorRequest(
            request_id=f"{forecast_object_type}-run",
            manifest_path=Path(f"/tmp/{forecast_object_type}.yaml"),
            dataset_csv=Path("/tmp/minimal_dataset.csv"),
            cutoff_available_at=None,
            quantization_step="0.5",
            minimum_description_gain_bits=0.0,
            search_family_ids=(
                ("constant",)
                if forecast_object_type == "point"
                else ("algorithmic_last_observation",)
            ),
            search_class="bounded_heuristic",
            search_seed="0",
            forecast_object_type=forecast_object_type,
        )

        assert request.forecast_object_type == forecast_object_type
        if forecast_object_type == "point":
            assert request.extension_lane_ids == ()
        else:
            assert forecast_object_type in request.extension_lane_ids
