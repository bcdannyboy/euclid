from __future__ import annotations

import importlib.metadata
from pathlib import Path

import euclid
from euclid.contracts.loader import load_contract_catalog
from euclid.manifests.runtime_models import (
    DistributionPredictionRow,
    PredictionArtifactManifest,
)
from euclid.modules import probabilistic_evaluation as pe
from euclid.modules.scoring import score_probabilistic_prediction_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "fixtures/runtime/phase06"
DISTRIBUTION_MANIFEST = FIXTURE_ROOT / "probabilistic-distribution-demo.yaml"
CALIBRATION_FAILURE_MANIFEST = (
    FIXTURE_ROOT / "probabilistic-distribution-calibration-failure-demo.yaml"
)


def test_failed_calibration_blocks_publication(tmp_path: Path) -> None:
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=CALIBRATION_FAILURE_MANIFEST,
        output_root=tmp_path / "distribution-calibration-failure",
    )
    entry = euclid.publish_demo_run_to_catalog(output_root=result.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=result.paths.output_root,
        publication_id=entry.publication_id,
    )

    assert result.calibration.status == "failed"
    assert result.summary.calibration_status == "failed"
    assert entry.publication_mode == "candidate_publication"
    assert inspection.scorecard is not None
    assert inspection.scorecard.manifest.body["predictive_status"] == "blocked"
    assert inspection.scorecard.manifest.body["predictive_reason_codes"] == [
        "calibration_failed"
    ]
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == "descriptive_structure"
    assert inspection.claim_card.manifest.body["predictive_support_status"] == "blocked"
    assert (
        "probabilistic_forecast_within_declared_validation_scope"
        not in inspection.claim_card.manifest.body["allowed_interpretation_codes"]
    )


def test_calibration_success_does_not_override_predictive_gate(
    tmp_path: Path,
) -> None:
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=DISTRIBUTION_MANIFEST,
        output_root=tmp_path / "distribution-calibration-success",
    )
    entry = euclid.publish_demo_run_to_catalog(output_root=result.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=result.paths.output_root,
        publication_id=entry.publication_id,
    )

    assert result.calibration.status == "passed"
    assert result.summary.calibration_status == "passed"
    assert entry.publication_mode == "candidate_publication"
    assert inspection.scorecard is not None
    assert inspection.scorecard.manifest.body["predictive_status"] == "blocked"
    assert inspection.scorecard.manifest.body["predictive_reason_codes"] == [
        "insufficient_paired_count"
    ]
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == "descriptive_structure"
    assert inspection.claim_card.manifest.body["predictive_support_status"] == "blocked"
    assert (
        "probabilistic_forecast_within_declared_validation_scope"
        not in inspection.claim_card.manifest.body["allowed_interpretation_codes"]
    )


def test_score_result_carries_mapie_unavailable_calibration_provenance(
    monkeypatch,
) -> None:
    def missing_version(package_name: str) -> str:
        assert package_name == "mapie"
        raise importlib.metadata.PackageNotFoundError("mapie")

    monkeypatch.setattr(pe.importlib_metadata, "version", missing_version)
    catalog = load_contract_catalog(PROJECT_ROOT)
    score_policy = _distribution_score_policy(catalog)
    calibration_method = pe.build_mapie_calibration_method_metadata(
        method_id="enbpi_time_series_v1",
        guarantee_tier="approximate_mixing_time_series",
        assumption_ids=("weak_dependence_or_mixing",),
        assumptions={"weak_dependence_or_mixing": "rolling residual split declared"},
        assumption_scope="mixing_time_series",
        calibration_partition_ids=("partition-h1",),
        horizon_ids=(1,),
        calibration_indices=(4, 5, 6, 7),
    )
    artifact = _distribution_prediction_artifact(
        catalog=catalog,
        score_policy=score_policy,
        calibration_method=calibration_method,
    )

    result = score_probabilistic_prediction_artifact(
        catalog=catalog,
        score_policy_manifest=score_policy,
        prediction_artifact_manifest=artifact,
    )

    assert result.body["comparison_status"] == "comparable"
    assert (
        result.body["effective_probabilistic_config"]["calibration_method"]
        == calibration_method
    )
    assert result.body["effective_probabilistic_config"]["calibration_method"][
        "backend"
    ]["reason_codes"] == ["calibration_backend_unavailable"]


def _distribution_score_policy(catalog):
    return pe.ManifestEnvelope.build(
        schema_name="probabilistic_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": "integration_distribution_policy_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": "distribution",
            "primary_score": "continuous_ranked_probability_score",
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [{"horizon": 1, "weight": "1.0"}],
            "entity_aggregation_mode": (
                "single_entity_only_no_cross_entity_aggregation"
            ),
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )


def _distribution_prediction_artifact(
    *,
    catalog,
    score_policy,
    calibration_method: dict[str, object],
):
    row = DistributionPredictionRow(
        origin_time="2026-01-01T00:00:00Z",
        available_at="2026-01-02T00:00:00Z",
        horizon=1,
        distribution_family="gaussian_location_scale",
        location=10.0,
        scale=1.0,
        support_kind="all_real",
        realized_observation=10.0,
    )
    return PredictionArtifactManifest(
        prediction_artifact_id="integration_distribution_prediction",
        candidate_id="integration_distribution_candidate",
        stage_id="confirmatory_holdout",
        fit_window_id="fit_window",
        test_window_id="confirmatory_segment",
        model_freeze_status="global_finalist_frozen",
        refit_rule_applied="pre_holdout_development_refit",
        score_policy_ref=score_policy.ref,
        rows=(row,),
        forecast_object_type="distribution",
        score_law_id="continuous_ranked_probability_score",
        horizon_weights=({"horizon": 1, "weight": "1.0"},),
        scored_origin_panel=(
            {
                "scored_origin_id": "integration_origin_0",
                "origin_time": row.origin_time,
                "available_at": row.available_at,
                "horizon": row.horizon,
            },
        ),
        scored_origin_set_id="integration_distribution_panel",
        comparison_key={
            "forecast_object_type": "distribution",
            "horizon_set": [1],
            "score_law_id": "continuous_ranked_probability_score",
            "scored_origin_set_id": "integration_distribution_panel",
        },
        missing_scored_origins=(),
        timeguard_checks=(
            {
                "scored_origin_id": "integration_origin_0",
                "expected_available_at": row.available_at,
                "observed_available_at": row.available_at,
                "status": "passed",
            },
        ),
        effective_probabilistic_config={
            "distribution_family": "gaussian_location_scale",
            "calibration_method": calibration_method,
        },
    ).to_manifest(catalog)
