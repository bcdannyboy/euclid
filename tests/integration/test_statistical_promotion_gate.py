from __future__ import annotations

from euclid.modules.predictive_tests import evaluate_predictive_promotion


def test_statistical_promotion_gate_uses_statsmodels_uncertainty_evidence() -> None:
    candidate_losses = tuple(0.8 + (index % 5) * 0.01 for index in range(60))
    baseline_losses = tuple(
        loss + 0.4 + (0.02 if index % 2 else -0.01)
        for index, loss in enumerate(candidate_losses)
    )

    result = evaluate_predictive_promotion(
        candidate_losses=candidate_losses,
        baseline_losses=baseline_losses,
        split_protocol_id="declared_walk_forward",
        baseline_id="naive",
        practical_margin=0.1,
        paired_stream_identity={
            "candidate_id": "candidate",
            "candidate_prediction_artifact_ref": {
                "schema_name": "prediction_artifact_manifest@1.1.0",
                "object_id": "candidate",
            },
            "baseline_prediction_artifact_ref": {
                "schema_name": "prediction_artifact_manifest@1.1.0",
                "object_id": "baseline",
            },
            "score_policy_ref": {
                "schema_name": "score_policy_manifest@1.0.0",
                "object_id": "absolute-error",
            },
            "horizon_set": [1],
            "scored_origin_set_id": "declared-walk-forward",
        },
    )

    manifest = result.as_manifest()

    assert result.promotion_allowed is True
    assert manifest["statistical_test_backend"] == "diebold_mariano_hln_v1"
    assert manifest["confidence_interval_method"] == "dm_hln_hac_t_interval"
    assert manifest["raw_pair_count"] == 60
    assert manifest["effective_sample_size"] == 60.0
