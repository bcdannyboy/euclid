from __future__ import annotations

from euclid.modules.predictive_tests import build_prequential_score_stream


def test_prequential_stream_records_origin_horizon_entity_and_rolling_degradation() -> None:
    stream = build_prequential_score_stream(
        stream_id="candidate_vs_baseline",
        candidate_id="candidate",
        baseline_id="baseline",
        rows=(
            {
                "origin_id": "o1",
                "horizon": 1,
                "entity": "a",
                "regime": "calm",
                "candidate_loss": 0.5,
                "baseline_loss": 1.0,
            },
            {
                "origin_id": "o2",
                "horizon": 1,
                "entity": "a",
                "regime": "volatile",
                "candidate_loss": 1.5,
                "baseline_loss": 1.0,
            },
        ),
        rolling_window=2,
    )

    manifest = stream.as_manifest()

    assert manifest["schema_name"] == "prequential_score_stream@1.0.0"
    assert manifest["per_origin"][0]["loss_difference"] == 0.5
    assert manifest["per_horizon"][0]["horizon"] == 1
    assert manifest["per_entity"][0]["entity"] == "a"
    assert manifest["per_regime"][0]["regime"] == "calm"
    assert manifest["rolling_degradation"][0]["status"] == "degraded"
    assert stream.replay_identity.startswith("prequential-stream:")
