from __future__ import annotations

from pathlib import Path

import euclid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISTRIBUTION_MANIFEST = (
    PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-distribution-demo.yaml"
)


def test_probabilistic_pipeline_emits_run_result_bundle_and_publication_chain(
    tmp_path: Path,
) -> None:
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=DISTRIBUTION_MANIFEST,
        output_root=tmp_path / "probabilistic-output",
    )

    assert result.summary.run_result_ref.schema_name == "run_result_manifest@1.1.0"
    assert (
        result.summary.bundle_ref.schema_name == "reproducibility_bundle_manifest@1.0.0"
    )
    assert (
        result.summary.publication_record_ref.schema_name
        == "publication_record_manifest@1.1.0"
    )
    assert (
        result.summary.comparison_universe_ref.schema_name
        == "comparison_universe_manifest@1.0.0"
    )
    assert result.summary.scorecard_ref.schema_name == "scorecard_manifest@1.1.0"
    assert result.summary.claim_card_ref is not None

    graph = euclid.load_demo_run_artifact_graph(
        output_root=result.paths.output_root,
        run_id=result.summary.run_result_ref.object_id,
    )
    root = graph.inspect(graph.root_ref).manifest

    assert graph.root_ref == result.summary.run_result_ref
    assert root.body["result_mode"] == "candidate_publication"
    assert root.body["prediction_artifact_refs"] == [
        result.summary.prediction_artifact_ref.as_dict()
    ]
    assert (
        root.body["primary_score_result_ref"]
        == result.summary.score_result_ref.as_dict()
    )
    assert root.body["primary_calibration_result_ref"] == (
        result.summary.calibration_result_ref.as_dict()
    )
    assert root.body["forecast_object_type"] == "distribution"

    replay_bundle = euclid.inspect_demo_replay_bundle(
        output_root=result.paths.output_root
    )
    recorded_roles = {
        record.artifact_role for record in replay_bundle.artifact_hash_records
    }
    assert {
        "prediction_artifact",
        "score_result",
        "calibration_result",
    } <= recorded_roles
