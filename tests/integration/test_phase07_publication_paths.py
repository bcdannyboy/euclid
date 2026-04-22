from __future__ import annotations

from pathlib import Path

import euclid


def test_candidate_publication_path_supports_catalog_and_replay_surfaces(
    phase07_candidate_demo_output_root: Path,
) -> None:
    graph = euclid.load_demo_run_artifact_graph(
        output_root=phase07_candidate_demo_output_root
    )

    assert graph.inspect(graph.root_ref).manifest.body["result_mode"] == (
        "candidate_publication"
    )

    replay = euclid.replay_demo(output_root=phase07_candidate_demo_output_root)
    assert replay.summary.replay_verification_status == "verified"

    published = euclid.publish_demo_run_to_catalog(
        output_root=phase07_candidate_demo_output_root
    )
    assert published.publication_mode == "candidate_publication"
    assert published.comparator_exposure_status == "satisfied"
    assert published.claim_card_ref is not None
    assert published.abstention_ref is None

    inspection = euclid.inspect_demo_catalog_entry(
        output_root=phase07_candidate_demo_output_root,
        publication_id=published.publication_id,
    )
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == "descriptive_structure"
    assert inspection.abstention is None
    assert inspection.replay_bundle.replay_verification_status == "verified"
