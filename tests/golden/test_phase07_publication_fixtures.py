from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import euclid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "fixtures/runtime/phase07"


def _ref_string(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    schema_name = payload.get("schema_name")
    object_id = payload.get("object_id")
    if isinstance(schema_name, str) and isinstance(object_id, str):
        return f"{schema_name}:{object_id}"
    return None


def _phase07_snapshot(output_root: Path) -> dict[str, Any]:
    graph = euclid.load_demo_run_artifact_graph(output_root=output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    comparison_ref_payload = run_result.body["comparison_universe_ref"]
    comparison = graph.inspect(
        euclid.TypedRef(
            schema_name=str(comparison_ref_payload["schema_name"]),
            object_id=str(comparison_ref_payload["object_id"]),
        )
    ).manifest

    replay = euclid.replay_demo(output_root=output_root)
    published = euclid.publish_demo_run_to_catalog(output_root=output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=output_root,
        publication_id=published.publication_id,
    )

    return {
        "run_result_ref": _ref_string(run_result.ref.as_dict()),
        "result_mode": str(run_result.body["result_mode"]),
        "claim_type": (
            str(inspection.claim_card.manifest.body["claim_type"])
            if inspection.claim_card is not None
            else None
        ),
        "abstention_type": (
            str(inspection.abstention.manifest.body["abstention_type"])
            if inspection.abstention is not None
            else None
        ),
        "published_entry": {
            "publication_mode": inspection.entry.publication_mode,
            "comparator_exposure_status": (
                inspection.entry.comparator_exposure_status
            ),
            "claim_card_ref": (
                _ref_string(inspection.entry.claim_card_ref.as_dict())
                if inspection.entry.claim_card_ref is not None
                else None
            ),
            "abstention_ref": (
                _ref_string(inspection.entry.abstention_ref.as_dict())
                if inspection.entry.abstention_ref is not None
                else None
            ),
        },
        "comparison_universe": {
            "comparison_class_status": str(
                comparison.body["comparison_class_status"]
            ),
            "candidate_beats_baseline": bool(comparison.body["candidate_beats_baseline"]),
            "candidate_score_result_ref": _ref_string(
                comparison.body.get("candidate_score_result_ref")
            ),
            "baseline_score_result_ref": _ref_string(
                comparison.body.get("baseline_score_result_ref")
            ),
            "comparator_score_result_refs": [
                ref
                for ref in (
                    _ref_string(item)
                    for item in comparison.body.get("comparator_score_result_refs", ())
                )
                if ref is not None
            ],
            "paired_comparison_record_count": len(
                comparison.body.get("paired_comparison_records", ())
            ),
        },
        "replay": {
            "bundle_ref": _ref_string(replay.summary.bundle_ref.as_dict()),
            "replay_verification_status": replay.summary.replay_verification_status,
            "required_manifest_refs": [
                f"{ref.schema_name}:{ref.object_id}"
                for ref in inspection.replay_bundle.required_manifest_refs
            ],
            "stage_order": list(inspection.replay_bundle.recorded_stage_order),
        },
        "run_result_refs": {
            "candidate": _ref_string(run_result.body.get("primary_reducer_artifact_ref")),
            "scorecard": _ref_string(run_result.body.get("primary_scorecard_ref")),
            "claim_card": _ref_string(run_result.body.get("primary_claim_card_ref")),
            "abstention": _ref_string(run_result.body.get("primary_abstention_ref")),
        },
    }


def _expected_fixture(name: str) -> dict[str, Any]:
    path = FIXTURE_ROOT / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_candidate_publication_matches_golden_fixture(
    phase07_candidate_demo_output_root: Path,
) -> None:
    assert _phase07_snapshot(phase07_candidate_demo_output_root) == _expected_fixture(
        "candidate-publication-golden.json"
    )


def test_abstention_publication_matches_golden_fixture(
    phase01_demo_output_root: Path,
) -> None:
    assert _phase07_snapshot(phase01_demo_output_root) == _expected_fixture(
        "abstention-publication-golden.json"
    )
