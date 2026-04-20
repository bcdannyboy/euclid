from __future__ import annotations

from pathlib import Path

import yaml

import euclid
from euclid.operator_runtime.run import run_operator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_RELEASE_MANIFEST = PROJECT_ROOT / "examples" / "current_release_run.yaml"
FULL_VISION_MANIFEST = PROJECT_ROOT / "examples" / "full_vision_run.yaml"


def test_severe_robustness_failure_blocks_publication(tmp_path: Path) -> None:
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        request_id="robustness-publication-leakage",
        robustness={
            "forced_leakage_canary_failures": ["future_target_level_feature"],
        },
    )

    result = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "robustness-publication-leakage-output",
    )
    entry = euclid.publish_demo_run_to_catalog(output_root=result.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=result.paths.output_root,
        publication_id=entry.publication_id,
    )

    assert result.summary.result_mode == "abstention_only_publication"
    assert entry.publication_mode == "abstention_only_publication"
    assert inspection.scorecard is not None
    assert inspection.scorecard.manifest.body["descriptive_status"] == (
        "blocked_robustness_failed"
    )
    assert (
        "leakage_canary_failed"
        in inspection.scorecard.manifest.body["descriptive_reason_codes"]
    )
    assert inspection.abstention is not None
    assert inspection.abstention.manifest.body["abstention_type"] == "robustness_failed"


def test_expected_abstention_mode_is_emitted(tmp_path: Path) -> None:
    result = run_operator(
        manifest_path=CURRENT_RELEASE_MANIFEST,
        output_root=tmp_path / "robustness-current-release-output",
    )
    entry = euclid.publish_demo_run_to_catalog(output_root=result.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=result.paths.output_root,
        publication_id=entry.publication_id,
    )

    assert result.summary.result_mode == "abstention_only_publication"
    assert entry.publication_mode == "abstention_only_publication"
    assert inspection.abstention is not None
    assert inspection.abstention.manifest.body["abstention_type"] == "robustness_failed"
    assert inspection.scorecard is not None
    assert (
        "robustness_failed"
        in inspection.scorecard.manifest.body["descriptive_reason_codes"]
    )


def test_nonsevere_perturbation_does_not_overblock(tmp_path: Path) -> None:
    result = run_operator(
        manifest_path=FULL_VISION_MANIFEST,
        output_root=tmp_path / "robustness-nonsevere-output",
    )
    entry = euclid.publish_demo_run_to_catalog(output_root=result.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=result.paths.output_root,
        publication_id=entry.publication_id,
    )
    graph = euclid.load_demo_run_artifact_graph(output_root=result.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    robustness_report = graph.inspect(
        run_result.body["robustness_report_refs"][0]
    ).manifest

    assert entry.publication_mode == "candidate_publication"
    assert inspection.claim_card is not None
    assert robustness_report.body["final_robustness_status"] == "passed"
    assert len(robustness_report.body["perturbation_family_result_refs"]) >= 1


def _write_operator_manifest(
    *,
    tmp_path: Path,
    request_id: str,
    robustness: dict[str, object],
) -> Path:
    payload = yaml.safe_load(FULL_VISION_MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["request_id"] = request_id
    dataset_csv = Path(str(payload["dataset_csv"]))
    if not dataset_csv.is_absolute():
        payload["dataset_csv"] = str(
            (FULL_VISION_MANIFEST.parent / dataset_csv).resolve()
        )
    payload["robustness"] = robustness

    manifest_path = tmp_path / f"{request_id}.yaml"
    manifest_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path
