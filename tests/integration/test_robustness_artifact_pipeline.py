from __future__ import annotations

from pathlib import Path

import yaml

import euclid
from euclid.operator_runtime.replay import replay_operator
from euclid.operator_runtime.run import run_operator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_RELEASE_MANIFEST = PROJECT_ROOT / "examples" / "current_release_run.yaml"
FULL_VISION_MANIFEST = PROJECT_ROOT / "examples" / "full_vision_run.yaml"


def test_operator_run_emits_full_robustness_family(tmp_path: Path) -> None:
    result = run_operator(
        manifest_path=CURRENT_RELEASE_MANIFEST,
        output_root=tmp_path / "robustness-family-output",
    )
    graph = euclid.load_demo_run_artifact_graph(output_root=result.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    robustness_report = graph.inspect(
        run_result.body["robustness_report_refs"][0]
    ).manifest

    assert "null_result_ref" in robustness_report.body
    assert len(robustness_report.body["perturbation_family_result_refs"]) >= 2
    assert len(robustness_report.body["leakage_canary_result_refs"]) == 4
    assert len(robustness_report.body["sensitivity_analysis_refs"]) >= 1
    assert run_result.body["robustness_report_refs"] == [
        robustness_report.ref.as_dict()
    ]


def test_robustness_bundle_replays_exactly(tmp_path: Path) -> None:
    result = run_operator(
        manifest_path=CURRENT_RELEASE_MANIFEST,
        output_root=tmp_path / "robustness-replay-output",
    )
    replay = replay_operator(
        output_root=result.paths.output_root,
        run_id="current-release-run",
    )
    bundle = euclid.inspect_demo_replay_bundle(output_root=result.paths.output_root)
    graph = euclid.load_demo_run_artifact_graph(output_root=result.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    robustness_report = graph.inspect(
        run_result.body["robustness_report_refs"][0]
    ).manifest
    robustness_hash = next(
        record.sha256
        for record in bundle.artifact_hash_records
        if record.artifact_role == "robustness_report"
    )

    assert replay.summary.replay_verification_status == "verified"
    assert bundle.replay_verification_status == "verified"
    assert robustness_hash == robustness_report.content_hash


def test_leakage_canary_failure_is_persisted(tmp_path: Path) -> None:
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        request_id="robustness-leakage-failure",
        robustness={
            "forced_leakage_canary_failures": ["future_target_level_feature"],
        },
    )

    result = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "robustness-leakage-output",
    )
    graph = euclid.load_demo_run_artifact_graph(output_root=result.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    robustness_report = graph.inspect(
        run_result.body["robustness_report_refs"][0]
    ).manifest
    scorecard = graph.inspect(run_result.body["primary_abstention_ref"]).parents[0]
    failing_canaries = [
        graph.inspect(ref).manifest.body
        for ref in robustness_report.body["leakage_canary_result_refs"]
        if graph.inspect(ref).manifest.body["pass"] is False
    ]

    assert robustness_report.body["final_robustness_status"] == "failed"
    assert failing_canaries
    assert any(
        "unexpected_canary_survival" in canary["failure_reason_codes"]
        for canary in failing_canaries
    )
    assert (
        "leakage_canary_failed"
        in graph.inspect(scorecard).manifest.body["descriptive_reason_codes"]
    )


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
