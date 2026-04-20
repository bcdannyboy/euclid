from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

import euclid
from euclid.operator_runtime.run import run_operator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL_VISION_MANIFEST = PROJECT_ROOT / "examples" / "full_vision_run.yaml"
POSITIVE_TASK = (
    PROJECT_ROOT / "benchmarks/tasks/mechanistic/mechanistic-lane-medium-positive.yaml"
)
NEGATIVE_TASK = (
    PROJECT_ROOT / "benchmarks/tasks/mechanistic/mechanistic-lane-medium-negative.yaml"
)
INSUFFICIENT_TASK = (
    PROJECT_ROOT
    / "benchmarks/tasks/mechanistic/mechanistic-lane-medium-insufficient.yaml"
)
MECHANISTIC_FIXTURE = (
    PROJECT_ROOT / "fixtures/runtime/mechanistic/mechanistic-positive-evidence.yaml"
)


def test_positive_case_requires_mechanistic_dossier_semantics(tmp_path: Path) -> None:
    fixture = _load_yaml(MECHANISTIC_FIXTURE)
    benchmark = euclid.profile_benchmark_task(
        manifest_path=POSITIVE_TASK,
        benchmark_root=tmp_path / "mechanistic-positive-benchmark",
        resume=False,
    )
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        request_id="mechanistic-track-positive",
        external_evidence=_external_evidence_payload(fixture),
        mechanistic=_positive_mechanistic_payload(fixture),
    )

    run = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "mechanistic-track-positive-output",
    )
    entry = euclid.publish_demo_run_to_catalog(output_root=run.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=run.paths.output_root,
        publication_id=entry.publication_id,
    )

    assert benchmark.task_manifest.task_family == "mechanistic_lane_positive"
    assert benchmark.submitter_results[0].status == "selected"
    assert entry.mechanistic_evidence_ref is not None
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == (
        "mechanistically_compatible_hypothesis"
    )


def test_negative_case_detects_contradictory_mechanistic_support(
    tmp_path: Path,
) -> None:
    fixture = _load_yaml(MECHANISTIC_FIXTURE)
    benchmark = euclid.profile_benchmark_task(
        manifest_path=NEGATIVE_TASK,
        benchmark_root=tmp_path / "mechanistic-negative-benchmark",
        resume=False,
    )
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        request_id="mechanistic-track-contradictory",
        external_evidence=_external_evidence_payload(fixture),
        mechanistic=_contradictory_mechanistic_payload(fixture),
    )

    run = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "mechanistic-track-contradictory-output",
    )
    graph = euclid.load_demo_run_artifact_graph(output_root=run.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    dossier = graph.inspect(
        run_result.body["primary_mechanistic_evidence_ref"]
    ).manifest
    entry = euclid.publish_demo_run_to_catalog(output_root=run.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=run.paths.output_root,
        publication_id=entry.publication_id,
    )

    assert benchmark.task_manifest.task_family == "mechanistic_lane_negative"
    assert dossier.body["status"] == "downgraded_to_predictively_supported"
    assert "invariance_check_failed" in dossier.body["reason_codes"]
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == "predictively_supported"


def test_insufficient_case_lowers_claim_ceiling_without_false_pass(
    tmp_path: Path,
) -> None:
    fixture = _load_yaml(MECHANISTIC_FIXTURE)
    benchmark = euclid.profile_benchmark_task(
        manifest_path=INSUFFICIENT_TASK,
        benchmark_root=tmp_path / "mechanistic-insufficient-benchmark",
        resume=False,
    )
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        request_id="mechanistic-track-insufficient",
        external_evidence=_external_evidence_payload(fixture),
        mechanistic=_insufficient_mechanistic_payload(fixture),
    )

    run = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "mechanistic-track-insufficient-output",
    )
    graph = euclid.load_demo_run_artifact_graph(output_root=run.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    dossier = graph.inspect(
        run_result.body["primary_mechanistic_evidence_ref"]
    ).manifest
    entry = euclid.publish_demo_run_to_catalog(output_root=run.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=run.paths.output_root,
        publication_id=entry.publication_id,
    )

    assert benchmark.task_manifest.task_family == "mechanistic_lane_insufficient"
    assert dossier.body["status"] == "downgraded_to_predictively_supported"
    assert "units_check_incomplete" in dossier.body["reason_codes"]
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == "predictively_supported"
    assert (
        inspection.claim_card.manifest.body["claim_type"]
        != "mechanistically_compatible_hypothesis"
    )


def _write_operator_manifest(
    *,
    tmp_path: Path,
    request_id: str,
    external_evidence: dict[str, object],
    mechanistic: dict[str, object],
) -> Path:
    payload = _load_yaml(FULL_VISION_MANIFEST)
    payload["request_id"] = request_id
    dataset_csv = Path(str(payload["dataset_csv"]))
    if not dataset_csv.is_absolute():
        payload["dataset_csv"] = str(
            (FULL_VISION_MANIFEST.parent / dataset_csv).resolve()
        )
    probabilistic = payload.setdefault("probabilistic", {})
    assert isinstance(probabilistic, dict)
    calibration_thresholds = probabilistic.setdefault("calibration_thresholds", {})
    assert isinstance(calibration_thresholds, dict)
    calibration_thresholds["max_ks_distance"] = 1.0
    payload["external_evidence"] = external_evidence
    payload["mechanistic_evidence"] = mechanistic

    manifest_path = tmp_path / f"{request_id}.yaml"
    manifest_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path


def _external_evidence_payload(fixture: dict[str, object]) -> dict[str, object]:
    return {
        "bundle_id": fixture["bundle_id"],
        "domain_id": fixture["domain_id"],
        "acquisition_window": deepcopy(fixture["acquisition_window"]),
        "raw_sources": deepcopy(fixture["raw_sources"]),
    }


def _positive_mechanistic_payload(fixture: dict[str, object]) -> dict[str, object]:
    return {
        "mechanistic_evidence_id": "mechanistic_track_positive",
        "term_bindings": deepcopy(fixture["term_bindings"]),
        "term_units": deepcopy(fixture["term_units"]),
        "invariance_checks": deepcopy(fixture["invariance_checks"]),
    }


def _contradictory_mechanistic_payload(
    fixture: dict[str, object],
) -> dict[str, object]:
    payload = _positive_mechanistic_payload(fixture)
    payload["mechanistic_evidence_id"] = "mechanistic_track_contradictory"
    payload["invariance_checks"] = [{"check_id": "meal_shift", "status": "failed"}]
    return payload


def _insufficient_mechanistic_payload(
    fixture: dict[str, object],
) -> dict[str, object]:
    payload = _positive_mechanistic_payload(fixture)
    payload["mechanistic_evidence_id"] = "mechanistic_track_insufficient"
    payload["term_bindings"] = [
        *deepcopy(fixture["term_bindings"]),
        {
            "term_id": "lag2_state",
            "domain_entity": "circulating_glucose",
            "activity": "secondary_persistence",
        },
    ]
    return payload


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
