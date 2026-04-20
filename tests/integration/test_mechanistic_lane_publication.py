from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

import euclid
from euclid.operator_runtime.replay import replay_operator
from euclid.operator_runtime.run import run_operator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL_VISION_MANIFEST = PROJECT_ROOT / "examples" / "full_vision_run.yaml"
MECHANISTIC_FIXTURE = (
    PROJECT_ROOT / "fixtures/runtime/mechanistic/mechanistic-positive-evidence.yaml"
)


def test_failed_mechanistic_support_lowers_claim_ceiling(tmp_path: Path) -> None:
    fixture = _load_yaml(MECHANISTIC_FIXTURE)
    request_id = "mechanistic-publication-contradictory"
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        request_id=request_id,
        external_evidence=_external_evidence_payload(fixture),
        mechanistic=_contradictory_mechanistic_payload(fixture),
    )

    result = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "mechanistic-publication-contradictory-output",
    )
    entry = euclid.publish_demo_run_to_catalog(output_root=result.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=result.paths.output_root,
        publication_id=entry.publication_id,
    )
    graph = euclid.load_demo_run_artifact_graph(output_root=result.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    dossier = graph.inspect(
        run_result.body["primary_mechanistic_evidence_ref"]
    ).manifest

    assert result.summary.result_mode == "candidate_publication"
    assert dossier.body["status"] == "downgraded_to_predictively_supported"
    assert inspection.scorecard is not None
    assert inspection.scorecard.manifest.body["mechanistic_status"] == (
        "downgraded_to_predictively_supported"
    )
    assert (
        "invariance_check_failed"
        in inspection.scorecard.manifest.body["mechanistic_reason_codes"]
    )
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == "predictively_supported"
    assert inspection.claim_card.manifest.body["mechanistic_evidence_ref"] == (
        dossier.ref.as_dict()
    )
    assert entry.mechanistic_evidence_ref == dossier.ref


def test_strong_mechanistic_support_updates_publication_record(tmp_path: Path) -> None:
    fixture = _load_yaml(MECHANISTIC_FIXTURE)
    request_id = "mechanistic-publication-strong"
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        request_id=request_id,
        external_evidence=_external_evidence_payload(fixture),
        mechanistic=_positive_mechanistic_payload(fixture),
    )

    result = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "mechanistic-publication-strong-output",
    )
    entry = euclid.publish_demo_run_to_catalog(output_root=result.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=result.paths.output_root,
        publication_id=entry.publication_id,
    )
    graph = euclid.load_demo_run_artifact_graph(output_root=result.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    dossier = graph.inspect(
        run_result.body["primary_mechanistic_evidence_ref"]
    ).manifest

    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == (
        "mechanistically_compatible_hypothesis"
    )
    assert entry.mechanistic_evidence_ref == dossier.ref
    assert inspection.entry.mechanistic_evidence_ref == dossier.ref
    assert run_result.body["primary_mechanistic_evidence_ref"] == dossier.ref.as_dict()


def test_mechanistic_publication_replay_is_exact(tmp_path: Path) -> None:
    fixture = _load_yaml(MECHANISTIC_FIXTURE)
    request_id = "mechanistic-publication-replay"
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        request_id=request_id,
        external_evidence=_external_evidence_payload(fixture),
        mechanistic=_positive_mechanistic_payload(fixture),
    )

    result = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "mechanistic-publication-replay-output",
    )
    entry = euclid.publish_demo_run_to_catalog(output_root=result.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=result.paths.output_root,
        publication_id=entry.publication_id,
    )
    replay = replay_operator(output_root=result.paths.output_root, run_id=request_id)

    assert replay.summary.replay_verification_status == "verified"
    assert entry.mechanistic_evidence_ref is not None
    assert (
        entry.mechanistic_evidence_ref
        in inspection.replay_bundle.required_manifest_refs
    )
    assert inspection.replay_bundle.recorded_stage_order[-4:] == (
        "external_evidence_resolved",
        "mechanistic_evidence_resolved",
        "publication_decision_resolved",
        "run_result_assembled",
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
        "mechanistic_evidence_id": "mechanistic_publication_positive",
        "term_bindings": deepcopy(fixture["term_bindings"]),
        "term_units": deepcopy(fixture["term_units"]),
        "invariance_checks": deepcopy(fixture["invariance_checks"]),
    }


def _contradictory_mechanistic_payload(
    fixture: dict[str, object],
) -> dict[str, object]:
    payload = _positive_mechanistic_payload(fixture)
    payload["mechanistic_evidence_id"] = "mechanistic_publication_contradictory"
    payload["invariance_checks"] = [
        {"check_id": "meal_shift", "status": "failed"},
    ]
    return payload


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
