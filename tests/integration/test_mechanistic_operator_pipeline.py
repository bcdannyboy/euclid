from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import euclid
from euclid.contracts.errors import ContractValidationError
from euclid.operator_runtime.replay import replay_operator
from euclid.operator_runtime.run import run_operator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL_VISION_MANIFEST = PROJECT_ROOT / "examples" / "full_vision_run.yaml"
MECHANISTIC_FIXTURE = (
    PROJECT_ROOT / "fixtures/runtime/mechanistic/mechanistic-positive-evidence.yaml"
)


def test_operator_run_emits_mechanistic_dossier(tmp_path: Path) -> None:
    fixture = _load_yaml(MECHANISTIC_FIXTURE)
    request_id = "mechanistic-operator-positive"
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        base_manifest=FULL_VISION_MANIFEST,
        request_id=request_id,
        external_evidence=_external_evidence_payload(fixture),
        mechanistic=_positive_mechanistic_payload(fixture),
    )

    result = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "mechanistic-operator-output",
    )
    entry = euclid.publish_demo_run_to_catalog(output_root=result.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=result.paths.output_root,
        publication_id=entry.publication_id,
    )
    graph = euclid.load_demo_run_artifact_graph(output_root=result.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    mechanistic_ref = run_result.body["primary_mechanistic_evidence_ref"]
    mechanistic_dossier = graph.inspect(mechanistic_ref).manifest

    assert result.summary.result_mode == "candidate_publication"
    assert (
        mechanistic_dossier.schema_name == "mechanistic_evidence_dossier_manifest@1.0.0"
    )
    assert mechanistic_dossier.body["status"] == "passed"
    assert mechanistic_dossier.body["resolved_claim_ceiling"] == (
        "mechanistically_compatible_hypothesis"
    )
    assert inspection.scorecard is not None
    assert inspection.scorecard.manifest.body["mechanistic_status"] == "passed"
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == (
        "mechanistically_compatible_hypothesis"
    )
    assert entry.mechanistic_evidence_ref == mechanistic_dossier.ref
    assert (
        "mechanistic_evidence_resolved" in inspection.replay_bundle.recorded_stage_order
    )
    assert mechanistic_dossier.ref in inspection.replay_bundle.required_manifest_refs
    assert any(
        record.artifact_role == "mechanistic_evidence"
        for record in inspection.replay_bundle.artifact_hash_records
    )


def test_missing_required_mechanistic_inputs_fail_cleanly(tmp_path: Path) -> None:
    fixture = _load_yaml(MECHANISTIC_FIXTURE)
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        base_manifest=FULL_VISION_MANIFEST,
        request_id="mechanistic-operator-missing-inputs",
        mechanistic=_positive_mechanistic_payload(fixture),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        run_operator(
            manifest_path=manifest_path,
            output_root=tmp_path / "mechanistic-missing-inputs-output",
        )

    assert exc_info.value.code == "mechanistic_external_evidence_required"


def test_mechanistic_artifacts_roundtrip_through_registry(tmp_path: Path) -> None:
    fixture = _load_yaml(MECHANISTIC_FIXTURE)
    request_id = "mechanistic-operator-roundtrip"
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        base_manifest=FULL_VISION_MANIFEST,
        request_id=request_id,
        external_evidence=_external_evidence_payload(fixture),
        mechanistic=_positive_mechanistic_payload(fixture),
    )

    result = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "mechanistic-roundtrip-output",
    )
    replay = replay_operator(
        output_root=result.paths.output_root,
        run_id=request_id,
    )
    bundle = euclid.inspect_demo_replay_bundle(output_root=result.paths.output_root)
    graph = euclid.load_demo_run_artifact_graph(output_root=result.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest

    assert replay.summary.replay_verification_status == "verified"
    assert replay.summary.run_result_ref == result.summary.run_result_ref
    assert bundle.recorded_stage_order[-4:] == (
        "external_evidence_resolved",
        "mechanistic_evidence_resolved",
        "publication_decision_resolved",
        "run_result_assembled",
    )
    assert any(
        record.artifact_role == "mechanistic_evidence"
        for record in bundle.artifact_hash_records
    )
    assert "primary_mechanistic_evidence_ref" in run_result.body


def _write_operator_manifest(
    *,
    tmp_path: Path,
    base_manifest: Path,
    request_id: str,
    external_evidence: dict[str, object] | None = None,
    mechanistic: dict[str, object] | None = None,
    robustness: dict[str, object] | None = None,
) -> Path:
    payload = _load_yaml(base_manifest)
    payload["request_id"] = request_id
    dataset_csv = Path(str(payload["dataset_csv"]))
    if not dataset_csv.is_absolute():
        payload["dataset_csv"] = str((base_manifest.parent / dataset_csv).resolve())
    probabilistic = payload.setdefault("probabilistic", {})
    assert isinstance(probabilistic, dict)
    calibration_thresholds = probabilistic.setdefault("calibration_thresholds", {})
    assert isinstance(calibration_thresholds, dict)
    calibration_thresholds["max_ks_distance"] = 1.0
    if external_evidence is not None:
        payload["external_evidence"] = external_evidence
    if mechanistic is not None:
        payload["mechanistic_evidence"] = mechanistic
    if robustness is not None:
        payload["robustness"] = robustness

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
        "mechanistic_evidence_id": "operator_mechanistic_positive",
        "term_bindings": deepcopy(fixture["term_bindings"]),
        "term_units": deepcopy(fixture["term_units"]),
        "invariance_checks": deepcopy(fixture["invariance_checks"]),
    }


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
