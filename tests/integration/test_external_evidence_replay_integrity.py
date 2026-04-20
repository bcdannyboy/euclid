from __future__ import annotations

import sqlite3
from pathlib import Path
from textwrap import dedent

import euclid
from euclid.artifacts import FilesystemArtifactStore
from euclid.contracts.loader import load_contract_catalog
from euclid.manifests.base import ManifestEnvelope
from euclid.operator_runtime.replay import replay_operator
from euclid.operator_runtime.run import run_operator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CSV = PROJECT_ROOT / "examples/minimal_dataset.csv"


def test_replay_verifies_source_digests(tmp_path: Path) -> None:
    result = run_operator(
        manifest_path=_write_operator_manifest(tmp_path, "external-evidence-replay"),
        output_root=tmp_path / "external-evidence-replay-output",
    )

    replay = replay_operator(
        output_root=result.paths.output_root,
        run_id="external-evidence-replay",
    )

    assert replay.summary.replay_verification_status == "verified"
    assert replay.summary.failure_reason_codes == ()


def test_tampered_evidence_bundle_is_rejected(tmp_path: Path) -> None:
    result = run_operator(
        manifest_path=_write_operator_manifest(tmp_path, "external-evidence-tamper"),
        output_root=tmp_path / "external-evidence-tamper-output",
    )
    graph = euclid.load_demo_run_artifact_graph(output_root=result.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    bundle_ref = run_result.body["primary_external_evidence_ref"]
    bundle = graph.inspect(bundle_ref).manifest
    digest_ref = bundle.body["source_digest_refs"][0]

    _rewrite_manifest_in_registry(
        artifact_root=result.paths.artifact_root,
        metadata_store_path=result.paths.metadata_store_path,
        manifest=graph.inspect(digest_ref).manifest,
        body={
            **graph.inspect(digest_ref).manifest.body,
            "digest_sha256": "0" * 64,
        },
    )

    replay = replay_operator(
        output_root=result.paths.output_root,
        run_id="external-evidence-tamper",
    )

    assert replay.summary.replay_verification_status == "failed"
    assert "external_evidence_digest_mismatch" in replay.summary.failure_reason_codes


def test_independence_metadata_is_preserved(tmp_path: Path) -> None:
    result = run_operator(
        manifest_path=_write_operator_manifest(
            tmp_path,
            "external-evidence-independence",
        ),
        output_root=tmp_path / "external-evidence-independence-output",
    )
    graph = euclid.load_demo_run_artifact_graph(output_root=result.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    bundle = graph.inspect(run_result.body["primary_external_evidence_ref"]).manifest
    records = tuple(
        graph.inspect(record_ref).manifest for record_ref in bundle.body["record_refs"]
    )

    assert {record.body["independence_mode"] for record in records} == {
        "external_domain_source"
    }
    assert all(record.body["provenance"] for record in records)


def _rewrite_manifest_in_registry(
    *,
    artifact_root: Path,
    metadata_store_path: Path,
    manifest: ManifestEnvelope,
    body: dict[str, object],
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    rewritten = ManifestEnvelope.build(
        schema_name=manifest.schema_name,
        object_id=manifest.object_id,
        module_id=manifest.module_id,
        body=body,
        catalog=catalog,
    )
    stored = FilesystemArtifactStore(artifact_root).write_manifest(rewritten)
    connection = sqlite3.connect(metadata_store_path)
    try:
        connection.execute(
            """
            UPDATE manifests
            SET content_hash = ?, artifact_path = ?, size_bytes = ?
            WHERE schema_name = ? AND object_id = ?
            """,
            (
                stored.content_hash,
                str(stored.path),
                stored.size_bytes,
                rewritten.schema_name,
                rewritten.object_id,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _write_operator_manifest(tmp_path: Path, request_id: str) -> Path:
    manifest_path = tmp_path / f"{request_id}.yaml"
    manifest_path.write_text(
        dedent(
            f"""
            request_id: {request_id}
            dataset_csv: {DATASET_CSV}
            cutoff_available_at: null
            quantization_step: "0.5"
            minimum_description_gain_bits: 0.0
            min_train_size: 3
            horizon: 1
            search:
              class: bounded_heuristic
              seed: "0"
              family_ids:
                - constant
                - drift
                - linear_trend
                - seasonal_naive
            external_evidence:
              bundle_id: operator_external_bundle
              domain_id: glucose_regulation
              acquisition_window:
                start: "2026-02-01T00:00:00Z"
                end: "2026-02-04T00:00:00Z"
              raw_sources:
                - source_id: paper-b
                  citation: doi:10.1000/beta
                  evidence_kind: intervention
                  acquired_at: "2026-02-03T09:00:00Z"
                  content:
                    finding: glucose decreases after perturbation
                  provenance:
                    publisher: lab-beta
                  independence_mode: external_domain_source
                - source_id: paper-a
                  citation: doi:10.1000/alpha
                  evidence_kind: measurement
                  acquired_at: "2026-02-01T09:00:00Z"
                  content:
                    finding: glucose recovers toward baseline
                  provenance:
                    publisher: lab-alpha
                  independence_mode: external_domain_source
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return manifest_path
