from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

import euclid
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.operator_runtime.run import run_operator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CSV = PROJECT_ROOT / "examples/minimal_dataset.csv"


def test_operator_run_ingests_external_evidence_bundle(tmp_path: Path) -> None:
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        request_id="external-evidence-operator-run",
        external_evidence=_valid_external_evidence_payload(),
    )

    result = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "external-evidence-output",
    )
    graph = euclid.load_demo_run_artifact_graph(output_root=result.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    bundle = graph.inspect(run_result.body["primary_external_evidence_ref"]).manifest
    replay_bundle = euclid.inspect_demo_replay_bundle(
        output_root=result.paths.output_root
    )

    assert "external_evidence" in result.summary.extension_lane_ids
    assert bundle.body["ordered_source_ids"] == ["paper-a", "paper-b"]
    assert bundle.body["source_count"] == 2
    assert "external_evidence_resolved" in replay_bundle.recorded_stage_order


def test_missing_or_invalid_source_payload_fails_fast(tmp_path: Path) -> None:
    invalid_payload = _valid_external_evidence_payload()
    invalid_payload["raw_sources"] = [
        {
            "source_id": "paper-a",
            "citation": "doi:10.1000/alpha",
            "evidence_kind": "measurement",
            "acquired_at": "2026-02-01T09:00:00Z",
            "content": {"finding": "copied forecast"},
            "provenance": {"publisher": "lab-alpha"},
            "independence_mode": "derived_from_predictive_output",
        }
    ]
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        request_id="external-evidence-invalid-run",
        external_evidence=invalid_payload,
    )

    with pytest.raises(ContractValidationError) as exc_info:
        run_operator(
            manifest_path=manifest_path,
            output_root=tmp_path / "external-evidence-invalid-output",
        )

    assert exc_info.value.code == "invalid_external_evidence_source"


def test_external_evidence_refs_flow_into_run_result(tmp_path: Path) -> None:
    manifest_path = _write_operator_manifest(
        tmp_path=tmp_path,
        request_id="external-evidence-ref-flow",
        external_evidence=_valid_external_evidence_payload(),
    )

    result = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "external-evidence-ref-flow-output",
    )
    graph = euclid.load_demo_run_artifact_graph(output_root=result.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    bundle_ref = run_result.body["primary_external_evidence_ref"]
    replay_bundle = euclid.inspect_demo_replay_bundle(
        output_root=result.paths.output_root
    )

    assert bundle_ref["schema_name"] == "external_evidence_manifest@1.0.0"
    assert TypedRef(
        schema_name="external_evidence_manifest@1.0.0",
        object_id=str(bundle_ref["object_id"]),
    ) in replay_bundle.required_manifest_refs
    assert replay_bundle.recorded_stage_order[-3:] == (
        "external_evidence_resolved",
        "publication_decision_resolved",
        "run_result_assembled",
    )


def _write_operator_manifest(
    *,
    tmp_path: Path,
    request_id: str,
    external_evidence: dict[str, object],
) -> Path:
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
              bundle_id: {external_evidence["bundle_id"]}
              domain_id: {external_evidence["domain_id"]}
              acquisition_window:
                start: "{external_evidence["acquisition_window"]["start"]}"
                end: "{external_evidence["acquisition_window"]["end"]}"
              raw_sources:
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    with manifest_path.open("a", encoding="utf-8") as handle:
        for source in external_evidence["raw_sources"]:
            handle.write("      - source_id: " + str(source["source_id"]) + "\n")
            handle.write("        citation: " + str(source["citation"]) + "\n")
            handle.write(
                "        evidence_kind: " + str(source["evidence_kind"]) + "\n"
            )
            handle.write(
                "        acquired_at: \"" + str(source["acquired_at"]) + "\"\n"
            )
            handle.write("        content:\n")
            for key, value in dict(source["content"]).items():
                handle.write(f"          {key}: {value}\n")
            handle.write("        provenance:\n")
            for key, value in dict(source["provenance"]).items():
                handle.write(f"          {key}: {value}\n")
            handle.write(
                "        independence_mode: "
                + str(source["independence_mode"])
                + "\n"
            )
    return manifest_path


def _valid_external_evidence_payload() -> dict[str, object]:
    return {
        "bundle_id": "operator_external_bundle",
        "domain_id": "glucose_regulation",
        "acquisition_window": {
            "start": "2026-02-01T00:00:00Z",
            "end": "2026-02-04T00:00:00Z",
        },
        "raw_sources": [
            {
                "source_id": "paper-b",
                "citation": "doi:10.1000/beta",
                "evidence_kind": "intervention",
                "acquired_at": "2026-02-03T09:00:00Z",
                "content": {"finding": "glucose decreases after perturbation"},
                "provenance": {"publisher": "lab-beta"},
                "independence_mode": "external_domain_source",
            },
            {
                "source_id": "paper-a",
                "citation": "doi:10.1000/alpha",
                "evidence_kind": "measurement",
                "acquired_at": "2026-02-01T09:00:00Z",
                "content": {"finding": "glucose recovers toward baseline"},
                "provenance": {"publisher": "lab-alpha"},
                "independence_mode": "external_domain_source",
            },
        ],
    }
