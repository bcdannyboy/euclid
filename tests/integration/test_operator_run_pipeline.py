from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import euclid.operator_runtime._compat_runtime as compat_runtime
from euclid.cli.run import run_command as cli_run_command
from euclid.cir.models import (
    CandidateIntermediateRepresentation,
    CIRBackendOriginRecord,
    CIRCanonicalSerialization,
    CIREvidenceLayer,
    CIRExecutionLayer,
    CIRForecastOperator,
    CIRHistoryAccessContract,
    CIRInputSignature,
    CIRReplayHooks,
    CIRStateSignature,
    CIRStructuralLayer,
)
from euclid.contracts.errors import ContractValidationError
from euclid.operator_runtime.run import run_operator
from euclid.reducers.models import ReducerStateObject

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_RELEASE_MANIFEST = PROJECT_ROOT / "examples" / "current_release_run.yaml"
FULL_VISION_MANIFEST = PROJECT_ROOT / "examples" / "full_vision_run.yaml"


def _runtime_sha256(path: Path) -> str:
    return f"runtime_sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def test_operator_run_no_longer_depends_on_demo(tmp_path: Path) -> None:
    result = run_operator(
        manifest_path=CURRENT_RELEASE_MANIFEST,
        output_root=tmp_path / "operator-run",
    )

    assert result.paths.output_root == (tmp_path / "operator-run")
    assert "build/demo" not in str(result.paths.output_root)
    assert result.summary.selected_candidate_id.startswith("operator_")
    assert "prototype" not in result.paths.run_summary_path.read_text(encoding="utf-8")


def test_operator_run_emits_typed_bundle_and_run_result(tmp_path: Path) -> None:
    result = run_operator(
        manifest_path=CURRENT_RELEASE_MANIFEST,
        output_root=tmp_path / "operator-run",
    )

    assert result.paths.run_summary_path.is_file()
    assert result.paths.metadata_store_path.is_file()
    assert (
        result.summary.bundle_ref.schema_name
        == "reproducibility_bundle_manifest@1.0.0"
    )
    assert result.summary.run_result_ref.schema_name == "run_result_manifest@1.1.0"
    assert result.summary.publication_record_ref is not None


def test_operator_run_supports_extension_lane_manifest(tmp_path: Path) -> None:
    result = run_operator(
        manifest_path=FULL_VISION_MANIFEST,
        output_root=tmp_path / "operator-run",
    )

    assert result.summary.forecast_object_type == "distribution"
    assert "distribution" in result.summary.extension_lane_ids
    assert "algorithmic_last_observation" in result.summary.extension_lane_ids


def test_operator_run_evidence_report_emits_digest_and_run_id_binding(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "operator-run"
    report_path = tmp_path / "reports" / "full_vision_operator_run_evidence.json"
    report_path.parent.mkdir(parents=True)

    cli_run_command(
        config=FULL_VISION_MANIFEST,
        output_root=output_root,
        evidence_report=report_path,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    run_summary_path = Path(payload["run_summary_path"])
    assert payload["run_id"] == "full-vision-run"
    assert payload["source_tree_digest_or_wheel_digest"]
    assert payload["run_summary_sha256"] == _runtime_sha256(run_summary_path)
    assert payload["run_id_binding"] == {
        "request_id": "full-vision-run",
        "run_result_object_id": "full-vision-run_run_result",
        "run_summary_request_id": "full-vision-run",
    }


def test_operator_run_with_evidence_report_writes_declared_run_result_artifact(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "operator-run"
    report_path = tmp_path / "reports" / "full_vision_operator_run_evidence.json"
    report_path.parent.mkdir(parents=True)

    cli_run_command(
        config=FULL_VISION_MANIFEST,
        output_root=output_root,
        evidence_report=report_path,
    )

    run_result_path = output_root / "run-result.json"
    assert run_result_path.is_file()

    payload = json.loads(run_result_path.read_text(encoding="utf-8"))
    evidence_payload = json.loads(report_path.read_text(encoding="utf-8"))
    run_summary_path = Path(payload["run_summary_path"])
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))

    assert run_summary_path == (
        output_root / "sealed-runs" / "full-vision-run" / "run-summary.json"
    )
    assert payload["run_id"] == "full-vision-run"
    assert payload["run_summary_path"] == evidence_payload["run_summary_path"]
    assert payload["run_summary_sha256"] == _runtime_sha256(run_summary_path)
    assert payload["run_summary_sha256"] == evidence_payload["run_summary_sha256"]
    assert payload["run_result_ref"] == run_summary["run_result_ref"]
    assert payload["run_result_ref"] == evidence_payload["run_result_ref"]
    assert payload["bundle_ref"] == run_summary["bundle_ref"]
    assert payload["bundle_ref"] == evidence_payload["bundle_ref"]
    assert payload["output_root"] == str(output_root)
    assert payload["output_root"] == evidence_payload["output_root"]


def test_operator_run_rejects_non_cir_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_selector = compat_runtime._select_frontier_candidate

    def _select_unclosed_candidate(search_result):
        selected = original_selector(search_result)
        return CandidateIntermediateRepresentation(
            structural_layer=CIRStructuralLayer(
                cir_family_id=selected.structural_layer.cir_family_id,
                cir_form_class=selected.structural_layer.cir_form_class,
                input_signature=CIRInputSignature(target_series="target"),
                state_signature=CIRStateSignature(
                    persistent_state=ReducerStateObject()
                ),
            ),
            execution_layer=CIRExecutionLayer(
                history_access_contract=CIRHistoryAccessContract(
                    contract_id="full_prefix",
                    access_mode="full_prefix",
                ),
                state_update_law_id=selected.execution_layer.state_update_law_id,
                forecast_operator=CIRForecastOperator(
                    operator_id="one_step_point_forecast",
                    horizon=1,
                ),
                observation_model_binding=(
                    selected.execution_layer.observation_model_binding
                ),
            ),
            evidence_layer=CIREvidenceLayer(
                canonical_serialization=CIRCanonicalSerialization(
                    canonical_bytes=b"{}",
                    content_hash="sha256:placeholder",
                ),
                model_code_decomposition=selected.evidence_layer.model_code_decomposition,
                backend_origin_record=CIRBackendOriginRecord(
                    adapter_id="patched-test",
                    adapter_class="test",
                    source_candidate_id="patched_non_cir_candidate",
                    search_class=selected.evidence_layer.backend_origin_record.search_class,
                ),
                replay_hooks=CIRReplayHooks(),
            ),
        )

    monkeypatch.setattr(
        compat_runtime,
        "_select_frontier_candidate",
        _select_unclosed_candidate,
    )

    with pytest.raises(ContractValidationError) as exc_info:
        run_operator(
            manifest_path=FULL_VISION_MANIFEST,
            output_root=tmp_path / "operator-run",
        )

    assert exc_info.value.code == "cir_closure_required"
