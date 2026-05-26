from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from euclid.contracts.refs import TypedRef
from euclid.operator_runtime.models import OperatorReplayResult, OperatorRunResult


def runtime_sha256_file(path: Path) -> str:
    return f"runtime_sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def runtime_sha256_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"runtime_sha256:{hashlib.sha256(encoded).hexdigest()}"


def bind_operator_run_evidence_report(
    *,
    result: OperatorRunResult,
    report_path: Path,
) -> Path:
    payload = _read_report_payload(report_path)
    run_summary = _read_run_summary_payload(result.paths.run_summary_path)
    payload["run_summary_sha256"] = runtime_sha256_file(result.paths.run_summary_path)
    payload["run_id_binding"] = {
        "request_id": result.request.request_id,
        "run_result_object_id": result.summary.run_result_ref.object_id,
        "run_summary_request_id": _summary_request_id(run_summary),
    }
    return _write_report_payload(report_path, payload)


def write_operator_run_result_artifact(*, result: OperatorRunResult) -> Path:
    path = result.paths.output_root / "run-result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": result.request.request_id,
        "run_summary_path": str(result.paths.run_summary_path),
        "run_summary_sha256": runtime_sha256_file(result.paths.run_summary_path),
        "run_result_ref": _typed_ref_payload(result.summary.run_result_ref),
        "bundle_ref": _typed_ref_payload(result.summary.bundle_ref),
        "output_root": str(result.paths.output_root),
    }
    return _write_report_payload(path, payload)


def bind_operator_replay_evidence_report(
    *,
    run_id: str,
    result: OperatorReplayResult,
    report_path: Path,
) -> Path:
    payload = _read_report_payload(report_path)
    run_summary = _read_run_summary_payload(result.paths.run_summary_path)
    payload["run_summary_sha256"] = runtime_sha256_file(result.paths.run_summary_path)
    payload["run_id_binding"] = {
        "requested_run_id": run_id,
        "run_result_object_id": result.summary.run_result_ref.object_id,
        "run_summary_request_id": _summary_request_id(run_summary),
    }
    payload["replay_result_sha256"] = runtime_sha256_payload(
        _replay_result_digest_payload(
            run_id=run_id,
            result=result,
            run_summary_sha256=str(payload["run_summary_sha256"]),
        )
    )
    _bind_operator_run_report(payload)
    return _write_report_payload(report_path, payload)


def _bind_operator_run_report(payload: dict[str, Any]) -> None:
    raw_path = payload.get("operator_run_evidence_report_path")
    if not isinstance(raw_path, str) or not raw_path:
        return
    run_report_path = Path(raw_path)
    if not run_report_path.is_file():
        return
    run_report = _read_report_payload(run_report_path)
    run_report_sha256 = runtime_sha256_file(run_report_path)
    payload["operator_run_evidence_report_sha256"] = run_report_sha256
    payload["operator_run_evidence_report_binding"] = {
        "path": str(run_report_path),
        "sha256": run_report_sha256,
        "run_id": str(run_report.get("run_id", "")),
    }


def _replay_result_digest_payload(
    *,
    run_id: str,
    result: OperatorReplayResult,
    run_summary_sha256: str,
) -> dict[str, Any]:
    summary = result.summary
    return {
        "run_id": run_id,
        "run_summary_sha256": run_summary_sha256,
        "bundle_ref": _typed_ref_payload(summary.bundle_ref),
        "run_result_ref": _typed_ref_payload(summary.run_result_ref),
        "selected_candidate_ref": _typed_ref_payload(summary.selected_candidate_ref),
        "selected_family": summary.selected_family,
        "result_mode": summary.result_mode,
        "forecast_object_type": summary.forecast_object_type,
        "replay_verification_status": summary.replay_verification_status,
        "confirmatory_primary_score": summary.confirmatory_primary_score,
        "failure_reason_codes": list(summary.failure_reason_codes),
    }


def _typed_ref_payload(ref: TypedRef | None) -> dict[str, str] | None:
    return None if ref is None else ref.as_dict()


def _summary_request_id(payload: Mapping[str, Any]) -> str:
    value = payload.get("request_id")
    return value if isinstance(value, str) else ""


def _read_run_summary_payload(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("operator run summary must deserialize to a mapping")
    return payload


def _read_report_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("operator evidence report must deserialize to an object")
    return payload


def _write_report_payload(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "bind_operator_replay_evidence_report",
    "bind_operator_run_evidence_report",
    "runtime_sha256_file",
    "runtime_sha256_payload",
    "write_operator_run_result_artifact",
]
