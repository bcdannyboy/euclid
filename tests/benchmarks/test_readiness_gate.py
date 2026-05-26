from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from euclid.readiness import (
    ReadinessGateResult,
    judge_benchmark_suite_readiness,
    judge_readiness,
)
from euclid.readiness.judgment import _surface_replay_verification_status

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ASSET_DIVERGENCE_ALLOWLIST = {
    "benchmarks:root-only:baselines/adversarial_honesty/leakage-safe-naive.yaml": (
        "authoritative=benchmarks; reason=root benchmark runner owns baselines"
    ),
    "benchmarks:root-only:baselines/predictive_generalization/naive-last-value.yaml": (
        "authoritative=benchmarks; reason=root benchmark runner owns baselines"
    ),
    "benchmarks:root-only:baselines/predictive_generalization/seasonal-naive.yaml": (
        "authoritative=benchmarks; reason=root benchmark runner owns baselines"
    ),
    "benchmarks:root-only:baselines/rediscovery/reference-description.yaml": (
        "authoritative=benchmarks; reason=root benchmark runner owns baselines"
    ),
    "benchmarks:asset-only:suites/phase08-holistic-honesty.yaml": (
        "authoritative=src/euclid/_assets/benchmarks; reason=packaged phase08 suite"
    ),
    "benchmarks:asset-only:tasks/adversarial_honesty/interpolation-bait-canary.yaml": (
        "authoritative=src/euclid/_assets/benchmarks; reason=packaged phase08 canary"
    ),
    "benchmarks:asset-only:tasks/adversarial_honesty/near-persistence-canary.yaml": (
        "authoritative=src/euclid/_assets/benchmarks; reason=packaged phase08 canary"
    ),
    "benchmarks:asset-only:tasks/adversarial_honesty/random-walk-canary.yaml": (
        "authoritative=src/euclid/_assets/benchmarks; reason=packaged phase08 canary"
    ),
    "benchmarks:asset-only:tasks/adversarial_honesty/row-index-leakage-canary.yaml": (
        "authoritative=src/euclid/_assets/benchmarks; reason=packaged phase08 canary"
    ),
    "benchmarks:asset-only:tasks/adversarial_honesty/sample-wide-closure-canary.yaml": (
        "authoritative=src/euclid/_assets/benchmarks; reason=packaged phase08 canary"
    ),
    (
        "benchmarks:asset-only:"
        "tasks/predictive_generalization/planted-additive-composition-noisy-phase08.yaml"
    ): (
        "authoritative=src/euclid/_assets/benchmarks; "
        "reason=packaged phase08 planted-law task"
    ),
    (
        "benchmarks:asset-only:"
        "tasks/predictive_generalization/planted-affine-lag-noisy-phase08.yaml"
    ): (
        "authoritative=src/euclid/_assets/benchmarks; "
        "reason=packaged phase08 planted-law task"
    ),
    (
        "benchmarks:asset-only:"
        "tasks/predictive_generalization/planted-damped-harmonic-noisy-phase08.yaml"
    ): (
        "authoritative=src/euclid/_assets/benchmarks; "
        "reason=packaged phase08 planted-law task"
    ),
    (
        "benchmarks:asset-only:"
        "tasks/predictive_generalization/planted-linear-noisy-phase08.yaml"
    ): (
        "authoritative=src/euclid/_assets/benchmarks; "
        "reason=packaged phase08 planted-law task"
    ),
    (
        "benchmarks:asset-only:"
        "tasks/predictive_generalization/real-series-gld-price-close-honesty-20260418.yaml"
    ): (
        "authoritative=src/euclid/_assets/benchmarks; "
        "reason=packaged phase08 real-series honesty task"
    ),
    (
        "benchmarks:asset-only:"
        "tasks/predictive_generalization/real-series-spy-daily-return-honesty-20260418.yaml"
    ): (
        "authoritative=src/euclid/_assets/benchmarks; "
        "reason=packaged phase08 real-series honesty task"
    ),
    (
        "benchmarks:asset-only:"
        "tasks/predictive_generalization/real-series-spy-price-close-honesty-20260416.yaml"
    ): (
        "authoritative=src/euclid/_assets/benchmarks; "
        "reason=packaged phase08 real-series honesty task"
    ),
    "benchmarks:asset-only:tasks/rediscovery/planted-additive-composition-exact-phase08.yaml": (
        "authoritative=src/euclid/_assets/benchmarks; "
        "reason=packaged phase08 planted-law task"
    ),
    "benchmarks:asset-only:tasks/rediscovery/planted-affine-lag-exact-phase08.yaml": (
        "authoritative=src/euclid/_assets/benchmarks; "
        "reason=packaged phase08 planted-law task"
    ),
    "benchmarks:asset-only:tasks/rediscovery/planted-damped-harmonic-exact-phase08.yaml": (
        "authoritative=src/euclid/_assets/benchmarks; "
        "reason=packaged phase08 planted-law task"
    ),
    "benchmarks:asset-only:tasks/rediscovery/planted-linear-exact-phase08.yaml": (
        "authoritative=src/euclid/_assets/benchmarks; "
        "reason=packaged phase08 planted-law task"
    ),
    "benchmarks:content-diff:suites/current-release.yaml": (
        "authoritative=src/euclid/_assets/benchmarks; "
        "reason=packaged suite carries benchmark threshold gate metadata"
    ),
    "benchmarks:content-diff:suites/full-vision.yaml": (
        "authoritative=src/euclid/_assets/benchmarks; "
        "reason=packaged suite carries benchmark threshold gate metadata"
    ),
    "schemas:root-only:contracts/fixture-provenance.yaml": (
        "authoritative=schemas; reason=root contract catalog has not been packaged"
    ),
    "schemas:root-only:contracts/live-api-evidence.yaml": (
        "authoritative=schemas; reason=root contract catalog has not been packaged"
    ),
    "schemas:root-only:contracts/numerical-policy.yaml": (
        "authoritative=schemas; reason=root contract catalog has not been packaged"
    ),
    "schemas:root-only:contracts/numerical-runtime.yaml": (
        "authoritative=schemas; reason=root contract catalog has not been packaged"
    ),
    "schemas:asset-only:readiness/benchmark-threshold-gates-v1.yaml": (
        "authoritative=src/euclid/_assets/schemas; "
        "reason=packaged release gate policy is not in root catalog yet"
    ),
    "schemas:content-diff:readiness/shipped-releasable-v1.yaml": (
        "authoritative=schemas; reason=root schema uses packaged docs paths"
    ),
}


def _gate(
    gate_id: str,
    *,
    status: str,
    required: bool = True,
    summary: str | None = None,
) -> ReadinessGateResult:
    return ReadinessGateResult(
        gate_id=gate_id,
        status=status,
        required=required,
        summary=summary or f"{gate_id} => {status}",
        evidence={"gate_id": gate_id},
    )


def test_judge_readiness_returns_public_ready_when_all_required_gates_pass() -> None:
    judgment = judge_readiness(
        judgment_id="release_v1_readiness",
        gate_results=(
            _gate("contracts.catalog", status="passed"),
            _gate("benchmarks.rediscovery", status="passed"),
            _gate("benchmarks.predictive_generalization", status="passed"),
            _gate("benchmarks.adversarial_honesty", status="passed"),
            _gate("notebook.smoke", status="passed"),
            _gate("determinism.same_seed", status="passed"),
            _gate("performance.runtime_smoke", status="passed"),
        ),
    )

    assert judgment.final_verdict == "ready"
    assert judgment.catalog_scope == "public"
    assert judgment.reason_codes == ()
    assert judgment.required_gate_count == 7
    assert judgment.passed_gate_count == 7
    assert judgment.failed_gate_count == 0
    assert judgment.missing_gate_count == 0


def test_judge_readiness_blocks_when_required_benchmark_gate_fails() -> None:
    judgment = judge_readiness(
        judgment_id="release_v1_readiness",
        gate_results=(
            _gate("contracts.catalog", status="passed"),
            _gate("benchmarks.rediscovery", status="passed"),
            _gate("benchmarks.predictive_generalization", status="failed"),
            _gate("benchmarks.adversarial_honesty", status="passed"),
            _gate("notebook.smoke", status="passed"),
        ),
    )

    assert judgment.final_verdict == "blocked"
    assert judgment.catalog_scope == "internal"
    assert judgment.reason_codes == ("benchmarks.predictive_generalization_failed",)
    assert judgment.failed_gate_count == 1


def test_benchmark_readiness_distinguishes_missing_replay_artifact(
    tmp_path: Path,
) -> None:
    missing_replay_path = tmp_path / "replay-missing.json"
    suite_result = _phase3_replay_suite_result(
        tmp_path,
        replay_artifact_path=missing_replay_path,
        replay_verification_status="missing",
    )

    judgment = judge_benchmark_suite_readiness(
        judgment_id="phase3_missing_replay_artifact",
        suite_result=suite_result,
    )
    gate = {
        gate.gate_id: gate for gate in judgment.gate_results
    }["surface.phase3_replay_surface"]

    assert judgment.final_verdict == "blocked"
    assert gate.status == "failed"
    assert gate.evidence["missing_replay_artifacts"] == [str(missing_replay_path)]
    assert gate.evidence["unverified_replay_artifacts"] == []


def test_benchmark_readiness_blocks_unverified_replay_file(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "replay-unverified.json"
    replay_path.write_text(
        json.dumps(
            {
                "artifact_type": "benchmark_replay_ref",
                "replay_verification_status": "unverified",
            }
        ),
        encoding="utf-8",
    )
    suite_result = _phase3_replay_suite_result(
        tmp_path,
        replay_artifact_path=replay_path,
        replay_verification_status="unverified",
    )

    judgment = judge_benchmark_suite_readiness(
        judgment_id="phase3_unverified_replay_file",
        suite_result=suite_result,
    )
    gate = {
        gate.gate_id: gate for gate in judgment.gate_results
    }["surface.phase3_replay_surface"]

    assert judgment.final_verdict == "blocked"
    assert gate.status == "failed"
    assert gate.evidence["missing_replay_artifacts"] == []
    assert gate.evidence["unverified_replay_artifacts"] == [str(replay_path)]
    assert gate.evidence["replay_verification_status"] == "unverified"


def test_benchmark_readiness_blocks_replay_file_without_verification_status(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "replay-legacy-without-status.json"
    replay_path.write_text(
        json.dumps({"artifact_type": "benchmark_replay_ref"}),
        encoding="utf-8",
    )
    suite_result = _phase3_replay_suite_result(
        tmp_path,
        replay_artifact_path=replay_path,
        replay_verification_status=None,
    )

    judgment = judge_benchmark_suite_readiness(
        judgment_id="phase3_replay_file_without_status",
        suite_result=suite_result,
    )
    gate = {
        gate.gate_id: gate for gate in judgment.gate_results
    }["surface.phase3_replay_surface"]

    assert judgment.final_verdict == "blocked"
    assert gate.status == "failed"
    assert gate.evidence["missing_replay_artifacts"] == []
    assert gate.evidence["unverified_replay_artifacts"] == [str(replay_path)]
    assert gate.evidence["replay_verification_status"] == "unverified"


def test_surface_without_replay_status_or_artifacts_is_unverified() -> None:
    assert _surface_replay_verification_status({}) == "unverified"


def test_packaged_benchmarks_and_schemas_match_or_document_divergence() -> None:
    for key, reason in _ASSET_DIVERGENCE_ALLOWLIST.items():
        assert "authoritative=" in reason, key
        assert "reason=" in reason, key

    failures = [
        *_asset_drift_failures("benchmarks"),
        *_asset_drift_failures("schemas"),
    ]

    assert not failures, (
        "Root and packaged assets must be byte-equal or explicitly allowlisted. "
        "Each finding includes the authoritative file location.\n"
        + "\n".join(failures)
    )


def _phase3_replay_suite_result(
    tmp_path: Path,
    *,
    replay_artifact_path: Path,
    replay_verification_status: str | None,
):
    task_result_path = tmp_path / f"task-{replay_verification_status}.json"
    task_result_path.write_text("{}", encoding="utf-8")
    summary_path = tmp_path / f"suite-{replay_verification_status}.json"
    task_summary = {
        "task_id": "phase3_replay_task",
        "track_id": "rediscovery",
        "forecast_object_type": "point",
        "score_law": "mean_absolute_error",
        "calibration_verdict": "not_applicable",
        "abstention_mode": "structural_miss",
        "replay_verification": "candidate_and_score_replay",
        "replay_artifact_paths": [str(replay_artifact_path)],
    }
    surface_evidence = {
        "forecast_object_types": ["point"],
        "score_laws": ["mean_absolute_error"],
        "calibration_verdicts": ["not_applicable"],
        "abstention_modes": ["structural_miss"],
        "replay_verification": "candidate_and_score_replay",
        "replay_artifact_paths": [str(replay_artifact_path)],
    }
    if replay_verification_status is not None:
        task_summary["replay_verification_status"] = replay_verification_status
        surface_evidence["replay_verification_status"] = replay_verification_status
    summary_path.write_text(
        json.dumps(
            {
                "suite_id": "phase3_replay_suite",
                "task_results": [task_summary],
                "surface_statuses": [
                    {
                        "surface_id": "phase3_replay_surface",
                        "evidence": surface_evidence,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        suite_manifest=SimpleNamespace(
            suite_id="phase3_replay_suite",
            required_tracks=("rediscovery",),
            task_manifest_paths=(tmp_path / "phase3-replay-task.yaml",),
        ),
        task_results=(
            SimpleNamespace(
                task_manifest=SimpleNamespace(
                    task_id="phase3_replay_task",
                    track_id="rediscovery",
                ),
                report_paths=SimpleNamespace(task_result_path=task_result_path),
            ),
        ),
        surface_statuses=(
            SimpleNamespace(
                surface_id="phase3_replay_surface",
                benchmark_status="passed",
                replay_status="passed",
            ),
        ),
        summary_path=summary_path,
    )


def _asset_drift_failures(asset_name: str) -> list[str]:
    root_dir = PROJECT_ROOT / asset_name
    package_dir = PROJECT_ROOT / "src/euclid/_assets" / asset_name
    root_files = _file_index(root_dir)
    package_files = _file_index(package_dir)
    findings: list[tuple[str, str]] = []

    for rel_path in sorted(set(root_files) - set(package_files)):
        key = f"{asset_name}:root-only:{rel_path}"
        findings.append(
            (
                key,
                (
                    f"{key}; authoritative={root_files[rel_path]}; "
                    f"packaged_missing={package_dir / rel_path}"
                ),
            )
        )
    for rel_path in sorted(set(package_files) - set(root_files)):
        key = f"{asset_name}:asset-only:{rel_path}"
        findings.append(
            (
                key,
                (
                    f"{key}; authoritative={package_files[rel_path]}; "
                    f"root_missing={root_dir / rel_path}"
                ),
            )
        )
    for rel_path in sorted(set(root_files) & set(package_files)):
        if root_files[rel_path].read_bytes() == package_files[rel_path].read_bytes():
            continue
        key = f"{asset_name}:content-diff:{rel_path}"
        findings.append(
            (
                key,
                (
                    f"{key}; authoritative={_authoritative_path(key)}; "
                    f"root={root_files[rel_path]}; packaged={package_files[rel_path]}"
                ),
            )
        )

    used_allowlist_keys = {
        key for key, _message in findings if key in _ASSET_DIVERGENCE_ALLOWLIST
    }
    stale_allowlist_keys = sorted(
        key
        for key in _ASSET_DIVERGENCE_ALLOWLIST
        if key.startswith(f"{asset_name}:") and key not in used_allowlist_keys
    )
    failures = [
        message
        for key, message in findings
        if key not in _ASSET_DIVERGENCE_ALLOWLIST
    ]
    failures.extend(
        f"stale-allowlist:{key}; authoritative={_authoritative_path(key)}"
        for key in stale_allowlist_keys
    )
    return failures


def _file_index(base_dir: Path) -> dict[str, Path]:
    return {
        path.relative_to(base_dir).as_posix(): path
        for path in base_dir.rglob("*")
        if path.is_file()
    }


def _authoritative_path(allowlist_key: str) -> str:
    reason = _ASSET_DIVERGENCE_ALLOWLIST.get(allowlist_key, "")
    for segment in reason.split(";"):
        text = segment.strip()
        if text.startswith("authoritative="):
            return text.removeprefix("authoritative=")
    return "unknown"
