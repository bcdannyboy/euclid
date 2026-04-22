from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import euclid
import euclid.operator_runtime._compat_runtime as compat_runtime
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISTRIBUTION_MANIFEST = (
    PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-distribution-demo.yaml"
)
CALIBRATION_FAILURE_MANIFEST = (
    PROJECT_ROOT
    / (
        "fixtures/runtime/phase06/"
        "probabilistic-distribution-calibration-failure-demo.yaml"
    )
)


def _ref_string(ref: euclid.TypedRef) -> str:
    return f"{ref.schema_name}:{ref.object_id}"


def test_probabilistic_publication_surfaces_support_replay_and_catalog_without_run_id(
    tmp_path: Path,
) -> None:
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=DISTRIBUTION_MANIFEST,
        output_root=tmp_path / "probabilistic-public-output",
    )

    replay = euclid.replay_demo(output_root=result.paths.output_root)
    entry = euclid.publish_demo_run_to_catalog(output_root=result.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=result.paths.output_root,
        publication_id=entry.publication_id,
    )

    assert replay.summary.bundle_ref == result.summary.bundle_ref
    assert replay.summary.run_result_ref == result.summary.run_result_ref
    assert replay.summary.replay_verification_status == "verified"
    assert entry.publication_mode == "candidate_publication"
    assert entry.forecast_object_type == "distribution"
    assert entry.validation_scope_ref is not None
    assert entry.primary_score_result_ref == result.summary.score_result_ref
    assert entry.primary_calibration_result_ref == result.summary.calibration_result_ref
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == "predictive_within_declared_scope"
    assert (
        inspection.claim_card.manifest.body["predictive_support_status"]
        == "confirmatory_supported"
    )
    assert inspection.replay_bundle.replay_verification_status == "verified"
    assert {
        _ref_string(result.summary.prediction_artifact_ref),
        _ref_string(entry.validation_scope_ref),
        _ref_string(result.summary.score_result_ref),
        _ref_string(result.summary.calibration_result_ref),
    } <= {_ref_string(ref) for ref in inspection.replay_bundle.required_manifest_refs}


def test_probabilistic_publication_downgrade_is_preserved_in_catalog_and_replay(
    tmp_path: Path,
) -> None:
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=CALIBRATION_FAILURE_MANIFEST,
        output_root=tmp_path / "probabilistic-downgrade-output",
    )

    replay = euclid.replay_demo(output_root=result.paths.output_root)
    entry = euclid.publish_demo_run_to_catalog(output_root=result.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=result.paths.output_root,
        publication_id=entry.publication_id,
    )

    assert replay.summary.bundle_ref == result.summary.bundle_ref
    assert replay.summary.replay_verification_status == "verified"
    assert entry.publication_mode == "candidate_publication"
    assert entry.forecast_object_type == "distribution"
    assert entry.validation_scope_ref is not None
    assert entry.primary_score_result_ref == result.summary.score_result_ref
    assert entry.primary_calibration_result_ref == result.summary.calibration_result_ref
    assert inspection.scorecard is not None
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == "descriptive_structure"
    assert inspection.claim_card.manifest.body["predictive_support_status"] == "blocked"
    assert inspection.scorecard.manifest.body["predictive_status"] == "blocked"
    assert inspection.scorecard.manifest.body["predictive_reason_codes"] == [
        "calibration_failed"
    ]
    assert inspection.replay_bundle.replay_verification_status == "verified"


def test_compat_mechanistic_reresolution_preserves_descriptive_gate_for_fallback_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "compat-probabilistic-mechanistic.yaml"
    dataset_path = (
        PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-trend-series.csv"
    )
    manifest_path.write_text(
        DISTRIBUTION_MANIFEST.read_text(encoding="utf-8").replace(
            "dataset_csv: probabilistic-trend-series.csv",
            f"dataset_csv: {dataset_path.as_posix()}",
        )
        + (
            "\nmechanistic_evidence:\n"
            "  mechanistic_evidence_id: fallback_mechanistic_probe\n"
            "  term_bindings: []\n"
            "  term_units: []\n"
            "  invariance_checks: []\n"
        ),
        encoding="utf-8",
    )

    captured_calls: list[dict[str, object]] = []
    original_resolve = compat_runtime.resolve_scorecard_status

    def _capture_resolve_scorecard_status(**kwargs):
        captured_calls.append(dict(kwargs))
        if len(captured_calls) == 2:
            raise RuntimeError("captured_second_scorecard_resolution")
        return original_resolve(**kwargs)

    monkeypatch.setattr(
        compat_runtime,
        "_selected_candidate_law_semantics",
        lambda **kwargs: (False, ("outside_law_eligible_scope",)),
    )
    monkeypatch.setattr(
        compat_runtime,
        "_register_requested_mechanistic_evidence",
        lambda **kwargs: SimpleNamespace(
            dossier=SimpleNamespace(
                manifest=SimpleNamespace(
                    body={"status": "passed", "reason_codes": ()}
                )
            )
        ),
    )
    monkeypatch.setattr(
        compat_runtime,
        "resolve_scorecard_status",
        _capture_resolve_scorecard_status,
    )

    with pytest.raises(RuntimeError, match="captured_second_scorecard_resolution"):
        compat_runtime.run_demo_probabilistic_evaluation(
            manifest_path=manifest_path,
            output_root=tmp_path / "compat-probabilistic-out",
        )

    assert len(captured_calls) == 2
    assert captured_calls[0]["candidate_admissible"] is False
    assert captured_calls[0]["descriptive_failure_reason_codes"] == (
        "outside_law_eligible_scope",
    )
    assert captured_calls[1]["candidate_admissible"] is False
    assert captured_calls[1]["descriptive_failure_reason_codes"] == (
        "outside_law_eligible_scope",
    )
    assert captured_calls[1]["mechanistic_requested"] is True
