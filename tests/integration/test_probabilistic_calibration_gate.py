from __future__ import annotations

from pathlib import Path

import euclid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "fixtures/runtime/phase06"
DISTRIBUTION_MANIFEST = FIXTURE_ROOT / "probabilistic-distribution-demo.yaml"
CALIBRATION_FAILURE_MANIFEST = (
    FIXTURE_ROOT / "probabilistic-distribution-calibration-failure-demo.yaml"
)


def test_failed_calibration_blocks_publication(tmp_path: Path) -> None:
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=CALIBRATION_FAILURE_MANIFEST,
        output_root=tmp_path / "distribution-calibration-failure",
    )
    entry = euclid.publish_demo_run_to_catalog(output_root=result.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=result.paths.output_root,
        publication_id=entry.publication_id,
    )

    assert result.calibration.status == "failed"
    assert result.summary.calibration_status == "failed"
    assert entry.publication_mode == "candidate_publication"
    assert inspection.scorecard is not None
    assert inspection.scorecard.manifest.body["predictive_status"] == "blocked"
    assert inspection.scorecard.manifest.body["predictive_reason_codes"] == [
        "calibration_failed"
    ]
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == "descriptive_structure"
    assert inspection.claim_card.manifest.body["predictive_support_status"] == "blocked"
    assert (
        "probabilistic_forecast_within_declared_validation_scope"
        not in inspection.claim_card.manifest.body["allowed_interpretation_codes"]
    )


def test_calibration_success_allows_publication(tmp_path: Path) -> None:
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=DISTRIBUTION_MANIFEST,
        output_root=tmp_path / "distribution-calibration-success",
    )
    entry = euclid.publish_demo_run_to_catalog(output_root=result.paths.output_root)
    inspection = euclid.inspect_demo_catalog_entry(
        output_root=result.paths.output_root,
        publication_id=entry.publication_id,
    )

    assert result.calibration.status == "passed"
    assert result.summary.calibration_status == "passed"
    assert entry.publication_mode == "candidate_publication"
    assert inspection.scorecard is not None
    assert inspection.scorecard.manifest.body["predictive_status"] == "passed"
    assert inspection.claim_card is not None
    assert inspection.claim_card.manifest.body["claim_type"] == "predictive_within_declared_scope"
    assert (
        inspection.claim_card.manifest.body["predictive_support_status"]
        == "confirmatory_supported"
    )
    assert (
        "probabilistic_forecast_within_declared_validation_scope"
        in inspection.claim_card.manifest.body["allowed_interpretation_codes"]
    )
