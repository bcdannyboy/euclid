from __future__ import annotations

from pathlib import Path

from euclid.contracts.loader import load_contract_catalog
from euclid.falsification.counterexamples import discover_counterexamples
from euclid.falsification.dossier import build_falsification_dossier
from euclid.falsification.residuals import evaluate_residual_diagnostics
from euclid.manifests.base import ManifestEnvelope
from euclid.modules.claims import resolve_claim_publication
from euclid.modules.gate_lifecycle import resolve_scorecard_status

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_falsification_dossier_registers_and_blocks_predictive_publication() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    dossier = build_falsification_dossier(
        candidate_id="candidate_with_structured_residuals",
        residual_diagnostics=evaluate_residual_diagnostics(
            candidate_id="candidate_with_structured_residuals",
            residuals=(-3.0, -2.0, -1.0, 1.0, 2.0, 3.0),
        ),
        counterexample_result=discover_counterexamples(
            candidate_id="candidate_with_structured_residuals",
            cases=(
                {
                    "case_id": "far_holdout",
                    "prediction": 10.0,
                    "observed": 16.0,
                    "domain_valid": True,
                    "extrapolation_distance": 2.0,
                },
            ),
            error_tolerance=1.0,
            max_extrapolation_distance=1.0,
        ),
    )
    manifest = ManifestEnvelope.build(
        schema_name="falsification_dossier@1.0.0",
        module_id="gate_lifecycle",
        body=dossier.as_manifest(),
        catalog=catalog,
    )
    scorecard = resolve_scorecard_status(
        candidate_admissible=True,
        robustness_status="passed",
        candidate_beats_baseline=True,
        confirmatory_promotion_allowed=True,
        point_score_comparison_status="comparable",
        time_safety_status="passed",
        calibration_status="not_applicable_for_forecast_type",
        falsification_status=manifest.body["status"],
        falsification_reason_codes=manifest.body["reason_codes"],
    )
    decision = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": scorecard.descriptive_status,
            "descriptive_reason_codes": list(scorecard.descriptive_reason_codes),
            "predictive_status": scorecard.predictive_status,
            "predictive_reason_codes": list(scorecard.predictive_reason_codes),
        }
    )

    assert manifest.ref.schema_name == "falsification_dossier@1.0.0"
    assert manifest.body["replay_identity"].startswith("falsification-dossier:")
    assert scorecard.predictive_status == "blocked"
    assert decision.claim_type == "descriptive_structure"
