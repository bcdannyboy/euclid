from __future__ import annotations

from euclid.falsification.counterexamples import discover_counterexamples
from euclid.falsification.dossier import build_falsification_dossier
from euclid.falsification.residuals import evaluate_residual_diagnostics
from euclid.falsification.surrogate_tests import evaluate_parameter_stability


def test_dossier_collects_transport_and_stochastic_failures() -> None:
    residuals = evaluate_residual_diagnostics(
        candidate_id="candidate",
        residuals=(-1.0, -0.5, 0.5, 1.0),
        autocorrelation_threshold=0.95,
    )
    counterexamples = discover_counterexamples(candidate_id="candidate", cases=())
    stability = evaluate_parameter_stability(
        candidate_id="candidate",
        window_parameters=({"theta": 1.0}, {"theta": 1.01}),
    )

    dossier = build_falsification_dossier(
        candidate_id="candidate",
        residual_diagnostics=residuals,
        counterexample_result=counterexamples,
        parameter_stability=stability,
        transport_status="failed",
        calibration_status="failed",
    )

    assert dossier.status == "failed"
    assert set(dossier.reason_codes) >= {
        "transport_failed",
        "stochastic_miscalibration",
    }
    assert dossier.claim_effect == "downgrade_predictive_claim"
    assert dossier.replay_identity.startswith("falsification-dossier:")


def test_domain_violation_in_dossier_blocks_claims() -> None:
    counterexamples = discover_counterexamples(
        candidate_id="candidate",
        cases=({"case_id": "outside", "domain_valid": False},),
    )

    dossier = build_falsification_dossier(
        candidate_id="candidate",
        residual_diagnostics=evaluate_residual_diagnostics(
            candidate_id="candidate",
            residuals=(0.0, 0.1, -0.1, 0.0),
        ),
        counterexample_result=counterexamples,
    )

    assert dossier.status == "blocked"
    assert dossier.claim_effect == "block_claim"
    assert "domain_violation" in dossier.reason_codes
