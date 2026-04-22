from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from euclid.falsification._identity import replay_identity, unique_codes


@dataclass(frozen=True)
class FalsificationDossier:
    candidate_id: str
    status: str
    reason_codes: tuple[str, ...]
    claim_effect: str
    residual_diagnostics: Mapping[str, Any] | None
    counterexamples: Mapping[str, Any] | None
    parameter_stability: Mapping[str, Any] | None
    transport_status: str
    calibration_status: str
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "falsification_dossier@1.0.0",
            "candidate_id": self.candidate_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "claim_effect": self.claim_effect,
            "residual_diagnostics": self.residual_diagnostics,
            "counterexamples": self.counterexamples,
            "parameter_stability": self.parameter_stability,
            "transport_status": self.transport_status,
            "calibration_status": self.calibration_status,
            "replay_identity": self.replay_identity,
        }


def build_falsification_dossier(
    *,
    candidate_id: str,
    residual_diagnostics: Any | None = None,
    counterexample_result: Any | None = None,
    parameter_stability: Any | None = None,
    transport_status: str = "not_requested",
    calibration_status: str = "not_applicable_for_forecast_type",
) -> FalsificationDossier:
    residual_manifest = _manifest_or_none(residual_diagnostics)
    counterexample_manifest = _manifest_or_none(counterexample_result)
    stability_manifest = _manifest_or_none(parameter_stability)

    reason_codes: list[str] = []
    claim_effects: list[str] = []
    for manifest in (residual_manifest, counterexample_manifest, stability_manifest):
        if not isinstance(manifest, Mapping):
            continue
        reason_codes.extend(str(code) for code in manifest.get("reason_codes", ()))
        claim_effects.append(str(manifest.get("claim_effect", "allow_claim")))
    if transport_status == "failed":
        reason_codes.append("transport_failed")
        claim_effects.append("downgrade_predictive_claim")
    if calibration_status in {"failed", "coverage_failed", "poor_coverage"}:
        reason_codes.append("stochastic_miscalibration")
        claim_effects.append("downgrade_predictive_claim")

    resolved_reasons = unique_codes(reason_codes)
    if "block_claim" in claim_effects or "domain_violation" in resolved_reasons:
        status = "blocked"
        claim_effect = "block_claim"
    elif resolved_reasons:
        status = "failed"
        claim_effect = "downgrade_predictive_claim"
    else:
        status = "passed"
        claim_effect = "allow_claim"

    payload = {
        "calibration_status": calibration_status,
        "candidate_id": candidate_id,
        "claim_effect": claim_effect,
        "counterexamples": counterexample_manifest,
        "parameter_stability": stability_manifest,
        "reason_codes": list(resolved_reasons),
        "residual_diagnostics": residual_manifest,
        "status": status,
        "transport_status": transport_status,
    }
    return FalsificationDossier(
        candidate_id=str(candidate_id),
        status=status,
        reason_codes=resolved_reasons,
        claim_effect=claim_effect,
        residual_diagnostics=residual_manifest,
        counterexamples=counterexample_manifest,
        parameter_stability=stability_manifest,
        transport_status=str(transport_status),
        calibration_status=str(calibration_status),
        replay_identity=replay_identity("falsification-dossier", payload),
    )


def _manifest_or_none(value: Any | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "as_manifest"):
        manifest = value.as_manifest()
        if isinstance(manifest, Mapping):
            return manifest
    if isinstance(value, Mapping):
        return value
    return None


__all__ = ["FalsificationDossier", "build_falsification_dossier"]
