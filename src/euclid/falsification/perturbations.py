from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

from euclid.falsification._identity import replay_identity, stable_float, unique_codes


@dataclass(frozen=True)
class PerturbationStabilityResult:
    candidate_id: str
    status: str
    reason_codes: tuple[str, ...]
    claim_effect: str
    retained_rate: float | None
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "perturbation_stability@1.0.0",
            "candidate_id": self.candidate_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "claim_effect": self.claim_effect,
            "retained_rate": self.retained_rate,
            "replay_identity": self.replay_identity,
        }


def evaluate_perturbation_stability(
    *,
    candidate_id: str,
    perturbation_runs: Sequence[Mapping[str, Any]],
    min_retained_rate: float = 0.8,
) -> PerturbationStabilityResult:
    retained = [
        1.0 if bool(run.get("canonical_form_matches", False)) else 0.0
        for run in perturbation_runs
    ]
    if not retained:
        return _result(
            candidate_id=candidate_id,
            status="abstained",
            reason_codes=("missing_perturbation_runs",),
            retained_rate=None,
            claim_effect="block_claim",
        )
    retained_rate = stable_float(fmean(retained))
    failed = retained_rate < float(min_retained_rate)
    return _result(
        candidate_id=candidate_id,
        status="failed" if failed else "passed",
        reason_codes=("perturbation_instability",) if failed else (),
        retained_rate=retained_rate,
        claim_effect="downgrade_predictive_claim" if failed else "allow_claim",
    )


def _result(
    *,
    candidate_id: str,
    status: str,
    reason_codes: Sequence[str],
    retained_rate: float | None,
    claim_effect: str,
) -> PerturbationStabilityResult:
    payload = {
        "candidate_id": candidate_id,
        "claim_effect": claim_effect,
        "reason_codes": list(reason_codes),
        "retained_rate": retained_rate,
        "status": status,
    }
    return PerturbationStabilityResult(
        candidate_id=str(candidate_id),
        status=status,
        reason_codes=unique_codes(reason_codes),
        claim_effect=claim_effect,
        retained_rate=retained_rate,
        replay_identity=replay_identity("perturbation-stability", payload),
    )


__all__ = ["PerturbationStabilityResult", "evaluate_perturbation_stability"]
