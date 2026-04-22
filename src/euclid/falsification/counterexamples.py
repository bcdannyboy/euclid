from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from euclid.falsification._identity import replay_identity, stable_float, unique_codes


@dataclass(frozen=True)
class CounterexampleSearchResult:
    candidate_id: str
    status: str
    reason_codes: tuple[str, ...]
    claim_effect: str
    counterexamples: tuple[Mapping[str, Any], ...]
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "counterexample_report@1.0.0",
            "candidate_id": self.candidate_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "claim_effect": self.claim_effect,
            "counterexamples": [dict(item) for item in self.counterexamples],
            "replay_identity": self.replay_identity,
        }


def discover_counterexamples(
    *,
    candidate_id: str,
    cases: Sequence[Mapping[str, Any]],
    error_tolerance: float = 1.0,
    max_extrapolation_distance: float = 1.0,
) -> CounterexampleSearchResult:
    reasons: list[str] = []
    counterexamples: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_id = str(case.get("case_id", f"case_{index}"))
        domain_valid = bool(case.get("domain_valid", True))
        if not domain_valid:
            reasons.append("domain_violation")
            counterexamples.append(
                {
                    "case_id": case_id,
                    "reason_code": "domain_violation",
                }
            )
            continue
        prediction = _optional_float(case.get("prediction"))
        observed = _optional_float(case.get("observed"))
        if prediction is None or observed is None:
            continue
        absolute_error = abs(prediction - observed)
        extrapolation_distance = _optional_float(
            case.get("extrapolation_distance", 0.0)
        ) or 0.0
        case_reasons: list[str] = []
        if absolute_error > float(error_tolerance):
            case_reasons.append("counterexample_discovered")
        if (
            extrapolation_distance > float(max_extrapolation_distance)
            and absolute_error > float(error_tolerance)
        ):
            case_reasons.append("extrapolation_failure")
        if case_reasons:
            reasons.extend(case_reasons)
            counterexamples.append(
                {
                    "case_id": case_id,
                    "absolute_error": stable_float(absolute_error),
                    "extrapolation_distance": stable_float(extrapolation_distance),
                    "reason_codes": case_reasons,
                }
            )

    reason_codes = unique_codes(reasons)
    claim_effect = (
        "block_claim"
        if "domain_violation" in reason_codes
        else "downgrade_predictive_claim"
        if reason_codes
        else "allow_claim"
    )
    status = "failed" if reason_codes else "passed"
    payload = {
        "candidate_id": candidate_id,
        "claim_effect": claim_effect,
        "counterexamples": counterexamples,
        "reason_codes": list(reason_codes),
        "status": status,
    }
    return CounterexampleSearchResult(
        candidate_id=str(candidate_id),
        status=status,
        reason_codes=reason_codes,
        claim_effect=claim_effect,
        counterexamples=tuple(counterexamples),
        replay_identity=replay_identity("counterexamples", payload),
    )


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


__all__ = ["CounterexampleSearchResult", "discover_counterexamples"]
