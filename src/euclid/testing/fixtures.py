from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixtureProvenance:
    source_kind: str
    source_ref: str
    license: str


@dataclass(frozen=True)
class FixtureGateResult:
    gate_id: str
    status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class FixtureGate:
    gate_id: str
    provenance: FixtureProvenance
    edge_cases: tuple[str, ...]
    regression_reason: str

    def validate(self) -> FixtureGateResult:
        reason_codes: list[str] = []
        if not self.provenance.source_kind.strip():
            reason_codes.append("missing_fixture_source_kind")
        if not self.provenance.source_ref.strip():
            reason_codes.append("missing_fixture_source_ref")
        if not self.provenance.license.strip():
            reason_codes.append("missing_fixture_license")
        if not self.edge_cases:
            reason_codes.append("missing_edge_case_coverage")
        if not self.regression_reason.strip():
            reason_codes.append("missing_regression_reason")
        if reason_codes:
            raise ValueError(
                f"{self.gate_id} fixture gate failed: {', '.join(reason_codes)}"
            )
        return FixtureGateResult(
            gate_id=self.gate_id,
            status="passed",
            reason_codes=(),
        )


__all__ = ["FixtureGate", "FixtureGateResult", "FixtureProvenance"]
