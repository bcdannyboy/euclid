from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from euclid.cir.models import CandidateIntermediateRepresentation
from euclid.contracts.errors import ContractValidationError
from euclid.runtime.hashing import sha256_digest

_CLAIM_BOUNDARY = {
    "claim_publication_allowed": False,
    "reason_codes": ["search_engine_not_claim_authority"],
}
_RUN_STATUSES = frozenset({"completed", "partial", "failed", "timeout"})


def engine_claim_boundary() -> dict[str, Any]:
    return {
        "claim_publication_allowed": False,
        "reason_codes": ["search_engine_not_claim_authority"],
    }


def _require_non_empty(value: str, *, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(
            code="invalid_engine_contract",
            message=f"{field_path} must be a non-empty string",
            field_path=field_path,
        )
    return value.strip()


def _require_claim_boundary(claim_boundary: Mapping[str, Any]) -> dict[str, Any]:
    if bool(claim_boundary.get("claim_publication_allowed")):
        raise ContractValidationError(
            code="direct_claim_publication_forbidden",
            message="search engines cannot publish claims directly",
            field_path="claim_boundary.claim_publication_allowed",
        )
    reason_codes = tuple(claim_boundary.get("reason_codes", ()))
    if "search_engine_not_claim_authority" not in reason_codes:
        raise ContractValidationError(
            code="missing_engine_claim_boundary",
            message="search engine outputs must declare they are not claim authority",
            field_path="claim_boundary.reason_codes",
        )
    return {
        "claim_publication_allowed": False,
        "reason_codes": list(reason_codes),
    }


@dataclass(frozen=True)
class RowsFeaturesAccess:
    row_count: int
    feature_names: tuple[str, ...]
    row_fingerprints: tuple[dict[str, Any], ...]
    first_event_time: str | None = None
    last_event_time: str | None = None

    @classmethod
    def from_rows(
        cls,
        *,
        rows: Sequence[Mapping[str, Any]],
        feature_names: Sequence[str],
    ) -> "RowsFeaturesAccess":
        fingerprints = tuple(_row_fingerprint(row) for row in rows)
        return cls(
            row_count=len(rows),
            feature_names=tuple(str(name) for name in feature_names),
            row_fingerprints=fingerprints,
            first_event_time=None if not fingerprints else fingerprints[0]["event_time"],
            last_event_time=None if not fingerprints else fingerprints[-1]["event_time"],
        )


@dataclass(frozen=True)
class EngineInputContext:
    search_plan_id: str
    search_class: str
    random_seed: str
    proposal_limit: int
    frontier_axes: tuple[str, ...]
    rows_features_access: RowsFeaturesAccess
    timeout_seconds: float
    engine_ids: tuple[str, ...]
    runtime_search_plan: Any = field(default=None, repr=False, compare=False)
    runtime_feature_view: Any = field(default=None, repr=False, compare=False)
    runtime_rows: tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    allowed_candidate_ids: tuple[str, ...] = ()
    claim_boundary: Mapping[str, Any] = field(default_factory=engine_claim_boundary)

    def __post_init__(self) -> None:
        _require_non_empty(self.search_plan_id, field_path="search_plan_id")
        _require_non_empty(self.search_class, field_path="search_class")
        _require_non_empty(str(self.random_seed), field_path="random_seed")
        if self.proposal_limit < 0:
            raise ContractValidationError(
                code="invalid_engine_contract",
                message="proposal_limit must be non-negative",
                field_path="proposal_limit",
            )
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise ContractValidationError(
                code="invalid_engine_contract",
                message="timeout_seconds must be positive and finite",
                field_path="timeout_seconds",
            )
        object.__setattr__(self, "random_seed", str(self.random_seed))
        object.__setattr__(self, "frontier_axes", tuple(self.frontier_axes))
        object.__setattr__(self, "engine_ids", tuple(self.engine_ids))
        object.__setattr__(
            self,
            "allowed_candidate_ids",
            tuple(str(candidate_id) for candidate_id in self.allowed_candidate_ids),
        )
        object.__setattr__(self, "runtime_rows", tuple(self.runtime_rows))
        object.__setattr__(
            self,
            "claim_boundary",
            _require_claim_boundary(self.claim_boundary),
        )

    @property
    def row_count(self) -> int:
        return self.rows_features_access.row_count

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.rows_features_access.feature_names

    def replay_metadata(self) -> dict[str, Any]:
        return {
            "search_plan_id": self.search_plan_id,
            "search_class": self.search_class,
            "random_seed": self.random_seed,
            "proposal_limit": self.proposal_limit,
            "frontier_axes": list(self.frontier_axes),
            "engine_ids": list(self.engine_ids),
            "feature_names": list(self.feature_names),
            "row_count": self.row_count,
            "row_fingerprints": list(self.rows_features_access.row_fingerprints),
        }


@dataclass(frozen=True)
class EngineFailureDiagnostic:
    engine_id: str
    reason_code: str
    message: str
    recoverable: bool
    candidate_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "engine_id",
            _require_non_empty(self.engine_id, field_path="engine_id"),
        )
        object.__setattr__(
            self,
            "reason_code",
            _require_non_empty(self.reason_code, field_path="reason_code"),
        )
        object.__setattr__(
            self,
            "message",
            _require_non_empty(self.message, field_path="message"),
        )
        object.__setattr__(self, "details", dict(self.details))

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "engine_id": self.engine_id,
            "reason_code": self.reason_code,
            "message": self.message,
            "recoverable": self.recoverable,
            "details": dict(self.details),
        }
        if self.candidate_id is not None:
            payload["candidate_id"] = self.candidate_id
        return payload


@dataclass(frozen=True)
class EngineCandidateRecord:
    candidate_id: str
    engine_id: str
    engine_version: str
    search_class: str
    search_space_declaration: str
    budget_declaration: Mapping[str, Any]
    rows_used: tuple[str, ...]
    features_used: tuple[str, ...]
    random_seed: str
    candidate_trace: Mapping[str, Any]
    omission_disclosure: Mapping[str, Any]
    claim_boundary: Mapping[str, Any]
    proposed_cir: CandidateIntermediateRepresentation | None = None
    lowering_kind: str = "proposed_cir"
    lowerable_payload: Mapping[str, Any] = field(default_factory=dict)
    published_claim_payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_id",
            "engine_id",
            "engine_version",
            "search_class",
            "search_space_declaration",
            "lowering_kind",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_non_empty(getattr(self, field_name), field_path=field_name),
            )
        if self.published_claim_payload is not None:
            raise ContractValidationError(
                code="direct_claim_publication_forbidden",
                message="search engines cannot attach published claim payloads",
                field_path="published_claim_payload",
            )
        object.__setattr__(self, "rows_used", tuple(self.rows_used))
        object.__setattr__(self, "features_used", tuple(self.features_used))
        object.__setattr__(self, "random_seed", str(self.random_seed))
        object.__setattr__(self, "budget_declaration", dict(self.budget_declaration))
        object.__setattr__(self, "candidate_trace", dict(self.candidate_trace))
        object.__setattr__(self, "omission_disclosure", dict(self.omission_disclosure))
        object.__setattr__(self, "lowerable_payload", dict(self.lowerable_payload))
        object.__setattr__(
            self,
            "claim_boundary",
            _require_claim_boundary(self.claim_boundary),
        )

    def replay_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "engine_id": self.engine_id,
            "engine_version": self.engine_version,
            "search_class": self.search_class,
            "lowering_kind": self.lowering_kind,
            "budget_declaration": dict(self.budget_declaration),
            "rows_used": list(self.rows_used),
            "features_used": list(self.features_used),
            "random_seed": self.random_seed,
            "candidate_trace": dict(self.candidate_trace),
            "omission_disclosure": dict(self.omission_disclosure),
        }


@dataclass(frozen=True)
class EngineRunResult:
    engine_id: str
    engine_version: str
    status: str
    candidates: tuple[EngineCandidateRecord, ...]
    failure_diagnostics: tuple[EngineFailureDiagnostic, ...]
    trace: Mapping[str, Any]
    omission_disclosure: Mapping[str, Any]
    replay_metadata: Mapping[str, Any]
    claim_boundary: Mapping[str, Any] = field(default_factory=engine_claim_boundary)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "engine_id",
            _require_non_empty(self.engine_id, field_path="engine_id"),
        )
        object.__setattr__(
            self,
            "engine_version",
            _require_non_empty(self.engine_version, field_path="engine_version"),
        )
        if self.status not in _RUN_STATUSES:
            raise ContractValidationError(
                code="invalid_engine_run_status",
                message="engine run status must be completed, partial, failed, or timeout",
                field_path="status",
                details={"status": self.status},
            )
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "failure_diagnostics", tuple(self.failure_diagnostics))
        object.__setattr__(self, "trace", dict(self.trace))
        object.__setattr__(self, "omission_disclosure", dict(self.omission_disclosure))
        object.__setattr__(self, "replay_metadata", dict(self.replay_metadata))
        object.__setattr__(
            self,
            "claim_boundary",
            _require_claim_boundary(self.claim_boundary),
        )

    def replay_identity(self) -> str:
        return sha256_digest(
            {
                "engine_id": self.engine_id,
                "engine_version": self.engine_version,
                "status": self.status,
                "candidates": [
                    candidate.replay_payload() for candidate in self.candidates
                ],
                "failures": [
                    diagnostic.as_dict() for diagnostic in self.failure_diagnostics
                ],
                "omission_disclosure": dict(self.omission_disclosure),
                "replay_metadata": dict(self.replay_metadata),
            }
        )


class SearchEngine(Protocol):
    engine_id: str
    engine_version: str

    def run(self, context: EngineInputContext) -> EngineRunResult:
        ...


def _row_fingerprint(row: Mapping[str, Any]) -> dict[str, Any]:
    event_time = str(row.get("event_time", ""))
    if not event_time:
        raise ContractValidationError(
            code="invalid_engine_rows",
            message="engine rows require event_time",
            field_path="rows.event_time",
        )
    return {
        "event_time": event_time,
        "available_at": str(row.get("available_at", event_time)),
        "row_hash": sha256_digest(
            {
                key: value
                for key, value in sorted(row.items())
                if key not in {"raw_payload", "provider_headers"}
            }
        ),
    }


__all__ = [
    "EngineCandidateRecord",
    "EngineFailureDiagnostic",
    "EngineInputContext",
    "EngineRunResult",
    "RowsFeaturesAccess",
    "SearchEngine",
    "engine_claim_boundary",
]
