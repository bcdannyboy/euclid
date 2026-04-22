from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from euclid.cir.models import CandidateIntermediateRepresentation
from euclid.contracts.errors import ContractValidationError
from euclid.math.observation_models import PointObservationModel
from euclid.modules.features import FeatureView
from euclid.search.engine_contracts import (
    EngineCandidateRecord,
    EngineFailureDiagnostic,
    EngineInputContext,
    EngineRunResult,
    RowsFeaturesAccess,
    SearchEngine,
    engine_claim_boundary,
)
from euclid.runtime.hashing import sha256_digest


@dataclass(frozen=True)
class OrchestratedSearchResult:
    accepted_candidates: tuple[CandidateIntermediateRepresentation, ...]
    engine_runs: Mapping[str, EngineRunResult]
    failure_diagnostics: tuple[EngineFailureDiagnostic, ...]
    duplicate_diagnostics: tuple[EngineFailureDiagnostic, ...]
    replay_metadata: Mapping[str, Any]
    replay_identity: str
    claim_boundary: Mapping[str, Any] = field(default_factory=engine_claim_boundary)


class EngineRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, SearchEngine] = {}

    def register(self, engine: SearchEngine) -> None:
        if engine.engine_id in self._engines:
            raise ContractValidationError(
                code="duplicate_search_engine",
                message="search engine registry cannot contain duplicate engine IDs",
                field_path="engine.engine_id",
                details={"engine_id": engine.engine_id},
            )
        self._engines[engine.engine_id] = engine

    def resolve(self, engine_id: str) -> SearchEngine:
        try:
            return self._engines[engine_id]
        except KeyError as exc:
            raise ContractValidationError(
                code="unknown_search_engine",
                message=f"unknown search engine {engine_id!r}",
                field_path="engine_id",
            ) from exc

    def engines(self) -> tuple[SearchEngine, ...]:
        return tuple(self._engines[engine_id] for engine_id in sorted(self._engines))


class SearchEngineOrchestrator:
    @staticmethod
    def context_from_rows(
        *,
        search_plan_id: str,
        search_class: str,
        random_seed: str,
        proposal_limit: int,
        frontier_axes: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        feature_names: Sequence[str],
        timeout_seconds: float,
        engine_ids: Sequence[str],
        allowed_candidate_ids: Sequence[str] = (),
        runtime_search_plan: Any = None,
        runtime_feature_view: FeatureView | None = None,
    ) -> EngineInputContext:
        return EngineInputContext(
            search_plan_id=search_plan_id,
            search_class=search_class,
            random_seed=random_seed,
            proposal_limit=proposal_limit,
            frontier_axes=tuple(frontier_axes),
            rows_features_access=RowsFeaturesAccess.from_rows(
                rows=rows,
                feature_names=feature_names,
            ),
            timeout_seconds=timeout_seconds,
            engine_ids=tuple(engine_ids),
            runtime_search_plan=runtime_search_plan,
            runtime_feature_view=runtime_feature_view,
            runtime_rows=tuple(rows),
            allowed_candidate_ids=tuple(allowed_candidate_ids),
        )

    def run(
        self,
        *,
        context: EngineInputContext,
        engines: Sequence[SearchEngine],
    ) -> OrchestratedSearchResult:
        engine_runs: dict[str, EngineRunResult] = {}
        failures: list[EngineFailureDiagnostic] = []
        duplicates: list[EngineFailureDiagnostic] = []
        accepted: list[CandidateIntermediateRepresentation] = []
        seen_hashes: set[str] = set()

        for engine in engines:
            run_result = self._run_engine(context=context, engine=engine)
            engine_runs[engine.engine_id] = run_result
            failures.extend(run_result.failure_diagnostics)
            for record in run_result.candidates:
                try:
                    candidate = self._lower_candidate(context=context, record=record)
                except ContractValidationError as exc:
                    failures.append(
                        EngineFailureDiagnostic(
                            engine_id=record.engine_id,
                            candidate_id=record.candidate_id,
                            reason_code="failed_lowering",
                            message=exc.message,
                            recoverable=True,
                            details={"error_code": exc.code, "field_path": exc.field_path},
                        )
                    )
                    continue
                candidate_hash = candidate.canonical_hash()
                if candidate_hash in seen_hashes:
                    duplicates.append(
                        EngineFailureDiagnostic(
                            engine_id=record.engine_id,
                            candidate_id=record.candidate_id,
                            reason_code="duplicate_canonical_output",
                            message="engine candidate lowered to an existing canonical CIR hash",
                            recoverable=True,
                            details={"candidate_hash": candidate_hash},
                        )
                    )
                    continue
                seen_hashes.add(candidate_hash)
                accepted.append(candidate)

        replay_metadata = {
            "context": context.replay_metadata(),
            "engine_runs": {
                engine_id: run.replay_identity()
                for engine_id, run in sorted(engine_runs.items())
            },
            "accepted_candidate_hashes": [
                candidate.canonical_hash() for candidate in accepted
            ],
            "failure_diagnostics": [failure.as_dict() for failure in failures],
            "duplicate_diagnostics": [failure.as_dict() for failure in duplicates],
        }
        return OrchestratedSearchResult(
            accepted_candidates=tuple(accepted),
            engine_runs=engine_runs,
            failure_diagnostics=tuple(failures),
            duplicate_diagnostics=tuple(duplicates),
            replay_metadata=replay_metadata,
            replay_identity=sha256_digest(replay_metadata),
            claim_boundary=engine_claim_boundary(),
        )

    def _run_engine(
        self,
        *,
        context: EngineInputContext,
        engine: SearchEngine,
    ) -> EngineRunResult:
        started = time.monotonic()
        try:
            raw_result = engine.run(context)
        except Exception as exc:
            return EngineRunResult(
                engine_id=engine.engine_id,
                engine_version=getattr(engine, "engine_version", "unknown"),
                status="failed",
                candidates=(),
                failure_diagnostics=(
                    EngineFailureDiagnostic(
                        engine_id=engine.engine_id,
                        reason_code="engine_crash",
                        message=type(exc).__name__,
                        recoverable=True,
                    ),
                ),
                trace={"exception_type": type(exc).__name__},
                omission_disclosure={"omitted_due_to_crash": True},
                replay_metadata=context.replay_metadata(),
            )
        elapsed = time.monotonic() - started
        if not isinstance(raw_result, EngineRunResult):
            return EngineRunResult(
                engine_id=engine.engine_id,
                engine_version=getattr(engine, "engine_version", "unknown"),
                status="failed",
                candidates=(),
                failure_diagnostics=(
                    EngineFailureDiagnostic(
                        engine_id=engine.engine_id,
                        reason_code="malformed_engine_result",
                        message="engine returned a non-EngineRunResult payload",
                        recoverable=True,
                    ),
                ),
                trace={"result_type": type(raw_result).__name__},
                omission_disclosure={"omitted_due_to_malformed_result": True},
                replay_metadata=context.replay_metadata(),
            )
        if elapsed > context.timeout_seconds:
            timeout_failure = EngineFailureDiagnostic(
                engine_id=engine.engine_id,
                reason_code="engine_timeout",
                message="engine exceeded its wall-clock timeout",
                recoverable=True,
                details={
                    "timeout_seconds": context.timeout_seconds,
                    "elapsed_seconds": round(elapsed, 9),
                },
            )
            return EngineRunResult(
                engine_id=raw_result.engine_id,
                engine_version=raw_result.engine_version,
                status="timeout",
                candidates=raw_result.candidates,
                failure_diagnostics=(
                    *raw_result.failure_diagnostics,
                    timeout_failure,
                ),
                trace={**dict(raw_result.trace), "timed_out": True},
                omission_disclosure={
                    **dict(raw_result.omission_disclosure),
                    "omitted_due_to_timeout": True,
                },
                replay_metadata=raw_result.replay_metadata,
            )
        return raw_result

    def _lower_candidate(
        self,
        *,
        context: EngineInputContext,
        record: EngineCandidateRecord,
    ) -> CandidateIntermediateRepresentation:
        if record.proposed_cir is not None:
            return record.proposed_cir
        if record.lowering_kind == "descriptive_proposal":
            return _lower_descriptive_proposal(context=context, record=record)
        raise ContractValidationError(
            code="unsupported_engine_lowering",
            message=f"unsupported engine lowering kind {record.lowering_kind!r}",
            field_path="candidate.lowering_kind",
        )


def run_search_engines(
    *,
    context: EngineInputContext,
    engines: Sequence[SearchEngine],
) -> OrchestratedSearchResult:
    return SearchEngineOrchestrator().run(context=context, engines=engines)


def _lower_descriptive_proposal(
    *,
    context: EngineInputContext,
    record: EngineCandidateRecord,
) -> CandidateIntermediateRepresentation:
    if context.runtime_search_plan is None or context.runtime_feature_view is None:
        raise ContractValidationError(
            code="missing_engine_lowering_context",
            message="descriptive proposal lowering requires runtime search plan and feature view",
            field_path="context.runtime_search_plan",
        )
    from euclid.search import backends
    from euclid.reducers.models import BoundObservationModel

    proposal = backends.DescriptiveSearchProposal(**dict(record.lowerable_payload))
    adapters = {adapter.family_id: adapter for adapter in backends._default_adapters()}
    adapter = adapters.get(proposal.primitive_family)
    if adapter is None:
        raise ContractValidationError(
            code="unknown_engine_proposal_family",
            message="descriptive proposal family has no native lowering adapter",
            field_path="candidate.lowerable_payload.primitive_family",
        )
    try:
        return adapter.realize_proposal(
            proposal=proposal,
            proposal_rank=int(record.candidate_trace.get("proposal_rank", 0)),
            search_plan=context.runtime_search_plan,
            feature_view=context.runtime_feature_view,
            observation_model=BoundObservationModel.from_runtime(PointObservationModel()),
            coverage_disclosures=backends._coverage_disclosures(
                search_plan=context.runtime_search_plan,
                canonical_program_count=int(
                    record.omission_disclosure.get("canonical_program_count", 0)
                ),
                restart_count_used=int(
                    record.omission_disclosure.get("restart_count_used", 0)
                ),
            ),
        )
    except Exception as exc:
        if isinstance(exc, ContractValidationError):
            raise
        raise ContractValidationError(
            code="engine_descriptive_lowering_failed",
            message=type(exc).__name__,
            field_path="candidate.lowerable_payload",
        ) from exc


__all__ = [
    "EngineRegistry",
    "OrchestratedSearchResult",
    "SearchEngineOrchestrator",
    "run_search_engines",
]
