from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from euclid.cir.models import (
    CandidateIntermediateRepresentation,
    CIRBackendOriginRecord,
    CIRModelCodeDecomposition,
    CIRReplayHook,
    CIRReplayHooks,
)
from euclid.cir.normalize import build_cir_candidate_from_expression
from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import (
    DistributionParameter,
    NoiseTerm,
    Parameter,
    State,
    walk_expression,
)
from euclid.expr.serialization import expression_from_dict
from euclid.rewrites.egglog_runner import (
    EqualitySaturationConfig,
    EqualitySaturationResult,
    run_equality_saturation,
)
from euclid.rewrites.extraction import expression_cost
from euclid.search.engine_contracts import (
    EngineCandidateRecord,
    EngineFailureDiagnostic,
    EngineInputContext,
    EngineRunResult,
    SearchEngine,
    engine_claim_boundary,
)


@dataclass(frozen=True)
class EGraphEngineConfig:
    max_iterations: int = 16
    node_limit: int = 256
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.max_iterations < 0:
            raise ContractValidationError(
                code="invalid_egraph_config",
                message="max_iterations must be non-negative",
                field_path="max_iterations",
            )
        if self.node_limit <= 0:
            raise ContractValidationError(
                code="invalid_egraph_config",
                message="node_limit must be positive",
                field_path="node_limit",
            )
        if self.timeout_seconds <= 0 or not math.isfinite(float(self.timeout_seconds)):
            raise ContractValidationError(
                code="invalid_egraph_config",
                message="timeout_seconds must be positive and finite",
                field_path="timeout_seconds",
            )

    def saturation_config(
        self,
        context: EngineInputContext,
    ) -> EqualitySaturationConfig:
        return EqualitySaturationConfig(
            max_iterations=self.max_iterations,
            node_limit=self.node_limit,
            timeout_seconds=min(self.timeout_seconds, context.timeout_seconds),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "node_limit": self.node_limit,
            "timeout_seconds": self.timeout_seconds,
        }


class RewriteCandidateProvider(Protocol):
    def candidates(
        self,
        context: EngineInputContext,
    ) -> tuple[CandidateIntermediateRepresentation, ...]:
        ...


@dataclass(frozen=True)
class StaticRewriteCandidateProvider:
    source_candidates: tuple[CandidateIntermediateRepresentation, ...]

    def __init__(
        self,
        source_candidates: Sequence[CandidateIntermediateRepresentation],
    ) -> None:
        object.__setattr__(self, "source_candidates", tuple(source_candidates))

    def candidates(
        self,
        context: EngineInputContext,
    ) -> tuple[CandidateIntermediateRepresentation, ...]:
        return self.source_candidates


class NativeFragmentRewriteCandidateProvider:
    def candidates(
        self,
        context: EngineInputContext,
    ) -> tuple[CandidateIntermediateRepresentation, ...]:
        search_plan = context.runtime_search_plan
        feature_view = context.runtime_feature_view
        if search_plan is None or feature_view is None:
            return ()
        if hasattr(search_plan, "model_copy"):
            source_plan = search_plan.model_copy(
                update={
                    "search_class": "bounded_heuristic",
                    "proposal_limit": max(
                        int(getattr(search_plan, "proposal_limit", 1)),
                        int(context.proposal_limit),
                    ),
                }
            )
        else:
            source_plan = search_plan
        from euclid.search import backends

        result = backends.run_descriptive_search_backends(
            search_plan=source_plan,
            feature_view=feature_view,
            include_default_grammar=True,
        )
        return tuple(result.accepted_candidates)


class EGraphEngine(SearchEngine):
    engine_id = "egraph-engine-v1"
    engine_version = "1.0"

    def __init__(
        self,
        *,
        config: EGraphEngineConfig | None = None,
        candidate_provider: RewriteCandidateProvider | None = None,
    ) -> None:
        self.config = config or EGraphEngineConfig()
        self.candidate_provider = (
            candidate_provider or NativeFragmentRewriteCandidateProvider()
        )

    def run(self, context: EngineInputContext) -> EngineRunResult:
        source_candidates = self.candidate_provider.candidates(context)
        failures: list[EngineFailureDiagnostic] = []
        records: list[EngineCandidateRecord] = []
        partial = False

        for rank, source_candidate in enumerate(
            source_candidates[: context.proposal_limit]
        ):
            try:
                record, saturation = self._record_from_candidate(
                    context=context,
                    source_candidate=source_candidate,
                    proposal_rank=rank,
                )
            except ContractValidationError as exc:
                failures.append(
                    EngineFailureDiagnostic(
                        engine_id=self.engine_id,
                        reason_code="rewrite_lowering_failed",
                        message=exc.message,
                        recoverable=True,
                        details={
                            "error_code": exc.code,
                            "field_path": exc.field_path,
                            **dict(exc.details),
                        },
                    )
                )
                continue
            records.append(record)
            if saturation.status != "completed":
                partial = True

        status = "partial" if partial and records else "completed"
        if not records and failures:
            status = "failed"
        return EngineRunResult(
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            status=status,
            candidates=tuple(records),
            failure_diagnostics=tuple(failures),
            trace={
                "engine_config": self.config.as_dict(),
                "source_candidate_count": len(source_candidates),
                "rewritten_candidate_count": len(records),
                "rewrite_engine": "equality_saturation_over_expression_cir",
            },
            omission_disclosure={
                "source_candidate_count": len(source_candidates),
                "omitted_by_proposal_limit": max(
                    len(source_candidates) - context.proposal_limit,
                    0,
                ),
            },
            replay_metadata={
                **context.replay_metadata(),
                "engine_id": self.engine_id,
                "engine_version": self.engine_version,
                "engine_config": self.config.as_dict(),
            },
            claim_boundary=engine_claim_boundary(),
        )

    def _record_from_candidate(
        self,
        *,
        context: EngineInputContext,
        source_candidate: CandidateIntermediateRepresentation,
        proposal_rank: int,
    ) -> tuple[EngineCandidateRecord, EqualitySaturationResult]:
        expression_payload = source_candidate.structural_layer.expression_payload
        if expression_payload is None:
            raise ContractValidationError(
                code="unsupported_rewrite_candidate",
                message="e-graph engine can rewrite only expression CIR candidates",
                field_path="source_candidate.expression_payload",
            )
        expression = expression_from_dict(expression_payload.expression_tree)
        saturation = run_equality_saturation(
            expression,
            assumptions=expression_payload.assumptions,
            config=self.config.saturation_config(context),
        )
        rewritten_candidate = _build_rewritten_candidate(
            source_candidate=source_candidate,
            rewritten_expression=saturation.best_expression,
            saturation=saturation,
            context=context,
            proposal_rank=proposal_rank,
            engine_id=self.engine_id,
        )
        candidate_id = (
            f"egraph-{context.search_plan_id}-{proposal_rank:02d}-"
            f"{saturation.extraction.best_expression_hash.removeprefix('sha256:')[:12]}"
        )
        trace = {
            "source_candidate_hash": source_candidate.canonical_hash(),
            "source_expression_hash": expression_payload.expression_canonical_hash,
            "rewrite_trace": saturation.as_dict(),
            "claim_boundary": engine_claim_boundary(),
        }
        return (
            EngineCandidateRecord(
                candidate_id=candidate_id,
                engine_id=self.engine_id,
                engine_version=self.engine_version,
                search_class=context.search_class,
                search_space_declaration="egraph_expression_rewrite_space_v1",
                budget_declaration={
                    "proposal_limit": context.proposal_limit,
                    "timeout_seconds": context.timeout_seconds,
                    "max_iterations": self.config.max_iterations,
                    "node_limit": self.config.node_limit,
                },
                rows_used=tuple(
                    row["event_time"]
                    for row in context.rows_features_access.row_fingerprints
                ),
                features_used=tuple(expression_payload.feature_dependencies),
                random_seed=context.random_seed,
                candidate_trace=trace,
                omission_disclosure=dict(saturation.omission_disclosure),
                claim_boundary=engine_claim_boundary(),
                proposed_cir=rewritten_candidate,
                lowering_kind="proposed_cir",
            ),
            saturation,
        )


def _build_rewritten_candidate(
    *,
    source_candidate: CandidateIntermediateRepresentation,
    rewritten_expression,
    saturation: EqualitySaturationResult,
    context: EngineInputContext,
    proposal_rank: int,
    engine_id: str,
) -> CandidateIntermediateRepresentation:
    source_payload = source_candidate.structural_layer.expression_payload
    assert source_payload is not None
    origin = source_candidate.evidence_layer.backend_origin_record
    diagnostics = {
        "rewrite_trace": saturation.as_dict(),
        "source_candidate_hash": source_candidate.canonical_hash(),
        "source_backend_origin": origin.as_dict(),
        "production_evidence_kind": "rewrite_equivalence_evidence_only",
        "claim_publication_allowed": False,
    }
    return build_cir_candidate_from_expression(
        expression=rewritten_expression,
        cir_family_id=source_candidate.structural_layer.cir_family_id,
        cir_form_class="egraph_rewrite_expression_ir",
        input_signature=source_candidate.structural_layer.input_signature,
        forecast_operator=source_candidate.execution_layer.forecast_operator,
        model_code_decomposition=_model_code_for_expression(
            rewritten_expression,
            base=source_candidate.evidence_layer.model_code_decomposition,
        ),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id=engine_id,
            adapter_class="egraph_rewrite_engine",
            source_candidate_id=(
                f"{origin.source_candidate_id}__egraph_rewrite_{proposal_rank:02d}"
            ),
            search_class=context.search_class,
            backend_family="egraph_rewrite",
            proposal_rank=proposal_rank,
            normalization_scope="rewrite_equivalent_expression_cir",
            comparability_scope="candidate_fitting_and_scoring",
            backend_private_fields=("rewrite_trace", "source_backend_origin"),
        ),
        replay_hooks=_rewrite_replay_hooks(
            source_candidate=source_candidate,
            saturation=saturation,
        ),
        assumptions=source_payload.assumptions,
        domain_constraints=source_payload.domain_constraints,
        unit_constraints=source_payload.unit_constraints,
        transient_diagnostics=diagnostics,
    )


def _model_code_for_expression(
    expression,
    *,
    base: CIRModelCodeDecomposition,
) -> CIRModelCodeDecomposition:
    nodes = walk_expression(expression)
    literal_count = sum(1 for node in nodes if node.__class__.__name__ == "Literal")
    parameter_count = sum(1 for node in nodes if isinstance(node, Parameter))
    state_count = sum(1 for node in nodes if isinstance(node, State))
    stochastic_count = sum(
        1 for node in nodes if isinstance(node, (NoiseTerm, DistributionParameter))
    )
    return CIRModelCodeDecomposition(
        L_family_bits=base.L_family_bits,
        L_structure_bits=float(expression_cost(expression)),
        L_literals_bits=float(literal_count),
        L_params_bits=float(parameter_count),
        L_state_bits=float(state_count + stochastic_count),
    )


def _rewrite_replay_hooks(
    *,
    source_candidate: CandidateIntermediateRepresentation,
    saturation: EqualitySaturationResult,
) -> CIRReplayHooks:
    hooks = list(source_candidate.evidence_layer.replay_hooks.hooks)
    hooks.extend(
        (
            CIRReplayHook(
                hook_name="rewrite_system",
                hook_ref="egraph_expression_rewrite_space_v1",
            ),
            CIRReplayHook(
                hook_name="rewrite_replay_identity",
                hook_ref=saturation.replay_identity,
            ),
            CIRReplayHook(
                hook_name="rewrite_extractor_cost",
                hook_ref=saturation.extraction.tie_break_rule,
            ),
        )
    )
    return CIRReplayHooks(hooks=tuple(hooks))


__all__ = [
    "EGraphEngine",
    "EGraphEngineConfig",
    "NativeFragmentRewriteCandidateProvider",
    "RewriteCandidateProvider",
    "StaticRewriteCandidateProvider",
]
