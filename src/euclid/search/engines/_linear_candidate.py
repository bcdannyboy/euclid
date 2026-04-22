from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from euclid.cir.models import (
    CIRBackendOriginRecord,
    CIRForecastOperator,
    CIRInputSignature,
    CIRModelCodeDecomposition,
    CIRReplayHook,
    CIRReplayHooks,
)
from euclid.cir.normalize import build_cir_candidate_from_expression
from euclid.expr.ast import BinaryOp, Expr, Feature, Literal, NaryOp
from euclid.search.engine_contracts import (
    EngineCandidateRecord,
    EngineFailureDiagnostic,
    EngineInputContext,
    EngineRunResult,
    engine_claim_boundary,
)


@dataclass(frozen=True)
class NumericTrainingData:
    x: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    rows_used: tuple[str, ...]
    omitted_row_count: int


def numeric_training_data(context: EngineInputContext) -> NumericTrainingData:
    rows: list[list[float]] = []
    targets: list[float] = []
    rows_used: list[str] = []
    feature_names = tuple(context.feature_names)

    for row in context.runtime_rows:
        target = _finite_float(row.get("target"))
        features = [_finite_float(row.get(name)) for name in feature_names]
        if target is None or any(value is None for value in features):
            continue
        targets.append(target)
        rows.append([float(value) for value in features if value is not None])
        rows_used.append(str(row.get("event_time", "")))

    return NumericTrainingData(
        x=np.asarray(rows, dtype=float).reshape(len(rows), len(feature_names)),
        y=np.asarray(targets, dtype=float),
        feature_names=feature_names,
        rows_used=tuple(rows_used),
        omitted_row_count=max(len(context.runtime_rows) - len(rows), 0),
    )


def result_for_insufficient_data(
    *,
    context: EngineInputContext,
    engine_id: str,
    engine_version: str,
    reason_code: str,
    message: str,
    details: Mapping[str, Any],
) -> EngineRunResult:
    return EngineRunResult(
        engine_id=engine_id,
        engine_version=engine_version,
        status="partial",
        candidates=(),
        failure_diagnostics=(
            EngineFailureDiagnostic(
                engine_id=engine_id,
                reason_code=reason_code,
                message=message,
                recoverable=True,
                details=dict(details),
            ),
        ),
        trace={
            "engine_execution": "completed_without_candidate",
            "reason_code": reason_code,
        },
        omission_disclosure={
            "candidate_omitted": True,
            "reason_code": reason_code,
        },
        replay_metadata={
            **context.replay_metadata(),
            "engine_id": engine_id,
            "engine_version": engine_version,
        },
        claim_boundary=engine_claim_boundary(),
    )


def linear_expression(
    *,
    intercept: float,
    coefficients: Mapping[str, float],
    coefficient_threshold: float,
) -> Expr:
    terms: list[Expr] = []
    if abs(intercept) > coefficient_threshold:
        terms.append(Literal(_round_float(intercept)))
    for feature_name, coefficient in sorted(coefficients.items()):
        if abs(coefficient) <= coefficient_threshold:
            continue
        terms.append(
            BinaryOp(
                "mul",
                Literal(_round_float(coefficient)),
                Feature(feature_name),
            )
        )
    if not terms:
        return Literal(0.0)
    if len(terms) == 1:
        return terms[0]
    return NaryOp("add", tuple(terms))


def linear_candidate_record(
    *,
    context: EngineInputContext,
    engine_id: str,
    engine_version: str,
    candidate_id: str,
    proposal_rank: int,
    expression: Expr,
    active_feature_names: Sequence[str],
    rows_used: Sequence[str],
    search_space_declaration: str,
    cir_family_id: str,
    cir_form_class: str,
    backend_family: str,
    candidate_trace: Mapping[str, Any],
    omission_disclosure: Mapping[str, Any],
) -> EngineCandidateRecord:
    active_features = tuple(sorted(str(name) for name in active_feature_names))
    model_code = CIRModelCodeDecomposition(
        L_family_bits=2.0,
        L_structure_bits=max(1.0, float(len(active_features))),
        L_literals_bits=1.0 + float(len(active_features)),
        L_params_bits=0.0,
        L_state_bits=0.0,
    )
    candidate = build_cir_candidate_from_expression(
        expression=expression,
        cir_family_id=cir_family_id,
        cir_form_class=cir_form_class,
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=active_features,
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=model_code,
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id=engine_id,
            adapter_class=engine_id.replace("-", "_"),
            source_candidate_id=candidate_id,
            search_class=context.search_class,
            backend_family=backend_family,
            proposal_rank=proposal_rank,
            comparability_scope="non_claim_candidate_search_only",
            backend_private_fields=("engine_trace", "fit_coefficients"),
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(
                CIRReplayHook(
                    hook_name=f"{engine_id}-replay",
                    hook_ref=f"engine:{engine_id}:{context.search_plan_id}:{proposal_rank}",
                ),
            )
        ),
        transient_diagnostics={
            "engine_trace": dict(candidate_trace),
            "rows_used": list(rows_used),
            "claim_publication_allowed": False,
            "reason_codes": ["search_engine_not_claim_authority"],
        },
    )
    return EngineCandidateRecord(
        candidate_id=candidate_id,
        engine_id=engine_id,
        engine_version=engine_version,
        search_class=context.search_class,
        search_space_declaration=search_space_declaration,
        budget_declaration={
            "proposal_limit": context.proposal_limit,
            "timeout_seconds": context.timeout_seconds,
        },
        rows_used=tuple(str(row_id) for row_id in rows_used),
        features_used=active_features,
        random_seed=context.random_seed,
        candidate_trace={
            **dict(candidate_trace),
            "claim_boundary": engine_claim_boundary(),
        },
        omission_disclosure=dict(omission_disclosure),
        claim_boundary=engine_claim_boundary(),
        proposed_cir=candidate,
        lowering_kind="proposed_cir",
    )


def engine_result(
    *,
    context: EngineInputContext,
    engine_id: str,
    engine_version: str,
    records: Sequence[EngineCandidateRecord],
    trace: Mapping[str, Any],
    omission_disclosure: Mapping[str, Any],
) -> EngineRunResult:
    return EngineRunResult(
        engine_id=engine_id,
        engine_version=engine_version,
        status="completed",
        candidates=tuple(records[: context.proposal_limit]),
        failure_diagnostics=(),
        trace=dict(trace),
        omission_disclosure={
            **dict(omission_disclosure),
            "omitted_by_proposal_limit": max(len(records) - context.proposal_limit, 0),
        },
        replay_metadata={
            **context.replay_metadata(),
            "engine_id": engine_id,
            "engine_version": engine_version,
        },
        claim_boundary=engine_claim_boundary(),
    )


def stable_int_seed(seed: str) -> int:
    digest = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def rounded_mapping(values: Mapping[str, float]) -> dict[str, float]:
    return {name: _round_float(value) for name, value in sorted(values.items())}


def rows_used_from_data(data: NumericTrainingData) -> tuple[str, ...]:
    return tuple(data.rows_used)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round_float(value: float) -> float:
    return float(round(float(value), 12))


__all__ = [
    "NumericTrainingData",
    "engine_result",
    "linear_candidate_record",
    "linear_expression",
    "numeric_training_data",
    "result_for_insufficient_data",
    "rounded_mapping",
    "rows_used_from_data",
    "stable_int_seed",
]
