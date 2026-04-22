from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from euclid.cir.models import CandidateIntermediateRepresentation
from euclid.contracts.errors import ContractValidationError
from euclid.expr.evaluators import evaluate_expression
from euclid.expr.serialization import expression_from_dict
from euclid.fit.diagnostics import (
    sanitized_data_window,
    statistical_baseline_diagnostics,
)
from euclid.fit.objectives import get_objective
from euclid.fit.parameterization import ParameterDeclaration
from euclid.fit.scipy_optimizers import fit_least_squares
from euclid.runtime.hashing import sha256_digest


@dataclass(frozen=True)
class FitDataSplit:
    train_rows: tuple[Mapping[str, Any], ...]
    validation_rows: tuple[Mapping[str, Any], ...] = ()
    test_rows: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class UnifiedFitResult:
    candidate_hash: str
    fit_window_id: str
    objective_id: str
    status: str
    parameter_estimates: dict[str, float]
    uncertainty_diagnostics: dict[str, Any]
    optimizer_diagnostics: dict[str, Any]
    replay_metadata: dict[str, Any]
    replay_identity: str
    failure_reasons: tuple[str, ...]
    rows_used: tuple[dict[str, Any], ...]
    split_counts: dict[str, int]
    loss: float
    claim_boundary: dict[str, Any]

    def as_redacted_evidence(self) -> dict[str, Any]:
        return {
            "candidate_hash": self.candidate_hash,
            "fit_window_id": self.fit_window_id,
            "objective_id": self.objective_id,
            "status": self.status,
            "parameter_estimates": dict(self.parameter_estimates),
            "optimizer_diagnostics": dict(self.optimizer_diagnostics),
            "uncertainty_diagnostics": dict(self.uncertainty_diagnostics),
            "replay_metadata": dict(self.replay_metadata),
            "replay_identity": self.replay_identity,
            "failure_reasons": list(self.failure_reasons),
            "split_counts": dict(self.split_counts),
            "loss": self.loss,
            "claim_boundary": dict(self.claim_boundary),
        }


def fit_cir_candidate(
    *,
    candidate: CandidateIntermediateRepresentation,
    data: FitDataSplit,
    fit_window_id: str,
    parameter_declarations: Sequence[ParameterDeclaration] | None = None,
    objective_id: str = "squared_error",
    seed: int | str = 0,
    max_nfev: int | None = None,
) -> UnifiedFitResult:
    payload = candidate.structural_layer.expression_payload
    if payload is None:
        raise ContractValidationError(
            code="unified_fit_requires_expression_payload",
            message="unified fitting requires CIR expression_payload",
            field_path="candidate.structural_layer.expression_payload",
        )
    objective = get_objective(objective_id)
    seed_value = int(seed)
    expression = expression_from_dict(payload.expression_tree)
    train_rows = tuple(data.train_rows)
    validation_rows = tuple(data.validation_rows)
    test_rows = tuple(data.test_rows)
    _validate_rows(train_rows, split_name="train")
    _validate_rows(validation_rows, split_name="validation")
    _validate_rows(test_rows, split_name="test")

    declarations = tuple(parameter_declarations or _default_declarations(payload.parameter_declarations))
    feature_names = tuple(payload.feature_dependencies)
    split_counts = {
        "train": len(train_rows),
        "validation": len(validation_rows),
        "test": len(test_rows),
    }
    rows_used = tuple(_sanitized_row_ref(row) for row in train_rows)
    replay_identity = _replay_identity(
        candidate=candidate,
        fit_window_id=fit_window_id,
        objective_id=objective_id,
        declarations=declarations,
        seed=seed_value,
        data=data,
    )
    claim_boundary = {
        "claim_publication_allowed": False,
        "reason_codes": ["fit_is_not_claim_authority"],
    }

    if len(train_rows) < len([declaration for declaration in declarations if not declaration.fixed]):
        return UnifiedFitResult(
            candidate_hash=candidate.canonical_hash(),
            fit_window_id=fit_window_id,
            objective_id=objective_id,
            status="failed",
            parameter_estimates={declaration.name: declaration.initial_value for declaration in declarations},
            uncertainty_diagnostics={"rank": len(train_rows), "parameter_count": len(declarations)},
            optimizer_diagnostics={
                "fit_layer": "src/euclid/fit",
                "optimizer_backend": "preflight",
                "failure_stage": "rank_check",
                "data_window": sanitized_data_window(
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                    test_rows=test_rows,
                ),
            },
            replay_metadata={"seed": seed_value, "objective_id": objective_id},
            replay_identity=replay_identity,
            failure_reasons=("underdetermined_system",),
            rows_used=rows_used,
            split_counts=split_counts,
            loss=math.inf,
            claim_boundary=claim_boundary,
        )

    def residual_fn(params: Mapping[str, float]) -> tuple[float, ...]:
        predictions = tuple(
            _predict_row(expression=expression, row=row, params=params)
            for row in train_rows
        )
        observed = tuple(float(row["target"]) for row in train_rows)
        return objective.residuals(observed, predictions)

    if declarations:
        optimizer = fit_least_squares(
            parameter_declarations=declarations,
            residual_fn=residual_fn,
            objective_id=objective_id,
            seed=seed_value,
            max_nfev=max_nfev,
            scipy_loss=objective.scipy_loss,
        )
        parameter_estimates = dict(optimizer.parameter_estimates)
        optimizer_diagnostics = dict(optimizer.diagnostics)
        optimizer_diagnostics["fit_layer"] = "src/euclid/fit"
        optimizer_diagnostics["claim_boundary"] = claim_boundary
        optimizer_diagnostics["data_window"] = sanitized_data_window(
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
        )
        optimizer_diagnostics["statistical_baselines"] = statistical_baseline_diagnostics(
            train_rows=train_rows,
            feature_names=feature_names,
        )
        failure_reasons = tuple(optimizer.failure_reasons)
        converged = optimizer.converged
        loss = float(optimizer.loss)
        replay_metadata = dict(optimizer.replay_metadata)
    else:
        predictions = tuple(
            _predict_row(expression=expression, row=row, params={})
            for row in train_rows
        )
        observed = tuple(float(row["target"]) for row in train_rows)
        parameter_estimates = {}
        loss = objective.scalar_loss(observed, predictions)
        converged = True
        failure_reasons = ()
        replay_metadata = {"seed": seed_value, "objective_id": objective_id}
        optimizer_diagnostics = {
            "fit_layer": "src/euclid/fit",
            "optimizer_backend": "direct_expression_evaluation",
            "objective_id": objective_id,
            "iteration_count": 0,
            "function_evaluations": len(train_rows),
            "claim_boundary": claim_boundary,
            "data_window": sanitized_data_window(
                train_rows=train_rows,
                validation_rows=validation_rows,
                test_rows=test_rows,
            ),
        }

    return UnifiedFitResult(
        candidate_hash=candidate.canonical_hash(),
        fit_window_id=fit_window_id,
        objective_id=objective_id,
        status="converged" if converged else "failed",
        parameter_estimates=parameter_estimates,
        uncertainty_diagnostics={
            "train_row_count": len(train_rows),
            "parameter_count": len(declarations),
            "residual_degrees_of_freedom": max(len(train_rows) - len(declarations), 0),
        },
        optimizer_diagnostics=optimizer_diagnostics,
        replay_metadata=replay_metadata,
        replay_identity=replay_identity,
        failure_reasons=failure_reasons,
        rows_used=rows_used,
        split_counts=split_counts,
        loss=loss,
        claim_boundary=claim_boundary,
    )


def _default_declarations(parameter_names: Sequence[str]) -> tuple[ParameterDeclaration, ...]:
    return tuple(ParameterDeclaration(name, initial_value=0.0) for name in parameter_names)


def _predict_row(
    *,
    expression,
    row: Mapping[str, Any],
    params: Mapping[str, float],
) -> float:
    values = dict(params)
    for key, value in row.items():
        if key in {"event_time", "available_at", "entity"}:
            continue
        if isinstance(value, (int, float)):
            values[key] = float(value)
    try:
        return float(evaluate_expression(expression, values))
    except ContractValidationError as exc:
        raise ContractValidationError(
            code="unified_fit_domain_violation",
            message=f"expression domain violation during fitting: {exc.message}",
            field_path=exc.field_path,
            details={"source_code": exc.code},
        ) from exc


def _validate_rows(rows: Sequence[Mapping[str, Any]], *, split_name: str) -> None:
    previous_event_time: str | None = None
    seen_event_times: set[str] = set()
    for index, row in enumerate(rows):
        event_time = str(row.get("event_time", ""))
        if not event_time:
            raise ContractValidationError(
                code="invalid_fit_rows",
                message="fit rows require event_time",
                field_path=f"{split_name}[{index}].event_time",
            )
        if event_time in seen_event_times:
            raise ContractValidationError(
                code="duplicate_fit_timestamp",
                message="duplicate fit row timestamp",
                field_path=f"{split_name}[{index}].event_time",
            )
        if previous_event_time is not None and event_time <= previous_event_time:
            raise ContractValidationError(
                code="out_of_order_fit_timestamp",
                message="fit rows must be in strictly increasing event_time order; out-of-order timestamp encountered",
                field_path=f"{split_name}[{index}].event_time",
            )
        seen_event_times.add(event_time)
        previous_event_time = event_time
        if "target" not in row:
            raise ContractValidationError(
                code="missing_fit_value",
                message="fit rows require target values",
                field_path=f"{split_name}[{index}].target",
            )
        _finite_value(row["target"], field_path=f"{split_name}[{index}].target")
        for key, value in row.items():
            if key in {"event_time", "available_at", "entity"}:
                continue
            if isinstance(value, (int, float)):
                _finite_value(value, field_path=f"{split_name}[{index}].{key}")


def _finite_value(value: object, *, field_path: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            code="missing_fit_value",
            message=f"{field_path} must be numeric",
            field_path=field_path,
        ) from exc
    if not math.isfinite(normalized):
        raise ContractValidationError(
            code="nonfinite_fit_value",
            message=f"{field_path} must be finite",
            field_path=field_path,
        )
    return normalized


def _sanitized_row_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_time": str(row["event_time"]),
        "available_at": str(row.get("available_at", row["event_time"])),
    }


def _replay_identity(
    *,
    candidate: CandidateIntermediateRepresentation,
    fit_window_id: str,
    objective_id: str,
    declarations: Sequence[ParameterDeclaration],
    seed: int,
    data: FitDataSplit,
) -> str:
    return sha256_digest(
        {
            "candidate_hash": candidate.canonical_hash(),
            "fit_window_id": fit_window_id,
            "objective_id": objective_id,
            "declarations": [declaration.as_dict() for declaration in declarations],
            "seed": seed,
            "train_rows": _row_fingerprints(data.train_rows),
            "validation_rows": _row_fingerprints(data.validation_rows),
            "test_rows": _row_fingerprints(data.test_rows),
        }
    )


def _row_fingerprints(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "event_time": str(row["event_time"]),
            "available_at": str(row.get("available_at", row["event_time"])),
            "row_hash": sha256_digest(
                {
                    key: value
                    for key, value in sorted(row.items())
                    if key not in {"raw_payload", "provider_headers"}
                }
            ),
        }
        for row in rows
    ]


__all__ = ["FitDataSplit", "UnifiedFitResult", "fit_cir_candidate"]
