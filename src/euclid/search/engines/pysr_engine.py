from __future__ import annotations

import importlib.metadata as importlib_metadata
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from euclid.contracts.errors import ContractValidationError
from euclid.fit.refit import FitDataSplit, fit_cir_candidate
from euclid.search.engine_contracts import (
    EngineCandidateRecord,
    EngineFailureDiagnostic,
    EngineInputContext,
    EngineRunResult,
    SearchEngine,
    engine_claim_boundary,
)
from euclid.search.engines.pysr_lowering import (
    PySrHallOfFameRow,
    build_pysr_cir_candidate,
    lower_pysr_expression_to_expression_ir,
)


class PySrRuntimeUnavailable(RuntimeError):
    pass


_BINARY_OPERATOR_MAP = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "pow": "^",
}
_UNARY_OPERATOR_MAP = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "tanh": "tanh",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "abs": "abs",
}


@dataclass(frozen=True)
class PySrEngineConfig:
    binary_operators: tuple[str, ...] = ("add", "sub", "mul", "div")
    unary_operators: tuple[str, ...] = ("sin", "cos", "exp", "log")
    niterations: int = 40
    populations: int = 8
    maxsize: int = 20
    timeout_seconds: float = 10.0
    loss: str = "loss(x, y) = (x - y)^2"
    random_state: int | None = None
    refit_enabled: bool = True
    fit_window_id: str = "pysr_development_window"
    fit_max_nfev: int | None = None

    def __post_init__(self) -> None:
        if self.niterations <= 0:
            raise ContractValidationError(
                code="invalid_pysr_config",
                message="niterations must be positive",
                field_path="niterations",
            )
        if self.timeout_seconds <= 0 or not math.isfinite(float(self.timeout_seconds)):
            raise ContractValidationError(
                code="invalid_pysr_config",
                message="timeout_seconds must be positive and finite",
                field_path="timeout_seconds",
            )
        _translate_operators(self.binary_operators, mapping=_BINARY_OPERATOR_MAP)
        _translate_operators(self.unary_operators, mapping=_UNARY_OPERATOR_MAP)

    @property
    def allowed_operators(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.binary_operators, *self.unary_operators)))

    def operator_set(self) -> dict[str, list[str]]:
        return {
            "binary_operators": _translate_operators(
                self.binary_operators,
                mapping=_BINARY_OPERATOR_MAP,
            ),
            "unary_operators": _translate_operators(
                self.unary_operators,
                mapping=_UNARY_OPERATOR_MAP,
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "binary_operators": list(self.binary_operators),
            "unary_operators": list(self.unary_operators),
            "niterations": self.niterations,
            "populations": self.populations,
            "maxsize": self.maxsize,
            "timeout_seconds": self.timeout_seconds,
            "loss": self.loss,
            "random_state": self.random_state,
            "refit_enabled": self.refit_enabled,
            "fit_window_id": self.fit_window_id,
            "fit_max_nfev": self.fit_max_nfev,
        }


@dataclass(frozen=True)
class PySrDiscovery:
    status: str
    engine_version: str
    hall_of_fame: tuple[PySrHallOfFameRow, ...]
    trace: Mapping[str, Any]
    runtime_metadata: Mapping[str, Any]
    omission_disclosure: Mapping[str, Any]
    failure_diagnostics: tuple[EngineFailureDiagnostic | Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "engine_version", str(self.engine_version))
        object.__setattr__(self, "hall_of_fame", tuple(self.hall_of_fame))
        object.__setattr__(self, "trace", dict(self.trace))
        object.__setattr__(self, "runtime_metadata", dict(self.runtime_metadata))
        object.__setattr__(self, "omission_disclosure", dict(self.omission_disclosure))
        object.__setattr__(self, "failure_diagnostics", tuple(self.failure_diagnostics))


class PySrRunner(Protocol):
    def discover(
        self,
        *,
        context: EngineInputContext,
        config: PySrEngineConfig,
    ) -> PySrDiscovery:
        ...


class PySrEngine(SearchEngine):
    engine_id = "pysr-engine-v1"

    def __init__(
        self,
        *,
        config: PySrEngineConfig | None = None,
        runner: PySrRunner | None = None,
    ) -> None:
        self.config = config or PySrEngineConfig()
        self.runner = runner or _RuntimePySrRunner()
        self.engine_version = _package_version("pysr")

    def run(self, context: EngineInputContext) -> EngineRunResult:
        try:
            discovery = self.runner.discover(context=context, config=self.config)
        except PySrRuntimeUnavailable as exc:
            return _runtime_unavailable_result(
                context=context,
                engine_id=self.engine_id,
                engine_version=self.engine_version,
                message=str(exc),
            )

        failures = [
            _coerce_failure(self.engine_id, diagnostic)
            for diagnostic in discovery.failure_diagnostics
        ]
        records: list[EngineCandidateRecord] = []
        for rank, row in enumerate(discovery.hall_of_fame[: context.proposal_limit]):
            try:
                records.append(
                    self._record_from_row(
                        context=context,
                        discovery=discovery,
                        row=row,
                        proposal_rank=rank,
                    )
                )
            except ContractValidationError as exc:
                failures.append(
                    EngineFailureDiagnostic(
                        engine_id=self.engine_id,
                        reason_code="pysr_lowering_failed",
                        message=exc.message,
                        recoverable=True,
                        details={
                            "error_code": exc.code,
                            "field_path": exc.field_path,
                        },
                    )
                )

        status = discovery.status
        if not records and failures and status == "completed":
            status = "failed"
        return EngineRunResult(
            engine_id=self.engine_id,
            engine_version=discovery.engine_version or self.engine_version,
            status=status,
            candidates=tuple(records),
            failure_diagnostics=tuple(failures),
            trace={
                "engine_config": self.config.as_dict(),
                "operator_set": self.config.operator_set(),
                "runtime_metadata": dict(discovery.runtime_metadata),
                "pysr_trace": dict(discovery.trace),
            },
            omission_disclosure={
                **dict(discovery.omission_disclosure),
                "omitted_by_proposal_limit": max(
                    len(discovery.hall_of_fame) - context.proposal_limit,
                    0,
                ),
            },
            replay_metadata={
                **context.replay_metadata(),
                "engine_id": self.engine_id,
                "engine_version": discovery.engine_version or self.engine_version,
                "engine_config": self.config.as_dict(),
                "operator_set": self.config.operator_set(),
            },
        )

    def _record_from_row(
        self,
        *,
        context: EngineInputContext,
        discovery: PySrDiscovery,
        row: PySrHallOfFameRow,
        proposal_rank: int,
    ) -> EngineCandidateRecord:
        lowered = lower_pysr_expression_to_expression_ir(
            expression_source=row.equation,
            feature_names=context.feature_names,
            allowed_operators=self.config.allowed_operators,
        )
        candidate_id = f"pysr-{context.search_plan_id}-{proposal_rank:02d}"
        candidate = build_pysr_cir_candidate(
            row=row,
            feature_names=context.feature_names,
            allowed_operators=self.config.allowed_operators,
            search_class=context.search_class,
            source_candidate_id=candidate_id,
            proposal_rank=proposal_rank,
            transient_diagnostics={
                **dict(discovery.trace),
                "runtime_metadata": dict(discovery.runtime_metadata),
                "operator_set": self.config.operator_set(),
            },
        )
        fit_evidence = (
            _fit_evidence(
                candidate=candidate,
                context=context,
                fit_window_id=self.config.fit_window_id,
                parameter_declarations=lowered.parameter_declarations,
                max_nfev=self.config.fit_max_nfev,
            )
            if self.config.refit_enabled
            else {"status": "skipped", "reason": "refit_disabled"}
        )
        candidate_trace = {
            "pysr_trace": dict(discovery.trace),
            "runtime_metadata": dict(discovery.runtime_metadata),
            "lowering": dict(lowered.lowering_trace),
            "euclid_fit": fit_evidence,
            "claim_boundary": engine_claim_boundary(),
        }
        payload = candidate.structural_layer.expression_payload
        if payload is None:  # pragma: no cover - build helper guarantees this.
            raise AssertionError("PySR CIR candidate missing expression payload")
        return EngineCandidateRecord(
            candidate_id=candidate_id,
            engine_id=self.engine_id,
            engine_version=discovery.engine_version or self.engine_version,
            search_class=context.search_class,
            search_space_declaration="pysr_symbolic_regression_expression_ir_v1",
            budget_declaration={
                "proposal_limit": context.proposal_limit,
                "timeout_seconds": context.timeout_seconds,
                "config": self.config.as_dict(),
            },
            rows_used=_rows_used(context),
            features_used=tuple(payload.feature_dependencies),
            random_seed=context.random_seed,
            candidate_trace=candidate_trace,
            omission_disclosure=dict(discovery.omission_disclosure),
            claim_boundary=engine_claim_boundary(),
            proposed_cir=candidate,
            lowering_kind="expression_ir_cir",
        )


class _RuntimePySrRunner:
    def discover(
        self,
        *,
        context: EngineInputContext,
        config: PySrEngineConfig,
    ) -> PySrDiscovery:
        try:
            import numpy as np
            import pandas as pd
            from pysr import PySRRegressor
        except Exception as exc:  # pragma: no cover - environment dependent.
            raise PySrRuntimeUnavailable(f"{type(exc).__name__}: {exc}") from exc

        rows = _runtime_rows(context)
        feature_matrix, target = _feature_target_arrays(
            rows=rows,
            feature_names=context.feature_names,
            np=np,
        )
        model = PySRRegressor(
            niterations=config.niterations,
            populations=config.populations,
            maxsize=config.maxsize,
            timeout_in_seconds=min(config.timeout_seconds, context.timeout_seconds),
            binary_operators=config.operator_set()["binary_operators"],
            unary_operators=config.operator_set()["unary_operators"],
            loss=config.loss,
            random_state=(
                int(context.random_seed)
                if config.random_state is None
                else config.random_state
            ),
            deterministic=True,
            progress=False,
        )
        model.fit(feature_matrix, target, variable_names=list(context.feature_names))
        equations = getattr(model, "equations_", pd.DataFrame())
        hall_of_fame = tuple(_rows_from_equation_table(equations))
        runtime_metadata = {
            "pysr_version": _package_version("pysr"),
            "symbolic_regression_jl_version": "unknown",
            "julia_version": _julia_version_from_model(model),
        }
        return PySrDiscovery(
            status="completed",
            engine_version=runtime_metadata["pysr_version"],
            hall_of_fame=hall_of_fame,
            trace={
                "hall_of_fame_rows": len(hall_of_fame),
                "equation_columns": list(getattr(equations, "columns", ())),
            },
            runtime_metadata=runtime_metadata,
            omission_disclosure={"omitted_by_pareto_limit": 0},
            failure_diagnostics=(),
        )


def _rows_from_equation_table(equations) -> list[PySrHallOfFameRow]:
    rows: list[PySrHallOfFameRow] = []
    if equations is None:
        return rows
    for _, row in equations.iterrows():
        equation = str(row.get("equation", "")).strip()
        if not equation:
            continue
        rows.append(
            PySrHallOfFameRow(
                equation=equation,
                complexity=int(row.get("complexity", 0)),
                loss=_optional_float(row.get("loss")),
                score=_optional_float(row.get("score")),
                metadata={
                    key: _json_scalar(value)
                    for key, value in row.items()
                    if key not in {"equation", "complexity", "loss", "score"}
                },
            )
        )
    return rows


def _translate_operators(operators, *, mapping) -> list[str]:
    translated = []
    for operator in operators:
        if operator not in mapping:
            raise ContractValidationError(
                code="invalid_pysr_config",
                message=f"unsupported PySR operator {operator!r}",
                field_path="operators",
            )
        translated.append(mapping[operator])
    return translated


def _feature_target_arrays(*, rows, feature_names, np):
    if len(rows) < 2:
        raise ContractValidationError(
            code="invalid_pysr_training_rows",
            message="PySR requires at least two development rows",
            field_path="rows",
        )
    feature_matrix = np.asarray(
        [
            [_finite(row[name], field_path=name) for name in feature_names]
            for row in rows
        ],
        dtype=float,
    )
    target = np.asarray([_finite(row["target"], field_path="target") for row in rows])
    return feature_matrix, target


def _fit_evidence(
    *,
    candidate,
    context: EngineInputContext,
    fit_window_id: str,
    parameter_declarations,
    max_nfev: int | None,
) -> dict[str, Any]:
    fit = fit_cir_candidate(
        candidate=candidate,
        data=FitDataSplit(train_rows=_runtime_rows(context)),
        fit_window_id=fit_window_id,
        parameter_declarations=parameter_declarations,
        seed=context.random_seed,
        max_nfev=max_nfev,
    )
    return fit.as_redacted_evidence()


def _runtime_rows(context: EngineInputContext) -> tuple[Mapping[str, Any], ...]:
    if context.runtime_rows:
        return tuple(context.runtime_rows)
    feature_view = context.runtime_feature_view
    rows = getattr(feature_view, "rows", None)
    if rows is not None:
        return tuple(rows)
    raise ContractValidationError(
        code="missing_engine_runtime_rows",
        message="PySR engine requires runtime rows for external fitting",
        field_path="context.runtime_rows",
    )


def _rows_used(context: EngineInputContext) -> tuple[str, ...]:
    return tuple(
        row["event_time"] for row in context.rows_features_access.row_fingerprints
    )


def _finite(value: Any, *, field_path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            code="invalid_pysr_training_rows",
            message=f"{field_path} must be numeric",
            field_path=field_path,
        ) from exc
    if not math.isfinite(result):
        raise ContractValidationError(
            code="invalid_pysr_training_rows",
            message=f"{field_path} must be finite",
            field_path=field_path,
        )
    return result


def _coerce_failure(
    engine_id: str,
    diagnostic: EngineFailureDiagnostic | Mapping[str, Any],
) -> EngineFailureDiagnostic:
    if isinstance(diagnostic, EngineFailureDiagnostic):
        return diagnostic
    return EngineFailureDiagnostic(
        engine_id=engine_id,
        reason_code=str(diagnostic.get("reason_code", "pysr_failure")),
        message=str(diagnostic.get("message", "PySR returned a failure")),
        recoverable=bool(diagnostic.get("recoverable", True)),
        details=dict(diagnostic.get("details", {})),
    )


def _runtime_unavailable_result(
    *,
    context: EngineInputContext,
    engine_id: str,
    engine_version: str,
    message: str,
) -> EngineRunResult:
    return EngineRunResult(
        engine_id=engine_id,
        engine_version=engine_version,
        status="failed",
        candidates=(),
        failure_diagnostics=(
            EngineFailureDiagnostic(
                engine_id=engine_id,
                reason_code="pysr_runtime_unavailable",
                message=message or "PySR or Julia runtime is unavailable",
                recoverable=True,
            ),
        ),
        trace={"runtime_unavailable": True},
        omission_disclosure={"omitted_due_to_runtime_unavailable": True},
        replay_metadata={
            **context.replay_metadata(),
            "engine_id": engine_id,
            "engine_version": engine_version,
        },
    )


def _package_version(distribution_name: str) -> str:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def _julia_version_from_model(model) -> str:
    version = getattr(model, "julia_version_", None)
    return "unknown" if version is None else str(version)


def _optional_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_scalar(value):
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


__all__ = [
    "PySrDiscovery",
    "PySrEngine",
    "PySrEngineConfig",
    "PySrRuntimeUnavailable",
]
