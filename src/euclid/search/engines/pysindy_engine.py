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
from euclid.search.engines.pysindy_lowering import (
    PySindyDiscoveredEquation,
    PySindyTerm,
    build_pysindy_cir_candidate,
    lower_pysindy_terms_to_expression_ir,
)


class PySindyRuntimeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class PySindyEngineConfig:
    library_kind: str = "polynomial"
    polynomial_degree: int = 2
    include_bias: bool = True
    include_interaction: bool = True
    fourier_n_frequencies: int = 1
    custom_operators: tuple[str, ...] = ("sin", "cos")
    optimizer_kind: str = "stlsq"
    threshold: float = 0.1
    alpha: float = 0.05
    max_iter: int = 20
    differentiation_method: str = "external_x_dot"
    ensemble_enabled: bool = False
    refit_enabled: bool = True
    fit_window_id: str = "pysindy_development_window"
    fit_max_nfev: int | None = None

    def __post_init__(self) -> None:
        if self.library_kind not in {"polynomial", "fourier", "custom"}:
            raise ContractValidationError(
                code="invalid_pysindy_config",
                message="library_kind must be polynomial, fourier, or custom",
                field_path="library_kind",
            )
        if self.optimizer_kind not in {"stlsq", "sr3"}:
            raise ContractValidationError(
                code="invalid_pysindy_config",
                message="optimizer_kind must be stlsq or sr3",
                field_path="optimizer_kind",
            )
        if self.polynomial_degree <= 0:
            raise ContractValidationError(
                code="invalid_pysindy_config",
                message="polynomial_degree must be positive",
                field_path="polynomial_degree",
            )
        if self.fourier_n_frequencies <= 0:
            raise ContractValidationError(
                code="invalid_pysindy_config",
                message="fourier_n_frequencies must be positive",
                field_path="fourier_n_frequencies",
            )
        for field_name in ("threshold", "alpha"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ContractValidationError(
                    code="invalid_pysindy_config",
                    message=f"{field_name} must be finite and non-negative",
                    field_path=field_name,
                )
            object.__setattr__(self, field_name, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "library_kind": self.library_kind,
            "polynomial_degree": self.polynomial_degree,
            "include_bias": self.include_bias,
            "include_interaction": self.include_interaction,
            "fourier_n_frequencies": self.fourier_n_frequencies,
            "custom_operators": list(self.custom_operators),
            "optimizer_kind": self.optimizer_kind,
            "threshold": self.threshold,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "differentiation_method": self.differentiation_method,
            "ensemble_enabled": self.ensemble_enabled,
            "refit_enabled": self.refit_enabled,
            "fit_window_id": self.fit_window_id,
            "fit_max_nfev": self.fit_max_nfev,
        }


@dataclass(frozen=True)
class PySindyDiscovery:
    status: str
    engine_version: str
    equations: tuple[PySindyDiscoveredEquation, ...]
    trace: Mapping[str, Any]
    omission_disclosure: Mapping[str, Any]
    failure_diagnostics: tuple[EngineFailureDiagnostic | Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "engine_version", str(self.engine_version))
        object.__setattr__(self, "equations", tuple(self.equations))
        object.__setattr__(self, "trace", dict(self.trace))
        object.__setattr__(self, "omission_disclosure", dict(self.omission_disclosure))
        object.__setattr__(self, "failure_diagnostics", tuple(self.failure_diagnostics))


class PySindyRunner(Protocol):
    def discover(
        self,
        *,
        context: EngineInputContext,
        config: PySindyEngineConfig,
    ) -> PySindyDiscovery:
        ...


class PySindyEngine(SearchEngine):
    engine_id = "pysindy-engine-v1"

    def __init__(
        self,
        *,
        config: PySindyEngineConfig | None = None,
        runner: PySindyRunner | None = None,
    ) -> None:
        self.config = config or PySindyEngineConfig()
        self.runner = runner or _RuntimePySindyRunner()
        self.engine_version = _package_version("pysindy")

    def run(self, context: EngineInputContext) -> EngineRunResult:
        try:
            discovery = self.runner.discover(context=context, config=self.config)
        except PySindyRuntimeUnavailable as exc:
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
        for rank, equation in enumerate(discovery.equations[: context.proposal_limit]):
            try:
                records.append(
                    self._record_from_equation(
                        context=context,
                        discovery=discovery,
                        equation=equation,
                        proposal_rank=rank,
                    )
                )
            except ContractValidationError as exc:
                failures.append(
                    EngineFailureDiagnostic(
                        engine_id=self.engine_id,
                        reason_code="pysindy_lowering_failed",
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
                "pysindy_trace": dict(discovery.trace),
            },
            omission_disclosure={
                **dict(discovery.omission_disclosure),
                "omitted_by_proposal_limit": max(
                    len(discovery.equations) - context.proposal_limit,
                    0,
                ),
            },
            replay_metadata={
                **context.replay_metadata(),
                "engine_id": self.engine_id,
                "engine_version": discovery.engine_version or self.engine_version,
                "engine_config": self.config.as_dict(),
            },
        )

    def _record_from_equation(
        self,
        *,
        context: EngineInputContext,
        discovery: PySindyDiscovery,
        equation: PySindyDiscoveredEquation,
        proposal_rank: int,
    ) -> EngineCandidateRecord:
        lowered = lower_pysindy_terms_to_expression_ir(
            terms=equation.terms,
            feature_names=context.feature_names,
            coefficient_threshold=self.config.threshold,
        )
        candidate_id = (
            f"pysindy-{context.search_plan_id}-{proposal_rank:02d}-"
            f"{lowered.lowering_trace['active_term_count']}"
        )
        candidate = build_pysindy_cir_candidate(
            equation=equation,
            feature_names=context.feature_names,
            search_class=context.search_class,
            source_candidate_id=candidate_id,
            proposal_rank=proposal_rank,
            coefficient_threshold=self.config.threshold,
            transient_diagnostics={
                **dict(discovery.trace),
                "engine_config": self.config.as_dict(),
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
            "pysindy_trace": dict(discovery.trace),
            "lowering": dict(lowered.lowering_trace),
            "euclid_fit": fit_evidence,
            "claim_boundary": engine_claim_boundary(),
        }
        return EngineCandidateRecord(
            candidate_id=candidate_id,
            engine_id=self.engine_id,
            engine_version=discovery.engine_version or self.engine_version,
            search_class=context.search_class,
            search_space_declaration="pysindy_sparse_dynamics_expression_ir_v1",
            budget_declaration={
                "proposal_limit": context.proposal_limit,
                "timeout_seconds": context.timeout_seconds,
                "config": self.config.as_dict(),
            },
            rows_used=_rows_used(context),
            features_used=tuple(candidate.structural_layer.expression_payload.feature_dependencies),
            random_seed=context.random_seed,
            candidate_trace=candidate_trace,
            omission_disclosure=dict(discovery.omission_disclosure),
            claim_boundary=engine_claim_boundary(),
            proposed_cir=candidate,
            lowering_kind="expression_ir_cir",
        )


class _RuntimePySindyRunner:
    def discover(
        self,
        *,
        context: EngineInputContext,
        config: PySindyEngineConfig,
    ) -> PySindyDiscovery:
        try:
            import numpy as np
            import pysindy as ps
        except Exception as exc:  # pragma: no cover - environment dependent.
            raise PySindyRuntimeUnavailable(f"{type(exc).__name__}: {exc}") from exc

        rows = _runtime_rows(context)
        feature_matrix, target_matrix = _feature_target_matrices(
            rows=rows,
            feature_names=context.feature_names,
        )
        library = _build_library(ps=ps, np=np, config=config)
        optimizer = _build_optimizer(ps=ps, config=config)
        model = ps.SINDy(
            optimizer=optimizer,
            feature_library=library,
            differentiation_method=None,
        )
        model.fit(
            feature_matrix,
            t=1.0,
            x_dot=target_matrix,
            feature_names=list(context.feature_names),
        )
        feature_library_names = tuple(model.get_feature_names())
        coefficients = tuple(float(value) for value in model.coefficients()[0])
        terms = tuple(
            PySindyTerm(term_name=name, coefficient=coefficient)
            for name, coefficient in zip(
                feature_library_names,
                coefficients,
                strict=True,
            )
        )
        support_mask = [
            abs(coefficient) > config.threshold for coefficient in coefficients
        ]
        trace = {
            "pysindy_version": getattr(ps, "__version__", _package_version("pysindy")),
            "library_kind": config.library_kind,
            "library_config": config.as_dict(),
            "optimizer_kind": config.optimizer_kind,
            "differentiation_method": config.differentiation_method,
            "feature_library_names": list(feature_library_names),
            "coefficients": list(coefficients),
            "support_mask": support_mask,
            "support_count": sum(1 for supported in support_mask if supported),
            "equations": list(model.equations()),
            "row_count": len(rows),
        }
        return PySindyDiscovery(
            status="completed",
            engine_version=str(trace["pysindy_version"]),
            equations=(
                PySindyDiscoveredEquation(
                    output_name="target",
                    terms=terms,
                    equation_text=str(model.equations()[0]),
                ),
            ),
            trace=trace,
            omission_disclosure={
                "omitted_by_sparsity": len(coefficients) - int(trace["support_count"]),
                "ensemble_enabled": config.ensemble_enabled,
            },
            failure_diagnostics=(),
        )


def _build_library(*, ps, np, config: PySindyEngineConfig):
    if config.library_kind == "polynomial":
        return ps.PolynomialLibrary(
            degree=config.polynomial_degree,
            include_interaction=config.include_interaction,
            include_bias=config.include_bias,
        )
    if config.library_kind == "fourier":
        return ps.FourierLibrary(n_frequencies=config.fourier_n_frequencies)
    functions = []
    names = []
    for operator in config.custom_operators:
        if operator == "sin":
            functions.append(lambda x: np.sin(x))
            names.append(lambda name: f"sin({name})")
        elif operator == "cos":
            functions.append(lambda x: np.cos(x))
            names.append(lambda name: f"cos({name})")
        elif operator == "tanh":
            functions.append(lambda x: np.tanh(x))
            names.append(lambda name: f"tanh({name})")
        else:
            raise ContractValidationError(
                code="invalid_pysindy_config",
                message=f"unsupported custom PySINDy operator {operator!r}",
                field_path="custom_operators",
            )
    return ps.CustomLibrary(
        library_functions=functions,
        function_names=names,
        include_bias=config.include_bias,
    )


def _build_optimizer(*, ps, config: PySindyEngineConfig):
    if config.optimizer_kind == "stlsq":
        return ps.STLSQ(
            threshold=config.threshold,
            alpha=config.alpha,
            max_iter=config.max_iter,
        )
    return ps.SR3(
        reg_weight_lam=max(config.threshold, 1e-12),
        max_iter=config.max_iter,
    )


def _feature_target_matrices(*, rows, feature_names):
    import numpy as np

    if len(rows) < 2:
        raise ContractValidationError(
            code="invalid_pysindy_training_rows",
            message="PySINDy requires at least two development rows",
            field_path="rows",
        )
    feature_matrix = np.asarray(
        [
            [_finite(row[name], field_path=name) for name in feature_names]
            for row in rows
        ],
        dtype=float,
    )
    target_matrix = np.asarray(
        [[_finite(row["target"], field_path="target")] for row in rows],
        dtype=float,
    )
    return feature_matrix, target_matrix


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
        message="PySINDy engine requires runtime rows for external fitting",
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
            code="invalid_pysindy_training_rows",
            message=f"{field_path} must be numeric",
            field_path=field_path,
        ) from exc
    if not math.isfinite(result):
        raise ContractValidationError(
            code="invalid_pysindy_training_rows",
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
        reason_code=str(diagnostic.get("reason_code", "pysindy_failure")),
        message=str(diagnostic.get("message", "PySINDy returned a failure")),
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
                reason_code="pysindy_runtime_unavailable",
                message=message or "PySINDy runtime is unavailable",
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


__all__ = [
    "PySindyDiscovery",
    "PySindyEngine",
    "PySindyEngineConfig",
    "PySindyRuntimeUnavailable",
]
