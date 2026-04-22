from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from euclid.cir.models import (
    CIRBackendOriginRecord,
    CIRForecastOperator,
    CIRInputSignature,
    CIRModelCodeDecomposition,
    CIRReplayHook,
    CIRReplayHooks,
)
from euclid.cir.normalize import build_cir_candidate_from_expression
from euclid.contracts.errors import ContractValidationError
from euclid.expr.ast import (
    BinaryOp,
    Expr,
    Feature,
    Literal,
    NaryOp,
    Parameter,
    UnaryOp,
    walk_expression,
)
from euclid.expr.sympy_bridge import from_sympy
from euclid.fit.parameterization import ParameterDeclaration
from euclid.runtime.hashing import sha256_digest

_SAFE_GLOBALS = {
    "__builtins__": {},
    "Abs": sp.Abs,
    "Float": sp.Float,
    "Integer": sp.Integer,
    "Rational": sp.Rational,
    "cos": sp.cos,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
    "tanh": sp.tanh,
}
_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)


@dataclass(frozen=True)
class PySrHallOfFameRow:
    equation: str
    complexity: int
    loss: float | None = None
    score: float | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        equation = str(self.equation).strip()
        if not equation:
            raise ContractValidationError(
                code="invalid_pysr_hall_of_fame",
                message="PySR hall-of-fame equations must be non-empty",
                field_path="equation",
            )
        if self.complexity < 0:
            raise ContractValidationError(
                code="invalid_pysr_hall_of_fame",
                message="PySR complexity must be non-negative",
                field_path="complexity",
            )
        object.__setattr__(self, "equation", equation)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class LoweredPySrExpression:
    expression: Expr
    parameter_declarations: tuple[ParameterDeclaration, ...]
    lowering_trace: Mapping[str, Any]


def lower_pysr_expression_to_expression_ir(
    *,
    expression_source: str,
    feature_names: Sequence[str],
    allowed_operators: Sequence[str],
    parameter_prefix: str = "c",
) -> LoweredPySrExpression:
    parsed = _parse_safe_sympy(expression_source, feature_names=feature_names)
    try:
        expression = from_sympy(sp.simplify(parsed))
    except ContractValidationError as exc:
        raise ContractValidationError(
            code="unsupported_pysr_expression",
            message=exc.message,
            field_path=exc.field_path,
            details={"source_code": exc.code},
        ) from exc
    _require_known_symbols(expression, feature_names)
    parameterized, declarations, literal_trace = _parameterize_literals(
        expression,
        parameter_prefix=parameter_prefix,
    )
    _require_allowed_operators(parameterized, allowed_operators)
    return LoweredPySrExpression(
        expression=parameterized,
        parameter_declarations=tuple(declarations),
        lowering_trace={
            "expression_source": expression_source,
            "parameterized_constants": literal_trace,
            "allowed_operators": list(allowed_operators),
        },
    )


def build_pysr_cir_candidate(
    *,
    row: PySrHallOfFameRow,
    feature_names: Sequence[str],
    allowed_operators: Sequence[str],
    search_class: str,
    source_candidate_id: str,
    proposal_rank: int,
    transient_diagnostics: Mapping[str, Any] | None = None,
):
    lowered = lower_pysr_expression_to_expression_ir(
        expression_source=row.equation,
        feature_names=feature_names,
        allowed_operators=allowed_operators,
    )
    trace = {
        "hall_of_fame_row": {
            "equation": row.equation,
            "complexity": row.complexity,
            "loss": row.loss,
            "score": row.score,
            "metadata": dict(row.metadata or {}),
        },
        "lowering": dict(lowered.lowering_trace),
        **dict(transient_diagnostics or {}),
    }
    return build_cir_candidate_from_expression(
        expression=lowered.expression,
        cir_family_id="analytic",
        cir_form_class="pysr_symbolic_regression_expression_ir",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=tuple(feature_names),
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_symbolic_regression",
            horizon=1,
        ),
        model_code_decomposition=_model_code(lowered.expression),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="pysr-engine-v1",
            adapter_class="external_symbolic_engine",
            source_candidate_id=source_candidate_id,
            search_class=search_class,
            backend_family="pysr",
            proposal_rank=proposal_rank,
            backend_private_fields=(
                "pysr_trace",
                "hall_of_fame",
                "julia_runtime",
            ),
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(
                CIRReplayHook(
                    hook_name="pysr_engine_trace",
                    hook_ref=f"pysr:{sha256_digest(trace)}",
                ),
            )
        ),
        transient_diagnostics={"pysr_trace": trace},
    )


def _parse_safe_sympy(expression_source: str, *, feature_names: Sequence[str]):
    local_dict = {name: sp.Symbol(name, real=True) for name in feature_names}
    try:
        return parse_expr(
            expression_source,
            local_dict=local_dict,
            global_dict=_SAFE_GLOBALS,
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:
        raise ContractValidationError(
            code="unsafe_pysr_expression",
            message="PySR expression could not be parsed by the safe SymPy boundary",
            field_path="expression_source",
            details={"exception_type": type(exc).__name__},
        ) from exc


def _require_known_symbols(expression: Expr, feature_names: Sequence[str]) -> None:
    allowed = set(feature_names)
    unknown = sorted(
        {
            node.name
            for node in walk_expression(expression)
            if isinstance(node, Feature) and node.name not in allowed
        }
    )
    if unknown:
        raise ContractValidationError(
            code="unknown_pysr_symbol",
            message="PySR expression references symbols outside the legal feature view",
            field_path="expression_source",
            details={"unknown_symbols": unknown},
        )


def _parameterize_literals(
    expression: Expr,
    *,
    parameter_prefix: str,
    in_power_exponent: bool = False,
    counter: list[int] | None = None,
    declarations: list[ParameterDeclaration] | None = None,
    trace: list[dict[str, Any]] | None = None,
):
    counter = [0] if counter is None else counter
    declarations = [] if declarations is None else declarations
    trace = [] if trace is None else trace

    if isinstance(expression, Literal):
        if in_power_exponent or isinstance(expression.value, bool):
            return expression, declarations, trace
        try:
            value = float(expression.value)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(
                code="invalid_pysr_constant",
                message="PySR numeric constants must be finite",
                field_path="expression_source",
            ) from exc
        if not math.isfinite(value):
            raise ContractValidationError(
                code="invalid_pysr_constant",
                message="PySR numeric constants must be finite",
                field_path="expression_source",
            )
        name = f"{parameter_prefix}_{counter[0]:02d}"
        counter[0] += 1
        declarations.append(ParameterDeclaration(name, initial_value=value))
        trace.append({"parameter_name": name, "initial_value": value})
        return Parameter(name), declarations, trace
    if isinstance(expression, (Feature, Parameter)):
        return expression, declarations, trace
    if isinstance(expression, UnaryOp):
        operand, declarations, trace = _parameterize_literals(
            expression.operand,
            parameter_prefix=parameter_prefix,
            counter=counter,
            declarations=declarations,
            trace=trace,
        )
        return UnaryOp(expression.operator, operand), declarations, trace
    if isinstance(expression, BinaryOp):
        left, declarations, trace = _parameterize_literals(
            expression.left,
            parameter_prefix=parameter_prefix,
            counter=counter,
            declarations=declarations,
            trace=trace,
        )
        right, declarations, trace = _parameterize_literals(
            expression.right,
            parameter_prefix=parameter_prefix,
            in_power_exponent=expression.operator == "pow",
            counter=counter,
            declarations=declarations,
            trace=trace,
        )
        return BinaryOp(expression.operator, left, right), declarations, trace
    if isinstance(expression, NaryOp):
        children = []
        for child in expression.children:
            lowered_child, declarations, trace = _parameterize_literals(
                child,
                parameter_prefix=parameter_prefix,
                counter=counter,
                declarations=declarations,
                trace=trace,
            )
            children.append(lowered_child)
        return NaryOp(expression.operator, tuple(children)), declarations, trace
    raise ContractValidationError(
        code="unsupported_pysr_expression",
        message=f"{type(expression).__name__} is not supported for PySR lowering",
        field_path="expression_source",
    )


def _require_allowed_operators(
    expression: Expr,
    allowed_operators: Sequence[str],
) -> None:
    allowed = set(allowed_operators)
    for node in walk_expression(expression):
        operator = getattr(node, "operator", None)
        if operator is not None and operator not in allowed:
            raise ContractValidationError(
                code="disallowed_pysr_operator",
                message=f"PySR operator {operator!r} is not approved by Euclid",
                field_path="allowed_operators",
                details={"operator": operator, "allowed_operators": sorted(allowed)},
            )


def _model_code(expression: Expr) -> CIRModelCodeDecomposition:
    nodes = walk_expression(expression)
    parameter_count = sum(1 for node in nodes if isinstance(node, Parameter))
    literal_count = sum(1 for node in nodes if isinstance(node, Literal))
    return CIRModelCodeDecomposition(
        L_family_bits=2.0,
        L_structure_bits=float(len(nodes)),
        L_literals_bits=float(literal_count),
        L_params_bits=float(parameter_count),
        L_state_bits=0.0,
    )


__all__ = [
    "LoweredPySrExpression",
    "PySrHallOfFameRow",
    "build_pysr_cir_candidate",
    "lower_pysr_expression_to_expression_ir",
]
