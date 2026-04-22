from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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
from euclid.fit.parameterization import ParameterDeclaration
from euclid.runtime.hashing import sha256_digest

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_POWER = re.compile(r"^(?P<feature>[A-Za-z_][A-Za-z0-9_]*)\^(?P<power>\d+)$")
_FOURIER = re.compile(
    r"^(?P<operator>sin|cos)\((?:(?P<frequency>\d+(?:\.\d+)?)\s+)?(?P<feature>[A-Za-z_][A-Za-z0-9_]*)\)$"
)


@dataclass(frozen=True)
class PySindyTerm:
    term_name: str
    coefficient: float

    def __post_init__(self) -> None:
        term_name = str(self.term_name).strip()
        if not term_name:
            raise ContractValidationError(
                code="invalid_pysindy_term",
                message="PySINDy term names must be non-empty",
                field_path="term_name",
            )
        coefficient = float(self.coefficient)
        if not math.isfinite(coefficient):
            raise ContractValidationError(
                code="invalid_pysindy_coefficient",
                message="PySINDy coefficients must be finite",
                field_path="coefficient",
            )
        object.__setattr__(self, "term_name", term_name)
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True)
class PySindyDiscoveredEquation:
    output_name: str
    terms: tuple[PySindyTerm, ...]
    equation_text: str = ""

    def __post_init__(self) -> None:
        output_name = str(self.output_name).strip() or "target"
        object.__setattr__(self, "output_name", output_name)
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "equation_text", str(self.equation_text))


@dataclass(frozen=True)
class LoweredPySindyExpression:
    expression: Expr
    parameter_declarations: tuple[ParameterDeclaration, ...]
    active_terms: tuple[PySindyTerm, ...]
    lowering_trace: Mapping[str, Any]


def lower_pysindy_terms_to_expression_ir(
    *,
    terms: Sequence[PySindyTerm],
    feature_names: Sequence[str],
    coefficient_threshold: float = 1e-12,
    parameter_prefix: str = "theta",
) -> LoweredPySindyExpression:
    threshold = abs(float(coefficient_threshold))
    allowed_features = tuple(str(name) for name in feature_names)
    active_terms = tuple(term for term in terms if abs(term.coefficient) > threshold)
    if not active_terms:
        raise ContractValidationError(
            code="empty_pysindy_support",
            message="PySINDy output did not contain any active support terms",
            field_path="terms",
        )

    components: list[Expr] = []
    declarations: list[ParameterDeclaration] = []
    lowering_events: list[dict[str, Any]] = []
    for index, term in enumerate(active_terms):
        parameter_name = f"{parameter_prefix}_{index:02d}"
        term_expression = _parse_pysindy_term(term.term_name, allowed_features)
        parameter = Parameter(parameter_name)
        declarations.append(
            ParameterDeclaration(parameter_name, initial_value=term.coefficient)
        )
        component = (
            parameter
            if _is_one(term_expression)
            else BinaryOp("mul", parameter, term_expression)
        )
        components.append(component)
        lowering_events.append(
            {
                "term_name": term.term_name,
                "coefficient": term.coefficient,
                "parameter_name": parameter_name,
            }
        )

    expression = (
        components[0] if len(components) == 1 else NaryOp("add", tuple(components))
    )
    return LoweredPySindyExpression(
        expression=expression,
        parameter_declarations=tuple(declarations),
        active_terms=active_terms,
        lowering_trace={
            "active_term_count": len(active_terms),
            "coefficient_threshold": threshold,
            "terms": lowering_events,
        },
    )


def build_pysindy_cir_candidate(
    *,
    equation: PySindyDiscoveredEquation,
    feature_names: Sequence[str],
    search_class: str,
    source_candidate_id: str,
    proposal_rank: int,
    coefficient_threshold: float = 1e-12,
    transient_diagnostics: Mapping[str, Any] | None = None,
):
    lowered = lower_pysindy_terms_to_expression_ir(
        terms=equation.terms,
        feature_names=feature_names,
        coefficient_threshold=coefficient_threshold,
    )
    trace = {
        "equation_text": equation.equation_text,
        "output_name": equation.output_name,
        "lowering": dict(lowered.lowering_trace),
        **dict(transient_diagnostics or {}),
    }
    return build_cir_candidate_from_expression(
        expression=lowered.expression,
        cir_family_id="analytic",
        cir_form_class="pysindy_sparse_dynamics_expression_ir",
        input_signature=CIRInputSignature(
            target_series=equation.output_name,
            side_information_fields=tuple(feature_names),
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_sparse_dynamics",
            horizon=1,
        ),
        model_code_decomposition=_model_code(lowered.expression),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="pysindy-engine-v1",
            adapter_class="external_symbolic_engine",
            source_candidate_id=source_candidate_id,
            search_class=search_class,
            backend_family="pysindy",
            proposal_rank=proposal_rank,
            backend_private_fields=(
                "pysindy_trace",
                "coefficient_path",
                "support_mask",
            ),
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(
                CIRReplayHook(
                    hook_name="pysindy_engine_trace",
                    hook_ref=f"pysindy:{sha256_digest(trace)}",
                ),
            )
        ),
        transient_diagnostics={"pysindy_trace": trace},
    )


def _parse_pysindy_term(term_name: str, allowed_features: tuple[str, ...]) -> Expr:
    normalized = term_name.strip().replace("**", "^")
    if normalized == "1":
        return Literal(1.0)
    if normalized in allowed_features:
        return Feature(normalized)
    fourier = _FOURIER.match(normalized)
    if fourier is not None:
        feature_name = fourier.group("feature")
        _require_feature(feature_name, allowed_features, term_name=term_name)
        frequency = float(fourier.group("frequency") or 1.0)
        operand: Expr = Feature(feature_name)
        if frequency != 1.0:
            operand = BinaryOp("mul", Literal(frequency), operand)
        return UnaryOp(fourier.group("operator"), operand)
    power = _POWER.match(normalized)
    if power is not None:
        feature_name = power.group("feature")
        _require_feature(feature_name, allowed_features, term_name=term_name)
        return BinaryOp(
            "pow",
            Feature(feature_name),
            Literal(int(power.group("power"))),
        )
    if " " in normalized:
        factors = tuple(part for part in normalized.split(" ") if part)
        if not factors:
            raise _unsupported_term(term_name)
        return NaryOp(
            "mul",
            tuple(
                _parse_pysindy_term(factor, allowed_features)
                for factor in factors
            ),
        )
    if not _IDENTIFIER.match(normalized):
        raise _unsupported_term(term_name)
    raise _unsupported_term(term_name)


def _require_feature(
    feature_name: str,
    allowed_features: tuple[str, ...],
    *,
    term_name: str,
) -> None:
    if feature_name not in allowed_features:
        raise _unsupported_term(term_name)


def _unsupported_term(term_name: str) -> ContractValidationError:
    return ContractValidationError(
        code="unsupported_pysindy_term",
        message=f"PySINDy term {term_name!r} cannot be lowered to Euclid expression IR",
        field_path="term_name",
        details={"term_name": term_name},
    )


def _is_one(expression: Expr) -> bool:
    return isinstance(expression, Literal) and float(expression.value or 0.0) == 1.0


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
    "LoweredPySindyExpression",
    "PySindyDiscoveredEquation",
    "PySindyTerm",
    "build_pysindy_cir_candidate",
    "lower_pysindy_terms_to_expression_ir",
]
