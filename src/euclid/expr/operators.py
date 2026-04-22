from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from euclid.contracts.errors import ContractValidationError


@dataclass(frozen=True)
class OperatorMetadata:
    name: str
    arity: int | tuple[int, int | None]
    input_domains: tuple[str, ...]
    output_domain: str
    unit_rule: str
    differentiability: str
    monotonicity: str | None
    commutative: bool
    associative: bool
    identity_element: float | None
    absorbing_element: float | None
    singularities: tuple[str, ...]
    safe_evaluation: str
    sympy_name: str | None
    numpy_evaluator: str | None

    def allows_arity(self, arity: int) -> bool:
        if isinstance(self.arity, int):
            return arity == self.arity
        lower, upper = self.arity
        if arity < lower:
            return False
        return upper is None or arity <= upper

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arity": self.arity,
            "input_domains": list(self.input_domains),
            "output_domain": self.output_domain,
            "unit_rule": self.unit_rule,
            "differentiability": self.differentiability,
            "monotonicity": self.monotonicity,
            "commutative": self.commutative,
            "associative": self.associative,
            "identity_element": self.identity_element,
            "absorbing_element": self.absorbing_element,
            "singularities": list(self.singularities),
            "safe_evaluation": self.safe_evaluation,
            "sympy_name": self.sympy_name,
            "numpy_evaluator": self.numpy_evaluator,
        }


def _metadata(
    name: str,
    *,
    arity: int | tuple[int, int | None],
    input_domains: tuple[str, ...] = ("real",),
    output_domain: str = "real",
    unit_rule: str = "dimensionless",
    differentiability: str = "piecewise_smooth",
    monotonicity: str | None = None,
    commutative: bool = False,
    associative: bool = False,
    identity_element: float | None = None,
    absorbing_element: float | None = None,
    singularities: tuple[str, ...] = (),
    safe_evaluation: str = "strict",
    sympy_name: str | None = None,
    numpy_evaluator: str | None = None,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        arity=arity,
        input_domains=input_domains,
        output_domain=output_domain,
        unit_rule=unit_rule,
        differentiability=differentiability,
        monotonicity=monotonicity,
        commutative=commutative,
        associative=associative,
        identity_element=identity_element,
        absorbing_element=absorbing_element,
        singularities=singularities,
        safe_evaluation=safe_evaluation,
        sympy_name=sympy_name or name,
        numpy_evaluator=numpy_evaluator or name,
    )


OPERATOR_REGISTRY: dict[str, OperatorMetadata] = {
    "add": _metadata(
        "add",
        arity=(2, None),
        unit_rule="same_dimension",
        differentiability="smooth",
        commutative=True,
        associative=True,
        identity_element=0.0,
        sympy_name="Add",
        numpy_evaluator="add",
    ),
    "sub": _metadata(
        "sub",
        arity=2,
        unit_rule="same_dimension",
        differentiability="smooth",
        sympy_name="Add",
        numpy_evaluator="subtract",
    ),
    "mul": _metadata(
        "mul",
        arity=(2, None),
        unit_rule="multiply",
        differentiability="smooth",
        commutative=True,
        associative=True,
        identity_element=1.0,
        absorbing_element=0.0,
        sympy_name="Mul",
        numpy_evaluator="multiply",
    ),
    "div": _metadata(
        "div",
        arity=2,
        unit_rule="divide",
        differentiability="smooth_except_singularities",
        singularities=("denominator_zero",),
        sympy_name="Mul",
        numpy_evaluator="divide",
    ),
    "protected_div": _metadata(
        "protected_div",
        arity=2,
        unit_rule="divide",
        differentiability="protected_piecewise",
        singularities=("denominator_near_zero",),
        safe_evaluation="protected",
        sympy_name="Piecewise",
        numpy_evaluator="where",
    ),
    "neg": _metadata("neg", arity=1, unit_rule="preserve", differentiability="smooth", sympy_name="Mul", numpy_evaluator="negative"),
    "abs": _metadata("abs", arity=1, unit_rule="preserve", differentiability="nondifferentiable_at_zero", sympy_name="Abs", numpy_evaluator="abs"),
    "min": _metadata("min", arity=(2, None), unit_rule="same_dimension", commutative=True, associative=True, sympy_name="Min", numpy_evaluator="minimum"),
    "max": _metadata("max", arity=(2, None), unit_rule="same_dimension", commutative=True, associative=True, sympy_name="Max", numpy_evaluator="maximum"),
    "pow": _metadata("pow", arity=2, unit_rule="power", differentiability="smooth_except_singularities", singularities=("invalid_base_exponent",), sympy_name="Pow", numpy_evaluator="power"),
    "pow2": _metadata("pow2", arity=1, unit_rule="square", differentiability="smooth", sympy_name="Pow", numpy_evaluator="square"),
    "sqrt": _metadata("sqrt", arity=1, input_domains=("nonnegative_real",), unit_rule="sqrt", differentiability="smooth_positive_only", singularities=("negative_radicand",), sympy_name="sqrt", numpy_evaluator="sqrt"),
    "exp": _metadata("exp", arity=1, unit_rule="require_dimensionless", differentiability="smooth", sympy_name="exp", numpy_evaluator="exp"),
    "log": _metadata("log", arity=1, input_domains=("positive_real",), unit_rule="require_dimensionless", differentiability="smooth_positive_only", singularities=("nonpositive_argument",), sympy_name="log", numpy_evaluator="log"),
    "protected_log": _metadata("protected_log", arity=1, input_domains=("real",), unit_rule="require_dimensionless", differentiability="protected_piecewise", singularities=("nonpositive_argument",), safe_evaluation="protected", sympy_name="log", numpy_evaluator="log"),
    "sin": _metadata("sin", arity=1, unit_rule="require_dimensionless", differentiability="smooth", sympy_name="sin", numpy_evaluator="sin"),
    "cos": _metadata("cos", arity=1, unit_rule="require_dimensionless", differentiability="smooth", sympy_name="cos", numpy_evaluator="cos"),
    "tan": _metadata("tan", arity=1, unit_rule="require_dimensionless", differentiability="smooth_except_singularities", singularities=("cos_zero",), sympy_name="tan", numpy_evaluator="tan"),
    "tanh": _metadata("tanh", arity=1, unit_rule="require_dimensionless", differentiability="smooth", sympy_name="tanh", numpy_evaluator="tanh"),
    "sigmoid": _metadata("sigmoid", arity=1, unit_rule="require_dimensionless", output_domain="probability", differentiability="smooth", sympy_name="sigmoid", numpy_evaluator="expit"),
    "logit": _metadata("logit", arity=1, input_domains=("probability",), unit_rule="require_dimensionless", differentiability="smooth_open_interval", singularities=("outside_unit_interval",), sympy_name="log", numpy_evaluator="logit"),
    "floor": _metadata("floor", arity=1, differentiability="none", sympy_name="floor", numpy_evaluator="floor"),
    "ceil": _metadata("ceil", arity=1, differentiability="none", sympy_name="ceiling", numpy_evaluator="ceil"),
    "clip": _metadata("clip", arity=3, unit_rule="same_dimension", differentiability="piecewise_smooth", sympy_name="Min", numpy_evaluator="clip"),
    "where": _metadata("where", arity=3, unit_rule="branch_same_dimension", differentiability="piecewise_smooth", sympy_name="Piecewise", numpy_evaluator="where"),
    "gt": _metadata("gt", arity=2, output_domain="boolean", unit_rule="same_dimension", differentiability="none", sympy_name="StrictGreaterThan", numpy_evaluator="greater"),
    "lt": _metadata("lt", arity=2, output_domain="boolean", unit_rule="same_dimension", differentiability="none", sympy_name="StrictLessThan", numpy_evaluator="less"),
    "ge": _metadata("ge", arity=2, output_domain="boolean", unit_rule="same_dimension", differentiability="none", sympy_name="GreaterThan", numpy_evaluator="greater_equal"),
    "le": _metadata("le", arity=2, output_domain="boolean", unit_rule="same_dimension", differentiability="none", sympy_name="LessThan", numpy_evaluator="less_equal"),
    "eq": _metadata("eq", arity=2, output_domain="boolean", unit_rule="same_dimension", differentiability="none", sympy_name="Eq", numpy_evaluator="equal"),
    "and": _metadata("and", arity=(2, None), input_domains=("boolean",), output_domain="boolean", unit_rule="boolean", differentiability="none", sympy_name="And", numpy_evaluator="logical_and"),
    "or": _metadata("or", arity=(2, None), input_domains=("boolean",), output_domain="boolean", unit_rule="boolean", differentiability="none", sympy_name="Or", numpy_evaluator="logical_or"),
    "not": _metadata("not", arity=1, input_domains=("boolean",), output_domain="boolean", unit_rule="boolean", differentiability="none", sympy_name="Not", numpy_evaluator="logical_not"),
    "lag": _metadata("lag", arity=1, unit_rule="preserve", differentiability="not_symbolic", sympy_name="Function", numpy_evaluator="lag"),
    "finite_difference": _metadata("finite_difference", arity=1, unit_rule="preserve_per_step", differentiability="not_symbolic", sympy_name="Function", numpy_evaluator="diff"),
    "rolling_mean": _metadata("rolling_mean", arity=1, unit_rule="preserve", differentiability="linear_filter", sympy_name="Function", numpy_evaluator="mean"),
    "rolling_sum": _metadata("rolling_sum", arity=1, unit_rule="preserve", differentiability="linear_filter", sympy_name="Function", numpy_evaluator="sum"),
    "cumulative_sum": _metadata("cumulative_sum", arity=1, unit_rule="preserve", differentiability="linear_filter", sympy_name="Function", numpy_evaluator="cumsum"),
    "convolution": _metadata("convolution", arity=2, unit_rule="multiply", differentiability="linear_filter", sympy_name="Function", numpy_evaluator="convolve"),
    "seasonal_phase": _metadata("seasonal_phase", arity=1, output_domain="bounded_interval", unit_rule="dimensionless", differentiability="none", sympy_name="Function", numpy_evaluator="mod"),
    "derivative_estimate": _metadata("derivative_estimate", arity=1, unit_rule="per_time", differentiability="linear_filter", sympy_name="Derivative", numpy_evaluator="gradient"),
    "integral_estimate": _metadata("integral_estimate", arity=1, unit_rule="times_time", differentiability="linear_filter", sympy_name="Integral", numpy_evaluator="trapz"),
    "location_parameter": _metadata("location_parameter", arity=1, unit_rule="preserve", differentiability="smooth", sympy_name="Function", numpy_evaluator="identity"),
    "scale_parameter": _metadata("scale_parameter", arity=1, input_domains=("positive_real",), unit_rule="preserve", differentiability="smooth_positive_only", sympy_name="Function", numpy_evaluator="identity"),
    "rate_parameter": _metadata("rate_parameter", arity=1, input_domains=("positive_real",), unit_rule="inverse_time", differentiability="smooth_positive_only", sympy_name="Function", numpy_evaluator="identity"),
    "probability_parameter": _metadata("probability_parameter", arity=1, input_domains=("probability",), output_domain="probability", unit_rule="require_dimensionless", differentiability="smooth_open_interval", sympy_name="Function", numpy_evaluator="identity"),
    "dispersion_parameter": _metadata("dispersion_parameter", arity=1, input_domains=("positive_real",), unit_rule="preserve", differentiability="smooth_positive_only", sympy_name="Function", numpy_evaluator="identity"),
}


def get_operator(name: str) -> OperatorMetadata:
    try:
        return OPERATOR_REGISTRY[name]
    except KeyError as exc:
        raise ContractValidationError(
            code="unsupported_expression_operator",
            message=f"{name!r} is not registered in the expression operator registry",
            field_path="operator",
            details={"operator": name},
        ) from exc


__all__ = ["OPERATOR_REGISTRY", "OperatorMetadata", "get_operator"]
