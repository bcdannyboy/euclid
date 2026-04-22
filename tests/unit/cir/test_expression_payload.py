from __future__ import annotations

from euclid.cir.models import (
    CIRBackendOriginRecord,
    CIRForecastOperator,
    CIRInputSignature,
    CIRModelCodeDecomposition,
    CIRReplayHooks,
)
from euclid.cir.normalize import build_cir_candidate_from_expression
from euclid.expr.ast import BinaryOp, Feature, Literal, NaryOp


def test_expression_payload_is_part_of_cir_identity_but_backend_origin_is_not() -> None:
    left_expression = NaryOp("add", (Feature("x"), Literal(1.0)))
    right_expression = NaryOp("add", (Literal(1.0), Feature("x")))

    left = build_cir_candidate_from_expression(
        expression=left_expression,
        cir_family_id="analytic",
        cir_form_class="closed_form_expression",
        input_signature=CIRInputSignature(target_series="target"),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=_code(),
        backend_origin_record=_origin("pysr", "candidate-a"),
        replay_hooks=CIRReplayHooks(),
    )
    right = build_cir_candidate_from_expression(
        expression=right_expression,
        cir_family_id="analytic",
        cir_form_class="closed_form_expression",
        input_signature=CIRInputSignature(target_series="target"),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=_code(),
        backend_origin_record=_origin("pysindy", "candidate-b"),
        replay_hooks=CIRReplayHooks(),
    )

    assert left.canonical_hash() == right.canonical_hash()
    assert left.structural_layer.expression_payload is not None
    assert "expression_canonical_hash" in left.structural_layer.expression_payload.as_dict()
    assert left.evidence_layer.backend_origin_record.adapter_id == "pysr"
    assert right.evidence_layer.backend_origin_record.adapter_id == "pysindy"


def test_expression_assumptions_and_units_are_part_of_cir_identity() -> None:
    expression = BinaryOp("pow", Feature("x"), Literal(0.5))

    unconstrained = build_cir_candidate_from_expression(
        expression=expression,
        cir_family_id="analytic",
        cir_form_class="closed_form_expression",
        input_signature=CIRInputSignature(target_series="target"),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=_code(),
        backend_origin_record=_origin("native", "sqrt-unconstrained"),
        replay_hooks=CIRReplayHooks(),
    )
    positive = build_cir_candidate_from_expression(
        expression=expression,
        cir_family_id="analytic",
        cir_form_class="closed_form_expression",
        input_signature=CIRInputSignature(target_series="target"),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=_code(),
        backend_origin_record=_origin("native", "sqrt-positive"),
        replay_hooks=CIRReplayHooks(),
        assumptions={"x": {"domain": "positive_real", "unit": "meter"}},
        unit_constraints={"x": "meter"},
    )

    assert unconstrained.canonical_hash() != positive.canonical_hash()


def _code() -> CIRModelCodeDecomposition:
    return CIRModelCodeDecomposition(
        L_family_bits=1.0,
        L_structure_bits=1.0,
        L_literals_bits=1.0,
        L_params_bits=0.0,
        L_state_bits=0.0,
    )


def _origin(adapter_id: str, candidate_id: str) -> CIRBackendOriginRecord:
    return CIRBackendOriginRecord(
        adapter_id=adapter_id,
        adapter_class="external_symbolic_engine",
        source_candidate_id=candidate_id,
        search_class="external_engine_proposal",
    )

