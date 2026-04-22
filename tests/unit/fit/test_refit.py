from __future__ import annotations

import math

import pytest

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
from euclid.expr.ast import BinaryOp, Feature, Parameter, UnaryOp
from euclid.fit.parameterization import ParameterBounds, ParameterDeclaration
from euclid.fit.refit import FitDataSplit, fit_cir_candidate


def test_unified_refit_uses_training_split_only_and_blocks_direct_claims() -> None:
    candidate = _linear_expression_candidate()
    split = FitDataSplit(
        train_rows=(
            _row("2026-01-01T00:00:00Z", target=1.0, lag_1=0.0),
            _row("2026-01-02T00:00:00Z", target=3.0, lag_1=1.0),
            _row("2026-01-03T00:00:00Z", target=5.0, lag_1=2.0),
        ),
        validation_rows=(
            _row("2026-01-04T00:00:00Z", target=999.0, lag_1=3.0),
        ),
        test_rows=(
            _row("2026-01-05T00:00:00Z", target=999.0, lag_1=4.0),
        ),
    )

    result = fit_cir_candidate(
        candidate=candidate,
        data=split,
        fit_window_id="outer_fold_0",
        parameter_declarations=(
            ParameterDeclaration("intercept", initial_value=0.0),
            ParameterDeclaration("slope", initial_value=0.0),
        ),
        seed=23,
    )

    assert result.status == "converged"
    assert result.parameter_estimates["intercept"] == pytest.approx(1.0)
    assert result.parameter_estimates["slope"] == pytest.approx(2.0)
    assert result.split_counts == {"train": 3, "validation": 1, "test": 1}
    assert [row["event_time"] for row in result.rows_used] == [
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
        "2026-01-03T00:00:00Z",
    ]
    assert result.claim_boundary["claim_publication_allowed"] is False
    assert "fit_is_not_claim_authority" in result.claim_boundary["reason_codes"]


def test_unified_refit_detects_underdetermined_singular_system() -> None:
    result = fit_cir_candidate(
        candidate=_linear_expression_candidate(),
        data=FitDataSplit(
            train_rows=(
                _row("2026-01-01T00:00:00Z", target=1.0, lag_1=0.0),
            )
        ),
        fit_window_id="outer_fold_0",
        parameter_declarations=(
            ParameterDeclaration("intercept", initial_value=0.0),
            ParameterDeclaration("slope", initial_value=0.0),
        ),
        seed=23,
    )

    assert result.status == "failed"
    assert "underdetermined_system" in result.failure_reasons
    assert result.claim_boundary["claim_publication_allowed"] is False


def test_unified_refit_fails_closed_for_bad_rows_and_domain_errors() -> None:
    with pytest.raises(ContractValidationError, match="duplicate"):
        fit_cir_candidate(
            candidate=_linear_expression_candidate(),
            data=FitDataSplit(
                train_rows=(
                    _row("2026-01-01T00:00:00Z", target=1.0, lag_1=0.0),
                    _row("2026-01-01T00:00:00Z", target=2.0, lag_1=1.0),
                )
            ),
            fit_window_id="outer_fold_0",
            parameter_declarations=(ParameterDeclaration("intercept", initial_value=0.0),),
        )

    with pytest.raises(ContractValidationError, match="out-of-order"):
        fit_cir_candidate(
            candidate=_linear_expression_candidate(),
            data=FitDataSplit(
                train_rows=(
                    _row("2026-01-02T00:00:00Z", target=2.0, lag_1=1.0),
                    _row("2026-01-01T00:00:00Z", target=1.0, lag_1=0.0),
                )
            ),
            fit_window_id="outer_fold_0",
            parameter_declarations=(ParameterDeclaration("intercept", initial_value=0.0),),
        )

    with pytest.raises(ContractValidationError, match="finite"):
        fit_cir_candidate(
            candidate=_linear_expression_candidate(),
            data=FitDataSplit(
                train_rows=(
                    _row("2026-01-01T00:00:00Z", target=math.inf, lag_1=0.0),
                )
            ),
            fit_window_id="outer_fold_0",
            parameter_declarations=(ParameterDeclaration("intercept", initial_value=0.0),),
        )

    with pytest.raises(ContractValidationError, match="domain"):
        fit_cir_candidate(
            candidate=_log_expression_candidate(),
            data=FitDataSplit(
                train_rows=(
                    _row("2026-01-01T00:00:00Z", target=1.0, lag_1=-1.0),
                )
            ),
            fit_window_id="outer_fold_0",
            parameter_declarations=(),
        )


def test_unified_refit_replay_identity_and_redacted_evidence_are_stable() -> None:
    candidate = _linear_expression_candidate()
    split = FitDataSplit(
        train_rows=(
            _row("2026-01-01T00:00:00Z", target=1.0, lag_1=0.0),
            _row("2026-01-02T00:00:00Z", target=3.0, lag_1=1.0),
            _row("2026-01-03T00:00:00Z", target=5.0, lag_1=2.0),
        )
    )
    kwargs = {
        "candidate": candidate,
        "data": split,
        "fit_window_id": "outer_fold_0",
        "parameter_declarations": (
            ParameterDeclaration(
                "intercept",
                initial_value=0.0,
                bounds=ParameterBounds(lower=-10.0, upper=10.0),
            ),
            ParameterDeclaration("slope", initial_value=0.0),
        ),
        "seed": 99,
    }

    first = fit_cir_candidate(**kwargs)
    second = fit_cir_candidate(**kwargs)

    assert first.replay_identity == second.replay_identity
    assert first.parameter_estimates == second.parameter_estimates
    redacted = first.as_redacted_evidence()
    assert redacted["replay_identity"] == first.replay_identity
    assert "train_target_values" not in redacted
    assert "raw_rows" not in redacted


def _linear_expression_candidate():
    expression = BinaryOp(
        "add",
        BinaryOp("mul", Parameter("slope"), Feature("lag_1")),
        Parameter("intercept"),
    )
    return build_cir_candidate_from_expression(
        expression=expression,
        cir_family_id="analytic",
        cir_form_class="expression_ir",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("lag_1",),
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=CIRModelCodeDecomposition(
            L_family_bits=1.0,
            L_structure_bits=1.0,
            L_literals_bits=0.0,
            L_params_bits=2.0,
            L_state_bits=0.0,
        ),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="test-expression",
            adapter_class="unit_test",
            source_candidate_id="linear-expression",
            search_class="unit_test",
            proposal_rank=0,
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(CIRReplayHook(hook_name="test", hook_ref="unit:test"),)
        ),
    )


def _log_expression_candidate():
    expression = UnaryOp("log", Feature("lag_1"))
    return build_cir_candidate_from_expression(
        expression=expression,
        cir_family_id="analytic",
        cir_form_class="expression_ir",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("lag_1",),
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=CIRModelCodeDecomposition(
            L_family_bits=1.0,
            L_structure_bits=1.0,
            L_literals_bits=0.0,
            L_params_bits=0.0,
            L_state_bits=0.0,
        ),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="test-expression",
            adapter_class="unit_test",
            source_candidate_id="log-expression",
            search_class="unit_test",
            proposal_rank=0,
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(CIRReplayHook(hook_name="test", hook_ref="unit:test"),)
        ),
    )


def _row(event_time: str, *, target: float, lag_1: float) -> dict[str, object]:
    return {
        "event_time": event_time,
        "available_at": event_time,
        "target": target,
        "lag_1": lag_1,
    }
