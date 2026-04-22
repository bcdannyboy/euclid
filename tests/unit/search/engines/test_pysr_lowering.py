from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.fit.refit import FitDataSplit, fit_cir_candidate
from euclid.search.engines.pysr_lowering import (
    PySrHallOfFameRow,
    build_pysr_cir_candidate,
    lower_pysr_expression_to_expression_ir,
)


def test_pysr_expression_lowers_to_expression_ir_cir_and_refits_constants() -> None:
    lowered = lower_pysr_expression_to_expression_ir(
        expression_source="2.0 * lag_1 + 1.0",
        feature_names=("lag_1",),
        allowed_operators=("add", "mul"),
    )
    candidate = build_pysr_cir_candidate(
        row=PySrHallOfFameRow(
            equation="2.0 * lag_1 + 1.0",
            complexity=5,
            loss=0.0,
            score=1.0,
        ),
        feature_names=("lag_1",),
        allowed_operators=("add", "mul"),
        search_class="bounded_heuristic",
        source_candidate_id="pysr-linear",
        proposal_rank=0,
        transient_diagnostics={"trace_id": "unit"},
    )

    payload = candidate.structural_layer.expression_payload
    assert payload is not None
    assert payload.feature_dependencies == ("lag_1",)
    assert payload.parameter_declarations == ("c_00", "c_01")
    assert candidate.evidence_layer.backend_origin_record.adapter_id == "pysr-engine-v1"
    assert "pysr_trace" in candidate.evidence_layer.transient_diagnostics

    fit = fit_cir_candidate(
        candidate=candidate,
        data=FitDataSplit(
            train_rows=(
                _row("2026-01-01T00:00:00Z", target=1.0, lag_1=0.0),
                _row("2026-01-02T00:00:00Z", target=3.0, lag_1=1.0),
                _row("2026-01-03T00:00:00Z", target=5.0, lag_1=2.0),
            )
        ),
        fit_window_id="pysr-unit",
        parameter_declarations=lowered.parameter_declarations,
    )

    assert fit.status == "converged"
    assert sorted(fit.parameter_estimates.values()) == pytest.approx([1.0, 2.0])
    assert fit.claim_boundary["claim_publication_allowed"] is False


def test_pysr_lowering_rejects_unsafe_expressions_and_disallowed_operators() -> None:
    with pytest.raises(ContractValidationError) as unsafe:
        lower_pysr_expression_to_expression_ir(
            expression_source="__import__('os').system('echo bad')",
            feature_names=("lag_1",),
            allowed_operators=("add", "mul"),
        )

    assert unsafe.value.code == "unsafe_pysr_expression"

    with pytest.raises(ContractValidationError) as disallowed:
        lower_pysr_expression_to_expression_ir(
            expression_source="log(lag_1)",
            feature_names=("lag_1",),
            allowed_operators=("add", "mul"),
        )

    assert disallowed.value.code == "disallowed_pysr_operator"


def test_pysr_duplicate_forms_have_same_canonical_cir_identity() -> None:
    first = build_pysr_cir_candidate(
        row=PySrHallOfFameRow(equation="2.0 * lag_1 + 1.0", complexity=5),
        feature_names=("lag_1",),
        allowed_operators=("add", "mul"),
        search_class="bounded_heuristic",
        source_candidate_id="first",
        proposal_rank=0,
    )
    second = build_pysr_cir_candidate(
        row=PySrHallOfFameRow(equation="1.0 + lag_1 * 2.0", complexity=5),
        feature_names=("lag_1",),
        allowed_operators=("add", "mul"),
        search_class="bounded_heuristic",
        source_candidate_id="second",
        proposal_rank=1,
    )

    assert first.canonical_hash() == second.canonical_hash()


def _row(event_time: str, *, target: float, lag_1: float) -> dict[str, object]:
    return {
        "event_time": event_time,
        "available_at": event_time,
        "target": target,
        "lag_1": lag_1,
    }
