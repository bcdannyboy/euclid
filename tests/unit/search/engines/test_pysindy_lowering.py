from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.expr.serialization import expression_to_dict
from euclid.fit.refit import FitDataSplit, fit_cir_candidate
from euclid.search.engines.pysindy_lowering import (
    PySindyDiscoveredEquation,
    PySindyTerm,
    build_pysindy_cir_candidate,
    lower_pysindy_terms_to_expression_ir,
)


def test_pysindy_active_terms_lower_to_expression_ir_cir_and_refit() -> None:
    lowered = lower_pysindy_terms_to_expression_ir(
        terms=(
            PySindyTerm(term_name="1", coefficient=1.0),
            PySindyTerm(term_name="lag_1", coefficient=2.0),
            PySindyTerm(term_name="lag_1^2", coefficient=0.0),
        ),
        feature_names=("lag_1",),
        coefficient_threshold=1e-12,
    )

    candidate = build_pysindy_cir_candidate(
        equation=PySindyDiscoveredEquation(
            output_name="target",
            terms=(
                PySindyTerm(term_name="1", coefficient=1.0),
                PySindyTerm(term_name="lag_1", coefficient=2.0),
            ),
            equation_text="1.000 1 + 2.000 lag_1",
        ),
        feature_names=("lag_1",),
        search_class="bounded_heuristic",
        source_candidate_id="pysindy-linear",
        proposal_rank=0,
        transient_diagnostics={"trace_id": "unit"},
    )

    payload = candidate.structural_layer.expression_payload
    assert payload is not None
    assert payload.feature_dependencies == ("lag_1",)
    assert payload.parameter_declarations == ("theta_00", "theta_01")
    assert (
        candidate.evidence_layer.backend_origin_record.adapter_id
        == "pysindy-engine-v1"
    )
    assert (
        candidate.evidence_layer.backend_origin_record.adapter_class
        == "external_symbolic_engine"
    )
    assert "pysindy_trace" in candidate.evidence_layer.transient_diagnostics
    assert expression_to_dict(lowered.expression) == payload.expression_tree

    fit = fit_cir_candidate(
        candidate=candidate,
        data=FitDataSplit(
            train_rows=(
                _row("2026-01-01T00:00:00Z", target=1.0, lag_1=0.0),
                _row("2026-01-02T00:00:00Z", target=3.0, lag_1=1.0),
                _row("2026-01-03T00:00:00Z", target=5.0, lag_1=2.0),
            )
        ),
        fit_window_id="pysindy-unit",
        parameter_declarations=lowered.parameter_declarations,
    )

    assert fit.status == "converged"
    assert fit.parameter_estimates["theta_00"] == pytest.approx(1.0)
    assert fit.parameter_estimates["theta_01"] == pytest.approx(2.0)
    assert fit.claim_boundary["claim_publication_allowed"] is False


def test_pysindy_lowering_rejects_invalid_terms_and_empty_support() -> None:
    with pytest.raises(ContractValidationError) as unknown_feature:
        lower_pysindy_terms_to_expression_ir(
            terms=(PySindyTerm(term_name="missing_feature", coefficient=1.0),),
            feature_names=("lag_1",),
        )

    assert unknown_feature.value.code == "unsupported_pysindy_term"

    with pytest.raises(ContractValidationError) as unsafe_term:
        lower_pysindy_terms_to_expression_ir(
            terms=(PySindyTerm(term_name="__import__('os')", coefficient=1.0),),
            feature_names=("lag_1",),
        )

    assert unsafe_term.value.code == "unsupported_pysindy_term"

    with pytest.raises(ContractValidationError) as no_support:
        lower_pysindy_terms_to_expression_ir(
            terms=(PySindyTerm(term_name="lag_1", coefficient=0.0),),
            feature_names=("lag_1",),
            coefficient_threshold=1e-12,
        )

    assert no_support.value.code == "empty_pysindy_support"


def _row(event_time: str, *, target: float, lag_1: float) -> dict[str, object]:
    return {
        "event_time": event_time,
        "available_at": event_time,
        "target": target,
        "lag_1": lag_1,
    }
