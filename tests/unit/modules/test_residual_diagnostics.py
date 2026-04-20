from __future__ import annotations

import importlib
import importlib.util


def _module():
    spec = importlib.util.find_spec("euclid.modules.residual_diagnostics")
    assert spec is not None
    return importlib.import_module("euclid.modules.residual_diagnostics")


def test_evaluate_residual_diagnostics_marks_structured_residual_as_eligible() -> (
    None
):
    module = _module()
    result = module.evaluate_residual_diagnostics(
        module.ResidualDiagnosticsEvidence(
            finite_dimensionality=module.FiniteDimensionalityEvidence(
                sample_count=96,
                dimension_upper_bound=3,
                replicated_across_splits=True,
            ),
            recoverability=module.RecoverabilityEvidence(
                sample_count=96,
                state_reconstruction_passed=True,
                replicated_across_splits=True,
            ),
        )
    )

    assert result.status == "structured_residual_remains"
    assert result.residual_law_search_eligible is True
    assert result.finite_dimensionality_status == "supported"
    assert result.recoverability_status == "supported"
    assert result.reason_codes == ()


def test_noise_like_status_requires_explicit_negative_evidence() -> None:
    module = _module()
    result = module.evaluate_residual_diagnostics(
        module.ResidualDiagnosticsEvidence(
            finite_dimensionality=module.FiniteDimensionalityEvidence(
                sample_count=96,
                explicit_infinite_dimensionality_evidence=True,
            ),
            recoverability=module.RecoverabilityEvidence(
                sample_count=96,
                explicit_unrecoverable_evidence=True,
            ),
        )
    )

    assert result.status == "noise_like_residual"
    assert result.residual_law_search_eligible is False
    assert result.finite_dimensionality_status == "unsupported"
    assert result.recoverability_status == "unsupported"
    assert result.reason_codes == (
        "finite_dimensionality_rejected",
        "recoverability_rejected",
    )


def test_evaluate_residual_diagnostics_stays_indeterminate_when_evidence_is_incomplete(
) -> None:
    module = _module()
    result = module.evaluate_residual_diagnostics(
        module.ResidualDiagnosticsEvidence(
            finite_dimensionality=module.FiniteDimensionalityEvidence(
                sample_count=12,
                dimension_upper_bound=2,
                replicated_across_splits=True,
            ),
            recoverability=module.RecoverabilityEvidence(
                sample_count=12,
                state_reconstruction_passed=True,
                replicated_across_splits=True,
            ),
        )
    )

    assert result.status == "indeterminate"
    assert result.residual_law_search_eligible is False
    assert result.finite_dimensionality_status == "indeterminate"
    assert result.recoverability_status == "indeterminate"
    assert result.reason_codes == (
        "finite_dimensionality_insufficient_data",
        "recoverability_insufficient_data",
    )


def test_evaluate_residual_diagnostics_remains_indeterminate_on_conflicting_signals(
) -> None:
    module = _module()
    result = module.evaluate_residual_diagnostics(
        module.ResidualDiagnosticsEvidence(
            finite_dimensionality=module.FiniteDimensionalityEvidence(
                sample_count=96,
                dimension_upper_bound=2,
                replicated_across_splits=True,
            ),
            recoverability=module.RecoverabilityEvidence(
                sample_count=96,
                explicit_unrecoverable_evidence=True,
            ),
        )
    )

    assert result.status == "indeterminate"
    assert result.residual_law_search_eligible is False
    assert result.finite_dimensionality_status == "supported"
    assert result.recoverability_status == "unsupported"
    assert result.reason_codes == (
        "recoverability_rejected",
        "diagnostic_conflict",
    )


def test_assess_finite_dimensionality_requires_replication_before_support() -> None:
    module = _module()
    assessment = module.assess_finite_dimensionality(
        module.FiniteDimensionalityEvidence(
            sample_count=96,
            dimension_upper_bound=4,
            replicated_across_splits=False,
        )
    )

    assert assessment.status == "indeterminate"
    assert assessment.reason_codes == ("finite_dimensionality_unreplicated",)


def test_assess_recoverability_flags_failed_reconstruction_as_unsupported() -> None:
    module = _module()
    assessment = module.assess_recoverability(
        module.RecoverabilityEvidence(
            sample_count=96,
            state_reconstruction_passed=False,
        )
    )

    assert assessment.status == "unsupported"
    assert assessment.reason_codes == ("recoverability_failed",)
