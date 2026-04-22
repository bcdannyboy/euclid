from euclid.falsification.counterexamples import (
    CounterexampleSearchResult,
    discover_counterexamples,
)
from euclid.falsification.dossier import (
    FalsificationDossier,
    build_falsification_dossier,
)
from euclid.falsification.perturbations import (
    PerturbationStabilityResult,
    evaluate_perturbation_stability,
)
from euclid.falsification.residuals import (
    ResidualDiagnosticResult,
    evaluate_residual_diagnostics,
)
from euclid.falsification.surrogate_tests import (
    ParameterStabilityResult,
    SurrogateResidualTestResult,
    evaluate_parameter_stability,
    evaluate_surrogate_residual_test,
)

__all__ = [
    "CounterexampleSearchResult",
    "FalsificationDossier",
    "ParameterStabilityResult",
    "PerturbationStabilityResult",
    "ResidualDiagnosticResult",
    "SurrogateResidualTestResult",
    "build_falsification_dossier",
    "discover_counterexamples",
    "evaluate_parameter_stability",
    "evaluate_perturbation_stability",
    "evaluate_residual_diagnostics",
    "evaluate_surrogate_residual_test",
]
