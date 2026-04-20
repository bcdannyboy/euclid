from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DEFAULT_FRONTIER_AXES": ("euclid.search.policies", "DEFAULT_FRONTIER_AXES"),
    "DEFAULT_FORBIDDEN_FRONTIER_AXES": (
        "euclid.search.policies",
        "DEFAULT_FORBIDDEN_FRONTIER_AXES",
    ),
    "DEFAULT_SEARCH_CLASS": ("euclid.search.policies", "DEFAULT_SEARCH_CLASS"),
    "FrontierPolicy": ("euclid.search.policies", "FrontierPolicy"),
    "AlgorithmicSearchBackendAdapter": (
        "euclid.search.backends",
        "AlgorithmicSearchBackendAdapter",
    ),
    "AnalyticSearchBackendAdapter": (
        "euclid.search.backends",
        "AnalyticSearchBackendAdapter",
    ),
    "DescriptionGainArtifact": ("euclid.search.backends", "DescriptionGainArtifact"),
    "DescriptiveAdmissibilityDiagnostic": (
        "euclid.search.backends",
        "DescriptiveAdmissibilityDiagnostic",
    ),
    "DescriptiveSearchFrontierResult": (
        "euclid.search.backends",
        "DescriptiveSearchFrontierResult",
    ),
    "DescriptiveSearchProposal": (
        "euclid.search.backends",
        "DescriptiveSearchProposal",
    ),
    "DescriptiveSearchRunResult": (
        "euclid.search.backends",
        "DescriptiveSearchRunResult",
    ),
    "FamilySearchBackendResult": (
        "euclid.search.backends",
        "FamilySearchBackendResult",
    ),
    "FrontierCandidateMetrics": (
        "euclid.search.frontier",
        "FrontierCandidateMetrics",
    ),
    "FrontierDominanceRecord": (
        "euclid.search.frontier",
        "FrontierDominanceRecord",
    ),
    "ParallelBudgetPolicy": ("euclid.search.policies", "ParallelBudgetPolicy"),
    "PortfolioBackendLedger": (
        "euclid.search.portfolio",
        "PortfolioBackendLedger",
    ),
    "PortfolioCandidateLedgerEntry": (
        "euclid.search.portfolio",
        "PortfolioCandidateLedgerEntry",
    ),
    "PortfolioSelectionRecord": (
        "euclid.search.portfolio",
        "PortfolioSelectionRecord",
    ),
    "RecursiveSearchBackendAdapter": (
        "euclid.search.backends",
        "RecursiveSearchBackendAdapter",
    ),
    "RETAINED_COMPOSITION_OPERATORS": (
        "euclid.search.policies",
        "RETAINED_COMPOSITION_OPERATORS",
    ),
    "RETAINED_PRIMITIVE_FAMILIES": (
        "euclid.search.policies",
        "RETAINED_PRIMITIVE_FAMILIES",
    ),
    "RejectedSearchDiagnostic": ("euclid.search.backends", "RejectedSearchDiagnostic"),
    "DescriptiveSearchPortfolioResult": (
        "euclid.search.portfolio",
        "DescriptiveSearchPortfolioResult",
    ),
    "run_descriptive_search_portfolio": (
        "euclid.search.portfolio",
        "run_descriptive_search_portfolio",
    ),
    "run_descriptive_search_backends": (
        "euclid.search.backends",
        "run_descriptive_search_backends",
    ),
    "SearchCoverageAccounting": ("euclid.search.backends", "SearchCoverageAccounting"),
    "SearchBudgetPolicy": ("euclid.search.policies", "SearchBudgetPolicy"),
    "SeedPolicy": ("euclid.search.policies", "SeedPolicy"),
    "SpectralSearchBackendAdapter": (
        "euclid.search.backends",
        "SpectralSearchBackendAdapter",
    ),
    "StageLocalFrontierCoverage": (
        "euclid.search.frontier",
        "StageLocalFrontierCoverage",
    ),
    "StageLocalFrontierResult": ("euclid.search.frontier", "StageLocalFrontierResult"),
    "construct_stage_local_frontier": (
        "euclid.search.frontier",
        "construct_stage_local_frontier",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
