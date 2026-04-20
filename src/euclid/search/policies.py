from __future__ import annotations

from dataclasses import dataclass

RETAINED_PRIMITIVE_FAMILIES: tuple[str, ...] = (
    "analytic",
    "recursive",
    "spectral",
    "algorithmic",
)
RETAINED_COMPOSITION_OPERATORS: tuple[str, ...] = (
    "piecewise",
    "additive_residual",
    "regime_conditioned",
)
DEFAULT_SEARCH_CLASS = "bounded_heuristic"
DEFAULT_FRONTIER_AXES: tuple[str, ...] = (
    "structure_code_bits",
    "description_gain_bits",
    "inner_primary_score",
)
SUPPORTED_FRONTIER_AXES: tuple[str, ...] = DEFAULT_FRONTIER_AXES
DEFAULT_FORBIDDEN_FRONTIER_AXES: tuple[str, ...] = (
    "holdout_results",
    "outer_fold_results",
    "null_results",
    "robustness_results",
)
DESCRIPTIVE_SCOPE_SELECTION_RULE = (
    "min_total_code_bits_then_max_description_gain_then_"
    "min_structure_code_bits_then_min_canonical_byte_length_then_candidate_id"
)
COMPACT_DESCRIPTIVE_FALLBACK_CANDIDATE_IDS: tuple[str, ...] = (
    "analytic_intercept",
    "recursive_level_smoother",
    "recursive_running_mean",
    "algorithmic_last_observation",
    "algorithmic_running_half_average",
)
BANNED_DESCRIPTIVE_SCOPE_COMPOSITION_OPERATORS: tuple[str, ...] = (
    "additive_residual",
)
BANNED_DESCRIPTIVE_SCOPE_FORM_CLASSES: dict[str, str] = {
    "exact_sample_closure": "requires_exact_sample_closure",
    "posthoc_symbolic_synthesis": "requires_posthoc_symbolic_synthesis",
}
FULL_COMPARABILITY_SEARCH_CLASSES: tuple[str, ...] = (
    "exact_finite_enumeration",
    "bounded_heuristic",
)
REQUIRED_SEARCH_COVERAGE_DISCLOSURES: dict[str, tuple[str, ...]] = {
    "exact_finite_enumeration": (
        "canonical_enumerator",
        "enumeration_cardinality",
        "stop_rule",
    ),
    "bounded_heuristic": (
        "proposer_mechanism",
        "pruning_rules",
        "stop_rule",
    ),
    "equality_saturation_heuristic": (
        "rewrite_system",
        "extractor_cost",
        "stop_rule",
    ),
    "stochastic_heuristic": (
        "proposal_distribution",
        "seed_policy",
        "restart_policy",
        "stop_rule",
    ),
}
REQUIRED_CANDIDATE_SEARCH_EVIDENCE: dict[str, tuple[str, ...]] = {
    "exact_finite_enumeration": (
        "search_class",
        "canonical_enumerator",
        "enumeration_cardinality",
        "exactness_scope",
    ),
    "bounded_heuristic": (
        "search_class",
        "proposer_mechanism",
        "pruning_rules",
        "stop_rule",
        "exactness_scope",
    ),
    "equality_saturation_heuristic": (
        "search_class",
        "rewrite_system",
        "extractor_cost",
        "rewrite_space_candidate_ids",
        "selected_candidate_id",
        "stop_rule",
        "exactness_scope",
    ),
    "stochastic_heuristic": (
        "search_class",
        "proposal_distribution",
        "seed_policy",
        "restart_policy",
        "seed_scopes",
        "restart_records",
        "declared_stochastic_surfaces",
        "stop_rule",
        "exactness_scope",
    ),
}


@dataclass(frozen=True)
class SearchBudgetPolicy:
    proposal_limit: int
    frontier_width: int
    shortlist_limit: int = 1
    wall_clock_budget_seconds: int = 1
    budget_accounting_rule: str = "proposal_count_then_candidate_id_tie_break"


@dataclass(frozen=True)
class ParallelBudgetPolicy:
    max_worker_count: int = 1
    candidate_batch_size: int = 1
    aggregation_rule: str = "deterministic_candidate_id_order"


@dataclass(frozen=True)
class SeedPolicy:
    root_seed: str = "0"
    seed_derivation_rule: str = "deterministic_scope_hash"
    seed_scopes: tuple[str, ...] = ("search", "candidate_generation", "tie_break")


@dataclass(frozen=True)
class FrontierPolicy:
    frontier_id: str = "retained_scope_search_frontier_v1"
    axes: tuple[str, ...] = DEFAULT_FRONTIER_AXES
    predictive_axis_rule: str = "inner_primary_score_allowed_only_when_fold_local"
    forbidden_axes: tuple[str, ...] = DEFAULT_FORBIDDEN_FRONTIER_AXES
