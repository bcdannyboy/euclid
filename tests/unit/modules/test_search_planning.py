from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    SearchCandidateRecord,
    build_canonicalization_policy,
    build_freeze_event,
    build_frontier,
    build_frozen_shortlist,
    build_rejected_diagnostics,
    build_search_ledger,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="demo-series",
        cutoff_available_at="2026-01-06T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=(
            SnapshotRow(
                event_time="2026-01-01T00:00:00Z",
                available_at="2026-01-01T00:00:00Z",
                observed_value=10.0,
                revision_id=0,
                payload_hash="sha256:a",
            ),
            SnapshotRow(
                event_time="2026-01-02T00:00:00Z",
                available_at="2026-01-02T00:00:00Z",
                observed_value=12.0,
                revision_id=0,
                payload_hash="sha256:b",
            ),
            SnapshotRow(
                event_time="2026-01-03T00:00:00Z",
                available_at="2026-01-03T00:00:00Z",
                observed_value=13.0,
                revision_id=0,
                payload_hash="sha256:c",
            ),
            SnapshotRow(
                event_time="2026-01-04T00:00:00Z",
                available_at="2026-01-04T00:00:00Z",
                observed_value=15.0,
                revision_id=0,
                payload_hash="sha256:d",
            ),
            SnapshotRow(
                event_time="2026-01-05T00:00:00Z",
                available_at="2026-01-05T00:00:00Z",
                observed_value=16.0,
                revision_id=0,
                payload_hash="sha256:e",
            ),
            SnapshotRow(
                event_time="2026-01-06T00:00:00Z",
                available_at="2026-01-06T00:00:00Z",
                observed_value=18.0,
                revision_id=0,
                payload_hash="sha256:f",
            ),
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    return materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    ), audit


def _search_plan(
    *,
    frontier_axes: tuple[str, ...] | None = None,
    predictive_mode: str = "predictive_requested",
):
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    canonicalization_policy = build_canonicalization_policy()
    return build_search_plan(
        evaluation_plan=evaluation_plan,
        canonicalization_policy_ref=TypedRef(
            "canonicalization_policy_manifest@1.0.0",
            canonicalization_policy.canonicalization_policy_id,
        ),
        codelength_policy_ref=TypedRef(
            "codelength_policy_manifest@1.1.0",
            "mdl_policy_default",
        ),
        reference_description_policy_ref=TypedRef(
            "reference_description_policy_manifest@1.1.0",
            "reference_description_default",
        ),
        observation_model_ref=TypedRef(
            "observation_model_manifest@1.1.0",
            "observation_model_default",
        ),
        candidate_family_ids=("constant", "drift", "linear_trend", "seasonal_naive"),
        predictive_mode=predictive_mode,
        frontier_axes=frontier_axes
        or (
            "structure_code_bits",
            "description_gain_bits",
            "inner_primary_score",
        ),
    )


def test_build_search_plan_freezes_budget_seed_and_frontier_policies() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    canonicalization_policy = build_canonicalization_policy()
    canonicalization_manifest = canonicalization_policy.to_manifest(catalog)
    search_plan = _search_plan()
    manifest = search_plan.to_manifest(catalog)

    assert canonicalization_manifest.body["canonicalization_policy_id"] == (
        canonicalization_policy.canonicalization_policy_id
    )
    assert canonicalization_manifest.body["canonical_form_id"] == (
        "search_normal_form_v1"
    )
    assert manifest.body["search_class"] == "bounded_heuristic"
    assert manifest.body["primitive_families"] == [
        "analytic",
        "recursive",
        "spectral",
        "algorithmic",
    ]
    assert manifest.body["composition_operators"] == [
        "piecewise",
        "additive_residual",
        "regime_conditioned",
    ]
    assert manifest.body["search_budget"] == {
        "proposal_limit": 4,
        "frontier_width": 4,
        "shortlist_limit": 1,
        "wall_clock_budget_seconds": 1,
        "budget_accounting_rule": "proposal_count_then_candidate_id_tie_break",
    }
    assert manifest.body["parallel_budget"] == {
        "max_worker_count": 1,
        "candidate_batch_size": 1,
        "aggregation_rule": "deterministic_candidate_id_order",
    }
    assert manifest.body["seed_policy"] == {
        "root_seed": "0",
        "seed_derivation_rule": "deterministic_scope_hash",
        "seed_scopes": ["search", "candidate_generation", "tie_break"],
    }
    assert manifest.body["frontier_policy"] == {
        "frontier_id": "retained_scope_search_frontier_v1",
        "axes": [
            "structure_code_bits",
            "description_gain_bits",
            "inner_primary_score",
        ],
        "predictive_axis_rule": "inner_primary_score_allowed_only_when_fold_local",
        "forbidden_axes": [
            "holdout_results",
            "outer_fold_results",
            "null_results",
            "robustness_results",
        ],
    }
    assert manifest.body["search_time_predictive_policy"] == "fold_local_only"


def test_build_search_plan_rejects_forbidden_frontier_axes() -> None:
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )

    with pytest.raises(ContractValidationError) as exc_info:
        build_search_plan(
            evaluation_plan=evaluation_plan,
            canonicalization_policy_ref=TypedRef(
                "canonicalization_policy_manifest@1.0.0",
                "canonicalization_default",
            ),
            codelength_policy_ref=TypedRef(
                "codelength_policy_manifest@1.1.0",
                "mdl_policy_default",
            ),
            reference_description_policy_ref=TypedRef(
                "reference_description_policy_manifest@1.1.0",
                "reference_description_default",
            ),
            observation_model_ref=TypedRef(
                "observation_model_manifest@1.1.0",
                "observation_model_default",
            ),
            candidate_family_ids=("constant",),
            frontier_axes=("description_gain_bits", "holdout_results"),
        )

    assert exc_info.value.code == "forbidden_frontier_axis"


def test_search_stage_artifacts_keep_confirmatory_metrics_out_of_search_surfaces() -> (
    None
):
    catalog = load_contract_catalog(PROJECT_ROOT)
    search_plan = _search_plan()
    candidate_records = (
        SearchCandidateRecord(
            candidate_id="candidate_a",
            family_id="constant",
            structure_code_bits=0.0,
            total_code_bits=10.0,
            description_gain_bits=3.0,
            inner_primary_score=0.35,
            admissible=True,
            canonical_byte_length=9,
            ranked=True,
            law_eligible=True,
        ),
        SearchCandidateRecord(
            candidate_id="candidate_b",
            family_id="drift",
            structure_code_bits=1.0,
            total_code_bits=12.0,
            description_gain_bits=2.0,
            inner_primary_score=0.20,
            admissible=True,
            canonical_byte_length=11,
            ranked=True,
            law_eligible=True,
        ),
        SearchCandidateRecord(
            candidate_id="candidate_c",
            family_id="seasonal_naive",
            structure_code_bits=1.0,
            total_code_bits=13.0,
            description_gain_bits=-0.5,
            inner_primary_score=0.15,
            canonical_byte_length=8,
            admissible=False,
            ranked=False,
            law_eligible=False,
            rejection_reason_codes=(
                "codelength_comparability_failed",
                "description_gain_non_positive",
            ),
            law_rejection_reason_codes=(
                "codelength_comparability_failed",
                "description_gain_non_positive",
            ),
        ),
    )

    search_ledger = build_search_ledger(
        search_plan=search_plan,
        candidate_records=candidate_records,
        selected_candidate_id="candidate_a",
        accepted_candidate_id="candidate_a",
    )
    frontier = build_frontier(
        search_plan=search_plan,
        candidate_records=candidate_records,
    )
    rejected = build_rejected_diagnostics(candidate_records=candidate_records)

    search_ledger_manifest = search_ledger.to_manifest(catalog)
    frontier_manifest = frontier.to_manifest(catalog)
    rejected_manifest = rejected.to_manifest(catalog)

    assert search_ledger_manifest.body["selection_rule"] == (
        "min_total_code_bits_then_max_description_gain_then_"
        "min_structure_code_bits_then_min_canonical_byte_length_then_candidate_id"
    )
    assert search_ledger_manifest.body["budget_accounting"] == {
        "proposal_limit": 4,
        "attempted_candidate_count": 3,
        "ranked_candidate_count": 2,
        "admissible_candidate_count": 2,
        "law_eligible_candidate_count": 2,
        "accepted_candidate_count": 1,
        "rejected_candidate_count": 1,
        "remaining_budget_count": 1,
        "best_overall_candidate_id": "candidate_a",
        "accepted_candidate_id": "candidate_a",
        "selected_candidate_scope": "descriptive_scope",
        "accepted_candidate_scope": "law_eligible_scope",
        "descriptive_scope_candidate_ids": [
            "candidate_a",
            "candidate_b",
        ],
        "law_eligible_scope_candidate_ids": [
            "candidate_a",
            "candidate_b",
        ],
        "accounting_status": "within_budget",
        "stop_reason": "candidate_list_exhausted",
    }
    assert all(
        "confirmatory_primary_score" not in record
        and "baseline_primary_score" not in record
        for record in search_ledger_manifest.body["candidates"]
    )
    assert frontier_manifest.body["frontier_candidate_ids"] == [
        "candidate_a",
        "candidate_b",
    ]
    assert frontier_manifest.body["frontier_axes"] == [
        "structure_code_bits",
        "description_gain_bits",
        "inner_primary_score",
    ]
    assert all(
        "confirmatory_primary_score" not in record
        for record in frontier_manifest.body["frontier_records"]
    )
    assert rejected_manifest.body["rejected_records"] == [
        {
            "candidate_id": "candidate_c",
            "family_id": "seasonal_naive",
            "rejection_reason_codes": [
                "codelength_comparability_failed",
                "description_gain_non_positive",
            ],
            "law_rejection_reason_codes": [
                "codelength_comparability_failed",
                "description_gain_non_positive",
            ],
            "law_eligible": False,
        }
    ]


def test_search_ledger_requires_explicit_scope_flags() -> None:
    search_plan = _search_plan()
    candidate_records = (
        SearchCandidateRecord(
            candidate_id="candidate_admissible_only",
            family_id="constant",
            structure_code_bits=0.0,
            total_code_bits=10.0,
            description_gain_bits=3.0,
            inner_primary_score=0.35,
            canonical_byte_length=9,
            admissible=True,
        ),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        build_search_ledger(
            search_plan=search_plan,
            candidate_records=candidate_records,
            selected_candidate_id="candidate_admissible_only",
            accepted_candidate_id="candidate_admissible_only",
        )

    assert exc_info.value.code == "missing_descriptive_scope_candidate"


def test_search_stage_artifacts_distinguish_ranked_scope_from_law_eligible_scope() -> (
    None
):
    catalog = load_contract_catalog(PROJECT_ROOT)
    search_plan = _search_plan()
    candidate_records = (
        SearchCandidateRecord(
            candidate_id="candidate_ranked_only",
            family_id="constant",
            structure_code_bits=0.0,
            total_code_bits=10.0,
            description_gain_bits=3.5,
            inner_primary_score=0.45,
            canonical_byte_length=9,
            admissible=True,
            ranked=True,
            law_eligible=False,
            law_rejection_reason_codes=("publishability_gate_failed",),
        ),
        SearchCandidateRecord(
            candidate_id="candidate_accepted",
            family_id="drift",
            structure_code_bits=1.0,
            total_code_bits=11.0,
            description_gain_bits=20.0,
            inner_primary_score=0.20,
            canonical_byte_length=11,
            admissible=True,
            ranked=True,
            law_eligible=True,
        ),
        SearchCandidateRecord(
            candidate_id="candidate_diagnostic_only",
            family_id="seasonal_naive",
            structure_code_bits=1.0,
            total_code_bits=13.0,
            description_gain_bits=-0.5,
            inner_primary_score=0.15,
            canonical_byte_length=8,
            admissible=False,
            ranked=False,
            rejection_reason_codes=("codelength_comparability_failed",),
            law_eligible=False,
            law_rejection_reason_codes=("codelength_comparability_failed",),
        ),
    )

    search_ledger = build_search_ledger(
        search_plan=search_plan,
        candidate_records=candidate_records,
        selected_candidate_id="candidate_ranked_only",
        accepted_candidate_id="candidate_accepted",
    )
    frontier = build_frontier(
        search_plan=search_plan,
        candidate_records=candidate_records,
    )
    rejected = build_rejected_diagnostics(candidate_records=candidate_records)

    search_ledger_manifest = search_ledger.to_manifest(catalog)
    frontier_manifest = frontier.to_manifest(catalog)
    rejected_manifest = rejected.to_manifest(catalog)

    assert search_ledger_manifest.body["budget_accounting"] == {
        "proposal_limit": 4,
        "attempted_candidate_count": 3,
        "ranked_candidate_count": 2,
        "admissible_candidate_count": 2,
        "law_eligible_candidate_count": 1,
        "accepted_candidate_count": 1,
        "rejected_candidate_count": 2,
        "remaining_budget_count": 1,
        "best_overall_candidate_id": "candidate_ranked_only",
        "accepted_candidate_id": "candidate_accepted",
        "selected_candidate_scope": "descriptive_scope",
        "accepted_candidate_scope": "law_eligible_scope",
        "descriptive_scope_candidate_ids": [
            "candidate_ranked_only",
            "candidate_accepted",
        ],
        "law_eligible_scope_candidate_ids": ["candidate_accepted"],
        "accounting_status": "within_budget",
        "stop_reason": "candidate_list_exhausted",
    }
    assert search_ledger_manifest.body["candidates"] == [
        {
            "candidate_id": "candidate_ranked_only",
            "family_id": "constant",
            "structure_code_bits": 0.0,
            "total_code_bits": 10.0,
            "description_gain_bits": 3.5,
            "inner_primary_score": 0.45,
            "canonical_byte_length": 9,
            "admissible": True,
            "ranked": True,
            "descriptive_scope_rank": 1,
            "law_eligible": False,
            "law_eligible_scope_rank": None,
            "law_rejection_reason_codes": ["publishability_gate_failed"],
        },
        {
            "candidate_id": "candidate_accepted",
            "family_id": "drift",
            "structure_code_bits": 1.0,
            "total_code_bits": 11.0,
            "description_gain_bits": 20.0,
            "inner_primary_score": 0.2,
            "canonical_byte_length": 11,
            "admissible": True,
            "ranked": True,
            "descriptive_scope_rank": 2,
            "law_eligible": True,
            "law_eligible_scope_rank": 1,
            "law_rejection_reason_codes": [],
        },
        {
            "candidate_id": "candidate_diagnostic_only",
            "family_id": "seasonal_naive",
            "structure_code_bits": 1.0,
            "total_code_bits": 13.0,
            "description_gain_bits": -0.5,
            "inner_primary_score": 0.15,
            "canonical_byte_length": 8,
            "admissible": False,
            "ranked": False,
            "descriptive_scope_rank": None,
            "law_eligible": False,
            "law_eligible_scope_rank": None,
            "law_rejection_reason_codes": ["codelength_comparability_failed"],
        },
    ]
    assert frontier_manifest.body["frontier_candidate_ids"] == [
        "candidate_ranked_only",
        "candidate_accepted",
    ]
    assert frontier_manifest.body["frontier_records"] == [
        {
            "candidate_id": "candidate_ranked_only",
            "family_id": "constant",
            "structure_code_bits": 0.0,
            "total_code_bits": 10.0,
            "description_gain_bits": 3.5,
            "inner_primary_score": 0.45,
            "ranked": True,
            "law_eligible": False,
            "law_rejection_reason_codes": ["publishability_gate_failed"],
        },
        {
            "candidate_id": "candidate_accepted",
            "family_id": "drift",
            "structure_code_bits": 1.0,
            "total_code_bits": 11.0,
            "description_gain_bits": 20.0,
            "inner_primary_score": 0.2,
            "ranked": True,
            "law_eligible": True,
            "law_rejection_reason_codes": [],
        },
    ]
    assert rejected_manifest.body["rejected_records"] == [
        {
            "candidate_id": "candidate_ranked_only",
            "family_id": "constant",
            "rejection_reason_codes": [],
            "law_rejection_reason_codes": ["publishability_gate_failed"],
            "law_eligible": False,
        },
        {
            "candidate_id": "candidate_diagnostic_only",
            "family_id": "seasonal_naive",
            "rejection_reason_codes": ["codelength_comparability_failed"],
            "law_rejection_reason_codes": ["codelength_comparability_failed"],
            "law_eligible": False,
        },
    ]


def test_search_ledger_counts_only_ranked_law_eligible_scope_records() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    search_plan = _search_plan()
    candidate_records = (
        SearchCandidateRecord(
            candidate_id="candidate_ranked_only",
            family_id="constant",
            structure_code_bits=0.0,
            total_code_bits=10.0,
            description_gain_bits=3.5,
            inner_primary_score=0.45,
            canonical_byte_length=9,
            admissible=True,
            ranked=True,
            law_eligible=False,
            law_rejection_reason_codes=("publishability_gate_failed",),
        ),
        SearchCandidateRecord(
            candidate_id="candidate_accepted",
            family_id="drift",
            structure_code_bits=1.0,
            total_code_bits=11.0,
            description_gain_bits=20.0,
            inner_primary_score=0.20,
            canonical_byte_length=11,
            admissible=True,
            ranked=True,
            law_eligible=True,
        ),
        SearchCandidateRecord(
            candidate_id="candidate_not_ranked_but_flagged",
            family_id="seasonal_naive",
            structure_code_bits=1.0,
            total_code_bits=12.0,
            description_gain_bits=0.5,
            inner_primary_score=0.10,
            canonical_byte_length=10,
            admissible=True,
            ranked=False,
            law_eligible=True,
        ),
    )

    search_ledger = build_search_ledger(
        search_plan=search_plan,
        candidate_records=candidate_records,
        selected_candidate_id="candidate_ranked_only",
        accepted_candidate_id="candidate_accepted",
    )

    manifest = search_ledger.to_manifest(catalog)

    assert manifest.body["budget_accounting"]["law_eligible_candidate_count"] == 1
    assert manifest.body["budget_accounting"]["law_eligible_scope_candidate_ids"] == [
        "candidate_accepted"
    ]


def test_search_stage_artifacts_keep_synthetic_exact_closure_and_posthoc_candidates_diagnostic_only(
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    search_plan = _search_plan()
    candidate_records = (
        SearchCandidateRecord(
            candidate_id="candidate_best_descriptive",
            family_id="constant",
            structure_code_bits=0.0,
            total_code_bits=10.0,
            description_gain_bits=4.0,
            inner_primary_score=0.4,
            canonical_byte_length=9,
            admissible=True,
            ranked=True,
            law_eligible=False,
            law_rejection_reason_codes=("publishability_gate_failed",),
        ),
        SearchCandidateRecord(
            candidate_id="candidate_exact_closure_sample",
            family_id="drift",
            structure_code_bits=0.5,
            total_code_bits=9.0,
            description_gain_bits=30.0,
            inner_primary_score=0.1,
            canonical_byte_length=8,
            admissible=False,
            ranked=False,
            rejection_reason_codes=("requires_exact_sample_closure",),
            law_eligible=False,
            law_rejection_reason_codes=("requires_exact_sample_closure",),
        ),
        SearchCandidateRecord(
            candidate_id="candidate_symbolic_synthesis_posthoc",
            family_id="linear_trend",
            structure_code_bits=0.25,
            total_code_bits=8.5,
            description_gain_bits=25.0,
            inner_primary_score=0.15,
            canonical_byte_length=7,
            admissible=False,
            ranked=False,
            rejection_reason_codes=("requires_posthoc_symbolic_synthesis",),
            law_eligible=False,
            law_rejection_reason_codes=("requires_posthoc_symbolic_synthesis",),
        ),
    )

    search_ledger = build_search_ledger(
        search_plan=search_plan,
        candidate_records=candidate_records,
        selected_candidate_id="candidate_best_descriptive",
        accepted_candidate_id=None,
    )
    frontier = build_frontier(
        search_plan=search_plan,
        candidate_records=candidate_records,
    )
    rejected = build_rejected_diagnostics(candidate_records=candidate_records)

    search_ledger_manifest = search_ledger.to_manifest(catalog)
    frontier_manifest = frontier.to_manifest(catalog)
    rejected_manifest = rejected.to_manifest(catalog)

    assert search_ledger_manifest.body["budget_accounting"] == {
        "proposal_limit": 4,
        "attempted_candidate_count": 3,
        "ranked_candidate_count": 1,
        "admissible_candidate_count": 1,
        "law_eligible_candidate_count": 0,
        "accepted_candidate_count": 0,
        "rejected_candidate_count": 3,
        "remaining_budget_count": 1,
        "best_overall_candidate_id": "candidate_best_descriptive",
        "accepted_candidate_id": None,
        "selected_candidate_scope": "descriptive_scope",
        "accepted_candidate_scope": "law_eligible_scope",
        "descriptive_scope_candidate_ids": ["candidate_best_descriptive"],
        "law_eligible_scope_candidate_ids": [],
        "accounting_status": "within_budget",
        "stop_reason": "candidate_list_exhausted",
    }
    candidates = {
        record["candidate_id"]: record
        for record in search_ledger_manifest.body["candidates"]
    }
    assert candidates["candidate_best_descriptive"]["descriptive_scope_rank"] == 1
    assert candidates["candidate_exact_closure_sample"]["descriptive_scope_rank"] is None
    assert candidates["candidate_exact_closure_sample"][
        "law_rejection_reason_codes"
    ] == ["requires_exact_sample_closure"]
    assert candidates["candidate_symbolic_synthesis_posthoc"][
        "descriptive_scope_rank"
    ] is None
    assert candidates["candidate_symbolic_synthesis_posthoc"][
        "law_rejection_reason_codes"
    ] == ["requires_posthoc_symbolic_synthesis"]
    assert frontier_manifest.body["frontier_candidate_ids"] == [
        "candidate_best_descriptive"
    ]
    rejected_records = {
        record["candidate_id"]: record
        for record in rejected_manifest.body["rejected_records"]
    }
    assert rejected_records["candidate_exact_closure_sample"] == {
        "candidate_id": "candidate_exact_closure_sample",
        "family_id": "drift",
        "rejection_reason_codes": ["requires_exact_sample_closure"],
        "law_rejection_reason_codes": ["requires_exact_sample_closure"],
        "law_eligible": False,
    }
    assert rejected_records["candidate_symbolic_synthesis_posthoc"] == {
        "candidate_id": "candidate_symbolic_synthesis_posthoc",
        "family_id": "linear_trend",
        "rejection_reason_codes": ["requires_posthoc_symbolic_synthesis"],
        "law_rejection_reason_codes": ["requires_posthoc_symbolic_synthesis"],
        "law_eligible": False,
    }


def test_build_search_ledger_rejects_mismatched_ranked_and_accepted_ids() -> None:
    search_plan = _search_plan()
    candidate_records = (
        SearchCandidateRecord(
            candidate_id="candidate_ranked_only",
            family_id="constant",
            structure_code_bits=0.0,
            total_code_bits=10.0,
            description_gain_bits=3.5,
            inner_primary_score=0.45,
            canonical_byte_length=9,
            admissible=True,
            ranked=True,
            law_eligible=False,
            law_rejection_reason_codes=("publishability_gate_failed",),
        ),
        SearchCandidateRecord(
            candidate_id="candidate_accepted",
            family_id="drift",
            structure_code_bits=1.0,
            total_code_bits=11.0,
            description_gain_bits=20.0,
            inner_primary_score=0.20,
            canonical_byte_length=11,
            admissible=True,
            ranked=True,
            law_eligible=True,
        ),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        build_search_ledger(
            search_plan=search_plan,
            candidate_records=candidate_records,
            selected_candidate_id="candidate_accepted",
            accepted_candidate_id="candidate_ranked_only",
        )

    assert exc_info.value.code == "ledger_selection_scope_mismatch"


def test_build_search_ledger_rejects_ranked_candidate_missing_total_code_bits(
) -> None:
    search_plan = _search_plan()
    candidate_records = (
        SearchCandidateRecord(
            candidate_id="candidate_ranked_only",
            family_id="constant",
            structure_code_bits=0.0,
            description_gain_bits=3.5,
            inner_primary_score=0.45,
            canonical_byte_length=9,
            admissible=True,
            ranked=True,
            law_eligible=True,
        ),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        build_search_ledger(
            search_plan=search_plan,
            candidate_records=candidate_records,
            selected_candidate_id="candidate_ranked_only",
            accepted_candidate_id="candidate_ranked_only",
        )

    assert exc_info.value.code == "missing_total_code_bits"


def test_build_search_ledger_rejects_ranked_candidate_missing_canonical_byte_length(
) -> None:
    search_plan = _search_plan()
    candidate_records = (
        SearchCandidateRecord(
            candidate_id="candidate_ranked_only",
            family_id="constant",
            structure_code_bits=0.0,
            total_code_bits=10.0,
            description_gain_bits=3.5,
            inner_primary_score=0.45,
            admissible=True,
            ranked=True,
            law_eligible=True,
        ),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        build_search_ledger(
            search_plan=search_plan,
            candidate_records=candidate_records,
            selected_candidate_id="candidate_ranked_only",
            accepted_candidate_id="candidate_ranked_only",
        )

    assert exc_info.value.code == "missing_canonical_byte_length"


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("total_code_bits", float("inf")),
        ("description_gain_bits", float("nan")),
        ("structure_code_bits", float("-inf")),
    ),
)
def test_build_search_ledger_rejects_non_finite_ranked_metrics(
    field_name: str,
    value: float,
) -> None:
    search_plan = _search_plan()
    candidate_payload = {
        "candidate_id": "candidate_ranked_only",
        "family_id": "constant",
        "structure_code_bits": 0.0,
        "total_code_bits": 10.0,
        "description_gain_bits": 3.5,
        "inner_primary_score": 0.45,
        "canonical_byte_length": 9,
        "admissible": True,
        "ranked": True,
        "law_eligible": True,
    }
    candidate_payload[field_name] = value
    candidate_records = (SearchCandidateRecord(**candidate_payload),)

    with pytest.raises(ContractValidationError) as exc_info:
        build_search_ledger(
            search_plan=search_plan,
            candidate_records=candidate_records,
            selected_candidate_id="candidate_ranked_only",
            accepted_candidate_id="candidate_ranked_only",
        )

    assert exc_info.value.code == "non_finite_ranked_candidate_metric"
    assert exc_info.value.details == {
        "candidate_id": "candidate_ranked_only",
        "metric_name": field_name,
        "metric_value": value,
    }


def test_search_candidate_record_requires_explicit_total_code_bits() -> None:
    class _RuntimeCandidate:
        def __init__(self) -> None:
            self.candidate_id = "candidate_ranked_only"
            self.description_components = {"L_total_bits": 10.0}

    def _build_record() -> SearchCandidateRecord:
        candidate = _RuntimeCandidate()
        return SearchCandidateRecord(
            candidate_id="candidate_ranked_only",
            family_id="constant",
            structure_code_bits=0.0,
            description_gain_bits=3.5,
            inner_primary_score=0.45,
            canonical_byte_length=9,
            admissible=True,
            ranked=True,
            law_eligible=True,
        )

    record = _build_record()

    assert record.total_code_bits is None


def test_build_frontier_honors_declared_non_default_axes() -> None:
    search_plan = _search_plan(
        frontier_axes=("description_gain_bits",),
        predictive_mode="descriptive_only",
    )
    candidate_records = (
        SearchCandidateRecord(
            candidate_id="candidate_gain_best",
            family_id="constant",
            structure_code_bits=3.0,
            total_code_bits=14.0,
            description_gain_bits=5.0,
            inner_primary_score=0.8,
            admissible=True,
            ranked=True,
        ),
        SearchCandidateRecord(
            candidate_id="candidate_structure_best",
            family_id="drift",
            structure_code_bits=0.0,
            total_code_bits=16.0,
            description_gain_bits=4.0,
            inner_primary_score=0.1,
            admissible=True,
            ranked=True,
        ),
        SearchCandidateRecord(
            candidate_id="candidate_inner_best",
            family_id="linear_trend",
            structure_code_bits=1.0,
            total_code_bits=15.0,
            description_gain_bits=4.5,
            inner_primary_score=0.0,
            admissible=True,
            ranked=True,
        ),
    )

    frontier = build_frontier(
        search_plan=search_plan,
        candidate_records=candidate_records,
    )

    assert frontier.frontier_axes == ("description_gain_bits",)
    assert frontier.frontier_candidate_ids == ("candidate_gain_best",)


def test_freeze_manifests_bind_pre_holdout_shortlist_rules() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    shortlist = build_frozen_shortlist(
        search_plan_ref=TypedRef("search_plan_manifest@1.0.0", "search"),
        candidate_ref=TypedRef("reducer_artifact_manifest@1.0.0", "candidate"),
    )
    freeze_event = build_freeze_event(
        frozen_candidate_ref=TypedRef(
            "reducer_artifact_manifest@1.0.0",
            "candidate",
        ),
        frozen_shortlist_ref=TypedRef(
            "frozen_shortlist_manifest@1.0.0",
            "shortlist",
        ),
        confirmatory_baseline_id="constant_baseline",
    )

    shortlist_manifest = shortlist.to_manifest(catalog)
    freeze_manifest = freeze_event.to_manifest(catalog)

    assert shortlist_manifest.body["selection_rule"] == (
        "single_lowest_total_bits_admissible_candidate"
    )
    assert shortlist_manifest.body["tie_break_rule"] == "lexicographic_candidate_id"
    assert shortlist_manifest.body["shortlist_cardinality"] == 1
    assert freeze_manifest.body["freeze_stage"] == "global_pair_freeze_pre_holdout"
    assert freeze_manifest.body["baseline_selection_rule"] == "ex_ante_fixed"
    assert freeze_manifest.body["holdout_materialized_before_freeze"] is False
    assert freeze_manifest.body["post_freeze_candidate_mutation_count"] == 0
