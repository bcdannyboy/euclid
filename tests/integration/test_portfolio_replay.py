from __future__ import annotations

from euclid.adapters.portfolio import normalize_cir_finalist
from euclid.contracts.refs import TypedRef
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.replay import (
    build_portfolio_replay_contract,
    verify_portfolio_replay_contract,
)
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.search.portfolio import run_descriptive_search_portfolio


def test_portfolio_replay_contract_round_trips_selected_winner_and_finalists() -> None:
    feature_view, search_plan = _context()
    portfolio_result = run_descriptive_search_portfolio(
        search_plan=search_plan,
        feature_view=feature_view,
    )

    compared_finalists = tuple(
        normalize_cir_finalist(
            candidate,
            total_code_bits=float(entry.total_code_bits),
            description_gain_bits=float(entry.description_gain_bits),
            structure_code_bits=float(entry.structure_code_bits),
            provenance_id=ledger.family_id,
            coverage_statement=ledger.coverage_statement,
            exactness_ceiling=ledger.exactness_ceiling,
            scope_declaration=ledger.scope_declaration,
        )
        for ledger in portfolio_result.backend_ledgers
        for entry, candidate in (
            [
                (
                    next(
                        item
                        for item in ledger.candidate_ledger
                        if item.candidate_id == ledger.finalist_candidate_id
                    ),
                    next(
                        candidate
                        for candidate in (
                            portfolio_result.search_result.accepted_candidates
                        )
                        if candidate.canonical_hash() == ledger.finalist_candidate_hash
                    ),
                )
            ]
            if ledger.finalist_candidate_id is not None
            else []
        )
    )

    replay_contract = build_portfolio_replay_contract(
        selection_record_id=portfolio_result.selection_record.selection_record_id,
        selection_scope=portfolio_result.selection_record.selection_scope,
        selection_rule=portfolio_result.selection_record.selection_rule,
        selected_provenance_id=portfolio_result.selection_record.selected_backend_family,
        selected_candidate_id=portfolio_result.selection_record.selected_candidate_id,
        selected_candidate_hash=portfolio_result.selection_record.selected_candidate_hash,
        compared_finalists=compared_finalists,
        decision_trace=portfolio_result.selection_record.decision_trace,
    )
    verification = verify_portfolio_replay_contract(
        replay_contract,
        selected_candidate_id=portfolio_result.selection_record.selected_candidate_id,
        selected_candidate_hash=portfolio_result.selection_record.selected_candidate_hash,
        compared_finalists=compared_finalists,
        decision_trace=portfolio_result.selection_record.decision_trace,
    )

    assert replay_contract["selection_scope"] == "shared_planning_cir_only"
    assert replay_contract["selection_rule"].startswith("min_total_code_bits")
    assert len(replay_contract["compared_finalists"]) == 3
    assert verification["replay_verification_status"] == "verified"
    assert verification["failure_reason_codes"] == []


def _context():
    snapshot = FrozenDatasetSnapshot(
        series_id="portfolio-replay-series",
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
    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    canonicalization_policy = build_canonicalization_policy()
    search_plan = build_search_plan(
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
        candidate_family_ids=(
            "analytic_intercept",
            "analytic_lag1_affine",
            "recursive_level_smoother",
            "recursive_running_mean",
            "spectral_harmonic_1",
            "spectral_harmonic_2",
            "algorithmic_last_observation",
            "algorithmic_running_half_average",
        ),
        search_class="exact_finite_enumeration",
        proposal_limit=8,
        seasonal_period=4,
    )
    return feature_view, search_plan
