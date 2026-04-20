from __future__ import annotations

from euclid.contracts.refs import TypedRef
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.search.backends import run_descriptive_search_backends


def test_exact_search_candidates_carry_enumeration_evidence_on_the_evidence_layer(
) -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=("analytic_intercept", "analytic_lag1_affine"),
        search_class="exact_finite_enumeration",
        proposal_limit=2,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )

    assert result.coverage.disclosures["enumeration_cardinality"] == 2
    assert result.coverage.disclosures["canonical_enumerator"] == (
        "declared_adapter_family_order_then_candidate_id"
    )

    for candidate in result.accepted_candidates:
        search_evidence = candidate.evidence_layer.transient_diagnostics[
            "search_evidence"
        ]
        assert search_evidence["search_class"] == "exact_finite_enumeration"
        assert search_evidence["enumeration_cardinality"] == 2
        assert search_evidence["canonical_enumerator"] == (
            "declared_adapter_family_order_then_candidate_id"
        )
        assert search_evidence["exactness_scope"] == (
            "finite_exactness_limited_to_declared_canonical_program_space"
        )


def test_declared_search_frontier_has_no_incomparable_axes() -> None:
    feature_view, audit = _feature_view()
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=("analytic_intercept", "analytic_lag1_affine"),
        search_class="exact_finite_enumeration",
        proposal_limit=2,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )

    assert result.frontier.coverage.incomparable_axes == ()
    assert result.frontier.coverage.comparable_axes == (
        "structure_code_bits",
        "description_gain_bits",
        "inner_primary_score",
    )


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="phase03-search-honesty-series",
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


def _build_search_plan(
    *,
    feature_view,
    audit,
    candidate_family_ids: tuple[str, ...],
    search_class: str,
    proposal_limit: int,
):
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
        candidate_family_ids=candidate_family_ids,
        search_class=search_class,
        proposal_limit=proposal_limit,
    )
