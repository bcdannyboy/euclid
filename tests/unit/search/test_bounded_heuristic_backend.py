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


def test_backend_emits_bound_and_stop_rule_disclosures() -> None:
    feature_view, audit = _feature_view()
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
        candidate_family_ids=("analytic_intercept", "analytic_lag1_affine"),
        search_class="bounded_heuristic",
        proposal_limit=1,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )

    assert result.coverage.disclosures == {
        "proposer_mechanism": "declared_adapter_default_or_user_supplied_proposals",
        "pruning_rules": (
            "proposal_limit_prefix_then_canonical_duplicate_screen"
            "_then_descriptive_admissibility"
        ),
        "stop_rule": "proposal_limit=1",
    }


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="bounded-heuristic-series",
        cutoff_available_at="2026-01-06T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-0{index + 1}T00:00:00Z",
                available_at=f"2026-01-0{index + 1}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:{index}",
            )
            for index, value in enumerate((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    return materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    ), audit

