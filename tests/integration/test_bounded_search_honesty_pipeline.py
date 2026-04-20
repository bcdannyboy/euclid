from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.search.backends import (
    _validate_search_honesty_evidence,
    run_descriptive_search_backends,
)


def test_replay_preserves_bounded_search_disclosures() -> None:
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

    first = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )
    second = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )

    assert first.coverage.disclosures == second.coverage.disclosures
    assert (
        first.accepted_candidates[0].evidence_layer.transient_diagnostics[
            "search_evidence"
        ]
        == second.accepted_candidates[0].evidence_layer.transient_diagnostics[
            "search_evidence"
        ]
    )


def test_bounded_search_overclaim_is_rejected() -> None:
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

    with pytest.raises(ContractValidationError) as exc_info:
        _validate_search_honesty_evidence(
            search_plan=search_plan,
            coverage_disclosures={"stop_rule": "proposal_limit=1"},
            accepted_candidates=result.accepted_candidates,
        )

    assert exc_info.value.code == "search_honesty_evidence_missing"


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="bounded-search-pipeline-series",
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

