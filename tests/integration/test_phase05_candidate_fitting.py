from __future__ import annotations

from pathlib import Path

from euclid.contracts.refs import TypedRef
from euclid.modules.candidate_fitting import fit_candidate_development_windows
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.search.backends import run_descriptive_search_backends

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_candidate_fitting_handoff_tracks_fold_local_windows_for_frozen_shortlist(
) -> None:
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    canonicalization_policy = build_canonicalization_policy()
    proposal_ids = (
        "analytic_intercept",
        "analytic_lag1_affine",
        "recursive_level_smoother",
        "recursive_running_mean",
        "spectral_harmonic_1",
        "spectral_harmonic_2",
    )
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
        candidate_family_ids=proposal_ids,
        search_class="exact_finite_enumeration",
        proposal_limit=len(proposal_ids),
        seasonal_period=4,
    )

    search_result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
    )
    fit_results = fit_candidate_development_windows(
        candidates=search_result.frontier.frozen_shortlist_cir_candidates,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        search_plan=search_plan,
    )

    assert [
        result.fit_window_id for result in fit_results
    ] == [
        segment.segment_id for segment in evaluation_plan.development_segments
    ]
    assert all(result.stage_id == "inner_search" for result in fit_results)
    assert [result.training_row_count for result in fit_results] == [
        segment.train_row_count for segment in evaluation_plan.development_segments
    ]
    assert all(
        result.candidate_id == "analytic_lag1_affine"
        for result in fit_results
    )


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="phase05-fit-series",
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
