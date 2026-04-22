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
from euclid.search.engines.native import NativeFragmentEngine
from euclid.search.orchestration import run_search_engines


def test_native_engine_emits_non_claim_records_for_bounded_mode() -> None:
    feature_view, search_plan = _feature_view_and_plan(search_class="bounded_heuristic")
    context = NativeFragmentEngine.context_from_plan(
        search_plan=search_plan,
        feature_view=feature_view,
        timeout_seconds=1.0,
    )

    result = NativeFragmentEngine(search_class="bounded_heuristic").run(context)

    assert result.status == "completed"
    assert result.candidates
    assert all(
        candidate.claim_boundary["claim_publication_allowed"] is False
        for candidate in result.candidates
    )
    assert result.trace["legacy_adapter_mode"] == "non_claim_structure_proposal"
    assert result.omission_disclosure["canonical_program_count"] >= len(
        result.candidates
    )


def test_native_engine_can_be_orchestrated_and_lowered_to_cir() -> None:
    feature_view, search_plan = _feature_view_and_plan(search_class="bounded_heuristic")
    context = NativeFragmentEngine.context_from_plan(
        search_plan=search_plan,
        feature_view=feature_view,
        timeout_seconds=1.0,
    )

    result = run_search_engines(
        context=context,
        engines=(NativeFragmentEngine(search_class="bounded_heuristic"),),
    )

    assert result.accepted_candidates
    assert result.engine_runs["native-bounded-fragment-v1"].status == "completed"
    assert result.claim_boundary["claim_publication_allowed"] is False


def test_native_engine_reports_empty_declared_space_without_claims() -> None:
    feature_view, search_plan = _feature_view_and_plan(
        search_class="bounded_heuristic",
        candidate_ids=("not_a_native_candidate",),
    )
    context = NativeFragmentEngine.context_from_plan(
        search_plan=search_plan,
        feature_view=feature_view,
        timeout_seconds=1.0,
    )

    result = NativeFragmentEngine(search_class="bounded_heuristic").run(context)

    assert result.status == "completed"
    assert result.candidates == ()
    assert result.omission_disclosure["omitted_by_candidate_filter"] > 0
    assert result.claim_boundary["claim_publication_allowed"] is False


def _feature_view_and_plan(
    *,
    search_class: str,
    candidate_ids: tuple[str, ...] = (
        "analytic_intercept",
        "analytic_lag1_affine",
        "recursive_level_smoother",
    ),
):
    snapshot = FrozenDatasetSnapshot(
        series_id="native-engine-series",
        cutoff_available_at="2026-01-06T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-{index + 1:02d}T00:00:00Z",
                available_at=f"2026-01-{index + 1:02d}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:native-{index}",
            )
            for index, value in enumerate((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
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
        candidate_family_ids=candidate_ids,
        search_class=search_class,
        proposal_limit=len(candidate_ids),
        random_seed="19",
    )
    return feature_view, search_plan
