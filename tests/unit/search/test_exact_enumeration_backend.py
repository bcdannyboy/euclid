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
from euclid.search.backends import (
    AlgorithmicSearchBackendAdapter,
    run_descriptive_search_backends,
)


def test_backend_emits_enumeration_cardinality() -> None:
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
        candidate_family_ids=(
            "algorithmic_last_observation",
            "algorithmic_running_half_average",
        ),
        search_class="exact_finite_enumeration",
        proposal_limit=2,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        adapters=(AlgorithmicSearchBackendAdapter(),),
    )

    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.accepted_candidates
    ] == ["algorithmic_last_observation"]
    assert result.coverage.disclosures["fragment_bounds"] == "proposal_limit=2"
    assert result.coverage.disclosures["enumeration_cardinality"] == 2
    assert result.coverage.disclosures["canonical_enumerator"] == (
        "declared_adapter_family_order_then_candidate_id"
    )
    assert result.accepted_candidates[0].evidence_layer.transient_diagnostics[
        "search_evidence"
    ]["exactness_scope"] == (
        "finite_exactness_limited_to_declared_canonical_program_space"
    )


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="exact-enumeration-series",
        cutoff_available_at="2026-01-07T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-0{index + 1}T00:00:00Z",
                available_at=f"2026-01-0{index + 1}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:{index}",
            )
            for index, value in enumerate((10.0, 12.0, 13.0, 15.0, 16.0, 18.0, 19.0))
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    return materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    ), audit
