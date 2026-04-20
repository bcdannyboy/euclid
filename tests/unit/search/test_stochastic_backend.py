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
    DescriptiveSearchProposal,
    run_descriptive_search_backends,
)


def test_backend_emits_stochastic_disclosures() -> None:
    feature_view, audit = _feature_view()
    proposals = tuple(
        DescriptiveSearchProposal(
            candidate_id=f"analytic_seed_{index}",
            primitive_family="analytic",
            form_class="closed_form_expression",
            parameter_values={"intercept": 10.0 + index},
        )
        for index in range(4)
    )
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=tuple(proposal.candidate_id for proposal in proposals),
        proposal_limit=3,
        random_seed="17",
        candidate_batch_size=1,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=proposals,
    )

    assert result.coverage.disclosures == {
        "proposal_distribution": "seeded_sha256_permutation_without_replacement",
        "seed_policy": "root_seed=17;derivation=deterministic_scope_hash",
        "restart_policy": (
            "batch_size=1;restart_until_budget_or_exhaustion;restarts_used=3"
        ),
        "stop_rule": "proposal_limit=3",
    }

    candidate = result.accepted_candidates[0]
    search_evidence = candidate.evidence_layer.transient_diagnostics["search_evidence"]
    hook_refs = {
        hook.hook_name: hook.hook_ref
        for hook in candidate.evidence_layer.replay_hooks.hooks
    }

    assert search_evidence["search_class"] == "stochastic_heuristic"
    assert search_evidence["proposal_distribution"] == (
        "seeded_sha256_permutation_without_replacement"
    )
    assert search_evidence["seed_policy"] == (
        "root_seed=17;derivation=deterministic_scope_hash"
    )
    assert len(search_evidence["restart_records"]) == 3
    assert hook_refs["restart_policy"] == (
        "batch_size=1;restart_until_budget_or_exhaustion;restarts_used=3"
    )
    assert (
        "search_evidence"
        in candidate.evidence_layer.backend_origin_record.backend_private_fields
    )


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="phase03-stochastic-series",
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
    proposal_limit: int,
    random_seed: str,
    candidate_batch_size: int,
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
        search_class="stochastic_heuristic",
        proposal_limit=proposal_limit,
        random_seed=random_seed,
        candidate_batch_size=candidate_batch_size,
    )
