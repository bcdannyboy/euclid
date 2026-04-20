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


def test_same_seed_replays_identically() -> None:
    feature_view, audit = _feature_view()
    proposal_specs = _proposal_specs()
    first_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=tuple(spec.candidate_id for spec in proposal_specs),
        random_seed="17",
    )
    second_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=tuple(spec.candidate_id for spec in proposal_specs),
        random_seed="17",
    )

    first = run_descriptive_search_backends(
        search_plan=first_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=proposal_specs,
    )
    second = run_descriptive_search_backends(
        search_plan=second_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=proposal_specs,
    )

    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in first.accepted_candidates
    ] == [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in second.accepted_candidates
    ]
    assert first.coverage.disclosures == second.coverage.disclosures


def test_different_seed_changes_only_declared_stochastic_surfaces() -> None:
    feature_view, audit = _feature_view()
    proposal_specs = _proposal_specs()
    seed_17_result = run_descriptive_search_backends(
        search_plan=_search_plan(
            feature_view=feature_view,
            audit=audit,
            candidate_ids=tuple(spec.candidate_id for spec in proposal_specs),
            random_seed="17",
        ),
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=proposal_specs,
    )
    seed_23_result = run_descriptive_search_backends(
        search_plan=_search_plan(
            feature_view=feature_view,
            audit=audit,
            candidate_ids=tuple(spec.candidate_id for spec in proposal_specs),
            random_seed="23",
        ),
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=proposal_specs,
    )

    seed_17_ids = [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in seed_17_result.accepted_candidates
    ]
    seed_23_ids = [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in seed_23_result.accepted_candidates
    ]

    assert seed_17_ids != seed_23_ids
    assert seed_17_result.coverage.disclosures["proposal_distribution"] == (
        seed_23_result.coverage.disclosures["proposal_distribution"]
    )
    assert seed_17_result.accepted_candidates[0].evidence_layer.transient_diagnostics[
        "search_evidence"
    ]["declared_stochastic_surfaces"] == seed_23_result.accepted_candidates[
        0
    ].evidence_layer.transient_diagnostics["search_evidence"][
        "declared_stochastic_surfaces"
    ]


def _proposal_specs():
    return tuple(
        DescriptiveSearchProposal(
            candidate_id=f"analytic_seed_{index}",
            primitive_family="analytic",
            form_class="closed_form_expression",
            parameter_values={"intercept": 10.0 + index},
        )
        for index in range(4)
    )


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="stochastic-search-pipeline-series",
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


def _search_plan(*, feature_view, audit, candidate_ids, random_seed):
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
        candidate_family_ids=candidate_ids,
        search_class="stochastic_heuristic",
        proposal_limit=3,
        random_seed=random_seed,
        candidate_batch_size=1,
    )
