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


def test_backend_emits_rewrite_and_extractor_disclosures(
) -> None:
    feature_view, audit = _feature_view()
    proposals = (
        DescriptiveSearchProposal(
            candidate_id="analytic_intercept_simple",
            primitive_family="analytic",
            form_class="closed_form_expression",
            parameter_values={"intercept": 14.0},
        ),
        DescriptiveSearchProposal(
            candidate_id="analytic_piecewise_complex",
            primitive_family="analytic",
            form_class="closed_form_expression",
            feature_dependencies=("lag_1",),
            parameter_values={"intercept": 14.0},
            literal_values={"upper_cut": 3.0, "lower_cut": 1.0},
            persistent_state={"step_count": 1, "running_total": 0.0},
            composition_payload={
                "operator_id": "piecewise",
                "ordered_partition": [
                    {"start_literal": 0.0, "end_literal": 1.0, "reducer_id": "head"},
                    {"start_literal": 1.0, "end_literal": 3.0, "reducer_id": "tail"},
                ],
            },
            history_access_mode="bounded_lag_window",
            max_lag=1,
        ),
    )
    search_plan = _build_search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_family_ids=tuple(proposal.candidate_id for proposal in proposals),
        search_class="equality_saturation_heuristic",
        proposal_limit=1,
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=proposals,
    )

    assert result.coverage.disclosures == {
        "rewrite_system": "egraph_engine_required_for_expression_cir_rewrites",
        "extractor_cost": "declared_by_egraph_engine_rewrite_trace",
        "legacy_fragment_backend_mode": "no_sort_only_equality_saturation",
        "stop_rule": "proposal_limit=1",
    }

    candidate = result.accepted_candidates[0]
    search_evidence = candidate.evidence_layer.transient_diagnostics["search_evidence"]

    assert search_evidence["search_class"] == "equality_saturation_heuristic"
    assert search_evidence["rewrite_system"] == (
        "egraph_engine_required_for_expression_cir_rewrites"
    )
    assert search_evidence["extractor_cost"] == (
        "declared_by_egraph_engine_rewrite_trace"
    )
    assert (
        search_evidence["legacy_fragment_backend_mode"]
        == "no_sort_only_equality_saturation"
    )
    assert set(search_evidence["rewrite_space_candidate_ids"]) == {
        "analytic_intercept_simple",
        "analytic_piecewise_complex",
    }
    assert search_evidence["selected_candidate_id"] == "analytic_intercept_simple"
    assert (
        "search_evidence"
        in candidate.evidence_layer.backend_origin_record.backend_private_fields
    )


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="phase03-equality-saturation-series",
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
