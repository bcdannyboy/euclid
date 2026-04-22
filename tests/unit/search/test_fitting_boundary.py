from __future__ import annotations

from euclid.contracts.refs import TypedRef
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import build_canonicalization_policy, build_search_plan
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.search.backends import AnalyticSearchBackendAdapter


def test_analytic_search_proposals_do_not_fit_parameters_from_targets() -> None:
    adapter = AnalyticSearchBackendAdapter()
    first_feature_view, first_audit = _feature_view((10.0, 20.0, 40.0, 80.0, 160.0))
    second_feature_view, second_audit = _feature_view((1000.0, -5.0, 12.0, 99.0, -42.0))

    first = adapter.default_proposals(
        search_plan=_search_plan(first_feature_view, first_audit),
        feature_view=first_feature_view,
    )
    second = adapter.default_proposals(
        search_plan=_search_plan(second_feature_view, second_audit),
        feature_view=second_feature_view,
    )

    first_params = {proposal.candidate_id: proposal.parameter_values for proposal in first}
    second_params = {
        proposal.candidate_id: proposal.parameter_values for proposal in second
    }
    assert first_params == second_params
    assert first_params["analytic_intercept"] == {"intercept": 13.0}
    assert first_params["analytic_lag1_affine"] == {
        "intercept": 1.0,
        "lag_coefficient": 1.0,
    }


def _search_plan(feature_view, audit):
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
        candidate_family_ids=("analytic_intercept", "analytic_lag1_affine"),
        search_class="exact_finite_enumeration",
        proposal_limit=2,
    )


def _feature_view(values: tuple[float, ...]):
    snapshot = FrozenDatasetSnapshot(
        series_id="search-fitting-boundary-series",
        cutoff_available_at=f"2026-01-{len(values):02d}T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-{index + 1:02d}T00:00:00Z",
                available_at=f"2026-01-{index + 1:02d}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:search-fit-{index}",
            )
            for index, value in enumerate(values)
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    return (
        materialize_feature_view(
            snapshot=snapshot,
            audit=audit,
            feature_spec=default_feature_spec(),
        ),
        audit,
    )
