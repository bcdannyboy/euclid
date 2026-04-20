from __future__ import annotations

import pytest

from euclid.benchmarks.submitters import _build_submitter_candidate_ledger
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.refs import TypedRef
from euclid.modules.evaluation_governance import (
    ComparisonKey,
    build_comparison_universe,
)
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.reducers.models import parse_reducer_composition
from euclid.search.backends import (
    DescriptiveSearchProposal,
    run_descriptive_search_backends,
)


def test_illegal_composition_structure_is_rejected() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        parse_reducer_composition(
            {
                "operator_id": "additive_residual",
                "base_reducer": "shared_component",
                "residual_reducer": "shared_component",
                "shared_observation_model": "point_identity",
            }
        )

    assert exc_info.value.code == "invalid_additive_residual_composition"


def test_incompatible_composition_candidates_cannot_be_compared() -> None:
    candidate_key = ComparisonKey(
        forecast_object_type="point",
        score_policy_ref=TypedRef("point_score_policy_manifest@1.0.0", "policy"),
        horizon_set=(1,),
        scored_origin_set_id="outer_fold_0",
        composition_signature="piecewise:head|tail",
    )
    baseline_key = ComparisonKey(
        forecast_object_type="point",
        score_policy_ref=TypedRef("point_score_policy_manifest@1.0.0", "policy"),
        horizon_set=(1,),
        scored_origin_set_id="outer_fold_0",
        composition_signature="regime:stable|volatile",
    )

    with pytest.raises(ContractValidationError) as exc_info:
        build_comparison_universe(
            selected_candidate_id="candidate",
            baseline_id="baseline",
            candidate_primary_score=1.0,
            baseline_primary_score=2.0,
            candidate_comparison_key=candidate_key,
            baseline_comparison_key=baseline_key,
        )

    assert exc_info.value.code == "comparison_key_mismatch"
    assert exc_info.value.details["composition_signature"] == {
        "candidate": "piecewise:head|tail",
        "baseline": "regime:stable|volatile",
    }


def test_reason_codes_are_persisted_into_artifacts() -> None:
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
        candidate_family_ids=("illegal_additive_residual",),
        search_class="exact_finite_enumeration",
        proposal_limit=1,
    )
    proposal = DescriptiveSearchProposal(
        candidate_id="illegal_additive_residual",
        primitive_family="analytic",
        form_class="closed_form_expression",
        composition_payload={
            "operator_id": "additive_residual",
            "base_reducer": "shared_component",
            "residual_reducer": "shared_component",
            "shared_observation_model": "point_identity",
        },
    )

    result = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=(proposal,),
    )
    candidate_ledger = _build_submitter_candidate_ledger(
        ordered_proposals=(proposal,),
        search_plan=search_plan,
        search_result=result,
    )

    assert result.rejected_diagnostics[0].reason_code == "syntax_invalid"
    assert candidate_ledger[0].ledger_status == "rejected"
    assert candidate_ledger[0].reason_codes == ("syntax_invalid",)
    assert candidate_ledger[0].details["diagnostics"][0]["reason_code"] == (
        "syntax_invalid"
    )


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="composition-legality-series",
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
