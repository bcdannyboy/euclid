from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.modules.candidate_fitting import fit_candidate_window
from euclid.modules.evaluation import emit_point_prediction_artifact
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
    _validate_search_honesty_evidence,
    run_descriptive_search_backends,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_exact_enumeration_artifacts_are_replayable() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=2,
    )
    search_plan = _search_plan(
        evaluation_plan=evaluation_plan,
        candidate_ids=(
            "algorithmic_last_observation",
            "algorithmic_running_half_average",
        ),
        search_class="exact_finite_enumeration",
        proposal_limit=2,
    )
    score_policy = _score_policy_manifest(catalog, evaluation_plan)

    first = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        adapters=(AlgorithmicSearchBackendAdapter(),),
    )
    second = run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        adapters=(AlgorithmicSearchBackendAdapter(),),
    )
    candidate_id = "algorithmic_last_observation"
    first_candidate = next(
        candidate
        for candidate in first.accepted_candidates
        if candidate.evidence_layer.backend_origin_record.source_candidate_id
        == candidate_id
    )
    second_candidate = next(
        candidate
        for candidate in second.accepted_candidates
        if candidate.evidence_layer.backend_origin_record.source_candidate_id
        == candidate_id
    )

    first_fit = fit_candidate_window(
        candidate=first_candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )
    second_fit = fit_candidate_window(
        candidate=second_candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )
    first_artifact = emit_point_prediction_artifact(
        catalog=catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.development_segments[0],
        fit_result=first_fit,
        score_policy_manifest=score_policy,
        stage_id="outer_test",
    )
    second_artifact = emit_point_prediction_artifact(
        catalog=catalog,
        feature_view=feature_view,
        evaluation_plan=evaluation_plan,
        evaluation_segment=evaluation_plan.development_segments[0],
        fit_result=second_fit,
        score_policy_manifest=score_policy,
        stage_id="outer_test",
    )

    assert (
        first_candidate.evidence_layer.transient_diagnostics["search_evidence"]
        == second_candidate.evidence_layer.transient_diagnostics["search_evidence"]
    )
    assert first_fit.final_state == second_fit.final_state
    assert first_artifact.body["rows"] == second_artifact.body["rows"]


def test_exact_enumeration_overclaim_is_rejected() -> None:
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    bounded_plan = _search_plan(
        evaluation_plan=evaluation_plan,
        candidate_ids=("analytic_intercept", "analytic_lag1_affine"),
        search_class="bounded_heuristic",
        proposal_limit=1,
    )
    bounded_result = run_descriptive_search_backends(
        search_plan=bounded_plan,
        feature_view=feature_view,
    )

    with pytest.raises(ContractValidationError) as exc_info:
        _validate_search_honesty_evidence(
            search_plan=replace(bounded_plan, search_class="exact_finite_enumeration"),
            coverage_disclosures=bounded_result.coverage.disclosures,
            accepted_candidates=bounded_result.accepted_candidates,
        )

    assert exc_info.value.code == "search_honesty_evidence_missing"


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="exact-enumeration-pipeline-series",
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


def _search_plan(*, evaluation_plan, candidate_ids, search_class, proposal_limit):
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
        search_class=search_class,
        proposal_limit=proposal_limit,
    )


def _score_policy_manifest(catalog, evaluation_plan):
    from euclid.manifests.base import ManifestEnvelope

    return ManifestEnvelope.build(
        schema_name="point_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": "exact_enumeration_prediction_policy_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": "point",
            "point_loss_id": "absolute_error",
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [
                weight.as_dict() for weight in evaluation_plan.horizon_weights
            ],
            "entity_aggregation_mode": (
                "single_entity_only_no_cross_entity_aggregation"
            ),
            "secondary_diagnostic_ids": [],
            "forbidden_primary_metric_ids": [],
            "lower_is_better": True,
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )
