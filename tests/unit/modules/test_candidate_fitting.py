from __future__ import annotations

from pathlib import Path

import pytest

from euclid.cir.models import (
    CIRBackendOriginRecord,
    CIRCanonicalSerialization,
    CIREvidenceLayer,
    CIRExecutionLayer,
    CIRForecastOperator,
    CIRHistoryAccessContract,
    CIRInputSignature,
    CIRModelCodeDecomposition,
    CIRReplayHooks,
    CIRStateSignature,
    CIRStructuralLayer,
    CandidateIntermediateRepresentation,
)
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.math.observation_models import PointObservationModel
from euclid.modules.candidate_fitting import (
    build_candidate_fit_artifacts,
    fit_candidate_window,
)
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.reducers.models import BoundObservationModel, ReducerStateObject
from euclid.search.backends import (
    AnalyticSearchBackendAdapter,
    RecursiveSearchBackendAdapter,
    SpectralSearchBackendAdapter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_fit_candidate_window_uses_only_fold_local_training_rows_for_recursive_state(
) -> None:
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("recursive_level_smoother",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        RecursiveSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="recursive_level_smoother",
    )

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    alpha = float(fit.parameter_summary["alpha"])
    segment = evaluation_plan.development_segments[0]
    training_targets = tuple(
        float(row["target"])
        for row in feature_view.rows[
            segment.train_start_index : segment.train_end_index + 1
        ]
    )
    level = training_targets[0]
    for observed in training_targets:
        level = (alpha * observed) + ((1.0 - alpha) * level)

    assert fit.backend_id == "deterministic_alpha_grid_v1"
    assert fit.optimizer_diagnostics["seed_value"] == "0"
    assert fit.training_row_count == 3
    assert fit.state_transition_count == 3
    assert fit.state_transitions[-1].observation_index == 2
    assert fit.final_state["level"] == pytest.approx(level)
    assert fit.final_state["step_count"] == 3


def test_fit_candidate_window_is_deterministic_for_spectral_candidates() -> None:
    feature_view, audit = _feature_view((0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=4,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("spectral_harmonic_1",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        SpectralSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="spectral_harmonic_1",
    )

    first = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )
    second = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    assert first.backend_id == "deterministic_harmonic_least_squares_v1"
    assert first.parameter_summary == second.parameter_summary
    assert first.final_state == second.final_state
    assert first.state_transitions == second.state_transitions
    assert first.fitted_candidate.canonical_hash() == (
        second.fitted_candidate.canonical_hash()
    )


def test_build_candidate_fit_artifacts_links_candidate_state_and_fit_backend() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_lag1_affine",),
        seasonal_period=4,
    )
    candidate = _realize_default_candidate(
        AnalyticSearchBackendAdapter(),
        search_plan=search_plan,
        feature_view=feature_view,
        candidate_id="analytic_lag1_affine",
    )
    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
    )

    artifacts = build_candidate_fit_artifacts(
        catalog=catalog,
        fit_result=fit,
        search_plan_ref=TypedRef(
            "search_plan_manifest@1.0.0",
            search_plan.search_plan_id,
        ),
        selection_floor_bits=0.0,
        description_gain_bits=1.25,
    )

    assert artifacts.candidate_spec.body["fit_window_id"] == "outer_fold_0"
    assert artifacts.candidate_state.body["optimizer_backend_id"] == fit.backend_id
    assert artifacts.candidate_state.body["state_transition_count"] == 3
    assert artifacts.reducer_artifact.body["candidate_state_ref"] == (
        artifacts.candidate_state.ref.as_dict()
    )
    assert artifacts.reducer_artifact.body["fit_backend_id"] == fit.backend_id


def test_fit_candidate_window_requires_closed_cir_candidate_before_frontier_scoring(
) -> None:
    feature_view, audit = _feature_view((10.0, 12.0, 13.0, 15.0, 16.0, 18.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    search_plan = _search_plan(
        feature_view=feature_view,
        audit=audit,
        candidate_ids=("analytic_intercept",),
        seasonal_period=4,
    )

    with pytest.raises(ContractValidationError, match="fully normalized CIR"):
        fit_candidate_window(
            candidate=_unnormalized_candidate(),
            feature_view=feature_view,
            fit_window=evaluation_plan.development_segments[0],
            search_plan=search_plan,
        )


def _search_plan(
    *,
    feature_view,
    audit,
    candidate_ids: tuple[str, ...],
    seasonal_period: int,
):
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3 if len(feature_view.rows) <= 6 else 4,
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
        search_class="exact_finite_enumeration",
        proposal_limit=len(candidate_ids),
        seasonal_period=seasonal_period,
    )


def _realize_default_candidate(
    adapter,
    *,
    search_plan,
    feature_view,
    candidate_id: str,
):
    observation_model = BoundObservationModel.from_runtime(PointObservationModel())
    for rank, proposal in enumerate(
        adapter.default_proposals(
            search_plan=search_plan,
            feature_view=feature_view,
        )
    ):
        if proposal.candidate_id != candidate_id:
            continue
        return adapter.realize_proposal(
            proposal=proposal,
            proposal_rank=rank,
            search_plan=search_plan,
            feature_view=feature_view,
            observation_model=observation_model,
        )
    raise AssertionError(f"missing candidate proposal: {candidate_id}")


def _feature_view(values: tuple[float, ...]):
    snapshot = FrozenDatasetSnapshot(
        series_id="candidate-fitting-series",
        cutoff_available_at=f"2026-01-0{len(values)}T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-{index + 1:02d}T00:00:00Z",
                available_at=f"2026-01-{index + 1:02d}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:{index}",
            )
            for index, value in enumerate(values)
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )
    return feature_view, audit


def _unnormalized_candidate() -> CandidateIntermediateRepresentation:
    return CandidateIntermediateRepresentation(
        structural_layer=CIRStructuralLayer(
            cir_family_id="analytic",
            cir_form_class="closed_form_expression",
            input_signature=CIRInputSignature(target_series="target"),
            state_signature=CIRStateSignature(
                persistent_state=ReducerStateObject()
            ),
        ),
        execution_layer=CIRExecutionLayer(
            history_access_contract=CIRHistoryAccessContract(
                contract_id="full_prefix",
                access_mode="full_prefix",
            ),
            state_update_law_id="analytic_identity_update",
            forecast_operator=CIRForecastOperator(
                operator_id="one_step_point_forecast",
                horizon=1,
            ),
            observation_model_binding=BoundObservationModel.from_runtime(
                PointObservationModel()
            ),
        ),
        evidence_layer=CIREvidenceLayer(
            canonical_serialization=CIRCanonicalSerialization(
                canonical_bytes=b"{}",
                content_hash="sha256:placeholder",
            ),
            model_code_decomposition=CIRModelCodeDecomposition(
                L_family_bits=1.0,
                L_structure_bits=1.0,
                L_literals_bits=0.0,
                L_params_bits=1.0,
                L_state_bits=0.0,
            ),
            backend_origin_record=CIRBackendOriginRecord(
                adapter_id="analytic-search",
                adapter_class="bounded_grammar",
                source_candidate_id="analytic_intercept",
                search_class="exact_finite_enumeration",
                proposal_rank=0,
            ),
            replay_hooks=CIRReplayHooks(),
            transient_diagnostics={},
        ),
    )
