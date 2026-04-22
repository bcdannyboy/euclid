from __future__ import annotations

from euclid.adapters.decomposition import (
    LegacyDecompositionProposal,
    normalize_legacy_decomposition_candidate,
)
from euclid.adapters.portfolio import (
    ComparableBackendFinalist,
    normalize_cir_finalist,
)
from euclid.adapters.sparse_library import (
    LegacySparseProposal,
    normalize_legacy_sparse_candidate,
)
from euclid.contracts.refs import TypedRef
from euclid.math.observation_models import PointObservationModel
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.reducers.models import BoundObservationModel


def test_backend_adapters_emit_public_normalization_contracts() -> None:
    feature_view, search_plan = _search_context()
    observation_model = BoundObservationModel.from_runtime(PointObservationModel())

    decomposition_candidate = normalize_legacy_decomposition_candidate(
        spec=LegacyDecompositionProposal(
            candidate_id="decomposition_lag1",
            feature_dependencies=("lag_1",),
            parameter_values={"intercept": 1.0, "lag_coefficient": 0.9},
            max_lag=1,
        ),
        search_plan=search_plan,
        feature_view=feature_view,
        observation_model=observation_model,
        proposal_rank=0,
    )
    sparse_candidate = normalize_legacy_sparse_candidate(
        spec=LegacySparseProposal(
            candidate_id="sparse_recursive",
            primitive_family="recursive",
            form_class="state_recurrence",
            literal_values={"alpha": 0.5},
            persistent_state={"level": 10.0, "step_count": 0},
            max_lag=1,
        ),
        search_plan=search_plan,
        feature_view=feature_view,
        observation_model=observation_model,
        proposal_rank=1,
    )

    assert (
        decomposition_candidate.evidence_layer.backend_origin_record.adapter_class
        == "legacy_non_claim_decomposition_adapter"
    )
    assert sparse_candidate.evidence_layer.backend_origin_record.adapter_class == (
        "legacy_non_claim_sparse_adapter"
    )
    assert (
        decomposition_candidate.evidence_layer.backend_origin_record.normalization_scope
        == "cir_structural_execution_model_code_only"
    )
    assert (
        sparse_candidate.evidence_layer.backend_origin_record.comparability_scope
        == "legacy_compatibility_only_not_production_evidence"
    )
    assert "legacy_compatibility_trace" in (
        decomposition_candidate.evidence_layer.backend_origin_record.backend_private_fields
    )
    assert (
        "legacy_compatibility_trace"
        in sparse_candidate.evidence_layer.backend_origin_record.backend_private_fields
    )
    assert (
        decomposition_candidate.evidence_layer.transient_diagnostics[
            "legacy_non_claim_adapter"
        ]["production_evidence_allowed"]
        is False
    )
    assert (
        sparse_candidate.evidence_layer.transient_diagnostics[
            "legacy_non_claim_adapter"
        ]["replacement_engine_id"]
        == "pysindy-engine-v1"
    )


def test_portfolio_adapter_normalization_extracts_comparable_finalist_record() -> None:
    feature_view, search_plan = _search_context()
    observation_model = BoundObservationModel.from_runtime(PointObservationModel())
    candidate = normalize_legacy_decomposition_candidate(
        spec=LegacyDecompositionProposal(
            candidate_id="decomposition_intercept",
            parameter_values={"intercept": 12.0},
        ),
        search_plan=search_plan,
        feature_view=feature_view,
        observation_model=observation_model,
        proposal_rank=0,
    )

    finalist = normalize_cir_finalist(
        candidate,
        total_code_bits=5.0,
        description_gain_bits=1.5,
        structure_code_bits=2.0,
        coverage_statement="complete_over_declared_canonical_program_space",
        exactness_ceiling="exact_over_declared_fragment_only",
        scope_declaration="finite_exactness_limited_to_declared_canonical_program_space",
        provenance_id="analytic_backend",
    )

    assert isinstance(finalist, ComparableBackendFinalist)
    assert finalist.provenance_id == "analytic_backend"
    assert finalist.adapter_class == "legacy_non_claim_decomposition_adapter"
    assert finalist.backend_family == "analytic"
    assert finalist.candidate_id == "decomposition_intercept"
    assert finalist.forecast_object_type == "point"
    assert finalist.total_code_bits == 5.0
    assert finalist.candidate_hash == candidate.canonical_hash()


def _search_context():
    snapshot = FrozenDatasetSnapshot(
        series_id="adapter-series",
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
        candidate_family_ids=("analytic_intercept", "analytic_lag1_affine"),
        search_class="exact_finite_enumeration",
        proposal_limit=2,
        seasonal_period=4,
    )
    return feature_view, search_plan
