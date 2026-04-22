from __future__ import annotations

import pytest

from euclid.cir.models import (
    CIRBackendOriginRecord,
    CIRForecastOperator,
    CIRInputSignature,
    CIRModelCodeDecomposition,
    CIRReplayHook,
    CIRReplayHooks,
)
from euclid.cir.normalize import build_cir_candidate_from_expression
from euclid.contracts.refs import TypedRef
from euclid.expr.ast import BinaryOp, Feature, Parameter
from euclid.modules.candidate_fitting import fit_candidate_window
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import build_canonicalization_policy, build_search_plan
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety


def test_candidate_fitting_routes_expression_cir_through_unified_fit_layer() -> None:
    feature_view, audit = _feature_view((0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    candidate = _linear_expression_candidate()

    fit = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=_search_plan(feature_view=feature_view, audit=audit),
    )

    assert fit.backend_id == "euclid_unified_fit_layer_v1"
    assert fit.objective_id == "squared_error"
    assert fit.parameter_summary["intercept"] == pytest.approx(1.0)
    assert fit.parameter_summary["slope"] == pytest.approx(1.0)
    assert fit.optimizer_diagnostics["fit_layer"] == "src/euclid/fit"
    assert fit.optimizer_diagnostics["claim_boundary"][
        "claim_publication_allowed"
    ] is False


def _search_plan(*, feature_view, audit):
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
        candidate_family_ids=("linear-expression",),
        search_class="exact_finite_enumeration",
        proposal_limit=1,
    )


def _feature_view(values: tuple[float, ...]):
    snapshot = FrozenDatasetSnapshot(
        series_id="candidate-fitting-expression-series",
        cutoff_available_at=f"2026-01-{len(values):02d}T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-{index + 1:02d}T00:00:00Z",
                available_at=f"2026-01-{index + 1:02d}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:fit-{index}",
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


def _linear_expression_candidate():
    expression = BinaryOp(
        "add",
        BinaryOp("mul", Parameter("slope"), Feature("lag_1")),
        Parameter("intercept"),
    )
    return build_cir_candidate_from_expression(
        expression=expression,
        cir_family_id="analytic",
        cir_form_class="expression_ir",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("lag_1",),
        ),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=CIRModelCodeDecomposition(
            L_family_bits=1.0,
            L_structure_bits=1.0,
            L_literals_bits=0.0,
            L_params_bits=2.0,
            L_state_bits=0.0,
        ),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="test-expression",
            adapter_class="unit_test",
            source_candidate_id="linear-expression",
            search_class="unit_test",
            proposal_rank=0,
        ),
        replay_hooks=CIRReplayHooks(
            hooks=(CIRReplayHook(hook_name="test", hook_ref="unit:test"),)
        ),
    )
