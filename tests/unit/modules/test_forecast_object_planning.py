from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ADMITTED_FORECAST_OBJECT_TYPES = (
    "point",
    "distribution",
    "interval",
    "quantile",
    "event_probability",
)


def _feature_view():
    snapshot = FrozenDatasetSnapshot(
        series_id="demo-series",
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


@pytest.mark.parametrize("forecast_object_type", ADMITTED_FORECAST_OBJECT_TYPES)
def test_planning_roundtrip_for_all_object_types(forecast_object_type: str) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
        forecast_object_type=forecast_object_type,
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
        candidate_family_ids=("constant", "drift"),
        search_class=(
            "bounded_heuristic"
            if forecast_object_type == "point"
            else "exact_finite_enumeration"
        ),
    )

    evaluation_manifest = evaluation_plan.to_manifest(
        catalog,
        time_safety_audit_ref=audit.to_manifest(catalog).ref,
    )
    search_manifest = search_plan.to_manifest(catalog)

    assert evaluation_plan.forecast_object_type == forecast_object_type
    assert evaluation_manifest.body["forecast_object_type"] == forecast_object_type
    assert (
        evaluation_manifest.body["comparison_key"]["forecast_object_type"]
        == forecast_object_type
    )
    assert search_plan.forecast_object_type == forecast_object_type
    assert search_manifest.body["forecast_object_type"] == forecast_object_type


def test_point_only_assumptions_are_rejected() -> None:
    feature_view, audit = _feature_view()

    with pytest.raises(ContractValidationError) as exc_info:
        build_evaluation_plan(
            feature_view=feature_view,
            audit=audit,
            min_train_size=3,
            horizon=1,
            forecast_object_type="categorical_label",
        )

    assert exc_info.value.code == "unsupported_forecast_object_type"


def test_illegal_object_type_and_search_class_pairs_fail_fast() -> None:
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
        forecast_object_type="distribution",
    )

    with pytest.raises(ContractValidationError) as exc_info:
        build_search_plan(
            evaluation_plan=evaluation_plan,
            canonicalization_policy_ref=TypedRef(
                "canonicalization_policy_manifest@1.0.0",
                "canonicalization_default",
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
            candidate_family_ids=("constant",),
            search_class="stochastic_heuristic",
        )

    assert exc_info.value.code == "illegal_object_type_search_class_pair"
