from __future__ import annotations

import importlib.metadata
from pathlib import Path

from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.math.observation_models import PointObservationModel
from euclid.modules import probabilistic_evaluation as pe
from euclid.modules.candidate_fitting import fit_candidate_window
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.scoring import score_probabilistic_prediction_artifact
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import EvaluationPlan, build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.reducers.models import BoundObservationModel
from euclid.search.backends import AnalyticSearchBackendAdapter
from euclid.search_planning import build_canonicalization_policy, build_search_plan

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_mapie_unavailable_metadata_has_stable_reason_and_scope(
    monkeypatch,
) -> None:
    def missing_version(package_name: str) -> str:
        assert package_name == "mapie"
        raise importlib.metadata.PackageNotFoundError("mapie")

    monkeypatch.setattr(pe.importlib_metadata, "version", missing_version)

    metadata = pe.build_mapie_calibration_method_metadata(
        method_id="enbpi_time_series_v1",
        guarantee_tier="approximate_mixing_time_series",
        assumption_ids=("weak_dependence_or_mixing",),
        assumptions={
            "weak_dependence_or_mixing": "declared rolling-origin residual diagnostics"
        },
        assumption_scope="mixing_time_series",
        calibration_partition_ids=("partition-h1", "partition-h2"),
        horizon_ids=(1, 2),
        calibration_indices=(2, 3, 5, 8),
    )

    assert metadata["method_id"] == "enbpi_time_series_v1"
    assert metadata["guarantee_tier"] == "approximate_mixing_time_series"
    assert metadata["assumption_ids"] == ["weak_dependence_or_mixing"]
    assert metadata["assumptions"] == {
        "weak_dependence_or_mixing": "declared rolling-origin residual diagnostics"
    }
    assert metadata["calibration_partition_id"] == "partition-h1"
    assert metadata["calibration_partition_ids"] == [
        "partition-h1",
        "partition-h2",
    ]
    assert metadata["horizon_ids"] == [1, 2]
    assert metadata["calibration_indices"] == [2, 3, 5, 8]
    assert metadata["backend"] == {
        "backend_id": "mapie",
        "status": "unavailable",
        "reason_codes": ["calibration_backend_unavailable"],
        "version": None,
        "provenance": {
            "package_name": "mapie",
            "import_status": "unavailable",
        },
    }


def test_prediction_and_score_artifacts_carry_calibration_method_metadata(
    monkeypatch,
) -> None:
    def missing_version(package_name: str) -> str:
        assert package_name == "mapie"
        raise importlib.metadata.PackageNotFoundError("mapie")

    monkeypatch.setattr(pe.importlib_metadata, "version", missing_version)
    context = _probabilistic_context()
    calibration_method_metadata = pe.build_mapie_calibration_method_metadata(
        method_id="enbpi_time_series_v1",
        guarantee_tier="approximate_mixing_time_series",
        assumption_ids=("weak_dependence_or_mixing",),
        assumptions={
            "weak_dependence_or_mixing": "declared rolling-origin residual diagnostics"
        },
        assumption_scope="mixing_time_series",
        calibration_partition_ids=("partition-h1", "partition-h2"),
        horizon_ids=(1, 2),
        calibration_indices=(2, 3, 5, 8),
    )

    prediction = pe.emit_probabilistic_prediction_artifact(
        catalog=context["catalog"],
        feature_view=context["feature_view"],
        evaluation_plan=context["evaluation_plan"],
        evaluation_segment=context["evaluation_segment"],
        fit_result=context["fit_result"],
        score_policy_manifest=context["score_policy"],
        stage_id="outer_test",
        forecast_object_type="distribution",
        calibration_method_metadata=calibration_method_metadata,
    )

    prediction_config = prediction.body["effective_probabilistic_config"]
    assert prediction_config["calibration_method"] == calibration_method_metadata

    score = score_probabilistic_prediction_artifact(
        catalog=context["catalog"],
        score_policy_manifest=context["score_policy"],
        prediction_artifact_manifest=prediction,
    )

    score_config = score.body["effective_probabilistic_config"]
    assert score.body["comparison_status"] == "comparable"
    assert score_config["calibration_method"] == calibration_method_metadata


def _probabilistic_context():
    catalog = load_contract_catalog(PROJECT_ROOT)
    feature_view, audit = _feature_view()
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=2,
    )
    search_plan = _search_plan(evaluation_plan)
    score_policy = _score_policy(catalog=catalog, evaluation_plan=evaluation_plan)
    adapter = AnalyticSearchBackendAdapter()
    observation_model = BoundObservationModel.from_runtime(PointObservationModel())
    candidate = None
    for rank, proposal in enumerate(
        adapter.default_proposals(
            search_plan=search_plan,
            feature_view=feature_view,
        )
    ):
        if proposal.candidate_id == "analytic_lag1_affine":
            candidate = adapter.realize_proposal(
                proposal=proposal,
                proposal_rank=rank,
                search_plan=search_plan,
                feature_view=feature_view,
                observation_model=observation_model,
            )
            break
    assert candidate is not None
    fit_result = fit_candidate_window(
        candidate=candidate,
        feature_view=feature_view,
        fit_window=evaluation_plan.development_segments[0],
        search_plan=search_plan,
        stage_id="outer_test",
    )
    return {
        "catalog": catalog,
        "feature_view": feature_view,
        "evaluation_plan": evaluation_plan,
        "evaluation_segment": evaluation_plan.development_segments[0],
        "score_policy": score_policy,
        "fit_result": fit_result,
    }


def _feature_view():
    observations = (1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0)
    snapshot = FrozenDatasetSnapshot(
        series_id="probabilistic-phase53-series",
        cutoff_available_at="2026-01-08T00:00:00Z",
        revision_policy="latest_available_revision_per_event_time",
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-0{index + 1}T00:00:00Z",
                available_at=f"2026-01-0{index + 1}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:{chr(ord('a') + index)}",
            )
            for index, value in enumerate(observations)
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


def _search_plan(evaluation_plan: EvaluationPlan):
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
        candidate_family_ids=("analytic_lag1_affine",),
        proposal_limit=1,
        search_class="exact_finite_enumeration",
        seasonal_period=4,
        minimum_description_gain_bits=-2.0,
    )


def _score_policy(*, catalog, evaluation_plan: EvaluationPlan):
    return pe.ManifestEnvelope.build(
        schema_name="probabilistic_score_policy_manifest@1.0.0",
        module_id="scoring",
        body={
            "score_policy_id": "phase53_distribution_policy_v1",
            "owner_prompt_id": "prompt.scoring-calibration-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "forecast_object_type": "distribution",
            "primary_score": "continuous_ranked_probability_score",
            "aggregation_mode": "per_horizon_mean_then_simplex_weighted_mean",
            "horizon_weights": [
                weight.as_dict() for weight in evaluation_plan.horizon_weights
            ],
            "entity_aggregation_mode": (
                "single_entity_only_no_cross_entity_aggregation"
            ),
            "comparison_class_rule": "identical_score_policy_required",
        },
        catalog=catalog,
    )
