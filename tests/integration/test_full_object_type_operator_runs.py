from __future__ import annotations

from pathlib import Path

import pytest

import euclid
from euclid.contracts.refs import TypedRef
from euclid.operator_runtime.models import (
    DEFAULT_ADMISSIBILITY_RULE_IDS,
    DEFAULT_RUN_SUPPORT_OBJECT_IDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINT_MANIFEST = PROJECT_ROOT / "fixtures/runtime/prototype-demo.yaml"
PROBABILISTIC_MANIFESTS = {
    "distribution": (
        PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-distribution-demo.yaml"
    ),
    "interval": (
        PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-interval-demo.yaml"
    ),
    "quantile": (
        PROJECT_ROOT / "fixtures/runtime/phase06/probabilistic-quantile-demo.yaml"
    ),
    "event_probability": (
        PROJECT_ROOT
        / "fixtures/runtime/phase06/probabilistic-event-probability-demo.yaml"
    ),
}


def test_operator_run_emits_point_artifact(tmp_path: Path) -> None:
    output_root = tmp_path / "point"
    result = euclid.run_demo_point_evaluation(
        manifest_path=POINT_MANIFEST,
        output_root=output_root,
    )

    run_result, evaluation_plan, search_plan, validation_scope, scope_ledger = (
        _runtime_surfaces(
            output_root=output_root,
            run_id=result.run.summary.run_result_ref.object_id,
        )
    )

    assert run_result.body["forecast_object_type"] == "point"
    assert run_result.body["prediction_artifact_refs"] == [
        result.prediction.prediction_artifact_ref.as_dict()
    ]
    assert evaluation_plan.body["forecast_object_type"] == "point"
    assert search_plan.body["forecast_object_type"] == "point"
    assert (
        run_result.body["primary_validation_scope_ref"]
        == validation_scope.ref.as_dict()
    )
    assert scope_ledger.body["run_support_object_ids"] == list(
        DEFAULT_RUN_SUPPORT_OBJECT_IDS
    )
    assert scope_ledger.body["admissibility_rule_ids"] == list(
        DEFAULT_ADMISSIBILITY_RULE_IDS
    )
    assert validation_scope.body["run_support_object_ids"] == list(
        DEFAULT_RUN_SUPPORT_OBJECT_IDS
    )
    assert validation_scope.body["admissibility_rule_ids"] == list(
        DEFAULT_ADMISSIBILITY_RULE_IDS
    )


def test_operator_run_emits_distribution_artifact(tmp_path: Path) -> None:
    output_root = tmp_path / "distribution"
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=PROBABILISTIC_MANIFESTS["distribution"],
        output_root=output_root,
    )

    run_result, evaluation_plan, search_plan, validation_scope, scope_ledger = (
        _runtime_surfaces(
            output_root=output_root,
            run_id=result.summary.run_result_ref.object_id,
        )
    )

    assert run_result.body["forecast_object_type"] == "distribution"
    assert run_result.body["prediction_artifact_refs"] == [
        result.summary.prediction_artifact_ref.as_dict()
    ]
    assert (
        run_result.body["primary_score_result_ref"]
        == result.summary.score_result_ref.as_dict()
    )
    assert (
        run_result.body["primary_calibration_result_ref"]
        == result.summary.calibration_result_ref.as_dict()
    )
    assert evaluation_plan.body["forecast_object_type"] == "distribution"
    assert search_plan.body["forecast_object_type"] == "distribution"
    assert validation_scope.body["public_forecast_object_type"] == "distribution"
    assert scope_ledger.body["forecast_object_type"] == "distribution"


@pytest.mark.parametrize(
    "forecast_object_type",
    ("interval", "quantile", "event_probability"),
)
def test_operator_run_emits_interval_quantile_and_event_probability_artifacts(
    tmp_path: Path,
    forecast_object_type: str,
) -> None:
    output_root = tmp_path / forecast_object_type
    result = euclid.run_demo_probabilistic_evaluation(
        manifest_path=PROBABILISTIC_MANIFESTS[forecast_object_type],
        output_root=output_root,
    )

    run_result, evaluation_plan, search_plan, validation_scope, scope_ledger = (
        _runtime_surfaces(
            output_root=output_root,
            run_id=result.summary.run_result_ref.object_id,
        )
    )

    assert result.summary.forecast_object_type == forecast_object_type
    assert result.prediction.forecast_object_type == forecast_object_type
    assert result.calibration.forecast_object_type == forecast_object_type
    assert run_result.body["forecast_object_type"] == forecast_object_type
    assert evaluation_plan.body["forecast_object_type"] == forecast_object_type
    assert search_plan.body["forecast_object_type"] == forecast_object_type
    assert validation_scope.body["public_forecast_object_type"] == forecast_object_type
    assert scope_ledger.body["forecast_object_type"] == forecast_object_type
    assert (
        run_result.body["primary_validation_scope_ref"]
        == validation_scope.ref.as_dict()
    )


def _runtime_surfaces(*, output_root: Path, run_id: str):
    graph = euclid.load_demo_run_artifact_graph(output_root=output_root, run_id=run_id)
    run_result = graph.inspect(graph.root_ref).manifest
    evaluation_plan = graph.inspect(
        TypedRef(**run_result.body["evaluation_plan_ref"])
    ).manifest
    search_plan = graph.inspect(TypedRef(**run_result.body["search_plan_ref"])).manifest
    validation_scope = graph.inspect(
        TypedRef(**run_result.body["primary_validation_scope_ref"])
    ).manifest
    scope_ledger = graph.inspect(
        TypedRef(**run_result.body["scope_ledger_ref"])
    ).manifest
    return run_result, evaluation_plan, search_plan, validation_scope, scope_ledger
