from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

import euclid
from euclid.contracts.errors import ContractValidationError
from euclid.operator_runtime.replay import replay_operator
from euclid.operator_runtime.run import run_operator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_LOCAL_DATASET = (
    PROJECT_ROOT / "fixtures/runtime/shared-local/shared-local-panel-series.csv"
)


def test_operator_run_emits_shared_local_artifacts(tmp_path: Path) -> None:
    manifest_path = _write_shared_local_manifest(
        tmp_path=tmp_path,
        request_id="shared-local-operator-run",
        dataset_csv=SHARED_LOCAL_DATASET,
    )

    result = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "shared-local-output",
    )
    graph = euclid.load_demo_run_artifact_graph(output_root=result.paths.output_root)
    run_result = graph.inspect(graph.root_ref).manifest
    prediction_artifact = graph.inspect(
        run_result.body["prediction_artifact_refs"][0]
    ).manifest
    validation_scope = graph.inspect(
        run_result.body["primary_validation_scope_ref"]
    ).manifest
    claim_card = graph.inspect(run_result.body["primary_claim_card_ref"]).manifest
    reducer_artifact = graph.inspect(
        run_result.body["primary_reducer_artifact_ref"]
    ).manifest

    assert result.summary.result_mode == "candidate_publication"
    assert "shared_plus_local_decomposition" in result.summary.extension_lane_ids
    assert prediction_artifact.body["entity_panel"] == ["entity-a", "entity-b"]
    assert validation_scope.body["entity_scope"] == "declared_entity_panel"
    assert validation_scope.body["entity_panel"] == ["entity-a", "entity-b"]
    assert "shared_lag_coefficient" in reducer_artifact.body["parameter_summary"]
    assert (
        "cross_entity_panel_forecast_within_declared_validation_scope"
        in claim_card.body["allowed_interpretation_codes"]
    )


def test_shared_local_replay_is_exact(tmp_path: Path) -> None:
    manifest_path = _write_shared_local_manifest(
        tmp_path=tmp_path,
        request_id="shared-local-replay-run",
        dataset_csv=SHARED_LOCAL_DATASET,
    )

    result = run_operator(
        manifest_path=manifest_path,
        output_root=tmp_path / "shared-local-replay-output",
    )
    replay = replay_operator(
        output_root=result.paths.output_root,
        run_id="shared-local-replay-run",
    )
    bundle = euclid.inspect_demo_replay_bundle(output_root=result.paths.output_root)

    assert replay.summary.replay_verification_status == "verified"
    assert replay.summary.failure_reason_codes == ()
    assert bundle.recorded_stage_order[-2:] == (
        "publication_decision_resolved",
        "run_result_assembled",
    )


def test_unseen_entity_rule_is_enforced(tmp_path: Path) -> None:
    dataset_csv = tmp_path / "shared-local-unseen-entity.csv"
    dataset_csv.write_text(
        dedent(
            """
            entity,event_time,availability_time,target
            entity-a,2026-01-01T00:00:00Z,2026-01-01T06:00:00Z,10
            entity-a,2026-01-02T00:00:00Z,2026-01-02T06:00:00Z,11
            entity-a,2026-01-03T00:00:00Z,2026-01-03T06:00:00Z,12
            entity-b,2026-01-01T00:00:00Z,2026-01-01T06:00:00Z,20
            entity-b,2026-01-02T00:00:00Z,2026-01-02T06:00:00Z,21
            entity-b,2026-01-03T00:00:00Z,2026-01-03T06:00:00Z,22
            entity-c,2026-01-07T00:00:00Z,2026-01-07T06:00:00Z,30
            entity-c,2026-01-08T00:00:00Z,2026-01-08T06:00:00Z,31
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    manifest_path = _write_shared_local_manifest(
        tmp_path=tmp_path,
        request_id="shared-local-unseen-entity-run",
        dataset_csv=dataset_csv,
    )

    with pytest.raises(ContractValidationError) as exc_info:
        run_operator(
            manifest_path=manifest_path,
            output_root=tmp_path / "shared-local-unseen-output",
        )

    assert exc_info.value.code == "unseen_entity_rule_violation"


def _write_shared_local_manifest(
    *,
    tmp_path: Path,
    request_id: str,
    dataset_csv: Path,
) -> Path:
    manifest_path = tmp_path / f"{request_id}.yaml"
    manifest_path.write_text(
        dedent(
            f"""
            request_id: {request_id}
            dataset_csv: {dataset_csv}
            cutoff_available_at: "2026-01-09T00:00:00Z"
            quantization_step: "0.5"
            minimum_description_gain_bits: 0.0
            min_train_size: 2
            horizon: 1
            declared_entity_panel:
              - entity-a
              - entity-b
            search:
              family_ids:
                - shared_plus_local_decomposition
              class: exact_finite_enumeration
              seed: "0"
              proposal_limit: 16
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return manifest_path
