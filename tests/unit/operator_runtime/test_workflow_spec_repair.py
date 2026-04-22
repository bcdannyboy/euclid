from __future__ import annotations

from importlib import import_module
import math
from pathlib import Path

import pytest

from euclid.artifacts import FilesystemArtifactStore
from euclid.contracts.loader import load_contract_catalog
from euclid.control_plane import SQLiteMetadataStore
from euclid.manifest_registry import ManifestRegistry
from euclid.modules.gate_lifecycle import ScorecardStatusDecision


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _workflow_module():
    return import_module("euclid.operator_runtime.workflow")


def _build_registry(tmp_path: Path) -> ManifestRegistry:
    catalog = load_contract_catalog(PROJECT_ROOT)
    return ManifestRegistry(
        catalog=catalog,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        metadata_store=SQLiteMetadataStore(tmp_path / "registry.sqlite3"),
    )


def _candidate(
    workflow,
    *,
    candidate_id: str,
    family_id: str,
    admissible: bool,
    total_bits: float,
    description_gain_bits: float,
    confirmatory_primary_score: float,
    baseline_primary_score: float,
    confirmatory_prediction_rows: tuple[dict[str, object], ...] | None = None,
    development_prediction_rows: tuple[dict[str, object], ...] = (),
) -> object:
    return workflow._CandidateEvaluation(
        family_id=family_id,
        candidate_id=candidate_id,
        parameters={"intercept": 1.0},
        exploratory_primary_score=confirmatory_primary_score,
        confirmatory_primary_score=confirmatory_primary_score,
        baseline_primary_score=baseline_primary_score,
        description_components={
            "L_family_bits": 1.0,
            "L_structure_bits": 1.0,
            "L_literals_bits": 1.0,
            "L_params_bits": 1.0,
            "L_state_bits": 1.0,
            "L_data_bits": total_bits - 6.0,
            "L_total_bits": total_bits,
            "reference_bits": total_bits + 1.0,
        },
        description_gain_bits=description_gain_bits,
        admissible=admissible,
        structure_signature=f"{family_id}:intercept=1.0",
        confirmatory_prediction_rows=confirmatory_prediction_rows
        or (
            {
                "origin_time": "2026-04-16T00:00:00Z",
                "available_at": "2026-04-16T21:00:00Z",
                "horizon": 1,
                "point_forecast": 11.2,
                "realized_observation": 11.0,
            },
        ),
        development_losses=(confirmatory_primary_score,),
        development_prediction_rows=development_prediction_rows,
    )


def _prediction_rows_from_residuals(
    residuals: list[float],
    *,
    point_forecast: float = 10.0,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "origin_time": f"2026-04-{index + 1:02d}T00:00:00Z",
            "available_at": f"2026-04-{index + 1:02d}T21:00:00Z",
            "horizon": 1,
            "point_forecast": point_forecast,
            "realized_observation": point_forecast + residual,
        }
        for index, residual in enumerate(residuals)
    )


def _structured_residual_sequence(length: int) -> list[float]:
    residual = 0.0
    values: list[float] = []
    for index in range(length):
        residual = (0.85 * residual) + math.sin(index / 3.0)
        values.append(round(residual, 6))
    return values


def _noise_like_residual_sequence(length: int) -> list[float]:
    return [
        round(
            0.45 * math.sin((index * 2.17) + 0.3)
            + 0.35 * math.cos((index * 1.71) + 0.1),
            6,
        )
        for index in range(length)
    ]


def test_workflow_keeps_best_descriptive_candidate_separate_from_publishable_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow_module()
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = _build_registry(tmp_path)

    best_descriptive_candidate = _candidate(
        workflow,
        candidate_id="operator_best_descriptive_candidate_v1",
        family_id="drift",
        admissible=False,
        total_bits=7.0,
        description_gain_bits=1.25,
        confirmatory_primary_score=0.3,
        baseline_primary_score=0.5,
    )
    accepted_candidate = _candidate(
        workflow,
        candidate_id="operator_publishable_candidate_v1",
        family_id="linear_trend",
        admissible=True,
        total_bits=9.0,
        description_gain_bits=0.75,
        confirmatory_primary_score=0.2,
        baseline_primary_score=0.5,
    )

    monkeypatch.setattr(
        workflow,
        "_evaluate_candidates",
        lambda **_: (best_descriptive_candidate, accepted_candidate),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_scorecard_status",
        lambda **_: ScorecardStatusDecision(
            descriptive_status="passed",
            descriptive_reason_codes=(),
            predictive_status="passed",
            predictive_reason_codes=(),
        ),
    )

    result = workflow.run_operator_reducer_workflow(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
    )

    search_ledger = result.search_ledger.manifest.body
    assert search_ledger["selected_candidate_id"] == best_descriptive_candidate.candidate_id
    assert (
        search_ledger["budget_accounting"]["accepted_candidate_id"]
        == accepted_candidate.candidate_id
    )

    candidate_roles = result.scorecard.manifest.body["candidate_roles"]
    assert candidate_roles["best_descriptive_candidate"]["candidate_id"] == (
        best_descriptive_candidate.candidate_id
    )
    assert candidate_roles["accepted_candidate"]["candidate_id"] == (
        accepted_candidate.candidate_id
    )
    assert candidate_roles["forward_validation_candidate"]["candidate_id"] == (
        accepted_candidate.candidate_id
    )

    predictive_law = {
        "claim_class": "predictive_law",
        "claim_card_ref": result.claim_card.manifest.ref.as_dict(),
        "scorecard_ref": result.scorecard.manifest.ref.as_dict(),
        "validation_scope_ref": result.validation_scope.manifest.ref.as_dict(),
        "publication_record_ref": result.publication_record.manifest.ref.as_dict(),
    }
    assert result.scorecard.manifest.body["predictive_law"] == predictive_law
    assert result.validation_scope.manifest.body["predictive_law"] is None
    assert result.claim_card.manifest.body["predictive_law"] is None


def test_workflow_emits_freeze_chain_predictive_blockers_and_residual_diagnostics(
    tmp_path: Path,
) -> None:
    workflow = _workflow_module()
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = _build_registry(tmp_path)

    result = workflow.run_operator_reducer_workflow(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
    )

    freeze_chain = result.validation_scope.manifest.body["freeze_chain"]
    assert freeze_chain == {
        "selected_candidate_ref": result.selected_candidate.manifest.ref.as_dict(),
        "selected_candidate_spec_ref": (
            result.selected_candidate_spec.manifest.ref.as_dict()
        ),
        "selected_candidate_structure_ref": (
            result.selected_candidate_structure.manifest.ref.as_dict()
        ),
        "frozen_shortlist_ref": result.frozen_shortlist.manifest.ref.as_dict(),
        "freeze_event_ref": result.freeze_event.manifest.ref.as_dict(),
    }
    assert result.scorecard.manifest.body["freeze_chain"] == freeze_chain

    residual_diagnostics = result.scorecard.manifest.body["residual_diagnostics"]
    assert residual_diagnostics == {
        "status": "indeterminate",
        "residual_law_search_eligible": False,
        "finite_dimensionality_status": "indeterminate",
        "recoverability_status": "indeterminate",
        "reason_codes": [
            "finite_dimensionality_insufficient_data",
            "recoverability_insufficient_data",
        ],
    }
    assert result.validation_scope.manifest.body["residual_diagnostics"] == (
        residual_diagnostics
    )
    assert result.scorecard.manifest.body["residual_law"] is None
    assert result.scorecard.manifest.body["predictive_law"] is None
    assert result.validation_scope.manifest.body["predictive_law"] is None
    assert result.claim_card is None

    run_result_body = result.run_result.manifest.body
    assert run_result_body["primary_score_result_ref"] == (
        result.point_score_result.manifest.ref.as_dict()
    )
    assert run_result_body["primary_calibration_result_ref"] == (
        result.calibration_result.manifest.ref.as_dict()
    )

    required_refs = result.reproducibility_bundle.manifest.body["required_manifest_refs"]
    assert result.prediction_artifact.manifest.ref.as_dict() in required_refs
    assert result.validation_scope.manifest.ref.as_dict() in required_refs
    assert result.point_score_result.manifest.ref.as_dict() in required_refs
    assert result.calibration_result.manifest.ref.as_dict() in required_refs


def test_workflow_residual_diagnostics_use_accepted_candidate_and_confirmatory_rows_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow_module()
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = _build_registry(tmp_path)

    best_descriptive_candidate = _candidate(
        workflow,
        candidate_id="operator_best_descriptive_candidate_v1",
        family_id="seasonal_naive",
        admissible=False,
        total_bits=7.0,
        description_gain_bits=1.5,
        confirmatory_primary_score=0.15,
        baseline_primary_score=0.5,
        confirmatory_prediction_rows=_prediction_rows_from_residuals(
            _structured_residual_sequence(40)
        ),
        development_prediction_rows=_prediction_rows_from_residuals(
            _structured_residual_sequence(40)
        ),
    )
    accepted_candidate = _candidate(
        workflow,
        candidate_id="operator_publishable_candidate_v1",
        family_id="drift",
        admissible=True,
        total_bits=9.0,
        description_gain_bits=0.75,
        confirmatory_primary_score=0.2,
        baseline_primary_score=0.5,
        confirmatory_prediction_rows=_prediction_rows_from_residuals([0.2]),
        development_prediction_rows=_prediction_rows_from_residuals(
            _structured_residual_sequence(40)
        ),
    )

    monkeypatch.setattr(
        workflow,
        "_evaluate_candidates",
        lambda **_: (best_descriptive_candidate, accepted_candidate),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_scorecard_status",
        lambda **_: ScorecardStatusDecision(
            descriptive_status="passed",
            descriptive_reason_codes=(),
            predictive_status="passed",
            predictive_reason_codes=(),
        ),
    )

    result = workflow.run_operator_reducer_workflow(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
    )

    assert result.scorecard.manifest.body["residual_diagnostics"] == {
        "status": "indeterminate",
        "residual_law_search_eligible": False,
        "finite_dimensionality_status": "indeterminate",
        "recoverability_status": "indeterminate",
        "reason_codes": [
            "finite_dimensionality_insufficient_data",
            "recoverability_insufficient_data",
        ],
    }
    assert result.scorecard.manifest.body["predictive_status"] == "passed"
    assert result.scorecard.manifest.body["predictive_law"] == {
        "claim_class": "predictive_law",
        "claim_card_ref": result.claim_card.manifest.ref.as_dict(),
        "scorecard_ref": result.scorecard.manifest.ref.as_dict(),
        "validation_scope_ref": result.validation_scope.manifest.ref.as_dict(),
        "publication_record_ref": result.publication_record.manifest.ref.as_dict(),
    }


def test_workflow_structured_confirmatory_residuals_block_predictive_law(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow_module()
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = _build_registry(tmp_path)

    accepted_candidate = _candidate(
        workflow,
        candidate_id="operator_publishable_candidate_v1",
        family_id="drift",
        admissible=True,
        total_bits=9.0,
        description_gain_bits=0.75,
        confirmatory_primary_score=0.2,
        baseline_primary_score=0.5,
        confirmatory_prediction_rows=_prediction_rows_from_residuals(
            _structured_residual_sequence(40)
        ),
    )

    monkeypatch.setattr(
        workflow,
        "_evaluate_candidates",
        lambda **_: (accepted_candidate,),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_scorecard_status",
        lambda **_: ScorecardStatusDecision(
            descriptive_status="passed",
            descriptive_reason_codes=(),
            predictive_status="passed",
            predictive_reason_codes=(),
        ),
    )

    result = workflow.run_operator_reducer_workflow(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
    )

    assert result.scorecard.manifest.body["residual_diagnostics"] == {
        "status": "structured_residual_remains",
        "residual_law_search_eligible": True,
        "finite_dimensionality_status": "supported",
        "recoverability_status": "supported",
        "reason_codes": [],
    }
    assert result.scorecard.manifest.body["predictive_status"] == "blocked"
    assert result.scorecard.manifest.body["predictive_reason_codes"] == [
        "structured_residual_remains"
    ]
    assert result.scorecard.manifest.body["predictive_law"] is None
    assert result.validation_scope.manifest.body["predictive_law"] is None
    assert result.claim_card.manifest.body["claim_type"] == "descriptive_structure"
    assert result.claim_card.manifest.body["predictive_law"] is None


def test_workflow_noise_like_confirmatory_residuals_remain_indeterminate_without_stronger_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow_module()
    catalog = load_contract_catalog(PROJECT_ROOT)
    registry = _build_registry(tmp_path)

    accepted_candidate = _candidate(
        workflow,
        candidate_id="operator_publishable_candidate_v1",
        family_id="drift",
        admissible=True,
        total_bits=9.0,
        description_gain_bits=0.75,
        confirmatory_primary_score=0.2,
        baseline_primary_score=0.5,
        confirmatory_prediction_rows=_prediction_rows_from_residuals(
            _noise_like_residual_sequence(40)
        ),
    )

    monkeypatch.setattr(
        workflow,
        "_evaluate_candidates",
        lambda **_: (accepted_candidate,),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_scorecard_status",
        lambda **_: ScorecardStatusDecision(
            descriptive_status="passed",
            descriptive_reason_codes=(),
            predictive_status="passed",
            predictive_reason_codes=(),
        ),
    )

    result = workflow.run_operator_reducer_workflow(
        csv_path=PROJECT_ROOT / "fixtures/runtime/prototype-series.csv",
        catalog=catalog,
        registry=registry,
    )

    assert result.scorecard.manifest.body["residual_diagnostics"] == {
        "status": "indeterminate",
        "residual_law_search_eligible": False,
        "finite_dimensionality_status": "indeterminate",
        "recoverability_status": "indeterminate",
        "reason_codes": [
            "finite_dimensionality_unmeasured",
            "recoverability_unmeasured",
        ],
    }
    assert result.scorecard.manifest.body["predictive_status"] == "passed"
