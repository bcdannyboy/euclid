from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from euclid.benchmarks.submitters import (
    BenchmarkSubmitterResult,
    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
)
from euclid.cir.models import (
    CIRBackendOriginRecord,
    CIRForecastOperator,
    CIRHistoryAccessContract,
    CIRInputSignature,
    CIRLiteralBlock,
    CIRModelCodeDecomposition,
    CIRReplayHooks,
)
from euclid.cir.normalize import build_cir_candidate_from_reducer
from euclid.math.observation_models import PointObservationModel
from euclid.reducers.models import (
    BoundObservationModel,
    ReducerAdmissibilityObject,
    ReducerCompositionObject,
    ReducerFamilyId,
    ReducerObject,
    ReducerParameter,
    ReducerParameterObject,
    ReducerStateObject,
    ReducerStateSemantics,
    ReducerStateUpdateContext,
    ReducerStateUpdateRule,
)
from euclid.workbench.service import (
    TARGET_SPECS,
    _build_descriptive_reconstruction,
    _build_descriptive_fit_from_submitter_results,
    _build_equation_curve,
    _render_equation_label,
    _format_portfolio_selection_explanation,
    _resolve_structure_signature,
    _select_descriptive_fit_submitter_result,
    build_equation_summary,
    create_workbench_analysis,
    default_workbench_date_range,
    normalize_analysis_payload,
    ordered_target_specs,
)


def _benchmark_submitter_result(
    *,
    submitter_id: str,
    total_code_bits: float,
    description_gain_bits: float,
    structure_code_bits: float,
    canonical_byte_length: int,
    selected: bool = True,
    selected_candidate: Any | None = None,
    selected_candidate_id: str | None = None,
) -> BenchmarkSubmitterResult:
    resolved_candidate = (
        selected_candidate
        if selected_candidate is not None
        else (object() if selected else None)
    )
    resolved_candidate_id = (
        selected_candidate_id
        if selected_candidate_id is not None
        else (
            f"{submitter_id}_candidate" if resolved_candidate is not None else None
        )
    )
    selected_candidate_metrics = (
        {
            "total_code_bits": total_code_bits,
            "description_gain_bits": description_gain_bits,
            "structure_code_bits": structure_code_bits,
            "canonical_byte_length": canonical_byte_length,
        }
        if resolved_candidate is not None
        else None
    )
    return BenchmarkSubmitterResult(
        submitter_id=submitter_id,
        submitter_class="portfolio"
        if submitter_id == PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
        else "backend",
        task_id="task",
        track_id="track",
        status="selected" if resolved_candidate is not None else "abstained",
        protocol_contract={},
        budget_consumption={},
        selected_candidate=resolved_candidate,
        selected_candidate_id=resolved_candidate_id,
        selected_candidate_metrics=selected_candidate_metrics,
        replay_contract={
            "selection_scope": "shared_planning_cir_only",
            "selection_rule": (
                "min_total_code_bits_then_max_description_gain_then_"
                "min_structure_code_bits_then_min_canonical_byte_length_then_"
                "candidate_id"
            ),
        },
    )


def _workbench_dataset_rows(values: list[float]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": "fixture",
            "series_id": "demo",
            "event_time": f"2026-01-{index + 1:02d}T00:00:00Z",
            "available_at": f"2026-01-{index + 1:02d}T00:00:00Z",
            "observed_value": value,
            "revision_id": 0,
        }
        for index, value in enumerate(values)
    ]


def _write_dataset_csv(dataset_csv: Path, rows: list[dict[str, Any]]) -> Path:
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return dataset_csv


def _constant_equation_curve(
    dataset_rows: list[dict[str, Any]],
    *,
    fitted_value: float,
) -> list[dict[str, Any]]:
    return [
        {
            "event_time": str(row["event_time"]),
            "fitted_value": fitted_value,
        }
        for row in dataset_rows
    ]


def _analytic_lag_candidate(
    *,
    candidate_id: str,
    intercept: float,
    lag_coefficient: float,
):
    reducer = ReducerObject(
        family=ReducerFamilyId("analytic"),
        composition_object=ReducerCompositionObject(),
        fitted_parameters=ReducerParameterObject(
            parameters=(
                ReducerParameter(name="intercept", value=intercept),
                ReducerParameter(
                    name="lag_coefficient",
                    value=lag_coefficient,
                ),
            )
        ),
        state_semantics=ReducerStateSemantics(
            persistent_state=ReducerStateObject(),
            update_rule=ReducerStateUpdateRule(
                update_rule_id=f"{candidate_id}_identity_update",
                implementation=_identity_update,
            ),
        ),
        observation_model=BoundObservationModel.from_runtime(PointObservationModel()),
        admissibility=ReducerAdmissibilityObject(
            family_membership=True,
            composition_closure=True,
            observation_model_compatibility=True,
            valid_state_semantics=True,
            codelength_comparability=True,
        ),
    )
    return build_cir_candidate_from_reducer(
        reducer=reducer,
        cir_form_class="closed_form_expression",
        input_signature=CIRInputSignature(
            target_series="target",
            side_information_fields=("lag_1",),
        ),
        history_access_contract=CIRHistoryAccessContract(
            contract_id=f"{candidate_id}_history",
            access_mode="full_prefix",
            allowed_side_information=("lag_1",),
        ),
        literal_block=CIRLiteralBlock(),
        forecast_operator=CIRForecastOperator(
            operator_id="one_step_point_forecast",
            horizon=1,
        ),
        model_code_decomposition=CIRModelCodeDecomposition(
            L_family_bits=2.0,
            L_structure_bits=0.0,
            L_literals_bits=0.0,
            L_params_bits=1.0,
            L_state_bits=0.0,
        ),
        backend_origin_record=CIRBackendOriginRecord(
            adapter_id="analytic-search",
            adapter_class="bounded_grammar",
            source_candidate_id=candidate_id,
            search_class="exact_finite_enumeration",
        ),
        replay_hooks=CIRReplayHooks(hooks=()),
    )


def _identity_update(
    state: ReducerStateObject,
    context: ReducerStateUpdateContext,
) -> ReducerStateObject:
    del context
    return state


def test_build_equation_curve_analytic_lag_rolls_forward_without_observed_lookup(
) -> None:
    curve = _build_equation_curve(
        candidate_id="analytic_lag1_affine",
        family_id="analytic",
        parameter_summary={"intercept": 0.0, "lag_coefficient": 1.0},
        dataset_rows=_workbench_dataset_rows([10.0, 20.0, 40.0, 80.0]),
    )

    assert [point["fitted_value"] for point in curve] == [
        10.0,
        10.0,
        10.0,
        10.0,
    ]


def test_build_equation_curve_algorithmic_last_observation_uses_rollout_state(
) -> None:
    curve = _build_equation_curve(
        candidate_id="algorithmic_last_observation",
        family_id="algorithmic",
        parameter_summary={},
        dataset_rows=_workbench_dataset_rows([3.0, 9.0, 27.0, 81.0]),
    )

    assert [point["fitted_value"] for point in curve] == [
        3.0,
        3.0,
        3.0,
        3.0,
    ]


def test_build_descriptive_fit_from_submitter_results_prefers_rollout_reconstruction_over_codelength(
) -> None:
    dataset_rows = _workbench_dataset_rows([10.0, 6.0, 4.0, 3.0])
    persistence = _benchmark_submitter_result(
        submitter_id="analytic_backend",
        total_code_bits=10.0,
        description_gain_bits=20.0,
        structure_code_bits=1.0,
        canonical_byte_length=10,
        selected_candidate=_analytic_lag_candidate(
            candidate_id="analytic_persistence",
            intercept=0.0,
            lag_coefficient=1.0,
        ),
        selected_candidate_id="analytic_persistence",
    )
    reconstructive = _benchmark_submitter_result(
        submitter_id="recursive_spectral_backend",
        total_code_bits=50.0,
        description_gain_bits=5.0,
        structure_code_bits=4.0,
        canonical_byte_length=40,
        selected_candidate=_analytic_lag_candidate(
            candidate_id="analytic_decay",
            intercept=1.0,
            lag_coefficient=0.5,
        ),
        selected_candidate_id="analytic_decay",
    )

    descriptive_fit = _build_descriptive_fit_from_submitter_results(
        submitter_results=(persistence, reconstructive),
        dataset_rows=dataset_rows,
    )

    assert descriptive_fit is not None
    assert descriptive_fit["candidate_id"] == "analytic_decay"


def test_build_descriptive_fit_from_submitter_results_returns_none_when_no_rollout_is_close(
) -> None:
    dataset_rows = _workbench_dataset_rows([10.0, 6.0, 4.0, 3.0])
    persistence = _benchmark_submitter_result(
        submitter_id="analytic_backend",
        total_code_bits=10.0,
        description_gain_bits=20.0,
        structure_code_bits=1.0,
        canonical_byte_length=10,
        selected_candidate=_analytic_lag_candidate(
            candidate_id="analytic_persistence",
            intercept=0.0,
            lag_coefficient=1.0,
        ),
        selected_candidate_id="analytic_persistence",
    )
    constant = _benchmark_submitter_result(
        submitter_id="recursive_spectral_backend",
        total_code_bits=12.0,
        description_gain_bits=18.0,
        structure_code_bits=2.0,
        canonical_byte_length=12,
        selected_candidate=_analytic_lag_candidate(
            candidate_id="analytic_constant",
            intercept=0.0,
            lag_coefficient=0.0,
        ),
        selected_candidate_id="analytic_constant",
    )

    descriptive_fit = _build_descriptive_fit_from_submitter_results(
        submitter_results=(persistence, constant),
        dataset_rows=dataset_rows,
    )

    assert descriptive_fit is None


def test_build_descriptive_fit_from_submitter_results_rejects_flat_return_rollout(
) -> None:
    dataset_rows = _workbench_dataset_rows([0.01, -0.01] * 16)
    flat_rollout = _benchmark_submitter_result(
        submitter_id="analytic_backend",
        total_code_bits=10.0,
        description_gain_bits=20.0,
        structure_code_bits=1.0,
        canonical_byte_length=10,
        selected_candidate=_analytic_lag_candidate(
            candidate_id="analytic_persistence",
            intercept=0.0,
            lag_coefficient=1.0,
        ),
        selected_candidate_id="analytic_persistence",
    )

    descriptive_fit = _build_descriptive_fit_from_submitter_results(
        submitter_results=(flat_rollout,),
        dataset_rows=dataset_rows,
    )

    assert descriptive_fit is None


def test_build_descriptive_reconstruction_is_descriptive_structure_and_non_exact() -> None:
    values = [
        (0.8 * math.cos((2.0 * math.pi * 3.0 * index) / 64.0))
        + (0.4 * math.sin((2.0 * math.pi * 7.0 * index) / 64.0))
        for index in range(64)
    ]
    dataset_rows = _workbench_dataset_rows(values)

    reconstruction = _build_descriptive_reconstruction(dataset_rows=dataset_rows)

    assert reconstruction is not None
    assert reconstruction["status"] == "completed"
    assert reconstruction["claim_class"] == "descriptive_reconstruction"
    assert reconstruction["is_law_claim"] is False
    assert reconstruction["law_eligible"] is False
    assert reconstruction["law_rejection_reason_codes"] == [
        "explicit_time_reconstruction_descriptive_structure"
    ]
    assert reconstruction["equation"]["candidate_id"] == "descriptive_fourier_reconstruction"
    assert reconstruction["equation"]["literals"]["harmonic_count"] < len(values) // 2
    assert "sum" in reconstruction["equation"]["label"]
    assert (
        reconstruction["reconstruction_metrics"]["r2_vs_mean_baseline"]
        > 0.999
    )


def test_resolve_structure_signature_falls_back_to_reducer_signature() -> None:
    reducer_body = {
        "canonical_structure_signature": "analytic:intercept=1.0,lag_coefficient=0.5"
    }

    assert (
        _resolve_structure_signature(
            structure_body=None,
            reducer_body=reducer_body,
        )
        == "analytic:intercept=1.0,lag_coefficient=0.5"
    )


def test_format_portfolio_selection_explanation_handles_mapping_payload() -> None:
    explanation = {
        "selection_rule": (
            "min_total_code_bits_then_max_description_gain_"
            "then_min_structure_code_bits"
        ),
        "winner": {
            "submitter_id": "analytic_backend",
            "candidate_id": "analytic_lag1_affine",
            "total_code_bits": 3642.0,
        },
        "runner_up": {
            "submitter_id": "algorithmic_search_backend",
            "candidate_id": "algorithmic_last_observation",
            "total_code_bits": 3666.0,
        },
    }

    summary = _format_portfolio_selection_explanation(explanation)

    assert "analytic_backend" in summary
    assert "analytic_lag1_affine" in summary
    assert "algorithmic_search_backend" in summary
    assert "3642" in summary
    assert "3666" in summary


def test_render_equation_label_tolerates_missing_analytic_parameters() -> None:
    assert (
        _render_equation_label(
            candidate_id="analytic_lag1_affine",
            family_id="analytic",
            parameter_summary={"intercept": 1.25},
        )
        is None
    )


def test_build_equation_curve_tolerates_missing_analytic_parameters() -> None:
    curve = _build_equation_curve(
        candidate_id="analytic_lag1_affine",
        family_id="analytic",
        parameter_summary={"intercept": 1.25},
        dataset_rows=[
            {"event_time": "2026-04-14T00:00:00Z", "observed_value": 700.0},
            {"event_time": "2026-04-15T00:00:00Z", "observed_value": 701.5},
        ],
    )

    assert curve == []


def test_normalize_analysis_payload_repairs_legacy_saved_workspace(
    tmp_path,
    monkeypatch,
) -> None:
    workspace_root = tmp_path / "20260416T160614Z-spy-price-close"
    dataset_dir = workspace_root / "datasets"
    dataset_dir.mkdir(parents=True)
    dataset_csv = dataset_dir / "spy-price-close.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "700.00",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "701.50",
                    "revision_id": "1",
                },
            ]
        )

    run_summary_path = (
        workspace_root
        / "runs"
        / "distribution"
        / "sealed-runs"
        / "workbench-spy-price-close-distribution"
        / "run-summary.json"
    )
    run_summary_path.parent.mkdir(parents=True)
    run_summary_path.write_text(
        (
            '{"selected_family":"analytic",'
            '"selected_candidate_id":"analytic_lag1_affine"}'
        ),
        encoding="utf-8",
    )

    reducer_manifest = SimpleNamespace(
        ref=SimpleNamespace(schema_name="reducer_artifact_manifest@1.0.0"),
        manifest=SimpleNamespace(
            body={
                "parameter_summary": {
                    "intercept": 1.25,
                    "lag_coefficient": 0.5,
                },
                "canonical_structure_signature": "analytic:lag1_affine",
            }
        ),
    )
    candidate_manifest = SimpleNamespace(
        ref=SimpleNamespace(schema_name="candidate_spec@1.0.0"),
        manifest=SimpleNamespace(
            body={
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
            }
        ),
    )

    monkeypatch.setattr(
        "euclid.workbench.service.load_demo_run_artifact_graph",
        lambda **_: SimpleNamespace(
            manifests=[reducer_manifest, candidate_manifest]
        ),
    )
    monkeypatch.setattr(
        "euclid.workbench.service.inspect_demo_probabilistic_prediction",
        lambda **_: SimpleNamespace(
            candidate_id="analytic_lag1_affine",
            aggregated_primary_score=0.42,
            rows=[
                {
                    "available_at": "2026-04-15T21:00:00Z",
                    "origin_time": "2026-04-15T00:00:00Z",
                    "location": 701.0,
                    "scale": 1.5,
                    "realized_observation": 701.5,
                }
            ],
        ),
    )
    monkeypatch.setattr(
        "euclid.workbench.service.inspect_demo_calibration",
        lambda **_: SimpleNamespace(
            status="passed",
            passed=True,
            failure_reason_code=None,
            gate_effect="publishable",
            diagnostics={"coverage": 1.0},
        ),
    )

    payload = {
        "workspace_root": str(workspace_root),
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "price_close"},
            "dataset_csv": str(dataset_csv),
        },
        "benchmark": {
            "portfolio_selection": {
                "selection_explanation": {
                    "selection_rule": "min_total_code_bits_then_max_description_gain",
                    "winner": {
                        "submitter_id": "analytic_backend",
                        "candidate_id": "analytic_lag1_affine",
                        "total_code_bits": 3642.0,
                    },
                }
            }
        },
        "probabilistic": {
            "distribution": {
                "status": "failed",
                "error": {
                    "type": "KeyError",
                    "message": (
                        "'artifact graph did not contain "
                        "canonical_structure_code_manifest@'"
                    ),
                },
            }
        },
    }

    normalized = normalize_analysis_payload(payload)

    lane = normalized["probabilistic"]["distribution"]
    assert lane["status"] == "completed"
    assert lane["selected_family"] == "analytic"
    assert lane["equation"]["structure_signature"] == "analytic:lag1_affine"
    assert lane["latest_row"]["location"] == 701.0
    assert lane["calibration"]["status"] == "passed"

    explanation = normalized["benchmark"]["portfolio_selection"][
        "selection_explanation"
    ]
    assert "analytic_backend" in explanation
    assert "3642" in explanation


def test_ordered_target_specs_promotes_daily_return_for_new_runs() -> None:
    specs = ordered_target_specs()

    assert specs[0]["id"] == "daily_return"
    assert TARGET_SPECS["daily_return"]["recommended"] is True


def test_default_workbench_date_range_is_explicit_and_stable() -> None:
    assert default_workbench_date_range(today=date(2026, 4, 16)) == {
        "start_date": "2021-04-16",
        "end_date": "2026-04-16",
    }


def test_normalize_analysis_payload_flags_persistence_like_price_fit(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "spy-price-close.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
                "close",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "500.00",
                    "revision_id": "1",
                    "close": "500.00",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "503.20",
                    "revision_id": "1",
                    "close": "503.20",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "502.60",
                    "revision_id": "1",
                    "close": "502.60",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-17T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "observed_value": "504.10",
                    "revision_id": "1",
                    "close": "504.10",
                },
            ]
        )

    payload = {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {
                "id": "price_close",
                "label": "Price Close",
                "description": "Predict the raw close level for each trading day.",
                "y_axis_label": "Close",
            },
            "dataset_csv": str(dataset_csv),
        },
        "descriptive_fit": {
            "status": "completed",
            "source": "benchmark_local_selection",
            "submitter_id": "portfolio_orchestrator",
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "metrics": {
                "total_code_bits": 8907.0,
            },
            "equation": {
                "label": "y(t) = 0.260812 + 0.999934*y(t-1)",
                "parameter_summary": {
                    "intercept": 0.260812,
                    "lag_coefficient": 0.999934,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 500.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 500.227812,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 503.427611,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 502.827571,
                    },
                ],
            },
            "chart": {
                "equation_curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 500.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 500.227812,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 503.427611,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 502.827571,
                    },
                ]
            },
        },
    }

    normalized = normalize_analysis_payload(payload)
    audit = normalized["descriptive_fit"]["semantic_audit"]

    assert audit["classification"] == "near_persistence"
    assert "persistence" in audit["headline"].lower()
    assert audit["relative_improvement_vs_naive_last_value"] < 0.05
    assert audit["delta_form_label"].startswith("Δy(t)")


def test_normalize_analysis_payload_backfills_equation_delta_forms(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "spy-price-close.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
                "close",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "500.00",
                    "revision_id": "1",
                    "close": "500.00",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "503.10",
                    "revision_id": "1",
                    "close": "503.10",
                },
            ]
        )

    payload = {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "price_close", "label": "Price Close"},
            "dataset_csv": str(dataset_csv),
        },
        "operator_point": {
            "status": "completed",
            "selected_family": "analytic",
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 0.25 + 1.01*y(t-1)",
                "parameter_summary": {
                    "intercept": 0.25,
                    "lag_coefficient": 1.01,
                },
            },
        },
        "probabilistic": {
            "distribution": {
                "status": "completed",
                "selected_family": "analytic",
                "equation": {
                    "candidate_id": "analytic_lag1_affine",
                    "family_id": "analytic",
                    "label": "y(t) = 0.25 + 1.01*y(t-1)",
                    "parameter_summary": {
                        "intercept": 0.25,
                        "lag_coefficient": 1.01,
                    },
                },
                "rows": [
                    {
                        "origin_time": "2026-04-14T00:00:00Z",
                        "available_at": "2026-04-15T21:00:00Z",
                        "horizon": 1,
                        "location": 505.0,
                        "scale": 2.0,
                        "realized_observation": 503.10,
                    }
                ],
                "calibration": {
                    "status": "passed",
                    "passed": True,
                    "gate_effect": "publishable",
                    "diagnostics": [],
                },
                "chart": {"forecast_bands": []},
            }
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert (
        normalized["operator_point"]["equation"]["delta_form_label"]
        == "Δy(t) = 0.25 + 0.01*y(t-1)"
    )
    assert (
        normalized["probabilistic"]["distribution"]["equation"]["delta_form_label"]
        == "Δy(t) = 0.25 + 0.01*y(t-1)"
    )


def test_select_descriptive_fit_submitter_result_uses_canonical_ranking_over_portfolio_winner(
) -> None:
    portfolio = _benchmark_submitter_result(
        submitter_id=PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
        total_code_bits=99.0,
        description_gain_bits=8.0,
        structure_code_bits=11.0,
        canonical_byte_length=24,
    )
    analytic = _benchmark_submitter_result(
        submitter_id="analytic_backend",
        total_code_bits=98.0,
        description_gain_bits=10.0,
        structure_code_bits=12.0,
        canonical_byte_length=20,
    )

    selected = _select_descriptive_fit_submitter_result((analytic, portfolio))

    assert selected is analytic


def test_select_descriptive_fit_submitter_result_falls_back_to_canonical_ranking(
) -> None:
    portfolio = _benchmark_submitter_result(
        submitter_id=PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
        total_code_bits=0.0,
        description_gain_bits=0.0,
        structure_code_bits=0.0,
        canonical_byte_length=0,
        selected=False,
    )
    algorithmic = _benchmark_submitter_result(
        submitter_id="algorithmic_search_backend",
        total_code_bits=100.0,
        description_gain_bits=3.0,
        structure_code_bits=5.0,
        canonical_byte_length=20,
    )
    analytic = _benchmark_submitter_result(
        submitter_id="analytic_backend",
        total_code_bits=100.0,
        description_gain_bits=9.0,
        structure_code_bits=10.0,
        canonical_byte_length=30,
    )

    selected = _select_descriptive_fit_submitter_result(
        (portfolio, algorithmic, analytic)
    )

    assert selected is analytic


def test_normalize_analysis_payload_clears_synthetic_holistic_and_reports_gaps(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "spy-daily-return.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "0.01",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "0.03",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "0.025",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-17T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "observed_value": "0.04",
                    "revision_id": "1",
                },
            ]
        )

    payload = {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return", "label": "Daily Return"},
            "dataset_csv": str(dataset_csv),
        },
        "descriptive_fit": {
            "status": "completed",
            "source": "benchmark_local_selection",
            "submitter_id": "portfolio_orchestrator",
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 0.005 + 0.8*y(t-1)",
                "parameter_summary": {
                    "intercept": 0.005,
                    "lag_coefficient": 0.8,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 0.01,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 0.013,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 0.029,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 0.025,
                    },
                ],
            },
            "chart": {
                "equation_curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 0.01,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 0.013,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 0.029,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 0.025,
                    },
                ]
            },
        },
        "operator_point": {
            "status": "completed",
            "result_mode": "abstention_only_publication",
            "selected_family": "analytic",
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 0.004 + 0.78*y(t-1)",
                "parameter_summary": {
                    "intercept": 0.004,
                    "lag_coefficient": 0.78,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 0.01,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 0.012,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 0.027,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 0.024,
                    },
                ],
            },
            "abstention": {"reason_codes": ["robustness_failed"]},
        },
        "probabilistic": {
            "distribution": {
                "status": "completed",
                "selected_family": "analytic",
                "equation": {
                    "candidate_id": "analytic_gaussian",
                    "family_id": "analytic",
                    "label": "mu(t) = 0.004 + 0.78*y(t-1)",
                },
                "rows": [
                    {
                        "origin_time": "2026-04-15T00:00:00Z",
                        "available_at": "2026-04-16T21:00:00Z",
                        "horizon": 1,
                        "location": 0.027,
                        "scale": 0.01,
                        "realized_observation": 0.025,
                    },
                    {
                        "origin_time": "2026-04-16T00:00:00Z",
                        "available_at": "2026-04-17T21:00:00Z",
                        "horizon": 1,
                        "location": 0.024,
                        "scale": 0.01,
                        "realized_observation": 0.04,
                    },
                ],
                "latest_row": {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 0.024,
                    "scale": 0.01,
                    "realized_observation": 0.04,
                },
                "calibration": {
                    "status": "passed",
                    "passed": True,
                    "gate_effect": "publishable",
                    "diagnostics": [
                        {
                            "diagnostic_id": "pit_or_randomized_pit_uniformity",
                            "sample_size": 2,
                            "status": "passed",
                        }
                    ],
                },
                "chart": {"forecast_bands": []},
            }
        },
        "holistic_equation": {
            "status": "completed",
            "claim_class": "descriptive_fit",
            "source": "descriptive_fit",
            "exactness": "sample_exact_closure",
            "equation": {"label": "y(t) = compact + exact_closure"},
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["claim_class"] == "descriptive_fit"
    assert normalized["publishable"] is False
    assert normalized["predictive_law"] is None
    assert normalized["holistic_equation"] is None
    assert normalized["would_have_abstained_because"] == ["robustness_failed"]
    assert normalized["descriptive_fit"]["claim_class"] == "descriptive_fit"
    assert normalized["descriptive_fit"]["is_law_claim"] is False
    assert normalized["descriptive_fit"]["law_eligible"] is False
    assert "operator_not_publishable" in normalized["gap_report"]
    assert "no_backend_joint_claim" in normalized["gap_report"]
    assert "probabilistic_evidence_thin" in normalized["gap_report"]
    assert "requires_exact_sample_closure" in normalized["gap_report"]
    assert normalized["not_holistic_because"] == normalized["gap_report"]


def test_build_equation_summary_preserves_composition_metadata() -> None:
    summary = build_equation_summary(
        candidate_id="additive_residual_candidate",
        family_id="analytic",
        parameter_summary={
            "intercept__trend_component": 1.0,
            "lag_coefficient__trend_component": 0.9,
            "intercept__seasonal_component": -0.25,
            "lag_coefficient__seasonal_component": 0.1,
        },
        structure_signature="analytic:additive_residual",
        dataset_rows=[
            {"event_time": "2026-04-14T00:00:00Z", "observed_value": 10.0},
            {"event_time": "2026-04-15T00:00:00Z", "observed_value": 11.0},
        ],
        composition_operator="additive_residual",
        composition_payload={
            "operator_id": "additive_residual",
            "base_reducer": "trend_component",
            "residual_reducer": "seasonal_component",
            "shared_observation_model": "point_identity",
        },
        literals={"period__seasonal_component": 5},
        state={"running_mean__seasonal_component": 0.0},
    )

    assert summary["composition_operator"] == "additive_residual"
    assert summary["composition_payload"]["base_reducer"] == "trend_component"
    assert summary["literals"]["period__seasonal_component"] == 5
    assert summary["state"]["running_mean__seasonal_component"] == 0.0


def _predictive_law_candidate_publication_payload(
    tmp_path: Path,
    *,
    operator_equation: dict[str, Any],
) -> dict[str, Any]:
    dataset_csv = tmp_path / "spy-price-close-predictive-scope.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "10.0",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "11.2",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "10.7",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-17T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "observed_value": "11.4",
                    "revision_id": "1",
                },
            ]
        )

    return {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "price_close", "label": "Price Close"},
            "dataset_csv": str(dataset_csv),
        },
        "descriptive_fit": {
            "status": "completed",
            "source": "benchmark_local_selection",
            "submitter_id": "portfolio_orchestrator",
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 1.0 + 0.9*y(t-1)",
                "parameter_summary": {
                    "intercept": 1.0,
                    "lag_coefficient": 0.9,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 10.8,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 11.3,
                    },
                ],
            },
            "chart": {
                "equation_curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 10.8,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 11.3,
                    },
                ]
            },
        },
        "operator_point": {
            "status": "completed",
            "result_mode": "candidate_publication",
            "selected_family": "analytic",
            "claim_card_ref": "claim_card_manifest@1.1.0:claim-card-1",
            "scorecard_ref": "scorecard_manifest@1.1.0:scorecard-1",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "equation": operator_equation,
            "claim_card": {
                "claim_type": "predictive_within_declared_scope",
                "claim_ceiling": "predictive_within_declared_scope",
                "predictive_support_status": "confirmatory_supported",
                "allowed_interpretation_codes": [
                    "historical_structure_summary",
                    "point_forecast_within_declared_validation_scope",
                ],
            },
            "scorecard": {
                "descriptive_status": "passed",
                "predictive_status": "passed",
            },
        },
    }


@pytest.mark.parametrize(
    ("claim_type", "allowed_interpretation_codes"),
    [
        (
            "predictive_within_declared_scope",
            [
                "historical_structure_summary",
                "point_forecast_within_declared_validation_scope",
            ],
        ),
        (
            "mechanistically_compatible_law",
            [
                "historical_structure_summary",
                "point_forecast_within_declared_validation_scope",
                "mechanism_claim",
            ],
        ),
    ],
)
def test_normalize_analysis_payload_projects_predictive_law_without_holistic_fallback(
    tmp_path: Path,
    claim_type: str,
    allowed_interpretation_codes: list[str],
) -> None:
    dataset_csv = tmp_path / "spy-price-close.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "10.0",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "11.2",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "10.7",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-17T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "observed_value": "11.4",
                    "revision_id": "1",
                },
            ]
        )

    payload = {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "price_close", "label": "Price Close"},
            "dataset_csv": str(dataset_csv),
        },
        "descriptive_fit": {
            "status": "completed",
            "source": "benchmark_local_selection",
            "submitter_id": "portfolio_orchestrator",
            "candidate_id": "additive_residual_candidate",
            "family_id": "analytic",
            "equation": {
                "candidate_id": "additive_residual_candidate",
                "family_id": "analytic",
                "parameter_summary": {
                    "intercept__trend_component": 1.0,
                    "lag_coefficient__trend_component": 0.9,
                    "intercept__seasonal_component": -0.25,
                    "lag_coefficient__seasonal_component": 0.1,
                },
                "composition_operator": "additive_residual",
                "composition_payload": {
                    "operator_id": "additive_residual",
                    "base_reducer": "trend_component",
                    "residual_reducer": "seasonal_component",
                    "shared_observation_model": "point_identity",
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 10.8,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 11.3,
                    },
                ],
            },
            "chart": {
                "equation_curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 10.8,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 11.3,
                    },
                ]
            },
        },
        "operator_point": {
            "status": "completed",
            "result_mode": "candidate_publication",
            "selected_family": "analytic",
            "claim_card_ref": "claim_card_manifest@1.1.0:claim-card-1",
            "scorecard_ref": "scorecard_manifest@1.1.0:scorecard-1",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 1.8 + 0.92*y(t-1)",
                "parameter_summary": {
                    "intercept": 1.8,
                    "lag_coefficient": 0.92,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 10.8,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 11.3,
                    },
                ],
            },
            "claim_card": {
                "claim_type": claim_type,
                "claim_ceiling": claim_type,
                "predictive_support_status": "confirmatory_supported",
                "allowed_interpretation_codes": allowed_interpretation_codes,
            },
            "scorecard": {
                "descriptive_status": "passed",
                "predictive_status": "passed",
            },
        },
        "probabilistic": {
            "distribution": {
                "status": "completed",
                "selected_family": "analytic",
                "equation": {
                    "candidate_id": "analytic_gaussian",
                    "family_id": "analytic",
                    "label": "mu(t) = 1.0 + 0.9*y(t-1)",
                },
                "rows": [
                    {
                        "origin_time": "2026-04-16T00:00:00Z",
                        "available_at": "2026-04-17T21:00:00Z",
                        "horizon": 1,
                        "location": 11.3,
                        "scale": 0.4,
                        "realized_observation": 11.4,
                    }
                ],
                "latest_row": {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 11.3,
                    "scale": 0.4,
                    "realized_observation": 11.4,
                },
                "calibration": {
                    "status": "passed",
                    "passed": True,
                    "gate_effect": "publishable",
                    "diagnostics": [
                        {
                            "diagnostic_id": "pit_or_randomized_pit_uniformity",
                            "sample_size": 1,
                            "status": "passed",
                        }
                    ],
                },
                "chart": {"forecast_bands": []},
            }
        },
    }

    normalized = normalize_analysis_payload(payload)

    predictive_law = normalized["predictive_law"]

    assert normalized["claim_class"] == "predictive_law"
    assert normalized["publishable"] is True
    assert predictive_law is not None
    assert predictive_law["claim_class"] == "predictive_law"
    assert predictive_law["claim_card_ref"] == "claim_card_manifest@1.1.0:claim-card-1"
    assert predictive_law["scorecard_ref"] == "scorecard_manifest@1.1.0:scorecard-1"
    assert (
        predictive_law["validation_scope_ref"]
        == "validation_scope_manifest@1.0.0:scope-1"
    )
    assert (
        predictive_law["publication_record_ref"]
        == "publication_record_manifest@1.1.0:publication-1"
    )
    assert predictive_law["evidence_summary"] == {
        "claim_card": {
            "ref": "claim_card_manifest@1.1.0:claim-card-1",
            "claim_type": claim_type,
            "claim_ceiling": claim_type,
            "predictive_support_status": "confirmatory_supported",
            "allowed_interpretation_codes": allowed_interpretation_codes,
        },
        "scorecard": {
            "ref": "scorecard_manifest@1.1.0:scorecard-1",
            "descriptive_status": "passed",
            "predictive_status": "passed",
        },
        "validation_scope": {
            "ref": "validation_scope_manifest@1.0.0:scope-1",
            "headline": (
                "Predictive-within-scope interpretation is bounded by the declared "
                "validation scope reference."
            ),
        },
        "publication_record": {
            "ref": "publication_record_manifest@1.1.0:publication-1",
            "status": "publishable",
            "headline": (
                "Operator replay published a point-lane candidate within the "
                "declared validation scope."
            ),
            "reason_codes": [],
        },
    }
    assert predictive_law["equation"]["label"] == "y(t) = 1.8 + 0.92*y(t-1)"
    assert predictive_law["equation"]["curve"] == [
        {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
        {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
        {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
        {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
    ]
    assert normalized["holistic_equation"] is None
    assert "no_backend_joint_claim" in normalized["gap_report"]
    assert "probabilistic_evidence_thin" in normalized["gap_report"]
    assert normalized["would_have_abstained_because"] == []
    assert normalized["not_holistic_because"] == normalized["gap_report"]


def test_normalize_analysis_payload_rejects_missing_claim_ceiling_for_predictive_law(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    payload["operator_point"]["claim_card"].pop("claim_ceiling", None)

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is None
    assert normalized["claim_class"] == "descriptive_fit"
    assert normalized["publishable"] is False


def test_normalize_analysis_payload_rejects_delta_only_operator_point_for_predictive_law(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "parameter_summary": {
                "intercept": 1.8,
                "lag_coefficient": 0.92,
            },
        },
    )

    normalized = normalize_analysis_payload(payload)

    assert normalized["operator_point"]["publication"]["status"] == "publishable"
    assert normalized["operator_point"]["equation"]["delta_form_label"] == (
        "Δy(t) = 1.8 - 0.08*y(t-1)"
    )
    assert normalized["predictive_law"] is None
    assert normalized["claim_class"] == "descriptive_fit"
    assert normalized["publishable"] is False


def test_normalize_analysis_payload_projects_predictive_law_with_curve_only_equation(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )

    normalized = normalize_analysis_payload(payload)

    predictive_law = normalized["predictive_law"]

    assert predictive_law is not None
    assert predictive_law["claim_class"] == "predictive_law"
    assert predictive_law["equation"]["candidate_id"] == "analytic_lag1_affine"
    assert predictive_law["equation"].get("label") is None
    assert predictive_law["equation"].get("delta_form_label") is None
    assert predictive_law["equation"]["curve"] == [
        {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
        {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
        {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
        {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
    ]
    assert predictive_law["publishable"] is True
    assert normalized["claim_class"] == "predictive_law"
    assert normalized["publishable"] is True


def test_normalize_analysis_payload_rejects_stale_saved_predictive_law_without_operator_gates(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "spy-price-close-stale-predictive.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "10.0",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "11.2",
                    "revision_id": "1",
                },
            ]
        )

    payload = {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "price_close", "label": "Price Close"},
            "dataset_csv": str(dataset_csv),
        },
        "descriptive_fit": {
            "status": "completed",
            "source": "benchmark_local_selection",
            "submitter_id": "portfolio_orchestrator",
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 1.0 + 0.9*y(t-1)",
                "parameter_summary": {
                    "intercept": 1.0,
                    "lag_coefficient": 0.9,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                ],
            },
            "chart": {
                "equation_curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                ]
            },
        },
        "predictive_law": {
            "status": "completed",
            "claim_class": "predictive_law",
            "claim_card_ref": "claim_card_manifest@1.1.0:claim-card-stale",
            "scorecard_ref": "scorecard_manifest@1.1.0:scorecard-stale",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-stale",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-stale"
            ),
            "equation": {
                "label": "y(t) = 999 + 0*y(t-1)",
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 999.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 999.0,
                    },
                ],
            },
        },
        "operator_point": {
            "status": "completed",
            "result_mode": "candidate_only",
            "selected_family": "analytic",
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 1.1 + 0.8*y(t-1)",
                "parameter_summary": {
                    "intercept": 1.1,
                    "lag_coefficient": 0.8,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 10.9,
                    },
                ],
            },
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is None
    assert normalized["claim_class"] == "descriptive_fit"
    assert normalized["publishable"] is False


def test_normalize_analysis_payload_rejects_descriptive_structure_claim_card_for_predictive_law(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "spy-price-close-descriptive-only.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "10.0",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "11.2",
                    "revision_id": "1",
                },
            ]
        )

    payload = {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "price_close", "label": "Price Close"},
            "dataset_csv": str(dataset_csv),
        },
        "descriptive_fit": {
            "status": "completed",
            "source": "benchmark_local_selection",
            "submitter_id": "portfolio_orchestrator",
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 1.0 + 0.9*y(t-1)",
                "parameter_summary": {
                    "intercept": 1.0,
                    "lag_coefficient": 0.9,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                ],
            },
            "chart": {
                "equation_curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                ]
            },
        },
        "operator_point": {
            "status": "completed",
            "result_mode": "candidate_publication",
            "selected_family": "analytic",
            "claim_card_ref": "claim_card_manifest@1.1.0:claim-card-1",
            "scorecard_ref": "scorecard_manifest@1.1.0:scorecard-1",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 1.8 + 0.92*y(t-1)",
                "parameter_summary": {
                    "intercept": 1.8,
                    "lag_coefficient": 0.92,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                ],
            },
            "claim_card": {
                "claim_type": "descriptive_structure",
                "claim_ceiling": "descriptive_structure",
                "predictive_support_status": "confirmatory_supported",
                "allowed_interpretation_codes": [
                    "historical_structure_summary",
                    "point_forecast_within_declared_validation_scope",
                ],
            },
            "abstention": {
                "reason_codes": ["robustness_failed"],
                "blocked_ceiling": "descriptive_structure",
            },
            "scorecard": {
                "descriptive_status": "passed",
                "predictive_status": "passed",
            },
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is None
    assert normalized["claim_class"] == "descriptive_fit"
    assert normalized["publishable"] is False
    assert "descriptive_structure" in normalized["would_have_abstained_because"]
    assert "descriptive_structure" in normalized["not_holistic_because"]


def test_normalize_analysis_payload_rejects_exact_closure_operator_point_for_predictive_law(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_exact_closure_surface",
            "family_id": "analytic",
            "exactness": "sample_exact_closure",
            "label": "y(t) = exact_closure(sample)",
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.2},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.7},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.4},
            ],
        },
    )

    normalized = normalize_analysis_payload(payload)

    assert normalized["operator_point"]["publication"]["status"] == "publishable"
    assert normalized["predictive_law"] is None
    assert normalized["claim_class"] == "descriptive_fit"
    assert normalized["publishable"] is False


@pytest.mark.parametrize(
    ("operator_equation", "case_id"),
    [
        (
            {
                "candidate_id": "analytic_additive_residual_surface",
                "family_id": "analytic",
                "label": "y(t) = trend + residual_lookup",
                "composition_operator": "additive_residual",
                "composition_payload": {
                    "operator_id": "additive_residual",
                    "base_reducer": "trend_component",
                    "residual_reducer": "seasonal_component",
                    "shared_observation_model": "point_identity",
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 10.8,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 11.3,
                    },
                ],
            },
            "lookup_residual_wrapper",
        ),
        (
            {
                "candidate_id": "analytic_symbolic_synthesis_surface",
                "family_id": "analytic",
                "label": "y(t) = symbolic_synthesis(base, residual)",
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 10.8,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 11.3,
                    },
                ],
            },
            "posthoc_symbolic_synthesis",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_normalize_analysis_payload_rejects_synthetic_operator_point_for_predictive_law(
    tmp_path: Path,
    operator_equation: dict[str, Any],
    case_id: str,
) -> None:
    del case_id
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation=operator_equation,
    )

    normalized = normalize_analysis_payload(payload)

    assert normalized["operator_point"]["publication"]["status"] == "publishable"
    assert normalized["predictive_law"] is None
    assert normalized["claim_class"] == "descriptive_fit"
    assert normalized["publishable"] is False


def test_normalize_analysis_payload_rejects_non_renderable_operator_point_for_predictive_law(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
        },
    )

    normalized = normalize_analysis_payload(payload)

    assert normalized["operator_point"]["publication"]["status"] == "publishable"
    assert normalized["operator_point"]["equation"] == {
        "candidate_id": "analytic_lag1_affine",
        "family_id": "analytic",
    }
    assert normalized["predictive_law"] is None
    assert normalized["claim_class"] == "descriptive_fit"
    assert normalized["publishable"] is False


def test_normalize_analysis_payload_does_not_infer_holistic_from_saved_alignment(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "spy-price-close-shape-valid.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "10.0",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "11.2",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "10.7",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-17T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "observed_value": "11.4",
                    "revision_id": "1",
                },
            ]
        )

    payload = {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "price_close", "label": "Price Close"},
            "dataset_csv": str(dataset_csv),
        },
        "descriptive_fit": {
            "status": "completed",
            "source": "benchmark_local_selection",
            "submitter_id": "portfolio_orchestrator",
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 1.0 + 0.9*y(t-1)",
                "parameter_summary": {
                    "intercept": 1.0,
                    "lag_coefficient": 0.9,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 10.8,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 11.3,
                    },
                ],
            },
            "chart": {
                "equation_curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 10.8,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 11.3,
                    },
                ]
            },
        },
        "operator_point": {
            "status": "completed",
            "result_mode": "candidate_publication",
            "selected_family": "analytic",
            "claim_card_ref": "claim_card_manifest@1.1.0:claim-card-2",
            "scorecard_ref": "scorecard_manifest@1.1.0:scorecard-2",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-2",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-2"
            ),
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 1.8 + 0.92*y(t-1)",
                "parameter_summary": {
                    "intercept": 1.8,
                    "lag_coefficient": 0.92,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 10.8,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 11.3,
                    },
                ],
            },
            "claim_card": {
                "claim_type": "predictive_within_declared_scope",
                "claim_ceiling": "predictive_within_declared_scope",
                "predictive_support_status": "confirmatory_supported",
                "allowed_interpretation_codes": [
                    "historical_structure_summary",
                    "point_forecast_within_declared_validation_scope",
                ],
            },
            "scorecard": {
                "descriptive_status": "passed",
                "predictive_status": "passed",
            },
        },
        "probabilistic": {
            "distribution": {
                "status": "completed",
                "selected_family": "analytic",
                "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-2",
                "publication_record_ref": (
                    "publication_record_manifest@1.1.0:publication-2"
                ),
                "rows": [
                    {
                        "origin_time": "2026-04-16T00:00:00Z",
                        "available_at": "2026-04-17T21:00:00Z",
                        "horizon": 1,
                        "location": 11.3,
                        "scale": 0.4,
                        "realized_observation": 11.4,
                    }
                ],
                "latest_row": {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 11.3,
                    "scale": 0.4,
                    "realized_observation": 11.4,
                },
                "calibration": {
                    "status": "passed",
                    "passed": True,
                    "gate_effect": "publishable",
                    "diagnostics": [
                        {
                            "diagnostic_id": "pit_or_randomized_pit_uniformity",
                            "sample_size": 1,
                            "status": "passed",
                        }
                    ],
                },
                "chart": {"forecast_bands": []},
            }
        },
        "holistic_equation": {
            "status": "completed",
            "claim_class": "holistic_equation",
            "deterministic_source": "predictive_law",
            "probabilistic_source": "distribution",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-2",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-2"
            ),
            "honesty_note": "Shape-valid stale payload",
            "equation": {
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 10.8,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 11.3,
                    },
                ],
            },
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is not None
    assert normalized["holistic_equation"] is None
    assert normalized["uncertainty_attachment"] is None
    assert normalized["claim_class"] == "predictive_law"
    assert normalized["publishable"] is True
    assert "no_backend_joint_claim" in normalized["gap_report"]
    assert normalized["not_holistic_because"] == normalized["gap_report"]


@pytest.mark.parametrize("deterministic_source", ["descriptive_fit", "operator_point"])
def test_normalize_analysis_payload_rejects_non_predictive_deterministic_source_for_holistic_claim(
    tmp_path: Path,
    deterministic_source: str,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "label": "y(t) = 1.8 + 0.92*y(t-1)",
            "parameter_summary": {
                "intercept": 1.8,
                "lag_coefficient": 0.92,
            },
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    payload["probabilistic"] = {
        "distribution": {
            "status": "completed",
            "selected_family": "analytic",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "rows": [
                {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 11.3,
                    "scale": 0.4,
                    "realized_observation": 11.4,
                }
            ],
            "latest_row": {
                "origin_time": "2026-04-16T00:00:00Z",
                "available_at": "2026-04-17T21:00:00Z",
                "horizon": 1,
                "location": 11.3,
                "scale": 0.4,
                "realized_observation": 11.4,
            },
            "calibration": {
                "status": "passed",
                "passed": True,
                "gate_effect": "publishable",
                "diagnostics": [
                    {
                        "diagnostic_id": "pit_or_randomized_pit_uniformity",
                        "sample_size": 1,
                        "status": "passed",
                    }
                ],
            },
            "chart": {"forecast_bands": []},
        }
    }
    payload["holistic_equation"] = {
        "status": "completed",
        "claim_class": "holistic_equation",
        "deterministic_source": deterministic_source,
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
        "honesty_note": (
            "Saved joint payload should not survive unless the deterministic "
            "source is the backend-backed predictive-within-scope claim and an explicit "
            "backend joint-claim gate is present."
        ),
        "equation": {
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is not None
    assert normalized["holistic_equation"] is None
    assert normalized["uncertainty_attachment"] is None
    assert normalized["claim_class"] == "predictive_law"
    assert normalized["not_holistic_because"] == normalized["gap_report"]


@pytest.mark.parametrize(
    "equation_payload",
    [
        {
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
            "exactness": "sample_exact_closure",
        },
        {
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
            "composition_operator": "additive_residual",
        },
        {
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
            "candidate_id": "symbolic_synthesis_candidate",
        },
    ],
)
def test_normalize_analysis_payload_rejects_holistic_claim_with_banned_equation_markers(
    tmp_path: Path,
    equation_payload: dict[str, Any],
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "label": "y(t) = 1.8 + 0.92*y(t-1)",
            "parameter_summary": {
                "intercept": 1.8,
                "lag_coefficient": 0.92,
            },
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    payload["probabilistic"] = {
        "distribution": {
            "status": "completed",
            "selected_family": "analytic",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "rows": [
                {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 11.3,
                    "scale": 0.4,
                    "realized_observation": 11.4,
                }
            ],
            "latest_row": {
                "origin_time": "2026-04-16T00:00:00Z",
                "available_at": "2026-04-17T21:00:00Z",
                "horizon": 1,
                "location": 11.3,
                "scale": 0.4,
                "realized_observation": 11.4,
            },
            "calibration": {
                "status": "passed",
                "passed": True,
                "gate_effect": "publishable",
                "diagnostics": [
                    {
                        "diagnostic_id": "pit_or_randomized_pit_uniformity",
                        "sample_size": 1,
                        "status": "passed",
                    }
                ],
            },
            "chart": {"forecast_bands": []},
        }
    }
    payload["holistic_equation"] = {
        "status": "completed",
        "claim_class": "holistic_equation",
        "joint_claim_gate": {
            "backend_authored": True,
            "status": "accepted",
        },
        "deterministic_source": "predictive_law",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
        "honesty_note": "Equation-level banned markers must veto stale holistic claims.",
        "equation": equation_payload,
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is not None
    assert normalized["holistic_equation"] is None
    assert normalized["uncertainty_attachment"] is None
    assert normalized["claim_class"] == "predictive_law"


@pytest.mark.parametrize("joint_claim_gate", [True, "publishable", "backend_authored"])
def test_normalize_analysis_payload_rejects_unstructured_joint_claim_gate(
    tmp_path: Path,
    joint_claim_gate: Any,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "label": "y(t) = 1.8 + 0.92*y(t-1)",
            "parameter_summary": {
                "intercept": 1.8,
                "lag_coefficient": 0.92,
            },
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    payload["probabilistic"] = {
        "distribution": {
            "status": "completed",
            "selected_family": "analytic",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "rows": [
                {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 11.3,
                    "scale": 0.4,
                    "realized_observation": 11.4,
                }
            ],
            "latest_row": {
                "origin_time": "2026-04-16T00:00:00Z",
                "available_at": "2026-04-17T21:00:00Z",
                "horizon": 1,
                "location": 11.3,
                "scale": 0.4,
                "realized_observation": 11.4,
            },
            "calibration": {
                "status": "passed",
                "passed": True,
                "gate_effect": "publishable",
                "diagnostics": [
                    {
                        "diagnostic_id": "pit_or_randomized_pit_uniformity",
                        "sample_size": 1,
                        "status": "passed",
                    }
                ],
            },
            "chart": {"forecast_bands": []},
        }
    }
    payload["holistic_equation"] = {
        "status": "completed",
        "claim_class": "holistic_equation",
        "joint_claim_gate": joint_claim_gate,
        "deterministic_source": "predictive_law",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
        "honesty_note": "Only structured backend-authored joint gates should count.",
        "equation": {
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is not None
    assert normalized["holistic_equation"] is None
    assert normalized["uncertainty_attachment"] is None


@pytest.mark.parametrize("joint_gate_status", [None, "completed", "published", "publishable"])
def test_normalize_analysis_payload_requires_accepted_joint_claim_gate_status(
    tmp_path: Path,
    joint_gate_status: str | None,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "label": "y(t) = 1.8 + 0.92*y(t-1)",
            "parameter_summary": {
                "intercept": 1.8,
                "lag_coefficient": 0.92,
            },
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    payload["probabilistic"] = {
        "distribution": {
            "status": "completed",
            "selected_family": "analytic",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "rows": [
                {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 11.3,
                    "scale": 0.4,
                    "realized_observation": 11.4,
                }
            ],
            "latest_row": {
                "origin_time": "2026-04-16T00:00:00Z",
                "available_at": "2026-04-17T21:00:00Z",
                "horizon": 1,
                "location": 11.3,
                "scale": 0.4,
                "realized_observation": 11.4,
            },
            "calibration": {
                "status": "passed",
                "passed": True,
                "gate_effect": "publishable",
                "diagnostics": [
                    {
                        "diagnostic_id": "pit_or_randomized_pit_uniformity",
                        "sample_size": 1,
                        "status": "passed",
                    }
                ],
            },
            "chart": {"forecast_bands": []},
        }
    }
    payload["holistic_equation"] = {
        "status": "completed",
        "claim_class": "holistic_equation",
        "joint_claim_gate": {
            "backend_authored": True,
            **({} if joint_gate_status is None else {"status": joint_gate_status}),
        },
        "deterministic_source": "predictive_law",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
        "honesty_note": "Only accepted backend joint gates may pass.",
        "equation": {
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is not None
    assert normalized["holistic_equation"] is None


def test_normalize_analysis_payload_projects_residual_law_from_residual_diagnostics(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    payload["residual_diagnostics"] = {
        "status": "structured_residual_remains",
        "residual_law_search_eligible": True,
        "finite_dimensionality_status": "supported",
        "recoverability_status": "supported",
        "reason_codes": [],
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["residual_law"] is None


def test_normalize_analysis_payload_projects_workflow_native_nested_raw_surfaces(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    nested_freeze_chain = {
        "selected_candidate_ref": {
            "schema_name": "candidate_manifest",
            "object_id": "candidate-1",
        },
        "selected_candidate_spec_ref": {
            "schema_name": "candidate_spec_manifest",
            "object_id": "candidate-spec-1",
        },
        "selected_candidate_structure_ref": {
            "schema_name": "canonical_structure_code_manifest",
            "object_id": "structure-1",
        },
        "frozen_shortlist_ref": {
            "schema_name": "frozen_shortlist_manifest",
            "object_id": "shortlist-1",
        },
        "freeze_event_ref": {
            "schema_name": "freeze_event_manifest",
            "object_id": "freeze-1",
        },
    }
    nested_residual_diagnostics = {
        "status": "structured_residual_remains",
        "residual_law_search_eligible": True,
        "finite_dimensionality_status": "supported",
        "recoverability_status": "supported",
        "reason_codes": ["requires_lookup_residual_wrapper"],
    }
    payload["operator_point"]["scorecard"]["freeze_chain"] = nested_freeze_chain
    payload["operator_point"]["scorecard"]["residual_diagnostics"] = (
        nested_residual_diagnostics
    )
    payload["operator_point"]["scorecard"]["residual_law"] = None
    payload["operator_point"]["claim_card"]["freeze_chain"] = nested_freeze_chain
    payload["operator_point"]["claim_card"]["residual_diagnostics"] = (
        nested_residual_diagnostics
    )
    payload["operator_point"]["claim_card"]["predictive_law"] = {
        "status": "publishable",
        "claim_class": "predictive_law",
        "claim_type": "predictive_within_declared_scope",
        "claim_ceiling": "predictive_within_declared_scope",
        "predictive_support_status": "confirmatory_supported",
        "allowed_interpretation_codes": [
            "point_forecast_within_declared_validation_scope"
        ],
        "forbidden_interpretation_codes": [],
        "reason_codes": [],
        "candidate_roles": {"selected_role": "winner"},
        "freeze_chain": nested_freeze_chain,
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is not None
    assert normalized["freeze_chain"] == nested_freeze_chain
    assert normalized["residual_diagnostics"] == nested_residual_diagnostics
    assert normalized["residual_law"] is None


def test_normalize_analysis_payload_projects_uncertainty_attachment_without_upgrading_descriptive_claim(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    payload["operator_point"]["result_mode"] = "candidate_only"
    payload["probabilistic"] = {
        "distribution": {
            "status": "completed",
            "selected_family": "analytic",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
            "rows": [
                {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 11.3,
                    "scale": 0.4,
                    "realized_observation": 11.4,
                }
            ],
            "latest_row": {
                "origin_time": "2026-04-16T00:00:00Z",
                "available_at": "2026-04-17T21:00:00Z",
                "horizon": 1,
                "location": 11.3,
                "scale": 0.4,
                "realized_observation": 11.4,
            },
            "calibration": {
                "status": "passed",
                "passed": True,
                "gate_effect": "publishable",
                "diagnostics": [
                    {
                        "diagnostic_id": "pit_or_randomized_pit_uniformity",
                        "sample_size": 1,
                        "status": "passed",
                    }
                ],
            },
            "chart": {"forecast_bands": []},
        }
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is None
    assert normalized["holistic_equation"] is None
    assert normalized["uncertainty_attachment"] is None
    assert normalized["claim_class"] == "descriptive_fit"
    assert normalized["publishable"] is False


def test_normalize_analysis_payload_rejects_stale_top_level_uncertainty_attachment(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    payload["probabilistic"] = {
        "distribution": {
            "status": "completed",
            "selected_family": "analytic",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "rows": [
                {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 11.3,
                    "scale": 0.4,
                    "realized_observation": 11.4,
                }
            ],
            "latest_row": {
                "origin_time": "2026-04-16T00:00:00Z",
                "available_at": "2026-04-17T21:00:00Z",
                "horizon": 1,
                "location": 11.3,
                "scale": 0.4,
                "realized_observation": 11.4,
            },
            "calibration": {
                "status": "passed",
                "passed": True,
                "gate_effect": "publishable",
                "diagnostics": [
                    {
                        "diagnostic_id": "pit_or_randomized_pit_uniformity",
                        "sample_size": 1,
                        "status": "passed",
                    }
                ],
            },
            "chart": {"forecast_bands": []},
        }
    }
    payload["uncertainty_attachment"] = {
        "status": "completed",
        "deterministic_source": "predictive_law",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is not None
    assert normalized["uncertainty_attachment"] is None


def test_normalize_analysis_payload_projects_workflow_native_uncertainty_attachment(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    payload["probabilistic"] = {
        "distribution": {
            "status": "completed",
            "selected_family": "analytic",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "rows": [
                {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 11.3,
                    "scale": 0.4,
                    "realized_observation": 11.4,
                }
            ],
            "latest_row": {
                "origin_time": "2026-04-16T00:00:00Z",
                "available_at": "2026-04-17T21:00:00Z",
                "horizon": 1,
                "location": 11.3,
                "scale": 0.4,
                "realized_observation": 11.4,
            },
            "calibration": {
                "status": "passed",
                "passed": True,
                "gate_effect": "publishable",
                "diagnostics": [
                    {
                        "diagnostic_id": "pit_or_randomized_pit_uniformity",
                        "sample_size": 1,
                        "status": "passed",
                    }
                ],
            },
            "chart": {"forecast_bands": []},
        }
    }
    payload["operator_point"]["claim_card"]["uncertainty_attachment"] = {
        "status": "completed",
        "joint_claim_gate": {
            "backend_authored": True,
            "status": "accepted",
        },
        "deterministic_source": "predictive_law",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is not None
    assert normalized["uncertainty_attachment"] == {
        "status": "completed",
        "deterministic_source": "predictive_law",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
    }


def test_normalize_analysis_payload_preserves_search_side_gap_reasons_and_not_holistic_blockers(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    payload["operator_point"]["result_mode"] = "candidate_only"
    payload["descriptive_fit"]["law_eligible"] = False
    payload["descriptive_fit"]["law_rejection_reason_codes"] = [
        "requires_lookup_residual_wrapper",
        "requires_posthoc_symbolic_synthesis",
    ]

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is None
    assert normalized["gap_report"] == [
        "requires_lookup_residual_wrapper",
        "requires_posthoc_symbolic_synthesis",
        "operator_not_publishable",
        "no_backend_joint_claim",
    ]
    assert normalized["not_holistic_because"] == normalized["gap_report"]


def test_normalize_analysis_payload_rejects_non_publishable_probabilistic_lane_for_holistic_claim(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "label": "y(t) = 1.8 + 0.92*y(t-1)",
            "parameter_summary": {
                "intercept": 1.8,
                "lag_coefficient": 0.92,
            },
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    payload["probabilistic"] = {
        "distribution": {
            "status": "completed",
            "selected_family": "analytic",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
            "rows": [
                {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 11.3,
                    "scale": 0.4,
                    "realized_observation": 11.4,
                }
            ],
            "latest_row": {
                "origin_time": "2026-04-16T00:00:00Z",
                "available_at": "2026-04-17T21:00:00Z",
                "horizon": 1,
                "location": 11.3,
                "scale": 0.4,
                "realized_observation": 11.4,
            },
            "calibration": {
                "status": "passed",
                "passed": True,
                "gate_effect": "candidate_only",
                "diagnostics": [
                    {
                        "diagnostic_id": "pit_or_randomized_pit_uniformity",
                        "sample_size": 1,
                        "status": "passed",
                    }
                ],
            },
            "chart": {"forecast_bands": []},
        }
    }
    payload["holistic_equation"] = {
        "status": "completed",
        "claim_class": "holistic_equation",
        "deterministic_source": "predictive_law",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
        "honesty_note": "Completed probabilistic lane is not enough without publishable calibration.",
        "equation": {
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is not None
    assert normalized["holistic_equation"] is None
    assert normalized["claim_class"] == "predictive_law"
    assert normalized["publishable"] is True
    assert "no_backend_joint_claim" in normalized["gap_report"]


def test_normalize_analysis_payload_rejects_label_only_backend_holistic_claim(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "label": "y(t) = 1.8 + 0.92*y(t-1)",
            "parameter_summary": {
                "intercept": 1.8,
                "lag_coefficient": 0.92,
            },
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    payload["probabilistic"] = {
        "distribution": {
            "status": "completed",
            "selected_family": "analytic",
            "rows": [
                {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 11.3,
                    "scale": 0.4,
                    "realized_observation": 11.4,
                }
            ],
            "latest_row": {
                "origin_time": "2026-04-16T00:00:00Z",
                "available_at": "2026-04-17T21:00:00Z",
                "horizon": 1,
                "location": 11.3,
                "scale": 0.4,
                "realized_observation": 11.4,
            },
            "calibration": {
                "status": "passed",
                "passed": True,
                "gate_effect": "publishable",
                "diagnostics": [
                    {
                        "diagnostic_id": "pit_or_randomized_pit_uniformity",
                        "sample_size": 1,
                        "status": "passed",
                    }
                ],
            },
            "chart": {"forecast_bands": []},
        }
    }
    payload["holistic_equation"] = {
        "status": "completed",
        "claim_class": "holistic_equation",
        "deterministic_source": "predictive_law",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
        "honesty_note": "Label-only holistic payload should not survive overlay gating.",
        "equation": {
            "label": "y(t) = predictive + epsilon",
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is not None
    assert normalized["holistic_equation"] is None
    assert normalized["claim_class"] == "predictive_law"
    assert normalized["publishable"] is True
    assert "no_backend_joint_claim" in normalized["gap_report"]
    assert normalized["not_holistic_because"] == normalized["gap_report"]


def test_normalize_analysis_payload_rejects_incomplete_backend_holistic_claim(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "spy-price-close-incomplete-holistic.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "10.0",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "11.2",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "10.7",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-17T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "observed_value": "11.4",
                    "revision_id": "1",
                },
            ]
        )

    payload = {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "price_close", "label": "Price Close"},
            "dataset_csv": str(dataset_csv),
        },
        "operator_point": {
            "status": "completed",
            "result_mode": "candidate_publication",
            "selected_family": "analytic",
            "claim_card_ref": "claim_card_manifest@1.1.0:claim-card-2",
            "scorecard_ref": "scorecard_manifest@1.1.0:scorecard-2",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-2",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-2"
            ),
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 1.8 + 0.92*y(t-1)",
                "parameter_summary": {
                    "intercept": 1.8,
                    "lag_coefficient": 0.92,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 10.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 11.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 10.8,
                    },
                    {
                        "event_time": "2026-04-17T00:00:00Z",
                        "fitted_value": 11.3,
                    },
                ],
            },
            "claim_card": {
                "claim_type": "predictive_within_declared_scope",
                "claim_ceiling": "predictive_within_declared_scope",
                "predictive_support_status": "confirmatory_supported",
                "allowed_interpretation_codes": [
                    "historical_structure_summary",
                    "point_forecast_within_declared_validation_scope",
                ],
            },
            "scorecard": {
                "descriptive_status": "passed",
                "predictive_status": "passed",
            },
        },
        "probabilistic": {
            "distribution": {
                "status": "completed",
                "selected_family": "analytic",
                "rows": [
                    {
                        "origin_time": "2026-04-16T00:00:00Z",
                        "available_at": "2026-04-17T21:00:00Z",
                        "horizon": 1,
                        "location": 11.3,
                        "scale": 0.4,
                        "realized_observation": 11.4,
                    }
                ],
                "latest_row": {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 11.3,
                    "scale": 0.4,
                    "realized_observation": 11.4,
                },
                "calibration": {
                    "status": "passed",
                    "passed": True,
                    "gate_effect": "publishable",
                    "diagnostics": [
                        {
                            "diagnostic_id": "pit_or_randomized_pit_uniformity",
                            "sample_size": 1,
                            "status": "passed",
                        }
                    ],
                },
                "chart": {"forecast_bands": []},
            }
        },
        "holistic_equation": {
            "status": "completed",
            "claim_class": "holistic_equation",
            "deterministic_source": "predictive_law",
            "probabilistic_source": "distribution",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-2",
            "honesty_note": "Missing publication record ref should fail strict gate.",
            "equation": {
                "label": "y(t) = predictive + epsilon",
            },
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is not None
    assert normalized["claim_class"] == "predictive_law"
    assert normalized["holistic_equation"] is None
    assert "no_backend_joint_claim" in normalized["gap_report"]


def test_normalize_analysis_payload_reconciles_saved_descriptive_fit_from_benchmark(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "spy-daily-return-reconciled.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "0.01",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "0.013",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "0.0154",
                    "revision_id": "1",
                },
            ]
        )

    payload = {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return", "label": "Daily Return"},
            "dataset_csv": str(dataset_csv),
        },
        "benchmark": {
            "status": "completed",
            "portfolio_selection": {
                "winner_submitter_id": "portfolio_orchestrator",
                "winner_candidate_id": "analytic_lag1_affine",
            },
            "descriptive_fit": {
                "status": "completed",
                "source": "benchmark_local_selection",
                "submitter_id": "portfolio_orchestrator",
                "candidate_id": "analytic_lag1_affine",
                "accepted_candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "metrics": {"total_code_bits": 123.0},
                "equation": {
                    "candidate_id": "analytic_lag1_affine",
                    "family_id": "analytic",
                    "label": "y(t) = 0.005 + 0.8*y(t-1)",
                    "parameter_summary": {
                        "intercept": 0.005,
                        "lag_coefficient": 0.8,
                    },
                    "curve": [
                        {
                            "event_time": "2026-04-14T00:00:00Z",
                            "fitted_value": 0.01,
                        },
                        {
                            "event_time": "2026-04-15T00:00:00Z",
                            "fitted_value": 0.013,
                        },
                        {
                            "event_time": "2026-04-16T00:00:00Z",
                            "fitted_value": 0.029,
                        },
                    ],
                },
                "chart": {
                    "equation_curve": [
                        {
                            "event_time": "2026-04-14T00:00:00Z",
                            "fitted_value": 0.01,
                        },
                        {
                            "event_time": "2026-04-15T00:00:00Z",
                            "fitted_value": 0.013,
                        },
                        {
                            "event_time": "2026-04-16T00:00:00Z",
                            "fitted_value": 0.029,
                        },
                    ]
                },
            },
        },
        "descriptive_fit": {
            "status": "completed",
            "source": "saved_payload",
            "submitter_id": "stale_saved_payload",
            "candidate_id": "stale_saved_fit",
            "family_id": "analytic",
            "metrics": {"total_code_bits": 999.0},
            "equation": {
                "candidate_id": "stale_saved_fit",
                "family_id": "analytic",
                "label": "y(t) = 999 + 0*y(t-1)",
                "parameter_summary": {
                    "intercept": 999.0,
                    "lag_coefficient": 0.0,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 999.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 999.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 999.0,
                    },
                ],
            },
            "chart": {
                "equation_curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 999.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 999.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 999.0,
                    },
                ]
            },
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["descriptive_fit"]["candidate_id"] == "analytic_lag1_affine"
    assert normalized["descriptive_fit"]["source"] == "benchmark_local_selection"
    assert (
        normalized["descriptive_fit"]["equation"]["label"]
        == "y(t) = 0.005 + 0.8*y(t-1)"
    )
    assert normalized["descriptive_fit"]["law_eligible"] is True


def test_normalize_analysis_payload_preserves_explicit_law_scope_for_best_overall_candidate(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "spy-daily-return-law-scope.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "0.01",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "0.03",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "0.025",
                    "revision_id": "1",
                },
            ]
        )

    payload = {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return", "label": "Daily Return"},
            "dataset_csv": str(dataset_csv),
        },
        "benchmark": {
            "status": "completed",
            "portfolio_selection": {
                "winner_submitter_id": "recursive_spectral_backend",
                "winner_candidate_id": "recursive_level_smoother",
            },
            "descriptive_fit": {
                "status": "completed",
                "source": "benchmark_local_selection",
                "submitter_id": "analytic_backend",
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "selection_scope": "best_overall_candidate",
                "selection_rule": (
                    "min_total_code_bits_then_max_description_gain_then_"
                    "min_structure_code_bits_then_min_canonical_byte_length_then_candidate_id"
                ),
                "law_eligible": False,
                "law_rejection_reason_codes": ["outside_law_eligible_scope"],
                "equation": {
                    "candidate_id": "analytic_lag1_affine",
                    "family_id": "analytic",
                    "label": "y(t) = 0.005 + 0.8*y(t-1)",
                    "parameter_summary": {
                        "intercept": 0.005,
                        "lag_coefficient": 0.8,
                    },
                    "curve": [
                        {
                            "event_time": "2026-04-14T00:00:00Z",
                            "fitted_value": 0.01,
                        },
                        {
                            "event_time": "2026-04-15T00:00:00Z",
                            "fitted_value": 0.013,
                        },
                        {
                            "event_time": "2026-04-16T00:00:00Z",
                            "fitted_value": 0.029,
                        },
                    ],
                },
                "chart": {
                    "equation_curve": [
                        {
                            "event_time": "2026-04-14T00:00:00Z",
                            "fitted_value": 0.01,
                        },
                        {
                            "event_time": "2026-04-15T00:00:00Z",
                            "fitted_value": 0.013,
                        },
                        {
                            "event_time": "2026-04-16T00:00:00Z",
                            "fitted_value": 0.029,
                        },
                    ]
                },
            },
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["descriptive_fit"]["candidate_id"] == "analytic_lag1_affine"
    assert normalized["descriptive_fit"]["selection_scope"] == "best_overall_candidate"
    assert normalized["descriptive_fit"]["law_eligible"] is False
    assert normalized["descriptive_fit"]["law_rejection_reason_codes"] == [
        "outside_law_eligible_scope"
    ]


def test_normalize_analysis_payload_requires_explicit_accepted_candidate_metadata(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "spy-daily-return-no-accepted-candidate.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "0.01",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "0.03",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "0.025",
                    "revision_id": "1",
                },
            ]
        )

    payload = {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return", "label": "Daily Return"},
            "dataset_csv": str(dataset_csv),
        },
        "benchmark": {
            "status": "completed",
            "portfolio_selection": {
                "winner_submitter_id": "recursive_spectral_backend",
                "winner_candidate_id": "analytic_lag1_affine",
            },
            "local_winner_candidate_id": "analytic_lag1_affine",
            "descriptive_fit": {
                "status": "completed",
                "source": "benchmark_local_selection",
                "submitter_id": "analytic_backend",
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "selection_scope": "best_overall_candidate",
                "selection_rule": (
                    "min_total_code_bits_then_max_description_gain_then_"
                    "min_structure_code_bits_then_min_canonical_byte_length_then_candidate_id"
                ),
                "equation": {
                    "candidate_id": "analytic_lag1_affine",
                    "family_id": "analytic",
                    "label": "y(t) = 0.005 + 0.8*y(t-1)",
                    "parameter_summary": {
                        "intercept": 0.005,
                        "lag_coefficient": 0.8,
                    },
                    "curve": [
                        {
                            "event_time": "2026-04-14T00:00:00Z",
                            "fitted_value": 0.01,
                        },
                        {
                            "event_time": "2026-04-15T00:00:00Z",
                            "fitted_value": 0.013,
                        },
                        {
                            "event_time": "2026-04-16T00:00:00Z",
                            "fitted_value": 0.029,
                        },
                    ],
                },
                "chart": {
                    "equation_curve": [
                        {
                            "event_time": "2026-04-14T00:00:00Z",
                            "fitted_value": 0.01,
                        },
                        {
                            "event_time": "2026-04-15T00:00:00Z",
                            "fitted_value": 0.013,
                        },
                        {
                            "event_time": "2026-04-16T00:00:00Z",
                            "fitted_value": 0.029,
                        },
                    ]
                },
            },
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["descriptive_fit"]["candidate_id"] == "analytic_lag1_affine"
    assert normalized["descriptive_fit"]["law_eligible"] is False
    assert normalized["descriptive_fit"]["law_rejection_reason_codes"] == [
        "no_accepted_candidate"
    ]


def test_normalize_analysis_payload_adds_lane_scope_and_evidence_warnings(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "spy-price-close.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
                "close",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "500.00",
                    "revision_id": "1",
                    "close": "500.00",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "503.20",
                    "revision_id": "1",
                    "close": "503.20",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "502.60",
                    "revision_id": "1",
                    "close": "502.60",
                },
            ]
        )

    operator_manifest = tmp_path / "operator-point.yaml"
    operator_manifest.write_text(
        "\n".join(
            [
                "workflow_id: euclid_current_release_candidate",
                "search:",
                "  class: bounded_heuristic",
                "  family_ids:",
                "  - constant",
                "  - drift",
                "  - linear_trend",
                "  - seasonal_naive",
            ]
        ),
        encoding="utf-8",
    )

    distribution_manifest = tmp_path / "distribution.yaml"
    distribution_manifest.write_text(
        "\n".join(
            [
                "search:",
                "  class: exact_finite_enumeration",
                "  family_ids:",
                "  - analytic_lag1_affine",
                "  proposal_limit: 1",
            ]
        ),
        encoding="utf-8",
    )

    payload = {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {
                "id": "price_close",
                "label": "Price Close",
                "description": "Predict the raw close level for each trading day.",
                "y_axis_label": "Close",
            },
            "dataset_csv": str(dataset_csv),
        },
        "operator_point": {
            "status": "completed",
            "manifest_path": str(operator_manifest),
            "selected_family": "drift",
            "result_mode": "abstention_only_publication",
            "equation": {
                "label": "y(t) = 412.17 + 0.229848*t",
                "parameter_summary": {"intercept": 412.17, "slope": 0.229848},
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 412.17,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 412.399848,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 412.629696,
                    },
                ],
            },
            "abstention": {"reason_codes": ["robustness_failed"]},
        },
        "probabilistic": {
            "distribution": {
                "status": "completed",
                "manifest_path": str(distribution_manifest),
                "selected_family": "analytic",
                "rows": [
                    {
                        "available_at": "2026-04-15T21:00:00Z",
                        "origin_time": "2026-04-14T00:00:00Z",
                        "horizon": 1,
                        "location": 501.0,
                        "scale": 0.4,
                        "realized_observation": 503.2,
                    },
                    {
                        "available_at": "2026-04-16T21:00:00Z",
                        "origin_time": "2026-04-14T00:00:00Z",
                        "horizon": 2,
                        "location": 501.4,
                        "scale": 0.5,
                        "realized_observation": 502.6,
                    },
                ],
                "calibration": {
                    "status": "passed",
                    "passed": True,
                    "diagnostics": [
                        {
                            "diagnostic_id": "pit_or_randomized_pit_uniformity",
                            "sample_size": 2,
                            "status": "passed",
                        }
                    ],
                },
            }
        },
    }

    normalized = normalize_analysis_payload(payload)

    operator_scope = normalized["operator_point"]["search_scope"]
    distribution_scope = normalized["probabilistic"]["distribution"]["search_scope"]
    distribution_evidence = normalized["probabilistic"]["distribution"]["evidence"]

    assert operator_scope["scope_kind"] == "narrow_release_surface"
    assert operator_scope["family_ids"] == [
        "constant",
        "drift",
        "linear_trend",
        "seasonal_naive",
    ]
    assert distribution_scope["scope_kind"] == "single_candidate_template"
    assert distribution_evidence["strength"] == "thin"
    assert distribution_evidence["sample_size"] == 2


def test_normalize_analysis_payload_surfaces_production_probabilistic_refs_and_family_bands() -> None:
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {
                "id": "price_close",
                "label": "Price Close",
                "y_axis_label": "Close",
            },
        },
        "probabilistic": {
            "distribution": {
                "status": "completed",
                "selected_family": "student_t",
                "distribution_family": "student_t",
                "lane_status": "downgraded",
                "downgrade_reason_codes": [
                    "heuristic_gaussian_compatibility_only",
                ],
                "residual_history_refs": [
                    {
                        "schema_name": "residual_history_manifest@1.0.0",
                        "object_id": "spy_residual_history",
                    }
                ],
                "stochastic_model_refs": [
                    {
                        "schema_name": "stochastic_model_manifest@1.0.0",
                        "object_id": "spy_student_t_model",
                    }
                ],
                "rows": [
                    {
                        "available_at": "2026-04-15T21:00:00Z",
                        "origin_time": "2026-04-14T00:00:00Z",
                        "horizon": 1,
                        "location": 501.0,
                        "scale": 50.0,
                        "configured_interval": {
                            "level": 0.9,
                            "lower": 498.25,
                            "upper": 504.75,
                        },
                        "quantiles": [
                            {"level": 0.1, "value": 498.5},
                            {"level": 0.5, "value": 501.0},
                            {"level": 0.9, "value": 504.5},
                        ],
                        "realized_observation": 502.0,
                    }
                ],
                "calibration": {
                    "status": "passed",
                    "passed": True,
                    "gate_effect": "publishable",
                    "diagnostics": [
                        {
                            "diagnostic_id": "pit_or_randomized_pit_uniformity",
                            "sample_size": 6,
                            "calibration_bins": [
                                {
                                    "bin_id": "0.0-0.5",
                                    "expected_count": 3,
                                    "observed_count": 2,
                                },
                                {
                                    "bin_id": "0.5-1.0",
                                    "expected_count": 3,
                                    "observed_count": 4,
                                },
                            ],
                        }
                    ],
                },
            }
        },
    }

    normalized = normalize_analysis_payload(payload)

    lane = normalized["probabilistic"]["distribution"]
    evidence = lane["evidence"]
    band = lane["chart"]["forecast_bands"][0]

    assert band["center"] == pytest.approx(501.0)
    assert band["lower"] == pytest.approx(498.25)
    assert band["upper"] == pytest.approx(504.75)
    assert band["source"] == "configured_interval"
    assert evidence["family"] == "student_t"
    assert evidence["lane_status"] == "downgraded"
    assert evidence["downgrade_reason_codes"] == [
        "heuristic_gaussian_compatibility_only",
    ]
    assert evidence["residual_history_refs"] == [
        "residual_history_manifest@1.0.0:spy_residual_history",
    ]
    assert evidence["stochastic_model_refs"] == [
        "stochastic_model_manifest@1.0.0:spy_student_t_model",
    ]
    assert evidence["calibration_bin_count"] == 2


def test_normalize_analysis_payload_builds_change_atlas_from_price_space_rows(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "spy-price-close.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
                "close",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-10T00:00:00Z",
                    "available_at": "2026-04-10T21:00:00Z",
                    "observed_value": "100.00",
                    "revision_id": "1",
                    "close": "100.00",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-13T00:00:00Z",
                    "available_at": "2026-04-13T21:00:00Z",
                    "observed_value": "102.00",
                    "revision_id": "1",
                    "close": "102.00",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "101.00",
                    "revision_id": "1",
                    "close": "101.00",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "104.00",
                    "revision_id": "1",
                    "close": "104.00",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "110.00",
                    "revision_id": "1",
                    "close": "110.00",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-17T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "observed_value": "108.00",
                    "revision_id": "1",
                    "close": "108.00",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-20T00:00:00Z",
                    "available_at": "2026-04-20T21:00:00Z",
                    "observed_value": "112.00",
                    "revision_id": "1",
                    "close": "112.00",
                },
            ]
        )

    payload = {
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "SPY",
            "target": {
                "id": "price_close",
                "label": "Price Close",
                "description": "Predict the raw close level for each trading day.",
                "y_axis_label": "Close",
            },
            "dataset_csv": str(dataset_csv),
        },
        "probabilistic": {
            "distribution": {
                "status": "completed",
                "selected_family": "analytic",
                "rows": [
                    {
                        "available_at": "2026-04-13T21:00:00Z",
                        "origin_time": "2026-04-10T00:00:00Z",
                        "horizon": 1,
                        "location": 101.8,
                        "scale": 1.2,
                        "realized_observation": 102.0,
                    },
                    {
                        "available_at": "2026-04-17T21:00:00Z",
                        "origin_time": "2026-04-10T00:00:00Z",
                        "horizon": 5,
                        "location": 109.0,
                        "scale": 4.0,
                        "realized_observation": 108.0,
                    },
                ],
                "latest_row": {
                    "available_at": "2026-04-17T21:00:00Z",
                    "origin_time": "2026-04-10T00:00:00Z",
                    "horizon": 5,
                    "location": 109.0,
                    "scale": 4.0,
                    "realized_observation": 108.0,
                },
                "calibration": {
                    "status": "passed",
                    "passed": True,
                    "gate_effect": "publishable",
                    "diagnostics": [],
                },
            }
        },
    }

    normalized = normalize_analysis_payload(payload)

    change_atlas = normalized["change_atlas"]
    assert change_atlas["status"] == "completed"
    assert change_atlas["horizons"] == [1, 5]
    assert [metric["id"] for metric in change_atlas["metrics"]] == [
        "delta",
        "return",
        "log_return",
    ]

    historical_return_h5 = change_atlas["historical"]["return"]["5"]
    assert historical_return_h5["sample_size"] == 2
    assert historical_return_h5["latest_value"] == pytest.approx(
        (112.0 / 102.0) - 1.0
    )

    forecast_return_h5 = change_atlas["forecast"]["lanes"]["distribution"]["return"][
        "5"
    ]
    assert forecast_return_h5["origin_close"] == 100.0
    assert forecast_return_h5["center"] == pytest.approx(0.09)
    assert forecast_return_h5["lower"] == pytest.approx(0.05)
    assert forecast_return_h5["upper"] == pytest.approx(0.13)
    assert forecast_return_h5["realized"] == pytest.approx(0.08)


def test_normalize_analysis_payload_summarizes_missing_descriptive_fit() -> None:
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {
                "id": "daily_return",
                "label": "Daily Return",
                "description": "Predict the close-to-close fractional return for each day.",
                "y_axis_label": "Return",
            },
        },
        "benchmark": {
            "status": "completed",
            "portfolio_selection": {
                "winner_submitter_id": None,
                "winner_candidate_id": None,
            },
            "submitters": [
                {
                    "submitter_id": "analytic_backend",
                    "status": "abstained",
                    "selected_candidate_id": None,
                    "candidate_ledger": [
                        {
                            "reason_codes": ["descriptive_admissibility_failed"],
                            "details": {
                                "diagnostics": [
                                    {
                                        "reason_codes": [
                                            "description_gain_non_positive"
                                        ]
                                    }
                                ]
                            },
                        }
                    ],
                },
                {
                    "submitter_id": "algorithmic_search_backend",
                    "status": "abstained",
                    "selected_candidate_id": None,
                    "candidate_ledger": [
                        {
                            "reason_codes": ["descriptive_admissibility_failed"],
                            "details": {
                                "diagnostics": [
                                    {
                                        "reason_codes": [
                                            "description_gain_non_positive"
                                        ]
                                    }
                                ]
                            },
                        }
                    ],
                },
            ],
        },
    }

    normalized = normalize_analysis_payload(payload)

    status = normalized["benchmark"]["descriptive_fit_status"]

    assert status["status"] == "absent_no_admissible_candidate"
    assert status["reason_codes"] == ["description_gain_non_positive"]
    assert "no admissible descriptive equation" in status["headline"].lower()


def test_normalize_analysis_payload_reports_reconstruction_floor_failure() -> None:
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {
                "id": "price_close",
                "label": "Price Close",
                "description": "Predict the raw close level for each trading day.",
                "y_axis_label": "Close",
            },
        },
        "benchmark": {
            "status": "completed",
            "descriptive_fit_materialization_reason": "reconstruction_floor_failed",
            "submitters": [
                {
                    "submitter_id": "analytic_backend",
                    "status": "selected",
                    "selected_candidate_id": "analytic_persistence",
                },
                {
                    "submitter_id": "recursive_spectral_backend",
                    "status": "selected",
                    "selected_candidate_id": "analytic_constant",
                },
            ],
        },
    }

    normalized = normalize_analysis_payload(payload)

    status = normalized["benchmark"]["descriptive_fit_status"]

    assert status["status"] == "absent_reconstruction_floor_failed"
    assert status["reason_codes"] == ["reconstruction_floor_failed"]
    assert "reproduced the path closely enough" in status["headline"].lower()
    assert normalized.get("descriptive_fit") is None


def test_normalize_analysis_payload_drops_stale_flat_daily_return_descriptive_fit(
    tmp_path: Path,
) -> None:
    dataset_rows = _workbench_dataset_rows([0.01, -0.01] * 16)
    dataset_csv = _write_dataset_csv(
        tmp_path / "spy-daily-return.csv",
        dataset_rows,
    )
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {
                "id": "daily_return",
                "label": "Daily Return",
                "description": "Predict the close-to-close fractional return for each day.",
                "y_axis_label": "Return",
            },
            "dataset_csv": str(dataset_csv),
        },
        "descriptive_fit": {
            "status": "completed",
            "source": "benchmark_local_selection",
            "submitter_id": "algorithmic_search_backend",
            "candidate_id": "algorithmic_last_observation",
            "family_id": "algorithmic",
            "equation": {"label": "y(t) = y(t-1)"},
            "chart": {
                "equation_curve": _constant_equation_curve(
                    dataset_rows,
                    fitted_value=0.01,
                )
            },
        },
        "benchmark": {
            "status": "completed",
            "submitters": [
                {
                    "submitter_id": "algorithmic_search_backend",
                    "status": "selected",
                    "selected_candidate_id": "algorithmic_last_observation",
                }
            ],
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized.get("descriptive_fit") is None
    assert (
        normalized["benchmark"]["descriptive_fit_materialization_reason"]
        == "reconstruction_floor_failed"
    )
    status = normalized["benchmark"]["descriptive_fit_status"]
    assert status["status"] == "absent_reconstruction_floor_failed"
    assert status["reason_codes"] == ["reconstruction_floor_failed"]


def test_normalize_analysis_payload_projects_legacy_operator_point_descriptive_fallback(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "gld-price-close.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "GLD",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "300.0",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "GLD",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "301.2",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "GLD",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "300.8",
                    "revision_id": "1",
                },
            ]
        )

    payload = {
        "analysis_version": "0.9.0",
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "GLD",
            "target": {"id": "price_close", "label": "Price Close"},
            "dataset_csv": str(dataset_csv),
        },
        "benchmark": {
            "status": "completed",
            "descriptive_fit_status": {
                "status": "candidate_available_but_not_loaded",
                "headline": (
                    "Saved analysis predates descriptive-bank payloads; project "
                    "a descriptive fallback instead of leaving the lane empty."
                ),
                "reason_codes": [],
            },
            "portfolio_selection": {
                "winner_submitter_id": None,
                "winner_candidate_id": None,
            },
        },
        "operator_point": {
            "status": "completed",
            "result_mode": "abstention_only_publication",
            "selected_family": "analytic",
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 1.2 + 0.99*y(t-1)",
                "parameter_summary": {
                    "intercept": 1.2,
                    "lag_coefficient": 0.99,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 300.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 301.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 301.1,
                    },
                ],
            },
            "abstention": {
                "reason_codes": ["robustness_failed", "perturbation_protocol_failed"]
            },
        },
        "probabilistic": {
            "distribution": {
                "status": "completed",
                "selected_family": "analytic",
                "rows": [
                    {
                        "origin_time": "2026-04-15T00:00:00Z",
                        "available_at": "2026-04-16T21:00:00Z",
                        "horizon": 1,
                        "location": 301.1,
                        "scale": 0.8,
                        "realized_observation": 300.8,
                    }
                ],
                "latest_row": {
                    "origin_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "horizon": 1,
                    "location": 301.1,
                    "scale": 0.8,
                    "realized_observation": 300.8,
                },
                "calibration": {
                    "status": "passed",
                    "passed": True,
                    "gate_effect": "publishable",
                    "diagnostics": [
                        {
                            "diagnostic_id": "pit_or_randomized_pit_uniformity",
                            "sample_size": 1,
                            "status": "passed",
                        }
                    ],
                },
                "chart": {"forecast_bands": []},
            }
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["descriptive_fit"]["status"] == "completed"
    assert normalized["descriptive_fit"]["source"] == "legacy_operator_point_fallback"
    status = normalized["benchmark"]["descriptive_fit_status"]
    assert status["status"] == "legacy_compatibility_projection"
    assert "compatibility projection" in status["headline"].lower()
    assert status["reason_codes"] == ["legacy_compatibility_projection"]
    assert (
        normalized["descriptive_fit"]["equation"]["label"]
        == normalized["operator_point"]["equation"]["label"]
    )
    assert normalized["descriptive_fit"]["claim_class"] == "descriptive_fit"
    assert normalized["descriptive_fit"]["is_law_claim"] is False
    assert normalized["descriptive_fit"]["law_eligible"] is False
    assert normalized["claim_class"] == "descriptive_fit"
    assert normalized["holistic_equation"] is None


def test_normalize_analysis_payload_does_not_project_legacy_operator_point_for_current_analysis(
    tmp_path: Path,
) -> None:
    dataset_csv = tmp_path / "gld-price-close-current.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "series_id",
                "event_time",
                "available_at",
                "observed_value",
                "revision_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fmp",
                    "series_id": "GLD",
                    "event_time": "2026-04-14T00:00:00Z",
                    "available_at": "2026-04-14T21:00:00Z",
                    "observed_value": "300.0",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "GLD",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "301.2",
                    "revision_id": "1",
                },
                {
                    "source_id": "fmp",
                    "series_id": "GLD",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "300.8",
                    "revision_id": "1",
                },
            ]
        )

    payload = {
        "analysis_version": "1.0.0",
        "workspace_root": str(tmp_path),
        "dataset": {
            "symbol": "GLD",
            "target": {"id": "price_close", "label": "Price Close"},
            "dataset_csv": str(dataset_csv),
        },
        "benchmark": {
            "status": "completed",
            "descriptive_fit_status": {
                "status": "candidate_available_but_not_loaded",
                "headline": (
                    "Current analysis should not project operator point through "
                    "the legacy descriptive fallback."
                ),
                "reason_codes": [],
            },
            "portfolio_selection": {
                "winner_submitter_id": None,
                "winner_candidate_id": None,
            },
        },
        "operator_point": {
            "status": "completed",
            "result_mode": "abstention_only_publication",
            "selected_family": "analytic",
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "label": "y(t) = 1.2 + 0.99*y(t-1)",
                "parameter_summary": {
                    "intercept": 1.2,
                    "lag_coefficient": 0.99,
                },
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 300.0,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": 301.0,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 301.1,
                    },
                ],
            },
            "abstention": {
                "reason_codes": ["robustness_failed", "perturbation_protocol_failed"]
            },
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized.get("descriptive_fit") is None
    assert normalized.get("claim_class") is None
    assert (
        normalized["benchmark"]["descriptive_fit_status"]["status"]
        == "candidate_available_but_not_loaded"
    )


def test_normalize_analysis_payload_preserves_completed_benchmark_without_winner() -> None:
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {
                "id": "daily_return",
                "label": "Daily Return",
                "description": "Predict the close-to-close fractional return for each day.",
                "y_axis_label": "Return",
            },
        },
        "benchmark": {
            "status": "completed",
            "local_winner_submitter_id": None,
            "local_winner_candidate_id": None,
            "portfolio_selection": {
                "status": "completed",
                "compared_finalists": [],
                "decision_trace": [
                    {
                        "step": "portfolio_selection",
                        "status": "no_admissible_finalist",
                        "message": "No admissible finalist was selected",
                    }
                ],
            },
            "submitters": [
                {
                    "submitter_id": "analytic_backend",
                    "status": "abstained",
                    "selected_candidate_id": None,
                },
                {
                    "submitter_id": "algorithmic_search_backend",
                    "status": "abstained",
                    "selected_candidate_id": None,
                },
            ],
        },
    }

    normalized = normalize_analysis_payload(payload)

    benchmark = normalized["benchmark"]
    selection = benchmark["portfolio_selection"]

    assert benchmark["status"] == "completed"
    assert benchmark["local_winner_submitter_id"] is None
    assert benchmark["local_winner_candidate_id"] is None
    assert selection["status"] == "completed"
    assert selection.get("winner_submitter_id") is None
    assert selection.get("winner_candidate_id") is None
    assert selection.get("selection_explanation") is None
    assert selection["selection_explanation_raw"] is None
    assert selection["decision_trace"] == [
        {
            "step": "portfolio_selection",
            "status": "no_admissible_finalist",
            "message": "No admissible finalist was selected",
        }
    ]


def test_create_workbench_analysis_requires_explicit_date_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "euclid.workbench.service.fetch_fmp_eod_history",
        lambda **_: [
            {
                "date": "2025-01-02",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000_000,
            },
            {
                "date": "2025-01-03",
                "open": 100.5,
                "high": 101.5,
                "low": 100.0,
                "close": 101.0,
                "volume": 1_100_000,
            },
        ],
    )

    with pytest.raises(ValueError, match="start_date and end_date are required"):
        create_workbench_analysis(
            symbol="SPY",
            api_key="test-key",
            target_id="price_close",
            output_root=tmp_path / "workbench",
            project_root=tmp_path,
            start_date="2025-01-02",
            end_date=None,
            include_probabilistic=False,
            include_benchmark=False,
        )


def test_normalize_analysis_payload_builds_descriptive_reconstruction_claim(
    tmp_path: Path,
) -> None:
    values = [
        (1.2 * math.cos((2.0 * math.pi * 2.0 * index) / 48.0))
        + (0.3 * math.sin((2.0 * math.pi * 5.0 * index) / 48.0))
        for index in range(48)
    ]
    dataset_csv = _write_dataset_csv(
        tmp_path / "descriptive-reconstruction.csv",
        _workbench_dataset_rows(values),
    )
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return"},
            "dataset_csv": str(dataset_csv),
        },
        "operator_point": {"status": "failed", "error": "fixture"},
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["claim_class"] == "descriptive_reconstruction"
    assert normalized["publishable"] is False
    assert normalized["predictive_law"] is None
    assert normalized["holistic_equation"] is None
    assert normalized["descriptive_reconstruction"]["status"] == "completed"
    assert normalized["descriptive_reconstruction"]["law_eligible"] is False
    assert (
        normalized["descriptive_reconstruction"]["equation"]["candidate_id"]
        == "descriptive_fourier_reconstruction"
    )


def test_create_workbench_analysis_rejects_inverted_date_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "euclid.workbench.service.fetch_fmp_eod_history",
        lambda **_: [
            {
                "date": "2025-01-02",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000_000,
            },
            {
                "date": "2025-01-03",
                "open": 100.5,
                "high": 101.5,
                "low": 100.0,
                "close": 101.0,
                "volume": 1_100_000,
            },
        ],
    )

    with pytest.raises(ValueError, match="start_date must be on or before end_date"):
        create_workbench_analysis(
            symbol="SPY",
            api_key="test-key",
            target_id="price_close",
            output_root=tmp_path / "workbench",
            project_root=tmp_path,
            start_date="2025-01-10",
            end_date="2025-01-02",
            include_probabilistic=False,
            include_benchmark=False,
        )
