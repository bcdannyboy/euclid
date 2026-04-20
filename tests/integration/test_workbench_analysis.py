from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from euclid.workbench.server import WorkbenchRequestHandler
from euclid.workbench.service import create_workbench_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fake_history(*, days: int = 48) -> list[dict[str, float | str | int]]:
    start = date(2025, 1, 2)
    history: list[dict[str, float | str | int]] = []
    close = 500.0
    for offset in range(days):
        current = start + timedelta(days=offset)
        if current.weekday() >= 5:
            continue
        close += 1.25 if offset % 3 else -0.4
        history.append(
            {
                "date": current.isoformat(),
                "open": round(close - 0.6, 4),
                "high": round(close + 0.8, 4),
                "low": round(close - 1.1, 4),
                "close": round(close, 4),
                "volume": 1_000_000 + offset * 10_000,
            }
        )
    return history


def _write_saved_daily_return_analysis_fixture(
    tmp_path: Path,
) -> dict[str, Path]:
    long_segment = "saved-analysis-regression-artifact-path"
    workspace_root = tmp_path / (
        "20260416T160614Z-spy-daily-return-"
        "saved-analysis-regression-fixture"
    )
    dataset_dir = workspace_root / "datasets" / long_segment / long_segment
    dataset_dir.mkdir(parents=True)
    dataset_csv = dataset_dir / "spy-daily-return.csv"
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
                "previous_close",
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
                    "observed_value": "0.0015",
                    "revision_id": "1",
                    "previous_close": "500.00",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-15T00:00:00Z",
                    "available_at": "2026-04-15T21:00:00Z",
                    "observed_value": "-0.0020",
                    "revision_id": "1",
                    "previous_close": "500.75",
                },
                {
                    "source_id": "fmp",
                    "series_id": "SPY",
                    "event_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-16T21:00:00Z",
                    "observed_value": "0.0045",
                    "revision_id": "1",
                    "previous_close": "499.75",
                },
            ]
        )

    artifact_root = workspace_root / "artifacts" / long_segment / long_segment
    operator_manifest_path = artifact_root / "manifests" / "operator-point.yaml"
    probabilistic_manifest_path = artifact_root / "manifests" / "distribution.yaml"
    benchmark_root = artifact_root / "benchmark-root" / long_segment
    report_path = benchmark_root / "reports" / "benchmark-report.md"
    task_result_path = benchmark_root / "reports" / "benchmark-task-result.json"
    telemetry_path = benchmark_root / "telemetry" / "benchmark-telemetry.json"

    analysis_path = workspace_root / "analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "analysis_version": "1.0.0",
                "workspace_root": str(workspace_root),
                "request": {
                    "symbol": "SPY",
                    "target_id": "daily_return",
                    "start_date": "2026-04-14",
                    "end_date": "2026-04-16",
                    "include_probabilistic": True,
                    "include_benchmark": True,
                },
                "dataset": {
                    "symbol": "SPY",
                    "target": {"id": "daily_return"},
                    "dataset_csv": str(dataset_csv),
                },
                "operator_point": {
                    "status": "completed",
                    "manifest_path": str(operator_manifest_path),
                    "selected_family": "drift",
                    "result_mode": "abstention_only_publication",
                    "equation": {
                        "label": "y(t) = 0.0004 + 0.1*t",
                        "parameter_summary": {
                            "intercept": 0.0004,
                            "slope": 0.1,
                        },
                        "curve": [
                            {
                                "event_time": "2026-04-14T00:00:00Z",
                                "fitted_value": 0.0004,
                            },
                            {
                                "event_time": "2026-04-15T00:00:00Z",
                                "fitted_value": 0.1004,
                            },
                            {
                                "event_time": "2026-04-16T00:00:00Z",
                                "fitted_value": 0.2004,
                            },
                        ],
                    },
                    "chart": {
                        "actual_series": [
                            {
                                "event_time": "2026-04-14T00:00:00Z",
                                "observed_value": 0.0015,
                            },
                            {
                                "event_time": "2026-04-15T00:00:00Z",
                                "observed_value": -0.0020,
                            },
                            {
                                "event_time": "2026-04-16T00:00:00Z",
                                "observed_value": 0.0045,
                            },
                        ],
                        "equation_curve": [
                            {
                                "event_time": "2026-04-14T00:00:00Z",
                                "fitted_value": 0.0004,
                            },
                            {
                                "event_time": "2026-04-15T00:00:00Z",
                                "fitted_value": 0.1004,
                            },
                            {
                                "event_time": "2026-04-16T00:00:00Z",
                                "fitted_value": 0.2004,
                            },
                        ],
                    },
                    "abstention": {
                        "reason_codes": [
                            "robustness_failed",
                            "perturbation_protocol_failed",
                        ]
                    },
                },
                "probabilistic": {
                    "distribution": {
                        "status": "failed",
                        "manifest_path": str(probabilistic_manifest_path),
                        "error": {
                            "message": (
                                "Insufficient realized holdout forecasts to publish "
                                "a probabilistic distribution lane."
                            ),
                        },
                    },
                    "interval": {
                        "status": "failed",
                        "error": {
                            "message": (
                                "No interval lane cleared the saved-analysis "
                                "publication gates."
                            ),
                        },
                    }
                },
                "holistic_equation": {
                    "status": "completed",
                    "source": "operator_point",
                    "exactness": "sample_exact_closure",
                    "equation": {
                        "label": (
                            "y(t) = \\left(0.0004 + 0.1*t\\right) + "
                            "exact_closure(sample)"
                        )
                    },
                },
                "benchmark": {
                    "status": "completed",
                    "benchmark_root": str(benchmark_root),
                    "report_path": str(report_path),
                    "task_result_path": str(task_result_path),
                    "telemetry_path": str(telemetry_path),
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
                            "candidate_ledger": [
                                {
                                    "reason_codes": [
                                        "descriptive_admissibility_failed"
                                    ],
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
                        },
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "analysis_path": analysis_path,
        "workspace_root": workspace_root,
        "operator_manifest_path": operator_manifest_path,
        "probabilistic_manifest_path": probabilistic_manifest_path,
        "report_path": report_path,
    }


def test_create_workbench_analysis_runs_real_runtime_with_fixture_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "euclid.workbench.service.fetch_fmp_eod_history",
        lambda **_: _fake_history(),
    )

    analysis = create_workbench_analysis(
        symbol="SPY",
        api_key="test-key",
        target_id="price_close",
        output_root=tmp_path / "workbench",
        project_root=PROJECT_ROOT,
        start_date="2025-01-08",
        end_date="2025-02-14",
        include_probabilistic=True,
        include_benchmark=True,
    )

    assert analysis["dataset"]["symbol"] == "SPY"
    assert analysis["dataset"]["target"]["id"] == "price_close"
    assert analysis["dataset"]["rows"] >= 20
    assert analysis["request"]["start_date"] == "2025-01-08"
    assert analysis["request"]["end_date"] == "2025-02-14"
    assert "row_limit" not in analysis["request"]
    assert analysis["operator_point"]["status"] == "completed"
    assert analysis["operator_point"]["selected_family"]
    assert analysis["operator_point"]["replay_verification"] == "verified"
    assert analysis["operator_point"]["equation"]["structure_signature"]
    assert "delta_form_label" in analysis["operator_point"]["equation"]
    assert analysis["operator_point"]["chart"]["actual_series"]
    assert analysis["probabilistic"]["distribution"]["status"] in {
        "completed",
        "failed",
    }
    if analysis["probabilistic"]["distribution"]["status"] == "completed":
        assert analysis["probabilistic"]["distribution"]["rows"]
        assert analysis["probabilistic"]["distribution"]["calibration"]["status"] in {
            "passed",
            "failed",
        }
    else:
        assert analysis["probabilistic"]["distribution"]["error"]["message"]
    assert analysis["benchmark"]["status"] == "completed"
    assert analysis["benchmark"]["submitters"]
    assert analysis["benchmark"]["portfolio_selection"]["winner_submitter_id"]
    assert analysis["descriptive_fit"]["status"] == "completed"
    assert analysis["descriptive_fit"]["source"] == "benchmark_local_selection"
    assert analysis["descriptive_fit"]["equation"]["label"]
    assert "delta_form_label" in analysis["descriptive_fit"]["equation"]
    assert analysis["descriptive_fit"]["chart"]["equation_curve"]
    assert Path(analysis["workspace_root"]).is_dir()


def test_saved_analysis_reload_preserves_no_winner_daily_return_fixture(
    tmp_path: Path,
) -> None:
    fixture_paths = _write_saved_daily_return_analysis_fixture(tmp_path)
    handler = SimpleNamespace(
        output_root=tmp_path,
        _attach_cached_workbench_explanations=lambda analysis: dict(analysis),
    )
    payload = WorkbenchRequestHandler._load_saved_analysis(
        handler,
        str(fixture_paths["analysis_path"]),
    )

    selection = payload["benchmark"]["portfolio_selection"]

    assert payload["analysis_path"] == str(fixture_paths["analysis_path"])
    assert payload["workspace_root"] == str(fixture_paths["workspace_root"])
    assert payload["dataset"]["target"]["id"] == "daily_return"
    assert payload["dataset"]["rows"] == 3
    assert payload["operator_point"]["status"] == "completed"
    assert payload["operator_point"]["manifest_path"] == str(
        fixture_paths["operator_manifest_path"]
    )
    assert payload["operator_point"]["publication"]["status"] == "abstained"
    assert payload["operator_point"]["publication"]["reason_codes"] == [
        "robustness_failed",
        "perturbation_protocol_failed",
    ]
    assert payload["probabilistic"]["distribution"]["status"] == "failed"
    assert payload["probabilistic"]["distribution"]["manifest_path"] == str(
        fixture_paths["probabilistic_manifest_path"]
    )
    assert payload["probabilistic"]["interval"]["status"] == "failed"
    assert payload["benchmark"]["report_path"] == str(fixture_paths["report_path"])
    assert payload["benchmark"]["local_winner_submitter_id"] is None
    assert payload["benchmark"]["local_winner_candidate_id"] is None
    assert selection["status"] == "completed"
    assert selection.get("winner_submitter_id") is None
    assert selection.get("winner_candidate_id") is None
    assert selection.get("selection_explanation") is None
    assert selection["selection_explanation_raw"] is None
    assert selection["decision_trace"][0]["message"] == (
        "No admissible finalist was selected"
    )
    assert payload["benchmark"]["descriptive_fit_status"]["status"] == (
        "absent_no_admissible_candidate"
    )
    assert payload.get("descriptive_fit") is None
    assert payload.get("claim_class") is None
    assert payload.get("predictive_law") is None
    assert payload.get("holistic_equation") is None
    assert payload.get("would_have_abstained_because") == [
        "robustness_failed",
        "perturbation_protocol_failed",
    ]
    assert payload.get("gap_report") == [
        "operator_not_publishable",
        "no_backend_joint_claim",
        "requires_exact_sample_closure",
    ]
    assert payload.get("not_holistic_because") == payload.get("gap_report")


def test_saved_analysis_reload_preserves_explicit_abstained_operator_publication(
    tmp_path: Path,
) -> None:
    fixture_paths = _write_saved_daily_return_analysis_fixture(tmp_path)
    analysis_path = fixture_paths["analysis_path"]
    saved_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    saved_payload["operator_point"].pop("abstention", None)
    saved_payload["operator_point"].update(
        {
            "result_mode": "candidate_publication",
            "claim_card_ref": "claim_card_manifest@1.1.0:claim-card-1",
            "scorecard_ref": "scorecard_manifest@1.1.0:scorecard-1",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "publication": {
                "status": "abstained",
                "headline": "Saved publication state recorded an abstained point claim.",
                "reason_codes": ["saved_explicit_abstention"],
            },
            "claim_card": {
                "claim_type": "predictively_supported",
                "claim_ceiling": "predictively_supported",
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
        }
    )
    analysis_path.write_text(
        json.dumps(saved_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    handler = SimpleNamespace(
        output_root=tmp_path,
        _attach_cached_workbench_explanations=lambda analysis: dict(analysis),
    )
    payload = WorkbenchRequestHandler._load_saved_analysis(
        handler,
        str(analysis_path),
    )

    assert payload["operator_point"]["publication"]["status"] == "abstained"
    assert payload["operator_point"]["publication"]["reason_codes"] == [
        "saved_explicit_abstention"
    ]
    assert payload["predictive_law"] is None
    assert payload["claim_class"] is None
    assert payload["publishable"] is False


def test_saved_analysis_reload_rejects_publishable_exact_closure_predictive_law(
    tmp_path: Path,
) -> None:
    fixture_paths = _write_saved_daily_return_analysis_fixture(tmp_path)
    analysis_path = fixture_paths["analysis_path"]
    saved_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    saved_payload["operator_point"].pop("abstention", None)
    saved_payload["operator_point"].update(
        {
            "result_mode": "candidate_publication",
            "claim_card_ref": "claim_card_manifest@1.1.0:claim-card-1",
            "scorecard_ref": "scorecard_manifest@1.1.0:scorecard-1",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "equation": {
                "candidate_id": "analytic_exact_closure_surface",
                "family_id": "analytic",
                "exactness": "sample_exact_closure",
                "label": "y(t) = exact_closure(sample)",
                "curve": [
                    {
                        "event_time": "2026-04-14T00:00:00Z",
                        "fitted_value": 0.0015,
                    },
                    {
                        "event_time": "2026-04-15T00:00:00Z",
                        "fitted_value": -0.0020,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 0.0045,
                    },
                ],
            },
            "claim_card": {
                "claim_type": "predictively_supported",
                "claim_ceiling": "predictively_supported",
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
        }
    )
    analysis_path.write_text(
        json.dumps(saved_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    handler = SimpleNamespace(
        output_root=tmp_path,
        _attach_cached_workbench_explanations=lambda analysis: dict(analysis),
    )
    payload = WorkbenchRequestHandler._load_saved_analysis(
        handler,
        str(analysis_path),
    )

    assert payload["operator_point"]["publication"]["status"] == "publishable"
    assert payload["predictive_law"] is None
    assert payload["claim_class"] is None
    assert payload["publishable"] is False


def test_saved_analysis_reload_rejects_publishable_delta_only_predictive_law(
    tmp_path: Path,
) -> None:
    fixture_paths = _write_saved_daily_return_analysis_fixture(tmp_path)
    analysis_path = fixture_paths["analysis_path"]
    saved_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    saved_payload["operator_point"].pop("abstention", None)
    saved_payload["operator_point"].update(
        {
            "result_mode": "candidate_publication",
            "claim_card_ref": "claim_card_manifest@1.1.0:claim-card-1",
            "scorecard_ref": "scorecard_manifest@1.1.0:scorecard-1",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "parameter_summary": {
                    "intercept": 0.0018,
                    "lag_coefficient": 0.92,
                },
            },
            "claim_card": {
                "claim_type": "predictively_supported",
                "claim_ceiling": "predictively_supported",
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
        }
    )
    analysis_path.write_text(
        json.dumps(saved_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    handler = SimpleNamespace(
        output_root=tmp_path,
        _attach_cached_workbench_explanations=lambda analysis: dict(analysis),
    )
    payload = WorkbenchRequestHandler._load_saved_analysis(
        handler,
        str(analysis_path),
    )

    assert payload["operator_point"]["publication"]["status"] == "publishable"
    assert payload["operator_point"]["equation"]["candidate_id"] == "analytic_lag1_affine"
    assert payload["operator_point"]["equation"]["family_id"] == "analytic"
    assert payload["operator_point"]["equation"]["delta_form_label"] == (
        "Δy(t) = 0.0018 - 0.08*y(t-1)"
    )
    assert payload["predictive_law"] is None
    assert payload["claim_class"] is None
    assert payload["publishable"] is False


def test_saved_analysis_reload_does_not_infer_holistic_from_saved_alignment(
    tmp_path: Path,
) -> None:
    fixture_paths = _write_saved_daily_return_analysis_fixture(tmp_path)
    analysis_path = fixture_paths["analysis_path"]
    saved_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    saved_payload["operator_point"].pop("abstention", None)
    saved_payload["operator_point"].update(
        {
            "result_mode": "candidate_publication",
            "claim_card_ref": "claim_card_manifest@1.1.0:claim-card-1",
            "scorecard_ref": "scorecard_manifest@1.1.0:scorecard-1",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "claim_card": {
                "claim_type": "mechanistically_compatible_hypothesis",
                "claim_ceiling": "mechanistically_compatible_hypothesis",
                "predictive_support_status": "confirmatory_supported",
                "allowed_interpretation_codes": [
                    "historical_structure_summary",
                    "point_forecast_within_declared_validation_scope",
                    "mechanism_claim",
                ],
            },
            "scorecard": {
                "descriptive_status": "passed",
                "predictive_status": "passed",
            },
        }
    )
    saved_payload["probabilistic"]["distribution"] = {
        "status": "completed",
        "manifest_path": str(fixture_paths["probabilistic_manifest_path"]),
        "selected_family": "analytic",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": (
            "publication_record_manifest@1.1.0:publication-1"
        ),
        "rows": [
            {
                "available_at": "2026-04-16T21:00:00Z",
                "origin_time": "2026-04-15T00:00:00Z",
                "horizon": 1,
                "location": 0.0025,
                "scale": 0.0035,
                "realized_observation": 0.0045,
            }
        ],
        "latest_row": {
            "available_at": "2026-04-16T21:00:00Z",
            "origin_time": "2026-04-15T00:00:00Z",
            "horizon": 1,
            "location": 0.0025,
            "scale": 0.0035,
            "realized_observation": 0.0045,
        },
        "calibration": {
            "status": "passed",
            "passed": True,
            "gate_effect": "publishable",
            "diagnostics": [
                {
                    "diagnostic_id": "pit_or_randomized_pit_uniformity",
                    "sample_size": 6,
                    "status": "passed",
                }
            ],
        },
        "chart": {"forecast_bands": []},
    }
    saved_payload["holistic_equation"] = {
        "status": "completed",
        "claim_class": "holistic_equation",
        "deterministic_source": "predictive_law",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": (
            "publication_record_manifest@1.1.0:publication-1"
        ),
        "honesty_note": (
            "Backend published a deterministic-plus-uncertainty claim within "
            "the declared validation scope."
        ),
        "equation": {
            "curve": [
                {
                    "event_time": "2026-04-14T00:00:00Z",
                    "fitted_value": 0.0015,
                },
                {
                    "event_time": "2026-04-15T00:00:00Z",
                    "fitted_value": -0.0020,
                },
                {
                    "event_time": "2026-04-16T00:00:00Z",
                    "fitted_value": 0.0045,
                },
            ],
        },
    }
    analysis_path.write_text(
        json.dumps(saved_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    handler = SimpleNamespace(
        output_root=tmp_path,
        _attach_cached_workbench_explanations=lambda analysis: dict(analysis),
    )
    payload = WorkbenchRequestHandler._load_saved_analysis(
        handler,
        str(analysis_path),
    )

    assert payload["predictive_law"] is not None
    assert payload["predictive_law"]["evidence_summary"]["claim_card"] == {
        "ref": "claim_card_manifest@1.1.0:claim-card-1",
        "claim_type": "mechanistically_compatible_hypothesis",
        "claim_ceiling": "mechanistically_compatible_hypothesis",
        "predictive_support_status": "confirmatory_supported",
        "allowed_interpretation_codes": [
            "historical_structure_summary",
            "point_forecast_within_declared_validation_scope",
            "mechanism_claim",
        ],
    }
    assert payload["holistic_equation"] is None
    assert payload["uncertainty_attachment"] is None
    assert payload["claim_class"] == "predictive_law"
    assert payload["publishable"] is True
    assert payload["not_holistic_because"] == payload["gap_report"]


def test_saved_analysis_reload_does_not_reconstruct_failed_backend_holistic_claim(
    tmp_path: Path,
) -> None:
    fixture_paths = _write_saved_daily_return_analysis_fixture(tmp_path)
    analysis_path = fixture_paths["analysis_path"]
    saved_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    saved_payload["operator_point"].pop("abstention", None)
    saved_payload["operator_point"].update(
        {
            "result_mode": "candidate_publication",
            "claim_card_ref": "claim_card_manifest@1.1.0:claim-card-1",
            "scorecard_ref": "scorecard_manifest@1.1.0:scorecard-1",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "claim_card": {
                "claim_type": "mechanistically_compatible_hypothesis",
                "claim_ceiling": "mechanistically_compatible_hypothesis",
                "predictive_support_status": "confirmatory_supported",
                "allowed_interpretation_codes": [
                    "historical_structure_summary",
                    "point_forecast_within_declared_validation_scope",
                    "mechanism_claim",
                ],
            },
            "scorecard": {
                "descriptive_status": "passed",
                "predictive_status": "passed",
            },
        }
    )
    saved_payload["probabilistic"]["distribution"] = {
        "status": "completed",
        "manifest_path": str(fixture_paths["probabilistic_manifest_path"]),
        "selected_family": "analytic",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": (
            "publication_record_manifest@1.1.0:publication-1"
        ),
        "rows": [
            {
                "available_at": "2026-04-16T21:00:00Z",
                "origin_time": "2026-04-15T00:00:00Z",
                "horizon": 1,
                "location": 0.0025,
                "scale": 0.0035,
                "realized_observation": 0.0045,
            }
        ],
        "latest_row": {
            "available_at": "2026-04-16T21:00:00Z",
            "origin_time": "2026-04-15T00:00:00Z",
            "horizon": 1,
            "location": 0.0025,
            "scale": 0.0035,
            "realized_observation": 0.0045,
        },
        "calibration": {
            "status": "passed",
            "passed": True,
            "gate_effect": "publishable",
            "diagnostics": [
                {
                    "diagnostic_id": "pit_or_randomized_pit_uniformity",
                    "sample_size": 6,
                    "status": "passed",
                }
            ],
        },
        "chart": {"forecast_bands": []},
    }
    saved_payload["holistic_equation"] = {
        "status": "failed",
        "claim_class": "holistic_equation",
        "deterministic_source": "predictive_law",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": (
            "publication_record_manifest@1.1.0:publication-1"
        ),
        "honesty_note": (
            "Failed backend holistic payload must not be reconstructed as "
            "completed."
        ),
        "equation": {
            "curve": [
                {
                    "event_time": "2026-04-14T00:00:00Z",
                    "fitted_value": 0.0015,
                },
                {
                    "event_time": "2026-04-15T00:00:00Z",
                    "fitted_value": -0.0020,
                },
                {
                    "event_time": "2026-04-16T00:00:00Z",
                    "fitted_value": 0.0045,
                },
            ],
        },
    }
    analysis_path.write_text(
        json.dumps(saved_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    handler = SimpleNamespace(
        output_root=tmp_path,
        _attach_cached_workbench_explanations=lambda analysis: dict(analysis),
    )
    payload = WorkbenchRequestHandler._load_saved_analysis(
        handler,
        str(analysis_path),
    )

    assert payload["predictive_law"] is not None
    assert payload["holistic_equation"] is None
    assert payload["uncertainty_attachment"] is None
    assert payload["claim_class"] == "predictive_law"
    assert payload["publishable"] is True
    assert payload["not_holistic_because"] == payload["gap_report"]


def test_saved_analysis_reload_projects_workflow_native_uncertainty_attachment(
    tmp_path: Path,
) -> None:
    fixture_paths = _write_saved_daily_return_analysis_fixture(tmp_path)
    analysis_path = fixture_paths["analysis_path"]
    saved_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    saved_payload["operator_point"].pop("abstention", None)
    saved_payload["operator_point"].update(
        {
            "result_mode": "candidate_publication",
            "claim_card_ref": "claim_card_manifest@1.1.0:claim-card-1",
            "scorecard_ref": "scorecard_manifest@1.1.0:scorecard-1",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "claim_card": {
                "claim_type": "mechanistically_compatible_hypothesis",
                "claim_ceiling": "mechanistically_compatible_hypothesis",
                "predictive_support_status": "confirmatory_supported",
                "allowed_interpretation_codes": [
                    "historical_structure_summary",
                    "point_forecast_within_declared_validation_scope",
                    "mechanism_claim",
                ],
                "uncertainty_attachment": {
                    "status": "completed",
                    "joint_claim_gate": {
                        "backend_authored": True,
                        "status": "accepted",
                    },
                    "deterministic_source": "predictive_law",
                    "probabilistic_source": "distribution",
                    "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
                    "publication_record_ref": (
                        "publication_record_manifest@1.1.0:publication-1"
                    ),
                },
            },
            "scorecard": {
                "descriptive_status": "passed",
                "predictive_status": "passed",
            },
        }
    )
    saved_payload["probabilistic"]["distribution"] = {
        "status": "completed",
        "manifest_path": str(fixture_paths["probabilistic_manifest_path"]),
        "selected_family": "analytic",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": (
            "publication_record_manifest@1.1.0:publication-1"
        ),
        "rows": [
            {
                "available_at": "2026-04-16T21:00:00Z",
                "origin_time": "2026-04-15T00:00:00Z",
                "horizon": 1,
                "location": 0.0025,
                "scale": 0.0035,
                "realized_observation": 0.0045,
            }
        ],
        "latest_row": {
            "available_at": "2026-04-16T21:00:00Z",
            "origin_time": "2026-04-15T00:00:00Z",
            "horizon": 1,
            "location": 0.0025,
            "scale": 0.0035,
            "realized_observation": 0.0045,
        },
        "calibration": {
            "status": "passed",
            "passed": True,
            "gate_effect": "publishable",
            "diagnostics": [
                {
                    "diagnostic_id": "pit_or_randomized_pit_uniformity",
                    "sample_size": 6,
                    "status": "passed",
                }
            ],
        },
        "chart": {"forecast_bands": []},
    }
    analysis_path.write_text(
        json.dumps(saved_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    handler = SimpleNamespace(
        output_root=tmp_path,
        _attach_cached_workbench_explanations=lambda analysis: dict(analysis),
    )
    payload = WorkbenchRequestHandler._load_saved_analysis(
        handler,
        str(analysis_path),
    )

    assert payload["predictive_law"] is not None
    assert payload["uncertainty_attachment"] == {
        "status": "completed",
        "deterministic_source": "predictive_law",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
    }


@pytest.mark.parametrize(
    "equation_payload",
    [
        {
            "curve": [
                {
                    "event_time": "2026-04-14T00:00:00Z",
                    "fitted_value": 0.0015,
                },
                {
                    "event_time": "2026-04-15T00:00:00Z",
                    "fitted_value": -0.0020,
                },
                {
                    "event_time": "2026-04-16T00:00:00Z",
                    "fitted_value": 0.0045,
                },
            ],
            "exactness": "sample_exact_closure",
        },
        {
            "curve": [
                {
                    "event_time": "2026-04-14T00:00:00Z",
                    "fitted_value": 0.0015,
                },
                {
                    "event_time": "2026-04-15T00:00:00Z",
                    "fitted_value": -0.0020,
                },
                {
                    "event_time": "2026-04-16T00:00:00Z",
                    "fitted_value": 0.0045,
                },
            ],
            "composition_operator": "additive_residual",
        },
        {
            "curve": [
                {
                    "event_time": "2026-04-14T00:00:00Z",
                    "fitted_value": 0.0015,
                },
                {
                    "event_time": "2026-04-15T00:00:00Z",
                    "fitted_value": -0.0020,
                },
                {
                    "event_time": "2026-04-16T00:00:00Z",
                    "fitted_value": 0.0045,
                },
            ],
            "candidate_id": "symbolic_synthesis_candidate",
        },
    ],
)
def test_saved_analysis_reload_rejects_holistic_equation_with_banned_equation_markers(
    tmp_path: Path,
    equation_payload: dict[str, object],
) -> None:
    fixture_paths = _write_saved_daily_return_analysis_fixture(tmp_path)
    analysis_path = fixture_paths["analysis_path"]
    saved_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    saved_payload["operator_point"].pop("abstention", None)
    saved_payload["operator_point"].update(
        {
            "result_mode": "candidate_publication",
            "claim_card_ref": "claim_card_manifest@1.1.0:claim-card-1",
            "scorecard_ref": "scorecard_manifest@1.1.0:scorecard-1",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": (
                "publication_record_manifest@1.1.0:publication-1"
            ),
            "claim_card": {
                "claim_type": "mechanistically_compatible_hypothesis",
                "claim_ceiling": "mechanistically_compatible_hypothesis",
                "predictive_support_status": "confirmatory_supported",
                "allowed_interpretation_codes": [
                    "historical_structure_summary",
                    "point_forecast_within_declared_validation_scope",
                    "mechanism_claim",
                ],
            },
            "scorecard": {
                "descriptive_status": "passed",
                "predictive_status": "passed",
            },
        }
    )
    saved_payload["probabilistic"]["distribution"] = {
        "status": "completed",
        "manifest_path": str(fixture_paths["probabilistic_manifest_path"]),
        "selected_family": "analytic",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": (
            "publication_record_manifest@1.1.0:publication-1"
        ),
        "rows": [
            {
                "available_at": "2026-04-16T21:00:00Z",
                "origin_time": "2026-04-15T00:00:00Z",
                "horizon": 1,
                "location": 0.0025,
                "scale": 0.0035,
                "realized_observation": 0.0045,
            }
        ],
        "latest_row": {
            "available_at": "2026-04-16T21:00:00Z",
            "origin_time": "2026-04-15T00:00:00Z",
            "horizon": 1,
            "location": 0.0025,
            "scale": 0.0035,
            "realized_observation": 0.0045,
        },
        "calibration": {
            "status": "passed",
            "passed": True,
            "gate_effect": "publishable",
            "diagnostics": [
                {
                    "diagnostic_id": "pit_or_randomized_pit_uniformity",
                    "sample_size": 6,
                    "status": "passed",
                }
            ],
        },
        "chart": {"forecast_bands": []},
    }
    saved_payload["holistic_equation"] = {
        "status": "completed",
        "claim_class": "holistic_equation",
        "joint_claim_gate": {
            "backend_authored": True,
            "status": "accepted",
        },
        "deterministic_source": "predictive_law",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": (
            "publication_record_manifest@1.1.0:publication-1"
        ),
        "honesty_note": (
            "Equation-level banned markers must veto stale holistic claims on "
            "saved-analysis reload."
        ),
        "equation": equation_payload,
    }
    analysis_path.write_text(
        json.dumps(saved_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    handler = SimpleNamespace(
        output_root=tmp_path,
        _attach_cached_workbench_explanations=lambda analysis: dict(analysis),
    )
    payload = WorkbenchRequestHandler._load_saved_analysis(
        handler,
        str(analysis_path),
    )

    assert payload["predictive_law"] is not None
    assert payload["holistic_equation"] is None
    assert payload["uncertainty_attachment"] is None
    assert payload["claim_class"] == "predictive_law"
