from __future__ import annotations

from collections import Counter
import csv
import json
import math
import pickle
import statistics
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from euclid.benchmarks import profile_benchmark_task
from euclid.benchmarks.submitters import (
    BenchmarkSubmitterResult,
    PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
)
from euclid.fmp_smoke import (
    PROBABILISTIC_FORECAST_OBJECT_TYPES,
    build_benchmark_task_manifest,
    build_csv_rows_from_fmp_history,
    build_operator_manifest,
    fetch_fmp_eod_history,
    write_euclid_dataset_csv,
    write_yaml_manifest,
)
from euclid.inspection import (
    compare_demo_baseline,
    inspect_demo_calibration,
    inspect_demo_point_prediction,
    inspect_demo_probabilistic_prediction,
    load_demo_run_artifact_graph,
)
from euclid.operator_runtime import replay_operator, run_operator
from euclid.operator_runtime.resources import resolve_asset_root
from euclid.testing.live_api import NON_CLAIM_EVIDENCE_BOUNDARY
from euclid.testing.redaction import redact_mapping

DEFAULT_TARGET_ID = "daily_return"
DEFAULT_DATE_RANGE_YEARS = 5
CHANGE_ATLAS_HORIZONS = (1, 5, 21)
_HOLISTIC_INLINE_COEFFICIENT_LIMIT = 32
_HOLISTIC_PREFERRED_LANES = (
    "distribution",
    "quantile",
    "interval",
    "event_probability",
)
_DESCRIPTIVE_SELECTION_SCOPE = "shared_planning_cir_only"
_DESCRIPTIVE_SELECTION_RULE = (
    "min_total_code_bits_then_max_description_gain_then_"
    "min_structure_code_bits_then_min_canonical_byte_length_then_candidate_id"
)
_RECONSTRUCTION_MAX_NORMALIZED_MAE = 0.25
_RECONSTRUCTION_MAX_NORMALIZED_MAX_ABS_ERROR = 0.5
_RECONSTRUCTION_MIN_SAMPLE_SIZE_FOR_VARIANCE_GATE = 20
_RECONSTRUCTION_MIN_R2_VS_MEAN_BASELINE = 0.5
_DESCRIPTIVE_RECONSTRUCTION_CANDIDATE_ID = "descriptive_fourier_reconstruction"
_DESCRIPTIVE_RECONSTRUCTION_SOURCE = "workbench_descriptive_reconstruction"
_DESCRIPTIVE_RECONSTRUCTION_REASON_CODE = (
    "explicit_time_reconstruction_descriptive_structure"
)
_DESCRIPTIVE_RECONSTRUCTION_TARGET_R2 = 0.97
_DESCRIPTIVE_RECONSTRUCTION_HARMONIC_LADDER = (
    16,
    32,
    64,
    128,
    256,
    384,
    512,
    608,
)
_PREDICTIVE_ALLOWED_INTERPRETATION = (
    "point_forecast_within_declared_validation_scope"
)
_PREDICTIVE_LAW_BANNED_CANDIDATE_ID_FRAGMENTS = (
    "exact_closure",
    "symbolic_synthesis",
    "holistic_",
)
_WORKBENCH_LIVE_EVIDENCE_KEYS = ("live_api_evidence", "live_evidence")
_INLINE_SECRET_VALUE_KEYS = frozenset(
    {
        "fmp_api_key",
        "openai_api_key",
        "api_key",
        "apikey",
        "authorization",
        "x-api-key",
        "api-key",
        "openai-api-key",
        "token",
        "access_token",
    }
)

TARGET_SPECS: dict[str, dict[str, Any]] = {
    "price_close": {
        "id": "price_close",
        "label": "Price Close",
        "description": "Predict the raw close level for each trading day.",
        "y_axis_label": "Close",
        "recommended": False,
        "recommended_reason": (
            "Raw levels are useful for replaying a published point path, but they "
            "often collapse into near-persistence equations."
        ),
        "analysis_note": (
            "Raw close levels usually reward persistence and drift. Treat any "
            "descriptive equation on this target as a level-tracking summary, not "
            "evidence of deeper market structure."
        ),
    },
    "daily_return": {
        "id": "daily_return",
        "label": "Daily Return",
        "description": "Predict the close-to-close fractional return for each day.",
        "y_axis_label": "Return",
        "recommended": True,
        "recommended_reason": (
            "Better default for interpretable descriptive equations because it makes "
            "level persistence explicit instead of hiding it inside raw prices."
        ),
        "analysis_note": (
            "Daily returns are the recommended default for analytical inspection. "
            "They make drift and persistence weaker, so descriptive fits have to earn "
            "their structure."
        ),
    },
    "log_return": {
        "id": "log_return",
        "label": "Log Return",
        "description": "Predict the log close-to-close return for each day.",
        "y_axis_label": "Log Return",
        "recommended": False,
        "recommended_reason": (
            "Good for scale-invariant analysis when you explicitly want additive "
            "return algebra."
        ),
        "analysis_note": (
            "Log returns emphasize multiplicative moves and make cross-regime "
            "comparisons easier than raw levels."
        ),
    },
    "next_day_up": {
        "id": "next_day_up",
        "label": "Next-Day Up",
        "description": "Predict whether the next close is above the previous close.",
        "y_axis_label": "Probability / Indicator",
        "recommended": False,
        "recommended_reason": (
            "Use when the decision is directional classification rather than a "
            "numeric path."
        ),
        "analysis_note": (
            "Directional targets are appropriate for event probabilities, but they "
            "do not support descriptive path equations."
        ),
    },
}

CHANGE_METRIC_SPECS: dict[str, dict[str, str]] = {
    "delta": {
        "id": "delta",
        "label": "Delta",
        "short_label": "Delta",
        "y_axis_label": "Delta",
    },
    "return": {
        "id": "return",
        "label": "Return",
        "short_label": "Return",
        "y_axis_label": "Return",
    },
    "log_return": {
        "id": "log_return",
        "label": "Log Return",
        "short_label": "Log Return",
        "y_axis_label": "Log Return",
    },
}

_CANONICAL_DATASET_COLUMNS = {
    "source_id",
    "series_id",
    "event_time",
    "available_at",
    "observed_value",
    "revision_id",
}


def ordered_target_specs() -> list[dict[str, Any]]:
    return [
        dict(spec)
        for spec in sorted(
            TARGET_SPECS.values(),
            key=lambda spec: (
                not bool(spec.get("recommended")),
                0 if spec.get("id") == DEFAULT_TARGET_ID else 1,
                str(spec.get("label") or spec.get("id") or ""),
            ),
        )
    ]


def default_workbench_date_range(*, today: date | None = None) -> dict[str, str]:
    resolved_today = today or date.today()
    return {
        "start_date": _shift_years(resolved_today, years=-DEFAULT_DATE_RANGE_YEARS).isoformat(),
        "end_date": resolved_today.isoformat(),
    }


def build_target_rows_from_history(
    history: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    target_id: str,
) -> list[dict[str, Any]]:
    if target_id not in TARGET_SPECS:
        supported = ", ".join(sorted(TARGET_SPECS))
        raise ValueError(
            f"unsupported target_id {target_id!r}; expected one of {supported}"
        )

    base_rows = build_csv_rows_from_fmp_history(history, symbol=symbol)
    if target_id == "price_close":
        return [dict(row) for row in base_rows]

    transformed: list[dict[str, Any]] = []
    for previous, current in zip(base_rows, base_rows[1:]):
        previous_close = float(previous["observed_value"])
        current_close = float(current["observed_value"])
        row = dict(current)
        row["previous_close"] = previous_close

        if target_id == "daily_return":
            row["observed_value"] = (current_close / previous_close) - 1.0
        elif target_id == "log_return":
            row["observed_value"] = math.log(current_close / previous_close)
        elif target_id == "next_day_up":
            row["observed_value"] = 1.0 if current_close > previous_close else 0.0
        else:  # pragma: no cover - guarded above
            raise AssertionError(f"unhandled target_id {target_id}")
        transformed.append(row)
    if not transformed:
        raise ValueError("target transform produced no rows")
    return transformed


def build_equation_summary(
    *,
    candidate_id: str,
    family_id: str,
    parameter_summary: Mapping[str, float | int],
    structure_signature: str | None,
    dataset_rows: Sequence[Mapping[str, Any]],
    composition_operator: str | None = None,
    composition_payload: Mapping[str, Any] | None = None,
    literals: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    label = _render_equation_label(
        candidate_id=candidate_id,
        family_id=family_id,
        parameter_summary=parameter_summary,
        literals=literals,
    )
    curve = _build_equation_curve(
        candidate_id=candidate_id,
        family_id=family_id,
        parameter_summary=parameter_summary,
        dataset_rows=dataset_rows,
        literals=literals,
        state=state,
        use_observed_lag_values=True,
    )
    intercept = _float_or_none(parameter_summary.get("intercept"))
    lag_coefficient = _float_or_none(parameter_summary.get("lag_coefficient"))
    return {
        "candidate_id": candidate_id,
        "family_id": family_id,
        "parameter_summary": {
            key: float(value) for key, value in parameter_summary.items()
        },
        "structure_signature": structure_signature,
        "composition_operator": composition_operator,
        "composition_payload": (
            dict(_jsonable(composition_payload))
            if isinstance(composition_payload, Mapping)
            else None
        ),
        "literals": dict(_jsonable(literals)) if isinstance(literals, Mapping) else {},
        "state": dict(_jsonable(state)) if isinstance(state, Mapping) else {},
        "label": label,
        "delta_form_label": _delta_form_label(
            candidate_id=candidate_id,
            family_id=family_id,
            intercept=intercept,
            lag_coefficient=lag_coefficient,
        ),
        "curve": curve,
        "render_status": "formula_supported" if label is not None else "structure_only",
    }


def _build_descriptive_reconstruction(
    *,
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if len(dataset_rows) < 8:
        return None
    observed_values = [
        _float_or_none(row.get("observed_value"))
        for row in dataset_rows
    ]
    if any(value is None or not math.isfinite(float(value)) for value in observed_values):
        return None
    resolved_values = [float(value) for value in observed_values if value is not None]
    candidate_records: list[dict[str, Any]] = []
    for harmonic_count in _descriptive_reconstruction_harmonic_ladder(
        len(resolved_values)
    ):
        fit = _fit_descriptive_reconstruction_fourier(
            observed_values=resolved_values,
            harmonic_count=harmonic_count,
        )
        equation = build_equation_summary(
            candidate_id=_DESCRIPTIVE_RECONSTRUCTION_CANDIDATE_ID,
            family_id="analytic",
            parameter_summary=fit["parameter_summary"],
            structure_signature=(
                "workbench_descriptive_reconstruction_fourier"
                f"@harmonics={fit['harmonic_count']}"
            ),
            dataset_rows=dataset_rows,
            literals=fit["literals"],
        )
        equation["coefficient_vector_label"] = (
            f"K={fit['harmonic_count']}, "
            f"N={fit['sample_size']}, "
            "coefficients stored in equation.literals"
        )
        reconstruction_metrics = _descriptive_fit_reconstruction_metrics(
            descriptive_fit={
                "candidate_id": _DESCRIPTIVE_RECONSTRUCTION_CANDIDATE_ID,
                "family_id": "analytic",
                "equation": equation,
                "chart": {"equation_curve": equation["curve"]},
            },
            dataset_rows=dataset_rows,
        )
        candidate_records.append(
            {
                "equation": equation,
                "reconstruction_metrics": reconstruction_metrics,
                "harmonic_count": fit["harmonic_count"],
                "sample_size": fit["sample_size"],
            }
        )

    if not candidate_records:
        return None

    qualifying = [
        record
        for record in candidate_records
        if _descriptive_reconstruction_clears_target(
            record["reconstruction_metrics"]
        )
    ]
    selected_record = (
        qualifying[0]
        if qualifying
        else min(
            candidate_records,
            key=lambda record: _descriptive_reconstruction_sort_key(
                record["reconstruction_metrics"],
                harmonic_count=int(record["harmonic_count"]),
            ),
        )
    )
    target_cleared = _descriptive_reconstruction_clears_target(
        selected_record["reconstruction_metrics"]
    )
    harmonic_count = int(selected_record["harmonic_count"])
    sample_size = int(selected_record["sample_size"])
    honesty_note = (
        "Workbench-generated descriptive reconstruction from an explicit time-index "
        f"Fourier series with {harmonic_count} retained harmonics over {sample_size} "
        "observations. This is descriptive-only, not an operator publication, not a "
        "benchmark-local compact winner, and not a predictive-within-scope claim."
    )
    if not target_cleared:
        honesty_note += (
            " No candidate in the non-exact harmonic ladder cleared the target "
            "reconstruction floor, so the workbench is showing the best available "
            "non-exact reconstruction instead."
        )
    return {
        "status": "completed",
        "claim_class": "descriptive_reconstruction",
        "source": _DESCRIPTIVE_RECONSTRUCTION_SOURCE,
        "candidate_id": _DESCRIPTIVE_RECONSTRUCTION_CANDIDATE_ID,
        "family_id": "analytic",
        "selection_scope": "full_sample_explicit_time_basis",
        "selection_rule": (
            "min_harmonic_count_subject_to_reconstruction_r2_floor_then_"
            "max_r2_then_min_normalized_mae"
        ),
        "is_law_claim": False,
        "law_eligible": False,
        "law_rejection_reason_codes": [_DESCRIPTIVE_RECONSTRUCTION_REASON_CODE],
        "honesty_note": honesty_note,
        "equation": selected_record["equation"],
        "chart": {
            "actual_series": _actual_series(dataset_rows),
            "equation_curve": selected_record["equation"]["curve"],
        },
        "reconstruction_metrics": selected_record["reconstruction_metrics"],
        "target_floor_cleared": target_cleared,
    }


def _descriptive_reconstruction_harmonic_ladder(
    sample_size: int,
) -> tuple[int, ...]:
    max_harmonic = max(1, (int(sample_size) // 2) - 2)
    ladder = [
        harmonic
        for harmonic in _DESCRIPTIVE_RECONSTRUCTION_HARMONIC_LADDER
        if harmonic <= max_harmonic
    ]
    ladder.append(max_harmonic)
    return tuple(sorted({max(1, int(harmonic)) for harmonic in ladder}))


def _fit_descriptive_reconstruction_fourier(
    *,
    observed_values: Sequence[float],
    harmonic_count: int,
) -> dict[str, Any]:
    sample_size = len(observed_values)
    max_supported_harmonic = max(
        1,
        min(int(harmonic_count), max(1, (sample_size // 2) - 2)),
    )
    spectrum = np.fft.rfft(np.asarray(observed_values, dtype=float))
    retained = spectrum.copy()
    retained[max_supported_harmonic + 1 :] = 0
    cosine_coefficients: list[float] = []
    sine_coefficients: list[float] = []
    for harmonic in range(1, max_supported_harmonic + 1):
        coefficient = retained[harmonic]
        scale = (
            1.0 / sample_size
            if sample_size % 2 == 0 and harmonic == sample_size // 2
            else 2.0 / sample_size
        )
        cosine_coefficients.append(float(scale * coefficient.real))
        sine_coefficients.append(float(-scale * coefficient.imag))
    return {
        "harmonic_count": max_supported_harmonic,
        "sample_size": sample_size,
        "parameter_summary": {
            "mean_term": float(retained[0].real / sample_size),
            "harmonic_count": float(max_supported_harmonic),
            "sample_size": float(sample_size),
        },
        "literals": {
            "harmonic_count": max_supported_harmonic,
            "sample_size": sample_size,
            "cosine_coefficients": cosine_coefficients,
            "sine_coefficients": sine_coefficients,
        },
    }


def _descriptive_reconstruction_clears_target(
    reconstruction_metrics: Mapping[str, Any],
) -> bool:
    r2 = _float_or_none(reconstruction_metrics.get("r2_vs_mean_baseline"))
    return r2 is not None and r2 >= _DESCRIPTIVE_RECONSTRUCTION_TARGET_R2


def _descriptive_reconstruction_sort_key(
    reconstruction_metrics: Mapping[str, Any],
    *,
    harmonic_count: int,
) -> tuple[float, float, float, int]:
    r2 = _float_or_none(reconstruction_metrics.get("r2_vs_mean_baseline"))
    normalized_mae = _float_or_none(reconstruction_metrics.get("normalized_mae"))
    normalized_max_abs_error = _float_or_none(
        reconstruction_metrics.get("normalized_max_abs_error")
    )
    return (
        -(r2 if r2 is not None else float("-inf")),
        normalized_mae if normalized_mae is not None else float("inf"),
        (
            normalized_max_abs_error
            if normalized_max_abs_error is not None
            else float("inf")
        ),
        int(harmonic_count),
    )


def create_workbench_analysis(
    *,
    symbol: str,
    api_key: str,
    target_id: str,
    output_root: Path,
    project_root: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    benchmark_workers: int = 1,
    include_probabilistic: bool = True,
    include_benchmark: bool = True,
) -> dict[str, Any]:
    resolved_project_root = (
        project_root.resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    workspace_root = _workspace_root(
        output_root=output_root,
        symbol=symbol,
        target_id=target_id,
    )
    validated_start_date, validated_end_date = _validated_requested_date_range(
        start_date=start_date,
        end_date=end_date,
    )
    raw_history = fetch_fmp_eod_history(symbol=symbol, api_key=api_key)
    filtered_history = _slice_history(
        raw_history,
        start_date=validated_start_date,
        end_date=validated_end_date,
    )
    dataset_rows = build_target_rows_from_history(
        filtered_history,
        symbol=symbol,
        target_id=target_id,
    )
    if len(dataset_rows) < 8:
        raise ValueError(
            "Euclid workbench needs at least 8 observations after the target "
            "transform. "
            f"Received {len(dataset_rows)}."
        )

    raw_history_path = workspace_root / "fmp-history.json"
    raw_history_path.parent.mkdir(parents=True, exist_ok=True)
    raw_history_path.write_text(
        json.dumps(filtered_history, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    dataset_csv = write_euclid_dataset_csv(
        dataset_rows,
        destination=workspace_root
        / "datasets"
        / f"{_slug(symbol)}-{_slug(target_id)}.csv",
    )

    analysis: dict[str, Any] = {
        "analysis_version": "1.0.0",
        "workspace_root": str(workspace_root),
        "request": {
            "symbol": symbol.upper(),
            "target_id": target_id,
            "start_date": validated_start_date,
            "end_date": validated_end_date,
            "include_probabilistic": include_probabilistic,
            "include_benchmark": include_benchmark,
        },
        "dataset": _build_dataset_summary(
            symbol=symbol,
            target_id=target_id,
            dataset_rows=dataset_rows,
            dataset_csv=dataset_csv,
            raw_history_path=raw_history_path,
        ),
    }

    try:
        analysis["operator_point"] = _run_point_analysis(
            project_root=resolved_project_root,
            workspace_root=workspace_root,
            dataset_rows=dataset_rows,
            dataset_csv=dataset_csv,
            symbol=symbol,
            target_id=target_id,
        )
    except Exception as exc:  # pragma: no cover - exercised in live/manual runs
        analysis["operator_point"] = _failure_payload(exc)

    if include_probabilistic:
        probabilistic: dict[str, Any] = {}
        for mode in PROBABILISTIC_FORECAST_OBJECT_TYPES:
            try:
                probabilistic[mode] = _run_probabilistic_analysis(
                    project_root=resolved_project_root,
                    workspace_root=workspace_root,
                    dataset_rows=dataset_rows,
                    dataset_csv=dataset_csv,
                    symbol=symbol,
                    target_id=target_id,
                    forecast_object_type=mode,
                )
            except Exception as exc:  # pragma: no cover - exercised in live/manual runs
                probabilistic[mode] = _failure_payload(exc)
        analysis["probabilistic"] = probabilistic

    if include_benchmark:
        try:
            benchmark_analysis = _run_benchmark_analysis(
                project_root=resolved_project_root,
                workspace_root=workspace_root,
                dataset_rows=dataset_rows,
                symbol=symbol,
                benchmark_workers=benchmark_workers,
            )
            analysis["benchmark"] = benchmark_analysis
            if isinstance(benchmark_analysis.get("descriptive_fit"), Mapping):
                analysis["descriptive_fit"] = dict(
                    benchmark_analysis["descriptive_fit"]
                )
        except Exception as exc:  # pragma: no cover - exercised in live/manual runs
            analysis["benchmark"] = _failure_payload(exc)

    descriptive_reconstruction = _build_descriptive_reconstruction(
        dataset_rows=dataset_rows,
    )
    if descriptive_reconstruction is not None:
        analysis["descriptive_reconstruction"] = descriptive_reconstruction

    analysis_path = workspace_root / "analysis.json"
    analysis["analysis_path"] = str(analysis_path)
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return analysis


def list_recent_analyses(*, output_root: Path, limit: int = 8) -> list[dict[str, Any]]:
    candidates = sorted(
        (output_root.resolve()).glob("*/analysis.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    recent: list[dict[str, Any]] = []
    for analysis_path in candidates[: max(1, limit)]:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        recent.append(
            {
                "analysis_path": str(analysis_path),
                "workspace_root": str(analysis_path.parent),
                "symbol": payload.get("dataset", {}).get("symbol"),
                "target_id": payload.get("dataset", {}).get("target", {}).get("id"),
                "created_at": analysis_path.parent.name.split("-")[0],
            }
        )
    return recent


def normalize_analysis_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    analysis = dict(_jsonable(payload))
    dataset_rows = _load_dataset_rows_from_analysis(analysis)
    raw_holistic_equation = analysis.get("holistic_equation")
    raw_residual_law_present, raw_residual_law = _workflow_native_surface_lookup(
        analysis=analysis,
        surface_name="residual_law",
    )
    _, raw_residual_diagnostics = _workflow_native_surface_lookup(
        analysis=analysis,
        surface_name="residual_diagnostics",
    )
    _, raw_freeze_chain = _workflow_native_surface_lookup(
        analysis=analysis,
        surface_name="freeze_chain",
    )
    _, raw_uncertainty_attachment = _workflow_native_surface_lookup(
        analysis=analysis,
        surface_name="uncertainty_attachment",
        include_top_level=False,
    )
    if (
        not isinstance(analysis.get("residual_diagnostics"), Mapping)
        and isinstance(raw_residual_diagnostics, Mapping)
    ):
        analysis["residual_diagnostics"] = dict(_jsonable(raw_residual_diagnostics))
    if (
        not isinstance(analysis.get("freeze_chain"), Mapping)
        and isinstance(raw_freeze_chain, Mapping)
    ):
        analysis["freeze_chain"] = dict(_jsonable(raw_freeze_chain))
    analysis["dataset"] = _normalize_dataset_payload(
        analysis=analysis,
        dataset_rows=dataset_rows,
    )
    analysis_path = _infer_analysis_path(analysis)
    if analysis_path is not None:
        analysis["analysis_path"] = analysis_path

    operator_point = analysis.get("operator_point")
    if isinstance(operator_point, Mapping):
        analysis["operator_point"] = _normalize_operator_point_payload(
            operator_point,
        )

    benchmark = analysis.get("benchmark")
    if isinstance(benchmark, Mapping):
        analysis["benchmark"] = _normalize_benchmark_payload(benchmark)

    descriptive_fit = _resolve_descriptive_fit_payload(
        analysis=analysis,
        dataset_rows=dataset_rows,
    )
    if descriptive_fit is not None:
        analysis["descriptive_fit"] = descriptive_fit

    descriptive_fit = analysis.get("descriptive_fit")
    if isinstance(descriptive_fit, Mapping):
        normalized_descriptive_fit = _normalize_descriptive_fit_payload(
            analysis=analysis,
            descriptive_fit=descriptive_fit,
            dataset_rows=dataset_rows,
        )
        if normalized_descriptive_fit is not None:
            analysis["descriptive_fit"] = normalized_descriptive_fit
        else:
            analysis.pop("descriptive_fit", None)
            if isinstance(analysis.get("benchmark"), Mapping):
                benchmark_payload = dict(analysis["benchmark"])
                benchmark_payload["descriptive_fit_materialization_reason"] = (
                    _string_or_none(
                        benchmark_payload.get(
                            "descriptive_fit_materialization_reason"
                        )
                    )
                    or "reconstruction_floor_failed"
                )
                analysis["benchmark"] = benchmark_payload
    if isinstance(analysis.get("benchmark"), Mapping):
        benchmark_payload = dict(analysis["benchmark"])
        descriptive_fit_status = _summarize_benchmark_descriptive_fit_status(
            benchmark=benchmark_payload,
            descriptive_fit=analysis.get("descriptive_fit"),
        )
        if descriptive_fit_status is not None:
            benchmark_payload["descriptive_fit_status"] = descriptive_fit_status
        analysis["benchmark"] = benchmark_payload

    descriptive_reconstruction = _build_descriptive_reconstruction(
        dataset_rows=dataset_rows,
    )
    if descriptive_reconstruction is not None:
        analysis["descriptive_reconstruction"] = descriptive_reconstruction
    else:
        analysis.pop("descriptive_reconstruction", None)

    probabilistic = analysis.get("probabilistic")
    if isinstance(probabilistic, Mapping):
        normalized_probabilistic: dict[str, Any] = {}
        for forecast_object_type, lane_payload in probabilistic.items():
            if not isinstance(lane_payload, Mapping):
                normalized_probabilistic[str(forecast_object_type)] = lane_payload
                continue
            normalized_probabilistic[str(forecast_object_type)] = (
                _normalize_probabilistic_lane(
                    analysis=analysis,
                    forecast_object_type=str(forecast_object_type),
                    lane_payload=lane_payload,
                    dataset_rows=dataset_rows,
                )
            )
        analysis["probabilistic"] = normalized_probabilistic

    analysis["predictive_law"] = _build_predictive_law(analysis=analysis)
    analysis["residual_law"] = _build_residual_law(
        analysis=analysis,
        raw_payload_present=raw_residual_law_present,
        raw_payload=raw_residual_law,
        raw_diagnostics=raw_residual_diagnostics,
    )
    analysis["holistic_equation"] = _build_holistic_equation(
        analysis=analysis,
        dataset_rows=dataset_rows,
        raw_payload=raw_holistic_equation,
    )
    analysis["uncertainty_attachment"] = _build_uncertainty_attachment(
        analysis=analysis,
        raw_payload=raw_uncertainty_attachment,
    )
    _apply_claim_taxonomy(
        analysis=analysis,
        raw_holistic_equation=raw_holistic_equation,
    )
    analysis["evidence_studio"] = _build_evidence_studio(analysis=analysis)

    change_atlas = _build_change_atlas(
        analysis=analysis,
        dataset_rows=dataset_rows,
    )
    if change_atlas is not None:
        analysis["change_atlas"] = change_atlas

    return analysis


def _normalize_dataset_payload(
    *,
    analysis: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dataset_payload = analysis.get("dataset")
    dataset = (
        dict(_jsonable(dataset_payload))
        if isinstance(dataset_payload, Mapping)
        else {}
    )
    target_payload = dataset.get("target")
    target = (
        dict(_jsonable(target_payload))
        if isinstance(target_payload, Mapping)
        else {}
    )
    target_id = (
        _string_or_none(target.get("id"))
        or _string_or_none(analysis.get("request", {}).get("target_id"))
        or DEFAULT_TARGET_ID
    )
    target_spec = TARGET_SPECS.get(target_id, {"id": target_id})
    dataset["target"] = {**dict(target_spec), **target}
    fallback_symbol = (
        _string_or_none(dataset_rows[0].get("series_id")) if dataset_rows else None
    )
    dataset.setdefault(
        "symbol",
        _string_or_none(dataset.get("symbol"))
        or _string_or_none(analysis.get("request", {}).get("symbol"))
        or fallback_symbol,
    )
    if dataset_rows:
        dataset.setdefault("rows", len(dataset_rows))
        dataset.setdefault(
            "date_range",
            {
                "start": str(dataset_rows[0]["event_time"]),
                "end": str(dataset_rows[-1]["event_time"]),
            },
        )
        dataset.setdefault("stats", _dataset_stats_from_rows(dataset_rows))
        dataset.setdefault("series", _dataset_series(dataset_rows))
    return dataset


def _workflow_native_surface_lookup(
    *,
    analysis: Mapping[str, Any],
    surface_name: str,
    include_top_level: bool = True,
) -> tuple[bool, Mapping[str, Any] | None]:
    if include_top_level:
        top_level = analysis.get(surface_name)
        if isinstance(top_level, Mapping):
            return True, dict(_jsonable(top_level))
        if surface_name in analysis and top_level is None:
            return True, None

    operator_point = analysis.get("operator_point")
    if not isinstance(operator_point, Mapping):
        return False, None

    artifact_bodies: list[Mapping[str, Any]] = [operator_point]
    for artifact_name in ("scorecard", "claim_card"):
        artifact_payload = operator_point.get(artifact_name)
        if isinstance(artifact_payload, Mapping):
            artifact_bodies.append(artifact_payload)

    for artifact_body in artifact_bodies:
        if surface_name in artifact_body and artifact_body.get(surface_name) is None:
            return True, None
        candidate = artifact_body.get(surface_name)
        if isinstance(candidate, Mapping):
            return True, dict(_jsonable(candidate))

    for artifact_body in artifact_bodies:
        predictive_law = artifact_body.get("predictive_law")
        if not isinstance(predictive_law, Mapping):
            continue
        if surface_name == "predictive_law":
            return True, dict(_jsonable(predictive_law))
        if surface_name in predictive_law and predictive_law.get(surface_name) is None:
            return True, None
        candidate = predictive_law.get(surface_name)
        if isinstance(candidate, Mapping):
            return True, dict(_jsonable(candidate))

    return False, None


def _infer_analysis_path(analysis: Mapping[str, Any]) -> str | None:
    explicit = _string_or_none(analysis.get("analysis_path"))
    if explicit:
        return explicit
    workspace_root = _string_or_none(analysis.get("workspace_root"))
    if not workspace_root:
        return None
    candidate = Path(workspace_root).expanduser() / "analysis.json"
    if not candidate.is_file():
        return None
    return str(candidate)


def _build_holistic_equation(
    *,
    analysis: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
    raw_payload: Any,
) -> dict[str, Any] | None:
    del dataset_rows
    if not isinstance(raw_payload, Mapping):
        return None
    if not _raw_holistic_equation_has_backend_joint_claim_gate(raw_payload):
        return None
    if raw_payload.get("exactness") == "sample_exact_closure":
        return None
    if raw_payload.get("status") != "completed":
        return None
    if raw_payload.get("claim_class") != "holistic_equation":
        return None
    if _raw_holistic_equation_is_posthoc_synthetic(raw_payload):
        return None

    operator_point = analysis.get("operator_point")
    predictive_law = analysis.get("predictive_law")
    if (
        not isinstance(operator_point, Mapping)
        or operator_point.get("status") != "completed"
        or operator_point.get("result_mode") != "candidate_publication"
        or isinstance(operator_point.get("abstention"), Mapping)
        or not isinstance(predictive_law, Mapping)
    ):
        return None
    publication = operator_point.get("publication")
    if (
        not isinstance(publication, Mapping)
        or publication.get("status") != "publishable"
    ):
        return None

    deterministic_source = _string_or_none(raw_payload.get("deterministic_source"))
    probabilistic_source = _string_or_none(raw_payload.get("probabilistic_source"))
    validation_scope_ref = _typed_ref_string(raw_payload.get("validation_scope_ref"))
    publication_record_ref = _typed_ref_string(
        raw_payload.get("publication_record_ref")
    )
    honesty_note = _string_or_none(raw_payload.get("honesty_note"))
    equation_payload = raw_payload.get("equation")
    normalized_equation = (
        dict(_jsonable(equation_payload))
        if isinstance(equation_payload, Mapping)
        else None
    )
    if (
        deterministic_source != "predictive_law"
        or probabilistic_source is None
        or validation_scope_ref is None
        or publication_record_ref is None
        or honesty_note is None
        or normalized_equation is None
        or not normalized_equation
    ):
        return None
    if not _has_curve_backed_normalized_equation(normalized_equation):
        return None
    if _predictive_law_uses_banned_symbolic_path(normalized_equation):
        return None

    if (
        validation_scope_ref != _string_or_none(predictive_law.get("validation_scope_ref"))
        or publication_record_ref
        != _string_or_none(predictive_law.get("publication_record_ref"))
    ):
        return None

    probabilistic = analysis.get("probabilistic")
    lane_payload = (
        probabilistic.get(probabilistic_source)
        if isinstance(probabilistic, Mapping)
        else None
    )
    if (
        not isinstance(lane_payload, Mapping)
        or lane_payload.get("status") != "completed"
        or not _probabilistic_lane_is_publishable(lane_payload)
    ):
        return None
    lane_validation_scope_ref = _typed_ref_string(
        lane_payload.get("validation_scope_ref")
    )
    lane_publication_record_ref = _typed_ref_string(
        lane_payload.get("publication_record_ref")
    )
    if lane_validation_scope_ref != validation_scope_ref:
        return None
    if lane_publication_record_ref != publication_record_ref:
        return None

    return {
        "status": "completed",
        "claim_class": "holistic_equation",
        "deterministic_source": deterministic_source,
        "probabilistic_source": probabilistic_source,
        "validation_scope_ref": validation_scope_ref,
        "publication_record_ref": publication_record_ref,
        "equation": normalized_equation,
        "honesty_note": honesty_note,
    }


def _build_residual_law(
    *,
    analysis: Mapping[str, Any],
    raw_payload_present: bool,
    raw_payload: Any,
    raw_diagnostics: Any,
) -> dict[str, Any] | None:
    del analysis, raw_diagnostics
    if raw_payload_present:
        if raw_payload is None:
            return None
        normalized = _normalize_residual_law_payload(raw_payload)
        if normalized is not None:
            return normalized
    return None


def _normalize_residual_law_payload(raw_payload: Any) -> dict[str, Any] | None:
    if not isinstance(raw_payload, Mapping):
        return None
    status = _string_or_none(raw_payload.get("status"))
    finite_dimensionality_status = _string_or_none(
        raw_payload.get("finite_dimensionality_status")
    )
    recoverability_status = _string_or_none(
        raw_payload.get("recoverability_status")
    )
    residual_law_search_eligible = raw_payload.get("residual_law_search_eligible")
    if (
        status is None
        or finite_dimensionality_status is None
        or recoverability_status is None
        or not isinstance(residual_law_search_eligible, bool)
    ):
        return None
    return {
        "status": status,
        "residual_law_search_eligible": residual_law_search_eligible,
        "finite_dimensionality_status": finite_dimensionality_status,
        "recoverability_status": recoverability_status,
        "reason_codes": _string_list(raw_payload.get("reason_codes")),
    }


def _build_predictive_law(
    *,
    analysis: Mapping[str, Any],
) -> dict[str, Any] | None:
    operator_point = analysis.get("operator_point")
    if (
        not isinstance(operator_point, Mapping)
        or operator_point.get("status") != "completed"
        or operator_point.get("result_mode") != "candidate_publication"
    ):
        return None
    publication = operator_point.get("publication")
    if (
        not isinstance(publication, Mapping)
        or publication.get("status") != "publishable"
    ):
        return None
    if isinstance(operator_point.get("abstention"), Mapping):
        return None
    claim_card = operator_point.get("claim_card")
    scorecard = operator_point.get("scorecard")
    equation_payload = operator_point.get("equation")
    if (
        not isinstance(claim_card, Mapping)
        or not isinstance(scorecard, Mapping)
        or not isinstance(equation_payload, Mapping)
    ):
        return None
    normalized_equation = dict(_jsonable(equation_payload))
    if not _has_curve_backed_normalized_equation(normalized_equation):
        return None
    if _predictive_law_uses_banned_symbolic_path(normalized_equation):
        return None
    allowed_codes = {
        str(code)
        for code in claim_card.get("allowed_interpretation_codes", []) or []
        if code is not None
    }
    if (
        not _claim_card_is_predictive_capable(claim_card)
        or claim_card.get("predictive_support_status")
        != "confirmatory_supported"
        or scorecard.get("descriptive_status") != "passed"
        or scorecard.get("predictive_status") != "passed"
        or _PREDICTIVE_ALLOWED_INTERPRETATION not in allowed_codes
    ):
        return None
    claim_card_ref = _typed_ref_string(operator_point.get("claim_card_ref"))
    scorecard_ref = _typed_ref_string(operator_point.get("scorecard_ref"))
    validation_scope_ref = _typed_ref_string(
        operator_point.get("validation_scope_ref")
    )
    publication_record_ref = _typed_ref_string(
        operator_point.get("publication_record_ref")
    )
    if (
        claim_card_ref is None
        or scorecard_ref is None
        or validation_scope_ref is None
        or publication_record_ref is None
    ):
        return None
    evidence_summary = _build_predictive_law_evidence_summary(
        operator_point=operator_point,
        claim_card_ref=claim_card_ref,
        scorecard_ref=scorecard_ref,
        validation_scope_ref=validation_scope_ref,
        publication_record_ref=publication_record_ref,
    )
    if not _predictive_law_evidence_summary_is_complete(evidence_summary):
        return None
    return {
        "status": "completed",
        "claim_class": "predictive_law",
        "publishable": True,
        "claim_card_ref": claim_card_ref,
        "scorecard_ref": scorecard_ref,
        "validation_scope_ref": validation_scope_ref,
        "publication_record_ref": publication_record_ref,
        "evidence_summary": evidence_summary,
        "honesty_note": (
            "Predictive symbolic law reflects the published point-lane claim "
            "inside the declared validation scope."
        ),
        "equation": normalized_equation,
    }


def _build_predictive_law_evidence_summary(
    *,
    operator_point: Mapping[str, Any],
    claim_card_ref: str,
    scorecard_ref: str,
    validation_scope_ref: str,
    publication_record_ref: str,
) -> dict[str, Any]:
    claim_card = (
        dict(_jsonable(operator_point.get("claim_card")))
        if isinstance(operator_point.get("claim_card"), Mapping)
        else {}
    )
    scorecard = (
        dict(_jsonable(operator_point.get("scorecard")))
        if isinstance(operator_point.get("scorecard"), Mapping)
        else {}
    )
    publication = (
        dict(_jsonable(operator_point.get("publication")))
        if isinstance(operator_point.get("publication"), Mapping)
        else {}
    )
    allowed_codes = [
        str(code)
        for code in claim_card.get("allowed_interpretation_codes", []) or []
        if code is not None
    ]
    reason_codes = [
        str(code)
        for code in publication.get("reason_codes", []) or []
        if code is not None
    ]
    return {
        "claim_card": {
            "ref": claim_card_ref,
            "claim_type": _string_or_none(claim_card.get("claim_type")),
            "claim_ceiling": _string_or_none(claim_card.get("claim_ceiling")),
            "predictive_support_status": _string_or_none(
                claim_card.get("predictive_support_status")
            ),
            "allowed_interpretation_codes": allowed_codes,
        },
        "scorecard": {
            "ref": scorecard_ref,
            "descriptive_status": _string_or_none(
                scorecard.get("descriptive_status")
            ),
            "predictive_status": _string_or_none(scorecard.get("predictive_status")),
        },
        "validation_scope": {
            "ref": validation_scope_ref,
            "headline": (
                "Predictive-within-scope interpretation is bounded by the declared "
                "validation scope reference."
            ),
        },
        "publication_record": {
            "ref": publication_record_ref,
            "status": _string_or_none(publication.get("status")),
            "headline": _string_or_none(publication.get("headline")),
            "reason_codes": reason_codes,
        },
    }


def _claim_card_is_predictive_capable(claim_card: Mapping[str, Any]) -> bool:
    claim_type = _string_or_none(claim_card.get("claim_type"))
    claim_ceiling = _string_or_none(claim_card.get("claim_ceiling"))
    if (
        claim_type is None
        or claim_type == "descriptive_structure"
        or claim_ceiling is None
    ):
        return False
    return claim_ceiling != "descriptive_structure"


def _predictive_law_evidence_summary_is_complete(
    summary: Mapping[str, Any],
) -> bool:
    claim_card = summary.get("claim_card")
    scorecard = summary.get("scorecard")
    validation_scope = summary.get("validation_scope")
    publication_record = summary.get("publication_record")
    if not all(
        isinstance(section, Mapping)
        for section in (
            claim_card,
            scorecard,
            validation_scope,
            publication_record,
        )
    ):
        return False
    if (
        _string_or_none(claim_card.get("ref")) is None
        or _string_or_none(claim_card.get("claim_type")) is None
        or _string_or_none(claim_card.get("claim_ceiling")) is None
        or _string_or_none(claim_card.get("predictive_support_status")) is None
        or _string_or_none(scorecard.get("ref")) is None
        or _string_or_none(scorecard.get("descriptive_status")) is None
        or _string_or_none(scorecard.get("predictive_status")) is None
        or _string_or_none(validation_scope.get("ref")) is None
        or _string_or_none(validation_scope.get("headline")) is None
        or _string_or_none(publication_record.get("ref")) is None
        or _string_or_none(publication_record.get("status")) is None
        or _string_or_none(publication_record.get("headline")) is None
    ):
        return False
    allowed_codes = claim_card.get("allowed_interpretation_codes")
    reason_codes = publication_record.get("reason_codes")
    if not isinstance(allowed_codes, Sequence) or isinstance(
        allowed_codes, (str, bytes)
    ):
        return False
    if not isinstance(reason_codes, Sequence) or isinstance(reason_codes, (str, bytes)):
        return False
    return all(isinstance(code, str) and code for code in allowed_codes) and all(
        isinstance(code, str) and code for code in reason_codes
    )


def _has_curve_backed_normalized_equation(
    equation: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(equation, Mapping):
        return False
    curve = equation.get("curve")
    return (
        isinstance(curve, Sequence) and not isinstance(curve, (str, bytes)) and bool(curve)
    )


def _probabilistic_lane_is_publishable(lane_payload: Mapping[str, Any]) -> bool:
    calibration = lane_payload.get("calibration")
    if not isinstance(calibration, Mapping):
        return False
    return (
        _string_or_none(calibration.get("status")) == "passed"
        and calibration.get("passed") is True
        and _string_or_none(calibration.get("gate_effect")) == "publishable"
    )


def _raw_holistic_equation_is_posthoc_synthetic(
    raw_payload: Mapping[str, Any],
) -> bool:
    return any(
        _string_or_none(raw_payload.get(field_name)) is not None
        for field_name in (
            "mode",
            "selected_probabilistic_lane",
            "composition_operator",
        )
    )


def _raw_holistic_equation_has_backend_joint_claim_gate(
    raw_payload: Mapping[str, Any],
) -> bool:
    gate = raw_payload.get("joint_claim_gate")
    if not isinstance(gate, Mapping):
        return False
    authored_by = _string_or_none(gate.get("authored_by")) or _string_or_none(
        gate.get("author")
    )
    backend_authored = gate.get("backend_authored")
    status = _string_or_none(gate.get("status"))
    if backend_authored is not True and authored_by != "backend":
        return False
    return status == "accepted"


def _predictive_law_gap_reason_codes(equation_payload: Mapping[str, Any]) -> list[str]:
    reason_codes: list[str] = []
    if equation_payload.get("exactness") == "sample_exact_closure":
        reason_codes.append("requires_exact_sample_closure")
    composition_operator = _string_or_none(
        equation_payload.get("composition_operator")
    ) or _string_or_none(
        (equation_payload.get("composition_payload") or {}).get("operator_id")
    )
    if composition_operator == "additive_residual":
        reason_codes.append("requires_lookup_residual_wrapper")
    candidate_markers = (
        _string_or_none(equation_payload.get("candidate_id")),
        _string_or_none(equation_payload.get("source_candidate_id")),
        _string_or_none(equation_payload.get("selected_candidate_id")),
    )
    if any(
        marker is not None
        and any(
            fragment in marker
            for fragment in _PREDICTIVE_LAW_BANNED_CANDIDATE_ID_FRAGMENTS
        )
        for marker in candidate_markers
    ):
        reason_codes.append("requires_posthoc_symbolic_synthesis")
    return reason_codes


def _predictive_law_uses_banned_symbolic_path(
    equation_payload: Mapping[str, Any],
) -> bool:
    return bool(_predictive_law_gap_reason_codes(equation_payload))


def _build_uncertainty_attachment(
    *,
    analysis: Mapping[str, Any],
    raw_payload: Any,
) -> dict[str, Any] | None:
    normalized = _normalize_uncertainty_attachment_payload(raw_payload)
    if normalized is not None:
        if (
            not isinstance(raw_payload, Mapping)
            or not _raw_holistic_equation_has_backend_joint_claim_gate(raw_payload)
            or normalized.get("status") != "completed"
            or normalized.get("deterministic_source") != "predictive_law"
        ):
            return None
        validation_scope_ref = _typed_ref_string(
            normalized.get("validation_scope_ref")
        )
        publication_record_ref = _typed_ref_string(
            normalized.get("publication_record_ref")
        )
        if validation_scope_ref is None or publication_record_ref is None:
            return None
        predictive_law = analysis.get("predictive_law")
        if (
            not isinstance(predictive_law, Mapping)
            or validation_scope_ref
            != _typed_ref_string(predictive_law.get("validation_scope_ref"))
            or publication_record_ref
            != _typed_ref_string(predictive_law.get("publication_record_ref"))
        ):
            return None
        probabilistic = analysis.get("probabilistic")
        lane_payload = (
            probabilistic.get(normalized["probabilistic_source"])
            if isinstance(probabilistic, Mapping)
            else None
        )
        if (
            not isinstance(lane_payload, Mapping)
            or lane_payload.get("status") != "completed"
            or not _probabilistic_lane_is_publishable(lane_payload)
            or validation_scope_ref
            != _typed_ref_string(lane_payload.get("validation_scope_ref"))
            or publication_record_ref
            != _typed_ref_string(lane_payload.get("publication_record_ref"))
        ):
            return None
        return normalized
    return None

def _normalize_uncertainty_attachment_payload(
    raw_payload: Any,
) -> dict[str, Any] | None:
    if not isinstance(raw_payload, Mapping):
        return None
    status = _string_or_none(raw_payload.get("status"))
    deterministic_source = _string_or_none(raw_payload.get("deterministic_source"))
    probabilistic_source = _string_or_none(raw_payload.get("probabilistic_source"))
    if (
        status is None
        or deterministic_source is None
        or probabilistic_source is None
    ):
        return None
    payload: dict[str, Any] = {
        "status": status,
        "deterministic_source": deterministic_source,
        "probabilistic_source": probabilistic_source,
    }
    validation_scope_ref = _typed_ref_string(raw_payload.get("validation_scope_ref"))
    publication_record_ref = _typed_ref_string(
        raw_payload.get("publication_record_ref")
    )
    if validation_scope_ref is not None:
        payload["validation_scope_ref"] = validation_scope_ref
    if publication_record_ref is not None:
        payload["publication_record_ref"] = publication_record_ref
    return payload


def _apply_claim_taxonomy(
    *,
    analysis: dict[str, Any],
    raw_holistic_equation: Any,
) -> None:
    predictive_law = analysis.get("predictive_law")
    holistic_equation = analysis.get("holistic_equation")
    descriptive_reconstruction = analysis.get("descriptive_reconstruction")
    claim_class = None
    publishable = False
    if isinstance(holistic_equation, Mapping):
        claim_class = "holistic_equation"
        publishable = True
    elif isinstance(predictive_law, Mapping):
        claim_class = "predictive_law"
        publishable = True
    elif isinstance(analysis.get("descriptive_fit"), Mapping):
        claim_class = "descriptive_fit"
    elif isinstance(descriptive_reconstruction, Mapping):
        claim_class = "descriptive_reconstruction"
    analysis["claim_class"] = claim_class
    analysis["publishable"] = publishable
    analysis["would_have_abstained_because"] = _operator_abstention_reason_codes(
        analysis
    )
    analysis["gap_report"] = _build_gap_report(
        analysis=analysis,
        raw_holistic_equation=raw_holistic_equation,
    )
    analysis["not_holistic_because"] = (
        []
        if isinstance(holistic_equation, Mapping)
        else list(analysis["gap_report"]) or ["no_backend_joint_claim"]
    )


def _build_evidence_studio(*, analysis: Mapping[str, Any]) -> dict[str, Any]:
    studio = {
        "surface_version": "1.0.0",
        "redaction_status": "sanitized",
        "claim_surface": _build_evidence_claim_surface(analysis=analysis),
        "live_evidence": _build_evidence_live_surface(analysis=analysis),
        "replay_artifacts": _build_evidence_replay_artifacts(analysis=analysis),
        "engine_provenance": _build_evidence_engine_provenance(analysis=analysis),
        "diagnostics": _build_evidence_diagnostics(analysis=analysis),
    }
    secrets = _collect_inline_secret_values(studio)
    sanitized = redact_mapping(studio, secrets=secrets)
    if isinstance(sanitized, Mapping):
        result = dict(sanitized)
        result["redaction_status"] = "sanitized"
        return result
    return studio


def _build_evidence_claim_surface(*, analysis: Mapping[str, Any]) -> dict[str, Any]:
    operator_point = analysis.get("operator_point")
    point = operator_point if isinstance(operator_point, Mapping) else {}
    publication = point.get("publication") if isinstance(point, Mapping) else None
    publication_payload = publication if isinstance(publication, Mapping) else {}
    claim_card = point.get("claim_card") if isinstance(point, Mapping) else None
    claim_card_payload = claim_card if isinstance(claim_card, Mapping) else {}
    claim_class = _string_or_none(analysis.get("claim_class"))
    claim_ceiling = (
        _string_or_none(claim_card_payload.get("claim_ceiling"))
        or _claim_ceiling_from_class(claim_class)
    )
    return {
        "claim_class": claim_class,
        "claim_lane": _claim_lane_from_class(claim_class),
        "publishable": bool(analysis.get("publishable")),
        "publication_status": (
            _string_or_none(publication_payload.get("status")) or "unavailable"
        ),
        "claim_type": _string_or_none(claim_card_payload.get("claim_type")),
        "claim_ceiling": claim_ceiling,
        "claim_ceiling_explanation": _claim_ceiling_explanation(
            claim_ceiling=claim_ceiling,
        ),
        "downgrade_reason_codes": _string_list(analysis.get("gap_report")),
        "abstention_reason_codes": _string_list(
            analysis.get("would_have_abstained_because")
        ),
        "not_holistic_because": _string_list(analysis.get("not_holistic_because")),
        "live_evidence_boundary": _non_claim_evidence_boundary(),
    }


def _claim_lane_from_class(claim_class: str | None) -> str:
    if claim_class == "holistic_equation":
        return "holistic"
    if claim_class == "predictive_law":
        return "predictive"
    if claim_class in {"descriptive_fit", "descriptive_reconstruction"}:
        return "descriptive"
    return "none"


def _claim_ceiling_from_class(claim_class: str | None) -> str | None:
    if claim_class == "holistic_equation":
        return "holistic_equation"
    if claim_class == "predictive_law":
        return "predictive_within_declared_scope"
    if claim_class in {"descriptive_fit", "descriptive_reconstruction"}:
        return "descriptive_structure"
    return None


def _claim_ceiling_explanation(*, claim_ceiling: str | None) -> str:
    if claim_ceiling == "holistic_equation":
        return "Claim lane cleared the backend-authored joint claim gate."
    if claim_ceiling == "predictive_within_declared_scope":
        return "Claim is bounded to the declared validation scope."
    if claim_ceiling == "descriptive_structure":
        return "Evidence supports descriptive structure only, not a predictive-within-scope claim."
    return "No publishable scientific claim ceiling is available."


def _build_evidence_live_surface(*, analysis: Mapping[str, Any]) -> dict[str, Any]:
    present, raw_evidence = _resolve_live_evidence_payload(analysis=analysis)
    if not present:
        return {
            "status": "unavailable",
            "reason_codes": ["live_evidence_missing"],
            "claim_evidence_status": "not_claim_evidence",
            "claim_boundary": _non_claim_evidence_boundary(),
        }
    if not isinstance(raw_evidence, Mapping):
        return {
            "status": "malformed",
            "reason_codes": ["live_evidence_malformed"],
            "claim_evidence_status": "not_claim_evidence",
            "claim_boundary": _non_claim_evidence_boundary(),
        }

    raw_payload = dict(_jsonable(raw_evidence))
    secrets = _collect_inline_secret_values(raw_payload)
    sanitized = redact_mapping(raw_payload, secrets=secrets)
    sanitized_payload = dict(sanitized) if isinstance(sanitized, Mapping) else {}
    sanitized_payload["claim_evidence_status"] = "not_claim_evidence"
    sanitized_payload["claim_boundary"] = _non_claim_evidence_boundary()
    return {
        "status": _string_or_none(sanitized_payload.get("status")) or "available",
        "provider": _string_or_none(sanitized_payload.get("provider")),
        "endpoint_class": _string_or_none(
            sanitized_payload.get("endpoint_class")
        ),
        "reason_codes": _string_list(sanitized_payload.get("reason_codes")),
        "claim_evidence_status": "not_claim_evidence",
        "claim_boundary": _non_claim_evidence_boundary(),
        "sanitized_evidence": sanitized_payload,
    }


def _resolve_live_evidence_payload(
    *,
    analysis: Mapping[str, Any],
) -> tuple[bool, Any]:
    for key in _WORKBENCH_LIVE_EVIDENCE_KEYS:
        if key in analysis:
            return True, analysis.get(key)
    return False, None


def _non_claim_evidence_boundary() -> dict[str, Any]:
    return dict(NON_CLAIM_EVIDENCE_BOUNDARY)


def _build_evidence_replay_artifacts(
    *,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    links: list[dict[str, str]] = []
    for section, payload in _evidence_artifact_sections(analysis=analysis):
        if not isinstance(payload, Mapping):
            continue
        links.extend(_artifact_links_from_mapping(section=section, payload=payload))
    return {
        "status": "available" if links else "unavailable",
        "links": links,
    }


def _evidence_artifact_sections(
    *,
    analysis: Mapping[str, Any],
) -> list[tuple[str, Any]]:
    sections: list[tuple[str, Any]] = [
        ("analysis", analysis),
        ("dataset", analysis.get("dataset")),
        ("operator_point", analysis.get("operator_point")),
        ("benchmark", analysis.get("benchmark")),
        ("descriptive_fit", analysis.get("descriptive_fit")),
        ("predictive_law", analysis.get("predictive_law")),
        ("holistic_equation", analysis.get("holistic_equation")),
    ]
    probabilistic = analysis.get("probabilistic")
    if isinstance(probabilistic, Mapping):
        for lane_name, lane_payload in sorted(probabilistic.items()):
            sections.append((f"probabilistic.{lane_name}", lane_payload))
    return sections


def _artifact_links_from_mapping(
    *,
    section: str,
    payload: Mapping[str, Any],
) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for key, value in sorted(payload.items(), key=lambda item: str(item[0])):
        role = str(key)
        if not _is_artifact_reference_role(role):
            continue
        rendered = _render_artifact_reference(value)
        if rendered is None:
            continue
        link = {"section": section, "role": role, "value": rendered}
        if link not in links:
            links.append(link)
    return links


def _is_artifact_reference_role(role: str) -> bool:
    return (
        role.endswith("_path")
        or role.endswith("_ref")
        or role
        in {
            "analysis_path",
            "dataset_csv",
            "manifest_path",
            "raw_history_path",
            "replay_ref",
            "workspace_root",
        }
    )


def _render_artifact_reference(value: Any) -> str | None:
    typed_ref = _typed_ref_string(value)
    if typed_ref is not None:
        return typed_ref
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Path):
        return str(value)
    return None


def _build_evidence_engine_provenance(
    *,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    operator_point = analysis.get("operator_point")
    point = operator_point if isinstance(operator_point, Mapping) else {}
    explicit_provenance = point.get("engine_provenance")
    point_lane: dict[str, Any] = (
        dict(_jsonable(explicit_provenance))
        if isinstance(explicit_provenance, Mapping)
        else {}
    )
    for key in (
        "candidate_id",
        "selected_candidate_id",
        "selected_family",
        "result_mode",
        "search_scope",
        "backend_origin_record",
    ):
        if key in point and key not in point_lane:
            point_lane[key] = _jsonable(point[key])

    benchmark = analysis.get("benchmark")
    submitters = benchmark.get("submitters") if isinstance(benchmark, Mapping) else None
    submitter_summaries: list[dict[str, Any]] = []
    if isinstance(submitters, Sequence) and not isinstance(submitters, (str, bytes)):
        for submitter in submitters:
            if not isinstance(submitter, Mapping):
                continue
            submitter_summaries.append(
                {
                    "submitter_id": _string_or_none(submitter.get("submitter_id")),
                    "submitter_class": _string_or_none(
                        submitter.get("submitter_class")
                    ),
                    "status": _string_or_none(submitter.get("status")),
                    "selected_candidate_id": _string_or_none(
                        submitter.get("selected_candidate_id")
                    ),
                    "replay_contract": _jsonable(
                        submitter.get("replay_contract")
                    ),
                }
            )
    return {
        "status": "available" if point_lane or submitter_summaries else "unavailable",
        "point_lane": point_lane,
        "benchmark_submitters": submitter_summaries,
    }


def _build_evidence_diagnostics(*, analysis: Mapping[str, Any]) -> dict[str, Any]:
    operator_point = analysis.get("operator_point")
    point = operator_point if isinstance(operator_point, Mapping) else {}
    scorecard = point.get("scorecard") if isinstance(point, Mapping) else None
    scorecard_payload = (
        dict(_jsonable(scorecard)) if isinstance(scorecard, Mapping) else {}
    )
    descriptive_fit = analysis.get("descriptive_fit")
    descriptive_payload = (
        dict(_jsonable(descriptive_fit))
        if isinstance(descriptive_fit, Mapping)
        else {}
    )
    return {
        "fitting": {
            "status": _string_or_none(descriptive_payload.get("status"))
            or "unavailable",
            "source": _string_or_none(descriptive_payload.get("source")),
            "metrics": _jsonable(descriptive_payload.get("metrics") or {}),
            "reconstruction_metrics": _jsonable(
                descriptive_payload.get("reconstruction_metrics") or {}
            ),
            "semantic_audit": _jsonable(
                descriptive_payload.get("semantic_audit") or {}
            ),
        },
        "scoring": {
            "scorecard": scorecard_payload,
            "probabilistic_lanes": _probabilistic_diagnostic_summaries(
                analysis=analysis
            ),
        },
        "falsification": {
            "gap_report": _string_list(analysis.get("gap_report")),
            "not_holistic_because": _string_list(
                analysis.get("not_holistic_because")
            ),
            "abstention_reason_codes": _string_list(
                analysis.get("would_have_abstained_because")
            ),
            "falsification": _jsonable(scorecard_payload.get("falsification") or {}),
            "residual_diagnostics": _jsonable(
                analysis.get("residual_diagnostics")
                or scorecard_payload.get("residual_diagnostics")
                or {}
            ),
        },
    }


def _probabilistic_diagnostic_summaries(
    *,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    probabilistic = analysis.get("probabilistic")
    if not isinstance(probabilistic, Mapping):
        return {}
    summaries: dict[str, Any] = {}
    for lane_name, lane_payload in sorted(probabilistic.items()):
        if not isinstance(lane_payload, Mapping):
            continue
        summaries[str(lane_name)] = {
            "status": _string_or_none(lane_payload.get("status")),
            "replay_verification": _string_or_none(
                lane_payload.get("replay_verification")
            ),
            "evidence": _jsonable(lane_payload.get("evidence") or {}),
            "calibration": _jsonable(lane_payload.get("calibration") or {}),
        }
    return summaries


def _collect_inline_secret_values(value: Any) -> tuple[str, ...]:
    secrets: list[str] = []

    def visit(item: Any, *, key_name: str | None = None) -> None:
        if isinstance(item, Mapping):
            for child_key, child_value in item.items():
                visit(child_value, key_name=str(child_key).lower())
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for child in item:
                visit(child)
            return
        if key_name not in _INLINE_SECRET_VALUE_KEYS or not isinstance(item, str):
            return
        text = item.strip()
        if not text:
            return
        secrets.append(text)
        if text.lower().startswith("bearer "):
            token = text.split(None, 1)[1].strip()
            if token:
                secrets.append(token)

    visit(value)
    return tuple(dict.fromkeys(secrets))


def _build_gap_report(
    *,
    analysis: Mapping[str, Any],
    raw_holistic_equation: Any,
) -> list[str]:
    gaps: list[str] = []
    for code in analysis.get("gap_report") or []:
        if isinstance(code, str) and code.strip() and code not in gaps:
            gaps.append(str(code))
    descriptive_fit = analysis.get("descriptive_fit")
    if (
        not isinstance(analysis.get("predictive_law"), Mapping)
        and isinstance(descriptive_fit, Mapping)
    ):
        for code in _string_list(descriptive_fit.get("law_rejection_reason_codes")):
            if code == "legacy_compatibility_projection":
                continue
            if code not in gaps:
                gaps.append(code)
    operator_point = analysis.get("operator_point")
    publication = (
        operator_point.get("publication")
        if isinstance(operator_point, Mapping)
        else None
    )
    if (
        isinstance(publication, Mapping)
        and publication.get("status") != "publishable"
        and "operator_not_publishable" not in gaps
    ):
        gaps.append("operator_not_publishable")
    claim_ceiling_blocker = _claim_ceiling_blocker_code(analysis)
    if claim_ceiling_blocker is not None and claim_ceiling_blocker not in gaps:
        gaps.append(claim_ceiling_blocker)
    if (
        not isinstance(analysis.get("holistic_equation"), Mapping)
        and "no_backend_joint_claim" not in gaps
    ):
        gaps.append("no_backend_joint_claim")
    if (
        _has_thin_probabilistic_evidence(analysis)
        and "probabilistic_evidence_thin" not in gaps
    ):
        gaps.append("probabilistic_evidence_thin")
    if isinstance(operator_point, Mapping):
        equation_payload = operator_point.get("equation")
        if isinstance(equation_payload, Mapping):
            for code in _predictive_law_gap_reason_codes(equation_payload):
                if code not in gaps:
                    gaps.append(code)
    if (
        isinstance(raw_holistic_equation, Mapping)
    ):
        if (
            raw_holistic_equation.get("exactness") == "sample_exact_closure"
            and "requires_exact_sample_closure" not in gaps
        ):
            gaps.append("requires_exact_sample_closure")
        if (
            _raw_holistic_equation_is_posthoc_synthetic(raw_holistic_equation)
            and "requires_posthoc_symbolic_synthesis" not in gaps
        ):
            gaps.append("requires_posthoc_symbolic_synthesis")
    return gaps


def _operator_abstention_reason_codes(analysis: Mapping[str, Any]) -> list[str]:
    operator_point = analysis.get("operator_point")
    if not isinstance(operator_point, Mapping):
        return []
    reason_codes: list[str] = []
    publication = operator_point.get("publication")
    if isinstance(publication, Mapping) and publication.get("status") == "abstained":
        raw_reason_codes = publication.get("reason_codes")
        if isinstance(raw_reason_codes, Sequence) and not isinstance(
            raw_reason_codes, (str, bytes)
        ):
            for reason_code in raw_reason_codes:
                text = str(reason_code)
                if text and text not in reason_codes:
                    reason_codes.append(text)
    claim_ceiling_blocker = _claim_ceiling_blocker_code(analysis)
    if claim_ceiling_blocker is not None and claim_ceiling_blocker not in reason_codes:
        reason_codes.append(claim_ceiling_blocker)
    return reason_codes


def _claim_ceiling_blocker_code(analysis: Mapping[str, Any]) -> str | None:
    operator_point = analysis.get("operator_point")
    if not isinstance(operator_point, Mapping):
        return None
    abstention = operator_point.get("abstention")
    blocked_ceiling = (
        _string_or_none(abstention.get("blocked_ceiling"))
        if isinstance(abstention, Mapping)
        else None
    )
    if blocked_ceiling is not None:
        return blocked_ceiling
    claim_card = operator_point.get("claim_card")
    claim_ceiling = (
        _string_or_none(claim_card.get("claim_ceiling"))
        if isinstance(claim_card, Mapping)
        else None
    )
    if claim_ceiling == "descriptive_structure":
        return claim_ceiling
    return None


def _has_thin_probabilistic_evidence(analysis: Mapping[str, Any]) -> bool:
    probabilistic = analysis.get("probabilistic")
    if not isinstance(probabilistic, Mapping):
        return False
    for lane_payload in probabilistic.values():
        if not isinstance(lane_payload, Mapping):
            continue
        evidence = lane_payload.get("evidence")
        if isinstance(evidence, Mapping) and evidence.get("strength") == "thin":
            return True
    return False


def _typed_ref_string(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Mapping):
        schema_name = value.get("schema_name")
        object_id = value.get("object_id")
        if isinstance(schema_name, str) and isinstance(object_id, str):
            return f"{schema_name}:{object_id}"
    schema_name = getattr(value, "schema_name", None)
    object_id = getattr(value, "object_id", None)
    if isinstance(schema_name, str) and isinstance(object_id, str):
        return f"{schema_name}:{object_id}"
    return None


def _typed_ref_payload(value: Any) -> dict[str, str] | None:
    if isinstance(value, Mapping):
        schema_name = value.get("schema_name")
        object_id = value.get("object_id")
        if isinstance(schema_name, str) and isinstance(object_id, str):
            return {"schema_name": schema_name, "object_id": object_id}
    schema_name = getattr(value, "schema_name", None)
    object_id = getattr(value, "object_id", None)
    if isinstance(schema_name, str) and isinstance(object_id, str):
        return {"schema_name": schema_name, "object_id": object_id}
    return None


def _preferred_holistic_source(
    analysis: Mapping[str, Any],
) -> tuple[str | None, Mapping[str, Any] | None]:
    descriptive_fit = analysis.get("descriptive_fit")
    if (
        isinstance(descriptive_fit, Mapping)
        and descriptive_fit.get("status") == "completed"
    ):
        return "descriptive_fit", descriptive_fit
    operator_point = analysis.get("operator_point")
    if (
        isinstance(operator_point, Mapping)
        and operator_point.get("status") == "completed"
    ):
        return "operator_point", operator_point
    return None, None


def _preferred_holistic_probabilistic_lane(
    analysis: Mapping[str, Any],
) -> tuple[str | None, Mapping[str, Any] | None]:
    probabilistic = analysis.get("probabilistic")
    if not isinstance(probabilistic, Mapping):
        return None, None
    for lane_kind in _HOLISTIC_PREFERRED_LANES:
        payload = probabilistic.get(lane_kind)
        if isinstance(payload, Mapping) and payload.get("status") == "completed":
            return lane_kind, payload
    return None, None


def _build_symbolic_holistic_equation(
    *,
    analysis: Mapping[str, Any],
    compact_source: str,
    equation_payload: Mapping[str, Any],
    compact_curve: Sequence[float] | None,
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    deterministic_label = _holistic_deterministic_label(equation_payload)
    lane_kind, lane_payload = _preferred_holistic_probabilistic_lane(analysis)
    stochastic_label = _holistic_stochastic_label(
        lane_kind=lane_kind,
        lane_payload=lane_payload,
    )
    if deterministic_label is None:
        return None
    if stochastic_label is None and compact_curve is not None:
        return None

    deterministic_curve = [
        {
            "event_time": str(row["event_time"]),
            "fitted_value": float(fitted_value),
        }
        for row, fitted_value in zip(dataset_rows, compact_curve or [], strict=False)
    ]
    residual_series = []
    if compact_curve is not None:
        residual_series = [
            {
                "event_time": str(row["event_time"]),
                "observed_value": float(
                    round(float(row["observed_value"]) - float(fitted_value), 12)
                ),
            }
            for row, fitted_value in zip(dataset_rows, compact_curve, strict=False)
        ]

    mode = "composition_stochastic" if stochastic_label else "composition_symbolic"
    main_label = deterministic_label
    if stochastic_label is not None:
        deterministic_rhs = _equation_rhs(deterministic_label) or deterministic_label
        main_label = f"y(t) = \\left({deterministic_rhs}\\right) + \\varepsilon_t"

    composition_operator = _string_or_none(
        equation_payload.get("composition_operator")
    ) or _string_or_none(
        (equation_payload.get("composition_payload") or {}).get("operator_id")
    )
    if composition_operator is None:
        composition_operator = "identity"

    honesty_note = (
        "Holistic equation renders Euclid's deterministic law as one symbolic "
        "expression and attaches the selected probabilistic lane as an explicit "
        "stochastic rule."
        if stochastic_label
        else "Holistic equation renders Euclid's deterministic law symbolically. "
        "No completed probabilistic lane was available, so the workbench cannot "
        "attach a typed uncertainty law here."
    )
    equation = {
        "candidate_id": "holistic_symbolic_law",
        "family_id": "holistic",
        "label": main_label,
        "deterministic_label": deterministic_label,
        "delta_form_label": stochastic_label,
        "curve": deterministic_curve,
        "render_status": "formula_supported",
    }
    return {
        "status": "completed",
        "mode": mode,
        "source": compact_source,
        "composition_operator": composition_operator,
        "selected_probabilistic_lane": lane_kind,
        "row_count": len(dataset_rows),
        "honesty_note": honesty_note,
        "equation": equation,
        "deterministic_label": deterministic_label,
        "stochastic_label": stochastic_label,
        "chart": {
            "actual_series": _actual_series(dataset_rows),
            "equation_curve": deterministic_curve,
        },
        "residual_series": residual_series,
    }


def _holistic_deterministic_label(equation_payload: Mapping[str, Any]) -> str | None:
    explicit = _string_or_none(equation_payload.get("label"))
    if explicit:
        return explicit
    family_id = _string_or_none(equation_payload.get("family_id")) or "unknown"
    params = (
        dict(equation_payload.get("parameter_summary", {}))
        if isinstance(equation_payload.get("parameter_summary"), Mapping)
        else {}
    )
    literals = (
        dict(equation_payload.get("literals", {}))
        if isinstance(equation_payload.get("literals"), Mapping)
        else {}
    )
    state = (
        dict(equation_payload.get("state", {}))
        if isinstance(equation_payload.get("state"), Mapping)
        else {}
    )
    composition_operator = _string_or_none(
        equation_payload.get("composition_operator")
    ) or _string_or_none(
        (equation_payload.get("composition_payload") or {}).get("operator_id")
    )
    composition_payload = (
        dict(equation_payload.get("composition_payload", {}))
        if isinstance(equation_payload.get("composition_payload"), Mapping)
        else {}
    )
    if composition_operator in {None, "identity"}:
        return _render_equation_label(
            candidate_id=_string_or_none(equation_payload.get("candidate_id")) or "",
            family_id=family_id,
            parameter_summary=params,
            literals=literals,
        )
    if composition_operator == "additive_residual":
        return _render_additive_residual_label(
            family_id=family_id,
            parameter_summary=params,
            literals=literals,
            composition_payload=composition_payload,
        )
    if composition_operator == "piecewise":
        return _render_piecewise_label(
            family_id=family_id,
            parameter_summary=params,
            literals=literals,
            composition_payload=composition_payload,
        )
    if composition_operator == "regime_conditioned":
        return _render_regime_conditioned_label(
            family_id=family_id,
            parameter_summary=params,
            literals=literals,
            composition_payload=composition_payload,
        )
    if composition_operator == "shared_plus_local_decomposition":
        return _render_shared_plus_local_label(
            parameter_summary=params,
            composition_payload=composition_payload,
            state=state,
        )
    return None


def _render_additive_residual_label(
    *,
    family_id: str,
    parameter_summary: Mapping[str, Any],
    literals: Mapping[str, Any],
    composition_payload: Mapping[str, Any],
) -> str | None:
    base_id = _string_or_none(composition_payload.get("base_reducer"))
    residual_id = _string_or_none(composition_payload.get("residual_reducer"))
    if base_id is None or residual_id is None:
        return None
    base_rhs = _component_equation_rhs(
        family_id=family_id,
        parameter_summary=parameter_summary,
        literals=literals,
        component_id=base_id,
    )
    residual_rhs = _component_equation_rhs(
        family_id=family_id,
        parameter_summary=parameter_summary,
        literals=literals,
        component_id=residual_id,
    )
    return f"y(t) = \\left({base_rhs}\\right) + \\left({residual_rhs}\\right)"


def _render_piecewise_label(
    *,
    family_id: str,
    parameter_summary: Mapping[str, Any],
    literals: Mapping[str, Any],
    composition_payload: Mapping[str, Any],
) -> str | None:
    partition = composition_payload.get("ordered_partition")
    if not isinstance(partition, Sequence) or isinstance(partition, (str, bytes)):
        return None
    branch_terms: list[str] = []
    for branch in partition:
        if not isinstance(branch, Mapping):
            continue
        reducer_id = _string_or_none(branch.get("reducer_id"))
        start_literal = _float_or_none(branch.get("start_literal"))
        end_literal = _float_or_none(branch.get("end_literal"))
        if reducer_id is None or start_literal is None or end_literal is None:
            continue
        branch_rhs = _component_equation_rhs(
            family_id=family_id,
            parameter_summary=parameter_summary,
            literals=literals,
            component_id=reducer_id,
        )
        branch_terms.append(
            "\\mathbf{1}["
            f"{_format_number(start_literal)} \\le g_t < {_format_number(end_literal)}"
            f"]\\left({branch_rhs}\\right)"
        )
    if not branch_terms:
        return None
    return "y(t) = " + " + ".join(branch_terms)


def _render_regime_conditioned_label(
    *,
    family_id: str,
    parameter_summary: Mapping[str, Any],
    literals: Mapping[str, Any],
    composition_payload: Mapping[str, Any],
) -> str | None:
    branches = composition_payload.get("branch_reducers")
    if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes)):
        return None
    selection_mode = _string_or_none(
        (composition_payload.get("gating_law") or {}).get("selection_mode")
    ) or "hard_switch"
    if selection_mode == "convex_weighting":
        weighted_terms: list[str] = []
        for index, branch in enumerate(branches, start=1):
            if not isinstance(branch, Mapping):
                continue
            reducer_id = _string_or_none(branch.get("reducer_id"))
            if reducer_id is None:
                continue
            branch_rhs = _component_equation_rhs(
                family_id=family_id,
                parameter_summary=parameter_summary,
                literals=literals,
                component_id=reducer_id,
            )
            weighted_terms.append(f"w_{{{index}}}(g_t)\\left({branch_rhs}\\right)")
        if not weighted_terms:
            return None
        return "y(t) = " + " + ".join(weighted_terms)

    branch_terms: list[str] = []
    for branch in branches:
        if not isinstance(branch, Mapping):
            continue
        reducer_id = _string_or_none(branch.get("reducer_id"))
        regime_value = branch.get("regime_value")
        if reducer_id is None or regime_value is None:
            continue
        branch_rhs = _component_equation_rhs(
            family_id=family_id,
            parameter_summary=parameter_summary,
            literals=literals,
            component_id=reducer_id,
        )
        regime_label = _latex_literal(regime_value)
        branch_terms.append(
            f"\\mathbf{{1}}[r_t = {regime_label}]\\left({branch_rhs}\\right)"
        )
    if not branch_terms:
        return None
    return "y(t) = " + " + ".join(branch_terms)


def _render_shared_plus_local_label(
    *,
    parameter_summary: Mapping[str, Any],
    composition_payload: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str | None:
    del composition_payload, state
    shared_rhs = _shared_component_rhs(parameter_summary)
    local_terms: list[str] = []
    if any(str(key).startswith("local_adjustment__") for key in parameter_summary):
        local_terms.append("\\alpha_e")
    if any(
        str(key).startswith("local_lag_adjustment__") for key in parameter_summary
    ):
        local_terms.append("\\beta_e y_{e}(t-1)")
    rhs = shared_rhs or "f_{shared}(e,t,H_t)"
    if local_terms:
        rhs = f"\\left({rhs}\\right) + " + " + ".join(local_terms)
    return f"y_{{e}}(t) = {rhs}"


def _shared_component_rhs(parameter_summary: Mapping[str, Any]) -> str | None:
    intercept = _float_or_none(parameter_summary.get("shared_intercept"))
    lag = _float_or_none(parameter_summary.get("shared_lag_coefficient"))
    if intercept is not None and lag is not None:
        return (
            f"{_format_number(intercept)} + {_format_number(lag)}*y_{{e}}(t-1)"
        )
    if intercept is not None:
        return _format_number(intercept)
    return None


def _component_equation_rhs(
    *,
    family_id: str,
    parameter_summary: Mapping[str, Any],
    literals: Mapping[str, Any],
    component_id: str,
) -> str:
    scoped_parameters = _component_mapping(parameter_summary, component_id=component_id)
    scoped_literals = _component_mapping(literals, component_id=component_id)
    label = _render_equation_label(
        candidate_id="",
        family_id=family_id,
        parameter_summary=scoped_parameters,
        literals=scoped_literals,
    )
    rhs = _equation_rhs(label)
    if rhs:
        return rhs
    return f"f_{{{_latex_identifier(component_id)}}}(t,H_t)"


def _component_mapping(
    mapping: Mapping[str, Any],
    *,
    component_id: str,
) -> dict[str, Any]:
    suffix = f"__{component_id}"
    scoped: dict[str, Any] = {}
    for key, value in mapping.items():
        key_string = str(key)
        if not key_string.endswith(suffix):
            continue
        scoped[key_string[: -len(suffix)]] = value
    return scoped


def _holistic_stochastic_label(
    *,
    lane_kind: str | None,
    lane_payload: Mapping[str, Any] | None,
) -> str | None:
    if lane_kind is None or not isinstance(lane_payload, Mapping):
        return None
    if lane_kind == "distribution":
        return "\\varepsilon_t \\sim \\mathcal{N}(0, \\sigma_t^2)"
    if lane_kind == "quantile":
        return "Q_p\\left(Y_t \\mid H_t\\right) = q_p(t)"
    if lane_kind == "interval":
        return (
            "\\Pr\\left(\\ell_t \\le Y_t \\le u_t \\mid H_t\\right) = 1-\\alpha"
        )
    if lane_kind == "event_probability":
        event_row = lane_payload.get("latest_row")
        threshold = None
        if isinstance(event_row, Mapping):
            event_definition = event_row.get("event_definition")
            if isinstance(event_definition, Mapping):
                threshold = _float_or_none(event_definition.get("threshold"))
        if threshold is None:
            return "\\Pr\\left(Y_t \\in E_t \\mid H_t\\right) = \\pi_t"
        return (
            f"\\Pr\\left(Y_t \\ge {_format_number(threshold)} \\mid H_t\\right) = \\pi_t"
        )
    return None


def _aligned_compact_curve(
    *,
    dataset_rows: Sequence[Mapping[str, Any]],
    compact_curve_payload: Sequence[Mapping[str, Any]] | object,
) -> list[float] | None:
    if not dataset_rows:
        return None
    curve_points = (
        list(compact_curve_payload)
        if isinstance(compact_curve_payload, Sequence)
        and not isinstance(compact_curve_payload, (str, bytes))
        else []
    )
    if not curve_points:
        return None
    by_time = {
        str(point["event_time"]): float(point["fitted_value"])
        for point in curve_points
        if isinstance(point, Mapping)
        and point.get("event_time") is not None
        and point.get("fitted_value") is not None
    }
    aligned: list[float] = []
    for index, row in enumerate(dataset_rows):
        fitted_value = by_time.get(str(row["event_time"]))
        if fitted_value is None and index < len(curve_points):
            fallback = curve_points[index]
            if (
                isinstance(fallback, Mapping)
                and fallback.get("fitted_value") is not None
            ):
                fitted_value = float(fallback["fitted_value"])
        if fitted_value is None or not math.isfinite(float(fitted_value)):
            return None
        aligned.append(float(fitted_value))
    return aligned


def _latex_identifier(value: str) -> str:
    return str(value).replace("_", "\\_")


def _latex_literal(value: Any) -> str:
    if isinstance(value, str):
        return f"\\mathrm{{{_latex_identifier(value)}}}"
    if isinstance(value, bool):
        return "\\mathrm{true}" if value else "\\mathrm{false}"
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return _format_number(float(value))
    return "\\mathrm{value}"


def _default_operator_point_publication(
    *,
    result_mode: str | None,
    reason_codes: Sequence[Any] | None,
) -> dict[str, Any]:
    normalized_reason_codes = [str(code) for code in reason_codes or [] if code]
    if result_mode == "candidate_publication" and not normalized_reason_codes:
        return {
            "status": "publishable",
            "headline": (
                "Operator replay published a point-lane candidate within the "
                "declared validation scope."
            ),
            "reason_codes": [],
        }
    if result_mode == "abstention_only_publication" or normalized_reason_codes:
        return {
            "status": "abstained",
            "headline": (
                "Operator replay verified a candidate, but the point lane did not "
                "clear publication gates."
            ),
            "reason_codes": normalized_reason_codes,
        }
    return {
        "status": "candidate_only",
        "headline": (
            "Operator output is shown as the selected point candidate. Check "
            "its scorecard and robustness artifacts before treating it as a "
            "publishable claim."
        ),
        "reason_codes": [],
    }


def _merge_operator_point_publication_conservatively(
    *,
    saved_publication: Mapping[str, Any] | None,
    computed_publication: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(saved_publication, Mapping):
        return dict(_jsonable(computed_publication))
    status_rank = {
        "abstained": 0,
        "candidate_only": 1,
        "publishable": 2,
    }
    saved = dict(_jsonable(saved_publication))
    computed = dict(_jsonable(computed_publication))
    saved_status = _string_or_none(saved.get("status"))
    computed_status = _string_or_none(computed.get("status"))
    if (
        saved_status not in status_rank
        or computed_status not in status_rank
        or status_rank[saved_status] > status_rank[computed_status]
    ):
        return computed
    saved["headline"] = _string_or_none(saved.get("headline")) or _string_or_none(
        computed.get("headline")
    )
    saved_reason_codes = [str(code) for code in saved.get("reason_codes") or [] if code]
    computed_reason_codes = [
        str(code) for code in computed.get("reason_codes") or [] if code
    ]
    saved["reason_codes"] = saved_reason_codes or computed_reason_codes
    return saved


def _normalize_operator_point_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    point = dict(_jsonable(payload))
    equation_payload = point.get("equation")
    if isinstance(equation_payload, Mapping):
        point["equation"] = _normalize_equation_payload(
            equation_payload,
            candidate_id=_string_or_none(point.get("candidate_id")),
            family_id=_string_or_none(point.get("selected_family")),
        )
    search_scope = _summarize_search_scope(
        manifest_path=_string_or_none(point.get("manifest_path")),
        lane_kind="point",
    )
    if search_scope is not None:
        point["search_scope"] = search_scope

    abstention_payload = point.get("abstention")
    abstention = (
        dict(_jsonable(abstention_payload))
        if isinstance(abstention_payload, Mapping)
        else {}
    )
    reason_codes = abstention.get("reason_codes") or abstention.get(
        "failure_reason_codes"
    )
    saved_publication = (
        dict(_jsonable(point.get("publication")))
        if isinstance(point.get("publication"), Mapping)
        else None
    )
    computed_publication = _default_operator_point_publication(
        result_mode=_string_or_none(point.get("result_mode")),
        reason_codes=reason_codes,
    )
    point["publication"] = _merge_operator_point_publication_conservatively(
        saved_publication=saved_publication,
        computed_publication=computed_publication,
    )
    point["publishable"] = point["publication"]["status"] == "publishable"
    return point


def _normalize_descriptive_fit_payload(
    *,
    analysis: Mapping[str, Any],
    descriptive_fit: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    normalized = dict(_jsonable(descriptive_fit))
    equation_payload = normalized.get("equation")
    if isinstance(equation_payload, Mapping):
        normalized["equation"] = _normalize_equation_payload(
            equation_payload,
            candidate_id=_string_or_none(normalized.get("candidate_id")),
            family_id=_string_or_none(normalized.get("family_id")),
        )
    normalized.setdefault(
        "honesty_note",
        "Descriptive fit from the broader benchmark-local search winner; not an "
        "operator publication.",
    )
    candidate_id = _string_or_none(normalized.get("candidate_id")) or _string_or_none(
        (normalized.get("equation") or {}).get("candidate_id")
    )
    law_eligible, law_rejection_reason_codes = _resolve_descriptive_fit_law_semantics(
        analysis=analysis,
        descriptive_fit=normalized,
        candidate_id=candidate_id,
    )
    normalized["selection_scope"] = (
        _string_or_none(normalized.get("selection_scope"))
        or _DESCRIPTIVE_SELECTION_SCOPE
    )
    normalized["selection_rule"] = (
        _string_or_none(normalized.get("selection_rule"))
        or _DESCRIPTIVE_SELECTION_RULE
    )
    reconstruction_curve = _descriptive_fit_equation_curve(
        descriptive_fit=normalized,
        dataset_rows=dataset_rows,
        use_observed_lag_values=True,
    )
    if reconstruction_curve is not None:
        equation = (
            dict(_jsonable(normalized.get("equation")))
            if isinstance(normalized.get("equation"), Mapping)
            else {}
        )
        equation["curve"] = reconstruction_curve
        normalized["equation"] = equation
        chart = (
            dict(_jsonable(normalized.get("chart")))
            if isinstance(normalized.get("chart"), Mapping)
            else {}
        )
        chart["equation_curve"] = reconstruction_curve
        if dataset_rows:
            chart["actual_series"] = _actual_series(dataset_rows)
        normalized["chart"] = chart
    reconstruction_metrics = _descriptive_fit_reconstruction_metrics(
        descriptive_fit=normalized,
        dataset_rows=dataset_rows,
    )
    if reconstruction_metrics is not None:
        normalized["reconstruction_metrics"] = dict(
            _jsonable(reconstruction_metrics)
        )
        if not _descriptive_fit_clears_reconstruction_floor(
            reconstruction_metrics
        ):
            return None
    normalized["law_eligible"] = bool(law_eligible)
    normalized["law_rejection_reason_codes"] = law_rejection_reason_codes
    normalized["claim_class"] = "descriptive_fit"
    normalized["is_law_claim"] = False
    normalized["semantic_audit"] = _build_descriptive_fit_semantic_audit(
        analysis=analysis,
        descriptive_fit=normalized,
        dataset_rows=dataset_rows,
    )
    return normalized


def _resolve_descriptive_fit_law_semantics(
    *,
    analysis: Mapping[str, Any],
    descriptive_fit: Mapping[str, Any],
    candidate_id: str | None,
) -> tuple[bool, list[str]]:
    explicit_law_eligible = descriptive_fit.get("law_eligible")
    explicit_reason_codes = _string_list(
        descriptive_fit.get("law_rejection_reason_codes")
    )
    if isinstance(explicit_law_eligible, bool):
        if explicit_law_eligible:
            return True, []
        return False, explicit_reason_codes or _default_descriptive_fit_rejection_codes(
            analysis=analysis,
            candidate_id=candidate_id,
        )
    if explicit_reason_codes:
        return False, explicit_reason_codes

    accepted_candidate_id = _accepted_candidate_id_from_analysis(analysis)
    if accepted_candidate_id is not None and candidate_id == accepted_candidate_id:
        return True, []
    return False, _default_descriptive_fit_rejection_codes(
        analysis=analysis,
        candidate_id=candidate_id,
    )


def _accepted_candidate_id_from_analysis(analysis: Mapping[str, Any]) -> str | None:
    benchmark = analysis.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return None
    descriptive_fit = benchmark.get("descriptive_fit")
    if isinstance(descriptive_fit, Mapping):
        explicit = _string_or_none(descriptive_fit.get("accepted_candidate_id"))
        if explicit is not None:
            return explicit
    return None


def _default_descriptive_fit_rejection_codes(
    *,
    analysis: Mapping[str, Any],
    candidate_id: str | None,
) -> list[str]:
    accepted_candidate_id = _accepted_candidate_id_from_analysis(analysis)
    if accepted_candidate_id is None:
        return ["no_accepted_candidate"]
    if candidate_id is None or candidate_id != accepted_candidate_id:
        return ["outside_law_eligible_scope"]
    return []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if item]


def _build_descriptive_fit_semantic_audit(
    *,
    analysis: Mapping[str, Any],
    descriptive_fit: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_id = str(
        analysis.get("dataset", {}).get("target", {}).get("id")
        or analysis.get("request", {}).get("target_id")
        or DEFAULT_TARGET_ID
    )
    equation = descriptive_fit.get("equation")
    equation_payload = (
        dict(_jsonable(equation)) if isinstance(equation, Mapping) else {}
    )
    params = (
        dict(equation_payload.get("parameter_summary", {}))
        if isinstance(equation_payload.get("parameter_summary"), Mapping)
        else {}
    )
    candidate_id = _string_or_none(descriptive_fit.get("candidate_id")) or _string_or_none(
        equation_payload.get("candidate_id")
    )
    family_id = _string_or_none(descriptive_fit.get("family_id")) or _string_or_none(
        equation_payload.get("family_id")
    )

    actual_values = [float(row["observed_value"]) for row in dataset_rows]
    fitted_points = (
        equation_payload.get("curve")
        or descriptive_fit.get("chart", {}).get("equation_curve")
        or []
    )
    fitted_values = [
        float(point["fitted_value"])
        for point in fitted_points
        if isinstance(point, Mapping) and point.get("fitted_value") is not None
    ]

    fit_mae: float | None = None
    naive_mae: float | None = None
    relative_improvement: float | None = None
    if len(actual_values) >= 2 and len(fitted_values) >= 2:
        comparison_count = min(len(actual_values), len(fitted_values))
        fit_errors = [
            abs(actual_values[index] - fitted_values[index])
            for index in range(1, comparison_count)
        ]
        naive_errors = [
            abs(actual_values[index] - actual_values[index - 1])
            for index in range(1, comparison_count)
        ]
        if fit_errors and naive_errors:
            fit_mae = statistics.fmean(fit_errors)
            naive_mae = statistics.fmean(naive_errors)
            if naive_mae > 0:
                relative_improvement = (naive_mae - fit_mae) / naive_mae

    intercept = _float_or_none(params.get("intercept"))
    lag_coefficient = _float_or_none(params.get("lag_coefficient"))
    delta_form_label = _delta_form_label(
        candidate_id=candidate_id,
        family_id=family_id,
        intercept=intercept,
        lag_coefficient=lag_coefficient,
    )
    near_persistence = (
        target_id == "price_close"
        and candidate_id in {"analytic_lag1_affine", "algorithmic_last_observation"}
        and relative_improvement is not None
        and relative_improvement < 0.05
    )
    if (
        target_id == "price_close"
        and lag_coefficient is not None
        and abs(1.0 - lag_coefficient) <= 0.02
        and relative_improvement is not None
        and relative_improvement < 0.05
    ):
        near_persistence = True

    if near_persistence:
        improvement_pct = (
            "n/a"
            if relative_improvement is None
            else f"{relative_improvement * 100:.1f}%"
        )
        headline = (
            "Raw close descriptive fit is effectively a persistence equation with a "
            "tiny intercept."
        )
        summary = (
            "Against a naive y(t)=y(t-1) baseline, this fit only improves one-step "
            f"MAE by {improvement_pct}. Treat it as a level-tracking summary, not a "
            "meaningful structural market equation."
        )
        classification = "near_persistence"
    elif target_id == "price_close":
        headline = (
            "Raw close descriptive fit should be read as a level model before it is "
            "read as market structure."
        )
        summary = (
            "Level targets often preserve persistence and drift. Inspect the naive "
            "last-value baseline and the delta form before inferring semantics."
        )
        classification = "level_fit"
    elif target_id in {"daily_return", "log_return"}:
        headline = (
            "Return-space descriptive fit is more interpretable than a raw-price "
            "level fit, but it still needs benchmark context."
        )
        summary = (
            "Return targets reduce trivial persistence, so improvements over simple "
            "baselines are more meaningful here."
        )
        classification = "return_space_fit"
    else:
        headline = (
            "Descriptive fit summarizes the observed target but is not itself an "
            "operator publication."
        )
        summary = (
            "Use the fitted equation as a descriptive compression of the series, then "
            "compare it against the operator and benchmark lanes."
        )
        classification = "descriptive_summary"

    return {
        "classification": classification,
        "headline": headline,
        "summary": summary,
        "delta_form_label": delta_form_label,
        "fit_mae": fit_mae,
        "naive_last_value_mae": naive_mae,
        "relative_improvement_vs_naive_last_value": relative_improvement,
        "recommended_target_id": (
            DEFAULT_TARGET_ID if target_id == "price_close" else None
        ),
        "recommended_target_label": (
            TARGET_SPECS[DEFAULT_TARGET_ID]["label"]
            if target_id == "price_close"
            else None
        ),
        "recommended_target_reason": (
            TARGET_SPECS[DEFAULT_TARGET_ID]["recommended_reason"]
            if target_id == "price_close"
            else None
        ),
    }


def _summarize_search_scope(
    *,
    manifest_path: str | None,
    lane_kind: str,
) -> dict[str, Any] | None:
    manifest = _load_yaml_payload(manifest_path)
    if not isinstance(manifest, Mapping):
        return None
    search = manifest.get("search")
    search_payload = dict(search) if isinstance(search, Mapping) else {}
    family_ids = [
        str(item)
        for item in search_payload.get("family_ids", [])
        if item is not None
    ]
    search_class = _string_or_none(search_payload.get("class"))
    proposal_limit = search_payload.get("proposal_limit")
    workflow_id = _string_or_none(manifest.get("workflow_id"))

    if len(family_ids) == 1 and int(proposal_limit or 1) == 1:
        scope_kind = "single_candidate_template"
        headline = (
            f"{humanize_lane_kind(lane_kind)} lane ran as a single-candidate "
            f"template: only {family_ids[0]} was eligible."
        )
    elif (
        lane_kind == "point"
        and workflow_id == "euclid_current_release_candidate"
    ):
        scope_kind = "narrow_release_surface"
        headline = (
            "Operator point lane only searched the current-release surface "
            f"({', '.join(family_ids) or 'configured families'})."
        )
    else:
        scope_kind = "configured_manifest"
        headline = (
            f"{humanize_lane_kind(lane_kind)} lane reflects the families and budget "
            "explicitly configured in its manifest."
        )

    return {
        "scope_kind": scope_kind,
        "headline": headline,
        "workflow_id": workflow_id,
        "search_class": search_class,
        "proposal_limit": proposal_limit,
        "family_ids": family_ids,
    }


def _load_yaml_payload(manifest_path: str | None) -> dict[str, Any] | None:
    if not manifest_path:
        return None
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _run_point_analysis(
    *,
    project_root: Path,
    workspace_root: Path,
    dataset_rows: Sequence[Mapping[str, Any]],
    dataset_csv: Path,
    symbol: str,
    target_id: str,
) -> dict[str, Any]:
    manifest_path = workspace_root / "manifests" / "operator-point.yaml"
    request_id = f"workbench-{_slug(symbol)}-{_slug(target_id)}-point"
    manifest = build_operator_manifest(
        project_root=project_root,
        dataset_csv=dataset_csv,
        manifest_path=manifest_path,
        request_id=request_id,
        forecast_object_type="point",
        row_count=len(dataset_rows),
    )
    manifest["quantization_step"] = _quantization_step_for_target(target_id)
    write_yaml_manifest(manifest, destination=manifest_path)
    output_root = workspace_root / "runs" / "point"
    run_result = run_operator(manifest_path=manifest_path, output_root=output_root)
    replay_result = replay_operator(output_root=output_root, run_id=request_id)

    run_summary = json.loads(
        run_result.paths.run_summary_path.read_text(encoding="utf-8")
    )
    graph = load_demo_run_artifact_graph(
        output_root=output_root,
        run_id=request_id,
    )
    point_prediction = inspect_demo_point_prediction(
        output_root=output_root,
        run_id=request_id,
    )
    baseline = compare_demo_baseline(output_root=output_root, run_id=request_id)

    reducer_body = _artifact_body(
        graph=graph,
        schema_prefix="reducer_artifact_manifest@",
    )
    structure_body = _artifact_body(
        graph=graph,
        schema_prefix="canonical_structure_code_manifest@",
    )
    equation = build_equation_summary(
        candidate_id=str(run_summary["selected_candidate_id"]),
        family_id=str(run_summary["selected_family"]),
        parameter_summary=dict(reducer_body.get("parameter_summary", {})),
        structure_signature=_resolve_structure_signature(
            structure_body=structure_body,
            reducer_body=reducer_body,
        ),
        dataset_rows=dataset_rows,
        composition_operator=_string_or_none(reducer_body.get("composition_operator")),
        composition_payload=(
            dict(_jsonable(reducer_body.get("composition_payload")))
            if isinstance(reducer_body.get("composition_payload"), Mapping)
            else None
        ),
        literals=(
            dict(_jsonable(reducer_body.get("literals")))
            if isinstance(reducer_body.get("literals"), Mapping)
            else None
        ),
        state=(
            dict(_jsonable(reducer_body.get("state")))
            if isinstance(reducer_body.get("state"), Mapping)
            else None
        ),
    )

    return {
        "status": "completed",
        "run_id": request_id,
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "selected_family": run_result.summary.selected_family,
        "result_mode": run_summary.get("result_mode"),
        "confirmatory_primary_score": _float_or_none(
            run_summary.get("confirmatory_primary_score")
        ),
        "scorecard_ref": _typed_ref_payload(run_result.summary.scorecard_ref),
        "claim_card_ref": _typed_ref_payload(run_result.summary.claim_card_ref),
        "publication_record_ref": _typed_ref_payload(
            run_result.summary.publication_record_ref
        ),
        "validation_scope_ref": _jsonable(
            run_summary.get("primary_validation_scope_ref")
        ),
        "replay_verification": replay_result.summary.replay_verification_status,
        "equation": equation,
        "prediction_rows": _jsonable(point_prediction.rows),
        "latest_prediction": (
            _jsonable(point_prediction.rows[-1])
            if point_prediction.rows
            else None
        ),
        "comparison": _jsonable(
            {
                "selected_candidate_id": baseline.selected_candidate_id,
                "baseline_id": baseline.baseline_id,
                "comparison_class_status": baseline.comparison_class_status,
                "candidate_primary_score": baseline.candidate_primary_score,
                "baseline_primary_score": baseline.baseline_primary_score,
                "score_delta": baseline.score_delta,
                "candidate_beats_baseline": baseline.candidate_beats_baseline,
                "practical_significance_margin": baseline.practical_significance_margin,
            }
        ),
        "scorecard": _artifact_body(graph=graph, schema_prefix="scorecard_manifest@"),
        "abstention": _artifact_body(
            graph=graph,
            schema_prefix="abstention_manifest@",
            default=None,
        ),
        "claim_card": _artifact_body(
            graph=graph,
            schema_prefix="claim_card_manifest@",
            default=None,
        ),
        "robustness": _artifact_body(
            graph=graph,
            schema_prefix="robustness_report_manifest@",
            default=None,
        ),
        "chart": {
            "actual_series": _actual_series(dataset_rows),
            "equation_curve": equation["curve"],
            "prediction_rows": _jsonable(point_prediction.rows),
        },
    }


def _run_probabilistic_analysis(
    *,
    project_root: Path,
    workspace_root: Path,
    dataset_rows: Sequence[Mapping[str, Any]],
    dataset_csv: Path,
    symbol: str,
    target_id: str,
    forecast_object_type: str,
) -> dict[str, Any]:
    manifest_path = workspace_root / "manifests" / f"{forecast_object_type}.yaml"
    request_id = (
        f"workbench-{_slug(symbol)}-{_slug(target_id)}-{_slug(forecast_object_type)}"
    )
    manifest = build_operator_manifest(
        project_root=project_root,
        dataset_csv=dataset_csv,
        manifest_path=manifest_path,
        request_id=request_id,
        forecast_object_type=forecast_object_type,
        row_count=len(dataset_rows),
    )
    manifest["quantization_step"] = _quantization_step_for_target(target_id)
    write_yaml_manifest(manifest, destination=manifest_path)
    output_root = workspace_root / "runs" / forecast_object_type
    run_result = run_operator(manifest_path=manifest_path, output_root=output_root)
    replay_result = replay_operator(output_root=output_root, run_id=request_id)

    graph = load_demo_run_artifact_graph(output_root=output_root, run_id=request_id)
    prediction = inspect_demo_probabilistic_prediction(output_root=output_root)
    calibration = inspect_demo_calibration(output_root=output_root)
    run_summary_path = output_root / "sealed-runs" / request_id / "run-summary.json"
    run_summary: dict[str, Any] = {}
    if run_summary_path.is_file():
        loaded = json.loads(run_summary_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            run_summary = loaded
    reducer_body = _artifact_body(
        graph=graph,
        schema_prefix="reducer_artifact_manifest@",
    )
    structure_body = _artifact_body(
        graph=graph,
        schema_prefix="canonical_structure_code_manifest@",
        default=None,
    )
    equation = build_equation_summary(
        candidate_id=prediction.candidate_id,
        family_id=run_result.summary.selected_family,
        parameter_summary=dict(reducer_body.get("parameter_summary", {})),
        structure_signature=_resolve_structure_signature(
            structure_body=structure_body,
            reducer_body=reducer_body,
        ),
        dataset_rows=dataset_rows,
        composition_operator=_string_or_none(reducer_body.get("composition_operator")),
        composition_payload=(
            dict(_jsonable(reducer_body.get("composition_payload")))
            if isinstance(reducer_body.get("composition_payload"), Mapping)
            else None
        ),
        literals=(
            dict(_jsonable(reducer_body.get("literals")))
            if isinstance(reducer_body.get("literals"), Mapping)
            else None
        ),
        state=(
            dict(_jsonable(reducer_body.get("state")))
            if isinstance(reducer_body.get("state"), Mapping)
            else None
        ),
    )
    return {
        "status": "completed",
        "run_id": request_id,
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "selected_family": run_result.summary.selected_family,
        "forecast_object_type": forecast_object_type,
        "scorecard_ref": _typed_ref_payload(run_result.summary.scorecard_ref),
        "claim_card_ref": _typed_ref_payload(run_result.summary.claim_card_ref),
        "publication_record_ref": _typed_ref_payload(
            run_result.summary.publication_record_ref
        ),
        "validation_scope_ref": _jsonable(
            run_summary.get("primary_validation_scope_ref")
        ),
        "replay_verification": replay_result.summary.replay_verification_status,
        "aggregated_primary_score": prediction.aggregated_primary_score,
        "equation": equation,
        "rows": _jsonable(prediction.rows),
        "latest_row": (
            _jsonable(prediction.rows[-1]) if prediction.rows else None
        ),
        "calibration": _jsonable(
            {
                "status": calibration.status,
                "passed": calibration.passed,
                "failure_reason_code": calibration.failure_reason_code,
                "gate_effect": calibration.gate_effect,
                "diagnostics": calibration.diagnostics,
            }
        ),
        "chart": _build_probabilistic_chart(
            forecast_object_type=forecast_object_type,
            dataset_rows=dataset_rows,
            prediction_rows=prediction.rows,
        ),
    }


def _run_benchmark_analysis(
    *,
    project_root: Path,
    workspace_root: Path,
    dataset_rows: Sequence[Mapping[str, Any]],
    symbol: str,
    benchmark_workers: int,
) -> dict[str, Any]:
    benchmark_project_root = resolve_asset_root(project_root)
    benchmark_dataset_csv = write_euclid_dataset_csv(
        dataset_rows,
        destination=(
            benchmark_project_root
            / "build"
            / "workbench"
            / workspace_root.name
            / "datasets"
            / f"{_slug(symbol)}.csv"
        ),
    )
    benchmark_task_path = workspace_root / "manifests" / "benchmark-task.yaml"
    benchmark_task = build_benchmark_task_manifest(
        project_root=benchmark_project_root,
        dataset_csv=benchmark_dataset_csv,
        manifest_path=benchmark_task_path,
        symbol=symbol,
        row_count=len(dataset_rows),
        availability_cutoff=str(dataset_rows[-1]["available_at"]),
    )
    write_yaml_manifest(benchmark_task, destination=benchmark_task_path)
    benchmark_root = workspace_root / "benchmark-root"
    benchmark_result = profile_benchmark_task(
        manifest_path=benchmark_task_path,
        benchmark_root=benchmark_root,
        project_root=benchmark_project_root,
        parallel_workers=max(1, int(benchmark_workers)),
        resume=False,
    )
    task_payload = json.loads(
        benchmark_result.report_paths.task_result_path.read_text(encoding="utf-8")
    )
    portfolio_payload = json.loads(
        (
            benchmark_root
            / "results"
            / benchmark_result.task_manifest.track_id
            / benchmark_result.task_manifest.task_id
            / "portfolio-selection-record.json"
        ).read_text(encoding="utf-8")
    )
    submitter_payloads = [
        json.loads(
            (
                benchmark_root
                / "results"
                / benchmark_result.task_manifest.track_id
                / benchmark_result.task_manifest.task_id
                / f"submitter-{result.submitter_id}.json"
            ).read_text(encoding="utf-8")
        )
        for result in benchmark_result.submitter_results
    ]
    descriptive_fit = _build_descriptive_fit_from_submitter_results(
        submitter_results=benchmark_result.submitter_results,
        dataset_rows=dataset_rows,
        allow_best_effort=True,
    )
    descriptive_fit_materialization_reason = (
        "reconstruction_floor_failed"
        if descriptive_fit is None
        and any(
            result.selected_candidate is not None
            and result.submitter_id != PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
            for result in benchmark_result.submitter_results
        )
        else None
    )
    return {
        "status": "completed",
        "manifest_path": str(benchmark_task_path),
        "benchmark_root": str(benchmark_root),
        "dataset_csv": str(benchmark_dataset_csv),
        "report_path": str(benchmark_result.report_paths.report_path),
        "task_result_path": str(benchmark_result.report_paths.task_result_path),
        "telemetry_path": str(benchmark_result.telemetry_path),
        "task_id": task_payload["task_id"],
        "track_id": task_payload["track_id"],
        "local_winner_submitter_id": task_payload.get("local_winner_submitter_id"),
        "local_winner_candidate_id": task_payload.get("local_winner_candidate_id"),
        "track_summary": _jsonable(task_payload.get("track_summary", {})),
        "submitters": [
            _summarize_submitter_payload(payload)
            for payload in submitter_payloads
        ],
        "portfolio_selection": {
            "status": portfolio_payload.get("status"),
            "winner_submitter_id": portfolio_payload.get("selected_submitter_id"),
            "winner_candidate_id": portfolio_payload.get("selected_candidate_id"),
            "selection_explanation": _format_portfolio_selection_explanation(
                portfolio_payload.get("selection_explanation")
            ),
            "selection_explanation_raw": _jsonable(
                portfolio_payload.get("selection_explanation")
            ),
            "decision_trace": _jsonable(
                portfolio_payload.get("decision_trace", [])
            ),
            "compared_finalists": _jsonable(
                portfolio_payload.get("compared_finalists", [])
            ),
        },
        "chart": {
            "total_code_bits": [
                {
                    "submitter_id": payload["submitter_id"],
                    "candidate_id": payload.get("selected_candidate_id"),
                    "total_code_bits": (
                        payload.get("selected_candidate_metrics", {}).get(
                            "total_code_bits"
                        )
                    ),
                }
                for payload in submitter_payloads
            ]
        },
        "descriptive_fit": descriptive_fit,
        "descriptive_fit_materialization_reason": (
            descriptive_fit_materialization_reason
        ),
    }


def _summarize_submitter_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    backend_families = [
        participant.get("backend_family")
        for participant in payload.get("backend_participation", [])
    ]
    return {
        "submitter_id": payload.get("submitter_id"),
        "submitter_class": payload.get("submitter_class"),
        "status": payload.get("status"),
        "backend_families": backend_families,
        "selected_candidate_id": payload.get("selected_candidate_id"),
        "selected_candidate_metrics": _jsonable(
            payload.get("selected_candidate_metrics", {})
        ),
        "semantic_disclosures": _jsonable(
            payload.get("semantic_disclosures", {})
        ),
        "budget_consumption": _jsonable(payload.get("budget_consumption", {})),
        "candidate_ledger": _jsonable(payload.get("candidate_ledger", [])),
    }


def _build_descriptive_fit_from_submitter_results(
    *,
    submitter_results: Sequence[BenchmarkSubmitterResult],
    dataset_rows: Sequence[Mapping[str, Any]],
    allow_best_effort: bool = False,
) -> dict[str, Any] | None:
    if dataset_rows:
        ranked_payloads: list[tuple[dict[str, Any], dict[str, float | None]]] = []
        for result in submitter_results:
            if (
                result.selected_candidate is None
                or result.submitter_id == PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
            ):
                continue
            payload = _build_descriptive_fit_from_submitter_result(
                submitter_result=result,
                dataset_rows=dataset_rows,
            )
            metrics = _descriptive_fit_reconstruction_metrics(
                descriptive_fit=payload,
                dataset_rows=dataset_rows,
            )
            if metrics is None:
                continue
            ranked_payloads.append((payload, metrics))
        eligible_payloads = [
            (payload, metrics)
            for payload, metrics in ranked_payloads
            if _descriptive_fit_clears_reconstruction_floor(metrics)
        ]
        if eligible_payloads:
            selected_payload, selected_metrics = min(
                eligible_payloads,
                key=lambda item: _descriptive_fit_reconstruction_sort_key(
                    descriptive_fit=item[0],
                    metrics=item[1],
                ),
            )
            selected_payload["reconstruction_metrics"] = dict(
                _jsonable(selected_metrics)
            )
            selected_payload["reconstruction_floor_cleared"] = True
            return selected_payload
        if ranked_payloads:
            if allow_best_effort:
                selected_payload, selected_metrics = min(
                    ranked_payloads,
                    key=lambda item: _descriptive_fit_reconstruction_sort_key(
                        descriptive_fit=item[0],
                        metrics=item[1],
                    ),
                )
                selected_payload["reconstruction_metrics"] = dict(
                    _jsonable(selected_metrics)
                )
                selected_payload["reconstruction_floor_cleared"] = False
                selected_payload["materialization_reason"] = (
                    "best_effort_benchmark_local_selection"
                )
                return selected_payload
            return None

    selected = _select_descriptive_fit_submitter_result(submitter_results)
    if selected is None or selected.selected_candidate is None:
        return None
    return _build_descriptive_fit_from_submitter_result(
        submitter_result=selected,
        dataset_rows=dataset_rows,
    )


def _descriptive_fit_equation_curve(
    *,
    descriptive_fit: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
    use_observed_lag_values: bool = False,
) -> list[dict[str, Any]] | None:
    equation = descriptive_fit.get("equation")
    candidate_id = _string_or_none(descriptive_fit.get("candidate_id")) or (
        _string_or_none(equation.get("candidate_id"))
        if isinstance(equation, Mapping)
        else None
    )
    family_id = _string_or_none(descriptive_fit.get("family_id")) or (
        _string_or_none(equation.get("family_id"))
        if isinstance(equation, Mapping)
        else None
    )
    parameter_summary = (
        dict(_jsonable(equation.get("parameter_summary")))
        if isinstance(equation, Mapping)
        and isinstance(equation.get("parameter_summary"), Mapping)
        else {}
    )
    literals = (
        dict(_jsonable(equation.get("literals")))
        if isinstance(equation, Mapping) and isinstance(equation.get("literals"), Mapping)
        else None
    )
    state = (
        dict(_jsonable(equation.get("state")))
        if isinstance(equation, Mapping) and isinstance(equation.get("state"), Mapping)
        else None
    )
    if dataset_rows and candidate_id is not None and family_id is not None:
        rebuilt_curve = _build_equation_curve(
            candidate_id=candidate_id,
            family_id=family_id,
            parameter_summary=parameter_summary,
            dataset_rows=dataset_rows,
            literals=literals,
            state=state,
            use_observed_lag_values=use_observed_lag_values,
        )
        if rebuilt_curve:
            return rebuilt_curve
    if isinstance(equation, Mapping):
        curve = equation.get("curve")
        if isinstance(curve, Sequence) and not isinstance(curve, (str, bytes)):
            return list(_jsonable(curve))
    chart_curve = descriptive_fit.get("chart", {}).get("equation_curve")
    if isinstance(chart_curve, Sequence) and not isinstance(
        chart_curve,
        (str, bytes),
    ):
        return list(_jsonable(chart_curve))
    return None


def _descriptive_fit_reconstruction_metrics(
    *,
    descriptive_fit: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | None] | None:
    actual_values = [float(row["observed_value"]) for row in dataset_rows]
    if not actual_values:
        return None
    curve = _descriptive_fit_equation_curve(
        descriptive_fit=descriptive_fit,
        dataset_rows=dataset_rows,
    )
    if curve is None:
        return None
    fitted_values = [
        float(point["fitted_value"])
        for point in curve
        if isinstance(point, Mapping) and point.get("fitted_value") is not None
    ]
    if len(fitted_values) != len(actual_values):
        return None

    absolute_errors = [
        abs(actual - fitted)
        for actual, fitted in zip(actual_values, fitted_values, strict=True)
    ]
    fit_mae = statistics.fmean(absolute_errors)
    fit_squared_error = sum(
        (actual - fitted) ** 2
        for actual, fitted in zip(actual_values, fitted_values, strict=True)
    )
    max_abs_error = max(absolute_errors, default=0.0)
    naive_values = [actual_values[0]] * len(actual_values)
    naive_errors = [
        abs(actual - naive)
        for actual, naive in zip(actual_values, naive_values, strict=True)
    ]
    naive_mae = statistics.fmean(naive_errors)
    actual_mean = statistics.fmean(actual_values)
    mean_baseline_sse = sum((actual - actual_mean) ** 2 for actual in actual_values)
    r2_vs_mean_baseline: float | None
    if mean_baseline_sse > 0.0:
        r2_vs_mean_baseline = 1.0 - (fit_squared_error / mean_baseline_sse)
    elif fit_squared_error == 0.0:
        r2_vs_mean_baseline = 1.0
    else:
        r2_vs_mean_baseline = None
    scale = max(
        max(actual_values) - min(actual_values),
        statistics.fmean(abs(value) for value in actual_values),
        1.0,
    )
    relative_improvement = (
        (naive_mae - fit_mae) / naive_mae if naive_mae > 0.0 else None
    )
    return {
        "sample_size": float(len(actual_values)),
        "fit_mae": fit_mae,
        "max_abs_error": max_abs_error,
        "naive_rollout_mae": naive_mae,
        "relative_improvement_vs_naive_rollout": relative_improvement,
        "normalized_mae": fit_mae / scale,
        "normalized_max_abs_error": max_abs_error / scale,
        "r2_vs_mean_baseline": r2_vs_mean_baseline,
    }


def _descriptive_fit_clears_reconstruction_floor(
    metrics: Mapping[str, float | None],
) -> bool:
    sample_size = int(_float_or_none(metrics.get("sample_size")) or 0)
    normalized_mae = _float_or_none(metrics.get("normalized_mae"))
    normalized_max_abs_error = _float_or_none(
        metrics.get("normalized_max_abs_error")
    )
    r2_vs_mean_baseline = _float_or_none(
        metrics.get("r2_vs_mean_baseline")
    )
    if (
        normalized_mae is None
        or normalized_max_abs_error is None
    ):
        return False
    clears_absolute_error_floor = (
        normalized_mae <= _RECONSTRUCTION_MAX_NORMALIZED_MAE
        and normalized_max_abs_error
        <= _RECONSTRUCTION_MAX_NORMALIZED_MAX_ABS_ERROR
    )
    if not clears_absolute_error_floor:
        return False
    if sample_size < _RECONSTRUCTION_MIN_SAMPLE_SIZE_FOR_VARIANCE_GATE:
        return True
    if r2_vs_mean_baseline is None:
        return False
    return r2_vs_mean_baseline >= _RECONSTRUCTION_MIN_R2_VS_MEAN_BASELINE


def _descriptive_fit_reconstruction_sort_key(
    *,
    descriptive_fit: Mapping[str, Any],
    metrics: Mapping[str, float | None],
) -> tuple[float, float, float, float, float, int, str]:
    candidate_metrics = (
        descriptive_fit.get("metrics")
        if isinstance(descriptive_fit.get("metrics"), Mapping)
        else {}
    )
    relative_improvement = _float_or_none(
        metrics.get("relative_improvement_vs_naive_rollout")
    )
    r2_vs_mean_baseline = _float_or_none(
        metrics.get("r2_vs_mean_baseline")
    )
    return (
        _float_or_none(metrics.get("normalized_mae")) or float("inf"),
        _float_or_none(metrics.get("normalized_max_abs_error")) or float("inf"),
        (
            -(r2_vs_mean_baseline)
            if r2_vs_mean_baseline is not None
            else float("inf")
        ),
        (
            -(relative_improvement)
            if relative_improvement is not None
            else float("inf")
        ),
        _float_or_none(candidate_metrics.get("total_code_bits")) or float("inf"),
        -(
            _float_or_none(candidate_metrics.get("description_gain_bits"))
            or float("-inf")
        ),
        int(candidate_metrics.get("canonical_byte_length") or 2**31),
        _string_or_none(descriptive_fit.get("candidate_id"))
        or _string_or_none(descriptive_fit.get("submitter_id"))
        or "unknown_candidate",
    )


def _select_descriptive_fit_submitter_result(
    submitter_results: Sequence[BenchmarkSubmitterResult],
) -> BenchmarkSubmitterResult | None:
    selected_results = [
        result
        for result in submitter_results
        if result.selected_candidate is not None
        and result.submitter_id != PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID
    ]
    if not selected_results:
        selected_results = [
            result
            for result in submitter_results
            if result.selected_candidate is not None
        ]
    if not selected_results:
        return None
    return min(
        selected_results,
        key=lambda result: (
            _float_or_none(
                (result.selected_candidate_metrics or {}).get("total_code_bits")
            )
            or float("inf"),
            -(
                _float_or_none(
                    (result.selected_candidate_metrics or {}).get(
                        "description_gain_bits"
                    )
                )
                or float("-inf")
            ),
            _float_or_none(
                (result.selected_candidate_metrics or {}).get(
                    "structure_code_bits"
                )
            )
            or float("inf"),
            int(
                (result.selected_candidate_metrics or {}).get(
                    "canonical_byte_length"
                )
                or 2**31
            ),
            _string_or_none(result.selected_candidate_id) or result.submitter_id,
        ),
    )


def _build_descriptive_fit_from_submitter_result(
    *,
    submitter_result: BenchmarkSubmitterResult,
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate = submitter_result.selected_candidate
    if candidate is None:
        raise ValueError("descriptive fit requires a selected candidate")

    candidate_id = _string_or_none(submitter_result.selected_candidate_id) or str(
        candidate.evidence_layer.backend_origin_record.source_candidate_id
    )
    family_id = str(candidate.structural_layer.cir_family_id)
    parameter_summary = _cir_parameter_summary(candidate)
    literals = _cir_literal_summary(candidate)
    state = _cir_state_summary(candidate)
    composition_operator, composition_payload = _cir_composition_summary(candidate)
    equation = build_equation_summary(
        candidate_id=candidate_id,
        family_id=family_id,
        parameter_summary=parameter_summary,
        structure_signature=candidate.canonical_hash(),
        dataset_rows=dataset_rows,
        composition_operator=composition_operator,
        composition_payload=composition_payload,
        literals=literals,
        state=state,
    )
    return {
        "status": "completed",
        "source": "benchmark_local_selection",
        "submitter_id": submitter_result.submitter_id,
        "submitter_class": submitter_result.submitter_class,
        "candidate_id": candidate_id,
        "family_id": family_id,
        "metrics": _jsonable(submitter_result.selected_candidate_metrics or {}),
        "honesty_note": (
            "Descriptive fit from the broader benchmark-local search winner; "
            "not an operator publication."
        ),
        **_descriptive_fit_scope_metadata(candidate),
        "equation": equation,
        "chart": {
            "actual_series": _actual_series(dataset_rows),
            "equation_curve": equation["curve"],
        },
    }


def _cir_parameter_summary(candidate) -> dict[str, float | int]:
    return {
        str(parameter.name): parameter.value
        for parameter in candidate.structural_layer.parameter_block.parameters
    }


def _cir_literal_summary(candidate) -> dict[str, Any]:
    return {
        str(literal.name): literal.value
        for literal in candidate.structural_layer.literal_block.literals
    }


def _cir_state_summary(candidate) -> dict[str, Any]:
    return {
        str(slot.name): slot.value
        for slot in candidate.structural_layer.state_signature.persistent_state.slots
    }


def _cir_composition_summary(
    candidate,
) -> tuple[str | None, Mapping[str, Any] | None]:
    composition_graph = candidate.structural_layer.composition_graph
    payload = composition_graph.as_dict()
    operator_id = _string_or_none(payload.get("operator_id"))
    if operator_id is None:
        return None, None
    return operator_id, payload


def _build_dataset_summary(
    *,
    symbol: str,
    target_id: str,
    dataset_rows: Sequence[Mapping[str, Any]],
    dataset_csv: Path,
    raw_history_path: Path,
) -> dict[str, Any]:
    target = TARGET_SPECS[target_id]
    return {
        "symbol": symbol.upper(),
        "target": dict(target),
        "rows": len(dataset_rows),
        "dataset_csv": str(dataset_csv),
        "raw_history_json": str(raw_history_path),
        "date_range": {
            "start": str(dataset_rows[0]["event_time"]),
            "end": str(dataset_rows[-1]["event_time"]),
        },
        "stats": _dataset_stats_from_rows(dataset_rows),
        "series": _dataset_series(dataset_rows),
    }


def _dataset_stats_from_rows(
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = [float(row["observed_value"]) for row in dataset_rows]
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "latest_value": values[-1],
        "latest_event_time": str(dataset_rows[-1]["event_time"]),
    }


def _dataset_series(
    dataset_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "event_time": str(row["event_time"]),
            "available_at": str(row["available_at"]),
            "observed_value": float(row["observed_value"]),
            **{
                key: row[key]
                for key in row
                if key not in _CANONICAL_DATASET_COLUMNS
                and key
                in {
                    "close",
                    "open",
                    "high",
                    "low",
                    "volume",
                    "previous_close",
                }
            },
        }
        for row in dataset_rows
    ]


def _build_probabilistic_chart(
    *,
    forecast_object_type: str,
    dataset_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    chart: dict[str, Any] = {"actual_series": _actual_series(dataset_rows)}
    if forecast_object_type == "distribution":
        chart["forecast_bands"] = [
            {
                "available_at": row["available_at"],
                "origin_time": row["origin_time"],
                "center": row["location"],
                "lower": row["location"] - row["scale"],
                "upper": row["location"] + row["scale"],
                "realized_observation": row.get("realized_observation"),
            }
            for row in prediction_rows
        ]
    elif forecast_object_type == "interval":
        chart["forecast_bands"] = [
            {
                "available_at": row["available_at"],
                "origin_time": row["origin_time"],
                "lower": row["lower_bound"],
                "upper": row["upper_bound"],
                "realized_observation": row.get("realized_observation"),
            }
            for row in prediction_rows
        ]
    elif forecast_object_type == "quantile":
        chart["forecast_quantiles"] = _jsonable(prediction_rows)
    elif forecast_object_type == "event_probability":
        chart["forecast_probabilities"] = [
            {
                "available_at": row["available_at"],
                "origin_time": row["origin_time"],
                "event_probability": row["event_probability"],
                "realized_event": row.get("realized_event"),
                "event_definition": row.get("event_definition"),
            }
            for row in prediction_rows
        ]
    return chart


def _failure_payload(exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": {
            "type": exc.__class__.__name__,
            "message": str(exc),
        },
    }


def _artifact_body(
    *,
    graph,
    schema_prefix: str,
    default: Any = ...,
) -> Any:
    for manifest in graph.manifests:
        if manifest.ref.schema_name.startswith(schema_prefix):
            return _jsonable(manifest.manifest.body)
    if default is ...:
        raise KeyError(f"artifact graph did not contain {schema_prefix}")
    return default


def _normalize_benchmark_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    benchmark = dict(_jsonable(payload))
    portfolio_selection = benchmark.get("portfolio_selection")
    if not isinstance(portfolio_selection, Mapping):
        return benchmark

    selection = dict(portfolio_selection)
    raw_explanation = selection.get("selection_explanation_raw")
    if raw_explanation is None:
        raw_explanation = selection.get("selection_explanation")
    formatted_explanation = _format_portfolio_selection_explanation(raw_explanation)
    if formatted_explanation is not None:
        selection["selection_explanation"] = formatted_explanation
    selection["selection_explanation_raw"] = _jsonable(raw_explanation)
    benchmark["portfolio_selection"] = selection
    return benchmark


def _summarize_benchmark_descriptive_fit_status(
    *,
    benchmark: Mapping[str, Any],
    descriptive_fit: Any,
) -> dict[str, Any] | None:
    if isinstance(descriptive_fit, Mapping) and descriptive_fit.get("status") == "completed":
        if _string_or_none(descriptive_fit.get("source")) == "legacy_operator_point_fallback":
            return {
                "status": "legacy_compatibility_projection",
                "headline": (
                    "Legacy saved analysis lacks a benchmark-local descriptive fit "
                    "artifact, so the operator point is projected only as a "
                    "compatibility projection fallback."
                ),
                "reason_codes": ["legacy_compatibility_projection"],
            }
        return {
            "status": "available",
            "headline": "Benchmark-local descriptive fit is available for this run.",
            "reason_codes": [],
        }

    submitters = benchmark.get("submitters")
    if not isinstance(submitters, Sequence) or isinstance(submitters, (str, bytes)):
        return None

    submitter_items = [
        submitter
        for submitter in submitters
        if isinstance(submitter, Mapping)
    ]
    if not submitter_items:
        return None

    selected_candidates = [
        submitter
        for submitter in submitter_items
        if submitter.get("selected_candidate_id")
    ]
    materialization_reason = _string_or_none(
        benchmark.get("descriptive_fit_materialization_reason")
    )
    if (
        materialization_reason == "reconstruction_floor_failed"
        and selected_candidates
    ):
        return {
            "status": "absent_reconstruction_floor_failed",
            "headline": (
                "Benchmark submitters produced descriptive candidates, but no "
                "candidate reproduced the path closely enough to materialize a "
                "canonical descriptive fit."
            ),
            "reason_codes": ["reconstruction_floor_failed"],
            "submitters_selected": len(selected_candidates),
        }
    if selected_candidates:
        return {
            "status": "candidate_available_but_not_loaded",
            "headline": (
                "Benchmark submitters selected candidates, but no descriptive fit "
                "was materialized into the saved analysis payload."
            ),
            "reason_codes": [],
        }

    abstained_count = sum(
        1 for submitter in submitter_items if submitter.get("status") == "abstained"
    )
    reason_codes = _collect_candidate_ledger_reason_codes(submitter_items)
    headline = (
        "Benchmark-local search found no admissible descriptive equation for this "
        "target, so the overview should not pretend the operator line is a fit."
    )
    if reason_codes:
        headline += f" Dominant rejection code: {reason_codes[0]}."
    return {
        "status": "absent_no_admissible_candidate",
        "headline": headline,
        "reason_codes": reason_codes,
        "submitters_abstained": abstained_count,
    }


def _load_descriptive_fit_from_cached_benchmark(
    *,
    analysis: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not dataset_rows:
        return None
    benchmark_root = _string_or_none(
        analysis.get("benchmark", {}).get("benchmark_root")
    )
    if not benchmark_root:
        return None
    cache_root = (
        Path(benchmark_root).expanduser()
        / "results"
        / str(analysis.get("benchmark", {}).get("track_id") or "")
        / str(analysis.get("benchmark", {}).get("task_id") or "")
        / "_profile_runtime"
        / "caches"
        / "submitter-results"
    )
    if not cache_root.is_dir():
        return None

    submitter_results = []
    for submitter_id in (
        PORTFOLIO_ORCHESTRATOR_SUBMITTER_ID,
        str(
            analysis.get("benchmark", {})
            .get("portfolio_selection", {})
            .get("winner_submitter_id")
            or ""
        ),
        str(analysis.get("benchmark", {}).get("local_winner_submitter_id") or ""),
        "analytic_backend",
        "recursive_spectral_backend",
        "algorithmic_search_backend",
    ):
        if not submitter_id:
            continue
        loaded = _load_cached_submitter_result(cache_root / f"{submitter_id}.pkl")
        if loaded is not None:
            submitter_results.append(loaded)
    if not submitter_results:
        return None
    return _build_descriptive_fit_from_submitter_results(
        submitter_results=submitter_results,
        dataset_rows=dataset_rows,
    )


def _resolve_descriptive_fit_payload(
    *,
    analysis: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    benchmark = analysis.get("benchmark")
    if isinstance(benchmark, Mapping):
        benchmark_descriptive_fit = benchmark.get("descriptive_fit")
        if isinstance(benchmark_descriptive_fit, Mapping):
            return dict(_jsonable(benchmark_descriptive_fit))

        descriptive_fit = _load_descriptive_fit_from_cached_benchmark(
            analysis=analysis,
            dataset_rows=dataset_rows,
        )
        if descriptive_fit is not None:
            return descriptive_fit

        if _is_legacy_saved_analysis(analysis):
            descriptive_fit = _project_legacy_descriptive_fit_from_operator_point(
                analysis=analysis,
                dataset_rows=dataset_rows,
            )
            if descriptive_fit is not None:
                return descriptive_fit

    existing = analysis.get("descriptive_fit")
    if isinstance(existing, Mapping):
        return dict(_jsonable(existing))
    return None


def _project_legacy_descriptive_fit_from_operator_point(
    *,
    analysis: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not _is_legacy_saved_analysis(analysis):
        return None
    if not dataset_rows:
        return None
    benchmark = analysis.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return None
    descriptive_fit_status = benchmark.get("descriptive_fit_status")
    if not isinstance(descriptive_fit_status, Mapping):
        return None
    if (
        _string_or_none(descriptive_fit_status.get("status"))
        != "candidate_available_but_not_loaded"
    ):
        return None

    operator_point = analysis.get("operator_point")
    if not isinstance(operator_point, Mapping):
        return None
    if operator_point.get("status") != "completed":
        return None

    equation_payload = operator_point.get("equation")
    if not isinstance(equation_payload, Mapping):
        return None

    chart_payload = (
        dict(_jsonable(operator_point.get("chart")))
        if isinstance(operator_point.get("chart"), Mapping)
        else {}
    )
    equation_curve = (
        chart_payload.get("equation_curve")
        if isinstance(chart_payload.get("equation_curve"), Sequence)
        and not isinstance(chart_payload.get("equation_curve"), (str, bytes))
        else equation_payload.get("curve")
    )
    if not isinstance(equation_curve, Sequence) or isinstance(
        equation_curve, (str, bytes)
    ):
        equation_curve = []

    candidate_id = _string_or_none(operator_point.get("candidate_id")) or _string_or_none(
        equation_payload.get("candidate_id")
    )
    family_id = _string_or_none(operator_point.get("selected_family")) or _string_or_none(
        equation_payload.get("family_id")
    )

    return {
        "status": "completed",
        "source": "legacy_operator_point_fallback",
        "submitter_id": "legacy_operator_point_fallback",
        "submitter_class": "legacy_compatibility_projection",
        "candidate_id": candidate_id,
        "family_id": family_id,
        "metrics": {},
        "honesty_note": (
            "Legacy saved analysis predates descriptive-bank payloads. "
            "Projecting the operator point equation as a descriptive-only "
            "fallback for display; not a publishable law claim."
        ),
        "law_eligible": False,
        "law_rejection_reason_codes": ["legacy_compatibility_projection"],
        "equation": dict(_jsonable(equation_payload)),
        "chart": {
            "actual_series": _actual_series(dataset_rows),
            "equation_curve": list(_jsonable(equation_curve)),
        },
    }


def _is_legacy_saved_analysis(analysis: Mapping[str, Any]) -> bool:
    analysis_version = _string_or_none(analysis.get("analysis_version"))
    if analysis_version is None:
        return _has_legacy_descriptive_projection_marker(analysis)
    major, minor, patch = _parse_version_triplet(analysis_version)
    if major is None:
        return _has_legacy_descriptive_projection_marker(analysis)
    return (major, minor, patch) < (1, 0, 0)


def _parse_version_triplet(value: str) -> tuple[int | None, int, int]:
    parts = value.strip().split(".")
    if not parts:
        return None, 0, 0
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        return None, 0, 0
    return major, minor, patch


def _has_legacy_descriptive_projection_marker(
    analysis: Mapping[str, Any],
) -> bool:
    benchmark = analysis.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return False
    descriptive_fit_status = benchmark.get("descriptive_fit_status")
    if not isinstance(descriptive_fit_status, Mapping):
        return False
    return (
        _string_or_none(descriptive_fit_status.get("status"))
        == "candidate_available_but_not_loaded"
    )


def _descriptive_fit_scope_metadata(candidate) -> dict[str, Any]:
    evidence_layer = getattr(candidate, "evidence_layer", None)
    diagnostics = getattr(evidence_layer, "transient_diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        return {}
    descriptive_scope = diagnostics.get("descriptive_scope")
    if not isinstance(descriptive_scope, Mapping):
        return {}
    payload: dict[str, Any] = {}
    selection_scope = _string_or_none(descriptive_scope.get("source"))
    selection_rule = _string_or_none(descriptive_scope.get("selection_rule"))
    law_eligible = descriptive_scope.get("law_eligible")
    law_rejection_reason_codes = _string_list(
        descriptive_scope.get("law_rejection_reason_codes")
    )
    if selection_scope is not None:
        payload["selection_scope"] = selection_scope
    if selection_rule is not None:
        payload["selection_rule"] = selection_rule
    if isinstance(law_eligible, bool):
        payload["law_eligible"] = law_eligible
    payload["law_rejection_reason_codes"] = law_rejection_reason_codes
    return payload


def _collect_candidate_ledger_reason_codes(
    submitters: Sequence[Mapping[str, Any]],
) -> list[str]:
    ledger_counter: Counter[str] = Counter()
    diagnostic_counter: Counter[str] = Counter()
    for submitter in submitters:
        ledger = submitter.get("candidate_ledger")
        if not isinstance(ledger, Sequence) or isinstance(ledger, (str, bytes)):
            continue
        for entry in ledger:
            if not isinstance(entry, Mapping):
                continue
            for reason_code in entry.get("reason_codes", []) or []:
                if reason_code:
                    ledger_counter[str(reason_code)] += 1
            details = entry.get("details")
            if not isinstance(details, Mapping):
                continue
            diagnostics = details.get("diagnostics")
            if not isinstance(diagnostics, Sequence) or isinstance(
                diagnostics,
                (str, bytes),
            ):
                continue
            for diagnostic in diagnostics:
                if not isinstance(diagnostic, Mapping):
                    continue
                for reason_code in diagnostic.get("reason_codes", []) or []:
                    if reason_code:
                        diagnostic_counter[str(reason_code)] += 1
    chosen = diagnostic_counter if diagnostic_counter else ledger_counter
    return [reason_code for reason_code, _ in chosen.most_common()]


def _load_cached_submitter_result(
    cache_path: Path,
) -> BenchmarkSubmitterResult | None:
    try:
        payload = pickle.loads(cache_path.read_bytes())
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    result = payload.get("payload")
    if not isinstance(result, BenchmarkSubmitterResult):
        return None
    return result


def _normalize_probabilistic_lane(
    *,
    analysis: Mapping[str, Any],
    forecast_object_type: str,
    lane_payload: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lane = dict(_jsonable(lane_payload))
    equation_payload = lane.get("equation")
    if isinstance(equation_payload, Mapping):
        lane["equation"] = _normalize_equation_payload(
            equation_payload,
            family_id=_string_or_none(lane.get("selected_family")),
        )
    if lane.get("status") != "failed":
        return _enrich_probabilistic_lane(
            forecast_object_type=forecast_object_type,
            lane=lane,
        )

    error = lane.get("error")
    if not isinstance(error, Mapping):
        return lane
    error_message = _string_or_none(error.get("message")) or ""
    if "canonical_structure_code_manifest@" not in error_message:
        return lane

    workspace_root = Path(str(analysis.get("workspace_root") or "")).expanduser()
    if not workspace_root.exists():
        return lane

    symbol = str(analysis.get("dataset", {}).get("symbol") or "").strip()
    target_id = str(
        analysis.get("dataset", {}).get("target", {}).get("id")
        or analysis.get("request", {}).get("target_id")
        or ""
    ).strip()
    if not symbol or not target_id or not dataset_rows:
        return lane

    run_id = (
        f"workbench-{_slug(symbol)}-{_slug(target_id)}-{_slug(forecast_object_type)}"
    )
    output_root = workspace_root / "runs" / forecast_object_type
    if not output_root.exists():
        return lane

    try:
        graph = load_demo_run_artifact_graph(output_root=output_root, run_id=run_id)
        prediction = inspect_demo_probabilistic_prediction(output_root=output_root)
        calibration = inspect_demo_calibration(output_root=output_root)
        reducer_body = _artifact_body(
            graph=graph,
            schema_prefix="reducer_artifact_manifest@",
        )
        structure_body = _artifact_body(
            graph=graph,
            schema_prefix="canonical_structure_code_manifest@",
            default=None,
        )
    except Exception:
        return lane

    run_summary_path = (
        output_root / "sealed-runs" / run_id / "run-summary.json"
    )
    run_summary: dict[str, Any] = {}
    if run_summary_path.is_file():
        loaded = json.loads(run_summary_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            run_summary = loaded

    selected_family = _string_or_none(run_summary.get("selected_family")) or _string_or_none(
        lane.get("selected_family")
    )
    if not selected_family:
        selected_family = _candidate_family_from_graph(
            graph=graph,
            candidate_id=prediction.candidate_id,
        )

    equation = build_equation_summary(
        candidate_id=prediction.candidate_id,
        family_id=selected_family or "unknown",
        parameter_summary=dict(reducer_body.get("parameter_summary", {})),
        structure_signature=_resolve_structure_signature(
            structure_body=structure_body,
            reducer_body=reducer_body,
        ),
        dataset_rows=dataset_rows,
    )

    manifest_path = workspace_root / "manifests" / f"{forecast_object_type}.yaml"
    repaired = {
        "status": "completed",
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "selected_family": selected_family,
        "forecast_object_type": forecast_object_type,
        "replay_verification": lane.get("replay_verification"),
        "aggregated_primary_score": prediction.aggregated_primary_score,
        "equation": equation,
        "rows": _jsonable(prediction.rows),
        "latest_row": (
            _jsonable(prediction.rows[-1]) if prediction.rows else None
        ),
        "calibration": _jsonable(
            {
                "status": calibration.status,
                "passed": calibration.passed,
                "failure_reason_code": calibration.failure_reason_code,
                "gate_effect": calibration.gate_effect,
                "diagnostics": calibration.diagnostics,
            }
        ),
        "chart": _build_probabilistic_chart(
            forecast_object_type=forecast_object_type,
            dataset_rows=dataset_rows,
            prediction_rows=prediction.rows,
        ),
    }
    return _enrich_probabilistic_lane(
        forecast_object_type=forecast_object_type,
        lane=repaired,
    )


def _enrich_probabilistic_lane(
    *,
    forecast_object_type: str,
    lane: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(_jsonable(lane))
    search_scope = _summarize_search_scope(
        manifest_path=_string_or_none(normalized.get("manifest_path")),
        lane_kind=forecast_object_type,
    )
    if search_scope is not None:
        normalized["search_scope"] = search_scope
    normalized["evidence"] = _build_probabilistic_evidence_summary(normalized)
    return normalized


def _load_dataset_rows_from_analysis(
    analysis: Mapping[str, Any],
) -> list[dict[str, Any]]:
    dataset_csv = _string_or_none(analysis.get("dataset", {}).get("dataset_csv"))
    if not dataset_csv:
        return []
    dataset_path = Path(dataset_csv).expanduser()
    if not dataset_path.is_file():
        return []

    rows: list[dict[str, Any]] = []
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row: dict[str, Any] = {}
            for key, value in raw_row.items():
                if value is None:
                    row[key] = None
                elif key in {
                    "source_id",
                    "series_id",
                    "event_time",
                    "available_at",
                    "revision_id",
                }:
                    row[key] = value
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            rows.append(row)
    return rows


def _build_change_atlas(
    *,
    analysis: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    price_rows = _load_price_rows_for_change_atlas(
        analysis=analysis,
        dataset_rows=dataset_rows,
    )
    if len(price_rows) < 2:
        return None

    historical = _build_historical_change_atlas(price_rows)
    forecast = _build_forecast_change_atlas(
        analysis=analysis,
        price_rows=price_rows,
    )
    horizons = _change_atlas_horizons(
        historical=historical,
        forecast=forecast,
    )
    if not horizons:
        return None

    return {
        "status": "completed",
        "headline": forecast["headline"],
        "metrics": [dict(spec) for spec in CHANGE_METRIC_SPECS.values()],
        "horizons": horizons,
        "historical": historical,
        "forecast": forecast,
    }


def _load_price_rows_for_change_atlas(
    *,
    analysis: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_history_path = _string_or_none(
        analysis.get("dataset", {}).get("raw_history_json")
    )
    symbol = _string_or_none(analysis.get("dataset", {}).get("symbol")) or "SERIES"
    if raw_history_path:
        candidate = Path(raw_history_path).expanduser()
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(payload, Sequence) and not isinstance(
                    payload, (str, bytes)
                ):
                    raw_rows = build_csv_rows_from_fmp_history(payload, symbol=symbol)
                    return [
                        {
                            "event_time": str(row["event_time"]),
                            "available_at": str(row["available_at"]),
                            "close": float(row["observed_value"]),
                        }
                        for row in raw_rows
                    ]
            except Exception:
                pass

    target_id = str(
        analysis.get("dataset", {}).get("target", {}).get("id")
        or analysis.get("request", {}).get("target_id")
        or DEFAULT_TARGET_ID
    )
    price_rows: list[dict[str, Any]] = []
    for row in dataset_rows:
        close_value = _float_or_none(row.get("close"))
        if close_value is None and target_id == "price_close":
            close_value = _float_or_none(row.get("observed_value"))
        event_time = _string_or_none(row.get("event_time"))
        available_at = _string_or_none(row.get("available_at"))
        if close_value is None or event_time is None or available_at is None:
            continue
        price_rows.append(
            {
                "event_time": event_time,
                "available_at": available_at,
                "close": close_value,
            }
        )
    return price_rows


def _build_historical_change_atlas(
    price_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    historical: dict[str, dict[str, Any]] = {}
    for metric_id in CHANGE_METRIC_SPECS:
        per_horizon: dict[str, Any] = {}
        for horizon in CHANGE_ATLAS_HORIZONS:
            summary = _build_historical_change_summary(
                price_rows=price_rows,
                metric_id=metric_id,
                horizon=horizon,
            )
            if summary is not None:
                per_horizon[str(horizon)] = summary
        historical[metric_id] = per_horizon
    return historical


def _build_historical_change_summary(
    *,
    price_rows: Sequence[Mapping[str, Any]],
    metric_id: str,
    horizon: int,
) -> dict[str, Any] | None:
    samples: list[dict[str, Any]] = []
    for origin_index in range(len(price_rows) - horizon):
        origin = price_rows[origin_index]
        realized = price_rows[origin_index + horizon]
        origin_close = _float_or_none(origin.get("close"))
        realized_close = _float_or_none(realized.get("close"))
        if origin_close is None or realized_close is None:
            continue
        projected = _project_change_value(
            metric_id=metric_id,
            origin_close=origin_close,
            target_value=realized_close,
        )
        if projected is None:
            continue
        samples.append(
            {
                "origin_time": str(origin["event_time"]),
                "available_at": str(realized["available_at"]),
                "origin_close": origin_close,
                "realized_close": realized_close,
                "value": projected,
            }
        )
    if not samples:
        return None

    values = [sample["value"] for sample in samples]
    latest = samples[-1]
    return {
        "horizon": horizon,
        "sample_size": len(samples),
        "latest_value": latest["value"],
        "latest_origin_time": latest["origin_time"],
        "latest_available_at": latest["available_at"],
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "quantiles": _quantile_summary(values),
        "histogram": _histogram_summary(values),
    }


def _build_forecast_change_atlas(
    *,
    analysis: Mapping[str, Any],
    price_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_id = str(
        analysis.get("dataset", {}).get("target", {}).get("id")
        or analysis.get("request", {}).get("target_id")
        or DEFAULT_TARGET_ID
    )
    probabilistic = analysis.get("probabilistic")
    if not isinstance(probabilistic, Mapping):
        return {
            "support": "historical_only",
            "headline": "No probabilistic lanes were available to project into change space.",
            "lanes": {},
        }
    if target_id != "price_close":
        return {
            "support": "historical_only",
            "headline": (
                "Forecast change projection is only available when the active target "
                "is raw price levels."
            ),
            "lanes": {},
        }

    close_by_time = {
        str(row["event_time"]): float(row["close"])
        for row in price_rows
        if row.get("event_time") is not None and row.get("close") is not None
    }
    lanes: dict[str, Any] = {}
    for lane_kind, lane_payload in probabilistic.items():
        if not isinstance(lane_payload, Mapping) or lane_payload.get("status") != "completed":
            continue
        projected_lane = _project_lane_change_views(
            lane_kind=str(lane_kind),
            lane_payload=lane_payload,
            close_by_time=close_by_time,
        )
        if projected_lane:
            lanes[str(lane_kind)] = projected_lane
    if lanes:
        return {
            "support": "price_projection",
            "headline": "Forecast lanes are shown as changes from the origin close.",
            "lanes": lanes,
        }
    return {
        "support": "historical_only",
        "headline": "Probabilistic rows were available, but none could be projected into change space.",
        "lanes": {},
    }


def _project_lane_change_views(
    *,
    lane_kind: str,
    lane_payload: Mapping[str, Any],
    close_by_time: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    rows = lane_payload.get("rows")
    row_items = (
        list(rows)
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes))
        else []
    )
    projected: dict[str, dict[str, Any]] = {}
    for metric_id in CHANGE_METRIC_SPECS:
        per_horizon: dict[str, Any] = {}
        for row in row_items:
            if not isinstance(row, Mapping):
                continue
            horizon = row.get("horizon")
            if horizon is None:
                continue
            try:
                normalized_horizon = int(horizon)
            except (TypeError, ValueError):
                continue
            if normalized_horizon not in CHANGE_ATLAS_HORIZONS:
                continue
            origin_time = _string_or_none(row.get("origin_time"))
            if origin_time is None:
                continue
            origin_close = _float_or_none(close_by_time.get(origin_time))
            if origin_close is None:
                continue
            projected_row = _project_probabilistic_row_to_change(
                lane_kind=lane_kind,
                row=row,
                metric_id=metric_id,
                origin_close=origin_close,
            )
            if projected_row is not None:
                per_horizon[str(normalized_horizon)] = projected_row
        if per_horizon:
            projected[metric_id] = per_horizon
    return projected


def _project_probabilistic_row_to_change(
    *,
    lane_kind: str,
    row: Mapping[str, Any],
    metric_id: str,
    origin_close: float,
) -> dict[str, Any] | None:
    base = {
        "horizon": int(row["horizon"]),
        "origin_time": str(row["origin_time"]),
        "available_at": str(row.get("available_at") or ""),
        "origin_close": origin_close,
    }
    if lane_kind == "distribution":
        location = _float_or_none(row.get("location"))
        scale = _float_or_none(row.get("scale"))
        if location is None or scale is None:
            return None
        lower = location - scale
        upper = location + scale
        return {
            **base,
            "center": _project_change_value(
                metric_id=metric_id,
                origin_close=origin_close,
                target_value=location,
            ),
            "lower": _project_change_value(
                metric_id=metric_id,
                origin_close=origin_close,
                target_value=lower,
            ),
            "upper": _project_change_value(
                metric_id=metric_id,
                origin_close=origin_close,
                target_value=upper,
            ),
            "realized": _project_change_value(
                metric_id=metric_id,
                origin_close=origin_close,
                target_value=_float_or_none(row.get("realized_observation")),
            ),
        }
    if lane_kind == "interval":
        lower_bound = _float_or_none(row.get("lower_bound"))
        upper_bound = _float_or_none(row.get("upper_bound"))
        if lower_bound is None or upper_bound is None:
            return None
        center_value = lower_bound + ((upper_bound - lower_bound) / 2.0)
        return {
            **base,
            "center": _project_change_value(
                metric_id=metric_id,
                origin_close=origin_close,
                target_value=center_value,
            ),
            "lower": _project_change_value(
                metric_id=metric_id,
                origin_close=origin_close,
                target_value=lower_bound,
            ),
            "upper": _project_change_value(
                metric_id=metric_id,
                origin_close=origin_close,
                target_value=upper_bound,
            ),
            "realized": _project_change_value(
                metric_id=metric_id,
                origin_close=origin_close,
                target_value=_float_or_none(row.get("realized_observation")),
            ),
        }
    if lane_kind == "quantile":
        quantiles_payload = row.get("quantiles")
        quantiles = (
            list(quantiles_payload)
            if isinstance(quantiles_payload, Sequence)
            and not isinstance(quantiles_payload, (str, bytes))
            else []
        )
        converted: list[dict[str, Any]] = []
        for quantile in quantiles:
            if not isinstance(quantile, Mapping):
                continue
            level = _float_or_none(quantile.get("level"))
            value = _float_or_none(quantile.get("value"))
            if level is None or value is None:
                continue
            converted_value = _project_change_value(
                metric_id=metric_id,
                origin_close=origin_close,
                target_value=value,
            )
            if converted_value is None:
                continue
            converted.append({"level": level, "value": converted_value})
        if not converted:
            return None
        median = next(
            (item["value"] for item in converted if item["level"] == 0.5),
            converted[len(converted) // 2]["value"],
        )
        return {
            **base,
            "center": median,
            "quantiles": converted,
            "realized": _project_change_value(
                metric_id=metric_id,
                origin_close=origin_close,
                target_value=_float_or_none(row.get("realized_observation")),
            ),
        }
    if lane_kind == "event_probability":
        probability = _float_or_none(row.get("event_probability"))
        if probability is None:
            return None
        return {
            **base,
            "probability": probability,
            "realized_event": row.get("realized_event"),
            "event_definition": row.get("event_definition"),
        }
    return None


def _project_change_value(
    *,
    metric_id: str,
    origin_close: float | None,
    target_value: float | None,
) -> float | None:
    if origin_close is None or target_value is None:
        return None
    if metric_id == "delta":
        return float(target_value - origin_close)
    if origin_close == 0:
        return None
    ratio = float(target_value / origin_close)
    if metric_id == "return":
        return ratio - 1.0
    if metric_id == "log_return":
        if ratio <= 0:
            return None
        return math.log(ratio)
    raise AssertionError(f"unsupported change metric {metric_id}")


def _change_atlas_horizons(
    *,
    historical: Mapping[str, Mapping[str, Any]],
    forecast: Mapping[str, Any],
) -> list[int]:
    horizons: set[int] = set()
    for per_metric in historical.values():
        if not isinstance(per_metric, Mapping):
            continue
        for horizon in per_metric:
            try:
                horizons.add(int(horizon))
            except (TypeError, ValueError):
                continue
    lanes = forecast.get("lanes")
    if isinstance(lanes, Mapping):
        for per_lane in lanes.values():
            if not isinstance(per_lane, Mapping):
                continue
            for per_metric in per_lane.values():
                if not isinstance(per_metric, Mapping):
                    continue
                for horizon in per_metric:
                    try:
                        horizons.add(int(horizon))
                    except (TypeError, ValueError):
                        continue
    return sorted(horizons)


def _quantile_summary(values: Sequence[float]) -> list[dict[str, float]]:
    return [
        {"level": level, "value": _percentile(values, level)}
        for level in (0.1, 0.5, 0.9)
    ]


def _percentile(values: Sequence[float], level: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, level)) * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = position - lower_index
    return lower_value + ((upper_value - lower_value) * weight)


def _histogram_summary(values: Sequence[float]) -> list[dict[str, float | int]]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return []
    if len(ordered) == 1 or ordered[0] == ordered[-1]:
        anchor = ordered[0]
        return [
            {
                "lower": anchor,
                "upper": anchor,
                "count": len(ordered),
            }
        ]
    bucket_count = min(16, max(4, int(math.sqrt(len(ordered))) + 1))
    lower_bound = ordered[0]
    upper_bound = ordered[-1]
    width = (upper_bound - lower_bound) / bucket_count
    buckets = [
        {
            "lower": lower_bound + (width * index),
            "upper": lower_bound + (width * (index + 1)),
            "count": 0,
        }
        for index in range(bucket_count)
    ]
    for value in ordered:
        if value == upper_bound:
            buckets[-1]["count"] += 1
            continue
        index = int((value - lower_bound) / width)
        buckets[max(0, min(bucket_count - 1, index))]["count"] += 1
    return buckets


def _candidate_family_from_graph(*, graph, candidate_id: str) -> str | None:
    for manifest in graph.manifests:
        if not manifest.ref.schema_name.startswith("candidate_spec@"):
            continue
        body = manifest.manifest.body
        if not isinstance(body, Mapping):
            continue
        if _string_or_none(body.get("candidate_id")) != candidate_id:
            continue
        return _string_or_none(body.get("family_id"))
    return None


def _normalize_equation_payload(
    payload: Mapping[str, Any],
    *,
    candidate_id: str | None = None,
    family_id: str | None = None,
) -> dict[str, Any]:
    equation = dict(_jsonable(payload))
    resolved_candidate_id = _string_or_none(equation.get("candidate_id")) or candidate_id
    resolved_family_id = _string_or_none(equation.get("family_id")) or family_id
    if resolved_candidate_id is not None:
        equation["candidate_id"] = resolved_candidate_id
    if resolved_family_id is not None:
        equation["family_id"] = resolved_family_id

    params = (
        dict(equation.get("parameter_summary", {}))
        if isinstance(equation.get("parameter_summary"), Mapping)
        else {}
    )
    intercept = _float_or_none(params.get("intercept"))
    lag_coefficient = _float_or_none(params.get("lag_coefficient"))
    delta_form_label = _delta_form_label(
        candidate_id=resolved_candidate_id,
        family_id=resolved_family_id,
        intercept=intercept,
        lag_coefficient=lag_coefficient,
    )
    if delta_form_label is not None:
        equation["delta_form_label"] = delta_form_label
    return equation


def _delta_form_label(
    *,
    candidate_id: str | None,
    family_id: str | None,
    intercept: float | None,
    lag_coefficient: float | None,
) -> str | None:
    if candidate_id == "algorithmic_last_observation":
        return "Δy(t) = 0"
    if (
        (candidate_id == "analytic_lag1_affine" or family_id == "analytic")
        and intercept is not None
        and lag_coefficient is not None
    ):
        return _format_affine_label(
            lhs="Δy(t)",
            intercept=intercept,
            coefficient=lag_coefficient - 1.0,
            term="y(t-1)",
        )
    return None


def _format_affine_label(
    *,
    lhs: str,
    intercept: float,
    coefficient: float,
    term: str,
) -> str:
    sign = "+" if coefficient >= 0 else "-"
    return (
        f"{lhs} = {_format_number(intercept)} {sign} "
        f"{_format_number(abs(coefficient))}*{term}"
    )


def _holistic_equation_label(
    *,
    compact_label: str,
    row_count: int,
    coefficient_vector_label: str,
) -> str:
    upper_index = max(0, row_count - 1)
    compact_rhs = _equation_rhs(compact_label) or "0"
    label = (
        "y(t) = "
        f"\\left({compact_rhs}\\right) + "
        f"\\sum_{{i=0}}^{{{upper_index}}} c_i "
        "\\prod_{j \\ne i}\\frac{\\tau(t)-j}{i-j},"
        "\\quad \\tau(t_i)=i"
    )
    if coefficient_vector_label.startswith("c="):
        label += f",\\quad {coefficient_vector_label}"
    return label


def _holistic_residual_label(*, row_count: int) -> str:
    upper_index = max(0, row_count - 1)
    return (
        f"\\sum_{{i=0}}^{{{upper_index}}} c_i "
        "\\prod_{j \\ne i}\\frac{\\tau(t)-j}{i-j}"
    )


def _holistic_coefficient_vector_label(
    *,
    residual_coefficients: Sequence[float],
    inline: bool,
) -> str:
    if inline:
        values = ", ".join(_format_number(value) for value in residual_coefficients)
        return f"c=\\left[{values}\\right]"
    return "c_i = y_i - \\hat{y}^{compact}_i"


def _equation_rhs(label: str | None) -> str | None:
    normalized = _string_or_none(label)
    if normalized is None:
        return None
    if "=" not in normalized:
        return normalized
    _lhs, rhs = normalized.split("=", 1)
    return rhs.strip()


def _build_probabilistic_evidence_summary(
    lane: Mapping[str, Any],
) -> dict[str, Any]:
    rows = lane.get("rows")
    row_items = list(rows) if isinstance(rows, Sequence) else []
    diagnostics_payload = lane.get("calibration", {}).get("diagnostics", [])
    diagnostics = (
        list(diagnostics_payload)
        if isinstance(diagnostics_payload, Sequence)
        and not isinstance(diagnostics_payload, (str, bytes))
        else [diagnostics_payload]
        if diagnostics_payload
        else []
    )

    sample_sizes = [
        int(diagnostic.get("sample_size"))
        for diagnostic in diagnostics
        if isinstance(diagnostic, Mapping)
        and diagnostic.get("sample_size") is not None
    ]
    sample_size = max(sample_sizes) if sample_sizes else len(row_items)
    origin_count = len(
        {
            str(row.get("origin_time"))
            for row in row_items
            if isinstance(row, Mapping) and row.get("origin_time") is not None
        }
    )
    horizon_count = len(
        {
            str(row.get("horizon"))
            for row in row_items
            if isinstance(row, Mapping) and row.get("horizon") is not None
        }
    )
    if sample_size < 5 or origin_count < 3:
        strength = "thin"
        headline = (
            f"Only {sample_size} realized forecasts across {origin_count or 0} "
            "origins. Calibration can pass here without proving robustness."
        )
    elif sample_size < 20 or origin_count < 10:
        strength = "limited"
        headline = (
            f"{sample_size} realized forecasts across {origin_count or 0} origins. "
            "Useful for inspection, but still a small calibration sample."
        )
    else:
        strength = "substantial"
        headline = (
            f"{sample_size} realized forecasts across {origin_count or 0} origins. "
            "Calibration evidence is large enough to read more seriously."
        )
    return {
        "strength": strength,
        "headline": headline,
        "sample_size": sample_size,
        "origin_count": origin_count,
        "horizon_count": horizon_count,
    }


def humanize_lane_kind(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def _render_equation_label(
    *,
    candidate_id: str,
    family_id: str,
    parameter_summary: Mapping[str, float | int],
    literals: Mapping[str, Any] | None = None,
) -> str | None:
    literal_values = dict(literals or {})
    if candidate_id == _DESCRIPTIVE_RECONSTRUCTION_CANDIDATE_ID:
        harmonic_count = int(literal_values.get("harmonic_count", 0))
        sample_size = int(literal_values.get("sample_size", 0))
        return (
            "y(t) = "
            f"{_format_number(parameter_summary.get('mean_term', 0.0))} + "
            f"\\sum_{{k=1}}^{{{harmonic_count}}}"
            "\\left[a_k\\cos\\left(2\\pi k t / "
            f"{sample_size}"
            "\\right) + b_k\\sin\\left(2\\pi k t / "
            f"{sample_size}"
            "\\right)\\right]"
        )
    if family_id in {"constant"} and "intercept" in parameter_summary:
        return f"y(t) = {_format_number(parameter_summary['intercept'])}"
    if family_id in {"drift", "linear_trend"} and {
        "intercept",
        "slope",
    } <= parameter_summary.keys():
        return (
            f"y(t) = {_format_number(parameter_summary['intercept'])} + "
            f"{_format_number(parameter_summary['slope'])}*t"
        )
    if (
        candidate_id == "analytic_lag1_affine" or family_id == "analytic"
    ) and {"intercept", "lag_coefficient"} <= parameter_summary.keys():
        return (
            f"y(t) = {_format_number(parameter_summary['intercept'])} + "
            f"{_format_number(parameter_summary['lag_coefficient'])}*y(t-1)"
        )
    if candidate_id == "algorithmic_last_observation":
        return "y(t) = y(t-1)"
    if candidate_id == "algorithmic_running_half_average":
        return "y(t) = 0.5*y(t-1) + 0.5*x(t-1)"
    if candidate_id == "recursive_level_smoother" and "alpha" in literal_values:
        alpha = float(literal_values["alpha"])
        return (
            f"level(t) = {_format_number(alpha)}*x(t-1) + "
            f"{_format_number(1.0 - alpha)}*level(t-1)"
        )
    if candidate_id == "recursive_running_mean":
        return "y(t) = mean(x[0:t-1])"
    if family_id == "spectral" and {
        "cosine_coefficient",
        "sine_coefficient",
    } <= parameter_summary.keys():
        harmonic = int(literal_values.get("harmonic", 1))
        season_length = int(literal_values.get("season_length", 5))
        return (
            "y(t) = "
            f"{_format_number(parameter_summary['cosine_coefficient'])}"
            f"*cos(2π*{harmonic}*t/{season_length}) + "
            f"{_format_number(parameter_summary['sine_coefficient'])}"
            f"*sin(2π*{harmonic}*t/{season_length})"
        )
    if family_id == "seasonal_naive":
        period = int(parameter_summary.get("seasonal_period", 5))
        return f"y(t) = y(t-{period})"
    return None


def _build_equation_curve(
    *,
    candidate_id: str,
    family_id: str,
    parameter_summary: Mapping[str, float | int],
    dataset_rows: Sequence[Mapping[str, Any]],
    literals: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    use_observed_lag_values: bool = False,
) -> list[dict[str, Any]]:
    if not dataset_rows:
        return []
    observed_values = [float(row["observed_value"]) for row in dataset_rows]
    timestamps = [str(row["event_time"]) for row in dataset_rows]
    literal_values = dict(literals or {})
    state_values = dict(state or {})
    fitted_values: list[float]

    if candidate_id == _DESCRIPTIVE_RECONSTRUCTION_CANDIDATE_ID:
        harmonic_count = int(literal_values.get("harmonic_count", 0))
        sample_size = max(int(literal_values.get("sample_size", len(observed_values))), 1)
        mean_term = float(parameter_summary.get("mean_term", 0.0))
        cosine_coefficients = [
            float(value)
            for value in literal_values.get("cosine_coefficients", [])
        ]
        sine_coefficients = [
            float(value)
            for value in literal_values.get("sine_coefficients", [])
        ]
        fitted_values = []
        for index in range(len(observed_values)):
            fitted_value = mean_term
            for harmonic in range(1, harmonic_count + 1):
                angle = (2.0 * math.pi * harmonic * index) / sample_size
                cosine = (
                    cosine_coefficients[harmonic - 1]
                    if harmonic - 1 < len(cosine_coefficients)
                    else 0.0
                )
                sine = (
                    sine_coefficients[harmonic - 1]
                    if harmonic - 1 < len(sine_coefficients)
                    else 0.0
                )
                fitted_value += (cosine * math.cos(angle)) + (
                    sine * math.sin(angle)
                )
            fitted_values.append(fitted_value)
    elif family_id == "constant" and "intercept" in parameter_summary:
        fitted_values = [float(parameter_summary["intercept"])] * len(observed_values)
    elif family_id in {"drift", "linear_trend"} and {
        "intercept",
        "slope",
    } <= parameter_summary.keys():
        intercept = float(parameter_summary["intercept"])
        slope = float(parameter_summary["slope"])
        fitted_values = [
            intercept + (slope * index)
            for index in range(len(observed_values))
        ]
    elif (
        candidate_id == "analytic_lag1_affine" or family_id == "analytic"
    ) and {"intercept", "lag_coefficient"} <= parameter_summary.keys():
        intercept = float(parameter_summary["intercept"])
        lag_coefficient = float(parameter_summary["lag_coefficient"])
        fitted_values = [observed_values[0]]
        previous_fitted = observed_values[0]
        for index in range(1, len(observed_values)):
            lag_value = (
                observed_values[index - 1]
                if use_observed_lag_values
                else previous_fitted
            )
            next_fitted = intercept + (lag_coefficient * lag_value)
            fitted_values.append(next_fitted)
            previous_fitted = next_fitted
    elif candidate_id == "algorithmic_last_observation":
        fitted_values = [observed_values[0]]
        previous_fitted = observed_values[0]
        for _ in observed_values[1:]:
            fitted_values.append(previous_fitted)
    elif candidate_id == "algorithmic_running_half_average":
        running = float(state_values.get("state_0", 0.0))
        fitted_values = []
        for index, observed in enumerate(observed_values):
            fitted_values.append(running)
            lag_value = (
                observed
                if use_observed_lag_values
                else fitted_values[index]
            )
            running = 0.5 * (running + lag_value)
    elif candidate_id == "recursive_level_smoother" and "alpha" in literal_values:
        alpha = float(literal_values["alpha"])
        level = float(state_values.get("level", observed_values[0]))
        fitted_values = [level]
        for index in range(1, len(observed_values)):
            lag_value = (
                observed_values[index - 1]
                if use_observed_lag_values
                else fitted_values[index - 1]
            )
            level = (alpha * lag_value) + ((1.0 - alpha) * level)
            fitted_values.append(level)
    elif candidate_id == "recursive_running_mean":
        running_mean = float(state_values.get("running_mean", observed_values[0]))
        step_count = int(state_values.get("step_count", 1))
        fitted_values = []
        for observed in observed_values:
            fitted_values.append(running_mean)
            lag_value = observed if use_observed_lag_values else running_mean
            next_step_count = step_count + 1
            running_mean = (
                (running_mean * step_count) + lag_value
            ) / next_step_count
            step_count = next_step_count
    elif family_id == "spectral" and {
        "cosine_coefficient",
        "sine_coefficient",
    } <= parameter_summary.keys():
        harmonic = int(literal_values.get("harmonic", 1))
        season_length = int(literal_values.get("season_length", 5))
        cosine = float(parameter_summary["cosine_coefficient"])
        sine = float(parameter_summary["sine_coefficient"])
        fitted_values = []
        for index in range(len(observed_values)):
            angle = (2.0 * math.pi * harmonic * index) / season_length
            fitted_values.append((cosine * math.cos(angle)) + (sine * math.sin(angle)))
    elif family_id == "seasonal_naive":
        period = int(parameter_summary.get("seasonal_period", 5))
        fitted_values = []
        for index, observed in enumerate(observed_values):
            if index < period:
                fitted_values.append(observed)
            else:
                fitted_values.append(observed_values[index - period])
    else:
        return []
    return [
        {
            "event_time": event_time,
            "fitted_value": float(fitted_value),
        }
        for event_time, fitted_value in zip(timestamps, fitted_values)
    ]


def _slice_history(
    history: Sequence[Mapping[str, Any]],
    *,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    lower = date.fromisoformat(start_date)
    upper = date.fromisoformat(end_date)
    filtered: list[dict[str, Any]] = []
    for entry in history:
        event_date = _extract_event_date(entry)
        if event_date < lower:
            continue
        if event_date > upper:
            continue
        filtered.append(dict(entry))
    filtered.sort(key=_extract_event_date)
    return filtered


def _validated_requested_date_range(
    *,
    start_date: str | None,
    end_date: str | None,
) -> tuple[str, str]:
    if not start_date or not end_date:
        raise ValueError("start_date and end_date are required")

    lower = _parse_iso_date(start_date, field_name="start_date")
    upper = _parse_iso_date(end_date, field_name="end_date")
    if lower > upper:
        raise ValueError("start_date must be on or before end_date")
    return lower.isoformat(), upper.isoformat()


def _parse_iso_date(value: str, *, field_name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a YYYY-MM-DD date") from exc


def _shift_years(anchor: date, *, years: int) -> date:
    try:
        return anchor.replace(year=anchor.year + years)
    except ValueError:
        return anchor.replace(month=2, day=28, year=anchor.year + years)


def _extract_event_date(entry: Mapping[str, Any]) -> date:
    raw_value = str(entry["date"])
    if "T" in raw_value:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00")).date()
    return date.fromisoformat(raw_value)


def _actual_series(dataset_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "event_time": str(row["event_time"]),
            "observed_value": float(row["observed_value"]),
        }
        for row in dataset_rows
    ]


def _workspace_root(*, output_root: Path, symbol: str, target_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace_root = (
        output_root.resolve() / f"{timestamp}-{_slug(symbol)}-{_slug(target_id)}"
    )
    workspace_root.mkdir(parents=True, exist_ok=True)
    return workspace_root


def _slug(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "-"
        for character in value
    ).strip("-")


def _quantization_step_for_target(target_id: str) -> str:
    if target_id == "price_close":
        return "0.01"
    if target_id in {"daily_return", "log_return"}:
        return "0.0001"
    return "0.001"


def _format_number(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.6f}".rstrip("0").rstrip(".")


def _resolve_structure_signature(
    *,
    structure_body: Mapping[str, Any] | None,
    reducer_body: Mapping[str, Any],
) -> str | None:
    if structure_body is not None:
        structured = _string_or_none(structure_body.get("structure_signature"))
        if structured:
            return structured
    return _string_or_none(reducer_body.get("canonical_structure_signature"))


def _format_portfolio_selection_explanation(explanation: Any) -> str | None:
    if explanation is None:
        return None
    if isinstance(explanation, str):
        return explanation
    if isinstance(explanation, Mapping):
        winner = explanation.get("winner")
        runner_up = explanation.get("runner_up")
        selection_rule = _string_or_none(explanation.get("selection_rule"))
        if isinstance(winner, Mapping):
            winner_bits = _float_or_none(winner.get("total_code_bits"))
            winner_label = " / ".join(
                part
                for part in (
                    _string_or_none(winner.get("submitter_id")),
                    _string_or_none(winner.get("candidate_id")),
                )
                if part
            )
            summary = [f"Winner {winner_label or 'unknown'}"]
            if winner_bits is not None:
                summary.append(f"{_format_number(winner_bits)} total code bits")
            if isinstance(runner_up, Mapping):
                runner_bits = _float_or_none(runner_up.get("total_code_bits"))
                runner_label = " / ".join(
                    part
                    for part in (
                        _string_or_none(runner_up.get("submitter_id")),
                        _string_or_none(runner_up.get("candidate_id")),
                    )
                    if part
                )
                if runner_label:
                    if runner_bits is None:
                        summary.append(f"runner-up {runner_label}")
                    else:
                        summary.append(
                            f"runner-up {runner_label} at "
                            f"{_format_number(runner_bits)}"
                        )
            if selection_rule:
                summary.append(
                    "selected by " + selection_rule.replace("_", " ")
                )
            return "; ".join(summary)
    return json.dumps(_jsonable(explanation), sort_keys=True)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "DEFAULT_TARGET_ID",
    "TARGET_SPECS",
    "build_equation_summary",
    "build_target_rows_from_history",
    "create_workbench_analysis",
    "list_recent_analyses",
    "normalize_analysis_payload",
    "ordered_target_specs",
]
