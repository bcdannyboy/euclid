from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import euclid.workbench.server as workbench_server
from euclid.workbench.server import create_workbench_server
from euclid.workbench.service import (
    build_target_rows_from_history,
    normalize_analysis_payload,
    ordered_target_specs,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_LIVE_DELAY_SECONDS = 2.5
SMOKE_API_KEY_ENV_VAR = "EUCLID_WORKBENCH_SMOKE_API_KEY"
SMOKE_OPENAI_API_KEY_ENV_VAR = "EUCLID_WORKBENCH_SMOKE_OPENAI_API_KEY"
_CANONICAL_DATASET_COLUMNS = (
    "source_id",
    "series_id",
    "event_time",
    "available_at",
    "observed_value",
    "revision_id",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start a local workbench server with a synthetic saved analysis and "
            "a stubbed delayed /api/analyze path, then verify the smoke API flow."
        )
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Host interface to bind the temporary workbench server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="TCP port to bind to. Use 0 to ask the OS for a free port.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional persistent output root. If omitted, the script writes into "
            "a temporary directory and cleans it up on exit."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Optional Euclid project root override for the local server.",
    )
    parser.add_argument(
        "--live-delay-seconds",
        type=float,
        default=DEFAULT_LIVE_DELAY_SECONDS,
        help=(
            "Artificial delay applied to the stubbed live /api/analyze response. "
            "Use this to exercise busy-state and stale-response checks in the browser."
        ),
    )
    parser.add_argument(
        "--hold-open",
        action="store_true",
        help="Keep the local server running after the API smoke checks pass.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with resolved_output_root(args.output_root) as output_root:
            saved_analysis = build_smoke_analysis(
                output_root=output_root,
                symbol="SPY",
                target_id="daily_return",
                scenario="saved-no-winner-layout-check",
                no_winner=True,
                include_descriptive_fit=False,
                long_paths=True,
                start_date="2025-01-08",
                end_date="2025-02-14",
            )
            live_factory = make_live_analysis_factory(
                output_root=output_root,
                live_delay_seconds=max(0.0, args.live_delay_seconds),
            )

            with smoke_server(
                host=args.host,
                port=args.port,
                output_root=output_root,
                project_root=(args.project_root or default_project_root()).resolve(),
                live_factory=live_factory,
            ) as base_url:
                result = run_smoke_checks(
                    base_url=base_url,
                    saved_analysis=saved_analysis,
                    live_delay_seconds=max(0.0, args.live_delay_seconds),
                )
                print_summary(
                    base_url=base_url,
                    output_root=output_root,
                    saved_analysis=result["saved_analysis"],
                    live_analysis=result["live_analysis"],
                    hold_open=args.hold_open,
                    live_delay_seconds=max(0.0, args.live_delay_seconds),
                )
                if args.hold_open:
                    hold_open_until_interrupted()
    except KeyboardInterrupt:
        print("Stopped workbench UI smoke server.")
        return 130
    except Exception as exc:
        print(f"workbench_ui_smoke failed: {exc}")
        return 1
    return 0


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@contextmanager
def resolved_output_root(output_root: Path | None) -> Iterator[Path]:
    if output_root is not None:
        resolved = output_root.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        yield resolved
        return
    with tempfile.TemporaryDirectory(prefix="euclid-workbench-ui-smoke-") as tmp_dir:
        yield Path(tmp_dir).resolve()


@contextmanager
def smoke_server(
    *,
    host: str,
    port: int,
    output_root: Path,
    project_root: Path,
    live_factory,
) -> Iterator[str]:
    server = None
    thread = None
    with patched_module_value(
        workbench_server,
        "create_workbench_analysis",
        live_factory,
    ), patched_env_var(SMOKE_API_KEY_ENV_VAR, "smoke-key"), patched_env_var(
        SMOKE_OPENAI_API_KEY_ENV_VAR, None
    ):
        server = create_workbench_server(
            host=host,
            port=port,
            output_root=output_root,
            project_root=project_root,
            api_key_env_var=SMOKE_API_KEY_ENV_VAR,
            openai_api_key_env_var=SMOKE_OPENAI_API_KEY_ENV_VAR,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        server_host, server_port = server.server_address[:2]
        try:
            yield f"http://{server_host}:{server_port}"
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


@contextmanager
def patched_module_value(module: Any, name: str, value: Any) -> Iterator[None]:
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


@contextmanager
def patched_env_var(name: str, value: str | None) -> Iterator[None]:
    original = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


def hold_open_until_interrupted() -> None:
    print("Holding the smoke server open for manual QA. Press Ctrl-C to stop.")
    while True:
        time.sleep(0.5)


def run_smoke_checks(
    *,
    base_url: str,
    saved_analysis: dict[str, Any],
    live_delay_seconds: float,
) -> dict[str, dict[str, Any]]:
    config = request_json(base_url, "/api/config")
    assert config["api_key_env_var"] == SMOKE_API_KEY_ENV_VAR
    assert config["has_api_key_env"] is True
    recent_paths = [entry.get("analysis_path") for entry in config.get("recent_analyses", [])]
    assert saved_analysis["analysis_path"] in recent_paths

    loaded_saved = request_json(
        base_url,
        "/api/analysis?" + urlencode({"analysis_path": saved_analysis["analysis_path"]}),
    )
    assert loaded_saved["analysis_path"] == saved_analysis["analysis_path"]
    assert loaded_saved["dataset"]["target"]["id"] == "daily_return"
    assert loaded_saved["benchmark"]["portfolio_selection"]["winner_submitter_id"] is None
    assert (
        loaded_saved["benchmark"]["descriptive_fit_status"]["status"]
        == "absent_no_admissible_candidate"
    )
    assert max(
        len(str(loaded_saved["workspace_root"])),
        len(str(loaded_saved["analysis_path"])),
        len(str(loaded_saved["dataset"]["dataset_csv"])),
        len(str(loaded_saved["benchmark"]["report_path"])),
    ) >= 100

    started = time.monotonic()
    live_analysis = request_json(
        base_url,
        "/api/analyze",
        method="POST",
        payload={
            "symbol": "SPY",
            "target_id": "price_close",
            "start_date": "2025-01-08",
            "end_date": "2025-02-14",
            "api_key": "manual-smoke-key",
            "include_probabilistic": True,
            "include_benchmark": True,
        },
    )
    elapsed = time.monotonic() - started
    if live_delay_seconds > 0:
        assert elapsed + 0.15 >= live_delay_seconds
    assert live_analysis["dataset"]["target"]["id"] == "price_close"
    assert live_analysis["operator_point"]["status"] == "completed"
    assert live_analysis["probabilistic"]["distribution"]["status"] == "completed"
    assert live_analysis["benchmark"]["status"] == "completed"
    assert live_analysis["benchmark"]["portfolio_selection"]["winner_submitter_id"]
    assert Path(live_analysis["analysis_path"]).is_file()

    refreshed_config = request_json(base_url, "/api/config")
    refreshed_paths = [
        entry.get("analysis_path") for entry in refreshed_config.get("recent_analyses", [])
    ]
    assert live_analysis["analysis_path"] in refreshed_paths

    return {
        "saved_analysis": loaded_saved,
        "live_analysis": live_analysis,
    }


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json"}
    request_data = None
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
        request_data = json.dumps(payload).encode("utf-8")
    request = Request(
        url=f"{base_url}{path}",
        headers=request_headers,
        data=request_data,
        method=method,
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310
            body = json.load(response)
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed with {exc.code}: {error_body}") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"{method} {path} did not return a JSON object")
    return body


def make_live_analysis_factory(*, output_root: Path, live_delay_seconds: float):
    counter = {"value": 0}

    def fake_create_workbench_analysis(
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
        del api_key, project_root, benchmark_workers, include_probabilistic, include_benchmark
        counter["value"] += 1
        if live_delay_seconds > 0:
            time.sleep(live_delay_seconds)
        return build_smoke_analysis(
            output_root=output_root,
            symbol=symbol,
            target_id=target_id,
            scenario=f"live-run-{counter['value']}",
            no_winner=False,
            include_descriptive_fit=True,
            long_paths=True,
            start_date=start_date or "2025-01-08",
            end_date=end_date or "2025-02-14",
        )

    return fake_create_workbench_analysis


def build_smoke_analysis(
    *,
    output_root: Path,
    symbol: str,
    target_id: str,
    scenario: str,
    no_winner: bool,
    include_descriptive_fit: bool,
    long_paths: bool,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    workspace_root = output_root / workspace_name(symbol=symbol, target_id=target_id, scenario=scenario)
    workspace_root.mkdir(parents=True, exist_ok=True)

    filtered_history = [
        entry
        for entry in fake_price_history(days=60)
        if start_date <= entry["date"] <= end_date
    ]
    dataset_rows = build_target_rows_from_history(
        filtered_history,
        symbol=symbol,
        target_id=target_id,
    )
    raw_history_path = long_artifact_path(
        workspace_root,
        "raw-inputs",
        "fmp-history-with-an-intentionally-long-filename-for-layout-checks.json",
        long_paths=long_paths,
    )
    write_json(raw_history_path, filtered_history)

    dataset_csv = long_artifact_path(
        workspace_root,
        "datasets",
        f"{slug(symbol)}-{slug(target_id)}-dataset-with-an-intentionally-long-filename-for-layout-checks.csv",
        long_paths=long_paths,
    )
    write_dataset_csv(dataset_csv, dataset_rows)

    point_manifest_path = long_artifact_path(
        workspace_root,
        "manifests",
        "operator-point-with-an-intentionally-long-manifest-name-for-layout-checks.yaml",
        long_paths=long_paths,
    )
    point_output_root = long_artifact_path(
        workspace_root,
        "runs",
        "point-lane-output-with-an-intentionally-long-directory-name-for-layout-checks",
        long_paths=long_paths,
    )
    benchmark_manifest_path = long_artifact_path(
        workspace_root,
        "manifests",
        "benchmark-task-with-an-intentionally-long-manifest-name-for-layout-checks.yaml",
        long_paths=long_paths,
    )
    benchmark_report_path = long_artifact_path(
        workspace_root,
        "reports",
        "benchmark-report-with-an-intentionally-long-filename-for-layout-checks.md",
        long_paths=long_paths,
    )
    write_text(point_manifest_path, "workflow_id: smoke_operator_point\n")
    point_output_root.mkdir(parents=True, exist_ok=True)
    write_text(benchmark_manifest_path, "task_id: smoke_benchmark\n")
    write_text(
        benchmark_report_path,
        "# Workbench UI smoke report\n\nSynthetic benchmark report for layout checks.\n",
    )

    actual_series = observed_series(dataset_rows)
    point_curve = fitted_curve(dataset_rows, slope=0.96, intercept=0.0004)
    descriptive_curve = fitted_curve(dataset_rows, slope=0.985, intercept=0.0001)
    distribution_rows = forecast_band_rows(dataset_rows)
    selected_family = "analytic_lag1_affine" if target_id != "next_day_up" else "analytic_logit"

    analysis_path = workspace_root / "analysis.json"
    analysis: dict[str, Any] = {
        "analysis_version": "1.0.0",
        "workspace_root": str(workspace_root),
        "analysis_path": str(analysis_path),
        "request": {
            "symbol": symbol.upper(),
            "target_id": target_id,
            "start_date": start_date,
            "end_date": end_date,
            "include_probabilistic": True,
            "include_benchmark": True,
        },
        "dataset": {
            "symbol": symbol.upper(),
            "target": {"id": target_id},
            "dataset_csv": str(dataset_csv),
            "raw_history_json": str(raw_history_path),
        },
        "operator_point": {
            "status": "completed",
            "selected_family": selected_family,
            "publication": {
                "status": "candidate_only",
                "headline": (
                    "Smoke payload selected a point-lane candidate; this sidecar "
                    "exists to exercise UI state and layout only."
                ),
            },
            "equation": {
                "label": "y(t) = 0.96 * y(t-1) + 0.0004",
                "structure_signature": "lag1_affine",
                "parameter_summary": {
                    "lag_coefficient": 0.96,
                    "intercept": 0.0004,
                },
                "curve": point_curve,
            },
            "prediction_rows": prediction_rows(dataset_rows),
            "replay_verification": "verified",
            "confirmatory_primary_score": 0.742,
            "comparison": {"comparison_class_status": "candidate_only"},
            "scorecard": {"descriptive_status": "candidate_only"},
            "output_root": str(point_output_root),
            "manifest_path": str(point_manifest_path),
            "chart": {"actual_series": actual_series},
        },
        "probabilistic": {
            "distribution": {
                "status": "completed",
                "selected_family": "analytic_distribution",
                "equation": {
                    "label": "distribution summary",
                    "structure_signature": "smoke_distribution",
                },
                "replay_verification": "verified",
                "aggregated_primary_score": 0.681,
                "rows": len(distribution_rows),
                "chart": {"forecast_bands": distribution_rows},
                "calibration": {
                    "status": "passed",
                    "passed": True,
                    "gate_effect": "passed_with_smoke_data",
                    "diagnostics": [{"diagnostic_id": "coverage", "value": 0.82}],
                },
                "evidence": {
                    "strength": "thin",
                    "headline": (
                        "Smoke-sized calibration evidence is enough for UI checks, "
                        "not for probabilistic confidence."
                    ),
                    "sample_size": len(distribution_rows),
                    "origin_count": len(distribution_rows),
                    "horizon_count": 1,
                },
            }
        },
        "benchmark": build_benchmark_payload(no_winner=no_winner),
    }
    analysis["benchmark"]["manifest_path"] = str(benchmark_manifest_path)
    analysis["benchmark"]["report_path"] = str(benchmark_report_path)

    if include_descriptive_fit and not no_winner:
        analysis["descriptive_fit"] = {
            "status": "completed",
            "source": "benchmark_local_selection",
            "submitter_id": "analytic_backend",
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "metrics": {"total_code_bits": 128.4},
            "honesty_note": (
                "Descriptive fit from the broader benchmark-local search winner; "
                "not an operator publication."
            ),
            "equation": {
                "label": "y(t) = 0.985 * y(t-1) + 0.0001",
                "structure_signature": "lag1_affine",
                "parameter_summary": {
                    "lag_coefficient": 0.985,
                    "intercept": 0.0001,
                },
                "curve": descriptive_curve,
            },
            "chart": {"equation_curve": descriptive_curve},
        }

    normalized = normalize_analysis_payload(analysis)
    write_json(analysis_path, normalized)
    return normalized


def workspace_name(*, symbol: str, target_id: str, scenario: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{timestamp}-{slug(symbol)}-{slug(target_id)}-"
        f"{slug(scenario)}-with-an-intentionally-long-workspace-name-for-layout-checks"
    )


def fake_price_history(*, days: int) -> list[dict[str, Any]]:
    start = date(2025, 1, 2)
    close = 575.0
    history: list[dict[str, Any]] = []
    business_days = 0
    offset = 0
    while business_days < days:
        current = start + timedelta(days=offset)
        offset += 1
        if current.weekday() >= 5:
            continue
        drift = 0.8 if business_days % 5 else -0.35
        seasonal = (business_days % 3) * 0.17
        close = round(close + drift + seasonal, 4)
        history.append(
            {
                "date": current.isoformat(),
                "open": round(close - 0.55, 4),
                "high": round(close + 0.95, 4),
                "low": round(close - 1.2, 4),
                "close": close,
                "volume": 1_250_000 + business_days * 9_500,
            }
        )
        business_days += 1
    return history


def observed_series(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "event_time": row["event_time"],
            "observed_value": float(row["observed_value"]),
        }
        for row in dataset_rows
    ]


def fitted_curve(
    dataset_rows: list[dict[str, Any]],
    *,
    slope: float,
    intercept: float,
) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    previous_value = float(dataset_rows[0]["observed_value"])
    for row in dataset_rows:
        observed_value = float(row["observed_value"])
        previous_value = previous_value * slope + intercept
        fitted_value = (previous_value + observed_value) / 2
        curve.append(
            {
                "event_time": row["event_time"],
                "fitted_value": round(fitted_value, 6),
            }
        )
        previous_value = observed_value
    return curve


def forecast_band_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = dataset_rows[-8:]
    payload: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        observed_value = float(row["observed_value"])
        band = 0.006 + index * 0.0005
        payload.append(
            {
                "available_at": row["available_at"],
                "origin_time": row["event_time"],
                "lower": round(observed_value - band, 6),
                "upper": round(observed_value + band, 6),
                "realized_observation": observed_value,
            }
        )
    return payload


def prediction_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = dataset_rows[-6:]
    payload: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        observed_value = float(row["observed_value"])
        payload.append(
            {
                "step": index,
                "event_time": row["event_time"],
                "observed_value": round(observed_value, 6),
                "predicted_value": round(observed_value * 0.97 + 0.0003, 6),
            }
        )
    return payload


def build_benchmark_payload(*, no_winner: bool) -> dict[str, Any]:
    if no_winner:
        return {
            "status": "completed",
            "track_summary": {"abstention_mode": "no_publication"},
            "chart": {
                "total_code_bits": [
                    {"submitter_id": "analytic_backend", "total_code_bits": 142.3},
                    {"submitter_id": "drift_backend", "total_code_bits": 146.9},
                ]
            },
            "portfolio_selection": {
                "winner_submitter_id": None,
                "winner_candidate_id": None,
                "selection_explanation": (
                    "No admissible finalist was selected for benchmark-local comparison."
                ),
                "selection_explanation_raw": {
                    "selection_rule": "no_admissible_finalist",
                    "runner_up": {"submitter_id": "analytic_backend"},
                },
                "decision_trace": [
                    {
                        "step": "collect_submitter_finalists",
                        "family_finalists": {
                            "analytic_backend": None,
                            "drift_backend": None,
                        },
                    },
                    {
                        "step": "select_portfolio_winner",
                        "selected_backend_family": None,
                        "selected_candidate_id": None,
                    },
                ],
            },
            "submitters": [
                {
                    "submitter_id": "analytic_backend",
                    "status": "abstained",
                    "backend_families": ["analytic"],
                    "selected_candidate_id": None,
                    "rejection_reason_codes": ["insufficient_signal"],
                },
                {
                    "submitter_id": "drift_backend",
                    "status": "abstained",
                    "backend_families": ["drift"],
                    "selected_candidate_id": None,
                    "rejection_reason_codes": ["insufficient_signal"],
                },
            ],
        }
    return {
        "status": "completed",
        "track_summary": {"abstention_mode": "no_publication"},
        "chart": {
            "total_code_bits": [
                {"submitter_id": "analytic_backend", "total_code_bits": 128.4},
                {"submitter_id": "drift_backend", "total_code_bits": 139.6},
            ]
        },
        "portfolio_selection": {
            "winner_submitter_id": "analytic_backend",
            "winner_candidate_id": "analytic_lag1_affine",
            "selection_explanation": {
                "winner": {
                    "submitter_id": "analytic_backend",
                    "candidate_id": "analytic_lag1_affine",
                    "total_code_bits": 128.4,
                },
                "runner_up": {
                    "submitter_id": "drift_backend",
                    "candidate_id": "drift_linear",
                    "total_code_bits": 139.6,
                },
                "selection_rule": "lowest_total_code_bits",
            },
            "decision_trace": [
                {
                    "step": "collect_submitter_finalists",
                    "family_finalists": {
                        "analytic_backend": "analytic_lag1_affine",
                        "drift_backend": "drift_linear",
                    },
                },
                {
                    "step": "rank_submitter_finalists",
                    "ordered_candidate_ids": [
                        "analytic_lag1_affine",
                        "drift_linear",
                    ],
                },
                {
                    "step": "select_portfolio_winner",
                    "selected_backend_family": "analytic_backend",
                    "selected_candidate_id": "analytic_lag1_affine",
                },
            ],
        },
        "submitters": [
            {
                "submitter_id": "analytic_backend",
                "status": "completed",
                "backend_families": ["analytic"],
                "selected_candidate_id": "analytic_lag1_affine",
                "selected_candidate_metrics": {"total_code_bits": 128.4},
            },
            {
                "submitter_id": "drift_backend",
                "status": "completed",
                "backend_families": ["drift"],
                "selected_candidate_id": "drift_linear",
                "selected_candidate_metrics": {"total_code_bits": 139.6},
            },
        ],
    }


def target_specs_by_id() -> dict[str, dict[str, Any]]:
    return {spec["id"]: spec for spec in ordered_target_specs()}


def long_artifact_path(
    workspace_root: Path,
    first_segment: str,
    leaf_name: str,
    *,
    long_paths: bool,
) -> Path:
    if not long_paths:
        return workspace_root / first_segment / leaf_name
    return (
        workspace_root
        / first_segment
        / "nested-artifacts-with-an-intentionally-long-segment-for-layout-checks"
        / "more-nesting-with-an-intentionally-long-segment-for-layout-checks"
        / leaf_name
    )


def write_dataset_csv(destination: Path, rows: list[dict[str, Any]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in _CANONICAL_DATASET_COLUMNS
        }
    )
    fieldnames = [*_CANONICAL_DATASET_COLUMNS, *extra_columns]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(destination: Path, payload: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_text(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def print_summary(
    *,
    base_url: str,
    output_root: Path,
    saved_analysis: dict[str, Any],
    live_analysis: dict[str, Any],
    hold_open: bool,
    live_delay_seconds: float,
) -> None:
    print("[ok] /api/config listed the synthetic saved analysis")
    print("[ok] /api/analysis loaded the saved no-winner payload with long paths")
    print("[ok] /api/analyze returned a delayed completed payload and refreshed recents")
    print(f"Workbench URL: {base_url}")
    print(f"Output root: {output_root}")
    print(f"Saved workspace: {saved_analysis['workspace_root']}")
    print(f"Saved analysis: {saved_analysis['analysis_path']}")
    print(f"Live workspace: {live_analysis['workspace_root']}")
    print(f"Live analysis: {live_analysis['analysis_path']}")
    if hold_open:
        print(
            "Manual QA: open the URL above, load the saved daily-return analysis, "
            f"then run a live analysis and use the {live_delay_seconds:.1f}s delay "
            "window for busy-state and stale-response checks."
        )


def slug(value: str) -> str:
    normalized = []
    for char in value.lower():
        if char.isalnum():
            normalized.append(char)
        else:
            normalized.append("-")
    text = "".join(normalized).strip("-")
    while "--" in text:
        text = text.replace("--", "-")
    return text or "item"


if __name__ == "__main__":
    raise SystemExit(main())
