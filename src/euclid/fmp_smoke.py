from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml

from euclid.benchmarks import profile_benchmark_task
from euclid.operator_runtime import replay_operator, run_operator
from euclid.operator_runtime.resources import resolve_asset_root

FMP_EOD_ENDPOINT = "https://financialmodelingprep.com/stable/historical-price-eod/full"
FMP_DOCS_URL = "https://site.financialmodelingprep.com/developer/docs/stable/historical-price-eod-full"
DEFAULT_SOURCE_ID = "financial_modeling_prep"
PROBABILISTIC_FORECAST_OBJECT_TYPES = (
    "distribution",
    "interval",
    "quantile",
    "event_probability",
)
_CANONICAL_DATASET_COLUMNS = (
    "source_id",
    "series_id",
    "event_time",
    "available_at",
    "observed_value",
    "revision_id",
)
_PROBABILISTIC_TEMPLATE_BY_TYPE = {
    "distribution": "fixtures/runtime/phase06/probabilistic-distribution-demo.yaml",
    "interval": "fixtures/runtime/phase06/probabilistic-interval-demo.yaml",
    "quantile": "fixtures/runtime/phase06/probabilistic-quantile-demo.yaml",
    "event_probability": (
        "fixtures/runtime/phase06/probabilistic-event-probability-demo.yaml"
    ),
}


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def extract_fmp_history_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("historical", "data", "results"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return [dict(item) for item in candidate if isinstance(item, Mapping)]
    raise ValueError(
        "FMP response did not match a supported history payload shape. "
        f"See {FMP_DOCS_URL}."
    )


def fetch_fmp_eod_history(
    *,
    symbol: str,
    api_key: str,
) -> list[dict[str, Any]]:
    query = urlencode({"symbol": symbol, "apikey": api_key})
    request = Request(
        url=f"{FMP_EOD_ENDPOINT}?{query}",
        headers={"User-Agent": "euclid-fmp-smoke/1.0"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    return extract_fmp_history_entries(payload)


def build_csv_rows_from_fmp_history(
    history: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    source_id: str = DEFAULT_SOURCE_ID,
    availability_hour: int = 21,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in history:
        event_date = _extract_event_date(entry)
        observed_value = _extract_numeric_value(
            entry,
            field_names=("close", "adjClose", "adj_close", "adjustedClose"),
        )
        row: dict[str, Any] = {
            "source_id": source_id,
            "series_id": symbol.upper(),
            "event_time": f"{event_date.isoformat()}T00:00:00Z",
            "available_at": f"{event_date.isoformat()}T{availability_hour:02d}:00:00Z",
            "observed_value": observed_value,
            "revision_id": 0,
        }
        for key, value in entry.items():
            if key == "date":
                continue
            normalized_key = _snake_case(key)
            if normalized_key in _CANONICAL_DATASET_COLUMNS:
                continue
            row[normalized_key] = value
        rows.append(row)
    rows.sort(key=lambda item: (item["event_time"], item["available_at"]))
    if not rows:
        raise ValueError("FMP history did not contain any usable daily bars")
    return rows


def write_euclid_dataset_csv(
    rows: Sequence[Mapping[str, Any]],
    *,
    destination: Path,
) -> Path:
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
    return destination


def build_operator_manifest(
    *,
    project_root: Path,
    dataset_csv: Path,
    manifest_path: Path,
    request_id: str,
    forecast_object_type: str,
    row_count: int,
) -> dict[str, Any]:
    payload = _load_operator_template(
        project_root=project_root,
        forecast_object_type=forecast_object_type,
    )
    payload["request_id"] = request_id
    payload["dataset_csv"] = _relative_path(dataset_csv, start=manifest_path.parent)
    payload["quantization_step"] = "0.01"
    payload["min_train_size"] = _default_min_train_size(row_count)
    payload["horizon"] = _default_horizon(row_count, forecast_object_type)
    seasonal_period = _default_seasonal_period(row_count)
    payload["seasonal_period"] = seasonal_period
    search = dict(payload.get("search") or {})
    search["seasonal_period"] = seasonal_period
    payload["search"] = search

    if forecast_object_type == "point":
        payload["forecast_object_type"] = "point"
        payload.pop("probabilistic", None)
        return payload

    payload.pop("forecast_object_type", None)
    probabilistic = dict(payload.get("probabilistic") or {})
    probabilistic["forecast_object_type"] = forecast_object_type
    payload["probabilistic"] = probabilistic
    return payload


def build_benchmark_task_manifest(
    *,
    project_root: Path,
    dataset_csv: Path,
    manifest_path: Path,
    symbol: str,
    row_count: int,
    availability_cutoff: str | None = None,
) -> dict[str, Any]:
    template_path = (
        project_root
        / "benchmarks"
        / "tasks"
        / "predictive_generalization"
        / "seasonal-trend-medium.yaml"
    )
    with template_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark template {template_path} is not a mapping")

    payload["task_id"] = f"fmp_{_slug(symbol)}_benchmark"
    payload["dataset_ref"] = _relative_path(dataset_csv, start=project_root)
    payload["generator_status"] = "unknown_real_world"
    payload["regime_tags"] = ["market_data", "fmp", "single_entity"]
    payload["snapshot_policy"] = dict(payload.get("snapshot_policy") or {})
    payload["snapshot_policy"]["availability_cutoff"] = availability_cutoff
    payload["split_policy"] = dict(payload.get("split_policy") or {})
    payload["split_policy"]["initial_window"] = _default_min_train_size(row_count)
    payload["horizon_policy"] = {"horizons": [_default_horizon(row_count, "point")]}
    payload["budget_policy"] = {
        "wall_clock_seconds": 90,
        "candidate_limit": 32,
    }
    payload["seed_policy"] = {"seed": 0, "restarts": 0}
    payload.pop("fixture_spec_id", None)
    payload.pop("fixture_family_id", None)
    return payload


def write_yaml_manifest(payload: Mapping[str, Any], *, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)
    return destination


def run_fmp_smoke(
    *,
    symbol: str,
    api_key: str,
    mode: str,
    output_root: Path,
    project_root: Path | None = None,
    row_limit: int = 180,
    start_date: str | None = None,
    end_date: str | None = None,
    benchmark_workers: int = 1,
) -> dict[str, Any]:
    resolved_project_root = (project_root or _default_project_root()).resolve()
    workspace_root = _workspace_root(output_root=output_root, symbol=symbol)
    raw_history = fetch_fmp_eod_history(symbol=symbol, api_key=api_key)
    filtered_history = _slice_history(
        raw_history,
        row_limit=row_limit,
        start_date=start_date,
        end_date=end_date,
    )
    rows = build_csv_rows_from_fmp_history(filtered_history, symbol=symbol)
    row_count = len(rows)
    if row_count < 8:
        raise ValueError(
            "Euclid smoke needs at least 8 daily observations after filtering. "
            f"Received {row_count}."
        )

    raw_history_path = workspace_root / "fmp-history.json"
    raw_history_path.parent.mkdir(parents=True, exist_ok=True)
    raw_history_path.write_text(
        json.dumps(filtered_history, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    dataset_csv = write_euclid_dataset_csv(
        rows,
        destination=workspace_root / "datasets" / f"{_slug(symbol)}.csv",
    )
    availability_cutoff = rows[-1]["available_at"]
    summary: dict[str, Any] = {
        "symbol": symbol.upper(),
        "workspace_root": str(workspace_root),
        "dataset_csv": str(dataset_csv),
        "raw_history_json": str(raw_history_path),
        "rows": row_count,
        "date_range": {
            "start": rows[0]["event_time"],
            "end": rows[-1]["event_time"],
        },
        "results": {},
    }

    if mode in {"current-release", "all"}:
        summary["results"]["current_release"] = _run_operator_mode(
            project_root=resolved_project_root,
            workspace_root=workspace_root,
            dataset_csv=dataset_csv,
            mode_name="current_release",
            forecast_object_type="point",
            symbol=symbol,
            row_count=row_count,
        )

    if mode in {"probabilistic", "all"}:
        for forecast_object_type in PROBABILISTIC_FORECAST_OBJECT_TYPES:
            summary["results"][forecast_object_type] = _run_operator_mode(
                project_root=resolved_project_root,
                workspace_root=workspace_root,
                dataset_csv=dataset_csv,
                mode_name=forecast_object_type,
                forecast_object_type=forecast_object_type,
                symbol=symbol,
                row_count=row_count,
            )

    if mode in {"benchmark", "all"}:
        benchmark_project_root = resolve_asset_root(resolved_project_root)
        benchmark_dataset_csv = write_euclid_dataset_csv(
            rows,
            destination=(
                benchmark_project_root
                / "build"
                / "fmp-smoke"
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
            row_count=row_count,
            availability_cutoff=availability_cutoff,
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
        summary["results"]["benchmark"] = {
            "task_id": benchmark_result.task_manifest.task_id,
            "manifest_path": str(benchmark_task_path),
            "benchmark_root": str(benchmark_root),
            "dataset_csv": str(benchmark_dataset_csv),
            "report_path": str(benchmark_result.report_paths.report_path),
            "task_result_path": str(benchmark_result.report_paths.task_result_path),
            "telemetry_path": str(benchmark_result.telemetry_path),
            "submitter_ids": [
                result.submitter_id for result in benchmark_result.submitter_results
            ],
        }

    return summary


def format_smoke_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "Euclid FMP smoke",
        f"Symbol: {summary['symbol']}",
        f"Workspace: {summary['workspace_root']}",
        f"Dataset: {summary['dataset_csv']}",
        (
            "Date range: "
            f"{summary['date_range']['start']} -> {summary['date_range']['end']}"
        ),
        f"Rows: {summary['rows']}",
        f"Raw history: {summary['raw_history_json']}"
        if "raw_history_json" in summary
        else "",
    ]
    lines = [line for line in lines if line]
    for mode_name, result in (summary.get("results") or {}).items():
        lines.append("")
        lines.append(f"Mode: {mode_name}")
        if "run_id" in result:
            lines.append(f"Run id: {result['run_id']}")
        if "manifest_path" in result:
            lines.append(f"Manifest: {result['manifest_path']}")
        if "dataset_csv" in result:
            lines.append(f"Dataset: {result['dataset_csv']}")
        if "output_root" in result:
            lines.append(f"Output root: {result['output_root']}")
        if "selected_family" in result:
            lines.append(f"Selected family: {result['selected_family']}")
        if "forecast_object_type" in result:
            lines.append(f"Forecast object type: {result['forecast_object_type']}")
        if "replay_verification" in result:
            lines.append(f"Replay verification: {result['replay_verification']}")
        if "run_summary_path" in result:
            lines.append(f"Run summary: {result['run_summary_path']}")
        if "report_path" in result:
            lines.append(f"Report path: {result['report_path']}")
        if "task_result_path" in result:
            lines.append(f"Task result path: {result['task_result_path']}")
        if "telemetry_path" in result:
            lines.append(f"Telemetry path: {result['telemetry_path']}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch daily EOD market data from Financial Modeling Prep and run "
            "Euclid smoke workflows against it."
        )
    )
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol to fetch.")
    parser.add_argument(
        "--mode",
        choices=("all", "current-release", "probabilistic", "benchmark"),
        default="all",
        help="Which Euclid surfaces to exercise.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_project_root() / "build" / "fmp-smoke",
        help="Directory where generated datasets, manifests, and artifacts are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=180,
        help="Trailing number of daily bars to keep after filtering.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional ISO date filter inclusive, e.g. 2024-01-01.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional ISO date filter inclusive, e.g. 2024-12-31.",
    )
    parser.add_argument(
        "--benchmark-workers",
        type=int,
        default=1,
        help="Deterministic worker count for benchmark mode.",
    )
    parser.add_argument(
        "--api-key-env-var",
        default="FMP_API_KEY",
        help="Environment variable that stores the FMP API key.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    api_key = _read_api_key(args.api_key_env_var)
    summary = run_fmp_smoke(
        symbol=args.symbol,
        api_key=api_key,
        mode=args.mode,
        output_root=args.output_root,
        row_limit=max(8, int(args.limit)),
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark_workers=max(1, int(args.benchmark_workers)),
    )
    print(format_smoke_report(summary))
    return 0


def _run_operator_mode(
    *,
    project_root: Path,
    workspace_root: Path,
    dataset_csv: Path,
    mode_name: str,
    forecast_object_type: str,
    symbol: str,
    row_count: int,
) -> dict[str, Any]:
    manifest_path = workspace_root / "manifests" / f"{mode_name}.yaml"
    request_id = f"fmp-{_slug(symbol)}-{mode_name}"
    manifest = build_operator_manifest(
        project_root=project_root,
        dataset_csv=dataset_csv,
        manifest_path=manifest_path,
        request_id=request_id,
        forecast_object_type=forecast_object_type,
        row_count=row_count,
    )
    write_yaml_manifest(manifest, destination=manifest_path)
    output_root = workspace_root / "runs" / mode_name
    run_result = run_operator(
        manifest_path=manifest_path,
        output_root=output_root,
    )
    replay_result = replay_operator(
        output_root=output_root,
        run_id=request_id,
    )
    return {
        "run_id": request_id,
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "forecast_object_type": forecast_object_type,
        "selected_family": run_result.summary.selected_family,
        "run_summary_path": str(run_result.paths.run_summary_path),
        "replay_verification": replay_result.summary.replay_verification_status,
    }


def _load_operator_template(
    *,
    project_root: Path,
    forecast_object_type: str,
) -> dict[str, Any]:
    if forecast_object_type == "point":
        template_path = project_root / "examples" / "current_release_run.yaml"
    else:
        template_path = project_root / _PROBABILISTIC_TEMPLATE_BY_TYPE[forecast_object_type]
    with template_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"operator template {template_path} is not a mapping")
    return payload


def _slice_history(
    history: Sequence[Mapping[str, Any]],
    *,
    row_limit: int,
    start_date: str | None,
    end_date: str | None,
) -> list[dict[str, Any]]:
    lower = date.fromisoformat(start_date) if start_date else None
    upper = date.fromisoformat(end_date) if end_date else None
    filtered = []
    for entry in history:
        event_date = _extract_event_date(entry)
        if lower is not None and event_date < lower:
            continue
        if upper is not None and event_date > upper:
            continue
        filtered.append(dict(entry))
    filtered.sort(key=lambda item: _extract_event_date(item))
    if row_limit > 0:
        filtered = filtered[-row_limit:]
    return filtered


def _extract_event_date(entry: Mapping[str, Any]) -> date:
    raw_value = entry.get("date")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("FMP history entry is missing a date field")
    trimmed = raw_value.strip()
    if "T" in trimmed:
        return datetime.fromisoformat(trimmed.replace("Z", "+00:00")).date()
    return date.fromisoformat(trimmed)


def _extract_numeric_value(
    entry: Mapping[str, Any],
    *,
    field_names: Sequence[str],
) -> float:
    for field_name in field_names:
        value = entry.get(field_name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str) and value.strip():
            return float(value)
    raise ValueError(f"FMP history entry missing numeric field from {field_names!r}")


def _snake_case(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_").lower()


def _relative_path(path: Path, *, start: Path) -> str:
    return os.path.relpath(path.resolve(), start.resolve())


def _default_min_train_size(row_count: int) -> int:
    return max(3, min(30, row_count // 2))


def _default_horizon(row_count: int, forecast_object_type: str) -> int:
    if forecast_object_type == "point":
        return 1
    if row_count >= 84:
        return 21
    if row_count >= 24:
        return 5
    return 1


def _default_seasonal_period(row_count: int) -> int:
    if row_count >= 40:
        return 5
    return 2


def _workspace_root(*, output_root: Path, symbol: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace_root = output_root.resolve() / f"{timestamp}-{_slug(symbol)}"
    workspace_root.mkdir(parents=True, exist_ok=True)
    return workspace_root


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "symbol"


def _read_api_key(env_var: str) -> str:
    import os

    api_key = os.environ.get(env_var, "").strip()
    if api_key:
        return api_key
    raise SystemExit(
        f"Set {env_var} before running the smoke script. "
        "The key is read from the environment and not stored in the repo."
    )


__all__ = [
    "PROBABILISTIC_FORECAST_OBJECT_TYPES",
    "build_arg_parser",
    "build_benchmark_task_manifest",
    "build_csv_rows_from_fmp_history",
    "build_operator_manifest",
    "extract_fmp_history_entries",
    "fetch_fmp_eod_history",
    "format_smoke_report",
    "main",
    "run_fmp_smoke",
    "write_euclid_dataset_csv",
    "write_yaml_manifest",
]
