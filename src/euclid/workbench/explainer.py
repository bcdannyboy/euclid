from __future__ import annotations

import base64
import io
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont

DEFAULT_WORKBENCH_EXPLAINER_MODEL = "gpt-5.4"
DEFAULT_OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_WORKBENCH_EXPLAINER_MODEL_ENV_VAR = "EUCLID_OPENAI_EXPLAINER_MODEL"
DEFAULT_OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
DEFAULT_WORKBENCH_EXPLAINER_TIMEOUT_SECONDS = 900.0
DEFAULT_WORKBENCH_EXPLAINER_COMPACT_RETRY_TIMEOUT_SECONDS = 900.0
WORKBENCH_EXPLAINER_PROMPT_VERSION = "workbench-holistic-multimodal-v1"
WORKBENCH_EXPLANATION_PAGE_KEYS = (
    "overview",
    "point",
    "probabilistic",
    "benchmark",
    "artifacts",
)

_SYSTEM_PROMPT = """You explain Euclid Market Workbench outputs to non-mathematicians.

Rules:
- Explain only what the provided analysis says.
- Ground every page in the concrete facts from the snapshot.
- Use concrete numbers from the snapshot when they help explain the run.
- Treat the numeric snapshot as authoritative. Images are supporting context.
- Do not simply restate the target label, date range, or generic finance boilerplate.
- When equations, fit metrics, or status headlines are present,
  explain those exact items.
- Never invent new metrics, evidence, confidence, macro stories, or publication status.
- Preserve honesty about abstentions, benchmark-local selections, thin evidence,
  and missing descriptive fits.
- Use simple direct English.
- Avoid jargon where possible, and define the few terms you keep.
- Return valid JSON matching the requested schema and nothing else.
"""

_PAGE_EXPLANATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        page_key: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "headline": {"type": "string"},
                "summary": {"type": "string"},
                "narrative": {"type": "string"},
                "key_takeaways": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "cautions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "term": {"type": "string"},
                            "meaning": {"type": "string"},
                        },
                        "required": ["term", "meaning"],
                    },
                },
            },
            "required": [
                "headline",
                "summary",
                "narrative",
                "key_takeaways",
                "cautions",
                "terms",
            ],
        }
        for page_key in WORKBENCH_EXPLANATION_PAGE_KEYS
    },
    "required": list(WORKBENCH_EXPLANATION_PAGE_KEYS),
}

_PREVIEW_WIDTH = 900
_PREVIEW_HEIGHT = 340
_PREVIEW_BG = "#f6f1e8"
_PREVIEW_PANEL = "#fffdf8"
_PREVIEW_TEXT = "#17212b"
_PREVIEW_MUTED = "#5f6c75"
_PREVIEW_ACTUAL = "#17212b"
_PREVIEW_OVERLAY = "#a45738"
_PREVIEW_BAND = "#567060"
_PREVIEW_BAR = "#2c617b"
_PREVIEW_DOT = "#a45738"


class _RetryableExplainerResponseError(RuntimeError):
    """Signals a recoverable Responses payload issue."""


def workbench_explainer_model() -> str:
    configured = os.environ.get(DEFAULT_WORKBENCH_EXPLAINER_MODEL_ENV_VAR, "").strip()
    return configured or DEFAULT_WORKBENCH_EXPLAINER_MODEL


def generate_workbench_explanations(
    analysis: Mapping[str, Any],
    *,
    api_key: str,
    model: str | None = None,
    endpoint: str = DEFAULT_OPENAI_RESPONSES_ENDPOINT,
    timeout_seconds: float = DEFAULT_WORKBENCH_EXPLAINER_TIMEOUT_SECONDS,
    urlopen=urlopen,
) -> dict[str, Any]:
    snapshot = build_workbench_explanation_snapshot(analysis)
    resolved_model = model or workbench_explainer_model()
    try:
        pages = _request_page_explanations(
            analysis=analysis,
            snapshot=snapshot,
            api_key=api_key,
            model=resolved_model,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            reasoning_effort="high",
            include_images=True,
            compact=False,
            urlopen=urlopen,
        )
    except Exception as exc:
        if not _is_retryable_explainer_exception(exc):
            raise
        try:
            pages = _request_page_explanations(
                analysis=analysis,
                snapshot=snapshot,
                api_key=api_key,
                model=resolved_model,
                endpoint=endpoint,
                timeout_seconds=(
                    DEFAULT_WORKBENCH_EXPLAINER_COMPACT_RETRY_TIMEOUT_SECONDS
                ),
                reasoning_effort="medium",
                include_images=False,
                compact=True,
                urlopen=urlopen,
            )
        except Exception as retry_exc:
            raise RuntimeError(
                _retry_failure_message(
                    primary_exc=exc,
                    retry_timeout_seconds=(
                        DEFAULT_WORKBENCH_EXPLAINER_COMPACT_RETRY_TIMEOUT_SECONDS
                    ),
                    retry_exc=retry_exc,
                )
            ) from retry_exc
    return {
        "status": "completed",
        "model": resolved_model,
        "generated_at": _utc_timestamp(),
        "prompt_version": WORKBENCH_EXPLAINER_PROMPT_VERSION,
        "pages": pages,
    }


def ensure_cached_workbench_explanations(
    analysis: Mapping[str, Any],
    *,
    api_key: str | None,
    analysis_path: Path | None = None,
    model: str | None = None,
    endpoint: str = DEFAULT_OPENAI_RESPONSES_ENDPOINT,
    timeout_seconds: float = DEFAULT_WORKBENCH_EXPLAINER_TIMEOUT_SECONDS,
    urlopen=urlopen,
    force: bool = False,
) -> dict[str, Any]:
    updated = dict(analysis)
    existing = updated.get("llm_explanations")
    if (
        not force
        and isinstance(existing, Mapping)
        and str(existing.get("status")) == "completed"
        and str(existing.get("prompt_version") or "")
        == WORKBENCH_EXPLAINER_PROMPT_VERSION
    ):
        return updated

    resolved_model = model or workbench_explainer_model()
    if not api_key:
        if isinstance(existing, Mapping) and str(existing.get("status")) == "completed":
            return updated
        updated["llm_explanations"] = {
            "status": "unavailable",
            "reason_code": "missing_openai_api_key",
            "message": (
                f"Plain-English explanations need {DEFAULT_OPENAI_API_KEY_ENV_VAR}."
            ),
            "model": resolved_model,
            "prompt_version": WORKBENCH_EXPLAINER_PROMPT_VERSION,
            "pages": {},
        }
        _persist_analysis_if_requested(updated, analysis_path)
        return updated

    try:
        updated["llm_explanations"] = generate_workbench_explanations(
            updated,
            api_key=api_key,
            model=resolved_model,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            urlopen=urlopen,
        )
    except Exception as exc:
        updated["llm_explanations"] = {
            "status": "failed",
            "reason_code": "openai_request_failed",
            "message": str(exc),
            "model": resolved_model,
            "prompt_version": WORKBENCH_EXPLAINER_PROMPT_VERSION,
            "pages": {},
        }
    _persist_analysis_if_requested(updated, analysis_path)
    return updated


def build_workbench_explanation_snapshot(
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = _mapping(analysis.get("dataset"))
    target = _mapping(dataset.get("target"))
    operator_point = _mapping(analysis.get("operator_point"))
    descriptive_fit = _mapping(analysis.get("descriptive_fit"))
    benchmark = _mapping(analysis.get("benchmark"))
    portfolio_selection = _mapping(benchmark.get("portfolio_selection"))
    probabilistic = _mapping(analysis.get("probabilistic"))
    dataset_stats = _mapping(dataset.get("stats"))
    probabilistic_lanes = {
        lane_id: {
            "status": str(_mapping(payload).get("status") or "unknown"),
            "selected_family": _mapping(payload).get("selected_family"),
            "equation": _mapping(_mapping(payload).get("equation")).get("label"),
            "calibration_status": _mapping(_mapping(payload).get("calibration")).get(
                "status"
            ),
            "evidence_headline": _mapping(_mapping(payload).get("evidence")).get(
                "headline"
            ),
            "search_scope_headline": _mapping(
                _mapping(payload).get("search_scope")
            ).get("headline"),
        }
        for lane_id, payload in probabilistic.items()
        if isinstance(payload, Mapping)
    }
    probabilistic_summary = _probabilistic_lane_summary(probabilistic_lanes)
    series_summary = _dataset_series_summary(
        stats=dataset_stats,
        series=dataset.get("series"),
    )
    benchmark_local_note = (
        "Benchmark-local winner remains separate from the operator publication lane."
        if portfolio_selection.get("winner_submitter_id")
        else "No benchmark-local winner was selected."
    )
    cross_page_relationships = [
        benchmark_local_note,
        (
            operator_point.get("publication", {}).get("headline")
            or (
                "Point-lane publication status should be read separately "
                "from the benchmark lane."
            )
        ),
    ]
    if descriptive_fit.get("status") == "completed":
        cross_page_relationships.append(
            descriptive_fit.get("semantic_audit", {}).get("headline")
            or descriptive_fit.get("honesty_note")
            or (
                "Benchmark-local descriptive fit is shown separately from "
                "the operator lane."
            )
        )
    if probabilistic_summary["thin_evidence_lane_count"] > 0:
        cross_page_relationships.append(
            (
                f"{probabilistic_summary['thin_evidence_lane_count']} "
                "probabilistic lane(s) still have smoke-sized evidence."
            )
        )

    return {
        "run_digest": {
            "symbol": dataset.get("symbol"),
            "target_label": target.get("label"),
            "target_description": target.get("description"),
            "rows": dataset.get("rows"),
            "date_range": _mapping(dataset.get("date_range")),
            "series_summary": series_summary,
            "target_semantics": target.get("analysis_note"),
            "cross_page_relationships": cross_page_relationships,
        },
        "dataset": {
            "symbol": dataset.get("symbol"),
            "target_label": target.get("label"),
            "target_description": target.get("description"),
            "rows": dataset.get("rows"),
            "date_range": _mapping(dataset.get("date_range")),
            "analysis_note": target.get("analysis_note"),
            "stats": dataset_stats,
            "series_summary": series_summary,
        },
        "operator_point": {
            "status": operator_point.get("status"),
            "selected_family": operator_point.get("selected_family"),
            "publication": _mapping(operator_point.get("publication")),
            "equation": _mapping(operator_point.get("equation")).get("label")
            or _mapping(operator_point.get("equation")).get("structure_signature"),
            "search_scope": _mapping(operator_point.get("search_scope")).get(
                "headline"
            ),
            "abstention_reason_codes": _string_list(
                _mapping(operator_point.get("abstention")).get("reason_codes")
            ),
        },
        "descriptive_fit": {
            "status": descriptive_fit.get("status"),
            "family_id": descriptive_fit.get("family_id"),
            "candidate_id": descriptive_fit.get("candidate_id"),
            "equation": _mapping(descriptive_fit.get("equation")).get("label"),
            "honesty_note": descriptive_fit.get("honesty_note"),
            "semantic_audit": _mapping(descriptive_fit.get("semantic_audit")),
        },
        "benchmark": {
            "status": benchmark.get("status"),
            "winner_submitter_id": portfolio_selection.get("winner_submitter_id"),
            "winner_candidate_id": portfolio_selection.get("winner_candidate_id"),
            "selection_explanation": portfolio_selection.get("selection_explanation"),
            "descriptive_fit_status": _mapping(
                benchmark.get("descriptive_fit_status")
            ),
        },
        "probabilistic": {
            "summary": probabilistic_summary,
            "lanes": probabilistic_lanes,
        },
        "artifacts": {
            "analysis_path": analysis.get("analysis_path"),
            "workspace_root": analysis.get("workspace_root"),
            "dataset_csv": dataset.get("dataset_csv"),
            "raw_history_json": dataset.get("raw_history_json"),
            "point_manifest": _mapping(operator_point).get("manifest_path"),
            "benchmark_report": benchmark.get("report_path"),
        },
        "pages": {
            "overview": {
                "page_goal": (
                    "Explain what the observed target did and the main "
                    "caution for reading this run."
                ),
                "dataset": {
                    "symbol": dataset.get("symbol"),
                    "target_label": target.get("label"),
                    "date_range": _mapping(dataset.get("date_range")),
                    "rows": dataset.get("rows"),
                    "series_summary": series_summary,
                    "target_semantics": target.get("analysis_note"),
                },
                "descriptive_fit": {
                    "status": descriptive_fit.get("status"),
                    "equation": _mapping(descriptive_fit.get("equation")).get("label"),
                    "semantic_audit": _mapping(descriptive_fit.get("semantic_audit")),
                    "honesty_note": descriptive_fit.get("honesty_note"),
                },
                "operator_lane": _mapping(operator_point.get("publication")),
                "probabilistic_summary": probabilistic_summary,
            },
            "point": {
                "page_goal": (
                    "Explain the point-lane candidate and whether it "
                    "cleared publication gates."
                ),
                "selected_family": operator_point.get("selected_family"),
                "equation": _mapping(operator_point.get("equation")).get("label"),
                "publication": _mapping(operator_point.get("publication")),
                "abstention_reason_codes": _string_list(
                    _mapping(operator_point.get("abstention")).get("reason_codes")
                ),
                "search_scope": _mapping(operator_point.get("search_scope")).get(
                    "headline"
                ),
            },
            "probabilistic": {
                "page_goal": (
                    "Explain what the probabilistic lanes say and how "
                    "strong the calibration evidence is."
                ),
                "summary": probabilistic_summary,
                "lanes": probabilistic_lanes,
            },
            "benchmark": {
                "page_goal": (
                    "Explain the benchmark-local comparison and why it "
                    "does not override operator publication status."
                ),
                "winner_submitter_id": portfolio_selection.get("winner_submitter_id"),
                "winner_candidate_id": portfolio_selection.get("winner_candidate_id"),
                "selection_explanation": portfolio_selection.get(
                    "selection_explanation"
                ),
                "descriptive_fit": {
                    "status": descriptive_fit.get("status"),
                    "equation": _mapping(descriptive_fit.get("equation")).get("label"),
                    "honesty_note": descriptive_fit.get("honesty_note"),
                    "semantic_audit": _mapping(descriptive_fit.get("semantic_audit")),
                },
                "descriptive_fit_status": _mapping(
                    benchmark.get("descriptive_fit_status")
                ),
                "operator_publication": _mapping(operator_point.get("publication")),
            },
            "artifacts": {
                "page_goal": (
                    "Explain how the saved files map back to the "
                    "workbench pages."
                ),
                "analysis_path": analysis.get("analysis_path"),
                "workspace_root": analysis.get("workspace_root"),
                "dataset_csv": dataset.get("dataset_csv"),
                "raw_history_json": dataset.get("raw_history_json"),
                "point_manifest": _mapping(operator_point).get("manifest_path"),
                "benchmark_report": benchmark.get("report_path"),
            },
        },
    }


def _build_explanation_input_items(
    *,
    snapshot: Mapping[str, Any],
    analysis: Mapping[str, Any],
    include_images: bool = True,
    compact: bool = False,
) -> list[dict[str, Any]]:
    page_images = _build_page_preview_images(analysis) if include_images else {}
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": _render_request_brief(snapshot, compact=compact),
        },
        _json_text_input(
            "RUN DIGEST",
            _mapping(snapshot.get("run_digest")),
        ),
    ]
    for page_key in WORKBENCH_EXPLANATION_PAGE_KEYS:
        page_context = _mapping(_mapping(snapshot.get("pages")).get(page_key))
        content.append(
            _json_text_input(
                f"{page_key.upper()} PAGE CONTEXT",
                page_context,
            )
        )
        image_url = page_images.get(page_key)
        if image_url:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"{page_key.upper()} PAGE IMAGE:\n"
                        "Use this chart preview as supporting visual context for "
                        "the same page."
                    ),
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_url,
                    "detail": "high",
                }
            )
    return [{"role": "user", "content": content}]


def _render_request_brief(
    snapshot: Mapping[str, Any],
    *,
    compact: bool = False,
) -> str:
    if compact:
        return (
            "Give a holistic explanation of the run across pages.\n"
            "Return JSON with five pages: overview, point, probabilistic, "
            "benchmark, artifacts.\n"
            "For each page return: headline, summary, narrative, "
            "key_takeaways, cautions, terms.\n"
            "This is the compact retry profile, so rely on the run digest "
            "and page contexts below as the authoritative evidence.\n"
            "Keep the explanation specific even without chart images.\n"
            "Do not write generic market commentary."
        )
    return (
        "Give a holistic explanation of the run across pages.\n"
        "Return JSON with five pages: overview, point, probabilistic, "
        "benchmark, artifacts.\n"
        "For each page return: headline, summary, narrative, "
        "key_takeaways, cautions, terms.\n"
        "Keep summaries short but let the narrative synthesize the page more fully.\n"
        "Page requirements:\n"
        "- Overview page: explain the observed series itself, including "
        "direction, scale, or range when available, then name the main "
        "caution from the actual run.\n"
        "- Point page: explain the actual point-lane status, publication "
        "outcome, selected family, equation, and abstention reason codes "
        "when present.\n"
        "- Probabilistic page: explain the actual lane statuses, "
        "calibration outcomes, and whether the evidence is thin or "
        "smoke-sized.\n"
        "- Benchmark page: explain the actual benchmark-local descriptive "
        "fit, including the equation, semantic audit, MAE-vs-naive "
        "context, and why it is not an operator publication.\n"
        "- Artifacts page: explain what the listed saved files correspond "
        "to for this run.\n"
        "Use the images as supporting context only. The numeric snapshot "
        "is authoritative.\n"
        "Do not write generic market commentary.\n"
        "RUN SNAPSHOT:\n"
        + json.dumps(snapshot, indent=2, sort_keys=True)
    )


def _json_text_input(title: str, payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        "type": "input_text",
        "text": f"{title}:\n" + json.dumps(payload, indent=2, sort_keys=True),
    }


def _request_page_explanations(
    *,
    analysis: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    api_key: str,
    model: str,
    endpoint: str,
    timeout_seconds: float,
    reasoning_effort: str,
    include_images: bool,
    compact: bool,
    urlopen,
) -> dict[str, Any]:
    request_body = {
        "model": model,
        "instructions": _SYSTEM_PROMPT,
        "input": _build_explanation_input_items(
            snapshot=snapshot,
            analysis=analysis,
            include_images=include_images,
            compact=compact,
        ),
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "workbench_page_explanations",
                "schema": _PAGE_EXPLANATION_SCHEMA,
                "strict": True,
            }
        },
        "max_output_tokens": 4200,
    }
    request = Request(
        endpoint,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        response_body = json.loads(response.read().decode("utf-8"))
    return _extract_page_explanations(response_body)


def _is_timeout_like_exception(exc: Exception) -> bool:
    queue: list[BaseException | None] = [exc]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (TimeoutError, socket.timeout)):
            return True
        if "timed out" in str(current).lower():
            return True
        queue.extend(
            [
                getattr(current, "__cause__", None),
                getattr(current, "__context__", None),
                getattr(current, "reason", None),
            ]
        )
    return False


def _is_retryable_explainer_exception(exc: Exception) -> bool:
    if _is_timeout_like_exception(exc):
        return True
    queue: list[BaseException | None] = [exc]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, _RetryableExplainerResponseError):
            return True
        queue.extend(
            [
                getattr(current, "__cause__", None),
                getattr(current, "__context__", None),
            ]
        )
    return False


def _retry_failure_message(
    *,
    primary_exc: Exception,
    retry_timeout_seconds: float,
    retry_exc: Exception,
) -> str:
    retry_detail = str(retry_exc).strip() or retry_exc.__class__.__name__
    if not _is_timeout_like_exception(primary_exc):
        primary_detail = str(primary_exc).strip() or primary_exc.__class__.__name__
        return (
            "The full multimodal explainer failed, and the compact retry also "
            f"failed: {primary_detail}; compact retry detail: {retry_detail}"
        )
    return (
        "The full multimodal explainer timed out, and the compact retry "
        f"failed after {retry_timeout_seconds:.0f}s: {retry_detail}"
    )


def _build_page_preview_images(analysis: Mapping[str, Any]) -> dict[str, str]:
    previews: dict[str, str] = {}
    overview = _build_overview_preview(analysis)
    if overview is not None:
        previews["overview"] = overview
    point = _build_point_preview(analysis)
    if point is not None:
        previews["point"] = point
    probabilistic = _build_probabilistic_preview(analysis)
    if probabilistic is not None:
        previews["probabilistic"] = probabilistic
    benchmark = _build_benchmark_preview(analysis)
    if benchmark is not None:
        previews["benchmark"] = benchmark
    return previews


def _build_overview_preview(analysis: Mapping[str, Any]) -> str | None:
    dataset = _mapping(analysis.get("dataset"))
    target = _mapping(dataset.get("target"))
    actual_series = _series_points(dataset.get("series"))
    descriptive_fit = _mapping(analysis.get("descriptive_fit"))
    benchmark = _mapping(analysis.get("benchmark"))
    operator_point = _mapping(analysis.get("operator_point"))

    overlay_series = _overlay_points(
        _mapping(descriptive_fit.get("chart")).get("equation_curve")
    )
    title = "Overview: observed vs descriptive fit"
    if not overlay_series:
        point_curve = _overlay_points(
            _mapping(operator_point.get("equation")).get("curve")
        )
        if point_curve:
            overlay_series = point_curve
            title = "Overview: observed vs point candidate"
        elif (
            _mapping(benchmark.get("descriptive_fit_status")).get("status")
            == "absent_no_admissible_candidate"
        ):
            overlay_series = _rolling_mean_series(actual_series, window_size=30)
            title = "Overview: observed vs rolling mean"
    if not actual_series or not overlay_series:
        return None
    return _line_chart_data_url(
        title=title,
        y_label=str(target.get("y_axis_label") or target.get("label") or "Value"),
        actual_series=actual_series,
        overlay_series=overlay_series,
    )


def _build_point_preview(analysis: Mapping[str, Any]) -> str | None:
    dataset = _mapping(analysis.get("dataset"))
    target = _mapping(dataset.get("target"))
    operator_point = _mapping(analysis.get("operator_point"))
    actual_series = _series_points(dataset.get("series"))
    overlay_series = _overlay_points(
        _mapping(operator_point.get("equation")).get("curve")
    )
    if not actual_series or not overlay_series:
        return None
    return _line_chart_data_url(
        title="Point page: observed vs point candidate",
        y_label=str(target.get("y_axis_label") or target.get("label") or "Value"),
        actual_series=actual_series,
        overlay_series=overlay_series,
    )


def _build_benchmark_preview(analysis: Mapping[str, Any]) -> str | None:
    benchmark = _mapping(analysis.get("benchmark"))
    series = benchmark.get("chart", {}).get("total_code_bits")
    if not isinstance(series, Sequence) or isinstance(series, (str, bytes)):
        return None
    normalized = [
        {
            "label": str(_mapping(item).get("submitter_id") or "unknown"),
            "value": _float_or_none(_mapping(item).get("total_code_bits")) or 0.0,
        }
        for item in series
        if isinstance(item, Mapping)
    ]
    if not normalized:
        return None
    return _bar_chart_data_url(
        title="Benchmark page: finalist total code bits",
        series=normalized,
    )


def _build_probabilistic_preview(analysis: Mapping[str, Any]) -> str | None:
    probabilistic = _mapping(analysis.get("probabilistic"))
    lanes: list[dict[str, Any]] = []
    for mode, payload in probabilistic.items():
        lane = _mapping(payload)
        if str(lane.get("status") or "") != "completed":
            continue
        chart = _mapping(lane.get("chart"))
        bands = _band_rows_for_chart(mode=str(mode), chart=chart)
        if bands:
            lanes.append(
                {
                    "mode": str(mode),
                    "kind": "bands",
                    "rows": bands,
                }
            )
            continue
        probabilities = _probability_rows(chart.get("forecast_probabilities"))
        if probabilities:
            lanes.append(
                {
                    "mode": str(mode),
                    "kind": "probabilities",
                    "rows": probabilities,
                }
            )
    if not lanes:
        return None
    return _probabilistic_chart_data_url(lanes[:3])


def _extract_page_explanations(response_body: Mapping[str, Any]) -> dict[str, Any]:
    _raise_for_incomplete_response(response_body)
    direct_mapping = _extract_direct_mapping(response_body)
    if direct_mapping is not None:
        return _normalize_page_explanations(direct_mapping)
    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return _normalize_page_explanations(json.loads(output_text))

    output_items = response_body.get("output")
    if not isinstance(output_items, Sequence):
        raise ValueError("OpenAI response did not contain any output items")

    text_fragments: list[str] = []
    for output_item in output_items:
        if not isinstance(output_item, Mapping):
            continue
        content_items = output_item.get("content")
        if not isinstance(content_items, Sequence):
            continue
        for content_item in content_items:
            if not isinstance(content_item, Mapping):
                continue
            direct_mapping = _extract_direct_mapping(content_item)
            if direct_mapping is not None:
                return _normalize_page_explanations(direct_mapping)
            text_value = content_item.get("text")
            if isinstance(text_value, str):
                text_fragments.append(text_value)

    if not text_fragments:
        raise ValueError("OpenAI response did not contain explanation text")
    return _normalize_page_explanations(json.loads("\n".join(text_fragments)))


def _raise_for_incomplete_response(response_body: Mapping[str, Any]) -> None:
    if str(response_body.get("status") or "").strip().lower() != "incomplete":
        return
    incomplete_details = _mapping(response_body.get("incomplete_details"))
    reason = str(incomplete_details.get("reason") or "unknown").strip() or "unknown"
    if reason == "max_output_tokens":
        raise _RetryableExplainerResponseError(
            "OpenAI response hit max_output_tokens before completing "
            "structured JSON"
        )
    if reason == "content_filter":
        raise ValueError(
            "OpenAI response was cut off by content filtering before "
            "completing structured JSON"
        )
    raise ValueError(
        "OpenAI response was incomplete before completing structured JSON: "
        f"{reason}"
    )


def _extract_direct_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("output_parsed", "parsed", "json"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _normalize_page_explanations(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for page_key in WORKBENCH_EXPLANATION_PAGE_KEYS:
        page_payload = _mapping(payload.get(page_key))
        headline = str(page_payload.get("headline") or "").strip()
        summary = str(page_payload.get("summary") or "").strip()
        narrative = str(page_payload.get("narrative") or summary).strip()
        if not headline or not summary:
            raise ValueError(f"OpenAI explanation payload missing {page_key} content")
        normalized[page_key] = {
            "headline": headline,
            "summary": summary,
            "narrative": narrative,
            "key_takeaways": _string_list(
                page_payload.get("key_takeaways") or page_payload.get("bullets")
            ),
            "cautions": _string_list(page_payload.get("cautions")),
            "terms": _normalize_terms(page_payload.get("terms")),
        }
    return normalized


def _normalize_terms(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        term = str(item.get("term") or "").strip()
        meaning = str(item.get("meaning") or "").strip()
        if term and meaning:
            normalized.append({"term": term, "meaning": meaning})
    return normalized


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _dataset_series_summary(
    *,
    stats: Mapping[str, Any],
    series: Any,
) -> dict[str, Any]:
    series_items = (
        [item for item in series if isinstance(item, Mapping)]
        if isinstance(series, Sequence) and not isinstance(series, (str, bytes))
        else []
    )
    if not series_items:
        return {}
    first_value = _float_or_none(series_items[0].get("observed_value"))
    latest_value = _float_or_none(
        series_items[-1].get("observed_value") or stats.get("latest_value")
    )
    if first_value is None or latest_value is None:
        return {}
    absolute_change = latest_value - first_value
    percent_change = None if first_value == 0 else absolute_change / first_value
    if absolute_change > 0:
        direction = "up"
    elif absolute_change < 0:
        direction = "down"
    else:
        direction = "flat"
    return {
        "first_value": first_value,
        "latest_value": latest_value,
        "absolute_change": absolute_change,
        "percent_change": percent_change,
        "direction": direction,
        "min": _float_or_none(stats.get("min")),
        "max": _float_or_none(stats.get("max")),
        "mean": _float_or_none(stats.get("mean")),
        "stdev": _float_or_none(stats.get("stdev")),
    }


def _probabilistic_lane_summary(
    probabilistic_summary: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    lanes = [
        payload
        for payload in probabilistic_summary.values()
        if isinstance(payload, Mapping)
    ]
    return {
        "completed_lane_count": sum(
            1 for lane in lanes if str(lane.get("status") or "") == "completed"
        ),
        "thin_evidence_lane_count": sum(
            1
            for lane in lanes
            if "smoke-sized" in str(lane.get("evidence_headline") or "").lower()
            or "thin" in str(lane.get("evidence_headline") or "").lower()
        ),
        "passed_calibration_lane_count": sum(
            1
            for lane in lanes
            if str(lane.get("calibration_status") or "") == "passed"
        ),
    }


def _series_points(series: Any) -> list[dict[str, Any]]:
    if not isinstance(series, Sequence) or isinstance(series, (str, bytes)):
        return []
    points: list[dict[str, Any]] = []
    for item in series:
        if not isinstance(item, Mapping):
            continue
        value = _float_or_none(item.get("observed_value"))
        if value is None:
            continue
        points.append(
            {
                "label": str(item.get("event_time") or ""),
                "value": value,
            }
        )
    return points


def _overlay_points(series: Any) -> list[dict[str, Any]]:
    if not isinstance(series, Sequence) or isinstance(series, (str, bytes)):
        return []
    points: list[dict[str, Any]] = []
    for item in series:
        if not isinstance(item, Mapping):
            continue
        value = _float_or_none(item.get("fitted_value"))
        if value is None:
            continue
        points.append(
            {
                "label": str(item.get("event_time") or ""),
                "value": value,
            }
        )
    return points


def _rolling_mean_series(
    actual_series: Sequence[Mapping[str, Any]],
    *,
    window_size: int,
) -> list[dict[str, Any]]:
    if not actual_series:
        return []
    window = max(2, min(len(actual_series), int(window_size)))
    values = [float(item["value"]) for item in actual_series]
    points: list[dict[str, Any]] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        points.append(
            {
                "label": str(actual_series[index].get("label") or ""),
                "value": running_sum / min(index + 1, window),
            }
        )
    return points


def _line_chart_data_url(
    *,
    title: str,
    y_label: str,
    actual_series: Sequence[Mapping[str, Any]],
    overlay_series: Sequence[Mapping[str, Any]],
) -> str | None:
    if not actual_series or not overlay_series:
        return None
    values = [
        *[float(item["value"]) for item in actual_series],
        *[float(item["value"]) for item in overlay_series],
    ]
    if not values:
        return None
    image, draw, font = _new_preview_canvas(height=_PREVIEW_HEIGHT)
    _draw_title(draw, font, title)
    left, top, right, bottom = 72, 76, _PREVIEW_WIDTH - 32, _PREVIEW_HEIGHT - 48
    _draw_axes(draw, left, top, right, bottom)
    _draw_text(draw, (left, 42), y_label, font, _PREVIEW_MUTED)
    _draw_text(
        draw,
        (left, bottom + 10),
        str(actual_series[0].get("label") or "")[:10],
        font,
        _PREVIEW_MUTED,
    )
    end_label = str(actual_series[-1].get("label") or "")[:10]
    end_width = _text_width(draw, end_label, font)
    _draw_text(
        draw,
        (right - end_width, bottom + 10),
        end_label,
        font,
        _PREVIEW_MUTED,
    )
    y_min, y_max = _expanded_range(values)
    actual_points = _polyline_points(
        actual_series,
        left,
        top,
        right,
        bottom,
        y_min,
        y_max,
    )
    overlay_points = _polyline_points(
        overlay_series,
        left,
        top,
        right,
        bottom,
        y_min,
        y_max,
    )
    if len(actual_points) >= 2:
        draw.line(actual_points, fill=_PREVIEW_ACTUAL, width=4)
    if len(overlay_points) >= 2:
        draw.line(overlay_points, fill=_PREVIEW_OVERLAY, width=3)
    _draw_legend(
        draw,
        font,
        items=[("Observed", _PREVIEW_ACTUAL), ("Overlay", _PREVIEW_OVERLAY)],
        x=left,
        y=top - 26,
    )
    return _image_data_url(image)


def _bar_chart_data_url(
    *,
    title: str,
    series: Sequence[Mapping[str, Any]],
) -> str | None:
    if not series:
        return None
    height = 120 + 48 * len(series)
    image, draw, font = _new_preview_canvas(height=height)
    _draw_title(draw, font, title)
    top = 80
    left = 40
    bar_left = 260
    max_value = max(float(item["value"]) for item in series) or 1.0
    for index, item in enumerate(series):
        y = top + index * 42
        label = str(item.get("label") or "unknown")
        value = float(item["value"])
        bar_width = int((value / max_value) * (_PREVIEW_WIDTH - bar_left - 80))
        _draw_text(draw, (left, y), label, font, _PREVIEW_TEXT)
        draw.rectangle(
            [(bar_left, y), (bar_left + bar_width, y + 18)],
            fill=_PREVIEW_BAR,
        )
        _draw_text(
            draw,
            (bar_left + bar_width + 10, y),
            f"{value:,.0f}",
            font,
            _PREVIEW_MUTED,
        )
    return _image_data_url(image)


def _band_rows_for_chart(
    *,
    mode: str,
    chart: Mapping[str, Any],
) -> list[dict[str, float]]:
    if mode in {"distribution", "interval"}:
        return _band_rows(chart.get("forecast_bands"))
    if mode == "quantile":
        return _band_rows_from_quantiles(chart.get("forecast_quantiles"))
    return []


def _band_rows(value: Any) -> list[dict[str, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[dict[str, float]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        lower = _float_or_none(item.get("lower"))
        upper = _float_or_none(item.get("upper"))
        actual = _float_or_none(item.get("realized_observation"))
        if lower is None or upper is None or actual is None:
            continue
        rows.append({"lower": lower, "upper": upper, "actual": actual})
    return rows


def _band_rows_from_quantiles(value: Any) -> list[dict[str, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[dict[str, float]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        quantiles = {
            str(_mapping(q).get("level")): _float_or_none(_mapping(q).get("value"))
            for q in item.get("quantiles", [])
            if isinstance(q, Mapping)
        }
        lower = quantiles.get("0.1")
        upper = quantiles.get("0.9")
        actual = _float_or_none(item.get("realized_observation"))
        if lower is None or upper is None or actual is None:
            continue
        rows.append({"lower": lower, "upper": upper, "actual": actual})
    return rows


def _probability_rows(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[float] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        probability = _float_or_none(item.get("event_probability"))
        if probability is None:
            continue
        rows.append(probability)
    return rows


def _probabilistic_chart_data_url(lanes: Sequence[Mapping[str, Any]]) -> str | None:
    if not lanes:
        return None
    lane_height = 180
    height = 80 + lane_height * len(lanes)
    image, draw, font = _new_preview_canvas(height=height)
    _draw_title(draw, font, "Probabilistic page: lane previews")
    for index, lane in enumerate(lanes):
        section_top = 64 + index * lane_height
        label = str(lane.get("mode") or "lane")
        _draw_text(
            draw,
            (32, section_top),
            label.replace("_", " ").title(),
            font,
            _PREVIEW_TEXT,
        )
        if lane.get("kind") == "bands":
            _draw_band_lane(
                draw=draw,
                rows=lane["rows"],
                top=section_top + 28,
                left=72,
                right=_PREVIEW_WIDTH - 36,
                height=110,
            )
        elif lane.get("kind") == "probabilities":
            _draw_probability_lane(
                draw=draw,
                rows=lane["rows"],
                top=section_top + 28,
                left=72,
                right=_PREVIEW_WIDTH - 36,
            )
    return _image_data_url(image)


def _draw_band_lane(
    *,
    draw: ImageDraw.ImageDraw,
    rows: Sequence[Mapping[str, float]],
    top: int,
    left: int,
    right: int,
    height: int,
) -> None:
    if len(rows) < 2:
        return
    values = [
        *(float(row["lower"]) for row in rows),
        *(float(row["upper"]) for row in rows),
        *(float(row["actual"]) for row in rows),
    ]
    y_min, y_max = _expanded_range(values)
    bottom = top + height
    _draw_axes(draw, left, top, right, bottom)
    for index, row in enumerate(rows):
        x = _x_position(index, len(rows), left, right)
        lower = _y_position(float(row["lower"]), top, bottom, y_min, y_max)
        upper = _y_position(float(row["upper"]), top, bottom, y_min, y_max)
        actual = _y_position(float(row["actual"]), top, bottom, y_min, y_max)
        draw.line([(x, lower), (x, upper)], fill=_PREVIEW_BAND, width=7)
        draw.ellipse(
            [(x - 4, actual - 4), (x + 4, actual + 4)],
            fill=_PREVIEW_DOT,
        )


def _draw_probability_lane(
    *,
    draw: ImageDraw.ImageDraw,
    rows: Sequence[float],
    top: int,
    left: int,
    right: int,
) -> None:
    for index, probability in enumerate(rows):
        y = top + index * 28
        width = int(max(0.0, min(1.0, float(probability))) * (right - left - 80))
        draw.rectangle([(left, y), (left + width, y + 16)], fill=_PREVIEW_OVERLAY)
        _draw_text(draw, (left - 36, y), f"h{index + 1}", _font(), _PREVIEW_MUTED)


def _new_preview_canvas(
    *,
    height: int,
) -> tuple[Image.Image, ImageDraw.ImageDraw, ImageFont.ImageFont]:
    image = Image.new("RGB", (_PREVIEW_WIDTH, height), _PREVIEW_BG)
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.rectangle(
        [(12, 12), (_PREVIEW_WIDTH - 12, height - 12)],
        fill=_PREVIEW_PANEL,
        outline="#ddd4c8",
        width=1,
    )
    return image, draw, font


def _draw_title(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    title: str,
) -> None:
    _draw_text(draw, (28, 24), title, font, _PREVIEW_TEXT)


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> None:
    draw.line([(left, bottom), (right, bottom)], fill="#d7cec3", width=2)
    draw.line([(left, top), (left, bottom)], fill="#d7cec3", width=2)


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    *,
    items: Sequence[tuple[str, str]],
    x: int,
    y: int,
) -> None:
    cursor_x = x
    for label, color in items:
        draw.rectangle([(cursor_x, y + 4), (cursor_x + 14, y + 10)], fill=color)
        _draw_text(draw, (cursor_x + 20, y), label, font, _PREVIEW_MUTED)
        cursor_x += 20 + _text_width(draw, label, font) + 28


def _draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    draw.text(position, str(text), fill=fill, font=font)


def _text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> int:
    bbox = draw.textbbox((0, 0), str(text), font=font)
    return int(bbox[2] - bbox[0])


def _polyline_points(
    series: Sequence[Mapping[str, Any]],
    left: int,
    top: int,
    right: int,
    bottom: int,
    y_min: float,
    y_max: float,
) -> list[tuple[int, int]]:
    return [
        (
            _x_position(index, len(series), left, right),
            _y_position(float(item["value"]), top, bottom, y_min, y_max),
        )
        for index, item in enumerate(series)
    ]


def _x_position(index: int, count: int, left: int, right: int) -> int:
    if count <= 1:
        return left
    return int(left + (index / (count - 1)) * (right - left))


def _y_position(value: float, top: int, bottom: int, y_min: float, y_max: float) -> int:
    return int(bottom - ((value - y_min) / (y_max - y_min or 1.0)) * (bottom - top))


def _expanded_range(values: Sequence[float]) -> tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    spread = max(max_value - min_value, 1.0)
    return min_value - spread * 0.08, max_value + spread * 0.08


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _persist_analysis_if_requested(
    analysis: Mapping[str, Any],
    analysis_path: Path | None,
) -> None:
    if analysis_path is None:
        return
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "DEFAULT_OPENAI_API_KEY_ENV_VAR",
    "DEFAULT_WORKBENCH_EXPLAINER_MODEL",
    "DEFAULT_WORKBENCH_EXPLAINER_MODEL_ENV_VAR",
    "DEFAULT_OPENAI_RESPONSES_ENDPOINT",
    "WORKBENCH_EXPLAINER_PROMPT_VERSION",
    "build_workbench_explanation_snapshot",
    "ensure_cached_workbench_explanations",
    "generate_workbench_explanations",
    "workbench_explainer_model",
]
