from __future__ import annotations

import json
from pathlib import Path

import pytest

from euclid.workbench.explainer import (
    DEFAULT_WORKBENCH_EXPLAINER_COMPACT_RETRY_TIMEOUT_SECONDS,
    DEFAULT_WORKBENCH_EXPLAINER_MODEL,
    DEFAULT_WORKBENCH_EXPLAINER_TIMEOUT_SECONDS,
    WORKBENCH_EXPLAINER_PROMPT_VERSION,
    build_workbench_explanation_snapshot,
    ensure_cached_workbench_explanations,
    generate_workbench_explanations,
)


def _sample_analysis() -> dict:
    return {
        "analysis_path": "/tmp/analysis.json",
        "workspace_root": "/tmp/workbench/sample",
        "dataset": {
            "symbol": "SPY",
            "target": {
                "id": "price_close",
                "label": "Price Close",
                "description": "Predict the raw close level for each trading day.",
            },
            "rows": 1255,
            "date_range": {
                "start": "2021-04-19T00:00:00Z",
                "end": "2026-04-16T00:00:00Z",
            },
            "stats": {
                "min": 412.17,
                "max": 612.42,
                "mean": 503.61,
                "stdev": 44.28,
                "latest_value": 612.42,
            },
            "series": [
                {
                    "event_time": "2021-04-19T00:00:00Z",
                    "observed_value": 412.17,
                },
                {
                    "event_time": "2023-10-02T00:00:00Z",
                    "observed_value": 503.84,
                },
                {
                    "event_time": "2026-04-16T00:00:00Z",
                    "observed_value": 612.42,
                },
            ],
        },
        "operator_point": {
            "status": "completed",
            "selected_family": "drift",
            "publication": {
                "status": "abstained",
                "headline": (
                    "Operator replay verified a candidate, but the point lane "
                    "did not clear publication gates."
                ),
            },
            "equation": {
                "label": "y(t) = 412.17 + 0.229848*t",
                "curve": [
                    {
                        "event_time": "2021-04-19T00:00:00Z",
                        "fitted_value": 412.17,
                    },
                    {
                        "event_time": "2023-10-02T00:00:00Z",
                        "fitted_value": 502.84,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 611.92,
                    },
                ],
            },
            "abstention": {
                "reason_codes": ["robustness_failed", "perturbation_protocol_failed"]
            },
        },
        "descriptive_fit": {
            "status": "completed",
            "family_id": "analytic",
            "candidate_id": "analytic_lag1_affine",
            "equation": {
                "label": "y(t) = 0.260812 + 0.999934*y(t-1)",
            },
            "semantic_audit": {
                "headline": (
                    "Raw close descriptive fit is effectively a persistence "
                    "equation with a tiny intercept."
                )
            },
            "chart": {
                "equation_curve": [
                    {
                        "event_time": "2021-04-19T00:00:00Z",
                        "fitted_value": 412.17,
                    },
                    {
                        "event_time": "2023-10-02T00:00:00Z",
                        "fitted_value": 503.22,
                    },
                    {
                        "event_time": "2026-04-16T00:00:00Z",
                        "fitted_value": 612.11,
                    },
                ]
            }
        },
        "benchmark": {
            "status": "completed",
            "portfolio_selection": {
                "winner_submitter_id": "analytic_backend",
                "winner_candidate_id": "analytic_lag1_affine",
                "selection_explanation": (
                    "analytic_backend won the benchmark-local comparison by the "
                    "configured code-length rule."
                ),
            },
            "descriptive_fit_status": {
                "status": "available",
            },
            "chart": {
                "total_code_bits": [
                    {"submitter_id": "analytic_backend", "total_code_bits": 3642},
                    {"submitter_id": "algorithmic_search", "total_code_bits": 3666},
                ]
            },
        },
        "probabilistic": {
            "distribution": {
                "status": "completed",
                "selected_family": "analytic",
                "equation": {"label": "distribution summary"},
                "calibration": {"status": "passed"},
                "evidence": {
                    "strength": "thin",
                    "headline": (
                        "Calibration evidence is smoke-sized and should not be "
                        "overread."
                    ),
                },
                "chart": {
                    "forecast_bands": [
                        {
                            "lower": 580.0,
                            "upper": 640.0,
                            "realized_observation": 612.0,
                        },
                        {
                            "lower": 600.0,
                            "upper": 660.0,
                            "realized_observation": 628.0,
                        },
                    ]
                },
            },
            "interval": {
                "status": "completed",
                "selected_family": "analytic",
                "equation": {"label": "interval summary"},
                "calibration": {"status": "passed"},
                "evidence": {
                    "strength": "thin",
                    "headline": (
                        "Interval calibration evidence is also smoke-sized."
                    ),
                },
                "chart": {
                    "forecast_bands": [
                        {
                            "lower": 575.0,
                            "upper": 635.0,
                            "realized_observation": 608.0,
                        },
                        {
                            "lower": 595.0,
                            "upper": 655.0,
                            "realized_observation": 624.0,
                        },
                    ]
                },
            }
        },
    }


def _sample_pages() -> dict:
    return {
        "overview": {
            "headline": "What this run is doing",
            "summary": (
                "This page summarizes the target, the date range, and "
                "the main caution for the run."
            ),
            "narrative": (
                "SPY rose materially across the window, but the descriptive fit is "
                "still basically tracking the last close and the operator did not "
                "clear publication gates."
            ),
            "key_takeaways": [
                "The observed series rose materially across the run.",
                "The descriptive fit behaves like near-persistence.",
                "The operator lane abstained from publication.",
            ],
            "cautions": [
                "Raw close levels can hide persistence inside level tracking."
            ],
            "terms": [
                {
                    "term": "abstained",
                    "meaning": "Euclid chose not to publish a claim.",
                }
            ],
        },
        "point": {
            "headline": "What the point lane means",
            "summary": (
                "The point lane found a simple path but did not clear "
                "publication checks."
            ),
            "narrative": (
                "The point lane selected a drift-style candidate, but replay "
                "verification is not enough for publication because the lane still "
                "failed later gates."
            ),
            "key_takeaways": [
                "This is not a published operator claim.",
                "The selected family was drift.",
                "The fitted path is only a candidate explanation.",
            ],
            "cautions": [
                "Publication status matters more than having a candidate equation."
            ],
            "terms": [
                {"term": "drift", "meaning": "A straight line over time."}
            ],
        },
        "probabilistic": {
            "headline": "What the probability pages mean",
            "summary": (
                "These lanes describe ranges and probabilities, but the "
                "evidence is thin here."
            ),
            "narrative": (
                "The probabilistic lanes completed and passed calibration gates, but "
                "the evidence base is still smoke-sized, so they should be treated as "
                "early support rather than strong probabilistic proof."
            ),
            "key_takeaways": [
                "Thin evidence means few calibration examples.",
                "Passed does not mean proven.",
                "Treat this as early evidence.",
            ],
            "cautions": [
                "Thin calibration evidence is easy to overread."
            ],
            "terms": [
                {
                    "term": "calibration",
                    "meaning": "How well forecast probabilities match "
                    "reality.",
                }
            ],
        },
        "benchmark": {
            "headline": "What benchmark-local means",
            "summary": (
                "Benchmark compares local finalists, but that winner is "
                "not the operator publication."
            ),
            "narrative": (
                "The benchmark-local winner is the descriptive-fit candidate that "
                "best survived the benchmark comparison, but it remains separate from "
                "the operator publication lane."
            ),
            "key_takeaways": [
                "Benchmark is a comparison lane.",
                "Its winner is local to that search.",
                "Do not treat it as the published result.",
            ],
            "cautions": [
                "Benchmark-local selection is not an operator publication."
            ],
            "terms": [
                {
                    "term": "benchmark-local",
                    "meaning": "Chosen inside the benchmark comparison only.",
                }
            ],
        },
        "artifacts": {
            "headline": "What the saved files are for",
            "summary": (
                "These files let you inspect the dataset, manifests, and "
                "saved run outputs behind the UI."
            ),
            "narrative": (
                "The artifact paths let you trace the saved analysis back to the "
                "dataset, manifests, and replay outputs that produced each page."
            ),
            "key_takeaways": [
                "analysis.json is the rendered payload.",
                "dataset CSV is what Euclid fit.",
                "manifests and runs are replay artifacts.",
            ],
            "cautions": [
                "Artifacts support inspection, not new semantic claims."
            ],
            "terms": [
                {
                    "term": "artifact",
                    "meaning": "A saved file produced during the run.",
                }
            ],
        },
    }


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_default_explainer_timeouts_are_900_seconds() -> None:
    assert DEFAULT_WORKBENCH_EXPLAINER_TIMEOUT_SECONDS == pytest.approx(900.0)
    assert DEFAULT_WORKBENCH_EXPLAINER_COMPACT_RETRY_TIMEOUT_SECONDS == pytest.approx(
        900.0
    )


def test_build_workbench_explanation_snapshot_includes_series_and_evidence_facts(
) -> None:
    snapshot = build_workbench_explanation_snapshot(_sample_analysis())

    series_summary = snapshot["dataset"]["series_summary"]

    assert series_summary["first_value"] == pytest.approx(412.17)
    assert series_summary["latest_value"] == pytest.approx(612.42)
    assert series_summary["absolute_change"] == pytest.approx(200.25)
    assert series_summary["percent_change"] == pytest.approx(200.25 / 412.17)
    assert series_summary["direction"] == "up"
    assert series_summary["min"] == pytest.approx(412.17)
    assert series_summary["max"] == pytest.approx(612.42)
    assert series_summary["mean"] == pytest.approx(503.61)
    assert series_summary["stdev"] == pytest.approx(44.28)
    assert snapshot["probabilistic"]["summary"] == {
        "completed_lane_count": 2,
        "thin_evidence_lane_count": 2,
        "passed_calibration_lane_count": 2,
    }


def test_generate_workbench_explanations_parses_structured_responses_payload() -> None:
    recorded: dict[str, dict] = {}

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        recorded["url"] = request.full_url
        recorded["body"] = json.loads(request.data.decode("utf-8"))
        recorded["timeout"] = timeout
        return _FakeResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(_sample_pages()),
                            }
                        ],
                    }
                ]
            }
        )

    bundle = generate_workbench_explanations(
        _sample_analysis(),
        api_key="test-key",
        model=DEFAULT_WORKBENCH_EXPLAINER_MODEL,
        urlopen=fake_urlopen,
    )

    assert recorded["url"].endswith("/responses")
    assert recorded["body"]["model"] == "gpt-5.4"
    assert recorded["timeout"] == DEFAULT_WORKBENCH_EXPLAINER_TIMEOUT_SECONDS
    assert recorded["body"]["text"]["format"]["type"] == "json_schema"
    assert recorded["body"]["reasoning"] == {"effort": "high"}
    assert isinstance(recorded["body"]["input"], list)
    content = recorded["body"]["input"][0]["content"]
    text_items = [item for item in content if item["type"] == "input_text"]
    image_items = [item for item in content if item["type"] == "input_image"]
    assert any(
        "Give a holistic explanation of the run across pages." in item["text"]
        for item in text_items
    )
    assert any(
        "Benchmark page: explain the actual benchmark-local descriptive fit"
        in item["text"]
        for item in text_items
    )
    assert len(image_items) >= 3
    assert all(item["detail"] == "high" for item in image_items)
    assert all(
        str(item["image_url"]).startswith("data:image/png;base64,")
        for item in image_items
    )
    assert bundle["status"] == "completed"
    assert bundle["pages"]["overview"]["summary"].startswith("This page summarizes")
    assert bundle["pages"]["overview"]["narrative"].startswith("SPY rose materially")
    assert bundle["pages"]["overview"]["key_takeaways"][0].startswith("The observed")
    assert bundle["pages"]["overview"]["cautions"][0].startswith("Raw close")
    assert bundle["pages"]["benchmark"]["terms"][0]["term"] == "benchmark-local"


def test_generate_workbench_explanations_retries_timeout_with_compact_payload() -> None:
    calls: list[dict[str, object]] = []

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        body = json.loads(request.data.decode("utf-8"))
        calls.append({"body": body, "timeout": timeout})
        if len(calls) == 1:
            raise TimeoutError("The read operation timed out")
        return _FakeResponse({"output_text": json.dumps(_sample_pages())})

    bundle = generate_workbench_explanations(
        _sample_analysis(),
        api_key="test-key",
        model=DEFAULT_WORKBENCH_EXPLAINER_MODEL,
        urlopen=fake_urlopen,
    )

    assert len(calls) == 2
    assert calls[0]["timeout"] == DEFAULT_WORKBENCH_EXPLAINER_TIMEOUT_SECONDS
    assert (
        calls[1]["timeout"]
        == DEFAULT_WORKBENCH_EXPLAINER_COMPACT_RETRY_TIMEOUT_SECONDS
    )
    first_content = calls[0]["body"]["input"][0]["content"]
    second_content = calls[1]["body"]["input"][0]["content"]
    assert any(item["type"] == "input_image" for item in first_content)
    assert not any(item["type"] == "input_image" for item in second_content)
    assert calls[0]["body"]["reasoning"] == {"effort": "high"}
    assert calls[1]["body"]["reasoning"] == {"effort": "medium"}
    assert bundle["status"] == "completed"


def test_generate_workbench_explanations_retries_incomplete_max_output_payload(
) -> None:
    calls: list[dict[str, object]] = []
    partial_output = json.dumps(_sample_pages())[:-19]

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        body = json.loads(request.data.decode("utf-8"))
        calls.append({"body": body, "timeout": timeout})
        if len(calls) == 1:
            return _FakeResponse(
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": partial_output,
                                }
                            ],
                        }
                    ],
                }
            )
        return _FakeResponse({"status": "completed", "output_text": json.dumps(_sample_pages())})

    bundle = generate_workbench_explanations(
        _sample_analysis(),
        api_key="test-key",
        model=DEFAULT_WORKBENCH_EXPLAINER_MODEL,
        urlopen=fake_urlopen,
    )

    assert len(calls) == 2
    second_content = calls[1]["body"]["input"][0]["content"]
    assert not any(item["type"] == "input_image" for item in second_content)
    assert calls[1]["body"]["reasoning"] == {"effort": "medium"}
    assert bundle["status"] == "completed"


def test_ensure_cached_workbench_explanations_does_not_retry_non_timeout_failures(
) -> None:
    call_count = 0

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        raise RuntimeError("upstream exploded")

    analysis = ensure_cached_workbench_explanations(
        _sample_analysis(),
        api_key="test-key",
        urlopen=fake_urlopen,
    )

    assert call_count == 1
    assert analysis["llm_explanations"]["status"] == "failed"
    assert analysis["llm_explanations"]["message"] == "upstream exploded"


def test_ensure_cached_workbench_explanations_marks_missing_api_key_unavailable(
) -> None:
    analysis = ensure_cached_workbench_explanations(
        _sample_analysis(),
        api_key=None,
    )

    assert analysis["llm_explanations"]["status"] == "unavailable"
    assert (
        analysis["llm_explanations"]["reason_code"]
        == "missing_openai_api_key"
    )


def test_ensure_cached_workbench_explanations_persists_generated_bundle(
    tmp_path: Path,
) -> None:
    analysis_path = tmp_path / "analysis.json"
    initial = _sample_analysis()
    initial["analysis_path"] = str(analysis_path)
    analysis_path.write_text(json.dumps(initial, indent=2), encoding="utf-8")

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        return _FakeResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(_sample_pages()),
                            }
                        ],
                    }
                ]
            }
        )

    enriched = ensure_cached_workbench_explanations(
        initial,
        api_key="test-key",
        analysis_path=analysis_path,
        urlopen=fake_urlopen,
    )

    persisted = json.loads(analysis_path.read_text(encoding="utf-8"))

    assert enriched["llm_explanations"]["status"] == "completed"
    assert persisted["llm_explanations"]["pages"]["artifacts"]["headline"] == (
        "What the saved files are for"
    )


def test_ensure_cached_workbench_explanations_regenerates_old_prompt_versions() -> None:
    recorded: dict[str, dict] = {}
    analysis = _sample_analysis()
    analysis["llm_explanations"] = {
        "status": "completed",
        "model": DEFAULT_WORKBENCH_EXPLAINER_MODEL,
        "generated_at": "2026-04-16T00:00:00Z",
        "prompt_version": "workbench-simple-english-v1",
        "pages": _sample_pages(),
    }

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        recorded["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"output_text": json.dumps(_sample_pages())})

    enriched = ensure_cached_workbench_explanations(
        analysis,
        api_key="test-key",
        urlopen=fake_urlopen,
    )

    assert recorded["body"]["model"] == DEFAULT_WORKBENCH_EXPLAINER_MODEL
    assert enriched["llm_explanations"]["prompt_version"] == (
        WORKBENCH_EXPLAINER_PROMPT_VERSION
    )
