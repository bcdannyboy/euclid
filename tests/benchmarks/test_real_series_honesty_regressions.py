from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

from euclid.benchmarks import load_benchmark_task_manifest
from euclid.workbench.service import normalize_analysis_payload

CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = CHECKOUT_ROOT / "src" / "euclid" / "_assets"
TASK_ROOT = ASSET_ROOT / "benchmarks" / "tasks" / "predictive_generalization"
FIXTURE_ROOT = ASSET_ROOT / "fixtures" / "runtime" / "phase08" / "real_series"


@dataclass(frozen=True)
class RealSeriesRegressionCase:
    case_id: str
    manifest_path: Path
    fixture_dir: Path
    dataset_name: str
    expected_benchmark_status: str
    expect_descriptive_fit: bool
    expected_claim_class: str | None
    expected_raw_holistic_status: str | None
    required_gap_codes: tuple[str, ...]


CASES = (
    RealSeriesRegressionCase(
        case_id="spy_daily_return_20260418",
        manifest_path=(
            TASK_ROOT / "real-series-spy-daily-return-honesty-20260418.yaml"
        ),
        fixture_dir=FIXTURE_ROOT / "spy-daily-return-20260418",
        dataset_name="spy-daily-return.csv",
        expected_benchmark_status="absent_no_admissible_candidate",
        expect_descriptive_fit=False,
        expected_claim_class=None,
        expected_raw_holistic_status="completed",
        required_gap_codes=(
            "operator_not_publishable",
            "no_backend_joint_claim",
            "requires_exact_sample_closure",
        ),
    ),
    RealSeriesRegressionCase(
        case_id="spy_price_close_20260416",
        manifest_path=(
            TASK_ROOT / "real-series-spy-price-close-honesty-20260416.yaml"
        ),
        fixture_dir=FIXTURE_ROOT / "spy-price-close-20260416",
        dataset_name="spy-price-close.csv",
        expected_benchmark_status="available",
        expect_descriptive_fit=True,
        expected_claim_class="descriptive_fit",
        expected_raw_holistic_status=None,
        required_gap_codes=(
            "no_accepted_candidate",
            "operator_not_publishable",
            "descriptive_only",
            "no_backend_joint_claim",
        ),
    ),
    RealSeriesRegressionCase(
        case_id="gld_price_close_20260418",
        manifest_path=(
            TASK_ROOT / "real-series-gld-price-close-honesty-20260418.yaml"
        ),
        fixture_dir=FIXTURE_ROOT / "gld-price-close-20260418",
        dataset_name="gld-price-close.csv",
        expected_benchmark_status="available",
        expect_descriptive_fit=True,
        expected_claim_class="descriptive_fit",
        expected_raw_holistic_status="completed",
        required_gap_codes=(
            "no_accepted_candidate",
            "operator_not_publishable",
            "descriptive_only",
            "no_backend_joint_claim",
            "requires_posthoc_symbolic_synthesis",
        ),
    ),
)


def _load_analysis_fixture(case: RealSeriesRegressionCase) -> dict[str, object]:
    payload = json.loads((case.fixture_dir / "analysis.json").read_text(encoding="utf-8"))
    rebound = deepcopy(payload)
    dataset = dict(rebound["dataset"])
    dataset["dataset_csv"] = str(case.fixture_dir / case.dataset_name)
    rebound["dataset"] = dataset
    return rebound


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_real_series_phase08_manifests_pin_copied_fixture_paths(
    case: RealSeriesRegressionCase,
) -> None:
    manifest = load_benchmark_task_manifest(case.manifest_path)

    assert manifest.task_family == "real_series_honesty_regression"
    assert manifest.track_id == "predictive_generalization"
    assert manifest.dataset_ref == (
        "fixtures/runtime/phase08/real_series/"
        f"{case.fixture_dir.name}/{case.dataset_name}"
    )
    assert manifest.frozen_protocol.snapshot_policy["freeze_mode"] == (
        "content_addressed_copy"
    )
    assert manifest.frozen_protocol.forecast_object_type == "point"
    assert manifest.source_path == case.manifest_path
    assert "real_series" in manifest.regime_tags
    assert "semantic_regression" in manifest.regime_tags
    assert "top_line_honesty" in manifest.adversarial_tags
    assert "sample_exact_closure" in manifest.forbidden_shortcuts
    assert "posthoc_symbolic_synthesis" in manifest.forbidden_shortcuts


@pytest.mark.parametrize(
    "case",
    tuple(case for case in CASES if case.expect_descriptive_fit),
    ids=lambda case: case.case_id,
)
def test_real_series_price_level_cases_keep_descriptive_lane_populated(
    case: RealSeriesRegressionCase,
) -> None:
    normalized = normalize_analysis_payload(_load_analysis_fixture(case))
    descriptive_fit = normalized.get("descriptive_fit")

    assert isinstance(descriptive_fit, dict)
    assert normalized["benchmark"]["descriptive_fit_status"]["status"] == "available"
    assert normalized.get("claim_class") == "descriptive_fit"
    assert descriptive_fit["claim_class"] == "descriptive_fit"
    assert descriptive_fit["is_law_claim"] is False
    assert descriptive_fit["source"] == "benchmark_local_selection"
    assert descriptive_fit["chart"]["equation_curve"]
    assert descriptive_fit["chart"]["actual_series"]
    assert normalized.get("holistic_equation") is None
    assert normalized.get("predictive_law") is None


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_real_series_saved_analysis_regressions_enforce_top_line_honesty(
    case: RealSeriesRegressionCase,
) -> None:
    payload = _load_analysis_fixture(case)
    raw_holistic = payload.get("holistic_equation")

    if case.expected_raw_holistic_status is None:
        assert raw_holistic in (None, {})
    else:
        assert isinstance(raw_holistic, dict)
        assert raw_holistic.get("status") == case.expected_raw_holistic_status

    normalized = normalize_analysis_payload(payload)
    gap_report = set(normalized.get("gap_report") or [])

    assert normalized["operator_point"]["publication"]["status"] == "abstained"
    assert normalized["benchmark"]["descriptive_fit_status"]["status"] == (
        case.expected_benchmark_status
    )
    assert normalized.get("holistic_equation") is None
    assert normalized.get("predictive_law") is None
    assert normalized.get("publishable") is False
    assert normalized.get("claim_class") == case.expected_claim_class
    assert set(case.required_gap_codes).issubset(gap_report)

    if case.expect_descriptive_fit:
        assert isinstance(normalized.get("descriptive_fit"), dict)
    else:
        assert normalized.get("descriptive_fit") is None
