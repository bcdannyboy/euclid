from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest


def _changepoints_module() -> Any:
    try:
        return importlib.import_module("euclid.nonstationarity.changepoints")
    except ModuleNotFoundError as exc:  # pragma: no cover - red-phase guard
        pytest.fail(f"missing change-point module: {exc}")


def test_piecewise_constant_series_recovers_breakpoints_within_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changepoints = _changepoints_module()
    fake_ruptures = _fake_ruptures_module(predicted_breakpoints=(31, 66))
    _install_fake_ruptures(monkeypatch, changepoints, fake_ruptures)
    series = [0.0] * 30 + [4.0] * 35 + [-2.0] * 30

    artifact = changepoints.detect_change_points(
        series,
        method="binseg",
        n_bkps=2,
        min_segment_size=10,
        tolerance=2,
    )

    assert artifact.status == "passed"
    assert artifact.breakpoints == (31, 66)
    assert _within_tolerance(artifact.breakpoints, (30, 65), tolerance=2)


def test_breakpoint_artifact_records_method_penalty_min_segment_size_and_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changepoints = _changepoints_module()
    fake_ruptures = _fake_ruptures_module(predicted_breakpoints=(25,))
    _install_fake_ruptures(monkeypatch, changepoints, fake_ruptures)

    artifact = changepoints.detect_change_points(
        [1.0] * 25 + [3.0] * 25,
        method="pelt",
        penalty=7.5,
        min_segment_size=8,
        tolerance=3,
    )

    manifest = artifact.as_manifest()
    assert manifest["artifact_type"] == "change_point"
    assert manifest["status"] == "passed"
    assert manifest["reason_codes"] == []
    assert manifest["breakpoints"] == [25]
    assert manifest["method"] == "pelt"
    assert manifest["penalty"] == 7.5
    assert manifest["min_segment_size"] == 8
    assert manifest["tolerance"] == 3


def test_segments_shorter_than_minimum_length_abstain() -> None:
    changepoints = _changepoints_module()

    artifact = changepoints.detect_change_points(
        [0.0, 1.0, 0.0, 1.0, 0.0],
        method="pelt",
        penalty=1.0,
        min_segment_size=3,
        tolerance=1,
    )

    assert artifact.status == "abstained"
    assert artifact.breakpoints == ()
    assert artifact.reason_codes == ("insufficient_observations_for_min_segment_size",)
    assert artifact.as_manifest()["reason_codes"] == [
        "insufficient_observations_for_min_segment_size"
    ]


def test_missing_ruptures_backend_abstains_with_stable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changepoints = _changepoints_module()

    def raise_import_error(name: str) -> Any:
        if name == "ruptures":
            raise ImportError("ruptures is intentionally unavailable")
        return importlib.import_module(name)

    monkeypatch.setattr(changepoints.importlib, "import_module", raise_import_error)

    artifact = changepoints.detect_change_points(
        [0.0] * 12 + [5.0] * 12,
        method="pelt",
        penalty=2.0,
        min_segment_size=6,
        tolerance=2,
    )

    assert artifact.status == "abstained"
    assert artifact.breakpoints == ()
    assert artifact.reason_codes == ("ruptures_backend_unavailable",)
    assert artifact.as_manifest()["metadata"]["backend"] == "ruptures"


def test_benchmark_change_point_detection_metrics_measure_tolerance_accuracy() -> None:
    from euclid.benchmarks.efficacy_metrics import (
        NONSTATIONARY_DETECTION_DELAY,
        NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE,
        NONSTATIONARY_DETECTION_PRECISION,
        NONSTATIONARY_DETECTION_RECALL,
        compute_efficacy_metric,
    )

    records = [
        {
            "task_id": "phase62_piecewise_demo",
            "truth_change_points": [20, 60],
            "change_point_artifact": {
                "status": "passed",
                "breakpoints": [21, 62, 90],
                "method": "pelt",
                "min_segment_size": 10,
                "tolerance": 2,
            },
        }
    ]

    precision = compute_efficacy_metric(NONSTATIONARY_DETECTION_PRECISION, records)
    recall = compute_efficacy_metric(NONSTATIONARY_DETECTION_RECALL, records)
    hausdorff = compute_efficacy_metric(
        NONSTATIONARY_DETECTION_HAUSDORFF_DISTANCE, records
    )
    delay = compute_efficacy_metric(NONSTATIONARY_DETECTION_DELAY, records)

    assert precision.status == "measured"
    assert precision.observed_value == pytest.approx(2 / 3)
    assert precision.numerator == 2
    assert precision.denominator == 3
    assert precision.details["tolerance"] == 2
    assert recall.observed_value == 1.0
    assert recall.numerator == 2
    assert recall.denominator == 2
    assert hausdorff.observed_value == 30.0
    assert hausdorff.unit == "samples"
    assert delay.observed_value == 1.5
    assert delay.numerator == 3.0
    assert delay.denominator == 2


def _within_tolerance(
    observed: tuple[int, ...],
    expected: tuple[int, ...],
    *,
    tolerance: int,
) -> bool:
    return len(observed) == len(expected) and all(
        abs(actual - target) <= tolerance
        for actual, target in zip(observed, expected, strict=True)
    )


def _fake_ruptures_module(*, predicted_breakpoints: tuple[int, ...]) -> Any:
    class _FakeAlgorithm:
        def __init__(self, *, model: str = "l2", min_size: int = 1, jump: int = 1):
            self.model = model
            self.min_size = min_size
            self.jump = jump
            self.values: list[float] = []

        def fit(self, values: Any) -> "_FakeAlgorithm":
            self.values = list(values)
            return self

        def predict(self, **_: Any) -> list[int]:
            return [*predicted_breakpoints, len(self.values)]

    return SimpleNamespace(Pelt=_FakeAlgorithm, Binseg=_FakeAlgorithm)


def _install_fake_ruptures(
    monkeypatch: pytest.MonkeyPatch,
    changepoints: Any,
    fake_ruptures: Any,
) -> None:
    def import_module(name: str) -> Any:
        if name == "ruptures":
            return fake_ruptures
        return importlib.import_module(name)

    monkeypatch.setattr(changepoints.importlib, "import_module", import_module)
