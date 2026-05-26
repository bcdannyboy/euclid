from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


def _load_change_point_detector() -> Callable[..., Any]:
    try:
        from euclid.nonstationarity.changepoints import detect_change_points
    except ModuleNotFoundError:
        pytest.fail(
            "Phase 6 change-point API is missing; expected "
            "euclid.nonstationarity.changepoints.detect_change_points"
        )
    return detect_change_points


def _assert_breakpoints_within_tolerance(
    detected: tuple[int, ...],
    expected: tuple[int, ...],
    *,
    tolerance: int,
) -> None:
    assert len(detected) == len(expected)
    for detected_break, expected_break in zip(detected, expected, strict=True):
        assert abs(detected_break - expected_break) <= tolerance


def test_synthetic_piecewise_constant_series_recovers_breakpoints_within_tolerance() -> None:
    detect_change_points = _load_change_point_detector()
    expected_breakpoints = (50, 90)
    series = tuple([1.0] * 50 + [6.0] * 40 + [-2.0] * 60)

    artifact = detect_change_points(
        series=series,
        method="pelt_l2",
        penalty=4.0,
        min_segment_size=20,
        tolerance=3,
    )
    manifest = artifact.as_manifest()

    assert manifest["schema_name"] == "change_point_artifact@1.0.0"
    assert manifest["status"] == "passed"
    _assert_breakpoints_within_tolerance(
        tuple(manifest["detected_change_points"]),
        expected_breakpoints,
        tolerance=3,
    )


def test_breakpoint_artifact_records_detector_configuration() -> None:
    detect_change_points = _load_change_point_detector()
    series = tuple([0.0] * 36 + [4.0] * 36 + [9.0] * 36)

    artifact = detect_change_points(
        series=series,
        method="binseg_l2",
        penalty=2.5,
        min_segment_size=18,
        tolerance=4,
    )
    manifest = artifact.as_manifest()

    assert manifest["method"] == "binseg_l2"
    assert manifest["penalty"] == 2.5
    assert manifest["min_segment_size"] == 18
    assert manifest["tolerance"] == 4


def test_segments_shorter_than_minimum_length_abstain() -> None:
    detect_change_points = _load_change_point_detector()
    series = tuple([0.0] * 8 + [10.0] * 8)

    artifact = detect_change_points(
        series=series,
        method="pelt_l2",
        penalty=1.0,
        min_segment_size=12,
        tolerance=2,
    )
    manifest = artifact.as_manifest()

    assert manifest["status"] == "abstained"
    assert manifest["detected_change_points"] == []
    assert "segment_shorter_than_minimum_length" in manifest["reason_codes"]
