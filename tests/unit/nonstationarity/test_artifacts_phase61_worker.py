from __future__ import annotations

import importlib

import pytest

from euclid.contracts.errors import ContractValidationError


def _artifacts():
    return importlib.import_module("euclid.nonstationarity.artifacts")


def test_status_manifest_has_stable_field_and_metadata_ordering() -> None:
    artifacts = _artifacts()

    status = artifacts.NonstationarityStatus.failed(
        ("instability_unhandled_by_nonstationary_lane",),
        metadata={"zeta": 2, "alpha": 1},
    )

    manifest = status.as_manifest()
    assert list(manifest) == [
        "status",
        "reason_codes",
        "evidence_refs",
        "metadata",
    ]
    assert list(manifest["metadata"]) == ["alpha", "zeta"]
    assert manifest["reason_codes"] == [
        "instability_unhandled_by_nonstationary_lane"
    ]


def test_status_vocabulary_accepts_only_explicit_runtime_states() -> None:
    artifacts = _artifacts()

    assert tuple(artifacts.NONSTATIONARITY_STATUSES) == (
        "passed",
        "failed",
        "abstained",
        "adapter_unavailable",
        "not_evaluated",
    )
    unavailable = artifacts.NonstationarityStatus.adapter_unavailable(
        "statsmodels_stability_backend_unavailable",
    )
    assert unavailable.status == "adapter_unavailable"

    with pytest.raises(ContractValidationError) as excinfo:
        artifacts.NonstationarityStatus(
            status="unavailable",
            reason_codes=("statsmodels_stability_backend_unavailable",),
        )

    assert excinfo.value.code == "unknown_nonstationarity_status"
    assert excinfo.value.details["status"] == "unavailable"


@pytest.mark.parametrize("generic_code", ["failed", "invalid"])
def test_reason_codes_must_be_specific_nonstationarity_codes(
    generic_code: str,
) -> None:
    artifacts = _artifacts()

    with pytest.raises(ContractValidationError) as excinfo:
        artifacts.NonstationarityStatus.failed((generic_code,))

    assert excinfo.value.code == "unknown_nonstationarity_reason_code"
    assert excinfo.value.details["reason_code"] == generic_code


def test_stability_diagnostic_artifact_preserves_required_fields() -> None:
    artifacts = _artifacts()

    artifact = artifacts.StabilityDiagnosticArtifact(
        diagnostic_id="stab-1",
        series_id="series-a",
        method="cusum_recursive_residuals",
        statistic_name="cusum_supremum",
        statistic_value=1.72,
        p_value=0.02,
        critical_value=1.36,
        status=artifacts.NonstationarityStatus.failed(
            ("cusum_instability_detected",),
            evidence_refs=(
                {
                    "schema_name": "series_manifest@1.0.0",
                    "object_id": "series-a",
                },
            ),
        ),
        window_start=0,
        window_end=127,
        sample_count=128,
        parameters={"max_lag": 4, "trim_fraction": 0.15},
        metadata={"worker": "6.1-common", "lane": "stability"},
    )

    manifest = artifact.as_manifest()
    assert list(manifest) == [
        "schema_name",
        "artifact_type",
        "diagnostic_id",
        "series_id",
        "method",
        "statistic_name",
        "statistic_value",
        "p_value",
        "critical_value",
        "status",
        "reason_codes",
        "evidence_refs",
        "window_start",
        "window_end",
        "sample_count",
        "parameters",
        "metadata",
        "claim_scope",
        "law_claim_allowed",
    ]
    assert manifest["schema_name"] == "stability_diagnostic_artifact@1.0.0"
    assert manifest["artifact_type"] == "stability_diagnostic"
    assert manifest["status"] == "failed"
    assert manifest["reason_codes"] == ["cusum_instability_detected"]
    assert list(manifest["parameters"]) == ["max_lag", "trim_fraction"]
    assert list(manifest["metadata"]) == ["lane", "worker"]


def test_change_point_artifact_preserves_required_fields() -> None:
    artifacts = _artifacts()

    artifact = artifacts.ChangePointArtifact(
        artifact_id="cp-1",
        series_id="series-a",
        method="pelt",
        detected_points=(84, 24),
        penalty=5.5,
        min_segment_size=20,
        tolerance=3,
        status=artifacts.NonstationarityStatus.passed(
            evidence_refs=(
                {
                    "schema_name": "series_manifest@1.0.0",
                    "object_id": "series-a",
                },
            ),
        ),
        sample_count=128,
        cost_model="l2",
        parameters={"jump": 2, "max_breaks": 4},
        metadata={"worker": "6.1-common", "lane": "changepoints"},
    )

    manifest = artifact.as_manifest()
    assert list(manifest) == [
        "schema_name",
        "artifact_type",
        "artifact_id",
        "series_id",
        "method",
        "detected_points",
        "penalty",
        "min_segment_size",
        "tolerance",
        "status",
        "reason_codes",
        "evidence_refs",
        "sample_count",
        "cost_model",
        "parameters",
        "metadata",
        "claim_scope",
        "law_claim_allowed",
    ]
    assert manifest["schema_name"] == "change_point_artifact@1.0.0"
    assert manifest["artifact_type"] == "change_point"
    assert manifest["detected_points"] == [24, 84]
    assert manifest["penalty"] == 5.5
    assert manifest["min_segment_size"] == 20
    assert manifest["tolerance"] == 3


def test_artifact_claim_scope_defaults_to_diagnostic_evidence_not_law_claim() -> None:
    artifacts = _artifacts()
    status = artifacts.NonstationarityStatus.not_evaluated(
        "stability_diagnostic_not_run",
    )

    stability_artifact = artifacts.StabilityDiagnosticArtifact(
        diagnostic_id="stab-1",
        series_id="series-a",
        method="cusum_recursive_residuals",
        statistic_name="cusum_supremum",
        statistic_value=None,
        p_value=None,
        critical_value=None,
        status=status,
        window_start=None,
        window_end=None,
        sample_count=0,
    )
    change_point_artifact = artifacts.ChangePointArtifact(
        artifact_id="cp-1",
        series_id="series-a",
        method="pelt",
        detected_points=(),
        penalty=None,
        min_segment_size=20,
        tolerance=3,
        status=status,
        sample_count=0,
    )

    forbidden_scopes = {"law_claim", "stationary_law", "universal_law"}
    assert stability_artifact.claim_scope not in forbidden_scopes
    assert change_point_artifact.claim_scope not in forbidden_scopes
    assert (
        stability_artifact.as_manifest()["claim_scope"]
        == "diagnostic_evidence_only"
    )
    assert (
        change_point_artifact.as_manifest()["claim_scope"]
        == "diagnostic_evidence_only"
    )
    assert stability_artifact.as_manifest()["law_claim_allowed"] is False
    assert change_point_artifact.as_manifest()["law_claim_allowed"] is False
