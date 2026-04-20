from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import validate_typed_ref_payload
from euclid.manifests.base import ManifestEnvelope

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_validate_typed_ref_payload_rejects_non_string_object_ids() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    with pytest.raises(ContractValidationError) as excinfo:
        validate_typed_ref_payload(
            {"schema_name": "scorecard_manifest@1.1.0", "object_id": ["bad"]},
            catalog=catalog,
        )

    assert excinfo.value.code == "invalid_typed_ref_shape"


def test_validate_typed_ref_payload_rejects_invalid_schema_name_format() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    with pytest.raises(ContractValidationError) as excinfo:
        validate_typed_ref_payload(
            {"schema_name": "scorecard_manifest", "object_id": "demo_scorecard"},
            catalog=catalog,
        )

    assert excinfo.value.code == "invalid_schema_name_format"


def test_validate_typed_ref_payload_rejects_forbidden_placeholder_keys() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    with pytest.raises(ContractValidationError) as excinfo:
        validate_typed_ref_payload(
            {
                "schema_name": "scorecard_manifest@1.1.0",
                "object_id": "demo_scorecard",
                "free_string_ref": "scorecard_manifest@1.1.0:demo_scorecard",
            },
            catalog=catalog,
        )

    assert excinfo.value.as_dict() == {
        "code": "forbidden_typed_ref_placeholder",
        "message": "typed refs may not include placeholder field 'free_string_ref'",
        "field_path": "body.free_string_ref",
        "details": {
            "forbidden_field": "free_string_ref",
            "forbidden_fields": [
                "free_string_ref",
                "prose_only_selector",
                "schema_less_object_id",
            ],
        },
    }


def test_manifest_envelope_rejects_incompatible_ref_target_schema() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    with pytest.raises(ContractValidationError) as excinfo:
        ManifestEnvelope.build(
            schema_name="baseline_registry_manifest@1.1.0",
            module_id="evaluation_governance",
            body={
                "baseline_registry_id": "demo_baselines",
                "compatible_point_score_policy_ref": {
                    "schema_name": "dataset_snapshot_manifest@1.0.0",
                    "object_id": "demo_wrong_target",
                },
            },
            catalog=catalog,
        )

    assert excinfo.value.code == "typed_ref_family_mismatch"
    assert excinfo.value.field_path == "body.compatible_point_score_policy_ref"
    assert excinfo.value.details == {
        "allowed_schema_families": [
            "event_probability_score_policy_manifest",
            "interval_score_policy_manifest",
            "point_score_policy_manifest",
            "probabilistic_score_policy_manifest",
            "quantile_score_policy_manifest",
        ],
        "allowed_schema_names": [
            "event_probability_score_policy_manifest@1.0.0",
            "interval_score_policy_manifest@1.0.0",
            "point_score_policy_manifest@1.0.0",
            "probabilistic_score_policy_manifest@1.0.0",
            "quantile_score_policy_manifest@1.0.0",
        ],
        "schema_family": "dataset_snapshot_manifest",
        "schema_name": "dataset_snapshot_manifest@1.0.0",
        "schema_version": "1.0.0",
    }


def test_manifest_envelope_rejects_publication_record_run_result_family_mismatch(
) -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    with pytest.raises(ContractValidationError) as excinfo:
        ManifestEnvelope.build(
            schema_name="publication_record_manifest@1.1.0",
            module_id="catalog_publishing",
            body={
                "publication_id": "demo_publication_record",
                "run_result_ref": {
                    "schema_name": "scorecard_manifest@1.1.0",
                    "object_id": "not_a_run_result",
                },
                "catalog_scope": "public",
                "publication_mode": "candidate_publication",
                "replay_verification_status": "verified",
                "comparator_exposure_status": "satisfied",
                "reproducibility_bundle_ref": {
                    "schema_name": "reproducibility_bundle_manifest@1.0.0",
                    "object_id": "bundle",
                },
                "readiness_judgment_ref": {
                    "schema_name": "readiness_judgment_manifest@1.0.0",
                    "object_id": "ready",
                },
                "schema_lifecycle_integration_closure_ref": {
                    "schema_name": (
                        "schema_lifecycle_integration_closure_manifest@1.0.0"
                    ),
                    "object_id": "closure",
                },
                "published_at": "2026-04-12T00:00:00Z",
            },
            catalog=catalog,
        )

    assert excinfo.value.code == "typed_ref_family_mismatch"
    assert excinfo.value.field_path == "body.run_result_ref"
