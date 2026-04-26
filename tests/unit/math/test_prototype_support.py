from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.math.codelength import CodelengthPolicy
from euclid.math.observation_models import build_gaussian_observation_model_manifest
from euclid.math.prototype_support import (
    CodelengthPolicyObject,
    ObservationModelObject,
    build_prototype_support_bundle,
    validate_support_object_compatibility,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_prototype_support_bundle_builds_retained_scope_manifests() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    bundle = build_prototype_support_bundle(
        catalog=catalog,
        observed_values=(10.0, 12.0, 14.0),
        quantization_step="0.5",
    )

    assert bundle.target_transform_manifest.schema_name == (
        "target_transform_manifest@1.0.0"
    )
    assert bundle.base_measure_policy_manifest.schema_name == (
        "base_measure_policy_manifest@1.0.0"
    )
    assert bundle.reference_description.quantized_sequence == (20, 24, 28)
    assert bundle.reference_description.reference_bits > 0
    assert bundle.observation_model_manifest.body["family"] == (
        "gaussian_location_scale"
    )
    assert bundle.codelength_policy_manifest.body["quantization_mode"] == (
        "fixed_step_mid_tread"
    )
    assert bundle.codelength_policy_manifest.body[
        "reference_description_policy_ref"
    ] == bundle.reference_description_policy_manifest.ref.as_dict()


def test_fixed_step_raw_reference_mdl_is_explicit_legacy_policy() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    bundle = build_prototype_support_bundle(
        catalog=catalog,
        observed_values=(10.0, 12.0, 14.0),
        quantization_step="0.5",
    )

    assert bundle.codelength_policy_manifest.body["quantization_mode"] == (
        "fixed_step_mid_tread"
    )
    assert bundle.reference_description_policy_manifest.body["reference_kind"] == (
        "raw_quantized_transformed_sequence"
    )
    assert bundle.codelength_policy_manifest.body["compatibility_policy_label"] == (
        "legacy_fixed_step_raw_reference_mdl"
    )
    assert bundle.reference_description_policy_manifest.body[
        "compatibility_policy_label"
    ] == "legacy_raw_reference_description"


def test_prototype_support_bundle_exposes_manifest_backed_runtime_objects() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    bundle = build_prototype_support_bundle(
        catalog=catalog,
        observed_values=(10.0, 12.0, 14.0),
        quantization_step="0.5",
    )

    assert bundle.target_transform_object.manifest == bundle.target_transform_manifest
    assert bundle.quantization_object.manifest == bundle.codelength_policy_manifest
    assert bundle.quantization_object.step_string == "0.5"
    assert bundle.observation_model_object.manifest == bundle.observation_model_manifest
    assert (
        bundle.reference_description_object.manifest
        == bundle.reference_description_policy_manifest
    )
    assert bundle.codelength_policy_object.manifest == bundle.codelength_policy_manifest
    bundle.require_supported_point_loss("absolute_error")

    with pytest.raises(ContractValidationError) as excinfo:
        bundle.require_supported_point_loss("pinball_loss")

    assert excinfo.value.code == "unsupported_point_loss_for_run"


def test_support_object_validation_rejects_target_transform_mismatch() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    bundle = build_prototype_support_bundle(
        catalog=catalog,
        observed_values=(10.0, 12.0, 14.0),
        quantization_step="0.5",
    )
    mismatched_manifest = build_gaussian_observation_model_manifest(
        catalog,
        target_transform_ref=TypedRef(
            "target_transform_manifest@1.0.0",
            "other_transform",
        ),
        base_measure_policy_ref=bundle.base_measure_policy_manifest.ref,
    )

    with pytest.raises(ContractValidationError) as excinfo:
        validate_support_object_compatibility(
            target_transform=bundle.target_transform_object,
            quantization=bundle.quantization_object,
            observation_model=ObservationModelObject(
                runtime=bundle.observation_model,
                manifest=mismatched_manifest,
            ),
            reference_description=bundle.reference_description_object,
            codelength_policy=bundle.codelength_policy_object,
        )

    assert excinfo.value.code == "incompatible_support_objects"
    assert excinfo.value.field_path == "observation_model.target_transform_ref"


def test_support_object_validation_rejects_quantization_runtime_mismatch() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    bundle = build_prototype_support_bundle(
        catalog=catalog,
        observed_values=(10.0, 12.0, 14.0),
        quantization_step="0.5",
    )

    with pytest.raises(ContractValidationError) as excinfo:
        validate_support_object_compatibility(
            target_transform=bundle.target_transform_object,
            quantization=bundle.quantization_object,
            observation_model=bundle.observation_model_object,
            reference_description=bundle.reference_description_object,
            codelength_policy=CodelengthPolicyObject(
                runtime=CodelengthPolicy(quantization_step="1.0"),
                manifest=bundle.codelength_policy_manifest,
            ),
        )

    assert excinfo.value.code == "incompatible_support_objects"
    assert excinfo.value.field_path == "codelength_policy.runtime.quantization_step"


def test_support_object_validation_rejects_reference_description_mismatch() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    bundle = build_prototype_support_bundle(
        catalog=catalog,
        observed_values=(10.0, 12.0, 14.0),
        quantization_step="0.5",
    )
    mismatched_manifest = replace(
        bundle.codelength_policy_manifest,
        body={
            **bundle.codelength_policy_manifest.body,
            "reference_description_policy_ref": {
                "schema_name": "reference_description_policy_manifest@1.1.0",
                "object_id": "other_reference_description",
            },
        },
    )

    with pytest.raises(ContractValidationError) as excinfo:
        validate_support_object_compatibility(
            target_transform=bundle.target_transform_object,
            quantization=bundle.quantization_object,
            observation_model=bundle.observation_model_object,
            reference_description=bundle.reference_description_object,
            codelength_policy=CodelengthPolicyObject(
                runtime=bundle.codelength_policy,
                manifest=mismatched_manifest,
            ),
        )

    assert excinfo.value.code == "incompatible_support_objects"
    assert (
        excinfo.value.field_path
        == "codelength_policy.reference_description_policy_ref"
    )
