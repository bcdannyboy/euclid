from __future__ import annotations

from pathlib import Path

from euclid.contracts.loader import load_contract_catalog
from euclid.math.codelength import build_codelength_policy_manifest
from euclid.math.observation_models import build_base_measure_policy_manifest
from euclid.math.quantization import FixedStepMidTreadQuantizer
from euclid.math.reference_descriptions import (
    build_reference_description_policy_manifest,
)
from euclid.math.target_transforms import build_identity_target_transform_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_codelength_policy_manifest_declares_coding_claim_tier() -> None:
    manifest = _build_policy_manifest()

    assert manifest.body["coding_claim_tier"] == "mdl_inspired_proxy_score"


def test_legacy_fixed_step_raw_reference_policy_is_not_universal_mdl_tier() -> None:
    manifest = _build_policy_manifest()

    assert manifest.body["data_code_family"] == (
        "residual_signed_integer_elias_delta_v1"
    )
    assert manifest.body["coding_claim_tier"] not in {
        "mdl_based_universal_code",
        "exact_prequential_symbol_code",
    }
    assert manifest.body["coding_claim_tier_reason_code"] == (
        "legacy_fixed_step_raw_reference_policy_is_proxy_score"
    )


def _build_policy_manifest():
    catalog = load_contract_catalog(PROJECT_ROOT)
    quantizer = FixedStepMidTreadQuantizer.from_string("0.5")
    target_transform_manifest = build_identity_target_transform_manifest(catalog)
    base_measure_policy_manifest = build_base_measure_policy_manifest(catalog)
    reference_description_policy_manifest = (
        build_reference_description_policy_manifest(catalog)
    )

    return build_codelength_policy_manifest(
        catalog,
        quantizer=quantizer,
        target_transform_ref=target_transform_manifest.ref,
        base_measure_policy_ref=base_measure_policy_manifest.ref,
        reference_description_policy_ref=reference_description_policy_manifest.ref,
    )
