from __future__ import annotations

from dataclasses import dataclass

from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.math.quantization import FixedStepMidTreadQuantizer


@dataclass(frozen=True)
class CodelengthPolicy:
    quantization_step: str
    literal_code_family: str = "zigzag_elias_delta_v1"
    parameter_code_family: str = "zigzag_elias_delta_v1"
    state_code_family: str = "zigzag_elias_delta_v1"


def build_codelength_policy_manifest(
    catalog: ContractCatalog,
    *,
    quantizer: FixedStepMidTreadQuantizer,
    target_transform_ref: TypedRef,
    base_measure_policy_ref: TypedRef,
    reference_description_policy_ref: TypedRef,
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="codelength_policy_manifest@1.1.0",
        module_id="candidate_fitting",
        body={
            "policy_id": "prototype_codelength_policy_v1",
            "owner_prompt_id": "prompt.mdl-observation-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "target_transform_ref": target_transform_ref.as_dict(),
            "base_measure_policy_ref": base_measure_policy_ref.as_dict(),
            "quantization_mode": "fixed_step_mid_tread",
            "quantization_step": quantizer.step_string,
            "family_code_policy": "fixed_v1_dyadic_2bit",
            "literal_code_family": "zigzag_elias_delta_v1",
            "literal_lattice_step": quantizer.step_string,
            "parameter_code_family": "zigzag_elias_delta_v1",
            "parameter_lattice_step": quantizer.step_string,
            "state_code_family": "zigzag_elias_delta_v1",
            "state_lattice_step": quantizer.step_string,
            "reference_description_policy_ref": (
                reference_description_policy_ref.as_dict()
            ),
            "cross_family_comparison_rule": (
                "all_policy_refs_and_observation_family_support_equal"
            ),
        },
        catalog=catalog,
    )
