from __future__ import annotations

from dataclasses import dataclass

from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope


@dataclass(frozen=True)
class PointObservationModel:
    family: str = "gaussian_location_scale"
    compatible_point_losses: tuple[str, ...] = ("squared_error", "absolute_error")

    def supports_point_loss(self, loss_name: str) -> bool:
        return loss_name in self.compatible_point_losses


def build_base_measure_policy_manifest(catalog: ContractCatalog) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="base_measure_policy_manifest@1.0.0",
        module_id="candidate_fitting",
        body={
            "base_measure_policy_id": "lebesgue_real_line_policy_v1",
            "owner_prompt_id": "prompt.mdl-observation-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "measure_kind": "lebesgue_on_transformed_real_line",
            "bin_mass_rule": "cdf_difference_over_quantization_bin",
            "support_kind": "all_real",
        },
        catalog=catalog,
    )


def build_gaussian_observation_model_manifest(
    catalog: ContractCatalog,
    *,
    target_transform_ref: TypedRef,
    base_measure_policy_ref: TypedRef,
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="observation_model_manifest@1.1.0",
        module_id="candidate_fitting",
        body={
            "observation_model_id": "gaussian_point_observation_model_v1",
            "owner_prompt_id": "prompt.mdl-observation-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "target_kind": "real_scalar",
            "forecast_type": "point",
            "target_transform_ref": target_transform_ref.as_dict(),
            "base_measure_policy_ref": base_measure_policy_ref.as_dict(),
            "family": "gaussian_location_scale",
            "location_semantics": "reducer_emits_location_parameter",
            "scale_scope": "global_scalar",
            "degrees_of_freedom": None,
            "point_forecast_output": "location_parameter",
            "compatible_point_losses": ["squared_error", "absolute_error"],
            "cdf_evaluation_required": True,
            "support_kind": "all_real",
        },
        catalog=catalog,
    )
