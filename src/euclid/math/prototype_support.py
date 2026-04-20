from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.base import ManifestEnvelope
from euclid.math.codelength import (
    CodelengthPolicy,
    build_codelength_policy_manifest,
)
from euclid.math.observation_models import (
    PointObservationModel,
    build_base_measure_policy_manifest,
    build_gaussian_observation_model_manifest,
)
from euclid.math.quantization import FixedStepMidTreadQuantizer
from euclid.math.reference_descriptions import (
    ReferenceDescription,
    build_reference_description,
    build_reference_description_policy_manifest,
)
from euclid.math.target_transforms import (
    IdentityTargetTransform,
    build_identity_target_transform_manifest,
)


def _typed_ref(payload: Any, *, field_path: str) -> TypedRef:
    if not isinstance(payload, Mapping):
        raise ContractValidationError(
            code="invalid_typed_ref_shape",
            message="typed refs must be mappings with schema_name and object_id",
            field_path=field_path,
        )
    schema_name = payload.get("schema_name")
    object_id = payload.get("object_id")
    if not isinstance(schema_name, str) or not isinstance(object_id, str):
        raise ContractValidationError(
            code="invalid_typed_ref_shape",
            message="typed refs require schema_name and object_id strings",
            field_path=field_path,
        )
    return TypedRef(schema_name=schema_name, object_id=object_id)


def _raise_incompatible(
    *,
    field_path: str,
    message: str,
    expected: Any,
    actual: Any,
) -> None:
    raise ContractValidationError(
        code="incompatible_support_objects",
        message=message,
        field_path=field_path,
        details={"expected": expected, "actual": actual},
    )


@dataclass(frozen=True)
class TargetTransformObject:
    runtime: IdentityTargetTransform
    manifest: ManifestEnvelope

    @property
    def transform_kind(self) -> str:
        return self.runtime.transform_kind

    def apply(self, values: Iterable[float]) -> tuple[float, ...]:
        return self.runtime.apply(values)

    def invert(self, values: Iterable[float]) -> tuple[float, ...]:
        return self.runtime.invert(values)


@dataclass(frozen=True)
class QuantizationObject:
    runtime: FixedStepMidTreadQuantizer
    manifest: ManifestEnvelope
    quantization_mode: str = "fixed_step_mid_tread"

    @property
    def step_string(self) -> str:
        return self.runtime.step_string

    def quantize_index(self, value: float) -> int:
        return self.runtime.quantize_index(value)

    def quantize_sequence(self, values: Iterable[float]) -> tuple[int, ...]:
        return self.runtime.quantize_sequence(values)


@dataclass(frozen=True)
class ObservationModelObject:
    runtime: PointObservationModel
    manifest: ManifestEnvelope

    @property
    def compatible_point_losses(self) -> tuple[str, ...]:
        return self.runtime.compatible_point_losses

    def supports_point_loss(self, loss_name: str) -> bool:
        return self.runtime.supports_point_loss(loss_name)

    def require_supported_point_loss(self, loss_name: str) -> None:
        if not self.supports_point_loss(loss_name):
            raise ContractValidationError(
                code="unsupported_point_loss_for_run",
                message=(
                    "point loss is not compatible with the run-level "
                    "observation model"
                ),
                field_path="point_loss_id",
                details={
                    "point_loss_id": loss_name,
                    "compatible_point_losses": self.compatible_point_losses,
                },
            )


@dataclass(frozen=True)
class ReferenceDescriptionObject:
    runtime: ReferenceDescription
    manifest: ManifestEnvelope

    @property
    def quantized_sequence(self) -> tuple[int, ...]:
        return self.runtime.quantized_sequence

    @property
    def reference_bits(self) -> int:
        return self.runtime.reference_bits


@dataclass(frozen=True)
class CodelengthPolicyObject:
    runtime: CodelengthPolicy
    manifest: ManifestEnvelope

    @property
    def quantization_step(self) -> str:
        return self.runtime.quantization_step


def validate_support_object_compatibility(
    *,
    target_transform: TargetTransformObject,
    quantization: QuantizationObject,
    observation_model: ObservationModelObject,
    reference_description: ReferenceDescriptionObject,
    codelength_policy: CodelengthPolicyObject,
) -> None:
    transform_kind = target_transform.manifest.body.get("transform_kind")
    if transform_kind != target_transform.transform_kind:
        _raise_incompatible(
            field_path="target_transform.transform_kind",
            message="target transform runtime and manifest disagree",
            expected=target_transform.transform_kind,
            actual=transform_kind,
        )

    observation_transform_ref = _typed_ref(
        observation_model.manifest.body.get("target_transform_ref"),
        field_path="observation_model.target_transform_ref",
    )
    if observation_transform_ref != target_transform.manifest.ref:
        _raise_incompatible(
            field_path="observation_model.target_transform_ref",
            message="observation model must bind the run target transform",
            expected=target_transform.manifest.ref.as_dict(),
            actual=observation_transform_ref.as_dict(),
        )

    codelength_transform_ref = _typed_ref(
        codelength_policy.manifest.body.get("target_transform_ref"),
        field_path="codelength_policy.target_transform_ref",
    )
    if codelength_transform_ref != target_transform.manifest.ref:
        _raise_incompatible(
            field_path="codelength_policy.target_transform_ref",
            message="codelength policy must bind the run target transform",
            expected=target_transform.manifest.ref.as_dict(),
            actual=codelength_transform_ref.as_dict(),
        )

    quantization_mode = codelength_policy.manifest.body.get("quantization_mode")
    if quantization_mode != quantization.quantization_mode:
        _raise_incompatible(
            field_path="codelength_policy.quantization_mode",
            message="codelength policy and quantization object disagree on mode",
            expected=quantization.quantization_mode,
            actual=quantization_mode,
        )

    for field_name in (
        "quantization_step",
        "literal_lattice_step",
        "parameter_lattice_step",
        "state_lattice_step",
    ):
        manifest_value = codelength_policy.manifest.body.get(field_name)
        if manifest_value != quantization.step_string:
            _raise_incompatible(
                field_path=f"codelength_policy.{field_name}",
                message=(
                    "codelength policy and quantization object disagree on "
                    "the retained lattice step"
                ),
                expected=quantization.step_string,
                actual=manifest_value,
            )

    if codelength_policy.quantization_step != quantization.step_string:
        _raise_incompatible(
            field_path="codelength_policy.runtime.quantization_step",
            message="codelength policy runtime and quantization object disagree",
            expected=quantization.step_string,
            actual=codelength_policy.quantization_step,
        )

    support_kind = observation_model.manifest.body.get("support_kind")
    transformed_domain = target_transform.manifest.body.get("transformed_domain")
    if support_kind != "all_real" or transformed_domain != "real_line":
        _raise_incompatible(
            field_path="observation_model.support_kind",
            message=(
                "retained support objects require an all-real observation model "
                "over a real-line transformed domain"
            ),
            expected={"support_kind": "all_real", "transformed_domain": "real_line"},
            actual={
                "support_kind": support_kind,
                "transformed_domain": transformed_domain,
            },
        )

    manifest_losses = tuple(
        str(value)
        for value in observation_model.manifest.body.get(
            "compatible_point_losses",
            (),
        )
    )
    if manifest_losses != observation_model.compatible_point_losses:
        _raise_incompatible(
            field_path="observation_model.compatible_point_losses",
            message=(
                "observation model runtime losses must match the manifest-backed "
                "run semantics"
            ),
            expected=observation_model.compatible_point_losses,
            actual=manifest_losses,
        )

    reference_policy_ref = _typed_ref(
        codelength_policy.manifest.body.get("reference_description_policy_ref"),
        field_path="codelength_policy.reference_description_policy_ref",
    )
    if reference_policy_ref != reference_description.manifest.ref:
        _raise_incompatible(
            field_path="codelength_policy.reference_description_policy_ref",
            message="codelength policy must bind the run reference description",
            expected=reference_description.manifest.ref.as_dict(),
            actual=reference_policy_ref.as_dict(),
        )


@dataclass(frozen=True)
class PrototypeSupportBundle:
    target_transform_object: TargetTransformObject
    target_transform: IdentityTargetTransform
    target_transform_manifest: ManifestEnvelope
    quantization_object: QuantizationObject
    quantizer: FixedStepMidTreadQuantizer
    base_measure_policy_manifest: ManifestEnvelope
    reference_description_object: ReferenceDescriptionObject
    reference_description: ReferenceDescription
    reference_description_policy_manifest: ManifestEnvelope
    observation_model_object: ObservationModelObject
    observation_model: PointObservationModel
    observation_model_manifest: ManifestEnvelope
    codelength_policy_object: CodelengthPolicyObject
    codelength_policy: CodelengthPolicy
    codelength_policy_manifest: ManifestEnvelope

    def require_supported_point_loss(self, loss_name: str) -> None:
        self.observation_model_object.require_supported_point_loss(loss_name)

    def point_loss(
        self,
        *,
        point_loss_id: str,
        point_forecast: float,
        realized_observation: float,
    ) -> float:
        self.require_supported_point_loss(point_loss_id)
        error = point_forecast - realized_observation
        if point_loss_id == "absolute_error":
            return abs(error)
        return error**2


def build_prototype_support_bundle(
    *,
    catalog: ContractCatalog,
    observed_values: Iterable[float],
    quantization_step: str = "0.5",
) -> PrototypeSupportBundle:
    target_transform = IdentityTargetTransform()
    target_transform_manifest = build_identity_target_transform_manifest(catalog)
    quantizer = FixedStepMidTreadQuantizer.from_string(quantization_step)
    base_measure_policy_manifest = build_base_measure_policy_manifest(catalog)
    reference_description = build_reference_description(
        observed_values,
        quantizer=quantizer,
    )
    reference_description_policy_manifest = (
        build_reference_description_policy_manifest(catalog)
    )
    observation_model = PointObservationModel()
    observation_model_manifest = build_gaussian_observation_model_manifest(
        catalog,
        target_transform_ref=target_transform_manifest.ref,
        base_measure_policy_ref=base_measure_policy_manifest.ref,
    )
    codelength_policy = CodelengthPolicy(quantization_step=quantizer.step_string)
    codelength_policy_manifest = build_codelength_policy_manifest(
        catalog,
        quantizer=quantizer,
        target_transform_ref=target_transform_manifest.ref,
        base_measure_policy_ref=base_measure_policy_manifest.ref,
        reference_description_policy_ref=reference_description_policy_manifest.ref,
    )
    target_transform_object = TargetTransformObject(
        runtime=target_transform,
        manifest=target_transform_manifest,
    )
    quantization_object = QuantizationObject(
        runtime=quantizer,
        manifest=codelength_policy_manifest,
    )
    reference_description_object = ReferenceDescriptionObject(
        runtime=reference_description,
        manifest=reference_description_policy_manifest,
    )
    observation_model_object = ObservationModelObject(
        runtime=observation_model,
        manifest=observation_model_manifest,
    )
    codelength_policy_object = CodelengthPolicyObject(
        runtime=codelength_policy,
        manifest=codelength_policy_manifest,
    )
    validate_support_object_compatibility(
        target_transform=target_transform_object,
        quantization=quantization_object,
        observation_model=observation_model_object,
        reference_description=reference_description_object,
        codelength_policy=codelength_policy_object,
    )
    return PrototypeSupportBundle(
        target_transform_object=target_transform_object,
        target_transform=target_transform,
        target_transform_manifest=target_transform_manifest,
        quantization_object=quantization_object,
        quantizer=quantizer,
        base_measure_policy_manifest=base_measure_policy_manifest,
        reference_description_object=reference_description_object,
        reference_description=reference_description,
        reference_description_policy_manifest=reference_description_policy_manifest,
        observation_model_object=observation_model_object,
        observation_model=observation_model,
        observation_model_manifest=observation_model_manifest,
        codelength_policy_object=codelength_policy_object,
        codelength_policy=codelength_policy,
        codelength_policy_manifest=codelength_policy_manifest,
    )
