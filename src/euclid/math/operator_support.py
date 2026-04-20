from __future__ import annotations

from euclid.math.prototype_support import (
    CodelengthPolicyObject,
    ObservationModelObject,
    PrototypeSupportBundle,
    QuantizationObject,
    ReferenceDescriptionObject,
    TargetTransformObject,
    build_prototype_support_bundle,
)

OperatorSupportBundle = PrototypeSupportBundle


def build_operator_support_bundle(*args, **kwargs) -> OperatorSupportBundle:
    return build_prototype_support_bundle(*args, **kwargs)


__all__ = [
    "CodelengthPolicyObject",
    "ObservationModelObject",
    "OperatorSupportBundle",
    "QuantizationObject",
    "ReferenceDescriptionObject",
    "TargetTransformObject",
    "build_operator_support_bundle",
]
