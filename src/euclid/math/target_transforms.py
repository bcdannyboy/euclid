from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.manifests.base import ManifestEnvelope


@dataclass(frozen=True)
class IdentityTargetTransform:
    transform_kind: str = "identity"

    def apply(self, values: Iterable[float]) -> tuple[float, ...]:
        normalized = tuple(float(value) for value in values)
        for value in normalized:
            if not math.isfinite(value):
                raise ContractValidationError(
                    code="nonfinite_target_value",
                    message="target transforms require finite inputs",
                    field_path="values",
                )
        return normalized

    def invert(self, values: Iterable[float]) -> tuple[float, ...]:
        return self.apply(values)


def build_identity_target_transform_manifest(
    catalog: ContractCatalog,
) -> ManifestEnvelope:
    return ManifestEnvelope.build(
        schema_name="target_transform_manifest@1.0.0",
        module_id="candidate_fitting",
        body={
            "transform_id": "identity_real_line_transform_v1",
            "owner_prompt_id": "prompt.mdl-observation-v1",
            "scope_id": "euclid_v1_binding_scope@1.0.0",
            "transform_kind": "identity",
            "jacobian_policy": "zero_log_abs_det",
            "transformed_domain": "real_line",
            "inverse_exists": True,
        },
        catalog=catalog,
    )
