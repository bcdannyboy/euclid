from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog
from euclid.contracts.refs import TypedRef, validate_manifest_body_refs
from euclid.runtime.hashing import (
    canonicalize_json,
    normalize_json_value,
    sha256_digest,
)
from euclid.runtime.ids import make_content_addressed_id


@dataclass(frozen=True)
class ManifestEnvelope:
    schema_name: str
    object_id: str
    module_id: str
    body: Mapping[str, Any]
    ref: TypedRef
    content_hash: str
    canonical_json: str

    @classmethod
    def build(
        cls,
        *,
        schema_name: str,
        module_id: str,
        body: Mapping[str, Any],
        catalog: ContractCatalog,
        object_id: str | None = None,
    ) -> "ManifestEnvelope":
        catalog.get_schema(schema_name)
        catalog.get_module(module_id)

        if not catalog.module_owns_schema(module_id, schema_name):
            owning_module = catalog.get_schema(schema_name).owning_module
            raise ContractValidationError(
                code="schema_module_mismatch",
                message=(
                    f"{schema_name} is owned by {owning_module}, "
                    f"not {module_id}"
                ),
                field_path="module_id",
                details={"schema_name": schema_name, "module_id": module_id},
            )

        normalized_body = normalize_json_value(body)
        validate_manifest_body_refs(
            schema_name=schema_name, body=normalized_body, catalog=catalog
        )

        identity_payload = {
            "body": normalized_body,
            "module_id": module_id,
            "schema_name": schema_name,
        }
        content_hash = sha256_digest(identity_payload)
        resolved_object_id = object_id or make_content_addressed_id(
            schema_name, identity_payload
        )
        if not isinstance(resolved_object_id, str) or not resolved_object_id.strip():
            raise ContractValidationError(
                code="invalid_object_id",
                message="object_id must be a non-empty string",
                field_path="object_id",
            )

        ref = TypedRef(schema_name=schema_name, object_id=resolved_object_id)
        canonical_json = canonicalize_json(
            {
                "body": normalized_body,
                "module_id": module_id,
                "object_id": resolved_object_id,
                "schema_name": schema_name,
            }
        )

        return cls(
            schema_name=schema_name,
            object_id=resolved_object_id,
            module_id=module_id,
            body=normalized_body,
            ref=ref,
            content_hash=content_hash,
            canonical_json=canonical_json,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "body": self.body,
            "module_id": self.module_id,
            "object_id": self.object_id,
            "schema_name": self.schema_name,
        }
