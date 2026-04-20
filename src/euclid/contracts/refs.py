from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import (
    ContractCatalog,
    ReferenceFieldDefinition,
    parse_schema_name,
)


@dataclass(frozen=True)
class TypedRef:
    schema_name: str
    object_id: str

    def as_dict(self) -> dict[str, str]:
        return {"schema_name": self.schema_name, "object_id": self.object_id}


@dataclass(frozen=True)
class ManifestBodyRef:
    field_path: str
    ref: TypedRef


@dataclass(frozen=True)
class _ExtractedRef:
    value: Any
    field_path: str
    container: Mapping[str, Any] | None


def validate_typed_ref_payload(
    payload: Any,
    *,
    catalog: ContractCatalog,
    allowed_schema_names: set[str] | frozenset[str] | None = None,
    field_path: str = "body",
) -> TypedRef:
    if not isinstance(payload, Mapping):
        raise ContractValidationError(
            code="invalid_typed_ref_shape",
            message="typed refs must be mappings with schema_name and object_id",
            field_path=field_path,
        )

    for forbidden_field in catalog.typed_ref_shape.forbidden_placeholders:
        if forbidden_field in payload:
            raise ContractValidationError(
                code="forbidden_typed_ref_placeholder",
                message=(
                    "typed refs may not include placeholder field "
                    f"{forbidden_field!r}"
                ),
                field_path=f"{field_path}.{forbidden_field}",
                details={
                    "forbidden_field": forbidden_field,
                    "forbidden_fields": list(
                        catalog.typed_ref_shape.forbidden_placeholders
                    ),
                },
            )

    schema_name = payload.get("schema_name")
    object_id = payload.get("object_id")

    if not isinstance(schema_name, str):
        raise ContractValidationError(
            code="invalid_typed_ref_shape",
            message="typed refs require schema_name to be a string",
            field_path=f"{field_path}.schema_name",
        )
    if not isinstance(object_id, str):
        raise ContractValidationError(
            code="invalid_typed_ref_shape",
            message="typed refs require object_id to be a string",
            field_path=f"{field_path}.object_id",
        )
    if not object_id.strip():
        raise ContractValidationError(
            code="invalid_typed_ref_shape",
            message="typed refs require a non-empty object_id",
            field_path=f"{field_path}.object_id",
        )

    schema_family, schema_version, _ = parse_schema_name(schema_name)
    catalog.get_schema(schema_name)
    if allowed_schema_names and schema_name not in allowed_schema_names:
        allowed_definitions = [
            catalog.get_schema(allowed_schema_name)
            for allowed_schema_name in sorted(allowed_schema_names)
        ]
        allowed_families = sorted(
            {definition.family for definition in allowed_definitions}
        )
        details = {
            "allowed_schema_families": allowed_families,
            "allowed_schema_names": sorted(allowed_schema_names),
            "schema_family": schema_family,
            "schema_name": schema_name,
            "schema_version": schema_version,
        }
        if schema_family not in allowed_families:
            raise ContractValidationError(
                code="typed_ref_family_mismatch",
                message=(
                    f"{schema_name!r} is not an allowed target family for "
                    f"{field_path}"
                ),
                field_path=field_path,
                details=details,
            )

        raise ContractValidationError(
            code="typed_ref_schema_mismatch",
            message=(
                f"{schema_name!r} is not an allowed target version for "
                f"{field_path}"
            ),
            field_path=field_path,
            details=details,
        )

    return TypedRef(schema_name=schema_name, object_id=object_id)


def validate_manifest_body_refs(
    *,
    schema_name: str,
    body: Mapping[str, Any],
    catalog: ContractCatalog,
) -> None:
    extract_manifest_body_refs(
        schema_name=schema_name,
        body=body,
        catalog=catalog,
    )


def extract_manifest_body_refs(
    *,
    schema_name: str,
    body: Mapping[str, Any],
    catalog: ContractCatalog,
) -> tuple[ManifestBodyRef, ...]:
    profile = catalog.get_reference_profile(schema_name)
    if profile is None:
        return ()

    body_refs: list[ManifestBodyRef] = []
    for field in profile.fields:
        matches = _extract_path_values(body, field.path, prefix="body")
        if not matches:
            if field.required:
                raise ContractValidationError(
                    code="missing_required_typed_ref",
                    message=f"{field.path} is required by {schema_name}",
                    field_path=f"body.{field.path}",
                    details={"schema_name": schema_name, "ref_path": field.path},
                )
            continue
        body_refs.extend(_extract_validated_refs(matches, field, catalog=catalog))
    return tuple(body_refs)


def _extract_validated_refs(
    matches: list[_ExtractedRef],
    field: ReferenceFieldDefinition,
    *,
    catalog: ContractCatalog,
) -> tuple[ManifestBodyRef, ...]:
    seen: set[tuple[str, str]] = set()
    resolved: list[ManifestBodyRef] = []
    for match in matches:
        allowed_schema_names = set(field.allowed_schema_names)
        if field.allowed_schema_names_by_discriminator:
            discriminator = None
            if match.container is not None and field.discriminator_field:
                discriminator_value = match.container.get(field.discriminator_field)
                if isinstance(discriminator_value, str):
                    discriminator = discriminator_value
            allowed_schema_names = set(
                field.allowed_schema_names_by_discriminator.get(
                    discriminator or "", frozenset()
                )
            )
        typed_ref = validate_typed_ref_payload(
            match.value,
            catalog=catalog,
            allowed_schema_names=allowed_schema_names or None,
            field_path=match.field_path,
        )
        if field.unique_items:
            key = (typed_ref.schema_name, typed_ref.object_id)
            if key in seen:
                raise ContractValidationError(
                    code="duplicate_typed_ref",
                    message=f"{match.field_path} contains a duplicate typed ref",
                    field_path=match.field_path,
                )
            seen.add(key)
        resolved.append(ManifestBodyRef(field_path=match.field_path, ref=typed_ref))
    return tuple(resolved)


def _extract_path_values(
    data: Any,
    path: str,
    *,
    prefix: str,
) -> list[_ExtractedRef]:
    segments = path.split(".")
    return _extract_segments(data, segments, prefix=prefix, parent=None)


def _extract_segments(
    current: Any,
    segments: list[str],
    *,
    prefix: str,
    parent: Mapping[str, Any] | None,
) -> list[_ExtractedRef]:
    if not segments:
        return [_ExtractedRef(value=current, field_path=prefix, container=parent)]

    segment = segments[0]
    is_list = segment.endswith("[]")
    key = segment[:-2] if is_list else segment

    if not isinstance(current, Mapping):
        raise ContractValidationError(
            code="invalid_ref_container",
            message=f"{prefix} must be a mapping to resolve {segment}",
            field_path=prefix,
        )

    if key not in current:
        return []

    next_value = current[key]
    next_prefix = f"{prefix}.{key}"

    if is_list:
        if not isinstance(next_value, list):
            raise ContractValidationError(
                code="invalid_ref_collection",
                message=f"{next_prefix} must be a list",
                field_path=next_prefix,
            )
        matches: list[_ExtractedRef] = []
        for index, item in enumerate(next_value):
            matches.extend(
                _extract_segments(
                    item,
                    segments[1:],
                    prefix=f"{next_prefix}[{index}]",
                    parent=item if isinstance(item, Mapping) else None,
                )
            )
        return matches

    return _extract_segments(
        next_value,
        segments[1:],
        prefix=next_prefix,
        parent=current if isinstance(current, Mapping) else parent,
    )
