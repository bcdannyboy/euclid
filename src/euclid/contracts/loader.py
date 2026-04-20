from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from euclid.contracts.errors import ContractValidationError

_SEMVER_PATTERN = re.compile(
    r"^(?P<family>[A-Za-z0-9_]+)@(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$"
)


@dataclass(frozen=True, order=True)
class SchemaVersion:
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class SchemaDefinition:
    schema_name: str
    family: str
    version: str
    version_info: SchemaVersion
    owner_ref: str
    owning_module: str
    canonical_source_path: str


@dataclass(frozen=True)
class ModuleDefinition:
    module: str
    owner_ref: str
    plane: str


@dataclass(frozen=True)
class EnumDefinition:
    enum_name: str
    owner_ref: str
    canonical_source_path: str
    allowed_values: tuple[str, ...]
    source_kind: str


@dataclass(frozen=True)
class TypedRefShape:
    required_fields: tuple[str, ...]
    forbidden_placeholders: tuple[str, ...]
    optional_ref_rule: str | None = None


@dataclass(frozen=True)
class ReferenceFieldDefinition:
    path: str
    required: bool
    cardinality: str
    allowed_schema_names: frozenset[str]
    allowed_schema_names_by_discriminator: dict[str, frozenset[str]]
    discriminator_field: str | None = None
    unique_items: bool = False
    required_when: str | None = None

    @property
    def allowed_schema_families(self) -> frozenset[str]:
        return frozenset(
            {
                parse_schema_name(schema_name)[0]
                for schema_name in self.allowed_schema_names
            }
        )


@dataclass(frozen=True)
class ReferenceProfile:
    schema_name: str
    fields: tuple[ReferenceFieldDefinition, ...]


@dataclass(frozen=True)
class ContractDocument:
    name: str
    path: str
    kind: str
    version: int | str | None
    payload: Mapping[str, Any]
    state_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContractCatalog:
    root: Path
    schemas: dict[str, SchemaDefinition]
    schemas_by_family: dict[str, tuple[SchemaDefinition, ...]]
    modules: dict[str, ModuleDefinition]
    enums: dict[str, EnumDefinition]
    typed_ref_shape: TypedRefShape
    reference_profiles: dict[str, ReferenceProfile]
    contract_documents_by_name: dict[str, ContractDocument]
    contract_documents_by_kind: dict[str, ContractDocument]

    def get_schema(self, schema_name: str) -> SchemaDefinition:
        parse_schema_name(schema_name)
        try:
            return self.schemas[schema_name]
        except KeyError as exc:
            raise ContractValidationError(
                code="unknown_schema_name",
                message=(
                    f"{schema_name!r} is not declared in "
                    "schemas/contracts/schema-registry.yaml"
                ),
                field_path="schema_name",
                details={"schema_name": schema_name},
            ) from exc

    def get_module(self, module_id: str) -> ModuleDefinition:
        try:
            return self.modules[module_id]
        except KeyError as exc:
            raise ContractValidationError(
                code="unknown_module_id",
                message=(
                    f"{module_id!r} is not declared in "
                    "schemas/contracts/module-registry.yaml"
                ),
                field_path="module_id",
                details={"module_id": module_id},
            ) from exc

    def get_enum(self, enum_name: str) -> EnumDefinition:
        try:
            return self.enums[enum_name]
        except KeyError as exc:
            raise ContractValidationError(
                code="unknown_enum_name",
                message=(
                    f"{enum_name!r} is not declared in "
                    "schemas/contracts/enum-registry.yaml"
                ),
                field_path="enum_name",
                details={"enum_name": enum_name},
            ) from exc

    def get_contract_document(self, name_or_kind: str) -> ContractDocument:
        document = self.contract_documents_by_kind.get(name_or_kind)
        if document is not None:
            return document
        document = self.contract_documents_by_name.get(name_or_kind)
        if document is not None:
            return document
        raise ContractValidationError(
            code="unknown_contract_document",
            message=f"{name_or_kind!r} is not a known contract document",
            field_path="contract_document",
            details={"contract_document": name_or_kind},
        )

    def schema_names_for_family(self, family: str) -> tuple[str, ...]:
        definitions = self.schemas_by_family.get(family, ())
        return tuple(definition.schema_name for definition in definitions)

    def module_owns_schema(self, module_id: str, schema_name: str) -> bool:
        return self.get_schema(schema_name).owning_module == module_id

    def enum_values(self, enum_name: str) -> tuple[str, ...]:
        return self.get_enum(enum_name).allowed_values

    def get_reference_profile(self, schema_name: str) -> ReferenceProfile | None:
        return self.reference_profiles.get(schema_name)

    def allowed_ref_schema_names(
        self,
        schema_name: str,
        field_path: str,
        *,
        discriminator: str | None = None,
    ) -> set[str]:
        profile = self.get_reference_profile(schema_name)
        if profile is None:
            return set()
        for field in profile.fields:
            if field.path != field_path:
                continue
            if field.allowed_schema_names_by_discriminator:
                if discriminator is None:
                    return set()
                return set(
                    field.allowed_schema_names_by_discriminator.get(
                        discriminator, frozenset()
                    )
                )
            return set(field.allowed_schema_names)
        return set()


def load_contract_catalog(root: Path | str | None = None) -> ContractCatalog:
    return _load_contract_catalog_cached(_resolve_contract_catalog_root(root))


def _resolve_contract_catalog_root(root: Path | str | None) -> Path:
    from euclid.operator_runtime.resources import resolve_contract_root

    return resolve_contract_root(root)


@lru_cache(maxsize=None)
def _load_contract_catalog_cached(root: Path) -> ContractCatalog:
    contract_documents = _load_contract_documents(root)
    schemas_data = _require_contract_document(
        contract_documents, "schema-registry", "schema_registry"
    ).payload
    modules_data = _require_contract_document(
        contract_documents, "module-registry", "module_registry"
    ).payload
    enums_data = _require_contract_document(
        contract_documents, "enum-registry", "enum_registry"
    ).payload
    refs_document = _require_contract_document(
        contract_documents, "reference-types", "reference_type_registry"
    )
    refs_data = refs_document.payload

    schemas = {
        item["schema_name"]: _build_schema_definition(item)
        for item in schemas_data["schemas"]
    }
    schemas_by_family = _group_schemas_by_family(schemas.values())
    modules = {
        item["module"]: ModuleDefinition(
            module=item["module"],
            owner_ref=item["owner_ref"],
            plane=item["plane"],
        )
        for item in modules_data["modules"]
    }
    reference_profiles = {
        item["schema_name"]: ReferenceProfile(
            schema_name=item["schema_name"],
            fields=tuple(
                ReferenceFieldDefinition(
                    path=field["path"],
                    required=bool(field.get("required", False)),
                    cardinality=field.get("cardinality", "unspecified"),
                    allowed_schema_names=frozenset(
                        field.get("allowed_schema_names", [])
                    ),
                    allowed_schema_names_by_discriminator={
                        discriminator: frozenset(values)
                        for discriminator, values in field.get(
                            "allowed_schema_names_by_discriminator",
                            {},
                        ).items()
                    },
                    discriminator_field=field.get("discriminator_field"),
                    unique_items=bool(field.get("unique_items", False)),
                    required_when=field.get("required_when"),
                )
                for field in item["fields"]
            ),
        )
        for item in refs_data["reference_profiles"]
    }
    enums = {
        item["enum_name"]: _build_enum_definition(root, item)
        for item in enums_data["enums"]
    }
    typed_ref_shape_data = refs_data["typed_ref_shape"]
    typed_ref_shape = TypedRefShape(
        required_fields=tuple(typed_ref_shape_data["required_fields"]),
        forbidden_placeholders=tuple(
            typed_ref_shape_data["forbidden_placeholders"]
        ),
        optional_ref_rule=typed_ref_shape_data.get("optional_ref_rule"),
    )

    contract_documents_by_name = {
        document.name: document for document in contract_documents
    }
    contract_documents_by_kind = {
        document.kind: document for document in contract_documents if document.kind
    }

    return ContractCatalog(
        root=root,
        schemas=schemas,
        schemas_by_family=schemas_by_family,
        modules=modules,
        enums=enums,
        typed_ref_shape=typed_ref_shape,
        reference_profiles=reference_profiles,
        contract_documents_by_name=contract_documents_by_name,
        contract_documents_by_kind=contract_documents_by_kind,
    )


def parse_schema_name(schema_name: str) -> tuple[str, str, SchemaVersion]:
    if not isinstance(schema_name, str):
        raise ContractValidationError(
            code="invalid_schema_name_format",
            message="schema_name must be a string in family@major.minor.patch form",
            field_path="schema_name",
            details={"schema_name": schema_name},
        )
    match = _SEMVER_PATTERN.match(schema_name)
    if match is None:
        raise ContractValidationError(
            code="invalid_schema_name_format",
            message=(
                "schema_name must use the form "
                "'family@major.minor.patch'"
            ),
            field_path="schema_name",
            details={"schema_name": schema_name},
        )
    version = SchemaVersion(
        major=int(match.group("major")),
        minor=int(match.group("minor")),
        patch=int(match.group("patch")),
    )
    return match.group("family"), str(version), version


def _build_schema_definition(item: Mapping[str, Any]) -> SchemaDefinition:
    family, version, version_info = parse_schema_name(item["schema_name"])
    return SchemaDefinition(
        schema_name=item["schema_name"],
        family=family,
        version=version,
        version_info=version_info,
        owner_ref=item["owner_ref"],
        owning_module=item["owning_module"],
        canonical_source_path=item["canonical_source_path"],
    )


def _group_schemas_by_family(
    definitions: Any,
) -> dict[str, tuple[SchemaDefinition, ...]]:
    grouped: dict[str, list[SchemaDefinition]] = {}
    for definition in definitions:
        grouped.setdefault(definition.family, []).append(definition)
    return {
        family: tuple(sorted(items, key=lambda item: item.version_info))
        for family, items in grouped.items()
    }


def _build_enum_definition(root: Path, item: Mapping[str, Any]) -> EnumDefinition:
    source_path = root / item["canonical_source_path"]
    source_kind = "registry_literal"
    allowed_values = tuple(item.get("allowed_values", []))

    if source_path.suffix in {".yaml", ".yml"} and source_path.is_file():
        payload = _load_yaml(source_path)
        derived_values, source_kind = _extract_enum_values_from_payload(payload)
        if derived_values:
            allowed_values = derived_values

    return EnumDefinition(
        enum_name=item["enum_name"],
        owner_ref=item["owner_ref"],
        canonical_source_path=item["canonical_source_path"],
        allowed_values=allowed_values,
        source_kind=source_kind,
    )


def _extract_enum_values_from_payload(
    payload: Mapping[str, Any]
) -> tuple[tuple[str, ...], str]:
    if "vocabulary" in payload and isinstance(payload.get("entries"), list):
        values = tuple(
            str(entry["id"])
            for entry in payload["entries"]
            if isinstance(entry, Mapping) and "id" in entry
        )
        return values, "core_vocabulary"
    if isinstance(payload.get("states"), list):
        values = tuple(
            str(state["state_id"])
            for state in payload["states"]
            if isinstance(state, Mapping) and "state_id" in state
        )
        return values, "contract_states"
    return (), "registry_literal"


def _load_contract_documents(root: Path) -> tuple[ContractDocument, ...]:
    documents: list[ContractDocument] = []
    for path in sorted((root / "schemas/contracts").glob("*.yaml")):
        payload = _load_yaml(path)
        documents.append(
            ContractDocument(
                name=path.stem,
                path=path.relative_to(root).as_posix(),
                kind=str(payload.get("kind", "")),
                version=payload.get("version"),
                payload=payload,
                state_ids=tuple(
                    str(item["state_id"])
                    for item in payload.get("states", [])
                    if isinstance(item, Mapping) and "state_id" in item
                ),
            )
        )
    return tuple(documents)


def _require_contract_document(
    documents: tuple[ContractDocument, ...],
    name: str,
    kind: str,
) -> ContractDocument:
    for document in documents:
        if document.name == name or document.kind == kind:
            return document
    raise ContractValidationError(
        code="missing_contract_document",
        message=f"missing required contract document {name!r}",
        field_path="contract_document",
        details={"contract_document": name, "kind": kind},
    )


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        return yaml.load(handle, Loader=loader)
