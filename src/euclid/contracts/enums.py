from __future__ import annotations

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import ContractCatalog


def validate_enum_value(
    catalog: ContractCatalog,
    enum_name: str,
    value: str,
    *,
    field_path: str | None = None,
) -> str:
    if value not in catalog.enum_values(enum_name):
        raise ContractValidationError(
            code="invalid_enum_value",
            message=f"{value!r} is not a legal value for {enum_name}",
            field_path=field_path,
            details={"enum_name": enum_name, "value": value},
        )
    return value
