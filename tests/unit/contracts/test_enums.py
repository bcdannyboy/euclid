from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.enums import validate_enum_value
from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_validate_enum_value_accepts_known_closed_vocabulary_value() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    assert (
        validate_enum_value(
            catalog,
            "claim_lanes",
            "predictive_within_declared_scope",
            field_path="body.claim_lane",
        )
        == "predictive_within_declared_scope"
    )


def test_validate_enum_value_rejects_unknown_closed_vocabulary_value() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)

    with pytest.raises(ContractValidationError) as excinfo:
        validate_enum_value(
            catalog,
            "claim_lanes",
            "unsupported_lane",
            field_path="body.claim_lane",
        )

    assert excinfo.value.as_dict() == {
        "code": "invalid_enum_value",
        "message": "'unsupported_lane' is not a legal value for claim_lanes",
        "field_path": "body.claim_lane",
        "details": {
            "enum_name": "claim_lanes",
            "value": "unsupported_lane",
        },
    }
