from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.expr.external import (
    ExternalEngineCandidate,
    lower_external_engine_candidate,
    require_euclid_claim_boundary,
)


def test_external_engine_lowering_records_provenance_but_blocks_direct_claims() -> None:
    lowered = lower_external_engine_candidate(
        ExternalEngineCandidate(
            engine_name="pysr",
            engine_candidate_id="hall-of-fame-7",
            expression_source="x + 1",
            feature_names=("x",),
            raw_score=0.125,
        )
    )

    assert lowered.expression is not None
    assert lowered.provenance["engine_name"] == "pysr"
    assert lowered.claim_boundary.claim_publication_allowed is False
    assert lowered.claim_boundary.reason_codes == ("external_engine_not_claim_authority",)

    with pytest.raises(ContractValidationError) as exc_info:
        require_euclid_claim_boundary(lowered)

    assert exc_info.value.code == "external_engine_claim_boundary"

