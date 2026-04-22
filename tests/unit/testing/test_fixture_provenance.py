from __future__ import annotations

import pytest

from euclid.testing.fixtures import FixtureGate, FixtureProvenance


def test_fixture_gate_requires_provenance_edge_cases_and_regression_reason() -> None:
    gate = FixtureGate(
        gate_id="P00-T06-fixture-redaction",
        provenance=FixtureProvenance(
            source_kind="hand_authored_adversarial",
            source_ref="tests/unit/testing/test_secret_redaction.py",
            license="repo",
        ),
        edge_cases=("fake_secret_in_header", "fake_secret_in_url"),
        regression_reason="Protect live API evidence redaction.",
    )

    assert gate.validate().status == "passed"


def test_fixture_gate_rejects_missing_edge_case_coverage() -> None:
    gate = FixtureGate(
        gate_id="P00-T06-empty",
        provenance=FixtureProvenance(
            source_kind="synthetic",
            source_ref="inline",
            license="repo",
        ),
        edge_cases=(),
        regression_reason="",
    )

    with pytest.raises(ValueError, match="edge"):
        gate.validate()
