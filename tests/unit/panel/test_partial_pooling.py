from __future__ import annotations

from euclid.panel.partial_pooling import fit_partial_pooling_baseline


def test_partial_pooling_records_shrinkage_and_dispersion() -> None:
    result = fit_partial_pooling_baseline(
        (
            {"entity": "a", "target": 10.0},
            {"entity": "a", "target": 12.0},
            {"entity": "b", "target": 20.0},
            {"entity": "b", "target": 22.0},
        ),
        entity_field="entity",
        target_field="target",
        ridge_strength=2.0,
    )

    assert result.status == "baseline_fit"
    assert result.pooling_strength == 0.5
    assert result.global_parameters == {"global_intercept": 16.0}
    assert result.entity_local_parameters == {
        "a": {"intercept_offset": -2.5},
        "b": {"intercept_offset": 2.5},
    }
    assert result.parameter_dispersion["intercept_offset_range"] == 5.0
    assert result.evidence_role == "baseline_only"
    assert result.universal_law_evidence_allowed is False


def test_partial_pooling_abstains_on_insufficient_entities() -> None:
    result = fit_partial_pooling_baseline(
        (
            {"entity": "a", "target": 10.0},
            {"entity": "a", "target": 12.0},
        ),
        entity_field="entity",
        target_field="target",
        min_entities=2,
    )

    assert result.status == "abstained"
    assert result.reason_codes == ("insufficient_entities",)
    assert result.claim_lane_ceiling == "descriptive_structure"
