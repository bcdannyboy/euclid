from __future__ import annotations

from euclid.panel.shared_structure import discover_shared_structure_panel


def test_shared_structure_pipeline_blocks_universal_law_claims() -> None:
    rows = (
        {"entity": "a", "target": 2.0, "x": 1.0},
        {"entity": "a", "target": 4.0, "x": 2.0},
        {"entity": "b", "target": 3.0, "x": 1.0},
        {"entity": "b", "target": 5.0, "x": 2.0},
    )

    result = discover_shared_structure_panel(
        rows,
        feature_fields=("x",),
        shared_skeleton_terms=("intercept", "x"),
    )

    assert result.status == "selected"
    assert result.claim_lane_ceiling == "predictive_within_declared_scope"
    assert result.universal_law_evidence_allowed is False
    assert result.replay_identity.startswith("shared-structure-panel:")
