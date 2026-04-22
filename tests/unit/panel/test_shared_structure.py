from __future__ import annotations

from euclid.panel.shared_structure import discover_shared_structure_panel


def test_discover_shared_structure_records_replay_and_no_universal_baseline_evidence() -> None:
    rows = (
        {"entity": "a", "target": 2.0, "x": 1.0},
        {"entity": "a", "target": 4.0, "x": 2.0},
        {"entity": "b", "target": 3.0, "x": 1.0},
        {"entity": "b", "target": 5.0, "x": 2.0},
    )

    result = discover_shared_structure_panel(
        rows,
        entity_field="entity",
        target_field="target",
        feature_fields=("x",),
        shared_skeleton_terms=("intercept", "x"),
    )
    replay = discover_shared_structure_panel(
        rows,
        entity_field="entity",
        target_field="target",
        feature_fields=("x",),
        shared_skeleton_terms=("intercept", "x"),
    )

    assert result.status == "selected"
    assert result.shared_skeleton_terms == ("intercept", "x")
    assert result.claim_lane_ceiling == "predictive_within_declared_scope"
    assert result.universal_law_evidence_allowed is False
    assert result.baseline_evidence_role == "baseline_only"
    assert result.replay_identity == replay.replay_identity


def test_discover_shared_structure_rejects_entity_leakage() -> None:
    result = discover_shared_structure_panel(
        (
            {"entity": "a", "target": 1.0, "entity_feature": "a"},
            {"entity": "b", "target": 2.0, "entity_feature": "b"},
        ),
        entity_field="entity",
        target_field="target",
        feature_fields=("entity_feature",),
        shared_skeleton_terms=("entity_feature",),
    )

    assert result.status == "rejected"
    assert result.reason_codes == ("entity_leakage_detected",)
    assert result.claim_lane_ceiling == "descriptive_structure"


def test_discover_shared_structure_abstains_on_local_only_supports() -> None:
    result = discover_shared_structure_panel(
        (
            {"entity": "a", "target": 1.0, "x_a": 1.0, "x_b": 0.0},
            {"entity": "a", "target": 2.0, "x_a": 2.0, "x_b": 0.0},
            {"entity": "b", "target": 3.0, "x_a": 0.0, "x_b": 1.0},
            {"entity": "b", "target": 4.0, "x_a": 0.0, "x_b": 2.0},
        ),
        entity_field="entity",
        target_field="target",
        feature_fields=("x_a", "x_b"),
        shared_skeleton_terms=("x_a", "x_b"),
        entity_supports={"a": {"x_a"}, "b": {"x_b"}},
    )

    assert result.status == "abstained"
    assert "local_only_overfit_candidate" in result.reason_codes
    assert result.universal_law_evidence_allowed is False
