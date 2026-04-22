from __future__ import annotations

from euclid.panel.leave_one_entity_out import (
    make_leave_one_entity_out_splits,
    validate_leave_one_entity_out,
)


def test_leave_one_entity_out_splits_are_deterministic() -> None:
    rows = (
        {"entity": "b", "target": 2.0},
        {"entity": "a", "target": 1.0},
        {"entity": "c", "target": 3.0},
    )

    splits = make_leave_one_entity_out_splits(rows, entity_field="entity")

    assert [split.heldout_entity for split in splits] == ["a", "b", "c"]
    assert splits[0].train_indices == (0, 2)
    assert splits[0].test_indices == (1,)


def test_leave_one_entity_out_requires_sufficient_entities() -> None:
    result = validate_leave_one_entity_out(
        (
            {"entity": "a", "target": 1.0},
            {"entity": "b", "target": 2.0},
        ),
        entity_field="entity",
        target_field="target",
        min_entities=3,
    )

    assert result.status == "failed"
    assert result.reason_codes == ("insufficient_entities",)
    assert result.transport_evidence_allowed is False


def test_leave_one_entity_out_emits_unseen_entity_replay_evidence() -> None:
    rows = (
        {"entity": "a", "target": 1.0},
        {"entity": "a", "target": 1.1},
        {"entity": "b", "target": 1.0},
        {"entity": "b", "target": 1.2},
        {"entity": "c", "target": 1.1},
        {"entity": "c", "target": 1.2},
    )

    result = validate_leave_one_entity_out(
        rows,
        entity_field="entity",
        target_field="target",
        max_mean_loss=0.05,
    )

    assert result.status == "passed"
    assert result.transport_evidence_allowed is True
    assert result.evidence_role == "unseen_entity_holdout"
    assert result.replay_identity
    assert [score["heldout_entity"] for score in result.fold_scores] == [
        "a",
        "b",
        "c",
    ]
