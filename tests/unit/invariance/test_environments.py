from __future__ import annotations

from euclid.invariance.environments import construct_environments


def test_construct_environments_uses_explicit_labels_and_records_policy() -> None:
    rows = (
        {"environment": "baseline", "entity": "a", "t": 1, "target": 1.0},
        {"environment": "baseline", "entity": "a", "t": 2, "target": 1.2},
        {"environment": "shock", "entity": "a", "t": 3, "target": 2.0},
    )

    result = construct_environments(
        rows,
        policy="explicit_label",
        label_field="environment",
        time_field="t",
    )
    replay = construct_environments(
        rows,
        policy="explicit_label",
        label_field="environment",
        time_field="t",
    )

    assert [slice.environment_id for slice in result.slices] == [
        "baseline",
        "shock",
    ]
    assert result.policy["policy"] == "explicit_label"
    assert result.policy["label_field"] == "environment"
    assert result.slices[0].row_indices == (0, 1)
    assert result.slices[1].row_indices == (2,)
    assert result.replay_identity == replay.replay_identity


def test_construct_environments_supports_entity_and_rolling_era_policies() -> None:
    rows = (
        {"entity": "a", "t": 1, "target": 1.0},
        {"entity": "b", "t": 2, "target": 2.0},
        {"entity": "a", "t": 3, "target": 1.5},
        {"entity": "b", "t": 4, "target": 2.5},
    )

    entity_result = construct_environments(
        rows,
        policy="entity",
        entity_field="entity",
        time_field="t",
    )
    era_result = construct_environments(
        rows,
        policy="rolling_era",
        time_field="t",
        era_count=2,
    )

    assert [slice.environment_id for slice in entity_result.slices] == [
        "entity:a",
        "entity:b",
    ]
    assert [slice.row_indices for slice in era_result.slices] == [(0, 1), (2, 3)]
    assert era_result.policy["era_count"] == 2


def test_construct_environments_supports_intervention_windows() -> None:
    rows = (
        {"t": 1, "target": 1.0},
        {"t": 2, "target": 1.1},
        {"t": 3, "target": 4.0},
        {"t": 4, "target": 4.1},
    )

    result = construct_environments(
        rows,
        policy="intervention_window",
        time_field="t",
        intervention_windows=(
            {"label": "shock", "start": 3, "end": 4},
        ),
    )

    assert [slice.environment_id for slice in result.slices] == [
        "outside_intervention",
        "intervention:shock",
    ]
    assert result.slices[1].metadata["window"] == {"end": 4, "start": 3}


def test_construct_environments_fails_closed_on_insufficient_environments() -> None:
    result = construct_environments(
        ({"environment": "only", "target": 1.0},),
        policy="explicit_label",
        label_field="environment",
        min_environment_count=2,
    )

    assert result.status == "insufficient_environments"
    assert result.reason_codes == ("insufficient_environments",)
    assert result.slices == ()
