from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.math.observation_models import PointObservationModel
from euclid.reducers.models import (
    AdditiveResidualComposition,
    BoundObservationModel,
    ReducerAdmissibilityObject,
    ReducerCompositionObject,
    ReducerFamilyId,
    ReducerObject,
    ReducerParameter,
    ReducerParameterObject,
    ReducerStateObject,
    ReducerStateSemantics,
    ReducerStateSlot,
    ReducerStateUpdateContext,
    ReducerStateUpdateRule,
    parse_reducer_composition,
)


def test_reducer_object_binds_typed_structures() -> None:
    reducer = ReducerObject(
        family=ReducerFamilyId("recursive"),
        composition_object=ReducerCompositionObject(),
        fitted_parameters=ReducerParameterObject(
            parameters=(ReducerParameter(name="alpha", value=0.75),)
        ),
        state_semantics=ReducerStateSemantics(
            persistent_state=ReducerStateObject(
                slots=(
                    ReducerStateSlot(name="running_total", value=0.0),
                    ReducerStateSlot(name="step_count", value=0),
                )
            ),
            update_rule=ReducerStateUpdateRule(
                update_rule_id="running_total_accumulator",
                implementation=_running_total_update,
            ),
        ),
        observation_model=BoundObservationModel.from_runtime(
            PointObservationModel()
        ),
        admissibility=ReducerAdmissibilityObject(
            family_membership=True,
            composition_closure=True,
            observation_model_compatibility=True,
            valid_state_semantics=True,
            codelength_comparability=True,
        ),
    )

    assert reducer.family.family_id == "recursive"
    assert reducer.composition_object.operator_id is None
    assert reducer.fitted_parameters.parameters[0].name == "alpha"
    assert reducer.observation_model.supports_point_loss("squared_error")
    assert reducer.admissibility.is_admissible


def test_state_semantics_replay_deterministic_updates_from_coded_initial_state(
) -> None:
    state_semantics = ReducerStateSemantics(
        persistent_state=ReducerStateObject(
            slots=(
                ReducerStateSlot(name="running_total", value=0.0),
                ReducerStateSlot(name="step_count", value=0),
            )
        ),
        update_rule=ReducerStateUpdateRule(
            update_rule_id="running_total_accumulator",
            implementation=_running_total_update,
        ),
    )

    initial_state = state_semantics.initialize_state()
    next_state = state_semantics.update_state(
        initial_state,
        ReducerStateUpdateContext(
            observation_index=0,
            history=(2.5,),
        ),
    )
    replay_state = state_semantics.update_state(
        initial_state,
        ReducerStateUpdateContext(
            observation_index=0,
            history=(2.5,),
        ),
    )

    assert next_state == replay_state
    assert next_state.get("running_total") == 2.5
    assert next_state.get("step_count") == 1
    assert initial_state.get("running_total") == 0.0


def test_reducer_vocabularies_and_state_slots_are_validated() -> None:
    with pytest.raises(ContractValidationError) as family_error:
        ReducerFamilyId("unknown")

    assert family_error.value.code == "invalid_reducer_family"

    with pytest.raises(ContractValidationError) as operator_error:
        ReducerCompositionObject(operator_id="not_real")

    assert operator_error.value.code == "invalid_composition_operator"

    with pytest.raises(ContractValidationError) as slot_error:
        ReducerStateObject(
            slots=(
                ReducerStateSlot(name="running_total", value=0.0),
                ReducerStateSlot(name="running_total", value=1.0),
            )
        )

    assert slot_error.value.code == "duplicate_reducer_state_slot"


def test_piecewise_normalization_collapses_equivalent_partition_encodings() -> None:
    left = parse_reducer_composition(
        {
            "operator_id": "piecewise",
            "branch_order_law": "ascending_split_literal",
            "ordered_partition": [
                {
                    "start_literal": 2.0,
                    "end_literal": 3.0,
                    "reducer_id": "tail",
                },
                {
                    "start_literal": 0.0,
                    "end_literal": 1.0,
                    "reducer_id": "head",
                },
                {
                    "start_literal": 1.0,
                    "end_literal": 2.0,
                    "reducer_id": "tail",
                },
            ],
        }
    )
    right = parse_reducer_composition(
        {
            "operator_id": "piecewise",
            "ordered_partition": [
                {
                    "start_literal": 0.0,
                    "end_literal": 1.0,
                    "reducer_id": "head",
                },
                {
                    "start_literal": 1.0,
                    "end_literal": 3.0,
                    "reducer_id": "tail",
                },
            ],
        }
    )

    normalized = left.normalize()

    assert normalized.child_reducer_ids == ("head", "tail")
    assert normalized.as_dict()["ordered_partition"] == [
        {"start_literal": 0.0, "end_literal": 1.0, "reducer_id": "head"},
        {"start_literal": 1.0, "end_literal": 3.0, "reducer_id": "tail"},
    ]
    assert left.canonical_bytes() == right.canonical_bytes()
    assert left.canonical_hash() == right.canonical_hash()


def test_piecewise_partition_rejects_overlapping_segments() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        parse_reducer_composition(
            {
                "operator_id": "piecewise",
                "ordered_partition": [
                    {
                        "start_literal": 0.0,
                        "end_literal": 2.0,
                        "reducer_id": "left",
                    },
                    {
                        "start_literal": 1.5,
                        "end_literal": 3.0,
                        "reducer_id": "right",
                    },
                ],
            }
        )

    assert exc_info.value.code == "invalid_piecewise_partition"


def test_additive_residual_requires_explicit_base_then_residual_structure() -> None:
    composition = ReducerCompositionObject(
        operator_id="additive_residual",
        composition=AdditiveResidualComposition(
            base_reducer="trend",
            residual_reducer="seasonal_adjustment",
            shared_observation_model="point_identity",
        ),
    )

    assert composition.child_reducer_ids == ("trend", "seasonal_adjustment")
    assert composition.canonical_payload()["child_reducer_ids"] == [
        "trend",
        "seasonal_adjustment",
    ]


def test_regime_conditioned_normalizes_branch_and_contract_order() -> None:
    left = parse_reducer_composition(
        {
            "operator_id": "regime_conditioned",
            "gating_law": {
                "gating_law_id": "market_regime_gate",
                "selection_mode": "hard_switch",
            },
            "regime_information_contract": ["weekday", "holiday_flag"],
            "branch_reducers": [
                {"regime_value": "weekend", "reducer_id": "slow_path"},
                {"regime_value": "weekday", "reducer_id": "fast_path"},
            ],
        }
    )
    right = parse_reducer_composition(
        {
            "operator_id": "regime_conditioned",
            "gating_law": {
                "selection_mode": "hard_switch",
                "gating_law_id": "market_regime_gate",
            },
            "regime_information_contract": ["holiday_flag", "weekday"],
            "branch_reducers": [
                {"regime_value": "weekday", "reducer_id": "fast_path"},
                {"regime_value": "weekend", "reducer_id": "slow_path"},
            ],
        }
    )

    normalized = left.normalize()

    assert normalized.child_reducer_ids == ("fast_path", "slow_path")
    assert normalized.as_dict()["regime_information_contract"] == [
        "holiday_flag",
        "weekday",
    ]
    assert normalized.as_dict()["branch_reducers"] == [
        {"regime_value": "weekday", "reducer_id": "fast_path"},
        {"regime_value": "weekend", "reducer_id": "slow_path"},
    ]
    assert left.canonical_hash() == right.canonical_hash()


def _running_total_update(
    state: ReducerStateObject,
    context: ReducerStateUpdateContext,
) -> ReducerStateObject:
    return ReducerStateObject(
        slots=(
            ReducerStateSlot(
                name="running_total",
                value=float(state.get("running_total")) + context.history[-1],
            ),
            ReducerStateSlot(
                name="step_count",
                value=int(state.get("step_count")) + 1,
            ),
        )
    )
