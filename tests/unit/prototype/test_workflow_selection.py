from __future__ import annotations

from euclid.prototype.workflow import _CandidateEvaluation, _select_candidate


def _candidate(
    *,
    family_id: str,
    exploratory_primary_score: float,
    description_gain_bits: float,
) -> _CandidateEvaluation:
    return _CandidateEvaluation(
        family_id=family_id,
        candidate_id=f"prototype_{family_id}_candidate_v1",
        parameters={"weight": 1.0},
        exploratory_primary_score=exploratory_primary_score,
        confirmatory_primary_score=exploratory_primary_score,
        baseline_primary_score=exploratory_primary_score + 1.0,
        description_components={"reference_bits": 10.0, "L_total_bits": 5.0},
        description_gain_bits=description_gain_bits,
        admissible=description_gain_bits > 0.0,
        structure_signature=f"{family_id}:weight=1.0",
        confirmatory_prediction_rows=(),
        development_losses=(exploratory_primary_score,),
    )


def test_select_candidate_returns_first_admissible_candidate() -> None:
    candidates = (
        _candidate(
            family_id="constant",
            exploratory_primary_score=0.2,
            description_gain_bits=-0.5,
        ),
        _candidate(
            family_id="linear_trend",
            exploratory_primary_score=0.3,
            description_gain_bits=1.25,
        ),
        _candidate(
            family_id="drift",
            exploratory_primary_score=0.4,
            description_gain_bits=2.0,
        ),
    )

    selected = _select_candidate(candidates, minimum_description_gain_bits=0.0)

    assert selected.family_id == "linear_trend"
    assert selected.candidate_id == "prototype_linear_trend_candidate_v1"


def test_select_candidate_falls_back_to_best_scoring_candidate_if_none_are_admissible(
) -> None:
    candidates = (
        _candidate(
            family_id="constant",
            exploratory_primary_score=0.2,
            description_gain_bits=-0.5,
        ),
        _candidate(
            family_id="linear_trend",
            exploratory_primary_score=0.3,
            description_gain_bits=-0.1,
        ),
    )

    selected = _select_candidate(candidates, minimum_description_gain_bits=0.0)

    assert selected.family_id == "constant"
    assert selected.candidate_id == "prototype_constant_candidate_v1"
