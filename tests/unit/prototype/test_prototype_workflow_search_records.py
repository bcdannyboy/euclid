from __future__ import annotations

import euclid.prototype.workflow as prototype_workflow


def test_search_candidate_records_keep_non_admissible_candidates_ranked_for_descriptive_scope() -> None:
    candidate = prototype_workflow._CandidateEvaluation(
        family_id="drift",
        candidate_id="prototype_drift_candidate_v1",
        parameters={"intercept": 1.0},
        exploratory_primary_score=0.25,
        confirmatory_primary_score=0.5,
        baseline_primary_score=1.0,
        description_components={"L_total_bits": 12.0, "L_structure_bits": 3.0},
        description_gain_bits=-0.5,
        admissible=False,
        structure_signature="drift:intercept=1.0",
        confirmatory_prediction_rows=(),
        development_losses=(0.25,),
    )

    records = prototype_workflow._search_candidate_records((candidate,))

    assert len(records) == 1
    record = records[0]
    assert record.total_code_bits == 12.0
    assert record.ranked_for_descriptive_scope is True
    assert record.law_eligible_for_claims is False
    assert record.rejection_reason_codes == (
        "codelength_comparability_failed",
        "description_gain_non_positive",
    )
