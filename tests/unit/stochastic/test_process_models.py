from __future__ import annotations

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.stochastic.process_models import fit_residual_stochastic_model


def test_residual_process_fit_emits_replayable_stochastic_model_manifest() -> None:
    model = fit_residual_stochastic_model(
        candidate_id="candidate",
        residuals=(-1.0, 0.0, 1.0),
        point_path={1: 10.0, 2: 11.0},
        family_id="student_t",
        horizon_scale_law="sqrt_horizon",
    )

    support = model.support_path()
    manifest = model.as_manifest()

    assert manifest["schema_name"] == "stochastic_model_manifest@1.0.0"
    assert manifest["production_evidence"] is True
    assert manifest["heuristic_gaussian_support"] is False
    assert manifest["observation_family"] == "student_t"
    assert support[2].scale > support[1].scale
    assert model.replay_identity.startswith("stochastic-model:")


def test_residual_process_rejects_train_only_overfit_and_nonfinite_residuals() -> None:
    with pytest.raises(ContractValidationError) as exc_info:
        fit_residual_stochastic_model(
            candidate_id="candidate",
            residuals=(0.0, float("nan")),
            point_path={1: 10.0},
            family_id="gaussian",
        )

    assert exc_info.value.code == "nonfinite_stochastic_training_value"

    with pytest.raises(ContractValidationError) as too_few:
        fit_residual_stochastic_model(
            candidate_id="candidate",
            residuals=(0.0,),
            point_path={1: 10.0},
            family_id="gaussian",
            min_residual_count=2,
        )

    assert too_few.value.code == "insufficient_stochastic_training_support"
