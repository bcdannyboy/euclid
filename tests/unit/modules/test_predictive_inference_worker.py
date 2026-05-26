from __future__ import annotations

import importlib
from typing import Any

import pytest

from euclid.contracts.errors import ContractValidationError


def _predictive_inference() -> Any:
    try:
        return importlib.import_module("euclid.modules.predictive_inference")
    except ModuleNotFoundError as exc:
        pytest.fail(f"predictive_inference module is missing: {exc}")


def _loss_rows(
    losses: tuple[float, ...],
    *,
    row_set_id: str = "holdout-panel",
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "origin_id": f"origin-{index}",
            "horizon": 1,
            "entity_id": "series-a",
            "row_set_id": row_set_id,
            "loss": loss,
        }
        for index, loss in enumerate(losses, start=1)
    )


def test_paired_loss_stream_rejects_identity_mismatch() -> None:
    module = _predictive_inference()
    candidate_rows = _loss_rows((0.20, 0.25))
    baseline_rows = list(_loss_rows((0.30, 0.40)))
    baseline_rows[1]["row_set_id"] = "different-panel"

    with pytest.raises(ContractValidationError) as exc_info:
        module.PairedLossDifferentialStream.from_loss_rows(
            stream_id="candidate-vs-baseline",
            candidate_id="candidate",
            baseline_id="baseline",
            loss_id="squared_error",
            candidate_rows=candidate_rows,
            baseline_rows=tuple(baseline_rows),
        )

    assert exc_info.value.code == "paired_loss_identity_mismatch"
    assert exc_info.value.details["field"] == "row_set_id"


def test_one_pair_declared_test_abstains() -> None:
    module = _predictive_inference()
    stream = module.PairedLossDifferentialStream.from_loss_rows(
        stream_id="candidate-vs-baseline",
        candidate_id="candidate",
        baseline_id="baseline",
        loss_id="absolute_error",
        candidate_rows=_loss_rows((0.20,)),
        baseline_rows=_loss_rows((0.30,)),
    )

    result = module.run_declared_predictive_test(
        stream=stream,
        declared_test_id="diebold_mariano_hln_v1",
        practical_margin=0.01,
    )

    assert result.status == "abstained"
    assert result.promotion_allowed is False
    assert result.reason_codes == ("insufficient_paired_count",)


def test_declared_dm_hln_result_uses_dm_identity_not_generic_hac_label() -> None:
    module = _predictive_inference()
    stream = module.PairedLossDifferentialStream.from_loss_rows(
        stream_id="candidate-vs-baseline",
        candidate_id="candidate",
        baseline_id="baseline",
        loss_id="absolute_error",
        candidate_rows=_loss_rows((0.20, 0.25, 0.22, 0.24, 0.23, 0.21)),
        baseline_rows=_loss_rows((0.34, 0.36, 0.35, 0.37, 0.38, 0.36)),
    )

    result = module.run_declared_predictive_test(
        stream=stream,
        declared_test_id="diebold_mariano_hln_v1",
        practical_margin=0.01,
    )

    assert result.declared_test_id == "diebold_mariano_hln_v1"
    assert result.statistical_test_backend == "diebold_mariano_hln_v1"
    assert result.confidence_interval_method == "dm_hln_hac_t_interval"
    assert result.as_manifest()["declared_test_id"] == "diebold_mariano_hln_v1"


def test_declared_block_bootstrap_records_block_metadata() -> None:
    module = _predictive_inference()
    stream = module.PairedLossDifferentialStream.from_loss_rows(
        stream_id="candidate-vs-baseline",
        candidate_id="candidate",
        baseline_id="baseline",
        loss_id="absolute_error",
        candidate_rows=_loss_rows((0.20, 0.25, 0.22, 0.24, 0.23, 0.21, 0.26, 0.22)),
        baseline_rows=_loss_rows((0.34, 0.36, 0.35, 0.37, 0.38, 0.36, 0.40, 0.35)),
    )

    result = module.run_declared_predictive_test(
        stream=stream,
        declared_test_id="paired_stationary_block_bootstrap_v1",
        practical_margin=0.01,
        block_length=3,
        bootstrap_count=200,
        seed=17,
    )

    assert result.declared_test_id == "paired_stationary_block_bootstrap_v1"
    assert result.statistical_test_backend == "paired_stationary_block_bootstrap_v1"
    assert result.metadata["block_length"] == 3
    assert result.metadata["bootstrap_count"] == 200
    assert result.metadata["seed"] == 17
    assert result.as_manifest()["metadata"]["block_length"] == 3


def test_unsupported_declared_predictive_test_id_fails_closed() -> None:
    module = _predictive_inference()
    stream = module.PairedLossDifferentialStream.from_loss_rows(
        stream_id="candidate-vs-baseline",
        candidate_id="candidate",
        baseline_id="baseline",
        loss_id="absolute_error",
        candidate_rows=_loss_rows((0.20, 0.25)),
        baseline_rows=_loss_rows((0.34, 0.36)),
    )

    result = module.run_declared_predictive_test(
        stream=stream,
        declared_test_id="raw_metric_delta",
        practical_margin=0.01,
    )

    assert result.status == "failed"
    assert result.promotion_allowed is False
    assert result.reason_codes == ("unsupported_declared_predictive_test_id",)


def test_gw_declared_test_cannot_be_default_without_state_declarations() -> None:
    module = _predictive_inference()
    stream = module.PairedLossDifferentialStream.from_loss_rows(
        stream_id="candidate-vs-baseline",
        candidate_id="candidate",
        baseline_id="baseline",
        loss_id="absolute_error",
        candidate_rows=_loss_rows((0.20, 0.25, 0.22)),
        baseline_rows=_loss_rows((0.34, 0.36, 0.35)),
    )

    result = module.run_declared_predictive_test(
        stream=stream,
        declared_test_id="giacomini_white_v1",
        practical_margin=0.01,
    )

    assert result.status == "failed"
    assert result.promotion_allowed is False
    assert result.reason_codes == ("gw_requires_instruments_or_state",)
