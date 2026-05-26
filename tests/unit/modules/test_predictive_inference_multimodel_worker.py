from __future__ import annotations

import importlib
from typing import Any


def _predictive_inference() -> Any:
    return importlib.import_module("euclid.modules.predictive_inference")


def _loss_rows(losses: tuple[float, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "origin_id": f"origin-{index}",
            "horizon": 1,
            "entity_id": "series-a",
            "row_set_id": "holdout-panel",
            "loss": loss,
        }
        for index, loss in enumerate(losses, start=1)
    )


def _paired_stream() -> Any:
    module = _predictive_inference()
    return module.PairedLossDifferentialStream.from_loss_rows(
        stream_id="candidate-vs-baseline",
        candidate_id="candidate",
        baseline_id="baseline",
        loss_id="absolute_error",
        candidate_rows=_loss_rows((0.20, 0.22, 0.21, 0.23)),
        baseline_rows=_loss_rows((0.34, 0.36, 0.35, 0.37)),
    )


def test_model_confidence_set_fails_closed_when_arch_backend_missing() -> None:
    module = _predictive_inference()

    result = module.run_declared_predictive_test(
        stream=_paired_stream(),
        declared_test_id="model_confidence_set_v1",
        practical_margin=0.01,
        optional_backend_overrides={"arch": None},
    )
    manifest = result.as_manifest()

    assert result.status == "abstained"
    assert result.promotion_allowed is False
    assert result.reason_codes == (
        "multi_model_test_backend_unavailable",
        "multi_model_superiority_not_tested",
    )
    assert manifest["declared_test_id"] == "model_confidence_set_v1"
    assert manifest["dependency_diagnostics"] == {
        "backend": "arch",
        "implementation": "MCS",
        "reason_code": "multi_model_test_backend_unavailable",
    }
