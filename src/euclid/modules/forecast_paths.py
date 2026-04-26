from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from euclid.cir.models import CandidateIntermediateRepresentation


@dataclass(frozen=True)
class ForecastPath:
    predictions: dict[int, float]
    runtime_evidence: dict[str, Any] | None = None


def forecast_path(
    *,
    candidate: CandidateIntermediateRepresentation,
    fit_result: Any,
    origin_row: Mapping[str, Any],
    max_horizon: int,
    entity: str | None = None,
) -> ForecastPath:
    """Shared forecast path facade during the evaluation-path migration."""

    from euclid.modules.evaluation import _legacy_forecast_path

    legacy_path = _legacy_forecast_path(
        candidate=candidate,
        fit_result=fit_result,
        origin_row=origin_row,
        max_horizon=max_horizon,
        entity=entity,
    )
    return ForecastPath(
        predictions=dict(legacy_path.predictions),
        runtime_evidence=(
            dict(legacy_path.runtime_evidence)
            if legacy_path.runtime_evidence is not None
            else None
        ),
    )
