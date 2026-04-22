from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np

from euclid.cir.models import CandidateIntermediateRepresentation
from euclid.contracts.errors import ContractValidationError
from euclid.reducers.composition import SharedPlusLocalComposition


@dataclass(frozen=True)
class SharedLocalFitSummary:
    entity_panel: tuple[str, ...]
    backend_id: str
    objective_id: str
    parameter_summary: Mapping[str, float]
    shared_component: Mapping[str, Any]
    local_components: tuple[Mapping[str, Any], ...]
    final_loss: float
    baseline_backend_id: str | None = None
    evidence_role: str = "legacy_non_claim_adapter"
    claim_lane_ceiling: str = "descriptive_structure"
    universal_law_evidence_allowed: bool = False

    def as_diagnostics(self) -> dict[str, Any]:
        diagnostics = {
            "entity_panel": list(self.entity_panel),
            "shared_component": dict(self.shared_component),
            "local_components": [
                dict(component) for component in self.local_components
            ],
            "sharing_map": ["intercept"],
            "unseen_entity_rule": "panel_entities_only",
            "evidence_role": self.evidence_role,
            "claim_lane_ceiling": self.claim_lane_ceiling,
            "universal_law_evidence_allowed": self.universal_law_evidence_allowed,
            "legacy_adapter_status": "legacy_non_claim_adapter",
        }
        if self.baseline_backend_id is not None:
            diagnostics["baseline_backend_id"] = self.baseline_backend_id
            diagnostics["selected_backend_id"] = self.backend_id
        return diagnostics


_BASELINE_BACKEND_ID = "legacy_non_claim_shared_local_mean_offsets_v1"
_BASELINE_OBJECTIVE_ID = "legacy_non_claim_shared_local_offsets_v1"
_OPTIMIZER_BACKEND_ID = "legacy_non_claim_shared_local_panel_optimizer_v1"
_OPTIMIZER_OBJECTIVE_ID = "legacy_non_claim_shared_local_panel_optimizer_v1"


def fit_shared_plus_local_decomposition(
    *,
    candidate: CandidateIntermediateRepresentation,
    training_rows: Sequence[Mapping[str, Any]],
    random_seed: str,
    backend_preference: str = "best",
) -> SharedLocalFitSummary:
    del random_seed
    if backend_preference not in {"best", "baseline", "optimizer"}:
        raise ContractValidationError(
            code="invalid_shared_local_backend_preference",
            message=(
                "shared-plus-local fitting backend_preference must be one of "
                "'best', 'baseline', or 'optimizer'"
            ),
            field_path="backend_preference",
            details={"backend_preference": backend_preference},
        )
    composition = candidate.structural_layer.composition_graph.composition
    if not isinstance(composition, SharedPlusLocalComposition):
        raise ContractValidationError(
            code="invalid_shared_local_candidate",
            message=(
                "shared-plus-local fitting requires a shared_plus_local_decomposition "
                "composition payload"
            ),
            field_path="candidate.structural_layer.composition_graph",
        )

    entity_rows: dict[str, list[Mapping[str, Any]]] = {
        entity: [] for entity in composition.entity_index_set
    }
    for index, row in enumerate(training_rows):
        entity = str(row.get("entity", ""))
        if entity not in entity_rows:
            raise ContractValidationError(
                code="entity_panel_mismatch",
                message="training rows must stay inside the declared entity panel",
                field_path=f"training_rows[{index}].entity",
                details={
                    "entity": entity,
                    "entity_panel": list(composition.entity_index_set),
                },
            )
        entity_rows[entity].append(row)

    missing_entities = [entity for entity, rows in entity_rows.items() if not rows]
    if missing_entities:
        raise ContractValidationError(
            code="entity_panel_mismatch",
            message="training rows must provide at least one row for each entity",
            field_path="training_rows",
            details={
                "missing_entities": missing_entities,
                "entity_panel": list(composition.entity_index_set),
            },
        )

    baseline_summary = _fit_mean_offset_baseline(
        composition=composition,
        entity_rows=entity_rows,
        training_rows=training_rows,
    )
    optimizer_summary = _fit_panel_joint_optimizer(
        composition=composition,
        entity_rows=entity_rows,
        training_rows=training_rows,
    )
    if backend_preference == "baseline":
        return baseline_summary
    if backend_preference == "optimizer":
        if optimizer_summary is None:
            raise ContractValidationError(
                code="shared_local_optimizer_unavailable",
                message=(
                    "shared-plus-local optimizer requires a lag-complete panel with "
                    "enough variation to fit the declared optimizer family"
                ),
                field_path="training_rows",
                details={"entity_panel": list(composition.entity_index_set)},
            )
        return SharedLocalFitSummary(
            entity_panel=optimizer_summary.entity_panel,
            backend_id=optimizer_summary.backend_id,
            objective_id=optimizer_summary.objective_id,
            parameter_summary=optimizer_summary.parameter_summary,
            shared_component=optimizer_summary.shared_component,
            local_components=optimizer_summary.local_components,
            final_loss=optimizer_summary.final_loss,
            baseline_backend_id=baseline_summary.backend_id,
            evidence_role="legacy_non_claim_adapter",
            claim_lane_ceiling="descriptive_structure",
            universal_law_evidence_allowed=False,
        )
    if (
        optimizer_summary is not None
        and optimizer_summary.final_loss + 1e-12 < baseline_summary.final_loss
    ):
        return SharedLocalFitSummary(
            entity_panel=optimizer_summary.entity_panel,
            backend_id=optimizer_summary.backend_id,
            objective_id=optimizer_summary.objective_id,
            parameter_summary=optimizer_summary.parameter_summary,
            shared_component=optimizer_summary.shared_component,
            local_components=optimizer_summary.local_components,
            final_loss=optimizer_summary.final_loss,
            baseline_backend_id=baseline_summary.backend_id,
            evidence_role="legacy_non_claim_adapter",
            claim_lane_ceiling="descriptive_structure",
            universal_law_evidence_allowed=False,
        )
    return baseline_summary


def _fit_mean_offset_baseline(
    *,
    composition: SharedPlusLocalComposition,
    entity_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    training_rows: Sequence[Mapping[str, Any]],
) -> SharedLocalFitSummary:
    targets = tuple(float(row["target"]) for row in training_rows)
    shared_intercept = _stable_float(fmean(targets))

    local_components: list[dict[str, Any]] = []
    parameter_summary: dict[str, float] = {
        "shared_intercept": shared_intercept,
    }
    for entity, component_id in zip(
        composition.entity_index_set,
        composition.local_component_refs,
        strict=True,
    ):
        entity_mean = fmean(float(row["target"]) for row in entity_rows[entity])
        local_adjustment = _stable_float(entity_mean - shared_intercept)
        parameter_key = f"local_adjustment__{entity}"
        parameter_summary[parameter_key] = local_adjustment
        local_components.append(
            {
                "component_id": component_id,
                "entity": entity,
                "fit_rule": "entity_mean_offset",
                "row_count": len(entity_rows[entity]),
                "parameter_summary": {parameter_key: local_adjustment},
            }
        )

    final_loss = 0.0
    for row in training_rows:
        entity = str(row["entity"])
        prediction = shared_intercept + parameter_summary[f"local_adjustment__{entity}"]
        final_loss += (float(row["target"]) - prediction) ** 2

    return SharedLocalFitSummary(
        entity_panel=composition.entity_index_set,
        backend_id=_BASELINE_BACKEND_ID,
        objective_id=_BASELINE_OBJECTIVE_ID,
        parameter_summary=parameter_summary,
        shared_component={
            "component_id": composition.shared_component_ref,
            "fit_rule": "panel_mean",
            "row_count": len(training_rows),
        },
        local_components=tuple(local_components),
        final_loss=_stable_float(final_loss),
    )


def _fit_panel_joint_optimizer(
    *,
    composition: SharedPlusLocalComposition,
    entity_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    training_rows: Sequence[Mapping[str, Any]],
) -> SharedLocalFitSummary | None:
    x_values: list[float] = []
    y_values: list[float] = []
    entity_index = {entity: index for index, entity in enumerate(composition.entity_index_set)}
    first_entity = composition.entity_index_set[0]
    rows_per_entity = {
        entity: len(rows) for entity, rows in entity_rows.items()
    }

    for row in training_rows:
        lag_value = row.get("lag_1")
        if lag_value is None:
            return None
        x_values.append(float(lag_value))
        y_values.append(float(row["target"]))
    if len(x_values) < len(composition.entity_index_set) + 1:
        return None
    if len({round(value, 12) for value in x_values}) < 2:
        return None

    design_rows: list[list[float]] = []
    for row, lag_value in zip(training_rows, x_values, strict=True):
        entity = str(row["entity"])
        row_terms: list[float] = [1.0, lag_value]
        for panel_entity in composition.entity_index_set[1:]:
            indicator = 1.0 if entity == panel_entity else 0.0
            row_terms.append(indicator)
        for panel_entity in composition.entity_index_set[1:]:
            indicator = lag_value if entity == panel_entity else 0.0
            row_terms.append(indicator)
        design_rows.append(row_terms)

    design = np.asarray(design_rows, dtype=float)
    targets = np.asarray(y_values, dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    predictions = design @ coefficients
    final_loss = _stable_float(float(np.square(targets - predictions).sum()))

    shared_intercept = _stable_float(float(coefficients[0]))
    shared_lag = _stable_float(float(coefficients[1]))
    parameter_summary: dict[str, float] = {
        "shared_intercept": shared_intercept,
        "shared_lag_coefficient": shared_lag,
    }
    intercept_offsets = {
        first_entity: 0.0,
    }
    lag_offsets = {
        first_entity: 0.0,
    }
    intercept_coefficients = coefficients[2 : 2 + (len(composition.entity_index_set) - 1)]
    lag_coefficients = coefficients[2 + (len(composition.entity_index_set) - 1) :]
    for panel_entity, coefficient in zip(
        composition.entity_index_set[1:],
        intercept_coefficients,
        strict=True,
    ):
        intercept_offsets[panel_entity] = _stable_float(float(coefficient))
    for panel_entity, coefficient in zip(
        composition.entity_index_set[1:],
        lag_coefficients,
        strict=True,
    ):
        lag_offsets[panel_entity] = _stable_float(float(coefficient))

    local_components: list[dict[str, Any]] = []
    for entity, component_id in zip(
        composition.entity_index_set,
        composition.local_component_refs,
        strict=True,
    ):
        adjustment_key = f"local_adjustment__{entity}"
        lag_adjustment_key = f"local_lag_adjustment__{entity}"
        parameter_summary[adjustment_key] = intercept_offsets[entity]
        parameter_summary[lag_adjustment_key] = lag_offsets[entity]
        local_components.append(
            {
                "component_id": component_id,
                "entity": entity,
                "fit_rule": "panel_joint_local_effects",
                "row_count": rows_per_entity[entity],
                "parameter_summary": {
                    adjustment_key: intercept_offsets[entity],
                    lag_adjustment_key: lag_offsets[entity],
                },
            }
        )

    return SharedLocalFitSummary(
        entity_panel=composition.entity_index_set,
        backend_id=_OPTIMIZER_BACKEND_ID,
        objective_id=_OPTIMIZER_OBJECTIVE_ID,
        parameter_summary=parameter_summary,
        shared_component={
            "component_id": composition.shared_component_ref,
            "fit_rule": "panel_joint_least_squares",
            "row_count": len(training_rows),
            "parameter_summary": {
                "shared_intercept": shared_intercept,
                "shared_lag_coefficient": shared_lag,
            },
        },
        local_components=tuple(local_components),
        final_loss=final_loss,
    )


def _stable_float(value: float) -> float:
    return float(round(float(value), 12))


__all__ = [
    "SharedLocalFitSummary",
    "fit_shared_plus_local_decomposition",
]
